#!/usr/bin/env python3
"""Conversion fidelity check of the served phase-1 GGUF (results/plan.md,
amendment 2026-10-08; method in NEXT.md).

Target: the served Q8_0 GGUF with MTP (sha256 a96c8c42...) against the HF
text-only checkpoint it was converted from (Qwen3.5-9B-text-mtp, bf16).
Run after phase 1, with the :8093 server stopped (the script refuses if it
answers). Never touches :8092.

Prompts (fixed): MGSM es ids 1-5 and en ids 1-5, each the phase-1 user message
(run_mgsm.PROMPTS[lang] + "\\n\\n" + question), rendered with the HF chat
template + generation prompt; the same HF token ids go to both sides (first
256 positions; all of them, since these prompts are shorter).

- Link 1, HF -> GGUF mapping: transformers bf16 (CPU) vs a BF16 GGUF made with
  the baseline's conversion command (--outtype bf16), next-token
  distributions at every prompt position: pooled top-1 agreement, mean
  KL(HF || GGUF) over the full vocabulary (float64).
- Link 2, BF16 GGUF -> Q8_0: llama-perplexity --kl-divergence-base on the
  BF16 GGUF, then --kl-divergence on the served Q8_0, over the text of the 10
  rendered prompts, -c 128 (scores the second half of each chunk); its "Same
  top p" and "Mean KLD", and the number of chunks.
- Greedy: 128 argmax tokens from the HF checkpoint (CPU) and from the served
  Q8_0 (no stop on EOS); first position where they diverge, per prompt.
- Recorded, not decisive: HF vs llama.cpp tokenization of the 10 rendered
  prompts (identical ids or not).

Verdict (fixed in the 2026-10-08 amendment): FAIL if top-1 agreement <= 95%
on either link (link 1: pooled top-1; link 2: "Same top p"); then phase-1
results are invalidated and phase 1 is repeated.

GGUF logits come from scripts/llama_logits.c on libllama (llama.cpp
3466812d), compiled here with gcc. Writes results/conversion_check.{json,md}.
The BF16 GGUF (~18 GB) is created if missing and deleted at the end unless
--keep-bf16.

  .venv-train/bin/python scripts/conversion_check.py [--ngl-bf16 0] [--ngl-q8 99] [--keep-bf16]

Resources: the HF side runs on CPU in bf16 (~18 GB RAM; the 16 GB card cannot
hold it). The BF16 GGUF runs on CPU by default (--ngl-bf16 0) for the same
reason; the Q8_0 runs on the GPU.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_mgsm import PROMPTS  # noqa: E402

LLAMA = Path("/home/jkzero/llama.cpp")
LLAMA_REV = "3466812d1f06728effe7c0f3c0671117f461672d"
HF_DIR = Path("/home/jkzero/models/Qwen3.5-9B-text-mtp")
Q8 = Path("/home/jkzero/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf")
Q8_SHA = "a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496"
BF16 = Path("/home/jkzero/models/Qwen3.5-9B-text-mtp-BF16/Qwen3.5-9B-text-mtp-BF16.gguf")
PY_TRAIN = ROOT / ".venv-train" / "bin" / "python"
IDS = [1, 2, 3, 4, 5]
MAX_POS = 256
N_GREEDY = 128
CTX_PPL = 128
THRESHOLD = 95.0


def log(msg):
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, **kw):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True, **kw)


def build_helper(tmp):
    exe = tmp / "llama_logits"
    run(["gcc", "-O2", "-o", exe, ROOT / "scripts" / "llama_logits.c", "-I", LLAMA / "include",
         "-I", LLAMA / "ggml" / "include", "-L", LLAMA / "build" / "bin", "-lllama",
         f"-Wl,-rpath,{LLAMA / 'build' / 'bin'}"])
    return exe


def gguf_logits(exe, model, ngl, ids, tmp):
    (tmp / "in.i32").write_bytes(np.asarray(ids, dtype=np.int32).tobytes())
    n_vocab = int(run([exe, model, ngl, "logits", tmp / "in.i32", tmp / "out.f32"]).stdout.split()[-1])
    out = np.fromfile(tmp / "out.f32", dtype=np.float32).reshape(len(ids), n_vocab)
    (tmp / "out.f32").unlink()
    return out


def gguf_greedy(exe, model, ngl, ids, tmp):
    (tmp / "in.i32").write_bytes(np.asarray(ids, dtype=np.int32).tobytes())
    run([exe, model, ngl, "greedy", tmp / "in.i32", tmp / "gen.i32", N_GREEDY])
    return np.fromfile(tmp / "gen.i32", dtype=np.int32).tolist()


def gguf_tokenize(exe, model, text, tmp):
    (tmp / "in.txt").write_text(text, encoding="utf-8")
    run([exe, model, 0, "tokenize", tmp / "in.txt", tmp / "tok.i32"])
    return np.fromfile(tmp / "tok.i32", dtype=np.int32).tolist()


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def port_up(port):
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngl-bf16", type=int, default=0)
    ap.add_argument("--ngl-q8", type=int, default=99)
    ap.add_argument("--keep-bf16", action="store_true")
    args = ap.parse_args()

    if port_up(8093):
        raise SystemExit(":8093 is up; stop the phase-1 server first")
    rev = run(["git", "-C", LLAMA, "rev-parse", "HEAD"]).stdout.strip()
    if rev != LLAMA_REV:
        raise SystemExit(f"llama.cpp at {rev}, expected {LLAMA_REV}")
    # The checkout's HEAD says nothing about the binary: a stale llama-perplexity
    # (2026-10-09: build 10434, 7e4c0a968) must not run silently.
    ppl_ver = subprocess.run([str(LLAMA / "build" / "bin" / "llama-perplexity"), "--version"],
                             capture_output=True, text=True)
    if ppl_ver.returncode != 0 or f"commit {LLAMA_REV[:9]}" not in ppl_ver.stdout + ppl_ver.stderr:
        raise SystemExit(f"llama-perplexity --version does not show commit {LLAMA_REV[:9]} "
                         f"(rc {ppl_ver.returncode}): {(ppl_ver.stdout + ppl_ver.stderr).strip()[:200]!r}")
    log("sha256 of the served GGUF")
    if sha256(Q8) != Q8_SHA:
        raise SystemExit(f"{Q8}: sha256 is not {Q8_SHA}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_DIR)
    prompts = []
    for lang in ("es", "en"):
        rows = {r["id"]: r for r in map(json.loads, (ROOT / "data" / f"mgsm_{lang}.jsonl").open(encoding="utf-8"))}
        for i in IDS:
            msgs = [{"role": "user", "content": PROMPTS[lang] + "\n\n" + rows[i]["question"]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_POS]
            prompts.append({"lang": lang, "id": i, "text": text, "ids": ids})
    res = {"date": dt.datetime.now().isoformat(timespec="seconds"), "llama_cpp": rev,
           "hf_dir": str(HF_DIR), "q8_gguf": str(Q8), "q8_sha256": Q8_SHA, "bf16_gguf": str(BF16),
           "ngl_bf16": args.ngl_bf16, "ngl_q8": args.ngl_q8, "threshold_top1_pct": THRESHOLD,
           "generation_prompt_tail": prompts[0]["text"][-40:],
           "prompts": [{"lang": p["lang"], "id": p["id"], "n_tokens": len(p["ids"])} for p in prompts]}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = build_helper(tmp)

        if not BF16.exists():
            log("converting the BF16 GGUF (baseline command, --outtype bf16)")
            BF16.parent.mkdir(parents=True, exist_ok=True)
            run([PY_TRAIN, "convert_hf_to_gguf.py", HF_DIR, "--outtype", "bf16", "--outfile", BF16],
                cwd=LLAMA, env={**os.environ, "PYTHONPATH": str(LLAMA / "gguf-py")})

        res["tokenizer_identical"] = [gguf_tokenize(exe, Q8, p["text"], tmp)[:MAX_POS] == p["ids"]
                                      for p in prompts]

        log("link 1: BF16 GGUF logits")
        g_logits = [gguf_logits(exe, BF16, args.ngl_bf16, p["ids"], tmp) for p in prompts]
        log("greedy: Q8_0")
        g_greedy = [gguf_greedy(exe, Q8, args.ngl_q8, p["ids"], tmp) for p in prompts]

        log("link 1 / greedy: HF bf16 on CPU")
        model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=torch.bfloat16, device_map="cpu").eval()
        agree = total = 0
        kls, per_prompt = [], []
        with torch.no_grad():
            for p, g in zip(prompts, g_logits):
                h = model(input_ids=torch.tensor([p["ids"]])).logits[0].float().numpy()
                v = min(h.shape[1], g.shape[1])
                lh, lg = log_softmax(h[:, :v]), log_softmax(g[:, :v])
                a = int((lh.argmax(-1) == lg.argmax(-1)).sum())
                kl = (np.exp(lh) * (lh - lg)).sum(-1)
                agree, total = agree + a, total + len(p["ids"])
                kls += kl.tolist()
                per_prompt.append({"lang": p["lang"], "id": p["id"], "top1": a, "positions": len(p["ids"]),
                                   "mean_kl": float(kl.mean()), "vocab_hf": h.shape[1], "vocab_gguf": g.shape[1]})
            greedy = []
            for p, gg in zip(prompts, g_greedy):
                x = torch.tensor([p["ids"]])
                out = model(input_ids=x, use_cache=True)
                hh = []
                for _ in range(N_GREEDY):
                    t = int(out.logits[0, -1].float().argmax())
                    hh.append(t)
                    out = model(input_ids=torch.tensor([[t]]), past_key_values=out.past_key_values, use_cache=True)
                div = next((k for k in range(N_GREEDY) if hh[k] != gg[k]), None)
                greedy.append({"lang": p["lang"], "id": p["id"], "first_divergence": div})
        del model
        res["link1"] = {"top1_agreement_pct": round(100 * agree / total, 3), "positions": total,
                        "mean_kl": float(np.mean(kls)), "per_prompt": per_prompt}
        res["greedy"] = greedy

        log("link 2: llama-perplexity KL, BF16 -> Q8_0")
        (tmp / "text.txt").write_text("".join(p["text"] for p in prompts), encoding="utf-8")
        ppl = LLAMA / "build" / "bin" / "llama-perplexity"
        run([ppl, "-m", BF16, "-f", tmp / "text.txt", "-c", CTX_PPL, "-ngl", args.ngl_bf16,
             "--kl-divergence-base", tmp / "base.kld"])
        out = run([ppl, "-m", Q8, "-c", CTX_PPL, "-ngl", args.ngl_q8,
                   "--kl-divergence-base", tmp / "base.kld", "--kl-divergence"])
        text = out.stdout + out.stderr
        m_top = re.search(r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)", text)
        m_kld = re.search(r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)", text)
        chunks = re.findall(r"^\s*(\d+)\s+[\d.]+\s+", text, flags=re.M)
        if not (m_top and m_kld):
            (ROOT / "results" / "conversion_check_link2.log").write_text(text)
            raise SystemExit("could not parse llama-perplexity output; saved results/conversion_check_link2.log")
        res["link2"] = {"same_top_pct": float(m_top.group(1)), "same_top_pm": float(m_top.group(2)),
                        "mean_kld": float(m_kld.group(1)), "mean_kld_pm": float(m_kld.group(2)),
                        "ctx": CTX_PPL, "chunks": len(chunks),
                        "positions_scored_approx": len(chunks) * (CTX_PPL - CTX_PPL // 2 - 1)}

    if not args.keep_bf16:
        BF16.unlink()
        res["bf16_deleted"] = True
    ok1 = res["link1"]["top1_agreement_pct"] > THRESHOLD
    ok2 = res["link2"]["same_top_pct"] > THRESHOLD
    res["verdict"] = "PASS" if ok1 and ok2 else "FAIL"
    (ROOT / "results" / "conversion_check.json").write_text(json.dumps(res, indent=1))
    g = ", ".join(f"{x['lang']}{x['id']}: {x['first_divergence'] if x['first_divergence'] is not None else 'none'}"
                  for x in res["greedy"])
    md = f"""# Conversion fidelity check

{res['date']}, llama.cpp `{rev}`. Served GGUF `{Q8}` (sha256 `{Q8_SHA}`) vs HF
`{HF_DIR}` (bf16, transformers, CPU). Method: `scripts/conversion_check.py`
(results/plan.md amendment 2026-10-08, NEXT.md). Raw: `results/conversion_check.json`.

**Verdict: {res['verdict']}** (FAIL if top-1 agreement <= {THRESHOLD}% on either link).

| link | top-1 agreement | KL | positions |
|---|---:|---:|---:|
| 1. HF bf16 -> BF16 GGUF (libllama, ngl {args.ngl_bf16}) | {res['link1']['top1_agreement_pct']}% | mean KL(HF‖GGUF) {res['link1']['mean_kl']:.6f} | {res['link1']['positions']} |
| 2. BF16 GGUF -> Q8_0 (llama-perplexity, -c {CTX_PPL}) | {res['link2']['same_top_pct']} ± {res['link2']['same_top_pm']}% | mean KLD {res['link2']['mean_kld']:.6f} ± {res['link2']['mean_kld_pm']:.6f} | {res['link2']['chunks']} chunks, ~{res['link2']['positions_scored_approx']} |

Greedy, {N_GREEDY} tokens, HF vs served Q8_0, first divergence (none = identical): {g}.

Tokenization HF vs llama.cpp identical on the 10 prompts: {sum(res['tokenizer_identical'])}/10 (recorded, not decisive).
"""
    (ROOT / "results" / "conversion_check.md").write_text(md)
    log(f"verdict {res['verdict']}: link1 {res['link1']['top1_agreement_pct']}%, link2 {res['link2']['same_top_pct']}%")


if __name__ == "__main__":
    main()
