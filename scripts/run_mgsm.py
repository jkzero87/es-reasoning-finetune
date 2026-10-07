#!/usr/bin/env python3
"""F1 baseline: run MGSM (en/es/zh) against the local llama-server.

- Loads mgsm test split (en, es, zh) from HF (question + answer_number).
- Interleaves by id (id0-en, id0-es, id0-zh, id1-...) so server drift spreads.
- POSTs /v1/chat/completions: temperature 1.0, top_p 0.95, top_k 20, seed 42,
  max_tokens 8192 (no reasoning_effort).
- Appends one JSON line per item to results/mgsm_raw.jsonl:
  id, lang, gold, content, reasoning_content, finish_reason, usage, timings,
  wall (seconds). Resumable: skips (id, lang) already present.
- Saves GET /props to results/server_props.json before the first item.

Optional: --langs en,es (subset), --system FILE (sent as a system message),
--tag NAME (writes results/mgsm_raw.<tag>.jsonl and server_props.<tag>.json;
records then also carry "system"/"tag"). No flags = the baseline run.

Run under nohup; logs go wherever stdout/stderr are redirected.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW_OUT = RESULTS_DIR / "mgsm_raw.jsonl"
PROPS_OUT = RESULTS_DIR / "server_props.json"
BASE = "http://127.0.0.1:8092"

LANGS = ["en", "es", "zh"]
PROMPTS = {
    "en": "Solve step by step. End with a final line 'Answer: <number>'.",
    "es": "Resuelve paso a paso. Termina con una línea final 'Respuesta: <número>'.",
    "zh": "请逐步解答。最后一行写 '答案：<数字>'.",
}

GENERATION = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 42,
    "max_tokens": 8192,
}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langs", default=",".join(LANGS),
                    help=f"comma-separated subset of {','.join(LANGS)} (default: all)")
    ap.add_argument("--system", type=Path, metavar="FILE",
                    help="text file sent as a system message (requires --tag)")
    ap.add_argument("--tag", metavar="NAME",
                    help="write results/mgsm_raw.<tag>.jsonl and server_props.<tag>.json")
    args = ap.parse_args(argv)
    langs = [l for l in LANGS if l in {x.strip() for x in args.langs.split(",")}]
    unknown = {x.strip() for x in args.langs.split(",")} - set(LANGS) - {""}
    if unknown or not langs:
        ap.error(f"--langs must be a non-empty subset of {','.join(LANGS)}, got {args.langs!r}")
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        ap.error("--tag may only contain letters, digits, '_', '-' and '.'")
    if args.system is not None and args.tag is None:
        ap.error("--system requires --tag, so tagged runs never mix into the baseline file")
    system = args.system.read_text(encoding="utf-8").strip() if args.system is not None else None
    return langs, system, args.tag


def load_questions():
    qs = {}
    for lang in LANGS:
        try:
            ds = load_dataset("juletxara/mgsm", lang, split="test")
        except Exception as e:
            print(f"ERROR: failed to load juletxara/mgsm config={lang!r}: {e}", file=sys.stderr)
            raise
        qs[lang] = {i: (row["question"], row["answer_number"])
                    for i, row in enumerate(ds, start=1)}
    return qs


def done_pairs(raw_out, langs):
    done = set()
    if raw_out.exists():
        with raw_out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec["lang"] in langs:
                        done.add((rec["id"], rec["lang"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    langs, system, tag = parse_args()
    raw_out = RESULTS_DIR / f"mgsm_raw.{tag}.jsonl" if tag else RAW_OUT
    props_out = RESULTS_DIR / f"server_props.{tag}.json" if tag else PROPS_OUT
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if tag:
        log(f"tag={tag} -> {raw_out.name}, {props_out.name}; langs={langs}; "
            f"system={'none' if system is None else repr(system[:80]) + f' ({len(system)} chars)'}")
    # Save server props before the first item (idempotent: only if absent).
    if not props_out.exists():
        r = requests.get(f"{BASE}/props", timeout=30)
        r.raise_for_status()
        props_out.write_text(json.dumps(r.json(), indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"saved server props -> {props_out.name}")

    qs = load_questions()
    max_id = max(max(qs[l]) for l in LANGS)
    done = done_pairs(raw_out, langs)
    pending = [(i, l) for i in range(1, max_id + 1) for l in langs if (i, l) not in done]
    total = max_id * len(langs)
    log(f"ids 1..{max_id}, langs {langs}; done={len(done)} pending={len(pending)} total={total}")
    if not pending:
        log("nothing to do.")
        return

    n_ok = 0
    for n, (i, lang) in enumerate(pending, start=1):
        question, gold = qs[lang][i]
        payload = {
            "messages": ([{"role": "system", "content": system}] if system is not None else [])
                        + [{"role": "user", "content": PROMPTS[lang] + "\n\n" + question}],
            "temperature": GENERATION["temperature"],
            "top_p": GENERATION["top_p"],
            "top_k": GENERATION["top_k"],
            "seed": GENERATION["seed"],
            "max_tokens": GENERATION["max_tokens"],
        }
        try:
            t0 = time.monotonic()
            r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=1800)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"id={i} {lang}: FAILED ({e}); will retry next run")
            continue
        wall = time.monotonic() - t0
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        rec = {
            "id": i,
            "lang": lang,
            "gold": gold,
            "content": msg.get("content"),
            "reasoning_content": msg.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "timings": data.get("timings"),
            "wall": wall,
        }
        if system is not None:
            rec["system"] = system
        if tag:
            rec["tag"] = tag
        with raw_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1
        tt = rec.get("timings") or {}
        n_pr = (rec.get("usage") or {}).get("completion_tokens", 0)
        log(f"({n}/{len(pending)}) id={i} {lang} finish={rec['finish_reason']} "
            f"completion={n_pr} wall={wall:.1f}s "
            f"predicted_per_s={tt.get('predicted_per_second', 0):.2f} "
            f"draft={tt.get('draft_n', 0)}/{tt.get('draft_n_accepted', 0)}")
    log(f"run finished: {n_ok}/{len(pending)} succeeded; total in file={len(done) + n_ok}/{total}")


if __name__ == "__main__":
    main()
