#!/usr/bin/env python3
"""Phase 1: MGSM baseline (es then en) against a local llama-server.

Same prompts and generation settings as the 27B baseline (imported from the
unmodified copy of es-eval's scripts/run_mgsm.py: temperature 1.0, top_p
0.95, top_k 20, seed 42, max_tokens 8192, no system prompt, thinking as the
server default). Reads data/mgsm_{lang}.jsonl, runs all ids of the first
language before starting the next (default order: es, en).

Appends one JSON line per item to results/mgsm_raw.<tag>.jsonl with the same
fields as es-eval (id, lang, gold, content, reasoning_content, finish_reason,
usage, timings, wall, tag), so score_mgsm.py --raw reads it unchanged.
Saves GET /props to results/server_props.<tag>.json before the first item.
Resumable: skips (id, lang) already present.

Options: --port (default 8093), --tag (default q14b), --langs (default es,en),
--limit N (at most N pending items, for testing),
--stop-at HH:MM (local time; no new item is started at or after it).
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mgsm import GENERATION, PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--tag", default="q14b")
    ap.add_argument("--langs", default="es,en", help="order matters (default: es,en)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stop-at", metavar="HH:MM")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    if not langs or set(langs) - set(PROMPTS):
        ap.error(f"--langs must be a subset of {','.join(PROMPTS)}")
    stop_at = None
    if args.stop_at:
        hh, mm = args.stop_at.split(":")
        stop_at = dt.datetime.now().replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)

    raw_out = RESULTS_DIR / f"mgsm_raw.{args.tag}.jsonl"
    props_out = RESULTS_DIR / f"server_props.{args.tag}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not props_out.exists():
        r = requests.get(f"{base}/props", timeout=30)
        r.raise_for_status()
        props_out.write_text(json.dumps(r.json(), indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"saved server props -> {props_out.name}")

    items = {l: [json.loads(x) for x in (ROOT / "data" / f"mgsm_{l}.jsonl").open(encoding="utf-8")
                 if x.strip()] for l in langs}
    done = set()
    if raw_out.exists():
        for line in raw_out.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                done.add((rec["id"], rec["lang"]))
            except (json.JSONDecodeError, KeyError):
                pass
    pending = [(l, it) for l in langs for it in items[l] if (it["id"], l) not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    log(f"tag={args.tag} port={args.port} langs={langs}; done={len(done)} "
        f"this run={len(pending)} stop_at={stop_at}")

    n_ok = 0
    for n, (lang, it) in enumerate(pending, start=1):
        if stop_at is not None and dt.datetime.now() >= stop_at:
            log(f"stop_at {args.stop_at} reached; stopping before id={it['id']} {lang}")
            break
        payload = {
            "messages": [{"role": "user", "content": PROMPTS[lang] + "\n\n" + it["question"]}],
            **GENERATION,
        }
        try:
            t0 = time.monotonic()
            r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=1800)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"id={it['id']} {lang}: FAILED ({e}); will retry next run")
            continue
        wall = time.monotonic() - t0
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        rec = {
            "id": it["id"],
            "lang": lang,
            "gold": it["answer_number"],
            "content": msg.get("content"),
            "reasoning_content": msg.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "timings": data.get("timings"),
            "wall": wall,
            "tag": args.tag,
        }
        with raw_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1
        u = rec["usage"] or {}
        log(f"({n}/{len(pending)}) id={it['id']} {lang} finish={rec['finish_reason']} "
            f"prompt={u.get('prompt_tokens', 0)} completion={u.get('completion_tokens', 0)} "
            f"wall={wall:.1f}s")
    log(f"run finished: {n_ok} written this run; total in file={len(done) + n_ok}/"
        f"{sum(len(v) for v in items.values())}")


if __name__ == "__main__":
    main()
