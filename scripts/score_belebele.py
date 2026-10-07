#!/usr/bin/env python3
"""Score Belebele results from results/belebele_raw.jsonl (runnable mid-run).

- correct = last standalone A-D letter in content equals gold.
- Same per-language table and prompt-language x detected-language matrices
  (content and reasoning_content) as score_mgsm.py.
- Reasoning tokens counted via POST /tokenize (skip with --no-tokenize).
- --raw PATH scores another file (e.g. a tagged run); --baseline PATH adds a
  paired baseline-vs-this table per language with exact McNemar p.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import requests

from score_mgsm import DETECT_COLS, LANGS, TOKENIZE_URL, detect, print_paired_vs_baseline

ROOT = Path(__file__).resolve().parents[1]
RAW_IN = ROOT / "results" / "belebele_raw.jsonl"

LETTER_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def last_letter(text):
    if not text:
        return None
    letters = LETTER_RE.findall(text)
    return letters[-1] if letters else None


def correct(record):
    pred = last_letter(record.get("content"))
    return pred is not None and pred == record.get("gold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_IN)
    ap.add_argument("--no-tokenize", action="store_true",
                    help="skip /tokenize calls (reasoning tokens reported as nan)")
    ap.add_argument("--baseline", type=Path, metavar="PATH",
                    help="pair against this raw file (same id + lang), e.g. results/belebele_raw.jsonl")
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"no {args.raw} yet", file=sys.stderr)
        sys.exit(1)

    records = []
    with args.raw.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"scored {len(records)} records so far")

    per_lang = {l: {
        "n": 0, "correct": 0, "completion_tokens": [], "reasoning_tokens": [],
        "pps": [], "drafted": 0, "accepted": 0, "length": 0,
        "reasoning_lang": {}, "content_lang": {},
    } for l in LANGS}
    ok = {}

    for rec in records:
        l = rec["lang"]
        bucket = per_lang[l]
        bucket["n"] += 1
        ok[(rec["id"], l)] = correct(rec)
        if ok[(rec["id"], l)]:
            bucket["correct"] += 1
        usage = rec.get("usage") or {}
        if "completion_tokens" in usage:
            bucket["completion_tokens"].append(usage["completion_tokens"])
        tt = rec.get("timings") or {}
        if tt.get("predicted_per_second") is not None:
            bucket["pps"].append(tt["predicted_per_second"])
        if tt.get("draft_n") is not None:
            bucket["drafted"] += tt["draft_n"]
        if tt.get("draft_n_accepted") is not None:
            bucket["accepted"] += tt["draft_n_accepted"]
        if rec.get("finish_reason") == "length":
            bucket["length"] += 1
        rl = detect(rec.get("reasoning_content"))
        bucket["reasoning_lang"][rl] = bucket["reasoning_lang"].get(rl, 0) + 1
        cl = detect(rec.get("content"))
        bucket["content_lang"][cl] = bucket["content_lang"].get(cl, 0) + 1
        rc = rec.get("reasoning_content")
        if rc and rc.strip() and not args.no_tokenize:
            try:
                r = requests.post(TOKENIZE_URL, json={"content": rc}, timeout=60)
                r.raise_for_status()
                bucket["reasoning_tokens"].append(len(r.json()["tokens"]))
            except Exception as e:
                print(f"  tokenize failed for id={rec['id']} {l}: {e}", file=sys.stderr)

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    print("\n=== Per language ===")
    header = (f"{'lang':<5} {'n':>4} {'acc':>7} {'med_compl':>10} {'med_reas_tk':>12} "
              f"{'med_pps':>9} {'mtp_acc':>9} {'%length':>8}")
    print(header)
    for l in LANGS:
        b = per_lang[l]
        acc = b["correct"] / b["n"] if b["n"] else float("nan")
        mtp = (b["accepted"] / b["drafted"]) if b["drafted"] else float("nan")
        pct_len = 100.0 * b["length"] / b["n"] if b["n"] else float("nan")
        print(f"{l:<5} {b['n']:>4} {acc:>7.4f} {med(b['completion_tokens']):>10.0f} "
              f"{med(b['reasoning_tokens']):>12.0f} {med(b['pps']):>9.2f} "
              f"{mtp:>9.4f} {pct_len:>8.1f}")

    for field, title in (("reasoning_lang", "reasoning_content"), ("content_lang", "content")):
        print(f"\n=== prompt-language x detected language of {title} ===")
        print(f"{'prompt':<8} " + "".join(f"{c:>8}" for c in DETECT_COLS))
        for l in LANGS:
            row = per_lang[l][field]
            print(f"{l:<8} " + "".join(f"{row.get(c, 0):>8}" for c in DETECT_COLS))

    if args.baseline:
        print_paired_vs_baseline(ok, args.baseline, correct)


if __name__ == "__main__":
    main()
