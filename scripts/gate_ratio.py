#!/usr/bin/env python3
"""Phase-1 GATE ratio (results/plan.md): MGSM es/en, mean total tokens per question.

Metric as in results/plan.md: mean total tokens per question (prompt +
reasoning + answer), from the server's `usage` (prompt_tokens +
completion_tokens). GATE: train only if the MGSM es/en ratio >= 1.15, applied
only on all 250 MGSM pairs. The ratio is the es mean over the en mean.

Also prints the completion-only ratio, INFORMATIONAL (not the gate).

The gate is NOT applied until the conversion check (NEXT.md c) passes
(results/plan.md, amendment 2026-10-08).

  python3 scripts/gate_ratio.py [--raw results/mgsm_raw.base9b.jsonl] > results/gate_mgsm.base9b.txt
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_IN = ROOT / "results" / "mgsm_raw.base9b.jsonl"
LANGS = ["es", "en"]
N_IDS = 250
THRESHOLD = 1.15


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=RAW_IN)
    args = ap.parse_args()

    recs = {l: {} for l in LANGS}
    for line in args.raw.open(encoding="utf-8"):
        r = json.loads(line)
        if r["lang"] in recs:
            if r["id"] in recs[r["lang"]]:
                raise SystemExit(f"duplicate (id, lang) = ({r['id']}, {r['lang']})")
            recs[r["lang"]][r["id"]] = r
    paired = set(recs["es"]) & set(recs["en"])

    print(f"raw: {args.raw.relative_to(ROOT) if args.raw.is_relative_to(ROOT) else args.raw}")
    print(f"raw sha256: {hashlib.sha256(args.raw.read_bytes()).hexdigest()}")
    print(f"records: es {len(recs['es'])}, en {len(recs['en'])}; pairs {len(paired)}/{N_IDS}")
    if len(paired) != N_IDS or any(len(recs[l]) != N_IDS for l in LANGS):
        print(f"INCOMPLETE: the gate is applied only on all {N_IDS} MGSM pairs")

    means = {}
    print()
    print(f"{'lang':<5} {'n':>4} {'mean_prompt':>12} {'mean_compl':>11} {'mean_total':>11}")
    for l in LANGS:
        us = [recs[l][i]["usage"] for i in sorted(paired)]
        prompt = statistics.mean(u["prompt_tokens"] for u in us)
        compl = statistics.mean(u["completion_tokens"] for u in us)
        total = statistics.mean(u["prompt_tokens"] + u["completion_tokens"] for u in us)
        means[l] = (total, compl)
        print(f"{l:<5} {len(us):>4} {prompt:>12.1f} {compl:>11.1f} {total:>11.1f}")

    ratio = means["es"][0] / means["en"][0]
    print()
    print(f"GATE metric, es/en mean total tokens (prompt + completion): {ratio:.4f} "
          f"(threshold >= {THRESHOLD})")
    print(f"informational only, es/en mean completion tokens: {means['es'][1] / means['en'][1]:.4f}")
    print()
    print("The gate is NOT applied until the conversion check (NEXT.md c) passes "
          "(results/plan.md, amendment 2026-10-08).")


if __name__ == "__main__":
    main()
