#!/usr/bin/env python3
"""DESCRIPTIVE token means per language for any phase-1/phase-3 raw file. NOT the gate
(the gate is scripts/gate_ratio.py, MGSM only).

Same definition as results/plan.md's metric: per record, usage.prompt_tokens +
usage.completion_tokens; mean per language over the records present; es/en
ratio of the means only when both languages are present. Partial files are
reported as such (n per language is printed).

  python3 scripts/token_means.py --raw results/belebele_raw.base9b.jsonl > results/token_means.belebele.base9b.txt
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--expected-per-lang", type=int, default=300,
                    help="items per language in a complete run (only used to flag partial files)")
    args = ap.parse_args()

    recs: dict[str, dict] = {}
    for line in args.raw.open(encoding="utf-8"):
        r = json.loads(line)
        key = (r["id"], r["lang"])
        if r["id"] in recs.setdefault(r["lang"], {}):
            raise SystemExit(f"duplicate (id, lang) = {key}")
        recs[r["lang"]][r["id"]] = r

    full = args.raw.resolve()
    raw = full.relative_to(ROOT) if full.is_relative_to(ROOT) else full
    print("DESCRIPTIVE, not the gate (gate: scripts/gate_ratio.py, MGSM).")
    print(f"raw: {raw}")
    print(f"raw sha256: {hashlib.sha256(args.raw.read_bytes()).hexdigest()}")
    print()
    print(f"{'lang':<5} {'n':>4} {'mean_prompt':>12} {'mean_compl':>11} {'mean_total':>11}")
    means = {}
    for lang in sorted(recs, key=lambda l: (l != "es", l)):
        us = [r["usage"] for r in recs[lang].values()]
        total = statistics.mean(u["prompt_tokens"] + u["completion_tokens"] for u in us)
        means[lang] = total
        flag = "" if len(us) == args.expected_per_lang else f"  (partial: {len(us)}/{args.expected_per_lang})"
        print(f"{lang:<5} {len(us):>4} {statistics.mean(u['prompt_tokens'] for u in us):>12.1f} "
              f"{statistics.mean(u['completion_tokens'] for u in us):>11.1f} {total:>11.1f}{flag}")
    print()
    complete = all(len(recs.get(l, {})) == args.expected_per_lang for l in ("es", "en"))
    if complete:
        print(f"es/en mean total tokens: {means['es'] / means['en']:.4f}")
    else:
        print("es/en ratio: not computed (a language is missing or partial)")


if __name__ == "__main__":
    main()
