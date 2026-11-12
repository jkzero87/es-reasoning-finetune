#!/usr/bin/env python3
"""Rows per raw file and language for the 2026-11-12 queue, against the expected counts.

Prints one line per (file, language): OK or MISSING/EXTRA. Exit 0 only if every
count is exact and no (id, lang) is duplicated. Used by scripts/run_queue.sh to
decide whether results/QUEUE_DONE is written, and by bench/TAREAS_AGENTE.md T1.

  .venv/bin/python scripts/queue_counts.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    ("results/mgsm_raw.base9b.jsonl", {"es": 250, "en": 250, "zh": 250}),
    ("results/belebele_raw.base9b.jsonl", {"es": 300, "en": 300, "zh": 300}),
    ("results/mgsm_raw.noreason9b.jsonl", {"es": 250, "en": 250, "zh": 250}),
    ("results/belebele_raw.noreason9b.jsonl", {"es": 300, "en": 300, "zh": 300}),
    ("results/proxlite_raw.prox9b.jsonl", {"es": 60, "en": 60, "zh": 60}),
]


def main():
    ok = True
    for rel, want in EXPECTED:
        p = ROOT / rel
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []
        n = collections.Counter(r["lang"] for r in rows)
        dup = len(rows) - len({(r["id"], r["lang"]) for r in rows})
        for lang, w in want.items():
            state = "OK" if n[lang] == w else ("MISSING" if n[lang] < w else "EXTRA")
            ok &= state == "OK"
            print(f"{state:<8} {rel} {lang} {n[lang]}/{w}")
        other = sorted(set(n) - set(want))
        if other or dup:
            ok = False
            print(f"EXTRA    {rel} other languages {other}, duplicate (id, lang) {dup}")
    print("ALL COUNTS OK" if ok else "COUNTS DIFFER FROM EXPECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
