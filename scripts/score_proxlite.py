#!/usr/bin/env python3
"""Score MMLU-ProX-Lite results from results/proxlite_raw.jsonl (runnable mid-run).

- Prediction = the letter right after the LAST 'Answer:' / 'Respuesta:' / '答案：'
  marker in content (ASCII or full-width colon; bold/markdown around the marker
  or the letter is allowed). No marker, or no A-J letter right after the last
  one -> unparsed: counted wrong and reported separately, never guessed.
- Same per-language table and prompt-language x detected-language matrices as
  score_mgsm.py, plus %unparsed, per-category accuracy per language, and
  paired en-vs-es / en-vs-zh counts with exact McNemar p-values.
- Reasoning tokens counted via POST /tokenize (skip with --no-tokenize).
- --raw PATH scores another file (e.g. a tagged run); --baseline PATH adds a
  paired baseline-vs-this table per language with exact McNemar p.
- --subset FILE scores only ids listed in FILE (one per line, e.g.
  results/proxlite_subset_<n>_keepdone.txt) that have all three languages.
  Without it, a warning is printed if a *_keepdone.txt subset file sits next
  to the raw file (the run was cut by proxlite_gate.sh).
- Category-weighted accuracy per language: per-category accuracy weighted by
  the category's share of the full 588-id test set (renormalized over the
  categories present). Paired McNemar always uses the raw paired counts.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import requests

from score_mgsm import DETECT_COLS, LANGS, TOKENIZE_URL, detect, mcnemar_exact, print_paired_vs_baseline

ROOT = Path(__file__).resolve().parents[1]
RAW_IN = ROOT / "results" / "proxlite_raw.jsonl"

# Category sizes of the full test set (li-lab/MMLU-ProX-Lite, test, revision
# e82aafb9, identical in en/es/zh): weights for the category-weighted accuracy.
FULL_CATEGORY_COUNTS = {
    "biology": 36, "business": 40, "chemistry": 56, "computer science": 20,
    "economics": 42, "engineering": 48, "health": 35, "history": 19, "law": 48,
    "math": 68, "other": 46, "philosophy": 25, "physics": 65, "psychology": 40,
}
FULL_N = sum(FULL_CATEGORY_COUNTS.values())
assert FULL_N == 588

# Marker, optionally wrapped in markdown: "**Answer:**", "## Respuesta:", "答案：".
MARKER_RE = re.compile(r"(?:(?i:answer|respuesta)|答案)\s*(?:\*\*|__)?\s*[:：]")
# Letter right after the marker, allowing bold/italics/code/parens/\boxed{...}.
LETTER_AFTER_RE = re.compile(r"[\s*_`$(\[{]*(?:\\(?:boxed|text|textbf)\{)?[\s*_`$(\[{]*([A-J])(?![A-Za-z])")


def parse_letter(text):
    """Letter after the last marker, or None if unparsed."""
    if not text:
        return None
    markers = list(MARKER_RE.finditer(text))
    if not markers:
        return None
    m = LETTER_AFTER_RE.match(text, markers[-1].end())
    return m.group(1) if m else None


def correct(record):
    pred = parse_letter(record.get("content"))
    return pred is not None and pred == record.get("gold")


def weighted_accuracy(cat):
    """Sum over present categories of full-set share * category accuracy,
    shares renormalized over the categories present. Returns (acc, n_cats)."""
    present = [c for c, (_, n) in cat.items() if n and c in FULL_CATEGORY_COUNTS]
    total = sum(FULL_CATEGORY_COUNTS[c] for c in present)
    if not total:
        return float("nan"), 0
    return sum(FULL_CATEGORY_COUNTS[c] / total * cat[c][0] / cat[c][1] for c in present), len(present)


def subset_filter(records, path):
    """Keep records whose id is listed in PATH and present in all LANGS."""
    wanted = {int(x) for x in path.read_text(encoding="utf-8").split()}
    langs_by_id = {}
    for rec in records:
        if rec["id"] in wanted:
            langs_by_id.setdefault(rec["id"], set()).add(rec["lang"])
    keep = {i for i, ls in langs_by_id.items() if ls >= set(LANGS)}
    print(f"--subset {path.name}: {len(keep)}/{len(wanted)} listed ids have all "
          f"{len(LANGS)} langs; scoring only those")
    return [r for r in records if r["id"] in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_IN)
    ap.add_argument("--no-tokenize", action="store_true",
                    help="skip /tokenize calls (reasoning tokens reported as nan)")
    ap.add_argument("--baseline", type=Path, metavar="PATH",
                    help="pair against this raw file (same id + lang), e.g. results/proxlite_raw.jsonl")
    ap.add_argument("--subset", type=Path, metavar="FILE",
                    help="score only ids listed in FILE that have all three langs")
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"no {args.raw} yet", file=sys.stderr)
        sys.exit(1)

    if args.subset is None:
        cut = sorted(p.name for p in args.raw.parent.glob("proxlite_subset_*_keepdone.txt"))
        if cut:
            print(f"WARNING: {', '.join(cut)} exists (run cut to a subset); scoring every "
                  f"record in {args.raw.name}; pass --subset to score only the subset")

    records = []
    with args.raw.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.subset is not None:
        records = subset_filter(records, args.subset)
    print(f"scored {len(records)} records so far")

    per_lang = {l: {
        "n": 0, "correct": 0, "unparsed": 0, "completion_tokens": [], "reasoning_tokens": [],
        "pps": [], "drafted": 0, "accepted": 0, "length": 0,
        "reasoning_lang": {}, "content_lang": {}, "cat": {},
    } for l in LANGS}
    ok = {}

    for rec in records:
        l = rec["lang"]
        bucket = per_lang[l]
        bucket["n"] += 1
        pred = parse_letter(rec.get("content"))
        is_ok = pred is not None and pred == rec.get("gold")
        ok[(rec["id"], l)] = is_ok
        bucket["correct"] += is_ok
        bucket["unparsed"] += pred is None
        cat = bucket["cat"].setdefault(rec.get("category", "?"), [0, 0])
        cat[0] += is_ok
        cat[1] += 1
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

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    print("\n=== Per language ===")
    header = (f"{'lang':<5} {'n':>4} {'acc':>7} {'%unpars':>8} {'med_compl':>10} {'med_reas_tk':>12} "
              f"{'med_pps':>9} {'mtp_acc':>9} {'%length':>8}")
    print(header)
    for l in LANGS:
        b = per_lang[l]
        acc = b["correct"] / b["n"] if b["n"] else float("nan")
        mtp = (b["accepted"] / b["drafted"]) if b["drafted"] else float("nan")
        print(f"{l:<5} {b['n']:>4} {acc:>7.4f} {pct(b['unparsed'], b['n']):>8.1f} "
              f"{med(b['completion_tokens']):>10.0f} {med(b['reasoning_tokens']):>12.0f} "
              f"{med(b['pps']):>9.2f} {mtp:>9.4f} {pct(b['length'], b['n']):>8.1f}")

    print(f"\n=== Category-weighted accuracy (weights = category share of the full {FULL_N} ids) ===")
    print(f"{'lang':<5} {'n':>4} {'acc':>7} {'w_acc':>7} {'cats':>6} {'min n/cat':>10}")
    for l in LANGS:
        b = per_lang[l]
        acc = b["correct"] / b["n"] if b["n"] else float("nan")
        wacc, ncat = weighted_accuracy(b["cat"])
        min_n = min((n for _, n in b["cat"].values()), default=0)
        print(f"{l:<5} {b['n']:>4} {acc:>7.4f} {wacc:>7.4f} {f'{ncat}/{len(FULL_CATEGORY_COUNTS)}':>6} "
              f"{min_n:>10}")

    for field, title in (("reasoning_lang", "reasoning_content"), ("content_lang", "content")):
        print(f"\n=== prompt-language x detected language of {title} ===")
        print(f"{'prompt':<8} " + "".join(f"{c:>8}" for c in DETECT_COLS))
        for l in LANGS:
            row = per_lang[l][field]
            print(f"{l:<8} " + "".join(f"{row.get(c, 0):>8}" for c in DETECT_COLS))

    print("\n=== Per-category accuracy (n) ===")
    cats = sorted({c for l in LANGS for c in per_lang[l]["cat"]})
    print(f"{'category':<18} " + "".join(f"{l:>14}" for l in LANGS) + f"{'full n':>8}")
    for c in cats:
        cells = []
        for l in LANGS:
            k, n = per_lang[l]["cat"].get(c, (0, 0))
            cells.append(f"{k / n:.3f} ({n:>3})" if n else "—")
        print(f"{c:<18} " + "".join(f"{x:>14}" for x in cells)
              + f"{FULL_CATEGORY_COUNTS.get(c, '?'):>8}")

    print("\n=== Paired vs en (ids run in both languages; unparsed = wrong) ===")
    print(f"{'pair':<9} {'n':>4} {'both✓':>6} {'both✗':>6} {'en✓ x✗':>7} {'x✓ en✗':>7} {'McNemar p':>10}")
    ids = sorted({i for i, _ in ok})
    for other in ("es", "zh"):
        common = [i for i in ids if (i, "en") in ok and (i, other) in ok]
        en_only = sum(1 for i in common if ok[(i, "en")] and not ok[(i, other)])
        other_only = sum(1 for i in common if ok[(i, other)] and not ok[(i, "en")])
        both_ok = sum(1 for i in common if ok[(i, "en")] and ok[(i, other)])
        both_bad = len(common) - both_ok - en_only - other_only
        print(f"{other + ' vs en':<9} {len(common):>4} {both_ok:>6} {both_bad:>6} {en_only:>7} "
              f"{other_only:>7} {mcnemar_exact(en_only, other_only):>10.4g}")

    if args.baseline:
        print_paired_vs_baseline(ok, args.baseline, correct)


if __name__ == "__main__":
    main()
