#!/usr/bin/env python3
"""Descriptive efficiency table per model x benchmark x language (results/plan.md,
amendment 2026-11-12). No gate.

Per language: n, accuracy, mean prompt / completion / total tokens (server
usage), count finish_reason=length, count empty answers (content missing or
only whitespace), count looping outputs (reasoning_content or content holds
one 60-character span at >= 5 non-overlapping positions), and
SUCCESS_PER_1K = (correct AND finish_reason == "stop") / (completion tokens
summed over ALL records) * 1000.
Between languages: es/en and zh/en of SUCCESS_PER_1K as %, the es/en and zh/en
ratio of mean total tokens, and the paired count of ids where es (zh) used
more total tokens than en.

Correctness comes from the existing scorers, imported unchanged:
score_mgsm.correct, score_belebele.correct, score_proxlite.parse_letter.

Two ways to run it (python: .venv/bin/python, the scorers need lingua):
  scripts/efficiency_baseline.py --raw FILE --tag TAG [--bench mgsm|belebele|proxlite]
        [--model NAME] [--ids FILE]      one raw file -> one table on stdout
  scripts/efficiency_baseline.py --all   every entry of results/efficiency_manifest.json
        -> results/efficiency_baseline.md and results/efficiency_baseline.json
        (an entry whose raw file does not exist yet is listed as "not run")

--bench defaults to the first word of the file name (mgsm_raw... -> mgsm).
--ids: a .json with an "ids" list or a text file of ids; only those ids count.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_belebele  # noqa: E402
import score_mgsm  # noqa: E402
import score_proxlite  # noqa: E402

MANIFEST = ROOT / "results" / "efficiency_manifest.json"
OUT_MD = ROOT / "results" / "efficiency_baseline.md"
OUT_JSON = ROOT / "results" / "efficiency_baseline.json"
LANGS = ["en", "es", "zh"]
SPAN, REPEATS = 60, 5
CORRECT = {
    "mgsm": score_mgsm.correct,
    "belebele": score_belebele.correct,
    "proxlite": lambda r: score_proxlite.parse_letter(r.get("content")) == r.get("gold")
    and r.get("gold") is not None,
}


def looping(text):
    """True if one SPAN-character span sits at >= REPEATS non-overlapping positions."""
    if not text or len(text) < SPAN * REPEATS:
        return False
    last, count = {}, {}
    for i in range(len(text) - SPAN + 1):
        w = text[i:i + SPAN]
        if w not in last or i - last[w] >= SPAN:
            last[w] = i
            count[w] = count.get(w, 0) + 1
            if count[w] >= REPEATS:
                return True
    return False


def load_ids(path):
    if path is None:
        return None
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    if p.suffix == ".json":
        return set(json.loads(p.read_text())["ids"])
    return {int(x) for x in p.read_text().split()}


def resolve(path):
    return Path(path) if Path(path).is_absolute() else ROOT / path


def table(raw, bench, ids=None):
    """Stats for one raw file. Returns {"langs": {lang: stats}, "pairs": {lang: ...}}."""
    recs = {l: {} for l in LANGS}
    for line in resolve(raw).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["lang"] not in recs or (ids is not None and r["id"] not in ids):
            continue
        if r["id"] in recs[r["lang"]]:
            raise SystemExit(f"STOP: duplicate (id, lang) = ({r['id']}, {r['lang']}) in {raw}")
        recs[r["lang"]][r["id"]] = r
    ok_fn = CORRECT[bench]
    out = {"langs": {}, "pairs": {}}
    for l in LANGS:
        rs = list(recs[l].values())
        if not rs:
            continue
        ok = [bool(ok_fn(r)) for r in rs]
        compl = [r["usage"]["completion_tokens"] for r in rs]
        prompt = [r["usage"]["prompt_tokens"] for r in rs]
        good = sum(1 for r, o in zip(rs, ok) if o and r.get("finish_reason") == "stop")
        out["langs"][l] = {
            "n": len(rs),
            "accuracy": sum(ok) / len(rs),
            "mean_prompt": statistics.mean(prompt),
            "mean_completion": statistics.mean(compl),
            "mean_total": statistics.mean(p + c for p, c in zip(prompt, compl)),
            "length": sum(r.get("finish_reason") == "length" for r in rs),
            "empty": sum(not (r.get("content") or "").strip() for r in rs),
            "looping": sum(looping(r.get("reasoning_content")) or looping(r.get("content")) for r in rs),
            "correct_and_stop": good,
            "completion_tokens_all": sum(compl),
            "success_per_1k": good / sum(compl) * 1000,
        }
    en = out["langs"].get("en")
    for l in ("es", "zh"):
        s = out["langs"].get(l)
        if not en or not s:
            continue
        common = sorted(set(recs[l]) & set(recs["en"]))
        tot = lambda r: r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]  # noqa: E731
        out["pairs"][l] = {
            "success_per_1k_pct_of_en": 100 * s["success_per_1k"] / en["success_per_1k"],
            "total_token_ratio": s["mean_total"] / en["mean_total"],
            "paired_ids": len(common),
            "ids_more_tokens_than_en": sum(tot(recs[l][i]) > tot(recs["en"][i]) for i in common),
        }
    return out


def render(t):
    lines = ["| lang | n | accuracy | mean prompt | mean completion | mean total | length | empty | looping "
             "| correct AND stop | SUCCESS_PER_1K |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for l, s in t["langs"].items():
        lines.append(f"| {l} | {s['n']} | {s['accuracy']:.4f} | {s['mean_prompt']:.1f} | {s['mean_completion']:.1f} "
                     f"| {s['mean_total']:.1f} | {s['length']} | {s['empty']} | {s['looping']} "
                     f"| {s['correct_and_stop']} | {s['success_per_1k']:.4f} |")
    if t["pairs"]:
        lines += ["", "| pair | SUCCESS_PER_1K, % of en | total-token ratio (old metric) | ids with more total tokens than en |",
                  "|---|---:|---:|---:|"]
        for l, p in t["pairs"].items():
            lines.append(f"| {l}/en | {p['success_per_1k_pct_of_en']:.1f}% | {p['total_token_ratio']:.4f} "
                         f"| {p['ids_more_tokens_than_en']}/{p['paired_ids']} |")
    return "\n".join(lines)


def run_all():
    entries = json.loads(MANIFEST.read_text())
    res, md = [], [
        "# Efficiency baseline (descriptive, no gate)", "",
        "Generated by `scripts/efficiency_baseline.py --all` from `results/efficiency_manifest.json`; "
        "definitions in `results/plan.md`, amendment 2026-11-12. Do not edit by hand.", "",
        "SUCCESS_PER_1K = (correct AND finish_reason = stop) / (completion tokens over ALL records) × 1000. "
        "Higher is better. \"% of en\" below 100 means that language gets fewer correct, complete answers "
        "per 1,000 generated tokens than English.", ""]
    for e in entries:
        title = f"## {e['model']} · {e['bench']} · {e['setting']} (tag `{e['tag']}`)"
        item = {**e, "status": "not run", "langs": {}, "pairs": {}}
        if resolve(e["raw"]).exists():
            t = table(e["raw"], e["bench"], load_ids(e["ids"]))
            item.update(t)
            short = [f"{l} {s['n']}/{e['expected_n']}" for l, s in t["langs"].items() if s["n"] != e["expected_n"]]
            item["status"] = "ok" if t["langs"] and not short else ("incomplete" if t["langs"] else "not run")
            item["missing_langs"] = [l for l in LANGS if l not in t["langs"]]
        md += [title, "", f"Raw: `{e['raw']}`" + (f", ids: `{e['ids']}`" if e["ids"] else "") + "."]
        if item["status"] == "not run":
            md += ["", "Not run yet (no records).", ""]
        else:
            if item["status"] == "incomplete":
                md += ["", "**INCOMPLETE:** " + ", ".join(short) + "."]
            if item["missing_langs"]:
                md += ["", "No records yet for: " + ", ".join(item["missing_langs"]) + "."]
            md += ["", render(item), ""]
        res.append(item)
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    for r in res:
        print(f"{r['status']:<10} {r['model']} | {r['bench']} | {r['setting']} | "
              + ", ".join(f"{l} n={s['n']}" for l, s in r["langs"].items()))
    print(f"tables: {len(res)} entries, {sum(r['status'] != 'not run' for r in res)} with data -> "
          f"{OUT_MD.relative_to(ROOT)}, {OUT_JSON.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--raw")
    ap.add_argument("--tag")
    ap.add_argument("--bench", choices=sorted(CORRECT))
    ap.add_argument("--model", default="")
    ap.add_argument("--ids")
    args = ap.parse_args()
    if args.all:
        return run_all()
    if not args.raw or not args.tag:
        ap.error("give --all, or --raw FILE --tag TAG")
    bench = args.bench or Path(args.raw).name.split("_")[0]
    if bench not in CORRECT:
        ap.error(f"cannot tell the benchmark from {args.raw}; give --bench")
    print(f"## {args.model} {bench} (tag `{args.tag}`), raw `{args.raw}`\n")
    print(render(table(args.raw, bench, load_ids(args.ids))))


if __name__ == "__main__":
    main()
