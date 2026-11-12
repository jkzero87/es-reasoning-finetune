#!/usr/bin/env python3
"""Export data/belebele_zh.jsonl in the same format as data/belebele_{en,es}.jsonl.

Uses es-eval's load_items() (its scripts/run_belebele.py, its data/ and its
local HF cache of facebook/belebele; offline), the way SOURCE.md describes for
en/es: one row per id 1..488 with id, qid, passage, question, options, gold.

Stops without writing unless:
- the same export for en and es reproduces data/belebele_{en,es}.jsonl byte for byte;
- every id has the same qid and gold in zh as in en;
- all ids of results/belebele_sample_300.json are present.

  HF_HOME=/home/jkzero/es-eval/.hf_cache .venv/bin/python scripts/export_belebele_zh.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ES_EVAL = Path("/home/jkzero/es-eval")
sys.path.insert(0, str(ES_EVAL / "scripts"))
import run_belebele  # noqa: E402  (es-eval's copy: reads es-eval's data/)


def dump(items):
    return "".join(json.dumps({"id": i, **items[i]}, ensure_ascii=False) + "\n" for i in sorted(items))


def main():
    items = run_belebele.load_items()
    for lang in ("en", "es"):
        have = (ROOT / "data" / f"belebele_{lang}.jsonl").read_text(encoding="utf-8")
        if dump(items[lang]) != have:
            raise SystemExit(f"STOP: export of {lang} does not reproduce data/belebele_{lang}.jsonl")
    for i, ref in items["en"].items():
        zh = items["zh"].get(i)
        if zh is None or (zh["qid"], zh["gold"]) != (ref["qid"], ref["gold"]):
            raise SystemExit(f"STOP: id {i} does not match between en and zh")
    sample = json.loads((ROOT / "results" / "belebele_sample_300.json").read_text())["ids"]
    missing = [i for i in sample if i not in items["zh"]]
    if missing or len(sample) != 300:
        raise SystemExit(f"STOP: sample has {len(sample)} ids, {len(missing)} missing in zh")
    (ROOT / "data" / "belebele_zh.jsonl").write_text(dump(items["zh"]), encoding="utf-8")
    print(f"en/es exports reproduce the tracked files; zh: {len(items['zh'])} ids written, "
          f"qid and gold equal to en on all of them; sample ids present: {len(sample)}/300")


if __name__ == "__main__":
    main()
