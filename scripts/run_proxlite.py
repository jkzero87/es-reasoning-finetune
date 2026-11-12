#!/usr/bin/env python3
"""F1 baseline: run MMLU-ProX-Lite (en/es/zh) against the local llama-server.

- Loads li-lab/MMLU-ProX-Lite test (588 items per language) offline from the
  local HF cache, pinned to revision e82aafb9460529687d3c7e51b401d8dd1dd309dd.
- Zero-shot. Lists only non-empty options, labeled A..J; the gold letter is
  remapped to that listing. Prompt per language: think step by step, end with a
  final line 'Answer: X' / 'Respuesta: X' / '答案：X'.
- Interleaves by question_id (qid-en, qid-es, qid-zh, next qid...).
- POSTs /v1/chat/completions: temperature 1.0, top_p 0.95, top_k 20, seed 42,
  max_tokens 16384 (no reasoning_effort).
- Appends one JSON line per item to results/proxlite_raw.jsonl:
  id (question_id), lang, gold (letter), content, reasoning_content,
  finish_reason, usage, timings, wall (seconds), plus category and n_options.
  Resumable: skips (id, lang) already present.
- Saves GET /props to results/server_props_proxlite.json before the first item.

Optional: --langs en,es (subset), --system FILE (sent as a system message),
--tag NAME (writes results/proxlite_raw.<tag>.jsonl and server_props.<tag>.json;
records then also carry "system"/"tag"). No flags = the baseline run.
--limit N runs only N question_ids, stratified by category (proportional,
at least 1 per category, seed 42), each in every selected language; the ids
are saved to results/proxlite_subset_<N>.txt. Records are unchanged, so a
later run without --limit on the same file just resumes the rest.
--ids FILE runs exactly the question_ids listed (one per line) instead.
--keep-done (with --limit) forces every id already done in all selected
languages into the subset (raising N if needed; per-category allocation never
drops below a category's finished ids) and fills the rest as above; the ids
go to results/proxlite_subset_<n>_keepdone.txt.

Run under nohup; logs go wherever stdout/stderr are redirected.
"""
import argparse
import json
import random
import re
import os
import sys
import time
from pathlib import Path

# The dataset is already in the HF cache; never hit the Hub.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW_OUT = RESULTS_DIR / "proxlite_raw.jsonl"
PROPS_OUT = RESULTS_DIR / "server_props_proxlite.json"
BASE = "http://127.0.0.1:8092"

DATASET = "li-lab/MMLU-ProX-Lite"
REVISION = "e82aafb9460529687d3c7e51b401d8dd1dd309dd"
LANGS = ["en", "es", "zh"]
LETTERS = "ABCDEFGHIJ"
PROMPTS = {
    "en": "Answer the following multiple-choice question. Think step by step, then end "
          "with a final line 'Answer: X', where X is the letter of the correct option.",
    "es": "Responde la siguiente pregunta de opción múltiple. Piensa paso a paso y termina "
          "con una línea final 'Respuesta: X', donde X es la letra de la opción correcta.",
    "zh": "回答下面的选择题。请逐步思考，最后一行写 '答案：X'，其中 X 是正确选项的字母。",
}
LABELS = {
    "en": ("Question", "Options"),
    "es": ("Pregunta", "Opciones"),
    "zh": ("问题", "选项"),
}

GENERATION = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 42,
    "max_tokens": 16384,
}
SUBSET_SEED = 42


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langs", default=",".join(LANGS),
                    help=f"comma-separated subset of {','.join(LANGS)} (default: all)")
    ap.add_argument("--system", type=Path, metavar="FILE",
                    help="text file sent as a system message (requires --tag)")
    ap.add_argument("--tag", metavar="NAME",
                    help="write results/proxlite_raw.<tag>.jsonl and server_props.<tag>.json")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="run only N question_ids, stratified by category (seed 42)")
    ap.add_argument("--keep-done", action="store_true",
                    help="with --limit: include every id already done in all selected languages")
    ap.add_argument("--ids", type=Path, metavar="FILE",
                    help="run exactly these question_ids (one per line; not with --limit)")
    args = ap.parse_args(argv)
    langs = [l for l in LANGS if l in {x.strip() for x in args.langs.split(",")}]
    unknown = {x.strip() for x in args.langs.split(",")} - set(LANGS) - {""}
    if unknown or not langs:
        ap.error(f"--langs must be a non-empty subset of {','.join(LANGS)}, got {args.langs!r}")
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        ap.error("--tag may only contain letters, digits, '_', '-' and '.'")
    if args.system is not None and args.tag is None:
        ap.error("--system requires --tag, so tagged runs never mix into the baseline file")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be a positive integer")
    if args.keep_done and args.limit is None:
        ap.error("--keep-done requires --limit")
    if args.ids is not None and args.limit is not None:
        ap.error("--ids and --limit are exclusive")
    ids = None
    if args.ids is not None:
        try:
            ids = sorted({int(x) for x in args.ids.read_text(encoding="utf-8").split()})
        except (OSError, ValueError) as e:
            ap.error(f"--ids {args.ids}: {e}")
        if not ids:
            ap.error(f"--ids {args.ids}: no ids")
    system = args.system.read_text(encoding="utf-8").strip() if args.system is not None else None
    return langs, system, args.tag, args.limit, args.keep_done, ids


def load_items():
    """Return {lang: {question_id: item}} with question, options, gold letter, category."""
    items = {}
    for lang in LANGS:
        try:
            ds = load_dataset(DATASET, lang, split="test", revision=REVISION)
        except Exception as e:
            print(f"ERROR: failed to load {DATASET} config={lang!r} rev={REVISION[:8]} "
                  f"from cache: {e}", file=sys.stderr)
            raise
        items[lang] = {}
        for row in ds:
            kept = [k for k in range(10) if (row[f"option_{k}"] or "").strip()]
            if row["answer_index"] not in kept:
                raise SystemExit(f"{lang} qid={row['question_id']}: answer on an empty option")
            items[lang][row["question_id"]] = {
                "question": row["question"],
                "options": [row[f"option_{k}"] for k in kept],
                "gold": LETTERS[kept.index(row["answer_index"])],
                "category": row["category"],
            }
    # Alignment sanity: same ids, gold and option count across languages.
    for lang in LANGS[1:]:
        if set(items[lang]) != set(items["en"]):
            raise SystemExit(f"question_id sets differ between en and {lang}")
    for qid, ref in items["en"].items():
        for lang in LANGS[1:]:
            other = items[lang][qid]
            if (other["gold"], len(other["options"])) != (ref["gold"], len(ref["options"])):
                raise SystemExit(f"misaligned qid={qid}: en={ref['gold']}/{len(ref['options'])} "
                                 f"{lang}={other['gold']}/{len(other['options'])}")
    return items


def stratified_subset(items, n, seed=SUBSET_SEED, keep=frozenset()):
    """Pick n question_ids, allocated to categories in proportion to their size
    (largest remainder, at least 1 each), sampled with a fixed seed. Sorted.
    Ids in keep are always chosen: a category gets at least its kept ids and n
    grows if they need more room. Empty keep = the plain stratified subset."""
    by_cat = {}
    for qid, item in items.items():
        by_cat.setdefault(item["category"], []).append(qid)
    cats = sorted(by_cat)
    total = len(items)
    if not len(cats) <= n <= total:
        raise SystemExit(f"--limit must be between {len(cats)} (one per category) and {total}, got {n}")
    kept = {c: sorted(q for q in by_cat[c] if q in keep) for c in cats}
    floor = {c: max(1, len(kept[c])) for c in cats}
    n = max(n, sum(floor.values()))
    quota = {c: n * len(by_cat[c]) / total for c in cats}
    alloc = {c: max(floor[c], int(quota[c])) for c in cats}
    # Hand out what is left by largest remainder; take back (from categories
    # above their floor) by smallest remainder if the floors overshot.
    by_remainder = sorted(cats, key=lambda c: (-(quota[c] - int(quota[c])), c))
    while sum(alloc.values()) < n:
        c = next(c for c in by_remainder if alloc[c] < len(by_cat[c]))
        alloc[c] += 1
        by_remainder.remove(c)
        by_remainder.append(c)
    while sum(alloc.values()) > n:
        c = next(c for c in reversed(by_remainder) if alloc[c] > floor[c])
        alloc[c] -= 1
        by_remainder.remove(c)
        by_remainder.insert(0, c)
    rng = random.Random(seed)
    chosen = []
    for c in cats:
        rest = [q for q in sorted(by_cat[c]) if q not in keep]
        chosen += kept[c] + rng.sample(rest, alloc[c] - len(kept[c]))
    return sorted(chosen), alloc


def build_prompt(lang, item):
    question_l, options_l = LABELS[lang]
    options = "\n".join(f"{LETTERS[k]}) {text}" for k, text in enumerate(item["options"]))
    return f"{PROMPTS[lang]}\n\n{question_l}: {item['question']}\n\n{options_l}:\n{options}"


def make_record(qid, lang, item, data, wall):
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    return {
        "id": qid,
        "lang": lang,
        "gold": item["gold"],
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "timings": data.get("timings"),
        "wall": wall,
        "category": item["category"],
        "n_options": len(item["options"]),
    }


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
    langs, system, tag, limit, keep_done, ids = parse_args()
    raw_out = RESULTS_DIR / f"proxlite_raw.{tag}.jsonl" if tag else RAW_OUT
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

    items = load_items()
    qids = sorted(items["en"])
    done = done_pairs(raw_out, langs)
    if limit is not None:
        keep = frozenset(q for q in qids if all((q, l) in done for l in langs)) \
            if keep_done else frozenset()
        qids, alloc = stratified_subset(items["en"], limit, keep=keep)
        name = f"proxlite_subset_{len(qids)}_keepdone.txt" if keep_done \
            else f"proxlite_subset_{limit}.txt"
        subset_out = RESULTS_DIR / name
        subset_out.write_text("".join(f"{q}\n" for q in qids), encoding="utf-8")
        log(f"--limit {limit}: {len(qids)} ids -> {subset_out.name}"
            + (f" ({len(keep)} finished ids kept)" if keep_done else "")
            + "; per category " + ", ".join(f"{c}={k}" for c, k in alloc.items()))
    if ids is not None:
        unknown = [q for q in ids if q not in items["en"]]
        if unknown:
            raise SystemExit(f"--ids: unknown question_ids {unknown[:10]}")
        qids = ids
        log(f"--ids: {len(qids)} ids from the file")
    pending = [(q, l) for q in qids for l in langs if (q, l) not in done]
    total = len(qids) * len(langs)
    log(f"question_id {qids[0]}..{qids[-1]} ({len(qids)} ids), langs {langs}; "
        f"done={len(done)} pending={len(pending)} total={total}")
    if not pending:
        log("nothing to do.")
        return

    n_ok = 0
    for n, (qid, lang) in enumerate(pending, start=1):
        item = items[lang][qid]
        payload = {
            "messages": ([{"role": "system", "content": system}] if system is not None else [])
                        + [{"role": "user", "content": build_prompt(lang, item)}],
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
            log(f"id={qid} {lang}: FAILED ({e}); will retry next run")
            continue
        wall = time.monotonic() - t0
        rec = make_record(qid, lang, item, data, wall)
        if system is not None:
            rec["system"] = system
        if tag:
            rec["tag"] = tag
        with raw_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1
        tt = rec.get("timings") or {}
        n_pr = (rec.get("usage") or {}).get("completion_tokens", 0)
        log(f"({n}/{len(pending)}) id={qid} {lang} finish={rec['finish_reason']} "
            f"completion={n_pr} wall={wall:.1f}s "
            f"predicted_per_s={tt.get('predicted_per_second', 0):.2f} "
            f"draft={tt.get('draft_n', 0)}/{tt.get('draft_n_accepted', 0)}")
    log(f"run finished: {n_ok}/{len(pending)} succeeded; total in file={len(done) + n_ok}/{total}")


if __name__ == "__main__":
    main()
