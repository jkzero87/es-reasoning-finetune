#!/usr/bin/env python3
"""F1 baseline: run Belebele (en/es/zh) against the local llama-server.

- Uses the 488 aligned questions from data/belebele_{en,es,zh}.jsonl (question
  rows, in file order -> id 1..488). Options (mc_answer1..4) and
  correct_answer_num are reloaded from the local HF cache (facebook/belebele,
  offline) when missing from the data files, matched by (link, question_number).
- Prompt per language: passage + question + options A-D, answer with only the
  letter on a final line ('Answer: X' / 'Respuesta: X' / '答案：X').
- Interleaves by id (id1-en, id1-es, id1-zh, id2-...) so server drift spreads.
- POSTs /v1/chat/completions: temperature 1.0, top_p 0.95, top_k 20, seed 42,
  max_tokens 8192 (no reasoning_effort).
- Appends one JSON line per item to results/belebele_raw.jsonl:
  id, lang, gold (letter), content, reasoning_content, finish_reason, usage,
  timings, wall (seconds), plus qid (link:question_number) for traceability.
  Resumable: skips (id, lang) already present.
- Saves GET /props to results/server_props_belebele.json before the first item.

Optional: --langs en,es (subset), --system FILE (sent as a system message),
--tag NAME (writes results/belebele_raw.<tag>.jsonl and server_props.<tag>.json;
records then also carry "system"/"tag"). No flags = the baseline run.

Run under nohup; logs go wherever stdout/stderr are redirected.
"""
import argparse
import json
import re
import os
import sys
import time
from pathlib import Path

# Options/answers come from the already-downloaded HF cache; never hit the Hub.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_OUT = RESULTS_DIR / "belebele_raw.jsonl"
PROPS_OUT = RESULTS_DIR / "server_props_belebele.json"
BASE = "http://127.0.0.1:8092"

LANGS = ["en", "es", "zh"]
HF_CONFIG = {"en": "eng_Latn", "es": "spa_Latn", "zh": "zho_Hans"}
LETTERS = ["A", "B", "C", "D"]
PROMPTS = {
    "en": "Read the passage and answer the question. End with a final line "
          "'Answer: X', where X is only the letter of the correct option (A, B, C or D).",
    "es": "Lee el texto y responde la pregunta. Termina con una línea final "
          "'Respuesta: X', donde X es solo la letra de la opción correcta (A, B, C o D).",
    "zh": "阅读短文并回答问题。最后一行写 '答案：X'，其中 X 只是正确选项的字母（A、B、C 或 D）。",
}
LABELS = {
    "en": ("Passage", "Question", "Options"),
    "es": ("Texto", "Pregunta", "Opciones"),
    "zh": ("短文", "问题", "选项"),
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
                    help="write results/belebele_raw.<tag>.jsonl and server_props.<tag>.json")
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


def load_items():
    """Return {lang: {id: item}} with passage, question, options, gold letter."""
    items = {}
    for lang in LANGS:
        rows = [json.loads(line) for line in
                (DATA_DIR / f"belebele_{lang}.jsonl").open(encoding="utf-8")]
        questions = [r for r in rows if "question" in r]
        hf = None
        if any("correct_answer_num" not in r or "mc_answer1" not in r for r in questions):
            try:
                ds = load_dataset("facebook/belebele", HF_CONFIG[lang], split="test")
            except Exception as e:
                print(f"ERROR: failed to load facebook/belebele config={HF_CONFIG[lang]!r} "
                      f"from cache: {e}", file=sys.stderr)
                raise
            hf = {(r["link"], r["question_number"]): r for r in ds}
        items[lang] = {}
        for i, q in enumerate(questions, start=1):
            src = q if "correct_answer_num" in q and "mc_answer1" in q \
                else hf[(q["link"], q["question_number"])]
            items[lang][i] = {
                "qid": f"{q['link']}:{q['question_number']}",
                "passage": q["flores_passage"],
                "question": q["question"],
                "options": [src[f"mc_answer{k}"] for k in range(1, 5)],
                "gold": LETTERS[int(src["correct_answer_num"]) - 1],
            }
    # Alignment sanity: same qid and gold across languages for every id.
    for i in items["en"]:
        ref = items["en"][i]
        for lang in LANGS[1:]:
            other = items[lang][i]
            if (other["qid"], other["gold"]) != (ref["qid"], ref["gold"]):
                raise SystemExit(f"misaligned id={i}: en={ref['qid']},{ref['gold']} "
                                 f"{lang}={other['qid']},{other['gold']}")
    return items


def build_prompt(lang, item):
    passage_l, question_l, options_l = LABELS[lang]
    options = "\n".join(f"{letter}) {text}" for letter, text in zip(LETTERS, item["options"]))
    return (f"{PROMPTS[lang]}\n\n{passage_l}:\n{item['passage']}\n\n"
            f"{question_l}: {item['question']}\n\n{options_l}:\n{options}")


def make_record(i, lang, item, data, wall):
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    return {
        "id": i,
        "lang": lang,
        "gold": item["gold"],
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "timings": data.get("timings"),
        "wall": wall,
        "qid": item["qid"],
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
    langs, system, tag = parse_args()
    raw_out = RESULTS_DIR / f"belebele_raw.{tag}.jsonl" if tag else RAW_OUT
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
    max_id = max(max(items[l]) for l in LANGS)
    done = done_pairs(raw_out, langs)
    pending = [(i, l) for i in range(1, max_id + 1) for l in langs if (i, l) not in done]
    total = max_id * len(langs)
    log(f"ids 1..{max_id}, langs {langs}; done={len(done)} pending={len(pending)} total={total}")
    if not pending:
        log("nothing to do.")
        return

    n_ok = 0
    for n, (i, lang) in enumerate(pending, start=1):
        item = items[lang][i]
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
            log(f"id={i} {lang}: FAILED ({e}); will retry next run")
            continue
        wall = time.monotonic() - t0
        rec = make_record(i, lang, item, data, wall)
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
