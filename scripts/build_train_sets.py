#!/usr/bin/env python3
"""Build the phase-2 training sets (Spanish) and the 2b control (English).

Same 2,000 ReasonXL-SFT items in both languages (results/plan.md, phase 2/2b).

1. Pool: the first POOL_ROWS global rows of toroe/ReasonXL-SFT (spanish shard
   00000 = 39,664 rows; the splits are row-aligned and interleaved across
   source datasets, so the head is not one source). English rows are read from
   the English shards covering the same global range and paired by
   `row_index` (checked equal, with dataset_name and ds_uid).
2. Length: each sample's full chat-formatted text (Qwen3.5 chat template of
   `messages`, tokenized with the Qwen3.5 tokenizer, no extra special tokens).
   Keep a pair only if BOTH versions are <= MAX_LEN (4096) tokens.
3. Overlap filter (drop the pair if either side matches), against MGSM en/es
   (250 each), GSM8K train + test (openai/gsm8k, main; 8,792 questions) and
   Belebele en/es (488 each: passage, question, options). Text is normalised
   (NFKC, lowercase, non-alphanumerics -> space, whitespace collapsed).
   - exact: the normalised user turn equals a normalised benchmark question;
   - near-duplicate: the user turn shares at least one word 13-gram with any
     benchmark text in the same language (GSM8K is English-only; its
     Spanish counterparts are caught through the pair rule).
4. Draw N_TRAIN pairs with random.Random(SEED).sample over the eligible pairs
   sorted by row_index.

Writes data/train/{es,en}.jsonl (gitignored: messages + ids), and commits-size
results/train_ids.json (row_index list, in draw order) and
results/train_sets_stats.json (counts at each step, length stats per language,
dataset_name mix).

  .venv-train/bin/python scripts/build_train_sets.py
"""
import json
import random
import re
import statistics as st
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "/home/jkzero/models/Qwen3.5-9B-text"
REPO = "datasets/toroe/ReasonXL-SFT/data"
POOL_ROWS = 39_664
MAX_LEN = 4096
NGRAM = 13
N_TRAIN = 2000
SEED = 20261007
COLS = ["messages", "source", "dataset_name", "ds_uid", "row_index"]


def norm(text):
    t = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(re.sub(r"[^\w]+", " ", t).split())


def grams(text, n=NGRAM):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def read_rows(fs, split, n_rows):
    out = []
    for path in sorted(fs.glob(f"{REPO}/{split}-*.parquet")):
        pf = pq.ParquetFile(fs.open(path))
        for rg in range(pf.num_row_groups):
            out += pf.read_row_group(rg, columns=COLS).to_pylist()
            if len(out) >= n_rows:
                return out[:n_rows]
    return out


def user_turn(msgs):
    return "\n".join(m["content"] for m in msgs if m["role"] == "user")


def lengths(tok, rows):
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]


def describe(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "mean": round(st.mean(xs), 1), "median": xs[n // 2], "p90": xs[int(n * .9)],
            "max": xs[-1], "sum": sum(xs)}


def main():
    fs = HfFileSystem()
    tok = AutoTokenizer.from_pretrained(BASE)
    stats = {"pool_rows": POOL_ROWS, "max_len": MAX_LEN, "ngram": NGRAM, "seed": SEED}

    es = read_rows(fs, "spanish", POOL_ROWS)
    en = read_rows(fs, "english", POOL_ROWS)
    pairs = []
    for a, b in zip(es, en):
        if (a["row_index"], a["dataset_name"], a["ds_uid"]) != (b["row_index"], b["dataset_name"], b["ds_uid"]):
            raise SystemExit(f"misaligned: es {a['row_index']} vs en {b['row_index']}")
        pairs.append((a, b))
    stats["pairs_aligned"] = len(pairs)

    len_es, len_en = lengths(tok, es), lengths(tok, en)
    keep = [i for i in range(len(pairs)) if len_es[i] <= MAX_LEN and len_en[i] <= MAX_LEN]
    stats["dropped_es_too_long"] = sum(x > MAX_LEN for x in len_es)
    stats["dropped_en_too_long"] = sum(x > MAX_LEN for x in len_en)
    stats["both_within_max_len"] = len(keep)

    # Benchmarks.
    bench = {"en": [], "es": []}
    for lang in ("en", "es"):
        bench[lang] += [json.loads(l)["question"] for l in (ROOT / "data" / f"mgsm_{lang}.jsonl").open()]
        for l in (ROOT / "data" / f"belebele_{lang}.jsonl").open():
            r = json.loads(l)
            bench[lang] += [r["passage"], r["question"], " ".join(r["options"])]
    gsm = load_dataset("openai/gsm8k", "main")
    gsm_q = [r["question"] for s in ("train", "test") for r in gsm[s]]
    bench["en"] += gsm_q
    stats["benchmark_texts"] = {k: len(v) for k, v in bench.items()}
    stats["gsm8k_questions"] = len(gsm_q)
    exact = {lang: {norm(t) for t in v} for lang, v in bench.items()}
    ng = {lang: set().union(*(grams(t) for t in v)) for lang, v in bench.items()}

    hits = Counter()
    eligible = []
    for i in keep:
        a, b = pairs[i]
        side = {"es": user_turn(a["messages"]), "en": user_turn(b["messages"])}
        why = []
        for lang, text in side.items():
            if norm(text) in exact[lang]:
                why.append(f"exact_{lang}")
            elif grams(text) & ng[lang]:
                why.append(f"ngram_{lang}")
        if why:
            for w in why:
                hits[w] += 1
            hits["pairs_dropped"] += 1
        else:
            eligible.append(i)
    stats["overlap"] = dict(hits)
    stats["eligible_pairs"] = len(eligible)

    eligible.sort(key=lambda i: pairs[i][0]["row_index"])
    draw = random.Random(SEED).sample(eligible, N_TRAIN)
    out_dir = ROOT / "data" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, k, lens in (("es", 0, len_es), ("en", 1, len_en)):
        with (out_dir / f"{lang}.jsonl").open("w", encoding="utf-8") as f:
            for i in draw:
                r = pairs[i][k]
                f.write(json.dumps({**r, "n_tokens": lens[i]}, ensure_ascii=False) + "\n")
        stats[f"length_{lang}"] = describe([lens[i] for i in draw])
    stats["length_ratio_es_en_sum"] = round(stats["length_es"]["sum"] / stats["length_en"]["sum"], 3)
    stats["dataset_name_mix"] = dict(Counter(pairs[i][0]["dataset_name"] for i in draw).most_common())
    (ROOT / "results" / "train_ids.json").write_text(json.dumps(
        {"seed": SEED, "n": N_TRAIN, "row_index": [pairs[i][0]["row_index"] for i in draw]}))
    (ROOT / "results" / "train_sets_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
