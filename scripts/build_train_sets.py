#!/usr/bin/env python3
"""Build the training sets for arms A, B and the control (results/plan.md).

Same items in all three arms (amendment 2026-10-07 18:2x):
- A (primary): Spanish messages, but the assistant turn's reasoning
  (everything up to and including `</think>`) is the ENGLISH version's, and
  the text after `</think>` (final answer) is the SPANISH version's.
- B: all Spanish.  Control: all English.

Steps:
1. Pool: the first POOL_ROWS global rows of toroe/ReasonXL-SFT (spanish shard
   00000 = 39,664 rows); English rows paired by `row_index` (checked equal with
   dataset_name and ds_uid). Cached in data/train/pool_{es,en}.jsonl.
2. Split point: the Spanish and the English versions must each contain
   exactly one `</think>` (whole sample), and it must be in the last message,
   which must be the assistant's (the Qwen3.5 template keeps reasoning only in
   the final assistant turn). Otherwise dropped and counted.
3. Length: every arm's version (A, B, control) <= MAX_LEN tokens, full
   chat-formatted sample (Qwen3.5 template + tokenizer). Longer: dropped.
4. Overlap filter (unchanged): normalised user turn exactly equal to a
   benchmark question, or any shared normalised word 13-gram with a benchmark
   text of the same language (MGSM en/es, GSM8K train+test, Belebele en/es
   passage/question/options). A match on either language drops the item.
5. Domain: "math" if "math" appears in dataset_name or source (case-
   insensitive); otherwise the dataset_name. Draw N_TRAIN items with
   random.Random(SEED): min(800, eligible math) math items, the rest split
   across the other domains in proportion to their eligible counts (largest
   remainder), each stratum sampled from its items sorted by row_index.

Writes data/train/{A,B,control}.jsonl (gitignored), results/train_ids.json
(row_index + domain, draw order) and results/train_sets_stats.json.

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
OUT = ROOT / "data" / "train"
BASE = "/home/jkzero/models/Qwen3.5-9B-text"
REPO = "datasets/toroe/ReasonXL-SFT/data"
POOL_ROWS = 39_664
MAX_LEN = 4096
NGRAM = 13
N_TRAIN = 2000
N_MATH = 800
SEED = 20261007
COLS = ["messages", "source", "dataset_name", "ds_uid", "row_index"]
END = "</think>"


def norm(text):
    t = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(re.sub(r"[^\w]+", " ", t).split())


def grams(text, n=NGRAM):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def read_rows(fs, split, n_rows, cache):
    if cache.exists():
        return [json.loads(l) for l in cache.open(encoding="utf-8")]
    out = []
    for path in sorted(fs.glob(f"{REPO}/{split}-*.parquet")):
        pf = pq.ParquetFile(fs.open(path))
        for rg in range(pf.num_row_groups):
            out += pf.read_row_group(rg, columns=COLS).to_pylist()
            if len(out) >= n_rows:
                break
        if len(out) >= n_rows:
            break
    out = out[:n_rows]
    with cache.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def user_turn(msgs):
    return "\n".join(m["content"] for m in msgs if m["role"] == "user")


def split_ok(msgs):
    total = sum(m["content"].count(END) for m in msgs)
    last = msgs[-1]
    return total == 1 and last["role"] == "assistant" and END in last["content"]


def arm_a(es_msgs, en_msgs):
    reasoning = en_msgs[-1]["content"].split(END)[0] + END
    answer = es_msgs[-1]["content"].split(END, 1)[1]
    return es_msgs[:-1] + [{"role": "assistant", "content": reasoning + answer}]


def lengths(tok, convs):
    texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
    out = []
    for i in range(0, len(texts), 2000):
        out += [len(x) for x in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"]]
    return out


def domain(r):
    return "math" if "math" in (r["dataset_name"] + " " + (r["source"] or "")).lower() else r["dataset_name"]


def describe(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "mean": round(st.mean(xs), 1), "median": xs[n // 2], "p90": xs[int(n * .9)],
            "max": xs[-1], "sum": sum(xs)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    tok = AutoTokenizer.from_pretrained(BASE)
    stats = {"pool_rows": POOL_ROWS, "max_len": MAX_LEN, "ngram": NGRAM, "seed": SEED,
             "n_train": N_TRAIN, "n_math_target": N_MATH}

    es = read_rows(fs, "spanish", POOL_ROWS, OUT / "pool_es.jsonl")
    en = read_rows(fs, "english", POOL_ROWS, OUT / "pool_en.jsonl")
    for a, b in zip(es, en):
        if (a["row_index"], a["dataset_name"], a["ds_uid"]) != (b["row_index"], b["dataset_name"], b["ds_uid"]):
            raise SystemExit(f"misaligned: es {a['row_index']} vs en {b['row_index']}")
    stats["pairs_aligned"] = len(es)

    idx = [i for i in range(len(es)) if split_ok(es[i]["messages"]) and split_ok(en[i]["messages"])]
    stats["dropped_think_split"] = {
        "es_bad": sum(not split_ok(r["messages"]) for r in es),
        "en_bad": sum(not split_ok(r["messages"]) for r in en),
        "pairs_dropped": len(es) - len(idx)}

    conv = {"A": [arm_a(es[i]["messages"], en[i]["messages"]) for i in idx],
            "B": [es[i]["messages"] for i in idx],
            "control": [en[i]["messages"] for i in idx]}
    lens = {arm: lengths(tok, c) for arm, c in conv.items()}
    fit = [k for k in range(len(idx)) if all(lens[a][k] <= MAX_LEN for a in lens)]
    stats["too_long"] = {a: sum(x > MAX_LEN for x in lens[a]) for a in lens}
    stats["all_arms_within_max_len"] = len(fit)

    bench = {"en": [], "es": []}
    for lang in ("en", "es"):
        bench[lang] += [json.loads(l)["question"] for l in (ROOT / "data" / f"mgsm_{lang}.jsonl").open()]
        for l in (ROOT / "data" / f"belebele_{lang}.jsonl").open():
            r = json.loads(l)
            bench[lang] += [r["passage"], r["question"], " ".join(r["options"])]
    gsm = load_dataset("openai/gsm8k", "main")
    bench["en"] += [r["question"] for s in ("train", "test") for r in gsm[s]]
    stats["benchmark_texts"] = {k: len(v) for k, v in bench.items()}
    exact = {lang: {norm(t) for t in v} for lang, v in bench.items()}
    ng = {lang: set().union(*(grams(t) for t in v)) for lang, v in bench.items()}

    hits = Counter()
    eligible = []
    for k in fit:
        i = idx[k]
        why = []
        for lang, text in (("es", user_turn(es[i]["messages"])), ("en", user_turn(en[i]["messages"]))):
            if norm(text) in exact[lang]:
                why.append(f"exact_{lang}")
            elif grams(text) & ng[lang]:
                why.append(f"ngram_{lang}")
        if why:
            hits.update(why)
            hits["pairs_dropped"] += 1
        else:
            eligible.append(k)
    stats["overlap"] = dict(hits)
    stats["eligible"] = len(eligible)

    by_dom = {}
    for k in sorted(eligible, key=lambda k: es[idx[k]]["row_index"]):
        by_dom.setdefault(domain(es[idx[k]]), []).append(k)
    stats["eligible_by_domain"] = {d: len(v) for d, v in sorted(by_dom.items(), key=lambda x: -len(x[1]))}
    n_math = min(N_MATH, len(by_dom.get("math", [])))
    rest = N_TRAIN - n_math
    others = {d: v for d, v in by_dom.items() if d != "math"}
    tot = sum(len(v) for v in others.values())
    raw = {d: rest * len(v) / tot for d, v in others.items()}
    quota = {d: int(x) for d, x in raw.items()}
    for d in sorted(raw, key=lambda d: -(raw[d] - quota[d]))[:rest - sum(quota.values())]:
        quota[d] += 1
    quota["math"] = n_math
    rng = random.Random(SEED)
    draw = []
    for d in sorted(quota):
        draw += [(d, k) for k in rng.sample(by_dom[d], quota[d])]
    rng.shuffle(draw)
    stats["drawn_by_domain"] = {d: quota[d] for d in sorted(quota, key=lambda d: -quota[d])}

    for arm in ("A", "B", "control"):
        with (OUT / f"{arm}.jsonl").open("w", encoding="utf-8") as f:
            for d, k in draw:
                i = idx[k]
                f.write(json.dumps({"row_index": es[i]["row_index"], "dataset_name": es[i]["dataset_name"],
                                    "source": es[i]["source"], "domain": d, "arm": arm,
                                    "messages": conv[arm][k], "n_tokens": lens[arm][k]},
                                   ensure_ascii=False) + "\n")
        stats[f"length_{arm}"] = describe([lens[arm][k] for _, k in draw])
        stats[f"length_{arm}_math"] = describe([lens[arm][k] for d, k in draw if d == "math"]) if n_math else None
    stats["length_ratio_sum"] = {f"{a}/control": round(stats[f"length_{a}"]["sum"] / stats["length_control"]["sum"], 3)
                                 for a in ("A", "B")}
    (ROOT / "results" / "train_ids.json").write_text(json.dumps(
        {"seed": SEED, "n": len(draw), "items": [[es[idx[k]]["row_index"], d] for d, k in draw]}))
    (ROOT / "results" / "train_sets_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
