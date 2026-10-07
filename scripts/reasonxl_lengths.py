#!/usr/bin/env python3
"""Token lengths of a fixed sample of ReasonXL-SFT (one split), Qwen3.5 chat format.

Reads remotely (no full download) one row group from each of N evenly spaced
parquet shards of toroe/ReasonXL-SFT/<split>, cycling the row-group index so
the sample is not all shard heads. Each row's `messages` is rendered with the
base model's chat template (as training would see it) and tokenized with its
tokenizer. Writes results/reasonxl_<split>_lengths.csv.gz (shard, row_group,
row, source, dataset_name, n_tokens) and prints the share at or under each
candidate max length.

  .venv-train/bin/python scripts/reasonxl_lengths.py [--split spanish] [--shards 8]
"""
import argparse
import csv
import gzip
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "/home/jkzero/models/Qwen3.5-4B-hf"
LENGTHS = (4096, 6144, 8192)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="spanish")
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE)
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/toroe/ReasonXL-SFT/data/{args.split}-*.parquet"))
    step = len(files) / args.shards
    picks = [files[int(i * step)] for i in range(args.shards)]
    out = ROOT / "results" / f"reasonxl_{args.split}_lengths.csv.gz"
    rows = []
    for k, path in enumerate(picks):
        pf = pq.ParquetFile(fs.open(path))
        rg = k % pf.num_row_groups
        t = pf.read_row_group(rg, columns=["messages", "source", "dataset_name"]).to_pylist()
        texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in t]
        lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        shard = path.rsplit("/", 1)[-1]
        rows += [(shard, rg, i, r["source"], r["dataset_name"], n) for i, (r, n) in enumerate(zip(t, lens))]
        print(f"{shard} rg={rg}: {len(t)} rows", flush=True)
    with gzip.open(out, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shard", "row_group", "row", "source", "dataset_name", "n_tokens"])
        w.writerows(rows)
    n = len(rows)
    lens = sorted(r[-1] for r in rows)
    print(f"\n{n} rows from {len(picks)} shards -> {out.name}")
    print(f"median {lens[n // 2]}, p90 {lens[int(n * .9)]}, p99 {lens[int(n * .99)]}, max {lens[-1]}")
    for L in LENGTHS:
        k = sum(x <= L for x in lens)
        print(f"<= {L}: {k}/{n} = {k / n:.1%} (dropped {n - k})")


if __name__ == "__main__":
    main()
