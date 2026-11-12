#!/usr/bin/env python3
"""Rebuild the training base ~/models/Qwen3.5-9B-text from ~/models/Qwen3.5-9B-text-mtp.

The two checkpoints come from scripts/make_text_only.py on the same official
weights and differ only in the 15 mtp.* tensors (--keep-mtp). This drops them
and writes the rest unchanged (same names, same dtype, bit-for-bit), with the
same config.json and tokenizer files and the same 4 GiB sharding rule.

Stops before writing unless the parameter counts are the ones in
results/feasibility.md: text-only 8,953,803,264 and MTP 243,290,624.
Refuses if the destination already exists. CPU only.

  .venv-train/bin/python scripts/rebuild_text_from_mtp.py
"""
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/home/jkzero/models/Qwen3.5-9B-text-mtp")
DST = Path("/home/jkzero/models/Qwen3.5-9B-text")
TEXT_PARAMS, MTP_PARAMS = 8_953_803_264, 243_290_624
SHARD_BYTES = 4 * 2**30


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DST.exists():
        raise SystemExit(f"STOP: {DST} already exists")
    files = sorted(SRC.glob("*.safetensors"))
    text = mtp = 0
    for f in files:
        with safe_open(f, "pt") as st:
            for k in st.keys():
                n = 1
                for d in st.get_slice(k).get_shape():
                    n *= d
                if k.startswith("mtp."):
                    mtp += n
                else:
                    text += n
    print(f"params: text {text:,} (expected {TEXT_PARAMS:,}), mtp {mtp:,} (expected {MTP_PARAMS:,})")
    if (text, mtp) != (TEXT_PARAMS, MTP_PARAMS):
        raise SystemExit("STOP: parameter counts differ from results/feasibility.md; nothing written")

    DST.mkdir(parents=True)
    for p in SRC.iterdir():
        if p.suffix != ".safetensors" and p.name != "model.safetensors.index.json":
            shutil.copy2(p, DST / p.name)
    shard, shard_bytes, n_shard, weight_map = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if shard:
            n_shard += 1
            fn = f"model-{n_shard:05d}.safetensors"
            save_file(shard, DST / fn, metadata={"format": "pt"})
            weight_map.update({k: fn for k in shard})
            shard, shard_bytes = {}, 0

    for f in files:
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if k.startswith("mtp."):
                    continue
                t = st.get_tensor(k)
                shard[k] = t.contiguous()
                shard_bytes += t.numel() * t.element_size()
                if shard_bytes >= SHARD_BYTES:
                    flush()
    flush()
    (DST / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum((DST / fn).stat().st_size for fn in set(weight_map.values()))},
         "weight_map": weight_map}, indent=2))
    print(f"tensors written: {len(weight_map)} in {n_shard} shards -> {DST}")
    for p in sorted(DST.iterdir()):
        print(f"{sha256(p)}  {p.name}  {p.stat().st_size}")


if __name__ == "__main__":
    main()
