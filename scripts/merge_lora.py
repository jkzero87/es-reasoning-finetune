#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the bf16 text-only base and re-attach the
base MTP head (results/plan.md: fine-tuned model = adapter merged into the bf16
text-only weights, then the same conversion; amendment 2026-10-07 18:00: for
serving, the base MTP head is added back untrained). Called by
scripts/merge_to_gguf.sh; CPU only (~18 GB RAM for the bf16 9B).

  .venv-train/bin/python scripts/merge_lora.py ADAPTER_DIR OUT_DIR

OUT_DIR gets: the merged text tensors under exactly the base checkpoint's
names and shapes (checked), the mtp.* tensors copied bit-for-bit from the
served baseline's source checkpoint (Qwen3.5-9B-text-mtp), and that
checkpoint's config.json and tokenizer/template files, so OUT_DIR has the same
layout as Qwen3.5-9B-text-mtp and converts with the same command.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

BASE = Path("/home/jkzero/models/Qwen3.5-9B-text")        # training base (no MTP)
BASE_MTP = Path("/home/jkzero/models/Qwen3.5-9B-text-mtp")  # served baseline source (with mtp.*)
COPY = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "LICENSE"]
SHARD_BYTES = 4 * 2**30


def read_all(d):
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for fn in sorted(set(idx.values())):
        with safe_open(d / fn, "pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def main():
    adapter, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists")
    base = read_all(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}

    missing = sorted(set(base) - set(sd))
    if missing:
        raise SystemExit(f"merged model lacks base tensors: {missing[:5]} ({len(missing)})")
    changed = 0
    for k, t in base.items():
        if sd[k].shape != t.shape or sd[k].dtype != t.dtype:
            raise SystemExit(f"{k}: merged {sd[k].shape} {sd[k].dtype} vs base {t.shape} {t.dtype}")
        changed += not torch.equal(sd[k], t)
    extra = sorted(set(sd) - set(base))  # e.g. a tied lm_head view; not written

    mtp = {k: v for k, v in read_all(BASE_MTP).items() if k.startswith("mtp.")}
    if not mtp:
        raise SystemExit(f"no mtp.* tensors in {BASE_MTP}")

    out.mkdir(parents=True)
    tensors = {k: sd[k] for k in base} | mtp
    shard, size, n, wmap = {}, 0, 0, {}
    for k, t in tensors.items():
        shard[k] = t
        size += t.numel() * t.element_size()
        if size >= SHARD_BYTES:
            n += 1
            save_file(shard, out / f"model-{n:05d}.safetensors", metadata={"format": "pt"})
            wmap.update({x: f"model-{n:05d}.safetensors" for x in shard})
            shard, size = {}, 0
    if shard:
        n += 1
        save_file(shard, out / f"model-{n:05d}.safetensors", metadata={"format": "pt"})
        wmap.update({x: f"model-{n:05d}.safetensors" for x in shard})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum((out / f).stat().st_size for f in set(wmap.values()))},
         "weight_map": wmap}, indent=2))
    for name in COPY:
        if (BASE_MTP / name).exists():
            shutil.copy2(BASE_MTP / name, out / name)
    print(json.dumps({"adapter": str(adapter), "out": str(out), "text_tensors": len(base),
                      "text_tensors_changed_by_merge": changed, "mtp_tensors": len(mtp),
                      "extra_state_dict_keys_not_written": extra}))


if __name__ == "__main__":
    main()
