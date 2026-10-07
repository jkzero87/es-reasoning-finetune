#!/usr/bin/env python3
"""Derive a text-only Qwen3.5 checkpoint from the official multimodal one.

Drops the vision tower (model.visual.*) and the MTP head (mtp.*), renames
model.language_model.* -> model.*, keeps lm_head.* (if untied), and writes
config.json = the original text_config with architectures
["Qwen3_5ForCausalLM"]. Tokenizer/chat-template files are copied. Tensors are
copied bit-for-bit (same dtype), only names change. Prints parameter counts
before/after.

  .venv-train/bin/python scripts/make_text_only.py SRC_DIR DST_DIR
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

COPY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "LICENSE"]
SHARD_BYTES = 4 * 2**30


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    tcfg = dict(cfg["text_config"])
    tcfg["architectures"] = ["Qwen3_5ForCausalLM"]
    tcfg["model_type"] = cfg["text_config"].get("model_type", "qwen3_5_text")
    tcfg.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    tcfg.setdefault("torch_dtype", cfg.get("torch_dtype", "bfloat16"))
    (dst / "config.json").write_text(json.dumps(tcfg, indent=2))
    for name in COPY:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    count = Counter()
    shard, shard_bytes, n_shard, weight_map = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if shard:
            n_shard += 1
            fn = f"model-{n_shard:05d}.safetensors"
            save_file(shard, dst / fn, metadata={"format": "pt"})
            weight_map.update({k: fn for k in shard})
            shard, shard_bytes = {}, 0

    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                part = ("vision" if k.startswith("model.visual.") else
                        "mtp" if k.startswith("mtp.") else "text")
                count[part] += t.numel()
                if part != "text":
                    continue
                nk = k.replace("model.language_model.", "model.", 1)
                shard[nk] = t.contiguous()
                shard_bytes += t.numel() * t.element_size()
                if shard_bytes >= SHARD_BYTES:
                    flush()
    flush()
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum((dst / fn).stat().st_size for fn in set(weight_map.values()))},
         "weight_map": weight_map}, indent=2))
    tot = sum(count.values())
    print(json.dumps({"params_total_before": tot, "params_vision": count["vision"],
                      "params_mtp": count["mtp"], "params_text_only_after": count["text"],
                      "tensors_written": len(weight_map),
                      "lm_head_present": any(k.startswith("lm_head") for k in weight_map),
                      "tie_word_embeddings": tcfg["tie_word_embeddings"]}))


if __name__ == "__main__":
    main()
