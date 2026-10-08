#!/usr/bin/env bash
# Trained arm -> served GGUF (results/plan.md: identical path to the baseline).
#   scripts/merge_to_gguf.sh ARM        (ARM = A | B | control)
# 1. merge checkpoints/ARM/final into the bf16 text-only base (CPU) and re-attach
#    the base MTP tensors (scripts/merge_lora.py) -> ~/models/Qwen3.5-9B-text-mtp-ARM
#    (same layout as Qwen3.5-9B-text-mtp, the served baseline's source);
# 2. convert with the baseline's command (llama.cpp 3466812d, .venv-train, no
#    --no-mtp, --outtype q8_0) -> ~/models/Qwen3.5-9B-text-mtp-ARM-Q8_0/*.gguf;
# 3. compare with the baseline GGUF a96c8c42 (scripts/gguf_compare.py): same
#    tensors/shapes/types, MTP block byte-identical; print sha256.
# The merged HF dir (~18 GB) is deleted after a passing check unless KEEP_HF=1
# (disk: 3 arms x (18 + 9.2) GB does not fit beside the current models).
# CPU only; does not touch :8092 / :8093.
set -euo pipefail
ARM=${1:?usage: merge_to_gguf.sh A|B|control}
case "$ARM" in A|B|control) ;; *) echo "bad arm: $ARM" >&2; exit 2 ;; esac
REPO=/home/jkzero/es-reasoning-finetune
LLAMA=/home/jkzero/llama.cpp
LLAMA_REV=3466812d1f06728effe7c0f3c0671117f461672d
PY=$REPO/.venv-train/bin/python
ADAPTER=$REPO/checkpoints/$ARM/final
HF=/home/jkzero/models/Qwen3.5-9B-text-mtp-$ARM
OUTDIR=/home/jkzero/models/Qwen3.5-9B-text-mtp-$ARM-Q8_0
GGUF=$OUTDIR/Qwen3.5-9B-text-mtp-$ARM-Q8_0.gguf
BASE_GGUF_SHA=a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496

[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "no adapter in $ADAPTER" >&2; exit 1; }
[ ! -e "$GGUF" ] || { echo "$GGUF exists" >&2; exit 1; }
[ "$(git -C "$LLAMA" rev-parse HEAD)" = "$LLAMA_REV" ] || { echo "llama.cpp is not at $LLAMA_REV" >&2; exit 1; }

echo "[$(date -Is)] merge $ADAPTER -> $HF"
"$PY" "$REPO/scripts/merge_lora.py" "$ADAPTER" "$HF"

echo "[$(date -Is)] convert -> $GGUF"
mkdir -p "$OUTDIR"
(cd "$LLAMA" && PYTHONPATH="$LLAMA/gguf-py" "$PY" convert_hf_to_gguf.py "$HF" \
    --outtype q8_0 --outfile "$GGUF")

echo "[$(date -Is)] compare with the baseline GGUF"
BASE_GGUF=/home/jkzero/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf
[ "$(sha256sum "$BASE_GGUF" | cut -d' ' -f1)" = "$BASE_GGUF_SHA" ] || { echo "baseline GGUF sha256 mismatch" >&2; exit 1; }
PYTHONPATH="$LLAMA/gguf-py" "$PY" "$REPO/scripts/gguf_compare.py" "$GGUF" "$BASE_GGUF"

echo "[$(date -Is)] $(stat -c %s "$GGUF") bytes, sha256 $(sha256sum "$GGUF" | cut -d' ' -f1)"
[ "${KEEP_HF:-0}" = 1 ] || { rm -rf "$HF"; echo "deleted $HF (regenerable; KEEP_HF=1 keeps it)"; }
echo "serve: MODEL=$GGUF scripts/serve_base.sh"
