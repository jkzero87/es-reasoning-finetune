#!/usr/bin/env bash
# Start a phase-1/phase-3 model (results/plan.md) on :8093 (not :8092, the
# 27B's port), detached. Flags as the 27B production server (~/bin/manifiestate).
#   [MODEL=...] [PARALLEL=4] [CTX=36864] [MTP=1] scripts/serve_base.sh
# PARALLEL: server slots (runners use --workers PARALLEL); CTX is split evenly
# across slots, so CTX/PARALLEL must hold prompt + 8192 completion tokens.
# MTP=1: MTP speculative decoding with the 27B's flags (draft-mtp, 3 drafted
# tokens, q8_0 draft KV); needs a GGUF with the MTP block (--keep-mtp).
# Waits for /health (<= 240 s). Stop: kill the PID it prints.
set -u
PORT=8093
MODEL=${MODEL:-/home/jkzero/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf}
PARALLEL=${PARALLEL:-4}
CTX=${CTX:-36864}
MTP=${MTP:-1}
SPEC=()
[ "$MTP" = 1 ] && SPEC=(--spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0)
LOG=~/logs/base_$(date +%Y%m%d_%H%M%S).log
mkdir -p ~/logs
if curl -sf "localhost:$PORT/health" >/dev/null; then echo "already up on :$PORT"; exit 0; fi
GGML_CUDA_DISABLE_GRAPHS=1 setsid nohup /home/jkzero/llama.cpp/build/bin/llama-server \
    -m "$MODEL" "${SPEC[@]}" \
    -ngl 99 -fa on -c "$CTX" -ub 256 \
    -ctk q8_0 -ctv q8_0 \
    -t 6 --parallel "$PARALLEL" --reasoning-effort medium \
    --port "$PORT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "llama-server PID $PID (log: $LOG) model=$(basename "$MODEL") parallel=$PARALLEL ctx=$CTX mtp=$MTP"
for ((i = 0; i < 240; i++)); do
    curl -sf "localhost:$PORT/health" >/dev/null && { echo "healthy after ${i}s"; exit 0; }
    kill -0 "$PID" 2>/dev/null || { echo "exited before healthy; see $LOG" >&2; exit 1; }
    sleep 1
done
echo "not healthy after 240s; see $LOG" >&2; exit 1
