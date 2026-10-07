#!/usr/bin/env bash
# Start the phase-1 base model (Qwen3.5-9B text-only, our own Q8_0 conversion,
# results/plan.md) on :8093 (not :8092, the 27B's port), detached.
# Same flags as the 27B production server (~/bin/manifiestate) minus MTP
# (the MTP head is dropped in the text-only checkpoint) and with -c 32768.
#   MODEL=... scripts/serve_base.sh   to serve another GGUF (e.g. the fine-tuned one)
#   scripts/serve_base.sh           start, wait for /health (<= 240 s)
# Stop: kill the PID it prints (log: ~/logs/base_*.log).
set -u
PORT=8093
MODEL=${MODEL:-/home/jkzero/models/Qwen3.5-9B-text-Q8_0/Qwen3.5-9B-text-Q8_0.gguf}
LOG=~/logs/base_$(date +%Y%m%d_%H%M).log
mkdir -p ~/logs
if curl -sf "localhost:$PORT/health" >/dev/null; then echo "already up on :$PORT"; exit 0; fi
GGML_CUDA_DISABLE_GRAPHS=1 setsid nohup /home/jkzero/llama.cpp/build/bin/llama-server \
    -m "$MODEL" \
    -ngl 99 -fa on -c 32768 -ub 256 \
    -ctk q8_0 -ctv q8_0 \
    -t 6 --parallel 1 --reasoning-effort medium \
    --port "$PORT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "llama-server PID $PID (log: $LOG)"
for ((i = 0; i < 240; i++)); do
    curl -sf "localhost:$PORT/health" >/dev/null && { echo "healthy after ${i}s"; exit 0; }
    kill -0 "$PID" 2>/dev/null || { echo "exited before healthy; see $LOG" >&2; exit 1; }
    sleep 1
done
echo "not healthy after 240s; see $LOG" >&2; exit 1
