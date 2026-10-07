#!/usr/bin/env bash
# Start Qwen3-14B Q4_K_M on :8093 (not :8092, the 27B's port), detached.
# Same flags as the 27B production server (~/bin/manifiestate) minus MTP
# (Qwen3-14B has no MTP head) and with -c 32768 (the model's native context).
#   scripts/serve_14b.sh            start, wait for /health (<= 240 s)
# Stop: kill the PID it prints (also in ~/logs/q14b_*.log's first lines).
set -u
PORT=8093
MODEL=/home/jkzero/models/Qwen3-14B/Qwen3-14B-Q4_K_M.gguf
LOG=~/logs/q14b_$(date +%Y%m%d_%H%M).log
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
