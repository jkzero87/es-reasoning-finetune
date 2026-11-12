#!/usr/bin/env bash
# Descriptive GPU queue on the 9B (results/plan.md, amendment 2026-11-12). No gate.
#   q1  zh baseline: MGSM 250 + Belebele same 300 ids          tag base9b (zh rows)
#   q2  reasoning off, es/en/zh: MGSM 250 + Belebele 300       tag noreason9b
#   q3  MMLU-ProX-Lite es/en/zh, 60 ids, thinking on           tag prox9b
# Starts the 9B on :8093 (scripts/serve_base.sh, phase-1 flags, phase-1
# libggml-cuda), runs q1..q3 in order, stops the server, and writes
# results/QUEUE_DONE (timestamp + row counts) only if every count is complete.
# Refuses to start if :8092 or :8093 is up. Resumable: rows already written
# are skipped, so an interrupted queue is continued with the same command.
#
# Launch (Juan):
#   cd ~/es-reasoning-finetune && setsid nohup scripts/run_queue.sh >> results/queue.log 2>&1 < /dev/null &
# Optional: STOP_AT=18:45 (no new item at or after it; the queue then ends
# incomplete and is relaunched the next day), WORKERS=4.
# Smoke test: SMOKE=1 scripts/run_queue.sh  -> 2 items per language per run,
# tags smoke_*, no QUEUE_DONE; delete results/*smoke_* afterwards.
set -u
cd /home/jkzero/es-reasoning-finetune || exit 1
WORKERS=${WORKERS:-4}
SMOKE=${SMOKE:-0}
STOP=()
[ -n "${STOP_AT:-}" ] && STOP=(--stop-at "$STOP_AT")
PY=.venv/bin/python
BAK=/home/jkzero/llama-build-3466812d1-phase1-bak
SERVER=/home/jkzero/llama.cpp/build/bin/llama-server
say() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"; }

for port in 8092 8093; do
    if ss -ltn 2>/dev/null | grep -qE ":$port\b"; then
        say "REFUSED: something is listening on :$port. Stop it first (the 16 GB card cannot hold both)."
        exit 1
    fi
done

# Rule 2026-10-09 (NEXT.md): serve with the phase-1 libggml-cuda.
export LD_LIBRARY_PATH=$BAK
lib=$(ldd "$SERVER" | sed -nE 's/.*libggml-cuda\.so\.0 => ([^ ]+).*/\1/p')
want=$(awk '$2 == "libggml-cuda.so.0.22.0" {print $1}' "$BAK/SHA256SUMS")
got=$(sha256sum "$(readlink -f "$lib")" | cut -d' ' -f1)
say "loaded: $lib -> $got"
if [ -z "$got" ] || [ "$got" != "$want" ] || [ "${got:0:8}" != bf382e74 ]; then
    say "REFUSED: not the phase-1 libggml-cuda"
    exit 1
fi

say "queue start (SMOKE=$SMOKE WORKERS=$WORKERS STOP_AT=${STOP_AT:-none})"
PARALLEL=$WORKERS scripts/serve_base.sh || { say "server did not start"; exit 1; }
PID=$(pgrep -f 'llama-server.*--port 8093' | head -1)
{ echo "# queue $(date -Iseconds), llama-server PID $PID, LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  tr '\0' ' ' < "/proc/$PID/cmdline"; echo; } >> results/serve_cmd.queue.txt

if [ "$SMOKE" = 1 ]; then
    T1=smoke_base9b; T2=smoke_noreason9b; T3=smoke_prox9b; LIM=(--limit-per-lang 2)
else
    T1=base9b; T2=noreason9b; T3=prox9b; LIM=()
fi
run() { say "run: $*"; $PY -u scripts/run_baseline.py --workers "$WORKERS" "${STOP[@]}" "${LIM[@]}" "$@"; }

say "q1 zh baseline"
run --bench mgsm     --tag "$T1" --langs zh
run --bench belebele --tag "$T1" --langs zh
say "q2 reasoning off"
run --bench mgsm     --tag "$T2" --langs es,en,zh --order interleave --thinking off
run --bench belebele --tag "$T2" --langs es,en,zh --order interleave --thinking off
say "q3 ProX-Lite"
run --bench proxlite --tag "$T3" --langs es,en,zh --order interleave

say "stopping llama-server PID $PID"
kill "$PID"
for _ in $(seq 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -0 "$PID" 2>/dev/null && say "WARNING: PID $PID still alive after 60 s"

if [ "$SMOKE" = 1 ]; then
    say "smoke run finished (no QUEUE_DONE). Rows:"; wc -l results/*_raw.smoke_*.jsonl
    exit 0
fi
if COUNTS=$($PY scripts/queue_counts.py); then
    { echo "QUEUE_DONE $(date -Iseconds)"; echo "$COUNTS"; } > results/QUEUE_DONE
    say "QUEUE_DONE written"
else
    echo "$COUNTS"
    say "INCOMPLETE: no QUEUE_DONE. Relaunch the same command to continue."
    exit 2
fi
