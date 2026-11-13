#!/usr/bin/env bash
# Causal test 2026-11-12 (results/plan.md, amendment "label audit + causal test"). No gate.
#   27B on :8093 (es-eval flags from ~/bin/manifiestate, --parallel 1), then the 9B
#   (scripts/serve_base.sh, --parallel 4); one model on the GPU at a time.
#   fix27b / fix9b: the 6 MGSM + 5 Belebele E items with data/fixes_2026-11-12.jsonl, es only.
#   rep27b: the 10 O items of mgsm27b worst-20, es and en.
#   rep9b:  the 12 O items of belebele9b worst-20, es and en.
#   Each job runs repeats 1..5 (seed 42+R).
# Refuses to start if :8092 or :8093 is up. Serves with the phase-1 libggml-cuda
# (rule 2026-10-09). Stops each server it started (by its PID) before the next model.
# Writes results/CAUSAL_DONE only if every row count is complete. Resumable.
#
# Launch:
#   cd ~/es-reasoning-finetune && setsid nohup scripts/run_causal.sh >> results/causal_run.log 2>&1 < /dev/null &
set -u
cd /home/jkzero/es-reasoning-finetune || exit 1
PY=.venv/bin/python
BAK=/home/jkzero/llama-build-3466812d1-phase1-bak
SERVER=/home/jkzero/llama.cpp/build/bin/llama-server
M27=/home/jkzero/models/Qwen3.8-27B-GSQ/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
FIX=data/fixes_2026-11-12.jsonl
FIX_MGSM=215,39,71,199,111,94
FIX_BELE=376,388,224,483,304
REP_MGSM=88,177,158,17,236,202,13,187,237,79
REP_BELE=270,33,487,127,278,374,203,134,377,466,324,140
say() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"; }

for port in 8092 8093; do
    if ss -ltn 2>/dev/null | grep -qE ":$port\b"; then
        say "REFUSED: something is listening on :$port. Stop it first (the 16 GB card cannot hold both)."
        exit 1
    fi
done
export LD_LIBRARY_PATH=$BAK
lib=$(ldd "$SERVER" | sed -nE 's/.*libggml-cuda\.so\.0 => ([^ ]+).*/\1/p')
got=$(sha256sum "$(readlink -f "$lib")" | cut -d' ' -f1)
say "loaded: $lib -> $got"
[ "${got:0:8}" = bf382e74 ] || { say "REFUSED: not the phase-1 libggml-cuda"; exit 1; }

record() { { echo "# causal $(date -Iseconds) $1, llama-server PID $2, LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
             tr '\0' ' ' < "/proc/$2/cmdline"; echo; } >> results/serve_cmd.causal.txt; }
stop() {
    kill "$1"
    for ((i = 0; i < 120; i++)); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
    kill -0 "$1" 2>/dev/null && { say "server PID $1 still alive after 120 s"; exit 1; }
    sleep 5
}
reps() {  # reps WORKERS ARGS...: repeats 1..5 of one job
    local w=$1; shift
    for r in 1 2 3 4 5; do
        say "run: --repeat $r $*"
        $PY -u scripts/run_baseline.py --workers "$w" --repeat "$r" "$@" || { say "runner failed"; exit 1; }
    done
}

# 27B: flags as ~/bin/manifiestate (:8092 production server), port 8093.
LOG=~/logs/causal27b_$(date +%Y%m%d_%H%M%S).log
GGML_CUDA_DISABLE_GRAPHS=1 setsid nohup "$SERVER" -m "$M27" \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    -ngl 99 -fa on -c 65536 -ub 256 \
    -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
    -t 6 --parallel 1 --reasoning-effort medium \
    --port 8093 > "$LOG" 2>&1 < /dev/null &
PID=$!
for ((i = 0; i < 240; i++)); do curl -sf localhost:8093/health >/dev/null && break
    kill -0 "$PID" 2>/dev/null || { say "27B exited before healthy; see $LOG"; exit 1; }; sleep 1; done
curl -sf localhost:8093/health >/dev/null || { say "27B not healthy after 240 s; see $LOG"; exit 1; }
say "27B up, PID $PID, log $LOG"; record 27b "$PID"
reps 1 --bench mgsm --tag fix27b --langs es --ids "$FIX_MGSM" --fixes "$FIX"
reps 1 --bench belebele --tag fix27b --langs es --ids "$FIX_BELE" --fixes "$FIX"
reps 1 --bench mgsm --tag rep27b --langs es,en --ids "$REP_MGSM"
stop "$PID"; say "27B stopped"

# 9B: phase-1 server.
PARALLEL=4 scripts/serve_base.sh || { say "9B did not start"; exit 1; }
PID=$(pgrep -f 'llama-server.*--port 8093' | head -1)
record 9b "$PID"
reps 4 --bench mgsm --tag fix9b --langs es --ids "$FIX_MGSM" --fixes "$FIX"
reps 4 --bench belebele --tag fix9b --langs es --ids "$FIX_BELE" --fixes "$FIX"
reps 4 --bench belebele --tag rep9b --langs es,en --ids "$REP_BELE"
stop "$PID"; say "9B stopped"

counts=$(for f in mgsm_raw.fix27b:30 belebele_raw.fix27b:25 mgsm_raw.rep27b:100 \
                  mgsm_raw.fix9b:30 belebele_raw.fix9b:25 belebele_raw.rep9b:120; do
    n=$(wc -l < "results/${f%%:*}.jsonl"); echo "${f%%:*} $n/${f##*:}"; done)
say "counts:"; echo "$counts"
if echo "$counts" | awk -F'[ /]' '$2 != $3 {bad=1} END {exit bad}'; then
    { date -Iseconds; echo "$counts"; } > results/CAUSAL_DONE; say "CAUSAL_DONE written"
else
    say "incomplete; relaunch the same command to resume"; exit 1
fi
