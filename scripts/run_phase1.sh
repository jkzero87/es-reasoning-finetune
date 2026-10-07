#!/usr/bin/env bash
# Phase 1 chain (results/plan.md): MGSM es, en, then Belebele es, en, on :8093.
# Resumable (each runner skips done items); no new item at/after STOP_AT.
#   STOP_AT=18:45 setsid nohup scripts/run_phase1.sh >> results/phase1_run.log 2>&1 < /dev/null &
set -u
cd /home/jkzero/es-reasoning-finetune || exit 1
STOP_AT=${STOP_AT:-18:45}
.venv/bin/python -u scripts/run_baseline.py --bench mgsm --langs es,en --stop-at "$STOP_AT"
.venv/bin/python -u scripts/run_baseline.py --bench belebele --langs es,en --stop-at "$STOP_AT"
