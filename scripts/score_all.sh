#!/usr/bin/env bash
# Run the existing scorers (unchanged copies of es-eval's) on every 9B raw file
# that exists, without /tokenize (never touches a server), and save each output
# as results/score_<bench>.<tag>.txt. A raw file that does not exist yet is
# reported as "not run" and skipped.
#   scripts/score_all.sh
set -u
cd /home/jkzero/es-reasoning-finetune || exit 1
n=0
for item in mgsm:base9b belebele:base9b mgsm:noreason9b belebele:noreason9b proxlite:prox9b; do
    bench=${item%%:*}; tag=${item##*:}
    raw=results/${bench}_raw.$tag.jsonl
    out=results/score_$bench.$tag.txt
    if [ ! -s "$raw" ]; then echo "not run   $raw"; continue; fi
    extra=()
    [ "$bench" = proxlite ] && extra=(--subset results/proxlite_subset_60_strat.txt)
    if .venv/bin/python "scripts/score_$bench.py" --raw "$raw" --no-tokenize "${extra[@]}" > "$out"; then
        echo "scored    $raw -> $out ($(head -1 "$out"))"; n=$((n + 1))
    else
        echo "FAILED    $raw"; exit 1
    fi
done
echo "score files written: $n"
