# CORRER — benchmark base del 9B (sin tocar el 27B en :8092)

Medido igual que el 27B (`~/es-eval`): mismos prompts (`data/*.jsonl`), misma
generación (temperature 1.0, top_p 0.95, top_k 20, seed 42, max_tokens 8192,
sin system prompt) y misma métrica (tokens de `reasoning_content` y `content`
vía `/tokenize` del servidor, como `es-eval/scripts/mgsm_token_split.py`).

**No ejecutar hasta que la GPU esté libre** (el 27B en :8092 la ocupa;
`journalctl -k | grep Xid` antes de arrancar).

## 1. Servidor (llama-server, puerto 8093)

Modelo: `~/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf`
(Q8_0, MTP; sha256 `a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496`).
Llamada exacta (idéntica a la de `results/serve_cmd.base9b.txt` y a
`scripts/serve_base.sh`; MTP incluido porque los raw base9b existentes se
corrieron así):

```bash
GGML_CUDA_DISABLE_GRAPHS=1 setsid nohup /home/jkzero/llama.cpp/build/bin/llama-server \
  -m /home/jkzero/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 \
  -ngl 99 -fa on -c 36864 -ub 256 -ctk q8_0 -ctv q8_0 \
  -t 6 --parallel 4 --reasoning-effort medium \
  --port 8093 > ~/logs/base_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Esperar a que `curl -sf localhost:8093/health` responda (~20 s).

> Nota: los raw base9b existentes de MGSM se corrieron **sin** MTP
> (llama.cpp b10749, `results/phase1_report.md`); los de Belebele **con** MTP
> (b10751). Con MTP las cifras de tokens cambian ligeramente; ver
> `results/plan.md` si hace falta decidir antes de llenar el hueco de zh.

## 2. Benchmark — trabajo pendiente: SOLO zh (es y en ya están completos)

```bash
cd ~/es-reasoning-finetune

# MGSM (250/idioma; es+en completos, falta solo zh):
.venv-train/bin/python -u bench/baseline_9b.py --bench mgsm --tag base9b --langs zh --stop-at 19:00 >> bench/mgsm_run.log 2>&1

# Belebele (es+en ya tienen los 300 ids del sample; falta solo zh con ESE MISMO
# sample de 300 ids — NO llenar el resto de los 488):
.venv-train/bin/python -u bench/baseline_9b.py --bench belebele --tag base9b --langs zh --stop-at 19:00 >> bench/belebele_run.log 2>&1
```

- El reanudable salta los `(id, lang)` ya escritos en
  `bench/<bench>_raw.base9b.jsonl`, así que en MGSM solo se corre zh.
- Para correr SOLO zh: `--langs zh` (el reanudable salta lo que ya hay).
- En Belebele, si el reanudable quisiera correr los es/en restantes (488−300),
  NO se corren: la comparación con el 27B se hace sobre el sample de 300 ids
  (`results/belebele_sample_300.json`) y ese ya está completo. El único trabajo
  pendiente en GPU es **zh: MGSM 250 + Belebele 300 (mismo sample de ids del 9B
  existente en es/en)**.
- Ojo: `data/belebele_zh.jsonl` no tiene `gold` ni `options` (formato distinto al
  de es/en; ver la nota en "Estado"). Para Belebele zh hay que preparar un
  dataset con esos campos antes de correr (o `--limit 0`/revisar antes).

> Nota MTP (observación, no se re-corre): los raw existentes de MGSM se
> corrieron **sin** MTP (llama.cpp b10749, `results/phase1_report.md`) y los de
> Belebele **con** MTP (b10751). MTP no cambia la longitud de las salidas
> (acelera la decodificación por draft, no el contenido), así que esa
> diferencia no se re-corre y queda como observación; para decidir antes de
> llenar el hueco de zh ver `results/plan.md`.

Prueba corta antes de lanzar en serio:

```bash
.venv-train/bin/python -u bench/baseline_9b.py --bench mgsm --tag test --limit 2
```

- `--stop-at 19:00` (hora local): no empieza nada a esa hora ni después; lo
  que esté en vuelo termina y se escribe. Para detener antes: `Ctrl-C`
  (SIGINT/SIGTERM) → termina el item en vuelo, escribe y sale limpio.
- **Reanudar al día siguiente**: re-lanzar el mismo comando; salta los
  `(id, lang)` ya en `bench/<bench>_raw.<tag>.jsonl` y regenera el resumen.
- `--limit N`: máximo de N pendientes por idioma (pruebas).

## 3. Salidas (en `bench/`)

| archivo | contenido |
|---|---|
| `<bench>_raw.base9b.jsonl` | crudo por petición (igual que es-eval) + `reasoning_tokens`, `answer_tokens` |
| `<bench>_raw.base9b.csv` | tabla cruda por petición (id, lang, gold, tokens, wall) |
| `server_props.base9b.json` | `GET /props` del 9B antes del primer item |
| `<bench>_summary.base9b.json` | resumen por idioma: medias/min/max de reasoning/answer/total y ratios es/en, zh/en |

Regenerar solo el resumen (sin tocar el servidor):

```bash
.venv-train/bin/python -u bench/baseline_9b.py --bench mgsm --tag base9b --summary-only
```

## 4. Cerrar

```bash
# par el llama-server del 9B (NUNCA el de :8092)
kill $(pgrep -f 'llama-server.*--port 8093')   # o el PID que imprimió el arranque
```

## 5. Estado

- **Ya hecho en CPU (20-10, antes de 19:00):**
  - Tokenizer CPU del 9B (`~/models/Qwen3.5-9B-text-mtp/tokenizer.json`,
    `add_special_tokens=False`) verificado contra el `/tokenize` de :8092:
    **30/30** (`bench/check_tokenizer_9b.py`: 5 textos por idioma de cada
    `data/*_{es,en,zh}.jsonl`, seed 20261020).
  - Splits derivados de los raw con ese tokenizer:
    `results/mgsm_token_split.base9b.jsonl` (500 filas = 250 es + 250 en) y
    `results/belebele_token_split.base9b.jsonl` (600 filas = 300 es + 300 en),
    formato igual que los `*_token_split.jsonl` del 27B (`bench/derive_token_split_9b.py`).
  - Comparación justa (mismos ids, `bench/compare_27b_9b.py`):
    - Belebele 27B-488: es/en reason 1.4902 · 27B-300: 1.5507 · 9B-300: 1.6290.
    - MGSM 250 vs 250: 27B es/en reason 1.5015 · 9B 1.1200.
- **Pendiente en GPU: solo zh** (MGSM 250 + Belebele 300 del sample).
- `data/belebele_zh.jsonl`: 976 líneas = 2×488. Bloque 1 (ids int 1..488):
  `id, link, flores_passage` (solo passage, como el raw del 27B). Bloque 2
  (ids str `q<url>:<n>`): `id, link, question_number, flores_passage, question`
  (versión HF completa de Belebele). Mismo set de 488 preguntas en ambos
  bloques; **sin `options` ni `gold`** → para correrlo con `baseline_9b.py`
  hay que añadir esos campos (o usar el bloque int tal cual para el prompt).
