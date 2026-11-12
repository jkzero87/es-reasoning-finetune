# CORRER — cola de GPU del 9B (la lanza Juan)

Enmienda 2026-11-12 de `results/plan.md`: corridas descriptivas, sin compuerta. Mismos flags de servidor, mismo
cuantizado y mismos ids que la fase 1.

| ítem | qué corre | etiqueta | filas nuevas |
|---|---|---|---:|
| q1 | zh: MGSM 250 + Belebele los mismos 300 ids | `base9b` (filas zh) | 550 |
| q2 | razonamiento apagado, es/en/zh: MGSM 250 + Belebele 300 | `noreason9b` | 1650 |
| q3 | MMLU-ProX-Lite es/en/zh, 60 ids de es-eval, `max_tokens` 8192 | `prox9b` | 180 |

## 1. Antes de lanzar

- **El 27B de :8092 tiene que estar detenido.** `scripts/run_queue.sh` se niega a arrancar si hay algo escuchando en
  :8092 o en :8093 (la tarjeta de 16 GB no aguanta los dos). El script nunca toca :8092.
- `nvidia-smi` sin procesos de cómputo.
- q3 lee los ítems de la caché local de es-eval (`~/es-eval/.hf_cache`, sin red); no borrar esa carpeta.

## 2. El comando (uno solo)

```bash
cd ~/es-reasoning-finetune && setsid nohup scripts/run_queue.sh >> results/queue.log 2>&1 < /dev/null &
```

Qué hace, en orden: comprueba la `libggml-cuda` de la fase 1 (`bf382e74…`, regla 2026-10-09), arranca el 9B en :8093
con `scripts/serve_base.sh` (los mismos flags que `results/serve_cmd.base9b.txt`, `--parallel 4`, MTP), guarda la
línea exacta del servidor en `results/serve_cmd.queue.txt`, corre q1, q2 y q3 con `scripts/run_baseline.py
--workers 4`, detiene el servidor y, **solo si todos los conteos son exactos** (`scripts/queue_counts.py`), escribe
`results/QUEUE_DONE` con la hora y los conteos.

- Seguir el avance: `tail -f results/queue.log`. Conteos en cualquier momento: `.venv/bin/python scripts/queue_counts.py`.
- **Reanudable:** si se corta (apagón, `STOP_AT`, Ctrl-C al servidor), relanzar el mismo comando; salta los
  `(id, lang)` ya escritos. Si el servidor quedó vivo: `kill $(pgrep -f 'llama-server.*--port 8093')` (NUNCA el de :8092).
- Hora de corte opcional: anteponer `STOP_AT=18:45` al comando (`STOP_AT=18:45 setsid nohup …`); no empieza ítems
  a esa hora ni después, termina incompleta (sin `QUEUE_DONE`) y se relanza al día siguiente.
- Al terminar: levantar el 27B y darle al agente local `bench/TAREAS_AGENTE.md` (T1–T6).

## 3. Horas proyectadas

Medido: en la fase 1 el servidor dio **214–225 tokens de salida por segundo en total** con 4 peticiones en vuelo
(`results/phase1_run.log`: 951.681 tokens en 4.240 s el 2026-10-08; 1.015.784 en 4.744 s el 2026-10-09). La
proyección es tokens esperados / 214. Los tokens por registro de zh, sin razonamiento y ProX-Lite salen de la prueba
de humo del 2026-11-12 (2 ítems por idioma: muy pocos, la incertidumbre es grande) y de las medias de la fase 1.

| ítem | registros | tokens de salida por registro (supuesto) | de dónde sale el supuesto | proyección |
|---|---:|---:|---|---:|
| q1 MGSM zh | 250 | ~2.800 | media es de la fase 1 (2.795,7); humo zh: 1.594 y 8.192 (corte) | ~55 min |
| q1 Belebele zh | 300 | ~1.700 | entre en (1.307) y es (2.079) de la fase 1; humo zh: 1.052 y 1.118 | ~40 min |
| q2 MGSM sin razonamiento | 750 | ~220 | humo: 166–290 (6 registros en 7 s) | ~15 min |
| q2 Belebele sin razonamiento | 900 | ~205 | humo: 77–420 (6 registros en 9 s) | ~25 min |
| q3 ProX-Lite | 180 | ~5.000 | humo: media 6.000 con 3 de 6 cortados en 8.192 (6 registros en 202 s) | ~70 min |
| **total** | 2.380 | | | **~3,5 h** |

Rango razonable: **3 a 4,5 h**. Techo si q3 entero llegara al corte de 8.192: q3 115 min. La prueba de humo completa
(22 registros, las 5 corridas) tardó 5 min 54 s de 15:10:07 a 15:16:01 incluyendo arranque y parada del servidor.

Observaciones de la prueba de humo (no son resultados; sus filas se borraron):
- `--thinking off` funciona: `reasoning_content` vacío en los 12 registros, todos terminan con la línea de respuesta.
- ProX-Lite id 75: los tres idiomas llegaron al corte de 8.192 sin respuesta (`content` vacío). En el 9B habrá
  cortes en q3; quedan contados en la columna `length` de `results/efficiency_baseline.md`.
- MGSM zh id 1 llegó al corte (8.192).

## 4. Cómo se midió la fase 1 (para comparar)

Modelo: `~/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf` (Q8_0, MTP; sha256
`a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496`). Generación: temperature 1.0, top_p 0.95,
top_k 20, seed 42, max_tokens 8192, sin system prompt. Línea del servidor: `results/serve_cmd.base9b.txt`.

> Nota MTP (corregida 2026-11-12): los raw base9b de MGSM y de Belebele se
> corrieron **con** MTP, en llama.cpp b10751-3466812d1. Evidencia: los 500
> registros de MGSM y los 600 de Belebele tienen `timings.draft_n > 0`; las
> dos líneas de `results/serve_cmd.base9b.txt` llevan `--spec-type draft-mtp`;
> los logs del servidor de las tres sesiones (`~/logs/base_20261007_175716.log`,
> `base_20261008_153937.log`, `base_20261009_170416.log`) dicen "creating MTP
> draft context"; `results/server_props.base9b.json` dice `b10751-3466812d1`.
> "b10749" no aparece en ningún otro archivo del repo.

`bench/baseline_9b.py` (runner alternativo del 2026-10-20 que escribe en `bench/`) **no se usa en la cola**: la cola
usa el runner de la fase 1, `scripts/run_baseline.py`, y escribe en `results/`.

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
- **Pendiente en GPU (2026-11-12):** la cola de arriba (q1, q2, q3).
- `data/belebele_zh.jsonl` (corregido 2026-11-12): el archivo anterior era el crudo de es-eval, 976 líneas = 488 filas
  de pasaje + 488 filas de pregunta, sin `options` ni `gold`. Ahora tiene el formato de en/es (488 ids; `qid` y `gold`
  iguales a en en todos; los 300 ids de la muestra presentes), exportado con `scripts/export_belebele_zh.py`.
