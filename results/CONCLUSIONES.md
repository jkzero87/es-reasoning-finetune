# Conclusiones (generado)

Generado por `scripts/make_conclusions.py` a partir de `results/efficiency_baseline.json`, `results/conversion_check.json` y `results/gate_mgsm.base9b.txt`. No editar a mano. Definiciones en `results/plan.md` (enmiendas 2026-11-12). Todo es descriptivo salvo la compuerta.

| # | hallazgo | número | archivo de datos | n |
|---:|---|---|---|---|
| 1 | Verificación de conversión del GGUF servido: PASS (falla si la coincidencia top-1 es ≤ 95.0% en algún enlace) | enlace 1 98.587%, enlace 2 98.81% | `results/conversion_check.json` | 920 posiciones (enlace 1) |
| 2 | Compuerta MGSM es/en ≥ 1.15: no se cumple en el 9B; no se entrena; resultado "problem not reproduced on this model" | 1.1235 < 1.15 | `results/gate_mgsm.base9b.txt` | 250 pares |
| 3 | 27B IQ3_S, mgsm, thinking: tokens totales por pregunta, español / inglés | 1.3355× | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 pares |
| 4 | 27B IQ3_S, mgsm, thinking: respuestas correctas y completas por 1.000 tokens generados, español como % del inglés | 70.7% (1.7037 vs 2.4108) | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 pares |
| 5 | 27B IQ3_S, mgsm, thinking: preguntas en que español gasta más tokens totales que inglés | 220/250 | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 pares |
| 6 | 27B IQ3_S, mgsm, thinking: exactitud español vs inglés; cortes por max_tokens español vs inglés | 0.9520 vs 0.9760; 2 vs 0 | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 y 250 |
| 7 | 27B IQ3_S, mgsm, thinking: tokens totales por pregunta, chino / inglés | 1.1565× | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 pares |
| 8 | 27B IQ3_S, mgsm, thinking: respuestas correctas y completas por 1.000 tokens generados, chino como % del inglés | 79.2% (1.9103 vs 2.4108) | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 pares |
| 9 | 27B IQ3_S, mgsm, thinking: preguntas en que chino gasta más tokens totales que inglés | 126/250 | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 pares |
| 10 | 27B IQ3_S, mgsm, thinking: exactitud chino vs inglés; cortes por max_tokens chino vs inglés | 0.9360 vs 0.9760; 0 vs 0 | `/home/jkzero/es-eval/results/mgsm_raw.jsonl` | 250 y 250 |
| 11 | 27B IQ3_S, belebele, thinking, 488 ids: tokens totales por pregunta, español / inglés | 1.3026× | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 pares |
| 12 | 27B IQ3_S, belebele, thinking, 488 ids: respuestas correctas y completas por 1.000 tokens generados, español como % del inglés | 72.6% (2.0651 vs 2.8448) | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 pares |
| 13 | 27B IQ3_S, belebele, thinking, 488 ids: preguntas en que español gasta más tokens totales que inglés | 441/488 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 pares |
| 14 | 27B IQ3_S, belebele, thinking, 488 ids: exactitud español vs inglés; cortes por max_tokens español vs inglés | 0.9652 vs 0.9816; 0 vs 0 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 y 488 |
| 15 | 27B IQ3_S, belebele, thinking, 488 ids: tokens totales por pregunta, chino / inglés | 1.0204× | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 pares |
| 16 | 27B IQ3_S, belebele, thinking, 488 ids: respuestas correctas y completas por 1.000 tokens generados, chino como % del inglés | 91.6% (2.6069 vs 2.8448) | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 pares |
| 17 | 27B IQ3_S, belebele, thinking, 488 ids: preguntas en que chino gasta más tokens totales que inglés | 244/488 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 pares |
| 18 | 27B IQ3_S, belebele, thinking, 488 ids: exactitud chino vs inglés; cortes por max_tokens chino vs inglés | 0.9611 vs 0.9816; 0 vs 0 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 488 y 488 |
| 19 | 27B IQ3_S, belebele, thinking, same 300 ids: tokens totales por pregunta, español / inglés | 1.3263× | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 pares |
| 20 | 27B IQ3_S, belebele, thinking, same 300 ids: respuestas correctas y completas por 1.000 tokens generados, español como % del inglés | 69.8% (2.0762 vs 2.9726) | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 pares |
| 21 | 27B IQ3_S, belebele, thinking, same 300 ids: preguntas en que español gasta más tokens totales que inglés | 276/300 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 pares |
| 22 | 27B IQ3_S, belebele, thinking, same 300 ids: exactitud español vs inglés; cortes por max_tokens español vs inglés | 0.9667 vs 0.9933; 0 vs 0 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 y 300 |
| 23 | 27B IQ3_S, belebele, thinking, same 300 ids: tokens totales por pregunta, chino / inglés | 1.0261× | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 pares |
| 24 | 27B IQ3_S, belebele, thinking, same 300 ids: respuestas correctas y completas por 1.000 tokens generados, chino como % del inglés | 91.2% (2.7118 vs 2.9726) | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 pares |
| 25 | 27B IQ3_S, belebele, thinking, same 300 ids: preguntas en que chino gasta más tokens totales que inglés | 150/300 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 pares |
| 26 | 27B IQ3_S, belebele, thinking, same 300 ids: exactitud chino vs inglés; cortes por max_tokens chino vs inglés | 0.9800 vs 0.9933; 0 vs 0 | `/home/jkzero/es-eval/results/belebele_raw.jsonl` | 300 y 300 |
| 27 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: tokens totales por pregunta, español / inglés | 1.2223× | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 pares |
| 28 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: respuestas correctas y completas por 1.000 tokens generados, español como % del inglés | 81.9% (0.4012 vs 0.4896) | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 pares |
| 29 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: preguntas en que español gasta más tokens totales que inglés | 43/60 | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 pares |
| 30 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: exactitud español vs inglés; cortes por max_tokens español vs inglés | 0.8333 vs 0.8167; 2 vs 0 | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 y 60 |
| 31 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: tokens totales por pregunta, chino / inglés | 1.0867× | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 pares |
| 32 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: respuestas correctas y completas por 1.000 tokens generados, chino como % del inglés | 87.2% (0.4270 vs 0.4896) | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 pares |
| 33 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: preguntas en que chino gasta más tokens totales que inglés | 35/60 | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 pares |
| 34 | 27B IQ3_S, proxlite, thinking, 60 ids, max_tokens 16384: exactitud chino vs inglés; cortes por max_tokens chino vs inglés | 0.8000 vs 0.8167; 1 vs 0 | `/home/jkzero/es-eval/results/proxlite_raw.jsonl` | 60 y 60 |
| 35 | 9B Q8_0, mgsm, thinking: tokens totales por pregunta, español / inglés | 1.1235× | `results/mgsm_raw.base9b.jsonl` | 250 pares |
| 36 | 9B Q8_0, mgsm, thinking: respuestas correctas y completas por 1.000 tokens generados, español como % del inglés | 89.8% (0.3305 vs 0.3680) | `results/mgsm_raw.base9b.jsonl` | 250 pares |
| 37 | 9B Q8_0, mgsm, thinking: preguntas en que español gasta más tokens totales que inglés | 160/250 | `results/mgsm_raw.base9b.jsonl` | 250 pares |
| 38 | 9B Q8_0, mgsm, thinking: exactitud español vs inglés; cortes por max_tokens español vs inglés | 0.9240 vs 0.9160; 7 vs 14 | `results/mgsm_raw.base9b.jsonl` | 250 y 250 |
| 39 | 9B Q8_0, belebele, thinking, same 300 ids: tokens totales por pregunta, español / inglés | 1.5401× | `results/belebele_raw.base9b.jsonl` | 300 pares |
| 40 | 9B Q8_0, belebele, thinking, same 300 ids: respuestas correctas y completas por 1.000 tokens generados, español como % del inglés | 60.9% (0.4570 vs 0.7498) | `results/belebele_raw.base9b.jsonl` | 300 pares |
| 41 | 9B Q8_0, belebele, thinking, same 300 ids: preguntas en que español gasta más tokens totales que inglés | 251/300 | `results/belebele_raw.base9b.jsonl` | 300 pares |
| 42 | 9B Q8_0, belebele, thinking, same 300 ids: exactitud español vs inglés; cortes por max_tokens español vs inglés | 0.9500 vs 0.9800; 4 vs 1 | `results/belebele_raw.base9b.jsonl` | 300 y 300 |
