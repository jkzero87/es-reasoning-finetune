# Resumen para el semillero (2026-11-12)

**Objetivo** (`README.md`): "Brújula: que usar la IA en español no sea una desventaja. Objetivo: con una sola GPU de 16 GB, producir la mayor cantidad posible de datos abiertos y conclusiones prácticas sobre cuánto, dónde y por qué el español sale más caro en modelos abiertos, y qué remedios funcionan y cuáles no. El fine-tuning es una prueba más, no la meta."

## Qué se midió
- Modelos: 27B IQ3_S y 9B Q8_0, en local con llama.cpp; idiomas en, es y zh (`results/efficiency_baseline.md`).
- 27B: MGSM 250, Belebele 488 (y los mismos 300 ids del 9B), MMLU-ProX-Lite 60 (`results/efficiency_baseline.md`).
- 9B: MGSM 250, Belebele 300, MMLU-ProX-Lite 60, con razonamiento; MGSM 250 y Belebele 300 sin razonamiento (`results/efficiency_baseline.md`).
- Métrica: respuestas correctas y completas por cada 1.000 tokens generados, español como % del inglés; menos de 100 % = el español sale más caro (`results/efficiency_baseline.md`).

## Números clave
- 27B, español como % del inglés: MGSM 70.7%, Belebele 72.6% (488 ids), ProX-Lite 81.9% (`results/efficiency_baseline.md`).
- 9B con razonamiento: MGSM 89.8%, Belebele 60.9%, ProX-Lite 117.2% (`results/efficiency_baseline.md`).
- 9B sin razonamiento: MGSM 91.0% (tokens es/en 1.0849), Belebele 114.8% (tokens es/en 1.0197) (`results/efficiency_baseline.md`).
- Compuerta MGSM 9B: 1.1235 < 1.15, así que no se entrena (`results/gate_mgsm.base9b.txt`, `results/gate_decision.md`).
- Auditoría de etiquetas, 20 peores ítems por exceso de razonamiento es−en (`results/label_audit_2026-11-12.md`):
  - MGSM 27B: los 6 ítems con error de traducción (E) llevan 0.411 del exceso de todo el benchmark (es/en 6.021). Los 10 bien traducidos (O) siguen en es/en 3.457.
  - Belebele 9B: 5 ítems E llevan 0.118 del exceso (es/en 4.408). Los 12 O llevan 0.227 (es/en 3.216).
  - En 20 ítems al azar, es/en es 1.150 (MGSM 27B) y 1.201 (Belebele 9B).
- Prueba causal (A), español corregido, 5 repeticiones; media de caracteres de razonamiento en español, original → corregido, y es/en (`results/causal_test_2026-11-12.md`):
  - MGSM 27B, 6 ítems: 7229 → 1309; es/en 6.021 → 1.090.
  - MGSM 9B, 6 ítems: 20105 → 11347; es/en 2.640 → 1.490.
  - Belebele 9B, 5 ítems: 27355 → 11915; es/en 4.408 → 1.920.
  - Belebele 27B, 5 ítems: 1852 → 1751; es/en 1.810 → 1.711.
- Prueba causal (B), ítems bien traducidos, 5 repeticiones es y en (`results/causal_test_2026-11-12.md`):
  - MGSM 27B, 10 ítems: 29/50 repeticiones en español superan 2× la media en inglés; es/en 2.196.
  - Belebele 9B, 12 ítems: 27/60 superan 2×; es/en 1.873.

## Límites
- Auditoría con una sola anotadora (Claude), solo preguntas y opciones; los pasajes no se revisaron (`results/label_audit_2026-11-12.md`).
- ProX-Lite en el 9B con tope de 8192 tokens, frente a 16384 en el 27B (`results/plan.md`, enmienda 2026-11-12, cola GPU).
- Prueba causal con muestras pequeñas: 22 ítems en (A) y 22 en (B), elegidos entre los peores casos (`results/causal_test_2026-11-12.md`).
- En la prueba causal, el español corregido se compara con el único registro en inglés original, no con repeticiones en inglés. Las repeticiones usan semillas 43–47 (`results/plan.md`, enmienda 2026-11-12, auditoría + prueba causal).
