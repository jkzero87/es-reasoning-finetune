# es-reasoning-finetune

Brújula: que usar la IA en español no sea una desventaja. Objetivo: con una sola GPU
de 16 GB, producir la mayor cantidad posible de datos abiertos y conclusiones prácticas
sobre cuánto, dónde y por qué el español sale más caro en modelos abiertos, y qué
remedios funcionan y cuáles no. El fine-tuning es una prueba más, no la meta.

**Estado (2026-11-12):** línea base del 9B medida y compuerta aplicada. En MGSM el 9B gasta en español 1,1235 × los
tokens del inglés, por debajo del umbral preregistrado de 1,15: **no se entrena**; resultado para MGSM: "problem not
reproduced on this model" (el 27B dio 1,34 ×). En Belebele el mismo 9B gasta 1,5401 × (descriptivo, fuera de la
compuerta). Decisión: `results/gate_decision.md`. Siguen corridas descriptivas (chino, sin razonamiento, ProX-Lite).

<!-- TABLES:BEGIN (generado por scripts/update_readme_tables.py; no editar a mano) -->
## Resultados

Tablas completas: `results/efficiency_baseline.md`; hallazgos uno por fila: `results/CONCLUSIONES.md`.

- Verificación de conversión del GGUF servido: PASS (falla si la coincidencia top-1 es ≤ 95.0% en algún enlace): **enlace 1 98.587%, enlace 2 98.81%** (`results/conversion_check.json`, 920 posiciones (enlace 1)).
- Compuerta MGSM es/en ≥ 1.15: no se cumple en el 9B; no se entrena; resultado "problem not reproduced on this model": **1.1235 < 1.15** (`results/gate_mgsm.base9b.txt`, 250 pares).

Éxito por 1k = respuestas correctas y completas por cada 1.000 tokens generados, como % del inglés (menos de 100 % = ese idioma sale más caro).

| modelo | prueba | condición | n (en/es/zh) | tokens totales es/en | éxito por 1k, es vs en | tokens totales zh/en | éxito por 1k, zh vs en |
|---|---|---|---:|---:|---:|---:|---:|
| 27B IQ3_S | mgsm | thinking | 250/250/250 | 1.3355× | 70.7 % | 1.1565× | 79.2 % |
| 27B IQ3_S | belebele | thinking, 488 ids | 488/488/488 | 1.3026× | 72.6 % | 1.0204× | 91.6 % |
| 27B IQ3_S | belebele | thinking, same 300 ids | 300/300/300 | 1.3263× | 69.8 % | 1.0261× | 91.2 % |
| 27B IQ3_S | proxlite | thinking, 60 ids, max_tokens 16384 | 60/60/60 | 1.2223× | 81.9 % | 1.0867× | 87.2 % |
| 9B Q8_0 | mgsm | thinking | 250/250 | 1.1235× | 89.8 % | — | — |
| 9B Q8_0 | belebele | thinking, same 300 ids | 300/300 | 1.5401× | 60.9 % | — | — |
<!-- TABLES:END -->

## Objetivo

Que el mismo modelo cueste los mismos tokens razonando en español que en
inglés, sin perder precisión.

## Punto de partida

La medición original está en [jkzero87/es-eval](https://github.com/jkzero87/es-eval)
(Qwen3.8-27B, IQ3_S, local). En MGSM (250 ítems) el modelo acierta casi
igual en los dos idiomas (en 97,6 % / es 95,2 %, diferencia no
significativa), pero en español razona alrededor de un 50 % más (razón
es/en de tokens de razonamiento 1,50) y cada pregunta cuesta 1,34 × los
tokens del inglés (659,6 vs 493,9 en total, prompt + completion). La causa
principal, según la auditoría, es que el modelo duda de enunciados que ya
había leído bien.

Dos palancas sin reentrenar quedaron descartadas allí:

- **Palanca 1**, un prompt de sistema ("lee las palabras en su sentido
  llano"): bajó la brecha un 28 %, pero no alcanzó la meta preregistrada.
- **Palanca 2**, traducir y responder (es→en, responder en inglés, en→es):
  2,15 × los tokens del inglés en 130 de 250 ítems (meta ≤ 1,10 ×). Ninguna
  palanca basada en traducción puede llegar: la sola llamada de traducción
  de entrada ya supera el margen (`es-eval/results/translation_floor.md`).

## Método

1. **Línea base**: MGSM (250 ítems) y una muestra fija de 300 ítems de
   Belebele, en y es, sobre Qwen/Qwen3.5-9B solo texto (sin torre de visión
   ni cabeza MTP), convertido por nosotros a GGUF Q8_0 y servido con
   llama-server local, con los mismos ajustes de generación que la línea
   base del 27B.
2. **Ajuste fino**: LoRA bf16 sobre la base en 8 bits (Unsloth), en la
   tarjeta local de 16 GB, con la parte en español de ReasonXL-SFT
   (muestras de más de 4096 tokens descartadas), más un control con la
   parte en inglés.
3. **Pruebas de falsación**: criterios fijados antes de mirar los resultados.

El plan completo, con fases, compuerta y criterios, está en
`results/plan.md`; la factibilidad en la tarjeta local, en
`results/feasibility.md`.

## Estructura

- `scripts/run_mgsm.py`, `scripts/score_mgsm.py`, `data/mgsm_{en,es}.jsonl`:
  copiados sin cambios de es-eval para puntuar igual (procedencia y hashes en
  `SOURCE.md`).
- `results/`: planes, salidas crudas y puntajes.
- Modelos, checkpoints, cachés y `.venv` no se versionan (`.gitignore`).

## Licencia

El código (`scripts/`) está bajo **MIT** (`LICENSE`, copyright 2026 Juan
Camilo Bejarano Triana).

### Licencias de datos

`data/mgsm_{en,es}.jsonl`, `data/belebele_{en,es}.jsonl` y las salidas
crudas en `results/` contienen texto de MGSM y Belebele y se comparten bajo
**CC BY-SA 4.0**, como exigen los conjuntos de datos.

| conjunto | vía | licencia |
|---|---|---|
| MGSM (Shi et al., 2022), basado en GSM8K (Cobbe et al., 2021) | Hugging Face `juletxara/mgsm` | CC BY-SA 4.0; GSM8K: MIT |
| Belebele (Bandarkar et al., 2024), pasajes de FLORES-200 | Hugging Face `facebook/belebele` | CC BY-SA 4.0 |

Atribución:
- Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush
  Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das,
  Jason Wei. *Language Models are Multilingual Chain-of-Thought Reasoners.*
  arXiv:2210.03057, 2022.
- Karl Cobbe et al. *Training Verifiers to Solve Math Word Problems.*
  arXiv:2110.14168, 2021.
- Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan
  Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer,
  Madian Khabsa. *The Belebele Benchmark: a Parallel Reading Comprehension
  Dataset in 122 Language Variants.* ACL 2024, pp. 749–775.
