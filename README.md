# es-reasoning-finetune

Norte: que usar un modelo de lenguaje en español no cueste más que usarlo en inglés. Es la dirección del trabajo, no lo que este proyecto promete resolver. Objetivo: identificar los factores que hacen más costoso usar modelos de lenguaje en español y atacarlos mediante fine-tuning, reentrenando sobre una NVIDIA RTX 5060 Ti de 16 GB, para obtener conclusiones que aporten a reducir esa desventaja. El fine-tuning es el centro del proyecto.

**Estado (2026-11-13):** fase de diagnóstico concluida (27B y 9B; en, es y zh) y prueba causal sobre errores de traducción hecha (`results/causal_test_2026-11-12.md`). La compuerta de MGSM de la fase 1 se aplicó tal como estaba escrita: en matemáticas el 9B gasta en español 1,1235 × los tokens del inglés, por debajo de 1,15, y ese resultado queda publicado (`results/gate_decision.md`). Desde la enmienda del 2026-11-13 el fine-tuning es el eje del proyecto: ataca el sobrecosto medido en lectura (1,5401 ×) y el que persiste con español bien traducido, con criterios fijados antes de entrenar.

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
| 9B Q8_0 | mgsm | thinking | 250/250/250 | 1.1235× | 89.8 % | 1.7901× | 43.6 % |
| 9B Q8_0 | belebele | thinking, same 300 ids | 300/300/300 | 1.5401× | 60.9 % | 1.4238× | 64.6 % |
| 9B Q8_0 | mgsm | reasoning off | 250/250/250 | 1.0849× | 91.0 % | 0.9750× | 94.0 % |
| 9B Q8_0 | belebele | reasoning off, same 300 ids | 300/300/300 | 1.0197× | 114.8 % | 0.9276× | 105.0 % |
| 9B Q8_0 | proxlite | thinking, 60 ids, max_tokens 8192 | 60/60/60 | 0.9957× | 117.2 % | 1.0510× | 99.2 % |
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
2. **Fine-tuning**: LoRA bf16 sobre la base en 8 bits (Unsloth), en la
   NVIDIA RTX 5060 Ti de 16 GB, con la parte en español de ReasonXL-SFT
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
