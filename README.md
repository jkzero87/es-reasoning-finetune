# es-reasoning-finetune

**Estado: en curso (iniciado 2026-10-07).**

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

1. **Línea base**: MGSM en y es sobre Qwen3-14B (GGUF Q4_K_M, llama-server
   local), con los mismos ajustes de generación que la línea base del 27B.
2. **Ajuste fino**: QLoRA sobre Qwen3-14B (Unsloth).
3. **Pruebas de falsación**: criterios fijados antes de mirar los resultados.

El plan completo, con fases y criterios, va en `results/plan.md` y se
publica antes de cualquier corrida del 14B.

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

`data/mgsm_{en,es}.jsonl` y las salidas crudas en `results/` contienen el
texto de MGSM y se comparten bajo **CC BY-SA 4.0**, como exige el conjunto
de datos.

| conjunto | vía | licencia |
|---|---|---|
| MGSM (Shi et al., 2022), basado en GSM8K (Cobbe et al., 2021) | Hugging Face `juletxara/mgsm` | CC BY-SA 4.0; GSM8K: MIT |

Atribución:
- Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush
  Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das,
  Jason Wei. *Language Models are Multilingual Chain-of-Thought Reasoners.*
  arXiv:2210.03057, 2022.
- Karl Cobbe et al. *Training Verifiers to Solve Math Word Problems.*
  arXiv:2110.14168, 2021.
