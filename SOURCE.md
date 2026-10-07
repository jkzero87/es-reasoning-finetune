# Procedencia de los archivos copiados

Copiados sin modificar de [jkzero87/es-eval](https://github.com/jkzero87/es-eval)
en el commit `1a44d21871dff4618b725669835d3316602b2917` (2026-10-07), para
reproducir exactamente el puntaje de MGSM.

| archivo | sha256 |
|---|---|
| `scripts/run_mgsm.py` | `2645434a7affdb3376af46e4fc12e91dfaea4e2d4c11d5cd61350ee40d14b5a9` |
| `scripts/score_mgsm.py` | `0f4e71421d993c09c15c715b694db0cbec33180d927d15ca05ee2e466ea157bf` |
| `data/mgsm_en.jsonl` | `0cb0acbad091022b4ab3c4502f9e634025ae7f1d81aec0e1e13de4d2b763d6ed` |
| `data/mgsm_es.jsonl` | `f90f2eda3c16770c91a3b6a78da9ee7199db55b77d5d6b28e6556fa12caea905` |

- `data/mgsm_{en,es}.jsonl`: split `test` de `juletxara/mgsm` (250 ítems por
  idioma; `id`, `question`, `answer_number`). Los 250 `answer_number` de cada
  idioma coinciden con el `gold` de la línea base del 27B
  (`es-eval/results/mgsm_raw.jsonl`).
- `scripts/run_mgsm.py` carga MGSM desde Hugging Face (no desde `data/`) y
  apunta a `http://127.0.0.1:8092`. Cualquier ajuste (puerto, modelo, lectura
  de `data/`, hora de corte) va en un runner nuevo, no en esta copia.
- `scripts/score_mgsm.py`: regla de puntaje (último número = gold; separadores
  de miles quitados) y McNemar exacto. Las funciones `last_number`, `correct`
  y `mcnemar_exact` no dependen del servidor; el conteo de tokens de
  razonamiento usa `/tokenize` del servidor (se omite con `--no-tokenize`).
