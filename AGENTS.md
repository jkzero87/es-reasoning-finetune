# AGENTS.md — es-reasoning-finetune

**Objetivo (Juan, 2026-11-12):** "Brújula: que usar la IA en español no sea una desventaja. Objetivo: con una sola GPU
de 16 GB, producir la mayor cantidad posible de datos abiertos y conclusiones prácticas
sobre cuánto, dónde y por qué el español sale más caro en modelos abiertos, y qué
remedios funcionan y cuáles no. El fine-tuning es una prueba más, no la meta."

## Reglas duras (sin excepciones)

1. Solo CPU. No uses la GPU ni lances nada que la use (nada de `train_lora.py`, `conversion_check.py`, `run_queue.sh`, `run_baseline.py`, `serve_base.sh`).
2. Nunca arranques ni detengas servidores. Nunca mates procesos.
3. Nunca toques `:8092`, salvo `POST /tokenize`.
4. Nunca hagas `git push`. Commits locales solo cuando la tarea lo diga.
5. Todo número sale de un comando. Muestra siempre el comando y su salida literal. No calcules de memoria ni redondees a mano.
6. Si un conteo o una salida difiere de lo esperado: **PARA**. No arregles nada. Informa qué comando, qué esperabas y qué salió.
7. No edites `results/plan.md`, la compuerta (1.15), ni archivos marcados "generado" o "no editar a mano". No borres archivos.
8. No inventes texto de conclusiones: las escriben los scripts.
9. Ejecuta solo los comandos escritos en `bench/TAREAS_AGENTE.md`, tal cual, desde `~/es-reasoning-finetune`.

## Mapa de carpetas

- `bench/TAREAS_AGENTE.md`: tus tareas T1–T6, en orden. `bench/CORRER.md`: cómo lanza Juan la cola de GPU (tú no).
- `scripts/`: runners, scorers (copias sin cambios de es-eval) y generadores de tablas.
- `data/`: ítems de MGSM y Belebele (en, es, zh). No se modifican.
- `results/`: planes (`plan.md`), crudos (`*_raw.<tag>.jsonl`), puntajes (`score_*.txt`), tablas (`efficiency_baseline.md`), `CONCLUSIONES.md`.
- `NEXT.md`: estado y punto de reanudación. `README.md`: resumen público (el bloque de tablas es generado).
- `~/es-eval/results/`: crudos del 27B. Solo lectura.
- Etiquetas: `base9b` (9B, pensando), `noreason9b` (9B, sin razonamiento), `prox9b` (9B, ProX-Lite), `base27b` (27B).

## Cómo informar

Una línea por tarea, con mi numeración: `T1 hecho` o `T1 NO hecho: <motivo>`, y debajo el comando y la última línea de su salida.
Si paraste por la regla 6, no sigas con la tarea siguiente.
