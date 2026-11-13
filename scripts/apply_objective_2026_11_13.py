import re, sys
NEW = ("Norte: que usar un modelo de lenguaje en español no cueste más que usarlo en inglés. "
       "Es la dirección del trabajo, no lo que este proyecto promete resolver. "
       "Objetivo: identificar los factores que hacen más costoso usar modelos de lenguaje en español "
       "y atacarlos mediante fine-tuning, reentrenando sobre una NVIDIA RTX 5060 Ti de 16 GB, "
       "para obtener conclusiones que aporten a reducir esa desventaja. El fine-tuning es el centro del proyecto.")
STATUS = ("**Estado (2026-11-13):** fase de diagnóstico concluida (27B y 9B; en, es y zh) y prueba causal sobre "
          "errores de traducción hecha (`results/causal_test_2026-11-12.md`). La compuerta de MGSM de la fase 1 se "
          "aplicó tal como estaba escrita: en matemáticas el 9B gasta en español 1,1235 × los tokens del inglés, por "
          "debajo de 1,15, y ese resultado queda publicado (`results/gate_decision.md`). Desde la enmienda del "
          "2026-11-13 el fine-tuning es el eje del proyecto: ataca el sobrecosto medido en lectura (1,5401 ×) y el que "
          "persiste con español bien traducido, con criterios fijados antes de entrenar.")
AMEND = """

## Amendment 2026-11-13 (project objective restated; fine-tuning becomes the core; before any training)

Objective as restated by Juan on 2026-11-13 (verbatim, Spanish; replaces the 2026-11-12 wording in README.md and AGENTS.md):

\"""" + NEW + """\"

Consequences, decided before any training run:
1. The 2026-11-12 objective text stays in this file as history; README.md, AGENTS.md and results/RESUMEN_REUNION.md now carry this one.
2. The Phase-1 gate (MGSM es/en >= 1.15) was applied as written and its result stands: "problem not reproduced on this model" for MGSM. It no longer decides whether the project trains.
3. Fine-tuning (arms A, B, control) is the core of the project. It targets the overcost measured on Belebele (9B es/en 1.5401) and the overcost that persists on correctly translated Spanish (results/causal_test_2026-11-12.md).
4. Success criteria, the evaluation set (cleaned of translation errors) and the guards (accuracy must not drop; non-terminating answers must not rise) will be written as a separate dated amendment BEFORE the first training run. No arm is trained until that amendment exists.
"""
def sub(path, pattern, repl, flags=re.S):
    txt = open(path, encoding="utf-8").read()
    new, n = re.subn(pattern, lambda m: repl, txt, flags=flags)
    if n != 1:
        sys.exit(f"STOP: {path}: pattern matched {n} times, expected 1: {pattern[:60]}")
    open(path, "w", encoding="utf-8").write(new)
    print(f"ok {path}")
sub("README.md", r"Brújula: que usar la IA.*?no la meta\.", NEW)
sub("README.md", r"\*\*Estado \(2026-11-12\):\*\*.*?ProX-Lite\)\.", STATUS)
sub("README.md", r"\*\*Ajuste fino\*\*", "**Fine-tuning**")
sub("README.md", r"tarjeta local de 16 GB", "NVIDIA RTX 5060 Ti de 16 GB")
sub("AGENTS.md", r'\*\*Objetivo \(Juan, 2026-11-12\):\*\* "Brújula:.*?no la meta\."', '**Objetivo (Juan, 2026-11-13):** "' + NEW + '"')
sub("results/RESUMEN_REUNION.md", r'"Brújula: que usar la IA.*?no la meta\."', '"' + NEW + '"')
with open("results/plan.md", "a", encoding="utf-8") as f:
    f.write(AMEND)
print("ok results/plan.md (amendment appended)")
