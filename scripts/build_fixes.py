#!/usr/bin/env python3
"""Write data/fixes_2026-11-12.jsonl: corrected Spanish for the 11 items labelled E
in results/label_audit_2026-11-12.md. One row per item; only the listed fields change.

Row: {"bench", "id", "field", "original", "fixed"}; "field" is "question" or
"options". For options, "original"/"fixed" are the full 4-option lists (0-indexed
A..D) so a runner can replace the field as a whole.
Each original must equal the value in data/<bench>_es.jsonl and differ from the fix;
otherwise the script stops without writing. data/*.jsonl are not modified.
Usage: scripts/build_fixes.py  (no arguments; run from the repo root)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fixes_2026-11-12.jsonl"
MGSM = {
    215: "Jerry tiene dos piscinas, ambas con fugas que las van vaciando a una velocidad de 4 galones/minuto. Hace 4 minutos, la piscina grande tenía el doble de agua que la piscina pequeña. Ahora, la piscina grande tiene cuatro veces la cantidad de agua de la piscina pequeña. ¿Qué cantidad de agua tiene ahora la piscina pequeña?",
    39: "John corre 60 millas por semana. Corre 3 días a la semana. El primer día corre 3 horas y los otros dos días corre la mitad de ese tiempo. ¿A qué velocidad corre?",
    71: "Judy da 5 clases de baile cada día entre semana y 8 clases el sábado. Si en cada clase hay 15 alumnos y ella cobra $15,00 por alumno, ¿cuánto dinero recauda en 1 semana?",
    199: "Al principio, Elaine tenía 20 cartas de Pokémon. Después de un mes, había reunido el triple de esa cantidad. El segundo mes reunió 20 cartas menos que las que reunió el primer mes. El tercer mes reunió el doble de la suma de las cartas que reunió el primer y el segundo mes. ¿Cuántas cartas de Pokémon tiene ahora en total?",
    111: "Marcell y Beatrice están haciendo una competencia para ver quién puede comer la mayor cantidad de rollitos de fruta, por lo que desenrollan todos los que pueden encontrar. Lamentablemente, alguien comete un error y lo de Beatrice tenía dos rollitos de ancho y 24 de largo, mientras que lo de Marcell tenía 3 rollitos de ancho y 14 de largo. Si ambos comieron su cantidad completa, ¿cuántos comieron en promedio?",
    94: "Lee antes podía correr los 400 metros con obstáculos dos segundos más rápido que Gerald. Pero Gerald cambió su dieta, lo que mejoró un 10 % su velocidad. Si Lee corre los 400 metros con obstáculos en 38 segundos, ¿en cuántos segundos puede correrlos Gerald con su dieta mejorada?",
}
# id -> (question or None, {option index: text})
BELEBELE = {
    376: (None, {1: "Acercarse demasiado despacio"}),
    388: ("Según el fragmento, ¿de qué están empezando a ofrecer una selección más variada los supermercados?", {}),
    224: (None, {0: "En los años 1800"}),
    483: (None, {3: "Mostrarse firme cuando sea necesario"}),
    304: (None, {2: "Activación de los propulsores", 3: "Recuperar la orientación normal"}),
}


def load(bench):
    return {r["id"]: r for r in map(json.loads, (ROOT / "data" / f"{bench}_es.jsonl").open(encoding="utf-8"))}


def main():
    rows = []
    m, b = load("mgsm"), load("belebele")
    for i, q in MGSM.items():
        rows.append({"bench": "mgsm", "id": i, "field": "question", "original": m[i]["question"], "fixed": q})
    for i, (q, opts) in BELEBELE.items():
        if q is not None:
            rows.append({"bench": "belebele", "id": i, "field": "question", "original": b[i]["question"], "fixed": q})
        if opts:
            orig = b[i]["options"]
            if len(orig) != 4:
                sys.exit(f"STOP: belebele {i}: {len(orig)} options, expected 4")
            new = [opts.get(k, o) for k, o in enumerate(orig)]
            rows.append({"bench": "belebele", "id": i, "field": "options", "original": orig, "fixed": new})
    reviewed = {}  # (tag, id) -> ES fields as shown to the annotator
    tag = None
    for line in (ROOT / "results" / "label_sample.md").open(encoding="utf-8"):
        if line.startswith("[") and " id=" in line:
            tag, i = line[1:].split()[0], int(line.split(" id=")[1].split()[0])
        elif line.startswith("ES: "):
            reviewed[(tag, i)] = json.loads(line[4:])
    for r in rows:
        shown = reviewed.get(("mgsm27b" if r["bench"] == "mgsm" else "belebele9b", r["id"]), {}).get(r["field"])
        if shown != r["original"]:
            sys.exit(f"STOP: {r['bench']} {r['id']} {r['field']}: data/ differs from the reviewed text in results/label_sample.md")
        if r["original"] == r["fixed"]:
            sys.exit(f"STOP: {r['bench']} {r['id']} {r['field']}: fix equals the original")
    if len({r["id"] for r in rows if r["bench"] == "mgsm"}) != 6 or len({r["id"] for r in rows if r["bench"] == "belebele"}) != 5:
        sys.exit("STOP: expected 6 MGSM and 5 Belebele items")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows, 11 items")


if __name__ == "__main__":
    main()
