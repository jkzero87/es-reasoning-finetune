#!/usr/bin/env python3
"""Regenerate the results block of README.md between the two TABLES markers.

Content: the gate rows (same text as results/CONCLUSIONES.md) and one summary
row per table of results/efficiency_baseline.json that has data. Everything
outside the markers is left as it is. Stops if a marker is missing.

  .venv/bin/python scripts/update_readme_tables.py     then: git diff README.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_conclusions  # noqa: E402

README = ROOT / "README.md"
BEGIN = "<!-- TABLES:BEGIN (generado por scripts/update_readme_tables.py; no editar a mano) -->"
END = "<!-- TABLES:END -->"


def cell(e, lang, key, fmt):
    p = e["pairs"].get(lang)
    return fmt.format(p[key]) if p else "—"


def main():
    s = README.read_text(encoding="utf-8")
    if s.count(BEGIN) != 1 or s.count(END) != 1:
        raise SystemExit("STOP: README.md must hold exactly one TABLES:BEGIN and one TABLES:END marker")
    entries = json.loads(make_conclusions.EFF.read_text())
    out = [BEGIN, "## Resultados", "",
           "Tablas completas: `results/efficiency_baseline.md`; hallazgos uno por fila: `results/CONCLUSIONES.md`.", ""]
    out += [f"- {a}: **{b}** (`{c}`, {d})." for a, b, c, d in make_conclusions.gate_rows()]
    out += ["", "Éxito por 1k = respuestas correctas y completas por cada 1.000 tokens generados, como % del inglés "
            "(menos de 100 % = ese idioma sale más caro).", "",
            "| modelo | prueba | condición | n (en/es/zh) | tokens totales es/en | éxito por 1k, es vs en "
            "| tokens totales zh/en | éxito por 1k, zh vs en |", "|---|---|---|---:|---:|---:|---:|---:|"]
    n_rows = 0
    for e in entries:
        if e["status"] == "not run":
            continue
        n_rows += 1
        n = "/".join(str(v["n"]) for v in e["langs"].values())
        out.append(f"| {e['model']} | {e['bench']} | {e['setting']} | {n} "
                   f"| {cell(e, 'es', 'total_token_ratio', '{:.4f}×')} | {cell(e, 'es', 'success_per_1k_pct_of_en', '{:.1f} %')} "
                   f"| {cell(e, 'zh', 'total_token_ratio', '{:.4f}×')} | {cell(e, 'zh', 'success_per_1k_pct_of_en', '{:.1f} %')} |")
    out.append(END)
    i, j = s.index(BEGIN), s.index(END) + len(END)
    README.write_text(s[:i] + "\n".join(out) + s[j:], encoding="utf-8")
    print(f"README.md results block regenerated: {n_rows} table rows")


if __name__ == "__main__":
    main()
