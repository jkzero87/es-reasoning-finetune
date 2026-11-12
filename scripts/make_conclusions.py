#!/usr/bin/env python3
"""Fill results/CONCLUSIONES.md from the tables. One row per finding:
# | hallazgo | número | archivo de datos | n. Every sentence is a fixed template
filled with numbers from the files below; nothing is written by hand.

Inputs:
- results/efficiency_baseline.json (scripts/efficiency_baseline.py --all);
- results/conversion_check.json (verdict of the conversion check), if present;
- results/gate_mgsm.base9b.txt (the gate ratio and its threshold, scripts/gate_ratio.py).

Rows: 2 gate rows + 4 rows per language pair (es/en, zh/en) of every table
that has data. The script prints the row count and the number it expects;
they must be equal.

  .venv/bin/python scripts/make_conclusions.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFF = ROOT / "results" / "efficiency_baseline.json"
CONV = ROOT / "results" / "conversion_check.json"
GATE = ROOT / "results" / "gate_mgsm.base9b.txt"
OUT = ROOT / "results" / "CONCLUSIONES.md"
NAMES = {"es": "español", "zh": "chino"}


def gate_rows():
    """Two rows: conversion-check verdict, and the gate as written in results/plan.md."""
    m = re.search(r"GATE metric.*?:\s*([\d.]+)\s*\(threshold >= ([\d.]+)\)", GATE.read_text())
    pairs = re.search(r"pairs (\d+)/(\d+)", GATE.read_text())
    if not m or not pairs:
        raise SystemExit(f"STOP: cannot read the ratio from {GATE}")
    ratio, thr, n = float(m.group(1)), float(m.group(2)), f"{pairs.group(1)} pares"
    gate_file = str(GATE.relative_to(ROOT))
    if not CONV.exists():
        return [("Verificación de conversión del GGUF servido: sin veredicto", "—", "results/conversion_check.json", "—"),
                (f"Compuerta MGSM es/en ≥ {thr}: NO aplicada (falta la verificación de conversión)",
                 f"{ratio:.4f}", gate_file, n)]
    c = json.loads(CONV.read_text())
    conv = (f"Verificación de conversión del GGUF servido: {c['verdict']} "
            f"(falla si la coincidencia top-1 es ≤ {c['threshold_top1_pct']}% en algún enlace)",
            f"enlace 1 {c['link1']['top1_agreement_pct']}%, enlace 2 {c['link2']['same_top_pct']}%",
            "results/conversion_check.json", f"{c['link1']['positions']} posiciones (enlace 1)")
    if c["verdict"] != "PASS":
        gate = (f"Compuerta MGSM es/en ≥ {thr}: no aplicada; la fase 1 queda invalidada y se repite (regla de la enmienda 2026-10-08)",
                f"{ratio:.4f}", gate_file, n)
    elif ratio >= thr:
        gate = (f"Compuerta MGSM es/en ≥ {thr}: se cumple en el 9B; se entrena según results/plan.md",
                f"{ratio:.4f}", gate_file, n)
    else:
        gate = (f"Compuerta MGSM es/en ≥ {thr}: no se cumple en el 9B; no se entrena; resultado \"problem not reproduced on this model\"",
                f"{ratio:.4f} < {thr}", gate_file, n)
    return [conv, gate]


def table_rows(e):
    rows = []
    where = f"{e['model']}, {e['bench']}, {e['setting']}"
    for lang, p in e["pairs"].items():
        s, en = e["langs"][lang], e["langs"]["en"]
        n = f"{p['paired_ids']} pares"
        rows += [
            (f"{where}: tokens totales por pregunta, {NAMES[lang]} / inglés", f"{p['total_token_ratio']:.4f}×", e["raw"], n),
            (f"{where}: respuestas correctas y completas por 1.000 tokens generados, {NAMES[lang]} como % del inglés",
             f"{p['success_per_1k_pct_of_en']:.1f}% ({s['success_per_1k']:.4f} vs {en['success_per_1k']:.4f})", e["raw"], n),
            (f"{where}: preguntas en que {NAMES[lang]} gasta más tokens totales que inglés",
             f"{p['ids_more_tokens_than_en']}/{p['paired_ids']}", e["raw"], n),
            (f"{where}: exactitud {NAMES[lang]} vs inglés; cortes por max_tokens {NAMES[lang]} vs inglés",
             f"{s['accuracy']:.4f} vs {en['accuracy']:.4f}; {s['length']} vs {en['length']}", e["raw"],
             f"{s['n']} y {en['n']}"),
        ]
    return rows


def build():
    """Returns (rows, expected_row_count)."""
    entries = json.loads(EFF.read_text())
    rows = gate_rows()
    pairs = 0
    for e in entries:
        if e["status"] == "not run":
            continue
        pairs += len(e["pairs"])
        rows += table_rows(e)
    return rows, 2 + 4 * pairs


def main():
    rows, expected = build()
    md = ["# Conclusiones (generado)", "",
          "Generado por `scripts/make_conclusions.py` a partir de `results/efficiency_baseline.json`, "
          "`results/conversion_check.json` y `results/gate_mgsm.base9b.txt`. No editar a mano. "
          "Definiciones en `results/plan.md` (enmiendas 2026-11-12). Todo es descriptivo salvo la compuerta.", "",
          "| # | hallazgo | número | archivo de datos | n |", "|---:|---|---|---|---|"]
    md += [f"| {i} | {a} | {b} | `{c}` | {d} |" for i, (a, b, c, d) in enumerate(rows, start=1)]
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"rows written: {len(rows)}; expected: {expected} -> {OUT.relative_to(ROOT)}")
    if len(rows) != expected:
        print("STOP: row count differs from expected")
        sys.exit(1)


if __name__ == "__main__":
    main()
