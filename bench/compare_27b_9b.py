#!/usr/bin/env python3
"""Comparación justa 27B vs 9B sobre los MISMOS ids.

Belebele: sample de 300 ids (results/belebele_sample_300.json) -> 27B sobre
sus 300 (de los 488 por idioma) vs 9B sobre sus 300.
MGSM: los 250 ids por idioma en ambos.

Salida: tabla reason/answer/total medios por idioma y ratios es/en de
reason (27B-488, 27B-300, 9B-300). Todo por tool, sin cifras de cabeza.
"""
import json
import os

SAMPLE_300 = json.load(open("results/belebele_sample_300.json"))["ids"]
T27_BEL = os.path.expanduser("~/es-eval/results/belebele_token_split.jsonl")
T27_MGSM = os.path.expanduser("~/es-eval/results/mgsm_token_split.jsonl")
T9B_BEL = "results/belebele_token_split.base9b.jsonl"
T9B_MGSM = "results/mgsm_token_split.base9b.jsonl"


def load(path, ids_by_lang=None):
    """rows -> dict lang -> list[(reason, answer)]. Optional filter por ids."""
    acc = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if ids_by_lang is not None and d["id"] not in ids_by_lang.get(d["lang"], ()):
            continue
        acc.setdefault(d["lang"], []).append((d["reasoning_tokens"], d["answer_tokens"]))
    return acc


def summarize(rows):
    out = {}
    for lang, lst in sorted(rows.items()):
        n = len(lst)
        r = sum(x[0] for x in lst) / n
        a = sum(x[1] for x in lst) / n
        out[lang] = (n, r, a, r + a)
    return out


def table(name, rows):
    print(f"\n=== {name} ===")
    print(f"{'lang':>5} {'n':>5} {'reason':>9} {'answer':>9} {'total':>9}")
    for lang in sorted(rows):
        n, r, a, t = rows[lang]
        print(f"{lang:>5} {n:>5} {r:>9.1f} {a:>9.1f} {t:>9.1f}")
    r_en = rows["en"][1]
    print("ratios (vs en):")
    for lang in sorted(rows):
        if lang != "en":
            print(f"  {lang}/en reason: {rows[lang][1] / r_en:.4f}")


def ids_of(path, lang):
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")
            if json.loads(l)["lang"] == lang}


def main():
    # Cruce de ids MGSM 250 vs 250
    for lang in ("es", "en"):
        a = ids_of(T27_MGSM, lang)
        b = ids_of(T9B_MGSM, lang)
        print(f"MGSM {lang}: 27B n={len(a)} 9B n={len(b)} iguales={a == b}")
    print(f"sample_300: n={len(SAMPLE_300)} ids unicos={len(set(SAMPLE_300))}")

    # Belebele 27B, los 488 (sin filtro)
    bel_27b_all = summarize(load(T27_BEL))
    # Belebele 27B, solo los 300 del sample
    filter300 = {"es": set(SAMPLE_300), "en": set(SAMPLE_300)}
    bel_27b_300 = summarize(load(T27_BEL, filter300))
    # Belebele 9B, 300 (el raw ya solo tiene esos)
    bel_9b = summarize(load(T9B_BEL))

    table("Belebele 27B, 488 por idioma (todo el set)", bel_27b_all)
    table("Belebele 27B, 300 del sample (mismos ids del 9B)", bel_27b_300)
    table("Belebele 9B base, 300 por idioma", bel_9b)

    # MGSM 250 vs 250 (los ids son 1..250 en ambos; se comprueba el cruce)
    mgsm_27b = summarize(load(T27_MGSM))
    mgsm_9b = summarize(load(T9B_MGSM))
    table("MGSM 27B, 250 por idioma", mgsm_27b)
    table("MGSM 9B base, 250 por idioma", mgsm_9b)


if __name__ == "__main__":
    main()
