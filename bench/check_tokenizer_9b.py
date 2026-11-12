#!/usr/bin/env python3
"""Verifica el tokenizer CPU del 9B (tokenizer.json, add_special_tokens=False)
contra el /tokenize de http://127.0.0.1:8092 en 30 textos de data/.

Solo hace POST /tokenize al 8092. No arranca servidor ni usa GPU.
Salida: X/30 y detalle de los que fallan.
"""
import json, random, sys, urllib.request

random.seed(20261020)
TOKENIZER_JSON = "/home/jkzero/models/Qwen3.5-9B-text-mtp/tokenizer.json"
BASE = "http://127.0.0.1:8092"
FILES = ["data/mgsm_es.jsonl", "data/belebele_es.jsonl", "data/mgsm_zh.jsonl",
         "data/belebele_zh.jsonl", "data/mgsm_en.jsonl", "data/belebele_en.jsonl"]
PER_LANG = 5  # es, en, zh de dos archivos c/u = 10 por idioma, 30 total


def server_len(text):
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.loads(r.read())["tokens"])


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    texts = []
    for path in FILES:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        random.shuffle(rows)
        for r in rows[:PER_LANG]:
            # texto = campo de contenido del prompt (question / flores_passage)
            field = "flores_passage" if "flores_passage" in r else "question"
            texts.append((path, r["id"], field, r[field]))
    print(f"total textos: {len(texts)}")
    ok, bad = 0, []
    for path, rid, field, text in texts:
        cpu = len(tok.encode(text, add_special_tokens=False).ids)
        srv = server_len(text)
        if cpu == srv:
            ok += 1
        else:
            bad.append((path, rid, field, cpu, srv))
    for b in bad:
        print(f"MISMATCH {b[0]} id={b[1]} campo={b[2]} cpu={b[3]} server={b[4]}")
    print(f"RESULT: {ok}/{len(texts)}")
    return 0 if ok == len(texts) else 1


if __name__ == "__main__":
    sys.exit(main())
