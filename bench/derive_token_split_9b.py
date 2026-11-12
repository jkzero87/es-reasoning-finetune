#!/usr/bin/env python3
"""Deriva <bench>_token_split.base9b.jsonl de los raw existentes con el
tokenizer CPU del 9B (tokenizer.json, add_special_tokens=False).

reasoning_content -> reasoning_tokens; content -> answer_tokens;
content vacío -> 0 (encode('') = 0 ids; el texto de content nunca está vacío
en los raw, pero se trata igual).

Formato igual que los token_split del 27B en ~/es-eval/results:
  {"id": ..., "lang": ..., "reasoning_tokens": N, "answer_tokens": N}
mismo orden de filas que el raw de entrada.
"""
import json, sys

TOKENIZER_JSON = "/home/jkzero/models/Qwen3.5-9B-text-mtp/tokenizer.json"

JOBS = [
    ("results/mgsm_raw.base9b.jsonl", "results/mgsm_token_split.base9b.jsonl"),
    ("results/belebele_raw.base9b.jsonl", "results/belebele_token_split.base9b.jsonl"),
]


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)

    for src, dst in JOBS:
        n = 0
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                r = rec.get("reasoning_content") or ""
                c = rec.get("content") or ""
                rt = len(tok.encode(r, add_special_tokens=False).ids) if r else 0
                at = len(tok.encode(c, add_special_tokens=False).ids) if c else 0
                out.write(json.dumps(
                    {"id": rec["id"], "lang": rec["lang"],
                     "reasoning_tokens": rt, "answer_tokens": at},
                    ensure_ascii=False) + "\n")
                n += 1
        print(f"{dst}: {n} filas")


if __name__ == "__main__":
    sys.exit(main())
