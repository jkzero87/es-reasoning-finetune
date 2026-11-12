#!/usr/bin/env python3
"""Benchmark base del Qwen3.5-9B, medido IGUAL que el 27B (~/es-eval).

Misma métrica que es-eval/scripts/mgsm_token_split.py: para cada petición se
cuenta reasoning_content y content por separado con el /tokenize del
llama-server del 9B (sin tokens especiales; texto vacío = 0). Mismo set de
prompts y misma configuración de generación que el 27B (importados de las
copias no modificadas de es-eval/scripts/run_mgsm.py y run_belebele.py en
scripts/ de este repo: temperature 1.0, top_p 0.95, top_k 20, seed 42,
max_tokens 8192, sin system prompt).

Salidas (bench/):
  <bench>_raw.<tag>.jsonl   crudo por petición (id, lang, gold, content,
                            reasoning_content, finish_reason, usage, timings,
                            wall, tag, [qid]) — igual que es-eval.
  <bench>_raw.<tag>.csv     tabla cruda por petición (mismo orden de append).
  <bench>_summary.<tag>.json  resumen por idioma: n, media/min/max de
                            reasoning_tokens, answer_tokens y total_tokens
                            (usage prompt+completion), es/en y zh/en ratios.

Reanudable: un (id, lang) ya escrito en el jsonl/csv no se vuelve a correr.
--limit N: como mucho N pendientes por idioma.
--stop-at HH:MM: no se empieza nada a esa hora ni después (lo in-flight se
termina y se escribe). Al detener (Ctrl-C o stop-at) el estado queda escrito:
el jsonl/csv son la fuente de verdad y summary se regenera al relanzar.

Uso:
  python bench/baseline_9b.py --bench mgsm --tag base9b --limit 3
  python bench/baseline_9b.py --bench belebele --tag base9b
  python bench/baseline_9b.py --summary-only          # regenera JSON desde los raw

NO lanza ni gestiona el llama-server; ver CORRER.md (puerto por defecto 8093,
para no chocar con el 8092 del 27B).
"""
import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import requests  # noqa: E402

from run_mgsm import GENERATION as MGSM_GEN, PROMPTS as MGSM_PROMPTS  # noqa: E402
from run_belebele import (  # noqa: E402
    GENERATION as BBL_GEN,
    build_prompt,
    LETTERS,
    PROMPTS as BBL_PROMPTS,
)

BENCHES = {
    "mgsm": {"data": "mgsm", "gen": MGSM_GEN, "prompts": MGSM_PROMPTS, "ids_start": 1},
    "belebele": {"data": "belebele", "gen": BBL_GEN, "prompts": BBL_PROMPTS, "ids_start": 1},
}
LANG_ORDER = ["es", "en", "zh"]
BENCH_DIR = ROOT / "bench"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def done_pairs(raw_out):
    done = set()
    if raw_out.exists():
        for line in raw_out.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                done.add((rec["id"], rec["lang"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_items(bench, langs):
    """{lang: {id: item}} desde data/<bench>_<lang>.jsonl (igual que es-eval)."""
    items = {}
    for lang in langs:
        path = ROOT / "data" / f"{BENCHES[bench]['data']}_{lang}.jsonl"
        if not path.exists():
            sys.exit(f"falta {path}; cópialo desde ~/es-eval/data/")
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if bench == "belebele":
            # filas de pregunta (id, qid, passage, question, options, gold);
            # las filas de opciones se ignoran (las preguntas ya traen todo).
            rows = [r for r in rows if "question" in r]
        items[lang] = {r["id"]: r for r in rows}
    return items


def build_user(bench, lang, item):
    if bench == "mgsm":
        return MGSM_PROMPTS[lang] + "\n\n" + item["question"]
    return build_prompt(lang, item)


def tokenize_count(base, text):
    """Como mgsm_token_split.count(): /tokenize sin tokens especiales; '' = 0."""
    if not text:
        return 0
    r = requests.post(f"{base}/tokenize", json={"content": text}, timeout=60)
    r.raise_for_status()
    return len(r.json()["tokens"])


def write_outputs(base, bench, tag, pending, stop_at, csv_lock, state):
    cfg = BENCHES[bench]
    raw_out = BENCH_DIR / f"{bench}_raw.{tag}.jsonl"
    csv_out = BENCH_DIR / f"{bench}_raw.{tag}.csv"
    props_out = BENCH_DIR / f"server_props.{tag}.json"
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    if not props_out.exists():
        r = requests.get(f"{base}/props", timeout=30)
        r.raise_for_status()
        props_out.write_text(json.dumps(r.json(), indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"server props -> {props_out.name}")

    header = ["id", "lang", "gold", "reasoning_tokens", "answer_tokens",
              "prompt_tokens", "completion_tokens", "total_tokens",
              "finish_reason", "wall_s"]
    csv_new = not csv_out.exists()
    if csv_new:
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    n_ok = 0
    for n, (lang, item) in enumerate(pending, start=1):
        if state.get("stop"):
            log("stop requested; no new items")
            break
        if stop_at is not None and datetime.now() >= stop_at:
            log(f"stop_at {stop_at:%H:%M} reached; no new items")
            break
        i = item["id"]
        payload = {
            "messages": [{"role": "user", "content": build_user(bench, lang, item)}],
            **cfg["gen"],
        }
        try:
            t0 = time.monotonic()
            r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=3600)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"id={i} {lang}: FAILED ({e}); will retry next run")
            continue
        wall = time.monotonic() - t0
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        rec = {
            "id": i,
            "lang": lang,
            "gold": item.get("gold", item.get("answer_number")),
            "content": msg.get("content"),
            "reasoning_content": msg.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "timings": data.get("timings"),
            "wall": wall,
            "tag": tag,
        }
        if bench == "belebele":
            rec["qid"] = item.get("qid")
        # Métrica es-eval: reasoning y answer tokenizados por separado.
        try:
            rec["reasoning_tokens"] = tokenize_count(base, rec.get("reasoning_content"))
            rec["answer_tokens"] = tokenize_count(base, rec.get("content"))
        except Exception as e:
            log(f"id={i} {lang}: /tokenize FAILED ({e}); counts left null, will retry next run")
            rec["reasoning_tokens"] = None
            rec["answer_tokens"] = None
        with csv_lock:
            with raw_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            u = rec.get("usage") or {}
            row = [i, lang, rec["gold"], rec["reasoning_tokens"], rec["answer_tokens"],
                   u.get("prompt_tokens"), u.get("completion_tokens"),
                   (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0),
                   rec["finish_reason"], f"{wall:.3f}"]
            with csv_out.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            n_ok += 1
        log(f"({n}/{len(pending)}) id={i} {lang} reason={rec['reasoning_tokens']} "
            f"ans={rec['answer_tokens']} finish={rec['finish_reason']} wall={wall:.1f}s")
    return n_ok


def summarize(bench, tag):
    """Resumen por idioma desde el raw (misma métrica que mgsm_token_split)."""
    raw = BENCH_DIR / f"{bench}_raw.{tag}.jsonl"
    out = BENCH_DIR / f"{bench}_summary.{tag}.json"
    stats = {}
    n_total = 0
    if raw.exists():
        for line in raw.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("reasoning_tokens") is None:
                continue
            n_total += 1
            s = stats.setdefault(rec["lang"], {"n": 0, "reason": 0, "ans": 0, "total": 0,
                                               "reason_min": 10**9, "reason_max": 0})
            u = rec.get("usage") or {}
            s["n"] += 1
            s["reason"] += rec["reasoning_tokens"]
            s["ans"] += rec["answer_tokens"]
            tot = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
            s["total"] += tot
            s["reason_min"] = min(s["reason_min"], rec["reasoning_tokens"])
            s["reason_max"] = max(s["reason_max"], rec["reasoning_tokens"])
    summary = {
        "bench": bench,
        "tag": tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metric": "reasoning_tokens/answer_tokens via server /tokenize (no special tokens), "
                  "same as es-eval mgsm_token_split.py; total_tokens = usage prompt+completion",
        "records": n_total,
        "langs": {},
    }
    for lang in sorted(stats):
        s = stats[lang]
        summary["langs"][lang] = {
            "n": s["n"],
            "mean_reasoning": round(s["reason"] / s["n"], 1),
            "mean_answer": round(s["ans"] / s["n"], 1),
            "mean_total": round(s["total"] / s["n"], 1),
            "reasoning_min": s["reason_min"],
            "reasoning_max": s["reason_max"],
        }
    for a, b in (("es", "en"), ("zh", "en")):
        if a in summary["langs"] and b in summary["langs"]:
            ra, rb = summary["langs"][a], summary["langs"][b]
            summary[f"{a}_vs_{b}"] = {
                "reasoning_ratio": round(ra["mean_reasoning"] / rb["mean_reasoning"], 3),
                "answer_ratio": round(ra["mean_answer"] / rb["mean_answer"], 3),
                "total_ratio": round(ra["mean_total"] / rb["mean_total"], 3),
            }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"summary -> {out.name} ({n_total} records)")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", choices=["mgsm", "belebele"], default="mgsm")
    ap.add_argument("--port", type=int, default=8093,
                    help="llama-server del 9B (8092 es el 27B; no tocar)")
    ap.add_argument("--tag", default="base9b")
    ap.add_argument("--langs", default=",".join(LANG_ORDER))
    ap.add_argument("--limit", type=int, help="máximo de pendientes por idioma")
    ap.add_argument("--stop-at", metavar="HH:MM",
                    help="no empezar items a esta hora ni después (local)")
    ap.add_argument("--summary-only", action="store_true",
                    help="no corre nada; regenera el JSON desde el raw existente")
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    unknown = set(langs) - set(LANG_ORDER)
    if unknown or not langs:
        ap.error(f"--langs must be a subset of {','.join(LANG_ORDER)}")
    langs = [l for l in LANG_ORDER if l in langs]  # orden canónico es,en,zh

    base = f"http://127.0.0.1:{args.port}"
    if args.summary_only:
        summarize(args.bench, args.tag)
        return

    stop_at = None
    if args.stop_at:
        hh, mm = args.stop_at.split(":")
        stop_at = datetime.now().replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)

    items = load_items(args.bench, langs)
    done = done_pairs(BENCH_DIR / f"{args.bench}_raw.{args.tag}.jsonl")
    pending = []
    for lang in langs:
        p = [it for it in sorted(items[lang], key=lambda r: r["id"])
             if (it["id"], lang) not in done]
        if args.limit is not None:
            p = p[:args.limit]
        pending.extend((lang, it) for it in p)
    log(f"bench={args.bench} tag={args.tag} port={args.port} langs={langs} "
        f"done={len(done)} pending={len(pending)} stop_at={stop_at or '-'}")
    if not pending:
        summarize(args.bench, args.tag)
        return

    state = {"stop": False}

    def handle_stop(signum, frame):
        log(f"signal {signum}: finishing in-flight item, then writing state and exiting")
        state["stop"] = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    import threading
    lock = threading.Lock()
    write_outputs(base, args.bench, args.tag, pending, stop_at, lock, state)
    summarize(args.bench, args.tag)
    log("done; state written to raw jsonl/csv (resume by re-running the same command)")


if __name__ == "__main__":
    main()
