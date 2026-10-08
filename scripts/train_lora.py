#!/usr/bin/env python3
"""Train one arm (A, B or control) of results/train_plan.md, amendment 2026-10-08.

Base: the text-only Qwen3.5-9B checkpoint (no vision, no MTP), loaded in 8-bit
(bitsandbytes) through Unsloth, LoRA adapters in bf16; same loading path as
scripts/feasibility_lora.py option A. Hyperparameters are constants below,
identical for all arms, and must match the train_plan amendment.

Data: data/train/{arm}.jsonl in file order (the build's seeded draw order),
1 epoch, batch 1, GRAD_ACCUM samples per optimizer step. Each sample is the
full chat-formatted conversation (Qwen3.5 template, tokenize=False, then the
tokenizer without extra special tokens: the same rendering that
build_train_sets.py measured). Loss only on the final assistant turn: the
tokens after the last `<|im_start|>assistant\\n` up to and including its
`<|im_end|>` (reasoning + answer); everything before is masked (-100). The
accumulated loss is token-weighted: sum of per-token losses over the
GRAD_ACCUM samples / number of loss tokens in them.

Checkpoints: CKPT_DIR/<arm>/step-NNNN/ every CKPT_EVERY optimizer steps and
when --stop-at is reached (adapter + optimizer + scheduler + RNG states +
step), written to a temp dir and renamed. --resume continues from the
latest; without --resume an existing checkpoint dir is an error. At the end
the adapter goes to CKPT_DIR/<arm>/final/. --stop-at HH:MM (today): no new
optimizer step starts at/after that time; save and exit 0.

--dry-run: tokenizer + data + batches only, no model, no GPU. Prints total
optimizer steps and tokens (all / loss) per step; checks every sample's
length equals the build's n_tokens and is <= MAX_LEN.

  .venv-train/bin/python scripts/train_lora.py --arm A --dry-run
  .venv-train/bin/python scripts/train_lora.py --arm A --stop-at 18:45 [--resume]
"""
import argparse
import datetime as dt
import json
import math
import os
import random
import shutil
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "/home/jkzero/models/Qwen3.5-9B-text"
CKPT_DIR = ROOT / "checkpoints"
ARMS = ("A", "B", "control")

# Hyperparameters (results/train_plan.md, amendment 2026-10-08). Same for all arms.
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 4096
EPOCHS = 1
BATCH = 1
GRAD_ACCUM = 8
LR = 1e-4
WARMUP_FRAC = 0.03
WEIGHT_DECAY = 0.01  # torch AdamW default, as in feasibility_lora.py
MAX_GRAD_NORM = 1.0
SEED = 20261007
CKPT_EVERY = 50

ASSIST = "<|im_start|>assistant\n"
END = "<|im_end|>"


def log(msg):
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def encode(tok, messages):
    """Token ids and labels (-100 outside the final assistant turn)."""
    text = tok.apply_chat_template(messages, tokenize=False)
    a = text.rindex(ASSIST) + len(ASSIST)
    b = text.index(END, a) + len(END)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    labels = [t if s >= a and e <= b else -100 for t, (s, e) in zip(ids, enc["offset_mapping"])]
    return ids, labels


def load_data(tok, arm):
    rows = [json.loads(l) for l in (ROOT / "data" / "train" / f"{arm}.jsonl").open(encoding="utf-8")]
    out = []
    for r in rows:
        ids, labels = encode(tok, r["messages"])
        if len(ids) != r["n_tokens"]:
            raise SystemExit(f"row {r['row_index']}: {len(ids)} tokens, build said {r['n_tokens']}")
        if len(ids) > MAX_LEN:
            raise SystemExit(f"row {r['row_index']}: {len(ids)} > {MAX_LEN}")
        n_loss = sum(x != -100 for x in labels[1:])  # labels are shifted inside the model
        if n_loss == 0:
            raise SystemExit(f"row {r['row_index']}: no loss tokens")
        out.append({"row_index": r["row_index"], "ids": ids, "labels": labels, "n_loss": n_loss})
    return out


def plan_steps(data):
    per_epoch = math.ceil(len(data) / (BATCH * GRAD_ACCUM))
    return [data[i * GRAD_ACCUM:(i + 1) * GRAD_ACCUM] for _ in range(EPOCHS) for i in range(per_epoch)]


def describe(xs):
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": st.median(xs),
            "min": xs[0], "max": xs[-1], "sum": sum(xs)}


def latest_ckpt(arm_dir):
    steps = sorted(arm_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    return steps[-1] if steps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--stop-at", help="HH:MM today; no new optimizer step at/after it")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stop_at = None
    if args.stop_at:
        hh, mm = args.stop_at.split(":")
        stop_at = dt.datetime.now().replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)

    if not args.dry_run:
        import unsloth  # noqa: F401  (must be imported before transformers)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    data = load_data(tok, args.arm)
    steps = plan_steps(data)
    total = len(steps)
    warmup = math.ceil(WARMUP_FRAC * total)
    tok_all = [sum(len(s["ids"]) for s in b) for b in steps]
    tok_loss = [sum(s["n_loss"] for s in b) for b in steps]
    summary = {"arm": args.arm, "samples": len(data), "epochs": EPOCHS, "batch": BATCH,
               "grad_accum": GRAD_ACCUM, "optimizer_steps": total, "warmup_steps": warmup,
               "tokens_per_step": describe(tok_all), "loss_tokens_per_step": describe(tok_loss),
               "loss_token_share": round(sum(tok_loss) / sum(tok_all), 3)}
    log("plan " + json.dumps(summary))
    if args.dry_run:
        return

    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_cosine_schedule_with_warmup
    from unsloth import FastLanguageModel

    arm_dir = CKPT_DIR / args.arm
    ck = latest_ckpt(arm_dir) if arm_dir.exists() else None
    if ck and not args.resume:
        raise SystemExit(f"{ck} exists; use --resume or move it")
    if (arm_dir / "final").exists():
        raise SystemExit(f"{arm_dir / 'final'} exists: arm already trained")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, _ = FastLanguageModel.from_pretrained(
        BASE, max_seq_length=MAX_LEN, dtype=torch.bfloat16, load_in_8bit=True, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=TARGETS, use_gradient_checkpointing="unsloth", random_state=SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total)

    step = 0
    if ck:
        set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        state = torch.load(ck / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        step = state["step"]
        log(f"resumed from {ck} at step {step}/{total}")

    def save(step):
        tmp = arm_dir / f".tmp-step-{step:04d}"
        dst = arm_dir / f"step-{step:04d}"
        if dst.exists():
            return
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(str(tmp))
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                    "step": step, "arm": args.arm, "total": total}, tmp / "trainer_state.pt")
        os.replace(tmp, dst)
        log(f"checkpoint {dst}")

    arm_dir.mkdir(parents=True, exist_ok=True)
    metrics = (arm_dir / "metrics.jsonl").open("a", encoding="utf-8")
    log(f"arm={args.arm} steps={total} start={step} stop_at={stop_at}")
    model.train()
    while step < total:
        if stop_at is not None and dt.datetime.now() >= stop_at:
            log(f"stop_at {args.stop_at} reached before step {step + 1}/{total}")
            save(step)
            return
        t0 = time.monotonic()
        batch = steps[step]
        n_loss = sum(s["n_loss"] for s in batch)
        loss_sum = 0.0
        for s in batch:
            x = torch.tensor([s["ids"]], device="cuda")
            y = torch.tensor([s["labels"]], device="cuda")
            out = model(input_ids=x, labels=y)  # mean over this sample's loss tokens
            (out.loss * s["n_loss"] / n_loss).backward()
            loss_sum += out.loss.item() * s["n_loss"]
        gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": round(loss_sum / n_loss, 4), "grad_norm": round(gnorm, 4),
               "lr": sched.get_last_lr()[0], "tokens": sum(len(s["ids"]) for s in batch),
               "loss_tokens": n_loss, "sec": round(time.monotonic() - t0, 1)}
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        log(f"step {step}/{total} loss={rec['loss']} gnorm={rec['grad_norm']} lr={rec['lr']:.2e} "
            f"tokens={rec['tokens']} sec={rec['sec']}")
        if step % CKPT_EVERY == 0:
            save(step)
    save(step)
    model.save_pretrained(str(arm_dir / "final"))
    log(f"done: {arm_dir / 'final'}")


if __name__ == "__main__":
    sys.exit(main())
