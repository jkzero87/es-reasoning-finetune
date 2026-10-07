#!/usr/bin/env python3
"""Feasibility: Unsloth LoRA on Qwen3.5 fits the 16 GB card at length L?

One process per length (clean CUDA state):
  .venv-train/bin/python scripts/feasibility_lora.py --seq-len 8192 \
      [--base DIR] [--quant bf16|8bit|4bit] [--backend unsloth|hf]

--quant: base weights in bf16, bitsandbytes 8-bit (LLM.int8) or 4-bit NF4
(bf16 compute, double quant); the LoRA adapters are bf16 in all cases.
--backend hf: plain transformers + peft + bitsandbytes (used if Unsloth
cannot load the requested quant).

- Gradient checkpointing on (Unsloth's, or HF's with --backend hf).
- LoRA r=16, alpha=16, dropout 0, on Unsloth's default projection modules
  (feasibility only; the real hyperparameters go in results/train_plan.md).
- Batch 1, 5 optimizer steps (AdamW, lr 1e-4) on real ReasonXL-SFT Spanish
  text: rows of one row group rendered with the chat template, concatenated
  and cut to exactly L tokens per step (every step is a full-length sequence).
- Peak VRAM: torch max_memory_reserved, and the device-wide peak from
  nvidia-smi polled every 0.2 s (includes the CUDA context). Headroom =
  total - nvidia-smi peak.
Prints one JSON line with the result (also on OOM).
"""
import argparse
import json
import subprocess
import threading
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

BASE = "/home/jkzero/models/Qwen3.5-4B-hf"
SHARD = "datasets/toroe/ReasonXL-SFT/data/spanish-00000-of-00064.parquet"


class SmiPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.total = 0
        self.stop = False

    def run(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
            self.peak, self.total = max(self.peak, used), total
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--quant", choices=["bf16", "8bit", "4bit"], default="bf16")
    ap.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    args = ap.parse_args()
    L = args.seq_len
    smi = SmiPeak()
    smi.start()
    res = {"seq_len": L, "steps": args.steps, "base": args.base, "quant": args.quant,
           "backend": args.backend}
    try:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if args.backend == "unsloth":
            from unsloth import FastLanguageModel
            import torch

            model, tok = FastLanguageModel.from_pretrained(
                args.base, max_seq_length=L, dtype=torch.bfloat16,
                load_in_4bit=args.quant == "4bit", load_in_8bit=args.quant == "8bit")
            tok = getattr(tok, "tokenizer", tok)
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=16, lora_dropout=0, bias="none", target_modules=targets,
                use_gradient_checkpointing="unsloth", random_state=3407)
        else:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            q = None
            if args.quant == "8bit":
                q = BitsAndBytesConfig(load_in_8bit=True)
            elif args.quant == "4bit":
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                       bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.base, dtype=torch.bfloat16, quantization_config=q, device_map="cuda")
            tok = AutoTokenizer.from_pretrained(args.base)
            if q is not None:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            else:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0,
                                                     bias="none", target_modules=targets,
                                                     task_type="CAUSAL_LM"))
        res["params_total"] = sum(p.numel() for p in model.parameters())
        res["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        pf = pq.ParquetFile(HfFileSystem().open(SHARD))
        rows = pf.read_row_group(0, columns=["messages"]).to_pylist()
        ids = []
        for r in rows:
            ids += tok(tok.apply_chat_template(r["messages"], tokenize=False),
                       add_special_tokens=False)["input_ids"]
            if len(ids) >= L * args.steps:
                break
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        losses = []
        for s in range(args.steps):
            x = torch.tensor([ids[s * L:(s + 1) * L]], device="cuda")
            out = model(input_ids=x, labels=x)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(out.loss.item(), 3))
        torch.cuda.synchronize()
        res.update(ok=True, losses=losses, sec_per_step=round((time.monotonic() - t0) / args.steps, 1),
                   torch_peak_reserved_mib=torch.cuda.max_memory_reserved() // 2**20,
                   torch_peak_alloc_mib=torch.cuda.max_memory_allocated() // 2**20)
    except Exception as e:  # OOM or anything else: report, don't hide
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    time.sleep(0.5)
    smi.stop = True
    res.update(smi_peak_mib=smi.peak, gpu_total_mib=smi.total, headroom_mib=smi.total - smi.peak)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
