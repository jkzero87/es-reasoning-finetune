# Feasibility: training Qwen3.5 locally on the 16 GB card

Measured 2026-10-07, 17:08–17:37, on an RTX 5060 Ti 16 GB (16,311 MiB),
driver 595.91.07. Training env `.venv-train` (Python 3.12): torch 2.14.1
(CUDA 13.0), unsloth 2026.10.2, transformers 5.17.0, peft 0.21.2,
bitsandbytes 0.50.2. No paid compute.

## Setup

`scripts/feasibility_lora.py`, one process per length (clean CUDA state):
batch 1, gradient checkpointing (Unsloth's), LoRA r=16, alpha 16, dropout 0
on the 7 projections (q, k, v, o, gate, up, down), 5 AdamW steps (lr 1e-4).
Text: real ReasonXL-SFT Spanish rows (shard `spanish-00000`, row group 0),
rendered with the chat template, concatenated and cut so every step is a full
sequence of exactly L tokens. Peak VRAM is the device-wide peak from
`nvidia-smi` polled every 0.2 s (includes the CUDA context); headroom =
16,311 MiB − that peak. s/step is the mean over the 5 steps (includes
warm-up, so it overstates steady state).

Text-only checkpoints come from `scripts/make_text_only.py` (vision tower
`model.visual.*` and MTP head `mtp.*` dropped, `model.language_model.*` →
`model.*`, tensors copied unchanged, config = the official `text_config` as
`Qwen3_5ForCausalLM`):

| | official params | vision | MTP | text-only |
|---|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B (rev. `c202236`) | 9,653,104,368 | 456,010,480 | 243,290,624 | **8,953,803,264** |
| Qwen/Qwen3.5-4B (rev. `851bf6e`) | 4,659,865,088 | 333,514,240 | 120,599,552 | **4,205,751,296** |

The 4B text-only model answered a sanity prompt correctly ("¿Cuánto es 17
por 3?" → 51).

## Results

Order tried (stop at the first that fits at ≥ 4096 tokens with ≥ 1 GB
headroom): A, then B, then C. A fits, so B was not run. C had already been
measured before the 9B was considered.

| option | L | peak VRAM (MiB) | headroom (MiB) | s/step | fits |
|---|---:|---:|---:|---:|---|
| **A: 9B text-only, 8-bit base + bf16 LoRA (Unsloth)** | 8192 | 14,771 | 1,540 | 24.4 | yes |
| | 6144 | 13,955 | 2,356 | 20.4 | yes |
| | 4096 | 13,169 | 3,142 | 10.2 | yes |
| B: 9B, 4-bit NF4 base + bf16 LoRA | — | not run (A fits) | | | |
| C: 4B (full official checkpoint), bf16 LoRA (Unsloth) | 8192 | 12,567 | 3,744 | 29.9 | yes |
| | 6144 | 11,847 | 4,464 | 18.7 | yes |
| | 4096 | 11,145 | 5,166 | 8.4 | yes |

Trainable LoRA parameters: 29.1 M (A), 21.2 M (C).

## ReasonXL-SFT Spanish lengths

`scripts/reasonxl_lengths.py`: 53,263 rows (one row group from each of 8
evenly spaced shards of the 64 Spanish shards; per-row data in
`results/reasonxl_spanish_lengths.csv.gz`), tokenizer and chat template of Qwen3.5-4B
(byte-identical `tokenizer.json` and `chat_template.jinja` to the 9B's). Median 2,239 tokens, p90 13,925, p99
26,035, max 32,544.

| max length | share ≤ L | dropped |
|---:|---:|---:|
| 4096 | 71.2% | 28.8% |
| 6144 | 81.8% | 18.2% |
| 8192 | 89.6% | 10.4% |

## Estimated training time, 2,000 samples, one epoch

Per-sample time interpolated from the measured s/step over each kept
sample's length (linear between the measured points, proportional below
4096), averaged over the length sample. Rough in both directions: no fixed
per-step overhead for short samples (optimistic), warm-up included in s/step
(pessimistic).

| option | L | s/sample | 2,000 samples | 3.5 h sessions |
|---|---:|---:|---:|---:|
| A (9B, 8-bit) | 8192 | 7.3 | 4.0 h | 1.2 |
| | 6144 | 5.9 | 3.3 h | 0.9 |
| | 4096 | 4.6 | 2.5 h | 0.7 |
| C (4B, bf16) | 8192 | 6.5 | 3.6 h | 1.0 |
| | 6144 | 5.0 | 2.8 h | 0.8 |
| | 4096 | 3.8 | 2.1 h | 0.6 |

## Caveats

- **8-bit is not pure bf16.** bitsandbytes' 8-bit matmul (LLM.int8) casts
  the bf16 inputs to fp16 during quantization (logged as a warning on every
  call: "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16").
- **Unsloth's guidance:** its Qwen3.5 fine-tuning page recommends against
  QLoRA (4-bit) on Qwen3.5 models, dense or MoE, and recommends bf16 LoRA
  (listed at 22 GB for the 9B). It says nothing about 8-bit.
- **Missing `causal_conv1d` kernel:** not installed (no `nvcc` on this
  machine to build it), so the short convolution of the linear-attention
  layers runs on transformers' PyTorch fallback in every run above.
- **C ran before `flash-linear-attention` was installed** (0.5.2, installed
  at ~17:17; A ran after). Unsloth patches the gated-delta-rule path itself,
  so the effect is uncertain, but C's timings may be pessimistic relative to
  A's; at 8192 the larger 9B was faster per step than the 4B.
- C used the full official 4B checkpoint (with vision tower and MTP), not
  the text-only one.
- Feasibility only: 5 steps, r=16. The real hyperparameters go in
  `results/train_plan.md`.

## Precedent

A GGUF re-quant card of `CrowdMind/PrimeMind-9B` (by `sizzlebop`, Sept 2026;
the original repo is no longer public) describes it as LoRA SFT on
Qwen/Qwen3.5-9B, rank 64 (alpha 128), 1,200 samples (600 text + 600
multimodal), published as bitsandbytes NF4 4-bit weights. The card gives no
hardware or training time.
