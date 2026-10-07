# Train plan (phase 2 and 2b control)

Written 2026-10-07, before any training, during phase 1 (no phase-1 scoring
yet). Implements phase 2 / 2b of `results/plan.md`; nothing here changes the
gate or criteria (1)–(5). Not run yet.

## Data

`scripts/build_train_sets.py` (CPU only) → `data/train/es.jsonl` (phase 2)
and `data/train/en.jsonl` (2b control), ids in `results/train_ids.json`,
counts and length stats in `results/train_sets_stats.json`.

- Pool: the first 39,664 global rows of `toroe/ReasonXL-SFT` (Spanish shard
  00000; English rows paired by `row_index`, checked equal with
  `dataset_name` and `ds_uid`). Only items present in **both** languages.
- Length: full chat-formatted sample (Qwen3.5 chat template, Qwen3.5
  tokenizer). A pair is kept only if **both** versions are ≤ 4096 tokens;
  longer ones are **dropped, not truncated**.
- Overlap filter against MGSM en/es, GSM8K train + test, Belebele en/es
  (passage, question, options): exact match of the normalised user turn, or
  any shared normalised word **13-gram** with a benchmark text of the same
  language. A match on either side drops the pair.
- Draw: **2,000** pairs, `random.Random(20261007).sample` over the eligible
  pairs sorted by `row_index`. **The Spanish set and the English control set
  are the same 2,000 items**, each in its own language.

Result (2026-10-07 18:13):

| step | count |
|---|---:|
| aligned pairs in the pool | 39,664 |
| es > 4096 tokens | 11,761 dropped |
| en > 4096 tokens | 12,200 dropped |
| both ≤ 4096 | 26,989 |
| overlap: exact (es / en) | 0 / 0 |
| overlap: 13-gram (es / en) | 0 / 57 |
| pairs dropped for overlap | 57 |
| eligible | 26,932 |
| drawn | 2,000 |

| drawn set | mean | median | p90 | max | total tokens |
|---|---:|---:|---:|---:|---:|
| Spanish (phase 2) | 1,716.9 | 1,728 | 2,783 | 4,092 | 3,433,772 |
| English (2b control) | 1,422.8 | 1,402 | 2,281 | 4,073 | 2,845,662 |

The Spanish versions are **1.207×** the English ones in total tokens (same
items). Source mix of the draw: Nemotron-Cascade Stage-1 general 915,
Stage-2 general 551, Nemotron-Science 162, Llama-Nemotron post-training 162,
Dolci-Think 72, Cascade Stage-1 math 67, instruction-following chat 30,
Cascade Stage-1 science 25, Cascade Stage-1 code 16.

## Training (identical for phase 2 and 2b, except the data language)

- Base: `~/models/Qwen3.5-9B-text` (text-only, no MTP head; `results/plan.md`).
- Unsloth (`FastLanguageModel`), base loaded in **8-bit** (bitsandbytes),
  LoRA adapters in **bf16**: r = 16, alpha = 32, dropout 0, bias none, on the
  7 projections q, k, v, o, gate, up, down. Gradient checkpointing
  ("unsloth").
- Loss: **completion-only**: only the assistant turn's tokens (from after
  `<|im_start|>assistant\n` to `<|im_end|>`, including `<think> … </think>`)
  count; the user turn is masked. The Qwen3.5 template keeps the reasoning
  in the final assistant turn (checked: `<think>…</think>` is rendered).
- 1 epoch over the 2,000 samples; batch size 1, gradient accumulation 8
  (**250 optimizer steps**); no packing; max length 4096 (no sample exceeds
  it).
- Optimizer: AdamW (8-bit Adam not used), lr **1e-4**, **cosine** schedule,
  **3% warmup** (8 steps), weight decay 0, max grad norm 1.0.
- Seed **20261007** (data order, LoRA init, everything seedable).
- Checkpoint every **50** optimizer steps (adapter + optimizer + scheduler +
  RNG state), resumable from the latest; **stop_at 18:45** daily (no new
  step after it; save and exit).
- After training: merge the adapter into the bf16 text-only weights, then the
  same text-only → GGUF → Q8_0 path as the baseline (`results/plan.md`); for
  serving, the base MTP head is added back untrained (plan amendment).

## Estimated time

From the feasibility measurement (option A, 10.2 s per 4096-token step,
`results/feasibility.md`), scaled by each drawn sample's length
(proportional; ignores fixed per-step overhead, so optimistic):
**Spanish ≈ 4.3 s/sample → ≈ 2.4 h** (0.7 of a 3.5 h session);
**English control ≈ 3.5 s/sample → ≈ 2.0 h**. Both fit in one session
each.

## Caveats

- 8-bit base: bitsandbytes' 8-bit matmul casts bf16 inputs to fp16.
- `causal_conv1d` kernel not installed (PyTorch fallback).
- transformers warns that the Qwen3.5 tokenizer has "an incorrect regex
  pattern" (a check aimed at Mistral tokenizers); to be cross-checked against
  llama.cpp's tokenization in the conversion fidelity check (NEXT.md) before
  training.
