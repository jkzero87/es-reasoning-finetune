# Train plan (phase 2 and 2b control)

Written 2026-10-07, before any training, during phase 1 (no phase-1 scoring
yet). Implements phase 2 / 2b of `results/plan.md`; nothing here changes the
gate or criteria (1)–(5). Not run yet.

## Data

> **Superseded por enmienda del 2026-10-08** (this whole Data section,
> including the 2026-10-07 18:13 result; see the end of this file).

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

> **Superseded por enmienda del 2026-10-08 (hyperparameters)**: values
> below are replaced by the table in "Amendment 2026-10-08 (hyperparameters)"
> at the end of this file (alpha and weight decay change).

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

## Amendment 2026-10-08 (before building the arm data; no training, no phase-1 scoring yet)

Supersedes the **Data** section above (and its 2026-10-07 18:13 result,
which is kept in git history; the es/en files it wrote are deleted). Implements
the 2026-10-07 18:2x three-arm amendment of `results/plan.md`. The
**Training** section applies unchanged to each arm (A, B, control) separately,
except that the data file is `data/train/{A,B,control}.jsonl`. The time
estimates above were for es/en and are not updated here.

**Script:** `scripts/build_train_sets.py` as committed with this amendment
(CPU only, no network: the pool is read from the cache below). Outputs
`data/train/A.jsonl`, `data/train/B.jsonl`, `data/train/control.jsonl`
(gitignored), `results/train_ids.json` (row_index + domain, in draw order),
`results/train_sets_stats.json` (counts at each step).

**Arms**, all on the **same 2,000 parallel items** of `toroe/ReasonXL-SFT`:

- **A (primary):** the Spanish version's messages, except the final
  assistant turn = the **English** version's text up to and including
  `</think>` + the **Spanish** version's text after `</think>`.
- **B:** the Spanish version, unchanged.
- **Control:** the English version, unchanged.

**Pool:** the first 39,664 global rows (Spanish shard 00000), English rows
paired by `row_index` (checked equal together with `dataset_name` and
`ds_uid`; mismatch = abort). Cached as `data/train/pool_{es,en}.jsonl`.

**Filters, in this order; each drops the whole item (all three arms):**

1. **Split point:** the Spanish and the English version must each contain
   exactly one `</think>` over all messages, located in the last message,
   which must be the assistant's. Otherwise dropped and counted.
2. **Length cap 4096 tokens per arm:** every arm's version (A, B, control),
   as the full chat-formatted sample (Qwen3.5 chat template, Qwen3.5
   tokenizer from `~/models/Qwen3.5-9B-text`, no extra special tokens), must
   be ≤ 4096 tokens. Longer = **dropped, not truncated**; counted per arm.
3. **Overlap** (unchanged from 2026-10-07): normalised (NFKC, lowercase,
   non-alphanumerics → space, whitespace collapsed) user turn exactly equal
   to a normalised benchmark question, or any shared normalised word
   **13-gram** with a benchmark text in the same language. Benchmarks: MGSM
   en/es (250 each), GSM8K train + test (`openai/gsm8k`, main), Belebele
   en/es (488 each: passage, question, options). A match in either language
   drops the item.

**Domain (preregistered definition of math):** an item is **"math"** if the
substring `math` appears, case-insensitive, in its `dataset_name` or its
`source` (Spanish row's fields; the code tests
`"math" in (dataset_name + " " + source).lower()`, `source` None → ""). It is
a substring test (e.g. "mathematics" also matches). Otherwise the item's
domain is its `dataset_name`.

Pool counts by domain, **before** any filter (39,664 items; computed from
`data/train/pool_es.jsonl` on 2026-10-08):

| domain | pool items |
|---|---:|
| Nemotron-Cascade-SFT-Stage-1-general | 12,910 |
| Dolci-Think-SFT-7B (non-math) | 8,628 |
| Nemotron-Cascade-SFT-Stage-2-general | 7,359 |
| **math** | **4,060** |
| Nemotron-Science-v1 | 2,490 |
| Llama-Nemotron-Post-Training-Dataset | 2,174 |
| Nemotron-Cascade-SFT-Stage-1-code | 866 |
| Nemotron-Cascade-SFT-Stage-1-science | 757 |
| Nemotron-Instruction-Following-Chat-v1 | 420 |

math = Nemotron-Cascade-SFT-Stage-1-math 2,976 + Nemotron-Cascade-SFT-Stage-2-math
669 + Dolci-Think-SFT-7B items whose `source` contains "math" 415.

**Draw:** N = **2,000** items. Seed **20261007**: one
`random.Random(20261007)` instance, used in this order: for each domain in
sorted name order, `rng.sample(items of that domain sorted by row_index,
quota)`; then `rng.shuffle` of the concatenated draw.

- math quota = **min(800, eligible math items)**;
- the remaining 2,000 − math quota split across the other domains in
  proportion to their **eligible** counts (after filters 1–3), by the
  **largest-remainder** method (floor of each share, then +1 to the domains
  with the largest fractional parts until the total is reached).

The exact per-domain quotas depend on the eligible counts after filters 1–3,
which are only known by running the script; they are a deterministic
function of the rule above and are reported, as numbers, in
`results/train_sets_stats.json` (`eligible_by_domain`, `drawn_by_domain`)
and in the results commit. **If eligible math < 800 or the draw is not 2,000
items in each arm, stop and report; nothing is adjusted.**

## Amendment 2026-10-08 (hyperparameters; before any training)

> **Superseded por enmienda del 2026-10-08 ("alpha, weight decay, loss")** for
> LoRA alpha, weight decay and the loss-over-a-step row; see the end of this file.

Supersedes the hyperparameter values of the **Training** section above.
Identical for the three arms (A, B, control); implemented as constants in
`scripts/train_lora.py`, committed with this amendment. Rule applied: take
the value from `scripts/feasibility_lora.py` (option A run of
`results/feasibility.md`) where feasibility set it; anything feasibility did
not exercise is marked **NOT TESTED** and keeps the 2026-10-07 value.
Feasibility was 5 optimizer steps measuring memory and speed only: no value
below has been validated for training quality.

| hyperparameter | value | source |
|---|---|---|
| base | `~/models/Qwen3.5-9B-text`, 8-bit bitsandbytes via Unsloth, bf16 compute | feasibility A |
| LoRA r | 16 | feasibility A |
| LoRA alpha | **16** (was 32 on 2026-10-07) | feasibility A (alpha 16) |
| LoRA dropout / bias | 0 / none | feasibility A |
| LoRA targets | q, k, v, o, gate, up, down projections (29.1 M trainable) | feasibility A |
| max length | 4096 (no sample exceeds it; longer were dropped at build) | feasibility A (13,169 MiB peak) |
| batch size | 1 | feasibility A |
| gradient checkpointing | Unsloth's | feasibility A |
| optimizer | torch AdamW, betas/eps default | feasibility A |
| learning rate | 1e-4 | feasibility A (5 steps; not validated for quality) |
| weight decay | **0.01** (torch AdamW default, implicit in feasibility; was 0 on 2026-10-07) | feasibility A |
| gradient accumulation | 8 → **250 optimizer steps** per arm | NOT TESTED (2026-10-07) |
| epochs | 1 | NOT TESTED (2026-10-07) |
| LR schedule | cosine, warmup 3% = 8 steps | NOT TESTED (2026-10-07) |
| max grad norm | 1.0 (feasibility had no clipping) | NOT TESTED (2026-10-07) |
| seed | **20261007** (data order = build draw order; LoRA init `random_state`; torch/numpy/python RNG) | NOT TESTED (2026-10-07; feasibility used Unsloth's default 3407) |
| loss | completion-only: final assistant turn from after `<|im_start|>assistant\n` through `<|im_end|>` | NOT TESTED (2026-10-07) |
| loss over an accumulated step | token-weighted: sum of per-token losses of the 8 samples / their loss tokens | NOT TESTED; implementation choice of 2026-10-08 (no earlier value) |
| checkpoints | every 50 optimizer steps + at stop_at; resumable | 2026-10-07 |
| stop_at | 18:45 daily (no new optimizer step after it; save and exit) | 2026-10-07 |

Target-module note: in Qwen3.5's linear-attention (Gated DeltaNet) layers the
projections have other names (fused qkv, gate, out), so "q, k, v, o" adapt
only the full-attention layers; gate/up/down adapt every layer's MLP. Same
as feasibility (29.1 M trainable parameters).

Dry run (`train_lora.py --dry-run`, 2026-10-08; tokens per optimizer step =
8 samples):

| arm | samples | optimizer steps | tokens/step mean (min–max) | total tokens | loss tokens |
|---|---:|---:|---:|---:|---:|
| A | 2,000 | 250 | 15,158 (8,401–22,010) | 3,789,393 | 3,571,463 (94.2%) |
| B | 2,000 | 250 | 16,279 (8,933–23,505) | 4,069,811 | 3,851,881 (94.6%) |
| control | 2,000 | 250 | 14,281 (7,986–21,567) | 3,570,319 | 3,379,937 (94.7%) |

Estimated time, from feasibility A at L = 4096 (10.2 s per 4096-token
optimizer step of batch 1, warm-up included), scaled linearly by tokens
(10.2 / 4096 = 2.49 ms per token): **A ≈ 2.6 h, B ≈ 2.8 h, control ≈ 2.5 h**
(7.9 h in all). Linear scaling ignores per-sample overhead (optimistic for
short samples) and includes warm-up (pessimistic); the 8 backward passes per
optimizer step are already counted as 8 samples.

Post-training tooling committed with this amendment (written, not run):
`scripts/merge_to_gguf.sh` (+ `merge_lora.py`, `gguf_compare.py`) and
`scripts/conversion_check.py` (+ `llama_logits.c`).

## Amendment 2026-10-08 (alpha, weight decay, loss; before any training)

Supersedes three rows of the hyperparameter table above. Same for A, B,
control; `scripts/train_lora.py` constants changed to match in the same
commit (`LORA_ALPHA = 32`, `WEIGHT_DECAY = 0.0`).

- **LoRA alpha = 32** and **weight decay = 0**, the 2026-10-07 values.
  Reason: feasibility (`results/feasibility.md`) measured only peak memory
  and speed, and neither alpha (a scalar on the adapter output) nor weight
  decay (an optimizer update term) changes either, so feasibility gives no
  reason to replace the values preregistered on 2026-10-07.
- **Loss within an optimizer step (rule):** mean per token over the 8
  samples of the step: every loss token of the step weighs the same (sum of
  per-token cross-entropies over the 8 samples / their total loss-token
  count), not a mean of per-sample means.
- **LoRA targets unchanged, and linear attention has no LoRA:** in the
  Gated DeltaNet (linear-attention) layers no module is named q/k/v/o_proj,
  so those layers get LoRA only on their MLP (gate/up/down); q/k/v/o adapt
  the full-attention layers. Same as feasibility. Trainable parameters:
  **29,097,984 (29.1 M)** with alpha 32, identical to alpha 16 (alpha does
  not change the count). Counted 2026-10-08 with peft on the text-only
  config, meta device (no weights, no GPU): 29,097,984 for both alpha 16 and
  32, and 0 target modules inside `linear_attn`.
