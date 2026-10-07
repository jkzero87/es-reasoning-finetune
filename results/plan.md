# Plan: same model, same token cost in Spanish and English

Committed 2026-10-07, before any phase-1 run. The rule is not changed after
seeing results.

**Goal:** a Spanish speaker should not pay more tokens than an English
speaker for the same question, on the same model. **Origin:** es-eval
measured the 27B at 1.34× (en 494, es 660 tokens/question on MGSM); prompt
and translation levers failed.

**Base model:** Qwen/Qwen3.5-9B (official, revision `c202236`), **text-only**:
vision tower and MTP head dropped by `scripts/make_text_only.py` (9.65 B →
8.95 B parameters). Evaluated locally at **GGUF Q8_0**. **Rationale:** newest
official dense model that trains on the 16 GB card with long reasoning traces
(`results/feasibility.md`); closest official architecture to our Qwen3.8-27B
(hybrid linear attention); Q8 removes quantization as a confound.

**Rule:** baseline and fine-tuned models always go through the **identical
path**: text-only checkpoint → GGUF → Q8_0, with the same llama.cpp build.
The baseline GGUF is converted from our text-only checkpoint, not taken from
a downloaded GGUF. The fine-tuned model: merge the LoRA adapter into the bf16
text-only weights, then the same conversion. Commands (llama.cpp
`3466812d1f06728effe7c0f3c0671117f461672d`, converter run with
`.venv-train`):

```
.venv-train/bin/python scripts/make_text_only.py ~/models/Qwen3.5-9B-hf ~/models/Qwen3.5-9B-text
cd ~/llama.cpp && PYTHONPATH=~/llama.cpp/gguf-py ~/es-reasoning-finetune/.venv-train/bin/python \
  convert_hf_to_gguf.py ~/models/Qwen3.5-9B-text --outtype q8_0 \
  --outfile ~/models/Qwen3.5-9B-text-Q8_0/Qwen3.5-9B-text-Q8_0.gguf
```

Baseline GGUF: 9,527,501,280 bytes, sha256
`d69faa4ac81dad5896a8bb69377639578dcb0c160d8096ebee36d28be3911c7c`.

**Metric:** mean total tokens per question (prompt + reasoning + answer),
from the server's `usage` (prompt_tokens + completion_tokens).

## Phase 1 (baseline)

MGSM en + es, 250 ids each, with the 27B baseline settings and scoring
(`scripts/run_mgsm.py` / `scripts/score_mgsm.py`, copied unchanged from
es-eval): **thinking on** (server default; llama-server started with
`--reasoning-effort medium` as for the 27B, no `chat_template_kwargs`),
**temperature 1.0**, top_p 0.95, top_k 20, seed 42, max_tokens 8192, no
system prompt. These are kept even where they differ from Qwen's recommended
sampling for Qwen3.5, for comparability with the 27B measurement. Also
Belebele en + es on a fixed stratified sample of **300 ids**
(`results/belebele_sample_300.json`: stratified by gold answer letter, seed
20261007, 488-id population; committed before the run), same settings and
the es-eval Belebele prompt and scorer. Server: `scripts/serve_base.sh`
(:8093). Order: MGSM es, MGSM en, Belebele es, Belebele en.

**GATE:** train only if the MGSM es/en ratio ≥ 1.15. Otherwise stop and
publish "problem not reproduced on this model".

## Phase 2 (train)

**Data:** the Spanish split of `toroe/ReasonXL-SFT`. Before training, check
the data for MGSM/GSM8K and Belebele overlap (exact + near-duplicate question
text) and drop matches; report counts.

**Training:** Unsloth, **8-bit base + bf16 LoRA** (r = 16, the 7 projections
q, k, v, o, gate, up, down), batch 1 + gradient accumulation, gradient
checkpointing, on the local 16 GB card. Checkpoints resumable; stop_at 18:45
daily.

**Max length 4096 tokens:** samples longer than that are **dropped, not
truncated** (report how many). Rationale: the training traces (median 2,239
tokens) are longer than the model's own outputs and the model imitates
length; 4096 also halves step time versus 8192 and leaves 3.1 GB headroom
(`results/feasibility.md`). In the length sample, 71.2% of Spanish samples
are ≤ 4096.

Exact hyperparameters, sample count and seed go in `results/train_plan.md`,
committed before training.

## Phase 2b (control)

Identical run (same base, steps, sample count, length distribution as close
as possible, hyperparameters, seed), but on the **English** split of
ReasonXL-SFT, with the same overlap filtering and the same 4096-token cap
(longer samples dropped). Run only if the Spanish model passes (1)–(4).

## Phase 3 (falsify)

The Spanish-trained model **PASSES only if ALL hold:**

1. MGSM: es total ≤ 1.10 × trained en total;
2. MGSM: trained en total ≤ 1.10 × baseline en total;
3. MGSM: es and en accuracy not significantly below the phase-1 baseline
   (exact McNemar; p ≤ 0.05 = fail);
4. criteria (1)–(3) also hold on the Belebele sample;
5. it beats the control on es/en ratio by ≥ 0.05 absolute on BOTH MGSM and
   Belebele.

If the control also meets (1)–(4), the README must say "training in general
shortened output", not "Spanish data fixed Spanish". % of reasoning in
Spanish is reported but does not decide.

## Caveats

- 8-bit is not pure bf16: bitsandbytes' 8-bit matmul casts bf16 inputs to
  fp16 during quantization.
- Unsloth's Qwen3.5 guidance recommends against 4-bit (QLoRA) training on
  Qwen3.5 and recommends bf16 LoRA (22 GB for the 9B, more than this card);
  it says nothing about 8-bit.
- The `causal_conv1d` kernel is not installed (no `nvcc`); training uses
  transformers' PyTorch fallback for it.
