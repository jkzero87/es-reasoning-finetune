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
  convert_hf_to_gguf.py ~/models/Qwen3.5-9B-text --no-mtp --outtype q8_0 \
  --outfile ~/models/Qwen3.5-9B-text-Q8_0/Qwen3.5-9B-text-Q8_0.gguf
```

`--no-mtp` is required: the text-only config keeps `mtp_num_hidden_layers: 1`,
and without the flag the converter declares an extra (MTP) block whose
tensors were dropped, so llama.cpp refuses to load the file (first attempt,
17:42, before any run). Baseline GGUF: 9,527,501,248 bytes, sha256
`a3e9970cce275f548079a839eb58731f638788693f67be34a2c13ef635906edb`.

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

## Amendment 2026-10-07 18:00 (before any scoring)

**Server config for ALL phase-1 and phase-3 runs (baseline, fine-tuned,
control):** `scripts/serve_base.sh` with `--parallel N`, runners with
`--workers N`. Changed after 4 serial records (a 5th was in flight when the
run was stopped and is lost), for throughput only, before any scoring; those
records are kept in `results/discarded_serial/` and not scored. N = 4,
`-c 36864` (9216 tokens per slot: prompt + 8192-token completion), idle VRAM
10.3 GB.

**Served GGUF with the MTP head.** The served baseline is the text-only
checkpoint **with** the official MTP tensors kept (vision still dropped),
same conversion path otherwise:

```
.venv-train/bin/python scripts/make_text_only.py ~/models/Qwen3.5-9B-hf ~/models/Qwen3.5-9B-text-mtp --keep-mtp
cd ~/llama.cpp && PYTHONPATH=~/llama.cpp/gguf-py ~/es-reasoning-finetune/.venv-train/bin/python \
  convert_hf_to_gguf.py ~/models/Qwen3.5-9B-text-mtp --outtype q8_0 \
  --outfile ~/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf
```

(9,786,060,160 bytes, sha256
`a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496`), served
with the 27B's MTP flags (`--spec-type draft-mtp --spec-draft-n-max 3
-ctkd q8_0 -ctvd q8_0`). The training checkpoint stays the one without MTP
(`--no-mtp` GGUF above is no longer served).

MTP speculative decoding is used for speed only; it verifies every drafted
token, so outputs and token counts are unaffected. The fine-tuned model keeps
the base MTP head untrained; acceptance may drop, which affects speed only.
Precision: at temperature 1.0 verification preserves the model's output
distribution (and so expected token counts), but an individual sampled
output is not identical to a non-MTP run with the same seed (MGSM es id 2:
626 tokens with MTP vs 744 without in the throughput probe). The 27B baseline
in es-eval was also served with MTP.

Measured 17:54–17:58 (MGSM es ids 1–4, 4 concurrent requests capped at 1024
tokens; single stream uncapped id 1):

| config | aggregate tok/s | draft acceptance |
|---|---:|---:|
| `--parallel 1` + MTP, single stream | 90.3 (id 1, correct, 1,460 tokens) | 76% |
| `--parallel 4` + MTP | **168.6** | 64–77% per request |
| `--parallel 4`, no MTP | 114.8 | — |
| (`--parallel 1`, no MTP, serial run) | ≈ 45 | — |

`--parallel 4` + MTP is the faster combination and is the config used.

## Note 2026-10-07 18:1x

Checked the cap's first reason against phase-1 data. ReasonXL Spanish samples
(mean 4,452 / median 2,239 tokens, prompt included) are longer than the 9B's
own MGSM es completions so far (mean 2,753 / median 1,810, completion only),
consistent with that reason. The cap stands for all three reasons. The rule
is unchanged.

## Amendment 2026-10-07 18:2x (before any training and before any phase-1 scoring)

**Three training arms**, all on the same 2,000 parallel ReasonXL-SFT items,
same hyperparameters, same seed (`results/train_plan.md`):

- **A (primary):** user turn **Spanish**, `<think>` = the **English**
  version's reasoning, final answer = the **Spanish** version's final answer.
  Rationale: the measured cost is doubt while reasoning, not writing Spanish;
  ReasonXL Spanish reasoning is 1.21× English for the same items, so Spanish
  reasoning sets a floor above criterion (1).
- **B:** all Spanish.
- **Control:** all English (was phase 2b).

Phase 3 criteria (1)–(4) apply to A and B separately; criterion (5) compares
each against the control. Order: A, then control, then B. Report % of
reasoning in English for A (expected high) and in Spanish for B; reported,
not decisive.

**Draw, stratified by domain:** 40% math (800 items), the rest proportional to
the eligible pool's other domains (categories and counts in
`results/train_plan.md`). Same overlap filter. The ≤ 4096-token cap applies
to **every arm's version** of each item (A's mixed sample must also fit). If
fewer than 800 math items are eligible, take all and report.

**Arm A split point:** every item's Spanish and English versions must contain
exactly one `</think>`; items that don't are dropped and counted.

## Amendment 2026-10-07 18:2x (before any scoring)

Phase-1 MGSM order changed to interleaved at 18:2x for earlier paired
visibility; per-request settings unchanged; done before any scoring. The
remaining items run as: first the English side of ids already done in
Spanish, then id by id (es id k, then en id k) (`run_baseline.py --order
interleave`). Applying it required restarting the runner process (server
untouched); requests in flight at the restart were dropped and re-run, so no
record was written twice or skipped. Belebele order unchanged (es, then en).
The gate (≥ 1.15) is applied only on all 250 MGSM pairs; any earlier paired
look is informal and labeled as such.
