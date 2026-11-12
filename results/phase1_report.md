# Phase 1 report (baseline 9B)

Written 2026-10-09 (commit `2df4b34`). Only metrics that
`results/plan.md` defines: items per language, accuracy (es-eval scorers,
copied unchanged), and **mean total tokens per question** (prompt + reasoning
+ answer, from the server's `usage`: prompt_tokens + completion_tokens), with
the es/en ratio of the means.

Model as served: `~/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf`
(sha256 `a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496`),
`scripts/serve_base.sh` on :8093 (`results/serve_cmd.base9b.txt`), llama.cpp
b10751-3466812d1. Settings as in `results/plan.md`, phase 1.

## MGSM (250 ids per language)

Raw: `results/mgsm_raw.base9b.jsonl` (sha256
`76fbf9fdac33c461024fdd1adf822e676adeb8b861e72b6508a204b9371b605c`, tracked
since 2026-11-12). Accuracy: `results/score_mgsm.base9b.txt`
(`score_mgsm.py --raw results/mgsm_raw.base9b.jsonl --no-tokenize`). Tokens:
`results/gate_mgsm.base9b.txt` (`scripts/gate_ratio.py`).

| lang | n | accuracy | mean total tokens / question |
|---|---:|---:|---:|
| es | 250 | 0.9240 | 2896.5 |
| en | 250 | 0.9160 | 2578.1 |

**es/en ratio (mean total tokens): 1.1235.**

Comparison with the 27B (es-eval): direct comparison of the models as served
(model + template + quant): 27B IQ3_S 1.34× vs 9B Q8_0 1.1235×. Same items,
metric, runner, scorer and build. Caveat: the quantization differs and can
affect reasoning length; the effect of model size cannot be separated from
the effect of the quant. What matches and what differs, with sha256: NEXT.md,
"Gate, after c)".

*Descriptive, not a plan metric:*

- Items cut at max_tokens 8192: es 2.8 %, en 5.6 % (source:
  `results/score_mgsm.base9b.txt`, column `%length`). Effect on the ratio:
  cuts underestimate the tokens of the language with more cuts. Since en is
  cut more, the measured es/en ratio (1.1235) tends to be HIGHER than the
  real one. If the gate is not reached, the cap cannot explain it. (Direction of
  the bias only; the ratio is not corrected or recomputed.)
- Accuracy: es 0.924 vs en 0.916.

## Belebele (sample of 300 ids per language, `results/belebele_sample_300.json`)

Raw: `results/belebele_raw.base9b.jsonl` (sha256
`8f7ffbae02b32d9110a7592b0a3a0ebfacefa57156d62efdf85529246ce4e81a`, tracked
since 2026-11-12; 600/600 at 2026-10-09 18:23, the 300 sample ids per language, no
duplicate (id, lang)). Accuracy: `results/score_belebele.base9b.txt`
(`score_belebele.py --raw results/belebele_raw.base9b.jsonl --no-tokenize`).
Tokens: `results/token_means.belebele.base9b.txt` (`scripts/token_means.py`,
descriptive, same definition as the MGSM metric).

| lang | n | accuracy | mean total tokens / question |
|---|---:|---:|---:|
| es | 300 | 0.9500 | 2326.4 |
| en | 300 | 0.9800 | 1510.6 |

**es/en ratio (mean total tokens): 1.5401** (descriptive; Belebele is not
part of the gate).

## Gate (MGSM es/en ≥ 1.15)

**Pending c)** (conversion check; results/plan.md, amendment 2026-10-08: the
gate is not applied until c) passes). Possible outcomes, per NEXT.md; none
is chosen here:

- c) passes and ratio ≥ 1.15 → d) (resume test, then arm A).
- c) passes and ratio < 1.15 → nothing is trained; the result is "problem not
  reproduced on this model" (results/plan.md), with the comparison above.
- c) fails → phase-1 results are invalidated and phase 1 is repeated, per the
  rule.
