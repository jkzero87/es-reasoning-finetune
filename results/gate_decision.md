# Gate decision (2026-11-12)

Applied exactly as written in `results/plan.md` ("**GATE:** train only if the MGSM es/en ratio ≥ 1.15. Otherwise stop and publish "problem not reproduced on this model"."), after the conversion check passed (amendment 2026-10-08).

- Conversion check: **PASS** (`results/conversion_check.md`, commit `7b40d9d`): link 1 top-1 98.587% over 920 positions, link 2 same-top 98.81 ± 0.484%; fail threshold ≤ 95% on either link.
- Gate metric, MGSM es/en mean total tokens, all 250 pairs: **1.1235** (`results/gate_mgsm.base9b.txt`, `scripts/gate_ratio.py`; re-run 2026-11-12, identical output). Threshold: ≥ 1.15.
- 1.1235 < 1.15 → **the gate is not met. Nothing is trained. Result for MGSM: "problem not reproduced on this model".**

Comparison with the 27B, with the label decided on 2026-10-09 (NEXT.md):

> Direct comparison of the models as served (model + template + quant): 27B IQ3_S 1.34× vs 9B Q8_0 1.1235×. Same items, metric, runner, scorer and build. Caveat: the quantization differs and can affect reasoning length; the effect of model size cannot be separated from the effect of the quant.

What matches and what differs between the two measurements, with sha256: NEXT.md, "Gate, after c)".

Descriptive, not part of the gate: Belebele (same 300 ids) es/en mean total tokens on the 9B is **1.5401** (`results/token_means.belebele.base9b.txt`). The gate is defined on MGSM only, so this does not change the decision; it is reported as an observation that the Spanish cost on this model depends on the benchmark.

Consequences: phase 2 (training arms A, control, B) and phase 3 are not run. The training scripts, data (`data/train/`, untracked) and the rebuilt training base stay as they are. The descriptive queue of amendment 2026-11-12 (zh, reasoning off, ProX-Lite) is independent of the gate.
