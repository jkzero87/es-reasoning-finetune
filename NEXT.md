# NEXT

## Status 2026-10-09 18:33 (end of day)

Done today (plan for 2026-10-09 below):
- a) Belebele resumed 17:04 (2026-10-08's run had written nothing: the PC
  shut down at 16:50:44 with 4 requests in flight). **Phase 1 complete:**
  MGSM 500/500, Belebele 600/600 (18:23:42), no duplicate (id, lang).
- b) Phase-1 commit `2df4b34`. Raw files not committed; sha256 in
  `results/phase1_report.md`: `mgsm_raw.base9b.jsonl` 76fbf9fd…,
  `belebele_raw.base9b.jsonl` 8f7ffbae…. Gate ratio 1.1235
  (`results/gate_mgsm.base9b.txt`) **not applied** (needs c)).
- :8093 stopped 18:24 (PID 18986). Nothing of this project is running.
- c0) `llama-perplexity` rebuilt at 3466812d1 (18:24–18:32). Both
  `llama-server --version` and `llama-perplexity --version` show
  `build 10751, commit 3466812d1`.
  **Open point for Juan, before c):** `cmake --target llama-perplexity` also
  recompiled ggml-cuda (the .cu sources were newer than this build dir's
  objects) and relinked the shared libs. All relinked libs are byte-identical
  to the copies taken at 18:30, EXCEPT `libggml-cuda.so.0.22.0`
  (bf382e74… → 8b541eb1…, same source rev, same nvcc 13.1.115). The phase-1
  copy is kept in `~/llama-build-3466812d1-phase1-bak/` (SHA256SUMS there);
  `libggml-base` / `libggml-cpu` had already been relinked at 18:24 before
  the backup, so their phase-1 bytes were not kept (the relinked ones match
  the backup). c)'s Q8_0 GPU side would run on the rebuilt libggml-cuda,
  not byte-for-byte the one that served phase 1. Decide: run c) as is, or
  with `LD_LIBRARY_PATH=~/llama-build-3466812d1-phase1-bak` for the Q8_0 side.
- c1) guard in `scripts/conversion_check.py`, commit `ee536ad`: aborts unless
  `llama-perplexity --version` shows 3466812d1 (checked: real binary passes;
  segfault and stale build abort).

## Tomorrow (2026-10-10), FIRST: c) conversion check

Nothing else before it (not d), not e)). :8093 must stay down; never touch
:8092. Decide the libggml-cuda point above first. Then:
```
cd ~/es-reasoning-finetune && ~/llama.cpp/build/bin/llama-perplexity --version   # must show commit 3466812d1
.venv-train/bin/python scripts/conversion_check.py
```
(HF side on CPU, ~18 GB RAM; 48 GB available on 2026-10-09; creates and
deletes a ~18 GB BF16 GGUF, 79 GB free.) Commit
`results/conversion_check.{md,json}` with the verdict, then apply the gate
with `results/gate_mgsm.base9b.txt` as in "Gate, after c)" below.

## Tomorrow, FIRST job (before resuming phase 1): conversion fidelity check

> **Superseded por enmienda del 2026-10-08** (target GGUF and timing; see
> `results/plan.md`, amendment 2026-10-08). Method below otherwise stands.

Check that our Q8_0 GGUF (`~/models/Qwen3.5-9B-text-Q8_0/Qwen3.5-9B-text-Q8_0.gguf`,
sha256 `a3e9970c…`) is a faithful conversion of our text-only HF checkpoint
(`~/models/Qwen3.5-9B-text`, bf16, transformers). Write `results/conversion_check.md`.

- **Prompts:** 10 fixed prompts, fixed before looking at any output: MGSM es
  ids 1–5 and MGSM en ids 1–5, each as the phase-1 user message (prompt +
  question), rendered with the chat template plus generation prompt, the same
  token ids fed to both sides. These ids have no special role (not used for
  any tuning or selection).
- **Next-token predictions:** for each prompt, at the first 256 prompt
  positions (all positions if the prompt is shorter):
  - top-1 agreement (HF argmax == GGUF argmax), pooled over all positions;
  - mean KL divergence KL(HF ‖ GGUF) of the next-token distributions.
- **Greedy decode:** 128 tokens from each side (temperature 0, same prompt,
  thinking as in phase 1), and report the first position where the two token
  sequences diverge (or "no divergence in 128").
- **Two links, reported separately** (the comparison above is split into
  these two; the greedy decode runs on the HF checkpoint and the Q8_0):
  1. **HF → GGUF mapping:** transformers bf16 text-only checkpoint vs a
     **BF16 GGUF** from the same conversion (same command as in
     `results/plan.md`, `--outtype bf16` instead of `q8_0`; CPU offload is
     fine), same 10 prompts: top-1 agreement + mean KL. This is the link that
     needs libllama / the bindings (see method note).
  2. **BF16 GGUF → Q8_0:** llama.cpp's own `llama-perplexity`, no custom
     code: `--kl-divergence-base <file>` on the BF16 GGUF, then
     `--kl-divergence` on the Q8_0, over the text of the same 10 prompts.
     Report its KL and "same top" figures. Caveat: llama-perplexity splits
     the text into chunks of `-c` tokens and scores only the second half of
     each chunk; the 10 prompts are ~1k tokens in all, so use a small `-c`
     (e.g. 128) and report how many positions were scored.
  Delete the BF16 GGUF after the check (regenerable).
- **Expected for a faithful Q8_0:** top-1 agreement well above 95%. If it is
  lower, **stop and report before resuming phase 1.**
- Method note (link 1): llama-server returns probabilities only for generated
  tokens, so per-position GGUF logits over the prompt need llama.cpp's own logits
  path (e.g. a small program on libllama with logits for all positions, or
  the llama-cpp-python bindings built against this llama.cpp,
  `3466812d1`). Record which one was used. Run on the GPU with the :8093
  server stopped (both models do not fit at once).

## Then: resume phase 1

> **Superseded por enmienda del 2026-10-08**: phase 1 was resumed on
> 2026-10-08 15:40 without the conversion check (see `results/plan.md`).

Only if the conversion check passes:

```
cd ~/es-reasoning-finetune && scripts/serve_base.sh
STOP_AT=18:45 setsid nohup scripts/run_phase1.sh >> results/phase1_run.log 2>&1 < /dev/null &
```

Resumable: each runner skips items already in `results/mgsm_raw.base9b.jsonl`
/ `results/belebele_raw.base9b.jsonl`. Order: MGSM es, MGSM en, Belebele es,
Belebele en.

## After phase 1

Score (score_mgsm.py / score_belebele.py with `--raw`), compute mean total
tokens per question and the MGSM es/en ratio, apply the GATE (≥ 1.15) from
`results/plan.md`.

## Plan for 2026-10-09 (written 2026-10-08, in this order)

Never touch the 27B server (:8092); it stays down while the 9B uses the GPU.

a) **Belebele, if it did not finish on 2026-10-08:** resume as today
   (resumable, skips done ids; same server, same settings):
   ```
   cd ~/es-reasoning-finetune && scripts/serve_base.sh
   STOP_AT=18:45 setsid nohup scripts/run_phase1.sh >> results/phase1_run.log 2>&1 < /dev/null &
   ```
   (MGSM is already 500/500, so the chain goes straight to Belebele.) Save
   the exact server line from `/proc/<pid>/cmdline` as on 2026-10-08
   (`results/serve_cmd.base9b.txt`). Continue only when MGSM is 500/500 and
   Belebele 600/600, with no duplicate (id, lang).
b) **Commit phase-1 results.** Not the raw files
   (`results/mgsm_raw.base9b.jsonl`, `results/belebele_raw.base9b.jsonl`:
   not committed). Commit: `results/phase1_run.log`,
   `results/serve_cmd.base9b.txt`, `results/server_props.base9b.json`, and
   the scorers' output (`score_mgsm.py` / `score_belebele.py --raw ...`, which
   print to stdout) saved as `results/score_mgsm.base9b.txt` and
   `results/score_belebele.base9b.txt` (added 2026-10-09:
   `results/gate_mgsm.base9b.txt` and `scripts/gate_ratio.py`). The plan does not name these summary
   files; this is the list. The gate (MGSM es/en ≥ 1.15) is **not decided**
   until c) passes (plan amendment 2026-10-08).
c) **Conversion check:** stop :8093 (kill the llama-server PID), then
   ```
   .venv-train/bin/python scripts/conversion_check.py
   ```
   (served GGUF a96c8c42; HF side on CPU, ~18 GB RAM; creates and deletes a
   ~18 GB BF16 GGUF). Commit `results/conversion_check.{md,json}` with the
   verdict.
   **If it FAILS** (top-1 ≤ 95% on either link): nothing is trained; phase-1
   results are invalidated and phase 1 is repeated, per the rule
   (results/plan.md, amendment 2026-10-08). Stop here and report.

   *Added 2026-10-09, before c):*

   c0) **Rebuild llama-perplexity** (with :8093 stopped). The installed
       `~/llama.cpp/build/bin/llama-perplexity` (and
       `libllama-perplexity-impl.so`) are from 2026-08-14, build 10434, commit
       `7e4c0a968`, linked against the 3466812d1 libs (`libllama.so.0` →
       0.3.0): it segfaults (rc 139) even on `--version`. Link 2 of c) needs
       it. Rebuild only that target at 3466812d1:
       ```
       git -C ~/llama.cpp rev-parse HEAD   # must be 3466812d1f06728effe7c0f3c0671117f461672d
       cmake --build ~/llama.cpp/build --target llama-perplexity -j
       ~/llama.cpp/build/bin/llama-perplexity --version   # must show commit 3466812d1
       ```
   c1) **Guard in conversion_check.py**, own commit, BEFORE running c): abort
       if `llama-perplexity --version` does not show the pinned commit
       (today it only checks `git HEAD` of ~/llama.cpp, so a stale binary
       that did not crash would have run silently).

   **Gate, after c)** (`results/gate_mgsm.base9b.txt`, from
   `scripts/gate_ratio.py`: MGSM es/en mean total tokens = **1.1235**,
   computed 2026-10-09, not yet applied). Threshold and metric are not
   touched:
   - c) passes and ratio ≥ 1.15 → d).
   - c) passes and ratio < 1.15 → nothing is trained. Write the result
     "problem not reproduced on this model" per results/plan.md, compared
     against the 27B's **1.34×** (decided 2026-10-09; es-eval's 2.15× is
     lever 2's cost and is not used here). Label (decided 2026-10-09, the
     chat template counts as part of the model):
     **"Direct comparison of the models as served (model + template +
     quant): 27B IQ3_S 1.34× vs 9B Q8_0 1.1235×. Same items, metric, runner,
     scorer and build. Caveat: the quantization differs and can affect
     reasoning length; the effect of model size cannot be separated from
     the effect of the quant."**
     Same (sha256):
     - metric: mean of prompt_tokens + completion_tokens from `usage`, ratio
       of means (es-eval `results/lever2_plan.md`, "Metric": en 493.9,
       es 659.6, 1.34×; here `scripts/gate_ratio.py`);
     - items: `data/mgsm_es.jsonl`
       `f90f2eda3c16770c91a3b6a78da9ee7199db55b77d5d6b28e6556fa12caea905`,
       `data/mgsm_en.jsonl`
       `0cb0acbad091022b4ab3c4502f9e634025ae7f1d81aec0e1e13de4d2b763d6ed`
       (identical in both repos, 250 ids each);
     - prompts and sampling: `scripts/run_mgsm.py`
       `2645434a7affdb3376af46e4fc12e91dfaea4e2d4c11d5cd61350ee40d14b5a9`
       (identical; here called through `scripts/run_baseline.py`
       `2f62b2eaaba3b151df3efc8ab12dd05b9c94ae3f02a2f37821b6c1d18b729ddd`);
       no system prompt;
     - scorer: `scripts/score_mgsm.py`
       `0f4e71421d993c09c15c715b694db0cbec33180d927d15ca05ee2e466ea157bf`
       (identical);
     - build: llama.cpp b10751-3466812d1 (both `server_props`);
     - mean prompt tokens: en 89.1, es 100.8 in both.
     Differs:
     - chat template (sha256 of `chat_template` in server props): 27B
       `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`,
       9B `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`;
     - quant: 27B `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (~3.5 bits/weight) vs
       9B `Qwen3.5-9B-text-mtp-Q8_0.gguf`
       `a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496`;
     - server: 1 slot, ctx 65536 vs 4 slots, ctx 36864.
     Not recomputed. The "1.50 → 1.34" in point 5 of es-eval's README is a
     different figure (lever 1, reasoning tokens) and is NOT used.
   - c) fails → phase 1 is repeated per the rule.

   `med_reas_tk` in `results/score_mgsm.base9b.txt` stays `nan`
   (`--no-tokenize`: the scorer's /tokenize points at :8092, the 27B); not
   the gate metric. Not addressed now.
d) **If c) passes and the gate passes, resume test of train_lora.py (throwaway, does not count
   as arm A):**
   ```
   .venv-train/bin/python scripts/train_lora.py --arm A --ckpt-dir checkpoints/_resume_test --stop-after-step 10
   .venv-train/bin/python scripts/train_lora.py --arm A --ckpt-dir checkpoints/_resume_test --stop-after-step 12 --resume
   ```
   Confirm: the second run logs `resumed from .../step-0010 at step 10/250`
   and then steps 11 and 12; loss falls over steps 1–12
   (`checkpoints/_resume_test/A/metrics.jsonl`); peak VRAM fits. Then delete
   `checkpoints/_resume_test/` (it must not be reused as arm A).
e) **Arm A, for real (only if d) ran, i.e. c) passed and the gate passed):**
   ```
   setsid nohup .venv-train/bin/python -u scripts/train_lora.py --arm A --stop-at 18:45 \
       >> results/train_A.log 2>&1 < /dev/null &
   ```
   Next days: same command with `--resume` until `checkpoints/A/final/`
   exists. Order of arms: A, then control, then B (results/plan.md).
