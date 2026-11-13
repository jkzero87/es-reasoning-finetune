# NEXT

## Status 2026-11-12 22:5x (causal test, interrupted for the night)

Ran tonight: label audit (`results/label_audit_2026-11-12.md`), fixes (`data/fixes_2026-11-12.jsonl`), causal test jobs `fix27b` MGSM 30/30 and Belebele 25/25 (`results/*_raw.fix27b.jsonl`). `rep27b` started (partial rows may exist in `results/mgsm_raw.rep27b.jsonl`, not committed).
Pending: `rep27b` (100 rows), `fix9b` (MGSM 30, Belebele 25), `rep9b` (120 rows); then `results/causal_test_2026-11-12.md` (regenerate with `.venv/bin/python scripts/causal_report.py`), `results/RESUMEN_REUNION.md`. Resume, with :8092 down: `cd ~/es-reasoning-finetune && setsid nohup scripts/run_causal.sh >> results/causal_run.log 2>&1 < /dev/null &` (skips rows already written).


## Status 2026-11-12 (gate applied; nothing running; the local agent takes over after Juan's queue)

Objective (Juan, 2026-11-12; verbatim in README.md and results/plan.md): data and practical conclusions on how much, where and why Spanish costs more in open models; fine-tuning is one test, not the goal.

Done today:
- **c) conversion check: PASS** (`results/conversion_check.md`, commit `7b40d9d`): link 1 top-1 98.587% (920 positions), link 2 same-top 98.81 ± 0.484%. Exact command of "Tomorrow (2026-10-10), FIRST" below, precheck bf382e74… passed. The script deleted the BF16 GGUF at its end (before the verdict commit, not after); nothing of it is left on disk.
- **Gate applied as written** (`results/gate_decision.md`): MGSM es/en 1.1235 < 1.15 → **nothing is trained**; MGSM result "problem not reproduced on this model" (27B 1.34×). Belebele 1.5401 is descriptive. d) resume test and e) arm A below are **not run and not pending**.
- New descriptive metric SUCCESS_PER_1K (`scripts/efficiency_baseline.py`, `results/efficiency_baseline.md`; plan amendment 2026-11-12), run on the 9B base raws and the 27B raws.
- zh data fixed: `data/belebele_zh.jsonl` was es-eval's raw 976-line file (488 passage rows + 488 question rows, no options or gold); re-exported in the en/es format, 488 ids, qid and gold equal to en, sample ids 300/300 (`scripts/export_belebele_zh.py`, `SOURCE.md`).
- Tracked: bench/, zh data, phase-1 raw files, token_split files. `scripts/peek_pairs.py` stays untracked on purpose.
- `bench/CORRER.md`: the MTP note was wrong; MGSM and Belebele base9b both ran with MTP on b10751 (evidence there).
- Training base `~/models/Qwen3.5-9B-text` was missing; rebuilt from `-mtp` by dropping only `mtp.*`, 8,953,803,264 params (`results/training_base_rebuild.md`). Caveat there: a Q8_0 conversion of the rebuild has the plan's exact byte size but not its sha256 (`34a75e46…` vs `a3e9970c…`; cause not established, test GGUF deleted). `~/models/Qwen3.5-9B-hf` and the no-MTP Q8_0 GGUF are not on disk; neither is needed unless training is reopened by a new amendment, which must then name the base.
- `scripts/run_baseline.py` gained `--bench proxlite`, `--thinking off`, `--limit-per-lang` (defaults unchanged). Its sha256 `2f62b2ea…` quoted under "Gate, after c)" is the phase-1 version (commit `2df4b34`).
- main pushed to origin with this commit (approved by Juan 2026-10-12 and again 2026-11-12).

**Next, in order:**
1. **Juan launches the GPU queue** (plan amendment 2026-11-12: q1 zh baseline, q2 reasoning off, q3 ProX-Lite). Command, preconditions and projected hours: `bench/CORRER.md`. :8092 must be down; the script refuses otherwise. Resumable with the same command; `results/QUEUE_DONE` is written only when every row count is exact.
2. **Local agent (27B via dsh), CPU only:** `AGENTS.md`, then `bench/TAREAS_AGENTE.md` T1–T6 (counts, scorers, efficiency tables, CONCLUSIONES.md, README tables, local commit, audit). It never pushes.
3. Juan reviews the agent's commit and pushes.

Rules that still hold: never change the gate, the threshold or a committed amendment; new decisions go in as dated amendments in results/plan.md before the run they affect; serve with the phase-1 libggml-cuda (rule 2026-10-09); never touch :8092.

Everything below is history, kept as written (dated figures such as "11 commits ahead" on 2026-10-12 or "79 GB free" on 2026-10-09 were true on their dates; on 2026-11-12: `df -h ~` 239G free before the rebuild).

## TOMORROW, AFTER the dsh items (Juan 2026-10-13: dsh first): c) conversion check, ~50 min, must start by ~17:45

- Timing 2026-10-13 16:29 (scratch script, conversion_check.py unchanged, torch gated-delta path): MGSM es id 1
  (109 tokens): load 2.5 s, logits forward 29.3 s, cached forward 23.4 s, greedy 1.51 s/token -> ~245 s/prompt,
  ~41 min for the 10 HF prompts; whole check ~47-55 min (BF16 GGUF already on disk, no conversion).
- 2026-10-13 run: started 16:31:08 with the exact command below; precheck PASS (bf382e74…), BF16 logits and Q8_0
  greedy done, HF CPU stage from 16:33:35; **stopped 16:40 by Juan's priority change (dsh first). Not a failure,
  no verdict, nothing written** (log appended to `results/conversion_check.log`). BF16 GGUF still on disk (reused).
- Plan: start it after the dsh items (stop the dsh stack first: nothing else on the GPU), same command as "Tomorrow (2026-10-10),
  FIRST" below; then the rest of the 2026-10-12 resume point (verdict commit, gate as written with threshold 1.15,
  delete BF16 GGUF only after the verdict, approved push of main).

## Status 2026-10-12 14:25 (unattended run; stopped, nothing running)

- Pre-check: dsh stack down, no compute processes, :8093 free.
- c) conversion check: **BLOCKED, not decided.** Run exactly as "Tomorrow (2026-10-10), FIRST: c)" below,
  14:16:18–14:20:14. Precheck PASS (`libggml-cuda.so.0 -> bf382e74…`, `llama-perplexity` build 10751 3466812d1).
  Done before the failure: served-GGUF sha256, tokenization, Q8_0 greedy x10 (GPU, 14:19:22–14:19:54).
  Failed at `link 1 / greedy: HF bf16 on CPU` (`scripts/conversion_check.py:199`): the HF Qwen3.5 gated-delta-rule
  layers run `fla/ops/gated_delta_rule/chunk.py` (Triton, CUDA-only) →
  `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`. Exit 1. No
  `results/conversion_check.{md,json}` written. Log: `results/conversion_check.log`.
  Not fixed (no improvised changes): the HF side as specified (bf16 on CPU) cannot run with this venv's
  transformers/fla path. Options for Juan to decide: a torch fallback for fla on CPU, or the HF side on GPU (does
  not fit 16 GB in bf16), or an amendment of the method. Any of them changes the check, so it needs an OK.
- Left on disk (regenerable, reused by the script if present): `~/models/Qwen3.5-9B-text-mtp-BF16/
  Qwen3.5-9B-text-mtp-BF16.gguf` (~18 GB; the script deletes it only at its end).
- Gate (ratio 1.1235, `results/gate_mgsm.base9b.txt`): still NOT applied (needs c)).
- Not run (training, waits for Juan): d) resume test, e) arm A. No `train_lora.py --dry-run` is listed here.
- Push: not done. main is 11 commits ahead of origin and those include code commits (conversion_check.py guard,
  train_lora.py, training scripts), so a results/docs-only push is not possible without them.

**Update 2026-10-12 20:15 (after hours, nothing run):** transformers 5.17.0 has reference torch kernels for the
gated-delta layers (`modeling_qwen3_5.py:301,438`), chosen at import time (`integrations/hub_kernels.py:829`:
hub kernels -> `fla` -> torch; no device check, no switch). Plan amendment 2026-10-12 (results/plan.md) + backend
switch in `scripts/conversion_check.py` (`sys.modules["fla"] = None` before importing transformers; nothing else
changed). Verified import-only: with the switch both functions resolve to the torch reference; without it, to fla.

**Resume point (next day, before 18:40):** 1) estimate the HF CPU runtime on the torch path (the reference
chunk kernel is far slower than fla; time one prompt's forward + a few greedy tokens in a scratch script, do not
change conversion_check.py); if the full check (10 prompts: logits + 128 greedy tokens each) cannot finish by 18:40,
report and do not start. 2) Run c) with the command below (same precheck). 3) Commit
`results/conversion_check.{md,json}`, apply the gate exactly as written, update README/NEXT.md; delete the BF16
GGUF only after a final verdict. 4) Push main (approved by Juan 2026-10-12, normal push, after the c) commit).

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
  **libggml-cuda:** `cmake --target llama-perplexity` also
  recompiled ggml-cuda (the .cu sources were newer than this build dir's
  objects) and relinked the shared libs. All relinked libs are byte-identical
  to the copies taken at 18:30, EXCEPT `libggml-cuda.so.0.22.0`
  (bf382e74… → 8b541eb1…, same source rev, same nvcc 13.1.115). The phase-1
  copy is kept in `~/llama-build-3466812d1-phase1-bak/` (SHA256SUMS there);
  `libggml-base` / `libggml-cpu` had already been relinked at 18:24 before
  the backup, so their phase-1 bytes were not kept (the relinked ones match
  the backup). c)'s Q8_0 GPU side would run on the rebuilt libggml-cuda,
  not byte-for-byte the one that served phase 1. Decided (Juan,
  2026-10-09): serve with the phase-1 copy, see "Rule (2026-10-09)" below.
- c1) guard in `scripts/conversion_check.py`, commit `ee536ad`: aborts unless
  `llama-perplexity --version` shows 3466812d1 (checked: real binary passes;
  segfault and stale build abort).

## Port :8094 (2026-10-09)

:8094 belongs to the multiplayer-ai thread (a separate 27B with `--jinja`
for its F1 part 3; see `~/multiplayer-ai/README.md`). It is NOT part of this
project and is used only AFTER this project's gate, never while :8093 or
c) use the GPU (the 16 GB card cannot hold both). :8092 and
`~/bin/manifiestate` are not touched by either thread.

Order for 2026-10-10: c) (block below) → verdict + commit of
`results/conversion_check.md` → gate with `results/gate_mgsm.base9b.txt` →
only then, with the GPU free, multiplayer-ai on :8094.

## Rule (2026-10-09): serve with the phase-1 libggml-cuda

Everything this project SERVES or evaluates with llama.cpp — c) and, if
training happens, the evaluation of the arms — runs with
`LD_LIBRARY_PATH=~/llama-build-3466812d1-phase1-bak`, i.e. with phase 1's
`libggml-cuda.so.0.22.0` (sha256 `bf382e74cc5178cf…`, full hash in that
dir's SHA256SUMS). The rebuilt one in `~/llama.cpp/build/bin`
(`8b541eb1731fe8e3…`) is not used to serve.
Limitation, as is: `libggml-base` and `libggml-cpu` were relinked (18:24)
before the copy (18:30), so their phase-1 bytes cannot be verified: same
sources and same commit, no byte-for-byte proof.

**Precondition — DONE 2026-10-09 18:37 (Juan's OK):** the backup dir holds only
the versioned files (`libggml-cuda.so.0.22.0`, …), not the soname names the
loader looks for (`libggml-cuda.so.0`, …). With the dir as it is,
`LD_LIBRARY_PATH` finds nothing there and the RUNPATH (`build/bin`) wins:
`ldd` shows the REBUILT lib. Checked 2026-10-09 18:35 in a scratch dir:
with soname symlinks to the backup files, `ldd` resolves libggml-cuda.so.0
to the bf382e74… file and `llama-perplexity --version` still shows
3466812d1. To add them (symlinks only; the files are not touched):
```
cd ~/llama-build-3466812d1-phase1-bak && for f in *.so.*; do
  so=$(readelf -d "$f" | sed -nE 's/.*SONAME.*\[(.*)\]/\1/p'); ln -s "$f" "$so"; done
```
(gives libggml-base.so.0, libggml-cpu.so.0, libggml-cuda.so.0, libggml.so.0,
libllama-common.so.0, libllama.so.0; libllama-server-impl.so and
llama-server need none.)
Done with that command: 6 symlinks, no file modified or moved
(`sha256sum -c SHA256SUMS` OK before and after; SHA256SUMS unchanged).
Pre-check below, run as written with the LD_LIBRARY_PATH, 18:37:
```
loaded: /home/jkzero/llama-build-3466812d1-phase1-bak/libggml-cuda.so.0 -> bf382e74cc5178cf9f0a00bf382a880626bb055053db6e80dc7ea3b72167124d
PRECHECK_PASS
llama-perplexity: version: 0.3.0-dev (build 10751, commit 3466812d1)
llama-server:     version: 0.3.0-dev (build 10751, commit 3466812d1)
```
(`ldd llama-server` under the same LD_LIBRARY_PATH also loads the backup's
libggml-cuda.so.0.)

## Tomorrow (2026-10-10), FIRST: c) conversion check

Nothing else before it (not d), not e)). :8093 must stay down; never touch
:8092. The precondition above is done. Then, in one shell:
```
cd ~/es-reasoning-finetune
export LD_LIBRARY_PATH=~/llama-build-3466812d1-phase1-bak
# Pre-check: the libggml-cuda actually loaded must be phase 1's (bf382e74...); if not, nothing after it runs.
precheck() {
  lib=$(ldd ~/llama.cpp/build/bin/llama-perplexity | sed -nE 's/.*libggml-cuda\.so\.0 => ([^ ]+).*/\1/p')
  want=$(awk '$2 == "libggml-cuda.so.0.22.0" {print $1}' ~/llama-build-3466812d1-phase1-bak/SHA256SUMS)
  got=$(sha256sum "$(readlink -f "$lib")" | cut -d' ' -f1)
  echo "loaded: $lib -> $got"
  [ -n "$got" ] && [ "$got" = "$want" ] && [ "${got:0:8}" = bf382e74 ] || { echo "NOT phase-1 libggml-cuda: c) does not run"; return 1; }
}
precheck && ~/llama.cpp/build/bin/llama-perplexity --version && \
.venv-train/bin/python scripts/conversion_check.py
```
(The script's own logits helper is built with `-rpath build/bin`, i.e.
RUNPATH, so the same `LD_LIBRARY_PATH` applies to it.)
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
