# NEXT

Status 2026-10-07: phase 1 (results/plan.md) running since 17:45 on :8093,
stop_at 18:45; it will be partial. Update the counts below when it stops.

## Tomorrow, FIRST job (before resuming phase 1): conversion fidelity check

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
     Report its KL and "same top" figures.
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
