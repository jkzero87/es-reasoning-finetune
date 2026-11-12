# Conversion fidelity check

2026-11-12T14:31:17, llama.cpp `3466812d1f06728effe7c0f3c0671117f461672d`. Served GGUF `/home/jkzero/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf` (sha256 `a96c8c42919aa1b963fd827e5cfa3b0fe8a14a7ba82a992797ce617c6f438496`) vs HF
`/home/jkzero/models/Qwen3.5-9B-text-mtp` (bf16, transformers, CPU). Method: `scripts/conversion_check.py`
(results/plan.md amendment 2026-10-08, NEXT.md). Raw: `results/conversion_check.json`.

**Verdict: PASS** (FAIL if top-1 agreement <= 95.0% on either link).

| link | top-1 agreement | KL | positions |
|---|---:|---:|---:|
| 1. HF bf16 -> BF16 GGUF (libllama, ngl 0) | 98.587% | mean KL(HF‖GGUF) 0.000506 | 920 |
| 2. BF16 GGUF -> Q8_0 (llama-perplexity, -c 128) | 98.81 ± 0.484% | mean KLD 0.001418 ± 0.000142 | 8 chunks, ~504 |

Greedy, 128 tokens, HF vs served Q8_0, first divergence (none = identical): es1: none, es2: 77, es3: 31, es4: none, es5: none, en1: 17, en2: none, en3: none, en4: 31, en5: 81.

Tokenization HF vs llama.cpp identical on the 10 prompts: 10/10 (recorded, not decisive).
