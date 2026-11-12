# Training base rebuilt (2026-11-12)

`~/models/Qwen3.5-9B-text` (the training base of `scripts/train_lora.py` and `scripts/merge_lora.py`) was missing on 2026-11-12, and so was its source `~/models/Qwen3.5-9B-hf`. It was rebuilt on CPU from `~/models/Qwen3.5-9B-text-mtp` by dropping only the 15 `mtp.*` tensors (`scripts/rebuild_text_from_mtp.py`); the other 427 tensors are copied bit-for-bit under the same names, and `config.json` and the tokenizer files are copied unchanged. No download.

Parameter counts, checked before writing, equal to `results/feasibility.md`: text-only 8,953,803,264; MTP dropped 243,290,624.

Not checked: byte equality with the deleted original directory (no hash of it was ever recorded; shard boundaries can differ, tensor contents cannot).

Output of the script (sha256, file, bytes):

```
params: text 8,953,803,264 (expected 8,953,803,264), mtp 243,290,624 (expected 243,290,624)
tensors written: 427 in 5 shards -> /home/jkzero/models/Qwen3.5-9B-text
bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a  LICENSE  11544
a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715  chat_template.jinja  7756
f546f8f4c88bafb57e60ec22f585a05f2d58aa2a65be689084e5a8341dd3d237  config.json  1892
a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d  merges.txt  3353259
e3a2660804dd58f7de814cab93a69feae07d8f4066e780c06809252ffde7512b  model-00001.safetensors  4370465352
0706fcf407e92f351cba1b3bf75595bd0616a38c2cca2f06637752cba5c71fa0  model-00002.safetensors  4328526696
d0190279f862f7feb3d3a4a52956d94e52ba31bc25cbd258200d2f0f4f009174  model-00003.safetensors  4347745488
d21439beb4aa058dcf22c58f6b3807ffc5b3a1213181bf25e4624162d06964ed  model-00004.safetensors  4303557776
72546e6be41a3cd5ef19e8c97a8ca29e4e130a8aae8a2c49d8e1f1922c10a28b  model-00005.safetensors  557367448
0153ffba9946f26067b69628a05cb4c97d15c778b54f8273d5a1e4f4c604dd7e  model.safetensors.index.json  31782
5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42  tokenizer.json  12807982
316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8  tokenizer_config.json  16710
ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003  vocab.json  6722759
```
