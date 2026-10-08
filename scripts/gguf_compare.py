#!/usr/bin/env python3
"""Compare a fine-tuned served GGUF with the baseline served GGUF (a96c8c42...).

Checks: same tensor names, shapes and types; the MTP (nextn) block tensors are
byte-identical (base head re-attached untrained); reports how many tensors
differ. Exit 1 on any structural mismatch or a changed MTP tensor.

  PYTHONPATH=~/llama.cpp/gguf-py python scripts/gguf_compare.py NEW.gguf [BASE.gguf]
"""
import json
import sys

import numpy as np
from gguf import GGUFReader

BASE = "/home/jkzero/models/Qwen3.5-9B-text-mtp-Q8_0/Qwen3.5-9B-text-mtp-Q8_0.gguf"


def field_int(r, suffix):
    for k, f in r.fields.items():
        if k.endswith(suffix):
            return int(f.parts[f.data[0]][0])
    raise SystemExit(f"no metadata key *{suffix}")


def main():
    new_p = sys.argv[1]
    base_p = sys.argv[2] if len(sys.argv) > 2 else BASE
    new, base = GGUFReader(new_p), GGUFReader(base_p)
    nt = {t.name: t for t in new.tensors}
    bt = {t.name: t for t in base.tensors}
    if set(nt) != set(bt):
        raise SystemExit(f"tensor names differ: only new {sorted(set(nt) - set(bt))[:5]}, "
                         f"only base {sorted(set(bt) - set(nt))[:5]}")
    n_blocks = field_int(base, ".block_count")
    n_mtp = field_int(base, ".nextn_predict_layers")
    mtp_prefixes = tuple(f"blk.{i}." for i in range(n_blocks - n_mtp, n_blocks))
    bad, differ, mtp_same, mtp_total = [], 0, 0, 0
    for name, b in bt.items():
        t = nt[name]
        if list(t.shape) != list(b.shape) or t.tensor_type != b.tensor_type:
            bad.append(name)
            continue
        same = np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(b.data).view(np.uint8))
        differ += not same
        if name.startswith(mtp_prefixes):
            mtp_total += 1
            mtp_same += same
    res = {"new": new_p, "base": base_p, "tensors": len(bt), "differ_from_base": differ,
           "mtp_block": list(mtp_prefixes), "mtp_tensors": mtp_total, "mtp_identical": mtp_same,
           "shape_or_type_mismatch": bad}
    print(json.dumps(res))
    if bad or mtp_total == 0 or mtp_same != mtp_total or differ == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
