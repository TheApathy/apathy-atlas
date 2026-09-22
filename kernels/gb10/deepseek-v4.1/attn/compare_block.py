"""Whole-block comparison: the driver's taps (fed-core arm vs real-core arm) against an oracle,
per layer and chunk, for attn_o_pre_inverse_rope / attn_out / h. rel_l2 and bf16 bit-exact %."""
import os, sys
import numpy as np
from oracle_io import Run, bf16_to_f32

ref = Run(sys.argv[1]); arms = sys.argv[2:]
print(f"{'tap':28s} " + " ".join(f"{os.path.basename(a):>22s}" for a in arms))
for L in range(21):
    for occ in (0, 1):
        for tap in ("attn_o_pre_inverse_rope", "attn_out", "h"):
            if not ref.has(L, tap, occ):
                continue
            want = ref.load(L, tap, occ, raw=True)
            cells = []
            for a in arms:
                p = f"{a}/L{L:02d}.{tap}.{occ:03d}.bin"
                if not os.path.exists(p):
                    cells.append("-"); continue
                got = np.fromfile(p, np.uint16).reshape(want.shape)
                g, w = bf16_to_f32(got).astype(np.float64), bf16_to_f32(want).astype(np.float64)
                rl = np.linalg.norm(g - w) / np.linalg.norm(w)
                cells.append(f"{rl:.2e} {np.mean(got == want):6.2%}")
            print(f"L{L:02d}.{occ} {tap:24s} " + " ".join(f"{c:>22s}" for c in cells))
