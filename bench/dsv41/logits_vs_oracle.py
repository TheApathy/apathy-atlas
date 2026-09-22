#!/usr/bin/env python3
"""Step-0 logits vs an oracle capture: KL(ref||ours), top-10 identity (order), argmax."""
import sys, numpy as np
ref = np.fromfile(sys.argv[1], dtype=np.float32).astype(np.float64)
c = (np.fromfile(sys.argv[2], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)[:ref.size].astype(np.float64)
p = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
pr, pc = p(ref), p(c)
kl = float((pr * (np.log(pr + 1e-30) - np.log(pc + 1e-30))).sum())
tr, tc = np.argsort(-ref, kind="stable")[:10], np.argsort(-c, kind="stable")[:10]
print(f"KL={kl:.3e} top10_identical_in_order={list(tr)==list(tc)} top10_set_overlap={len(set(tr)&set(tc))}/10 argmax ref={tr[0]} ours={tc[0]}")
