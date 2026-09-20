#!/usr/bin/env python3
"""Compare raw BF16 logits dumps: cmp_logits.py ref.bf16 cand.bf16 [...]"""
import sys, numpy as np
def load(p):
    u=np.fromfile(p,dtype=np.uint16).astype(np.uint32)<<16
    return u.view(np.float32).astype(np.float64)
def lsm(x):
    m=x.max(); return x-m-np.log(np.exp(x-m).sum())
ref=load(sys.argv[1]); lr=lsm(ref); pr=np.exp(lr)
print(f"ref argmax={ref.argmax()} top5={np.argsort(-ref)[:5].tolist()} max={ref.max():.4f}")
for p in sys.argv[2:]:
    c=load(p); lc=lsm(c); 
    kl=float((pr*(lr-lc)).sum()); d=np.abs(c-ref)
    t5r=set(np.argsort(-ref)[:5].tolist()); t5c=set(np.argsort(-c)[:5].tolist())
    print(f"{p}: argmax={c.argmax()} same_argmax={c.argmax()==ref.argmax()} identical={np.array_equal(c,ref)} "
          f"max|d|={d.max():.4f} mean|d|={d.mean():.5f} rel_l2={np.linalg.norm(c-ref)/np.linalg.norm(ref):.2e} KL(ref||c)={kl:.3e} top5_overlap={len(t5r&t5c)}/5 logp_ref_argmax: ref={lr[ref.argmax()]:.4f} cand={lc[ref.argmax()]:.4f}")
