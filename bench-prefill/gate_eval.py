#!/usr/bin/env python3
"""gate_eval.py <base-label> <cand-label>...: teacher-forced metrics from full-logits dumps."""
import sys, os, json, glob, numpy as np
R = '/home/flocka/atlas/glm53-prefill-work/bench/runs'
prompts = json.load(open('/home/flocka/atlas/glm53-prefill-work/bench/prompts_1024.json'))
V = 154880
def load(label, i):
    f = sorted(glob.glob(f'{R}/gate-{label}/logits/logits-{i}-r*.bf16'))[0]
    rows = int(f.split('-r')[-1].split('.')[0])
    u = np.fromfile(f, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(rows, V)
def lsm(x):
    m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
base = sys.argv[1]; cands = sys.argv[2:]
print("| arm | prompt | positions | argmax agree vs base | mean KL(base||cand) | max KL | top-1 acc vs true | NLL vs true |\n|---|---|---|---|---|---|---|---|")
agg = {}
for label in [base] + cands:
    for i, p in enumerate(prompts):
        L = load(label, i); tgt = np.array(p['ids'][1:]); lp = lsm(L[:-1].astype(np.float64)); n = len(tgt)
        top1 = float((L[:-1].argmax(1) == tgt).mean()); nll = float(-lp[np.arange(n), tgt].mean())
        if label == base:
            agree = kl = klmax = float('nan'); agg.setdefault(label, []).append((top1, nll))
        else:
            B = load(base, i); blp = lsm(B[:-1].astype(np.float64))
            agree = float((L[:-1].argmax(1) == B[:-1].argmax(1)).mean())
            klv = (np.exp(blp) * (blp - lp)).sum(1); kl = float(klv.mean()); klmax = float(klv.max())
            agg.setdefault(label, []).append((top1, nll, agree, kl))
        print(f"| {label} | {p['name']} | {n} | {agree:.4f} | {kl:.5f} | {klmax:.3f} | {top1:.4f} | {nll:.4f} |")
print("\n| arm | mean top-1 | mean NLL | mean argmax agree | mean KL |\n|---|---|---|---|---|")
for label, rows in agg.items():
    a = np.array(rows); extra = f"{a[:,2].mean():.4f} | {a[:,3].mean():.5f}" if a.shape[1] == 4 else "ref | ref"
    print(f"| {label} | {a[:,0].mean():.4f} | {a[:,1].mean():.4f} | {extra} |")
