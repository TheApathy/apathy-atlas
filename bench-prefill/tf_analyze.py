#!/usr/bin/env python3
"""tf_analyze.py REF_ARM CAND_ARM: teacher-forced comparison over captured full-logit dumps."""
import json, os, sys, math
import numpy as np
B="/home/flocka/atlas/flashnext-prefill-work/bench"
ref, cand = sys.argv[1], sys.argv[2]
names=json.load(open(f"{B}/tf_corpus.json"))
def logsoftmax(x):
    x=x.astype(np.float64); m=x.max(); z=x-m; return z-np.log(np.exp(z).sum())
tot=0; agree=0; kl=[]; nll_r=[]; nll_c=[]; top1_r=0; top1_c=0; per=[]
for name in names:
    req=json.load(open(f"{B}/tf-req-{name}.json")); ids=req["prompt_token_ids"]
    n=0; a=0; k=[]; tr=0; tc=0; lr=[]; lc=[]
    for f in sorted(os.listdir(f"{B}/results/tf-{ref}/{name}")):
        p=int(f[7:11]); cf=f"{B}/results/tf-{cand}/{name}/{f}"
        if not os.path.exists(cf): continue
        lr_=np.fromfile(f"{B}/results/tf-{ref}/{name}/{f}", dtype=np.float32); lc_=np.fromfile(cf, dtype=np.float32)
        pr=logsoftmax(lr_); pc=logsoftmax(lc_); true=ids[p]
        ar=int(pr.argmax()); ac=int(pc.argmax())
        n+=1; a+=(ar==ac); k.append(float((np.exp(pr)*(pr-pc)).sum()))
        tr+=(ar==true); tc+=(ac==true); lr.append(-pr[true]); lc.append(-pc[true])
    per.append((name,n,a/n,np.mean(k),tr/n,tc/n,np.mean(lr),np.mean(lc)))
    tot+=n; agree+=a; kl+=k; nll_r+=lr; nll_c+=lc; top1_r+=tr; top1_c+=tc
print(f"ref={ref} cand={cand}")
print(f"{'prompt':>7} {'n':>4} {'argmax agree':>12} {'mean KL':>9} {'top1 ref':>9} {'top1 cand':>9} {'NLL ref':>8} {'NLL cand':>8}")
for r in per: print(f"{r[0]:>7} {r[1]:4d} {r[2]:12.4f} {r[3]:9.5f} {r[4]:9.4f} {r[5]:9.4f} {r[6]:8.4f} {r[7]:8.4f}")
print(f"{'ALL':>7} {tot:4d} {agree/tot:12.4f} {np.mean(kl):9.5f} {top1_r/tot:9.4f} {top1_c/tot:9.4f} {np.mean(nll_r):8.4f} {np.mean(nll_c):8.4f}")
print(f"top-1 vs true: cand/ref = {top1_c/max(top1_r,1):.4f}; NLL cand/ref = {np.mean(nll_c)/np.mean(nll_r):.4f}; median KL {np.median(kl):.5f} max KL {max(kl):.4f}")
