#!/usr/bin/env python3
"""kl_outliers.py <prompts.json> <ref_dir> <atlas_logits_dir>... : top KL(ref||atlas) rows per arm.
For each of the worst rows: position, q mod 4, the true next token, and ref/atlas top-5 with probs."""
import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from admission_gate import atlas_chunks, load_atlas, load_ref, log_softmax
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/home/flocka/models/GLM-5.3-Flash-exl3-2.05bpw/tokenizer.json")
prompts = json.load(open(sys.argv[1])); ref_dir = sys.argv[2]
show = lambda ids: repr(tok.decode([int(ids)]))
worst = {}
for d in sys.argv[3:]:
    chunks = atlas_chunks(d); c = 0; rows_all = []
    for i, p in enumerate(prompts):
        n = len(p["ids"]); parts = []; have = 0
        while have < n:
            _, r, f = chunks[c]; parts.append(load_atlas(f, r)); have += r; c += 1
        atl = np.concatenate(parts)[:-1]; ref = load_ref(ref_dir, i, n)[:-1]
        rlp, alp = log_softmax(ref), log_softmax(atl)
        kl = (np.exp(rlp) * (rlp - alp)).sum(1)
        ent = -(np.exp(rlp) * rlp).sum(1)
        for pos in np.argsort(kl)[::-1][:6]:
            rows_all.append((kl[pos], i, pos, ent[pos], rlp[pos], alp[pos], p["ids"][pos + 1]))
        q = np.quantile(kl, [0.5, 0.9, 0.99]); print(f"{d.split('/')[-2]} {p['name']}: KL p50={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.3f} max={kl.max():.3f} rows>1.0: {(kl>1).sum()}")
        worst.setdefault(d, {})[i] = set(np.argsort(kl)[::-1][:20].tolist())
    rows_all.sort(key=lambda x: -x[0])
    for kl, i, pos, ent, r, a, truth in rows_all[:10]:
        rt, at = np.argsort(r)[::-1][:5], np.argsort(a)[::-1][:5]
        print(f"  {prompts[i]['name']} pos={pos} (predicting {pos+1}, q%4={pos%4}, pool_edge={(pos%4)==3}) KL={kl:.3f} ref_entropy={ent:.2f} true={show(truth)}")
        print("    ref  : " + "  ".join(f"{show(t)}:{np.exp(r[t]):.3f}" for t in rt))
        print("    atlas: " + "  ".join(f"{show(t)}:{np.exp(a[t]):.3f}" for t in at))
ds = list(worst)
if len(ds) == 2:
    for i in range(len(prompts)):
        o = worst[ds[0]][i] & worst[ds[1]][i]
        print(f"overlap of top-20 KL positions between arms, {prompts[i]['name']}: {len(o)}/20 {sorted(o)}")
