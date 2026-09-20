#!/usr/bin/env python3
"""Teacher-forced numerics gate: compare two arms' prefill residual streams through one offline
(BF16->f32) final-norm + lm_head, per position. Usage: analyze.py <runA> <runB> [names...]"""
import sys, json, os, numpy as np, torch
from safetensors import safe_open
torch.set_num_threads(16)
A, B = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or ['code', 'prose', 'legal']
shard = '/home/flocka/atlas/qwen38/optimized-qwen/model-00006-of-00006.safetensors'
with safe_open(shard, 'pt') as f:
    W = f.get_tensor('lm_head.weight').float()            # [V, H]
    g = f.get_tensor('model.language_model.norm.weight').float()
V, H = W.shape; eps = 1e-6
def load(path):
    fn = os.path.basename(path); 
    raw = np.fromfile(path, dtype=np.uint8)
    elem = int(fn.split('elem')[1].split('.')[0]) if 'elem' in fn else None
    # our dumps are renamed hidden_<text>.bin; infer element size from length
    if raw.size % (H * 4) == 0 and 'elem4' in fn: x = raw.view(np.float32).reshape(-1, H)
    else: x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, H)
    return torch.from_numpy(x.copy())
def logits(x):
    xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g
    return xn @ W.T
tot = {}
for name in names:
    ids = json.load(open(f'/home/flocka/atlas/qwen27b-prefill-work/bench/gate/ids_{name}.json'))['ids']
    xa, xb = load(f'{A}/hidden_{name}.bin'), load(f'{B}/hidden_{name}.bin')
    n = xa.shape[0]; assert xb.shape[0] == n
    la, lb = logits(xa), logits(xb)
    true = torch.tensor(ids[1:n+1])
    lpa, lpb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
    pa = lpa.exp()
    kl = (pa * (lpa - lpb)).sum(-1)                       # KL(A||B) per position, nats
    aa, ab = la.argmax(-1), lb.argmax(-1)
    agree = (aa == ab).float()
    nll_a, nll_b = -lpa[torch.arange(n), true], -lpb[torch.arange(n), true]
    top1_a, top1_b = (aa == true).float(), (ab == true).float()
    hid_rel = ((xa - xb).norm(dim=-1) / xa.norm(dim=-1))
    for tag, sl in [('all', slice(0, n)), ('last512', slice(n-512, n))]:
        r = dict(positions=int(n if tag=='all' else 512), argmax_agree=agree[sl].mean().item(), kl_mean=kl[sl].mean().item(),
                 kl_max=kl[sl].max().item(), kl_p99=kl[sl].quantile(0.99).item(),
                 nll_A=nll_a[sl].mean().item(), nll_B=nll_b[sl].mean().item(),
                 top1_A=top1_a[sl].mean().item(), top1_B=top1_b[sl].mean().item(),
                 hidden_rel_l2_mean=hid_rel[sl].mean().item(), hidden_rel_l2_max=hid_rel[sl].max().item(),
                 bitexact_hidden=bool(torch.equal(xa[sl], xb[sl])))
        tot[f'{name}/{tag}'] = r
        print(f'{name:6s} {tag:8s} ' + ' '.join(f'{k}={v:.5g}' if isinstance(v,float) else f'{k}={v}' for k, v in r.items()))
json.dump(tot, open(f'{B}/gate_vs_{os.path.basename(A)}.json', 'w'), indent=1)
