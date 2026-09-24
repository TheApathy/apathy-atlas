"""Negative control: feed the same comparison a WRONG K-order and watch it fail.

Without this the agreement number is a gate nobody has seen fail. Three wrong
orderings, each a permutation of the SAME multiset -- so any of them would look
finite and correctly-shaped, which is exactly the failure mode CB3_FORMAT.md
warns produces plausible garbage.
"""
import sys, json, struct, numpy as np, torch
sys.path.insert(0, "/home/flocka/atlas/dsv41-prefill-work/tools")
import cb3_moe as M

PACK = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3/layers/layer-00.safetensors"
def read_st(path, names):
    with open(path,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n; out={}
        for nm in names:
            m=hdr[nm]; s,e=m["data_offsets"]; fh.seek(base+s); buf=fh.read(e-s)
            out[nm]=torch.from_numpy(np.frombuffer(buf,dtype=np.uint8).reshape(m["shape"]).copy())
        return out

raw = read_st(PACK, ["w1_lo","w1_hi","w1_cb","s1","w3_lo","w3_hi","w3_cb","s3","w2_lo","w2_hi","w2_cb","s2"])
planes = {k: v[0].contiguous().view(torch.uint8) for k,v in raw.items()}
dev = torch.device("cuda")
arena = M.CB3ArenaV2(1, dev); arena.load_prepacked_slot(0, planes)
W1, W2, W3 = arena.dequant_slot(0)

torch.manual_seed(0)
T=4; LIMIT=10.0
x = (torch.randn(T, M.DIM, device=dev)*0.05).to(torch.bfloat16).contiguous()
slots = torch.zeros(T,1,dtype=torch.int32,device=dev); wgt = torch.ones(T,1,dtype=torch.float32,device=dev)
y_k = M.moe_forward_v2(x, slots, wgt, arena, swiglu_limit=LIMIT).double()

def score(w1, w3, label):
    xf=x.double()
    g=(xf@w1.double().T).clamp(max=LIMIT); u=(xf@w3.double().T).clamp(-LIMIT,LIMIT)
    y=((g*torch.sigmoid(g)*u)@W2.double().T)
    rel=((y_k-y).norm()/y.norm()).item()
    cos=torch.nn.functional.cosine_similarity(y_k.flatten(),y.flatten(),dim=0).item()
    print(f"  {label:38s} rel_l2={rel:.4e}  cos={cos:+.6f}")
    return rel

K=M.DIM
print("negative controls (all are permutations of the same multiset):")
r_true = score(W1, W3, "TRUE order (unpack_cb3_v2)")
# 1. swap the r sub-position: K index g*32+lane*2+r -> ...+(1-r)
idx = torch.arange(K, device=dev); idx_r = idx ^ 1
r_a = score(W1[:, idx_r], W3[:, idx_r], "r sub-position swapped (xor 1)")
# 2. byte-major vs field-major inside each 512 block
blk = torch.arange(512, device=dev).view(16,16,2).permute(1,0,2).reshape(-1)
idx_b = (torch.arange(0,K,512,device=dev)[:,None] + blk[None,:]).reshape(-1)
r_b = score(W1[:, idx_b], W3[:, idx_b], "field-major <-> byte-major in block")
# 3. whole-row shuffle
g2 = torch.Generator(device='cuda'); g2.manual_seed(1)
idx_s = torch.randperm(K, device=dev, generator=g2)
r_c = score(W1[:, idx_s], W3[:, idx_s], "random permutation of K")

print()
ok = r_true < 2e-2 and min(r_a, r_b, r_c) > 0.2
print("CONTROL VERDICT:", "the check CAN fail — agreement is evidence" if ok
      else "INCONCLUSIVE — a wrong order also passed; do not trust the agreement")
