"""Settle the CB3 K-ORDER: does the PTX kernel agree with unpack_cb3_v2's ordering?

A wrong ordering pairs every weight with the wrong activation and still produces
finite, correctly-shaped output -- so this compares the SHIPPED kernel against a
CPU reference built from the dequantizer, on the REAL pack. Agreement settles it;
disagreement says the doc's ambiguity is real and names which side to trust.
"""
import sys, json, struct, numpy as np, torch
sys.path.insert(0, "/home/flocka/atlas/dsv41-prefill-work/tools")
import cb3_moe as M
from cb3 import dequant_cb3_v2

PACK = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3/layers/layer-00.safetensors"
OUT  = "/home/flocka/atlas/apathy-deepseek/kernels/gb10/deepseek-v4.1/cb3"

def read_st(path, names):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for nm in names:
            m = hdr[nm]
            s, e = m["data_offsets"]
            fh.seek(base + s); buf = fh.read(e - s)
            dt = {"U8": np.uint8, "I8": np.int8}[m["dtype"]]
            out[nm] = torch.from_numpy(np.frombuffer(buf, dtype=dt).reshape(m["shape"]).copy())
        return out

EXPERT = 0
planes = {}
raw = read_st(PACK, ["w1_lo","w1_hi","w1_cb","s1","w3_lo","w3_hi","w3_cb","s3",
                     "w2_lo","w2_hi","w2_cb","s2"])
for k, v in raw.items():
    planes[k] = v[EXPERT].contiguous().view(torch.uint8)
for k, v in planes.items(): print(f"  {k:7s} {tuple(v.shape)}")

dev = torch.device("cuda")
arena = M.CB3ArenaV2(1, dev)
arena.load_prepacked_slot(0, planes)
W1, W2, W3 = arena.dequant_slot(0)
print("dequantized:", tuple(W1.shape), tuple(W2.shape), tuple(W3.shape))

torch.manual_seed(0)
T = 4
x = (torch.randn(T, M.DIM, device=dev) * 0.05).to(torch.bfloat16).contiguous()
slots = torch.zeros(T, 1, dtype=torch.int32, device=dev)
wgt = torch.ones(T, 1, dtype=torch.float32, device=dev)
LIMIT = 10.0
y_kernel = M.moe_forward_v2(x, slots, wgt, arena, swiglu_limit=LIMIT)

# CPU reference in float64 from the dequantized weights
xf  = x.double()
g   = (xf @ W1.double().T).clamp(max=LIMIT)
u   = (xf @ W3.double().T).clamp(-LIMIT, LIMIT)
h   = g * torch.sigmoid(g) * u
y_r = (h @ W2.double().T)

a = y_kernel.double(); b = y_r
rel = ((a - b).norm() / b.norm()).item()
cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
print(f"\nrel_l2 = {rel:.6e}   cosine = {cos:.8f}")
print("VERDICT:", "AGREE — K-order settled" if rel < 2e-2 else "DISAGREE — orderings differ")

if rel < 2e-2:
    np.save(f"{OUT}/korder_ref_W1.npy", W1[:16, :256].float().cpu().numpy())
    np.save(f"{OUT}/korder_ref_planes_lo.npy", planes["w1_lo"][:16, :64].numpy())
    np.save(f"{OUT}/korder_ref_planes_hi.npy", planes["w1_hi"][:16, :32].numpy())
    np.save(f"{OUT}/korder_ref_planes_cb.npy", planes["w1_cb"][:16].numpy())
    np.save(f"{OUT}/korder_ref_planes_s.npy",  planes["s1"][:16, :8].numpy())
    print(f"captured reference written to {OUT}/korder_ref_*.npy")
