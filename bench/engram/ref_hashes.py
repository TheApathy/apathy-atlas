import json, os, sys, numpy as np, torch
D="/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
sys.path.insert(0, os.path.join(D,"inference"))
import engram as E
from transformers import AutoTokenizer
cfg=json.load(open(os.path.join(D,"inference","config.json")))
tok=AutoTokenizer.from_pretrained(D, trust_remote_code=True)
class A:
    engram_layer_ids=tuple(cfg["engram_layer_ids"]); engram_max_ngram_size=cfg["engram_max_ngram_size"]
    engram_n_heads=cfg["engram_n_heads"]; engram_vocab_size=cfg["engram_vocab_size"]
    engram_num_embeddings=tuple(cfg["engram_num_embeddings"]); engram_head_dim=cfg["engram_head_dim"]
    engram_pad_id=cfg["engram_pad_id"]; engram_compressed_vocab_size=cfg["engram_compressed_vocab_size"]
    max_batch_size=1; max_seq_len=4096
layout=E.EngramLayout.from_args(A)
st=E.NgramHashState(A, layout, tok)
text=open(sys.argv[1]).read() if len(sys.argv)>1 else "The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX. def f(x):\n    return x*2\n"
ids=tok(text, return_tensors="pt").input_ids[0][:2048]
print("tokens:", ids.numel())
h=st(ids[None], 0)            # [1, T, 2, 24]
h=h[0].contiguous()
np.asarray(h.numpy(),dtype=np.int64).tofile("ref_hashes.bin")
np.asarray(ids.numpy(),dtype=np.int32).tofile("ref_ids.bin")
print("hashes shape", tuple(h.shape), "-> ref_hashes.bin")
print("multipliers:", st.multipliers.tolist())
print("first row layer0:", h[0,0,:8].tolist())
print("min/max row id:", int(h.min()), int(h.max()))
