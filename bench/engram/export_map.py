import json, sys, numpy as np
from transformers import AutoTokenizer
from tokenizers import Regex, normalizers
D="/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
tok=AutoTokenizer.from_pretrained(D, trust_remote_code=True)
sentinel=""
norm=normalizers.Sequence([normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),
  normalizers.Lowercase(),normalizers.Replace(Regex(r"[ \t\r\n]+")," "),
  normalizers.Replace(Regex(r"^ $"),sentinel),normalizers.Strip(),normalizers.Replace(sentinel," ")])
b=tok.backend_tokenizer; k2n={}; lookup=[0]*len(tok)
for i in range(len(tok)):
    t=b.decode([i],skip_special_tokens=False)
    key = b.id_to_token(i) if "�" in t else (norm.normalize_str(t) or t)
    n=k2n.get(key)
    if n is None: n=len(k2n); k2n[key]=n
    lookup[i]=n
V=len(k2n)
print("len(tokenizer) =",len(tok),"compressed vocab =",V,"config says 99092 ->", "MATCH" if V==99092 else "MISMATCH")
a=np.array(lookup,dtype=np.int32)
a.tofile("token_map_i32.bin")
import hashlib
print("token_map rows",a.shape,"sha256",hashlib.sha256(a.tobytes()).hexdigest()[:32])
print("pad: token_map[engram_pad_token_id=2] =",int(a[2]))
# collapse sanity: " The","the","THE" must share an id
for s in [" The","the","THE"," the"]:
    ids=b.encode(s,add_special_tokens=False).ids
    print(f"  {s!r} -> ids {ids} -> compressed {[int(a[i]) for i in ids]}")
