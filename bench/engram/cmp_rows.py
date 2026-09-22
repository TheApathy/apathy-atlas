import numpy as np, torch, os, sys
D="/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
SH=[(f"{D}/model-00047-of-00048.safetensors",664,98305579672),
    (f"{D}/model-00048-of-00048.safetensors",672,98308271264)]
h=np.fromfile("ref_hashes.bin",np.int64); T=h.size//48; h=h.reshape(T,2,24)
out=np.empty((T,2,24,256),np.float32)
for li,(p,woff,soff) in enumerate(SH):
    fd=os.open(p,os.O_RDONLY)
    ids=h[:,li,:].reshape(-1)
    uniq,inv=np.unique(ids,return_inverse=True)
    raw=np.empty((uniq.size,264),np.uint8)
    for i,r in enumerate(uniq):
        r=int(r)
        raw[i,:256]=np.frombuffer(os.pread(fd,256,woff+r*256),np.uint8)
        raw[i,256:]=np.frombuffer(os.pread(fd,8,soff+r*8),np.uint8)
    os.close(fd)
    t=torch.from_numpy(raw)
    vals=t[:,:256].view(torch.float8_e4m3fn).float()
    scales=torch.exp2(t[:,256:].float()-127.0)
    deq=(vals.unflatten(-1,(8,32))*scales.unsqueeze(-1)).flatten(-2)   # [n,256]
    out[:,li,:,:]=deq.numpy()[inv].reshape(T,24,256)
ref=out.reshape(-1)
cand=np.fromfile(sys.argv[1] if len(sys.argv)>1 else "rust_rows.bin",np.float32)
assert cand.size==ref.size,(cand.size,ref.size)
# both sides are exact operations in f32 (e2m1-class mantissa x power-of-two scale),
# so anything but bit-equality is a real bug, not rounding.
neq=int((cand.view(np.uint32)!=ref.view(np.uint32)).sum())
nz=int((ref!=0).sum())
print(f"values: {ref.size}   nonzero refs: {nz} ({100*nz/ref.size:.1f}%)")
print(f"bitwise mismatches: {neq} / {ref.size}   {'EXACT MATCH' if neq==0 else 'MISMATCH'}")
if neq:
    i=int(np.argmax(cand.view(np.uint32)!=ref.view(np.uint32)))
    print(f"  first at {i}: rust {cand[i]!r} vs python {ref[i]!r}")
print(f"ref range: [{ref.min():.6g}, {ref.max():.6g}]  distinct values {len(np.unique(ref[:100000]))}")
sys.exit(1 if neq else 0)
