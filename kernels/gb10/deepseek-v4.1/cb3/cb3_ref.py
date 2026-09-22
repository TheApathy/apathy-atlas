import numpy as np
U32 = np.uint32
def lop3(a,b,c,lut):
    # immLut over three inputs, bitwise
    r = np.zeros_like(a)
    for i,(ba,bb,bc) in enumerate([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]):
        if (lut >> i) & 1:
            m = (~a if not ba else a) & (~b if not bb else b) & (~c if not bc else c)
            r |= m
    return r & U32(0xFFFFFFFF)

def prmt(a,b,sel):
    """PTX prmt.b32: byte permute from the 8-byte value {b:a} by 4 nibbles of sel."""
    src = (np.uint64(b) << np.uint64(32)) | np.uint64(a)
    out = np.uint64(0)
    for i in range(4):
        idx = int((sel >> (4*i)) & 0xF)
        byte = int((src >> np.uint64(8*(idx & 7))) & np.uint64(0xFF))
        if idx & 8:  # sign-replicate mode
            byte = 0xFF if (byte & 0x80) else 0x00
        out |= np.uint64(byte) << np.uint64(8*i)
    return U32(out & np.uint64(0xFFFFFFFF))

def half(L, H, A, B, shift, bit):
    a = (L >> U32(shift)) & U32(0x03030303)
    b = (H >> U32(bit-2)) if bit >= 2 else (H << U32(2-bit))
    b &= U32(0xFFFFFFFF)
    ie = lop3(a, b, U32(0x04040404), 0xF8)          # a | (b & c)
    t  = ie >> U32(4)
    r  = lop3(ie, t, U32(0x00FF00FF), 0xA8)          # (a | b) & c
    t  = r >> U32(8)
    r  = (r | t) & U32(0xFFFFFFFF)
    return prmt(A, B, int(r) & 0xFFFF)                # low 16 bits select 4 bytes

def cb3_asm(L, H, A, B, sh, hb):
    ne = half(L, H, A, B, sh,   hb)
    no = half(L, H, A, B, sh+2, hb+1)
    return U32((ne | ((no << U32(4)) & U32(0xFFFFFFFF))) & U32(0xFFFFFFFF))

if __name__ == "__main__":
    rng = np.random.default_rng(20260921)
    n = 64
    L = rng.integers(0, 2**32, n, dtype=np.uint64).astype(U32)
    H = rng.integers(0, 2**32, n, dtype=np.uint64).astype(U32)
    A = rng.integers(0, 2**32, n, dtype=np.uint64).astype(U32)
    B = rng.integers(0, 2**32, n, dtype=np.uint64).astype(U32)
    rows = []
    for variant,(sh,hb) in enumerate([(0,0),(4,2),(0,4),(4,6)]):
        for i in range(n):
            out = cb3_asm(L[i],H[i],A[i],B[i],sh,hb)
            rows.append(f"{variant} {L[i]:08x} {H[i]:08x} {A[i]:08x} {B[i]:08x} {int(out):08x}")
    open("/home/flocka/atlas/apathy-deepseek/kernels/gb10/deepseek-v4.1/cb3/cb3_decode_vectors.txt","w").write("\n".join(rows)+"\n")
    print(f"wrote {len(rows)} vectors across 4 variants")
