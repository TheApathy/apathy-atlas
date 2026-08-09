#!/usr/bin/env python3
"""GB10 achieved-bandwidth ceiling probe.

What fraction of the 273 GB/s LPDDR5x can real access patterns reach?
Patterns, from friendliest to GEMV-realistic:
  copy      : d2d copy (counts read+write bytes)
  read_sum  : pure streaming read (reduction) - closest to weight streaming
  gemv_bf16 : cuBLAS bf16 GEMV at MoE decode shapes
  gemv_i8   : streaming int8 rows (MXFP4-byte-like) with manual dot
  gather    : 64B-granular random gather - worst-case paged/dedup pattern
"""
import torch, time

assert torch.cuda.is_available()
dev = "cuda"
THEORY = 273.0

def bench(fn, bytes_moved, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return bytes_moved / dt / 1e9

results = {}

# 1) d2d copy, 4 GiB working set (defeats any cache)
n = 1 << 31  # 2 GiB per buffer
a = torch.empty(n, dtype=torch.uint8, device=dev)
b = torch.empty(n, dtype=torch.uint8, device=dev)
results["copy (r+w)"] = bench(lambda: a.copy_(b), 2 * n)
del a, b; torch.cuda.empty_cache()

# 2) pure streaming read: sum over 4 GiB
n = 1 << 30  # elements of f32 = 4 GiB
x = torch.randn(n, dtype=torch.float32, device=dev)
results["read_sum f32"] = bench(lambda: x.sum(), 4 * n // 1)
del x; torch.cuda.empty_cache()

# 3) bf16 GEMV at expert shape: y = W @ x, W [4096, 2048*6] ~ one layer's
#    routed experts' w1 stacked; bytes ~= W bytes
W = torch.randn(6 * 2048, 4096, dtype=torch.bfloat16, device=dev)
x = torch.randn(4096, dtype=torch.bfloat16, device=dev)
results["gemv bf16 (6x2048,4096)"] = bench(lambda: W.mv(x), W.numel() * 2)
del W; torch.cuda.empty_cache()

# big GEMV: whole-layer scale [43*bytes too big]; use 1 GiB weight
W = torch.randn(16384, 32768, dtype=torch.bfloat16, device=dev)  # 1 GiB
x = torch.randn(32768, dtype=torch.bfloat16, device=dev)
results["gemv bf16 1GiB"] = bench(lambda: W.mv(x), W.numel() * 2)
del W, x; torch.cuda.empty_cache()

# 4) int8 streaming matvec-like: read 1 GiB of bytes as [16384, 65536] i8,
#    reduce rows (emulates MXFP4 byte streaming without dequant ALU)
Wi = torch.randint(0, 255, (16384, 65536), dtype=torch.uint8, device=dev)
results["i8 row-reduce 1GiB"] = bench(lambda: Wi.sum(dim=1, dtype=torch.int64), Wi.numel())
del Wi; torch.cuda.empty_cache()

# 5) random 64B-granule gather over 4 GiB pool (paged-KV / dedup worst case)
pool = torch.empty(1 << 32 - 0, dtype=torch.uint8, device=dev)[: (1 << 31)]
pool = pool.view(-1, 64)  # 64B granules
idx = torch.randint(0, pool.shape[0], (1 << 22,), device=dev)  # 256 MiB gathered
results["gather 64B random"] = bench(lambda: pool[idx], idx.numel() * 64 * 2)  # r+w
del pool, idx; torch.cuda.empty_cache()

print(f"{'pattern':28s} {'GB/s':>8s}  {'% of 273':>8s}")
for k, v in results.items():
    print(f"{k:28s} {v:8.1f}  {100*v/THEORY:7.1f}%")
