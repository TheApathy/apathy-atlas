# DeepSeek Vision fc1 GemmEx diagnostic

This isolated probe tests block-0 fc1 on the retained 4x5 shared native norm2
input. It does not run or qualify a model, encoder, vision request or benchmark.
All GPU execution requires the root agent's separate live resource claim.

The source-default Torch 2.10 contract is row-major X[20,1024] and W[5632,1024],
BF16 A/B/C, FP32 compute enum **68**, alpha 1, beta 0, `cublasGemmEx(T,N)`,
column-major m/n/k=5632/20/1024, lda/ldb/ldc=1024/1024/5632, algorithm 99.
The default and full-reduction modes explicitly set math mode 0 and 16.
Each run sets the owned nondefault stream first, then the 8,519,680-byte
workspace, then host pointer mode and math mode. Every attempted prefix fences
and restores mode 0, including errors. Handles and device resources have
explicit cleanup; any cleanup error prevents a successful result.

Pinned Torch git revision: `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`.
The relevant primary files are `aten/src/ATen/cuda/CUDABlas.cpp`,
`aten/src/ATen/native/cuda/Blas.cpp`, `aten/src/ATen/native/cuda/cuBlasCommonArgs.h`,
`aten/src/ATen/cuda/CublasHandlePool.cpp`, and `aten/src/ATen/native/Linear.cpp`.
The historical BLAS preference, TunableOp and workspace environment were not
captured. This candidate is not a proven historical backend until hashes match.

The production native control must reproduce its retained SHA and bytes before
any cuBLAS candidate is admitted. Native/default/full each run twice with reset
outputs, immutable inputs and guarded allocations. There are six allocations,
20,773,888 bytes including 256-byte redzones, below the explicit 32 MiB cap.
This cap excludes CUDA context and library-internal allocations.

Default teacher has a pinned SHA only: its per-element metrics are unavailable.
Full teacher's SHA equals an actual retained native fc1 payload; only that
payload supports FP64 cosine, worst-row cosine, reference-denominator relative
L2, max absolute error and F32 exact-fraction metrics. No payload is synthesized
from a reported hash. The pass gate is exact SHA per mode, not a tolerance.
This does not change the full-encoder .999/.995/.05 gate.

CPU build and tests (root only, no CUDA initialization):

```sh
cargo test --locked --offline --manifest-path /home/flocka/atlas/apathy-deepseek/bench/deepseek-vision-gemmex/Cargo.toml --tests
cargo build --release --locked --offline --manifest-path /home/flocka/atlas/apathy-deepseek/bench/deepseek-vision-gemmex/Cargo.toml
```

`inspect` is CPU-only. It reads pinned metadata, stage payloads and the selected
11,534,336-byte fc1 tensor, not the whole shard/checkpoint. It hashes requested
CUDA/cuBLAS/Lt libraries without loading them, records source receipts, and
creates a fresh output directory containing a selected weight copy and an
admission manifest. The shared P1 canonical-path I/O is reused; its file reader
is not an adversarial concurrent-filesystem sandbox. The selected shard reader
additionally rejects a symlink final component and verifies inode/time/size
before and after the bounded read. No upstream checkpoint files are written.

```sh
/ABS/REVIEWED-BINARY inspect \
  --model /home/flocka/models/DeepSeek-V4-Flash-Vision-EXL3-K2-c171bea5 \
  --corpus /var/tmp/atlas-deepseek-vision-parity-v7-20260905T0440Z \
  --out /ABS/NEW-ADMISSION
```

After reviewing and hashing `admission.json`, and obtaining a fresh GPU claim:

```sh
/ABS/REVIEWED-BINARY run \
  --admission /ABS/NEW-ADMISSION/admission.json \
  --admission-sha REVIEWED_SHA256 \
  --out /ABS/NEW-RUN
```

`run` reconstructs the admission, rechecks every receipt and the selected
weight bytes before CUDA initialization, then records its executable identity.
It uses only fresh output directories and create-new files. Results go to
`result.json`, with receipts on stdout. Exit 0 means both operator SHA gates
passed with successful cleanup; 2 means completed reference mismatch; 1 means
invalid admission, control mismatch, execution or cleanup failure. No downloads,
Python, model construction or reference generation are involved.
