# Qwen3.8 v3 Weschera 72 tok/s profile

This records the promoted single-stream speed profile without replacing the
historical BF16-KV production default.

## Measured identity

| Item | Exact identity |
|---|---|
| Hardware | one DGX Spark / GB10, C=1 |
| Atlas revision | `2677e6bcd5149aa3f1e0f47199603695b240f723` |
| `spark` binary SHA-256 | `ebe7a7c8408bcd2d2aa492273aba8bc608c625e3e4c6c22831d014824c44f7d5` |
| Target | local `optimized-qwen` NVFP4 export |
| Target config SHA-256 | `267be2125ee2ec272555748c87cc636b25a96107946f05491bd6043151c7fe4e` |
| Target index SHA-256 | `f9ba0436d933e2362fb1bfa0508931c28b68f1fddbd5b94e6d50712e36a0636c` |
| Drafter | `onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3` |
| Drafter weights SHA-256 | `c15685d680bd58939689dcb4c344bb325efb75536df16d77c38517ce3df2dd6c` |
| Draft profile | NVFP4, gamma 15, full vocabulary 248320 |
| Target KV | NVFP4, zero high-precision layers |
| Context / batch | 8192 / one sequence |

The target is not byte-identical to the current monolithic upstream Unsloth
repository: it is a sharded Atlas-ready export with an MTP tensor and W3
sidecar. Copy the exact target directory from a validated host or archive;
substituting an upstream directory invalidates the performance comparison.

## End-to-end evidence

The fixed Weschera MinHeap request uses temperature zero, reasoning disabled,
400 output tokens, and one request at a time. The five-run promotion rates were
70.6411, 72.1877, 72.2574, 72.0858, and 72.1689 tok/s; median 72.1689.

A later ten-run gate against the unchanged process measured rates 72.9290,
72.6333, 72.6133, 70.9318, 72.4969, 70.7989, 72.3140, 72.2913,
72.1710, and 72.3280 tok/s. Median was 72.3210, mean 72.1507, standard
deviation 0.7121, and coefficient of variation 0.987%. Every response used
stable output SHA-256
`f51d8358ea2a5c63353ca00a29208ae2cccd3039b070043cad514cc4af9761c4`.

The five-run result SHA-256 is
`c3ca1279930cfa7492d8a0ffe38a44c208d331c1ea27f93a515fc3cace4e41db`;
the ten-run result SHA-256 is
`a2d6e304f42602e7a33cf7e1d4551f8945b7d5037603392d60fd3e86c5326e61`.
Both JSON records are publicly downloadable from the drafter repository.

Immediately before publishing this branch update, the tracked harness itself
passed a fresh uncontaminated five-run gate against the unchanged measured
server: 72.4800, 72.3395, 72.2242, 72.0106, and 71.5133 tok/s; median
72.2242. Output remained deterministic with the same stable hash. The local
validation JSON SHA-256 is
`1617a021965fe03b827b83af41f6e0631793ea7f913d4e741a6b0ae0fc479cc7`.

Output is deterministic for this fixed probe. Timing is repeatable within
normal runtime variance, not numerically deterministic. Every response ends at
the 400-token cap, so the result does not certify task completion.

## Local, VM, and Vast.ai reproduction

Clone and pin the branch on the execution host:

```bash
git clone --branch perf/qwen38-gb10-dflash \
  https://github.com/TheApathy/apathy-atlas.git
cd apathy-atlas
git rev-parse HEAD
```

Record that checkout revision with the result. The measured engine source was
`2677e6bcd5149aa3f1e0f47199603695b240f723`; later documentation/profile-only
commits do not change the Rust or CUDA source, but a future source change is a
new binary identity and must be requalified.

Download the public drafter on that host:

```bash
hf download onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3 \
  --local-dir "$PWD/models/drafter-v3"
sha256sum "$PWD/models/drafter-v3/model.safetensors"
```

Copy the exact target from Spark or Jenova, preserving sparse files, modes,
and timestamps. Verify the config and index hashes above after transfer:

```bash
rsync -aH --partial --info=progress2 \
  SOURCE_HOST:/path/to/optimized-qwen/ "$PWD/models/optimized-qwen/"
sha256sum "$PWD/models/optimized-qwen/config.json" \
  "$PWD/models/optimized-qwen/model.safetensors.index.json"
```

Build and serve:

```bash
export PATH=/usr/local/cuda/bin:$PATH
touch crates/atlas-kernels/build.rs
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server

MODEL_DIR="$PWD/models/optimized-qwen" \
DRAFT="$PWD/models/drafter-v3" \
./bench/qwen38-gb10/serve-v3-72tps.sh
```

The client can run on the same machine or through an SSH tunnel from the host:

```bash
ssh -N -L 8896:127.0.0.1:8896 VM_ALIAS
python3 bench/qwen38-gb10/weschera_minheap_repro.py \
  --endpoint http://127.0.0.1:8896/v1/chat/completions \
  --output /tmp/weschera-v3.json --repetitions 5 --max-tokens 400 \
  --store-output
```

For Vast.ai, use its generated SSH host/port as `VM_ALIAS` and transfer models
directly to the mounted persistent volume. Do not put Hugging Face tokens in a
template, command history, repository, or benchmark JSON.

The exact 72 tok/s number is a GB10 measurement using its `sm_121f` kernels and
unified-memory bandwidth. A generic Vast.ai H100/H200/A100 VM can exercise the
API and BF16/FP8 paths but cannot validate this GB10/NVFP4 number. Treat any
other GPU as a new hardware row and preserve its model, binary, request, output,
and environment identities before comparing rates.

## Quality boundary

NVFP4 target KV changed the deterministic target trajectory relative to BF16
and FP8 KV. It increased draft acceptance on this prompt, but is not lossless.
Use BF16 KV when reference output is required, and run separate multi-task and
long-context qualification before making NVFP4 KV a general serving default.
