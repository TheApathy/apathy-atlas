# DeepSeek V4 Flash K2 on one Spark

Atlas branch: `apathy-deepseek`.

This lane ports the standard Hugging Face EXL3 K2 checkpoint into Atlas's
native DeepSeek-V4 loader and CUDA runtime. It does not wrap the Mia container.
The upstream recipe used for the checkpoint and benchmark contract is
`tpurtell/ds4-mia-exl3-k2-1spark` at
`f20b97dfd7666c00c316f29542e2e53f33cabb19`.

## Frozen target

- Model: `wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1`
- Revision: `68eaca43e99bfbfd697a5559c7796b983deb38f8`
- Layout: homogeneous 2-bit EXL3, 256 routed experts, 43 layers
- Context: checkpoint-native YaRN, `max_position_embeddings = 1048576`
- Draft: embedded five-token DSpark block

Download and launch:

```bash
scripts/download-deepseek-v4-k2.sh

ATLAS_TARGET_MODEL=deepseek-v4-flash \
ATLAS_TARGET_QUANT=nvfp4 \
CUDA_HOME=/usr/local/cuda \
CUDARC_CUDA_VERSION=13000 \
cargo build --release -p spark-server

GAMMA=5 scripts/exl3-serve.sh k2
```

Plain serving exposes the checkpoint-native 1M YaRN ceiling with paged-KV
overcommit. The resident pool is smaller and must be reported from the boot
log. Speculative serving currently defaults to 131072 because DSpark's target
capture is an absolute-position BF16 buffer (`3 * max_seq_len * hidden`); at
1M that buffer alone is 24.576 GB. A windowed capture ring is required before
claiming speculative 1M.

The speculative launcher enables the existing NVFP4 attention residency
profile to fit the target, embedded drafter, and useful KV together. That
transcode is lossy and its output must be quality-gated separately from the
plain EXL3 baseline.

The loader accepts both the original Atlas `w1/w2/w3.rank0.*` tensor names
and standard HF `gate_proj/up_proj/down_proj.*` names. Trellis tile width
selects K2 (`32` I16 words) or K3 (`48` I16 words). The current fused runtime
requires a homogeneous bitrate across experts; mixed K2.1 fails closed rather
than decoding a K3 expert through K2 geometry.

## Correctness gate

```bash
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=13000 \
cargo run -p spark-model --example exl3_gemv_microtest \
  --features 'cuda gpu-examples'
```

The first gate is a bit-exact K2 dequant comparison against the CPU reference.
The remaining gates retain the K3 GEMV, prefill, fused-decode, determinism, and
m-row verification coverage.

## Benchmark contract

Record target revision, Atlas commit, prompt-token count, output-token count,
concurrency, prefix-cache state, thinking mode, TTFT, prefill tok/s, visible
decode tok/s, aggregate tok/s, and an output hash. Do not promote a synthetic
short-prompt decode rate as a varied-content average. The upstream reference
reported 58.1 tok/s for its controlled K2 C1 case and materially lower rates
on prose; Atlas numbers must be measured independently.
