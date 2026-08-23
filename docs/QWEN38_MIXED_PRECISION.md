# Qwen3.8 mixed-precision checkpoint loading

## Artifact audited

The loader changes were checked against:

`/home/flocka/atlas/qwen38/optimized-qwen-unsloth-ct/model.safetensors`

This is not a uniform NVFP4 file. Representative tensors in the active
safetensors header are:

| Role | Weight dtype / shape | Scale dtype / shape |
|---|---|---|
| layer 3 attention `q_proj` | FP8 E4M3 `[12288, 5120]` | FP32 `[12288, 1]` |
| layer 0 GDN `in_proj_qkv` | FP8 E4M3 `[10240, 5120]` | FP32 `[10240, 1]` |
| layer 0 FFN `gate_proj` | compressed NVFP4 `weight_packed` | compressed-tensors metadata |
| layer 63 FFN `gate_proj` | FP8 E4M3 `[17408, 5120]` | FP32 `[17408, 1]` |
| `lm_head` | FP8 E4M3 `[248320, 5120]` | FP32 `[248320, 1]` |

Therefore each projection must be classified from its own keys. The
checkpoint-wide compressed-tensors declaration is insufficient: FP8 islands
need FP8-to-BF16 dequantization followed by the existing runtime NVFP4 path.
The LM head must be expanded to BF16 before its BF16 consumer; passing the raw
one-byte FP8 allocation to that consumer causes an out-of-bounds two-byte read.

## MTP artifact caveat

The active `model.safetensors` header contains 1,953 tensors and no `mtp.*`
keys. The active `model.safetensors.index.json` also excludes MTP. A separate
`model.safetensors.index.json.with-mtp` references 15 `mtp.*` keys in
`model_mtp.safetensors`, but the sidecar currently exists only as
`model_mtp.safetensors.bak`.

Consequently, a target configuration with `mtp_layers = 1` does not describe
the active artifact. Restore and validate the sidecar/index pairing before
claiming MTP results; do not interpret an MTP-disabled load as proof that the
MTP path works.

## Validation boundary

Pure metadata/layout tests and CPU compilation cover the dispatch decisions.
The large dequantizations, allocation lifetime, numerical output, and final
throughput still require a locked B200 run with the exact artifact. Vision,
video, and target-resolution changes are deliberately outside this port.
