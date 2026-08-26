# Qwen3.8-Flash-Next on one Spark

This profile serves the full 48-layer, 512-expert, 248077-token runtime
vocabulary, vision tower, and all 320M PLE rows on one GB10. It does not prune
experts, layers, PLE rows, or the vision tower.

## Public checkpoint provenance

The qualified base is RadixArk's NVFP4 revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`, converted with
`scripts/convert_qwen4_ple_offload.py`. The official FP8 comparison checkpoint
is pinned at `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`. Inferact's native NVFP4
MTP sidecar is pinned at `129972269565f7f4f664fdf8dd42268d3bbda9fd`.

The conversion moves only the enormous PLE embedding table into checksummed,
page-aligned NVFP4 sidecars. All 512 experts, all layers, the full vocabulary,
and all PLE rows remain present. The resulting resident logical weights are
83,955,875,834 bytes and the sparse PLE sidecars are 28,807,528,448 bytes.

## Build and serve

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server

MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-Offload \
  ./bench/qwen38-flash-next/serve.sh
```

The default 2048-token limit is the currently qualified exact-QSA ceiling; it
is a context cap, not model pruning. `PLE_CACHE_MB` controls the bounded
system-memory hot-page tier (default 512 MiB). Cold rows remain on NVMe and
only 16 selected rows are staged to the GPU per token.

## Qualification status

The first performance probe is Weschera's canonical MinHeap request at C=1,
temperature zero, reasoning disabled, and 400 output tokens. Target-only Atlas
with the PLE-boundary segmented CUDA graph measured 41.6410, 42.1010, and
42.3992 tok/s (median 42.1010), with identical stable output SHA-256
`f811cdc565dff063074f8bb1d0bb3fd55c8b10f3438c517cf4d59128f23cf790`.
The pre-graph eager control was 38.3594 tok/s median. Segmented graphs leave
layer 0 and sparse PLE row I/O/injection eager, then replay layers 1-47 plus
the terminal mixer and LM head. Set `ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS=0` only
for an eager diagnostic control.

MiaAI-Lab reports approximately 40 tok/s single-stream and 90 tok/s aggregate
for SGLang TP2 on two DGX Sparks. Atlas's 42.1010 result exceeds the published
single-stream number on one Spark under a deterministic output-hash gate; the
aggregate configurations are not equivalent and are not presented as an
aggregate win.

CPU prompt-lookup ngram speculation is not enabled by this launcher. Its
Flash-Next K=2 diagnostic was coherent but non-identical and slower (25.2982
tok/s median), so Atlas rejects it by default. The production PLE NVMe plus
system-memory ngram cache remains enabled; these are separate mechanisms.

The pinned Inferact native-MTP sidecar now passes the same output-hash gate.
K=2 verification uses exact multi-row hyperconnection projections and fused
MoE, producing the target-only stable SHA-256 above at 36.3819 tok/s in the
first 400-token run. This is up from the initial exact native-MTP result of
33.9389 tok/s, but still below target-only, so the production launcher does not
enable it by default. To reproduce the diagnostic, append:

```bash
--speculative \
--mtp-from-path /path/to/Qwen3.8-Flash-Next-Inferact-MTP \
--num-drafts 1 --mtp-vocab 248077
```

The dense Qwen3.8 V3 sidecar cannot directly share Flash-Next's 2560-wide
embedding and LM head. An explicit donor bridge was tested with proportional
capture-depth remapping and a 5120-wide residual slice. Its exact 64-token
diagnostic reached only 3.3200 tok/s with essentially zero draft acceptance,
so it is retained as an experimental compatibility probe and is not a
recommended launch path. DFlash2 remains shape-incompatible and fails closed.
A trained Flash-Next adapter or native Flash-Next drafter is required for a
real V3/DFlash2 speedup. No 70/80 tok/s claim is made until a qualified drafter
beats the target-only control under the same output-hash gate.
