# Mia DeepSeek reproduction on one Spark

This branch pins the public recipe at
`tpurtell/ds4-mia-exl3-k2-1spark@f20b97dfd7666c00c316f29542e2e53f33cabb19`.
The runtime image was rebuilt locally without allocating the GPU:

```text
image: sha256:0bc0d918acdce3078690f51afa0a8e63a5eee4d1e52a39c08e4059bc3a25184f
size:  18,922,435,700 bytes
```

Its build-time verification passed for Mia vLLM 0.25.2.dev, the EXL3 loader,
InstantTensor, the pinned B12X tree, and selected DeepSeek backports. The final
published GHCR manifest was unavailable during the audit, so this local image
is a source reproduction, not a byte-identical claim about the published
image.

Rebuild without using the GPU:

```bash
bash scripts/build-mia-deepseek-runtime.sh
```

When the GPU is intentionally free, launch the local K2-v1 checkpoint:

```bash
bash scripts/mia-deepseek-serve.sh --allow-gpu
docker logs -f atlas-mia-deepseek-k2
```

The launcher refuses to remove an existing container and refuses a busy GPU.
It defaults to one million maximum tokens, NVFP4 DS-MLA KV, five probabilistic
DSpark proposals, six maximum sequences, and thinking off.

After `/health` is ready, run the locked concurrency-one content sweep:

```bash
python3 bench/deepseek-v4/mia_decode_sweep.py \
  --output mia-decode-sweep.json
```

The valid upstream reference point is 58.1 tok/s for K2-v1's controlled short
decode. Content medians vary substantially; aggregate C6 throughput and the
incorrect repeated-token case are not substitutes for single-stream decode.

## Atlas port order

1. Reproduce Mia with the same local checkpoint and locked prompts.
2. Compare per-step proposal, verifier, acceptance, and dispatch counters.
3. Port the B12X/Trellis small-M expert-major verifier behavior, keeping exact
   output parity as a hard gate.
4. Fuse or graph DSpark projection and attention launches.
5. Add probabilistic rejection only with an explicit quality comparison.
6. Train DeepSeek-native DFlash2 after runtime costs are competitive; draft
   quality is not the current primary bottleneck.
