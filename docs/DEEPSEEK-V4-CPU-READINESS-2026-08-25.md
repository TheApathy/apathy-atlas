# DeepSeek V4 CPU readiness — 2026-08-25

This is the exact handoff for the `apathy-deepseek` lane. No GPU training,
inference, capture, or benchmark was started during this hardening pass.

## Ready offline

- The canonical 24,846-row corpus is restored and pinned at SHA-256
  `2824835f81288541eaa6a97362cd7e308e6f7f80c001d8a871860506f15f1bde`.
  The truncated 24,843-row copy is preserved under
  `/workspace/deepseek-dflash2-backups/20260825-corpus-tail/`.
- Capture, validation, and training select the same deterministic 128 usable
  rows after shuffle, tokenization, and loss-token filtering. The CPU dry-run
  found 128 unique rows, 361–1,736 tokens, and a 40.00 GiB hidden-tensor cost.
  Their shared preprocessing cache key hashes corpus and tokenizer contents,
  preprocessing code, length, template, and formatting mode, so replacing a
  file in place cannot silently reuse rows from the previous corpus. A fresh
  128-row CPU dry-run produced cache key
  `63fc6b90aec0dd60e131910c6ffde02958ca6d3a5afc518760efde817f8fe349`
  and reproduced 128 unique rows at 361–1,736 tokens and 40.00 GiB projected.
- The compact embedding and LM-head components match their original tensors by
  streaming value hash. The report is shipped as
  `target-components/component-parity.json`; it also pins tokenizer and
  generation metadata, the preserved DeepSeek target config, and the compact
  Qwen loader config ABI.
- The launcher rejects corpus drift, missing or malformed BF16 components,
  incomplete hidden tensors, unpatched SpecForge, and incompatible final
  checkpoints. `PREFLIGHT_ONLY=1` exits before credit access or `torchrun`;
  paid execution additionally requires `CREDIT_GUARD_CONFIRM=1`.
  Final checkpoint validation cross-checks every indexed tensor against its
  declared shard and records SHA-256 for the config, index, and weight shards.
  The launcher independently verifies the final bundle, requires at least 128
  declared hidden tensors, and binds its manifest hash into the run contract.
- The pre-capture Vast bundle verifies as 280 files and 2,217,662,913 bytes
  beneath `/workspace/deepseek-dflash2`; its manifest SHA-256 is
  `1010ec482619052c561c2abef9ab911b8441d5fd99e65822fb3f6398fded82fe`.
  It must be rebuilt with the captured hidden directory before training.
- The 1M context plan pins config, tokenizer, and prompt hashes and reserves
  generation headroom inside the declared 1,048,576-token YaRN window.
- The release gate separates reasoning and content, hashes exact output, binds
  model and implementation identities, and accepts only fresh server-log
  acceptance records from the measured run.
- The persistent expert-major work ABI is shared production code with CPU tests
  for layout, encoding, metadata, routing shape, duplicate experts, and bounds.
  It is not wired into production dispatch yet.

## Vast state and storage

At the last read-only check, the instance was reachable and the inherited Qwen
processes had exited; no vLLM, torchrun, DFlash trainer, or pair generator was
running. Vast credit was $144.77 and the volume had 42 GB free. This does not
authorize starting a DeepSeek GPU workload. Recheck ownership, free VRAM,
credit, disk, and the bundle immediately before a paid action.

Only 42 GB was free at the final check while twelve old Qwen checkpoints occupied
`/workspace/out` and `/workspace/out-v7`. The non-destructive plan at
`/workspace/checkpoint-prune-plan.json` identifies 212,869,128,140 bytes of old
checkpoint candidates while retaining each directory's final step. Nothing was
deleted. Review and explicitly authorize every listed path before reclaiming
space; capture plus a new checkpoint cannot safely fit in the observed free
space.

## Work that still requires the GPU

1. Re-run all preflights against the unchanged bundle, then capture the selected
   128 rows from an isolated plain target server and validate every tensor.
2. Train DFlash2, retain the exact run contract and logs, and pass the final
   checkpoint ABI validator. The saved Qwen drafter at
   `/workspace/out-v7/epoch_2_step_3784` is not a DeepSeek checkpoint.
3. Measure the persistent-worklist kernel and require at least 213 GB/s plus
   exact per-row parity before production dispatch is enabled.
4. Run plain and DFlash2 release arms separately with locked identities. Promote
   only if exactness/quality gates pass and median single-stream decode reaches
   the stated target; do not infer 65 tok/s from CPU readiness.
5. Run the 8K/128K/250K/512K/1M retrieval sweep and report TTFT, decode speed,
   output hashes, and retrieval success. Capacity alone does not qualify 1M.
