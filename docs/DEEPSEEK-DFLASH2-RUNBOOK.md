# DeepSeek DFlash2 Runbook — staged for jenova

Staged: 2026-08-25. Branch: `apathy-deepseek` @ `a40ed27e` (CPU-readiness handoff).
Driver host: **jenova** (no GPU, 768 GB free — orchestration only).
GPU host: **Vast H200 NVL instance `48572428`** ($3.844/hr, 143,771 MiB VRAM).

> All commands run from an `apathy-deepseek` checkout unless noted.
> The launcher is fail-closed: it refuses paid execution without
> `CREDIT_GUARD_CONFIRM=1` and an armed credit stop-guard.

## 0. Preconditions (check every time, before ANY paid action)

```bash
# Credit + instance state (from driver host)
vastai show instance 48572428 --raw | jq '.actual_status, .gpu_name'
./qwen38/vast/balance-guard.sh check   # or: vastai show limits

# On-instance disk + no co-tenant GPU jobs
ssh <h200> 'df -h /workspace; nvidia-smi --query-gpu=memory.used --format=csv,noheader'

# Corpus integrity (must match pinned SHA)
ssh <h200> 'sha256sum /workspace/corpus/v3_corpus.jsonl'  # expect 2824835f…f1bde
```

Reference docs on the branch:
- `docs/DEEPSEEK-V4-CPU-READINESS-2026-08-25.md` — full handoff state
- `docs/VAST-DEEPSEEK-DFLASH2-TRAINING.md` — ABI + launch gate details

## Known-good state facts (as of staging)

| Item | Value |
|---|---|
| Corpus | 24,846 rows, SHA-256 `2824835f81288541eaa6a97362cd7e308e6f7f80c001d8a871860506f15f1bde` |
| Selected rows | Deterministic 128 rows post-shuffle/filter (361–1,736 tokens, ~40 GiB hidden tensors) |
| CPU dry-run cache key | `63fc6b90aec0dd60e131910c6ffde02958ca6d3a5afc518760efde817f8fe349` |
| Pre-capture bundle | `/workspace/deepseek-dflash2`, 280 files / 2,217,662,913 bytes, manifest SHA `1010ec482619052c561c2abef9ab911b8441d5fd99e65822fb3f6398fded82fe` |
| Target components (local) | `/home/flocka/models/DeepSeek-V4-DFlash2-target-components` (2.0 GiB) |
| Serving target (local copy) | `/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1` (79 GB) |
| H200 free disk at staging | 180 GB (old-checkpoint pressure resolved) |

## Hard ABI (do not deviate)

- hidden size 4096 · vocab 129280 · layers 43
- capture-layer IDs `[1,11,22,32,43]` (HF hidden-state IDs)
- block size / verify width 16
- tokenizer + mask token declared in exported drafter config
- capture-layer IDs must equal `fc.weight` input width agreement exactly
- **No warm-start from the Qwen drafter — incompatible in every dimension.**

## DS1 — Capture (GPU, ~1–2h)

The serving checkpoint is Atlas-packed and NOT HF-loadable by SpecForge;
hidden states must come from the **Atlas target-serving path**, then be
converted to SpecForge keyed offline cache.

```bash
# 1. Serve the plain EXL3-K2 target with the atlas binary from this branch
#    (isolated port, no drafter attached). Use the branch-built binary:
#    target/release/spark serve … (mirror the flags used for qwen arms,
#    minus any --dflash/--draft-model flags).

# 2. Capture + convert (resumable):
python3 scripts/capture-deepseek-dflash2-offline.py \
  --specforge-dir /workspace/SpecForge \
  --target-components /workspace/deepseek-dflash2/target-components \
  --corpus /workspace/corpus/v3_corpus.jsonl \
  --dump /workspace/deepseek-dflash2/capture.dump \
  --hidden-dir /workspace/deepseek-dflash2/hidden \
  --cache-dir /workspace/deepseek-dflash2/cache \
  --url http://127.0.0.1:<port>/v1 \
  --resume
# add --dry-run first if anything about layout is uncertain (~free)

# 3. Validate every captured tensor, then REBUILD the bundle so the manifest
#    includes the hidden directory (the staged pre-capture manifest is invalid
#    for training):
python3 scripts/deepseek-dflash2-bundle.py --help   # check exact rebuild flags
```

Gate to pass before DS2: ≥128 keyed hidden rows, BF16 dtype,
`[padded_tokens, 20480]` shapes verified by the launcher's own preflight.

## DS2 — Train DFlash2 (GPU, est. $10–20)

```bash
# From driver host BEFORE launching: arm the stop guard (floor $4):
python3 scripts/vast-credit-guard.py 48572428 --floor 4.00 --interval 60 --arm

# Dry pass — allocates nothing, exits before torchrun:
PREFLIGHT_ONLY=1 bash scripts/train-deepseek-dflash2-vast.sh

# Paid run (guard must be armed; env confirms intent):
CREDIT_GUARD_CONFIRM=1 bash scripts/train-deepseek-dflash2-vast.sh
```

Notes:
- SpecForge snapshot MUST contain the safe single-GPU teardown patch
  (`3rdparty_patches/specforge/safe_single_gpu_teardown.patch`) or the
  launcher refuses — this is the same duplicate-destroy bug that "failed"
  after v8's successful save.
- After training: `scripts/validate-deepseek-dflash2-checkpoint.py` on the
  final bundle. Keep run contract + logs.
- `/workspace/out-v7/epoch_2_step_3784` is a QWEN checkpoint — not related.

## DS3 — Persistent-worklist kernel gate

Measure the kernel; require **≥213 GB/s AND exact per-row parity** vs the
reference path. Only then may production dispatch be enabled. (Test entry
points live under the branch's kernel tests; CPU tests cover layout/routing
already.)

## DS4 — Release arms (plain vs DFlash2)

Run both arms separately with locked model + implementation identities.
Promote **only if**: exactness/quality gates pass AND median single-stream
decode reaches the stated target. Do NOT infer tok/s from CPU readiness.

## DS5 — Long-context sweep

8K / 128K / 250K / 512K / 1M retrieval sweep. Report TTFT, decode speed,
output hashes, retrieval success per length. Capacity alone does not qualify
1M — YaRN window is 1,048,576; generation headroom must stay inside it.

## Qwen-lane interaction (read before scheduling)

- The H200 runs the Qwen 60k corpus regen until ~09:00 UTC 2026-08-26.
  A local watcher (`/tmp/v9-watch.sh`, log `/tmp/v9-watch.log` on reaper)
  stops the instance and pulls the corpus when regen hits 59k rows.
- Restarting the instance after that stop changes its SSH port — re-read via
  `vastai ssh-url 48572428`.
- Budget reality at staging: credit bottoms out as the Qwen regen finishes.
  **DS1 cannot start until more Vast credit lands.**
