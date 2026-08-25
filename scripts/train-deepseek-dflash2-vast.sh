#!/usr/bin/env bash
# Train the Atlas-native DeepSeek-V4 DFlash2 head on the retained Vast H200.
#
# This launcher intentionally handles only the paid training phase. Target
# hidden states must already have been captured by Atlas and converted to the
# SpecForge keyed offline cache. It refuses to start an online HF teacher: the
# serving checkpoint uses Atlas-packed names/kernels and is not an HF teacher.
set -euo pipefail

W="${DFLASH_WORKSPACE:-/workspace/deepseek-dflash2}"
SF="${SPECFORGE_DIR:-$W/SpecForge}"
TARGET="${TARGET_COMPONENTS_DIR:-$W/target-components}"
CONFIG="${DRAFT_CONFIG:-$W/deepseek-v4-dflash2.json}"
CORPUS="${TRAIN_CORPUS:-$W/corpus.jsonl}"
HIDDEN="${OFFLINE_HIDDEN_DIR:-$W/hidden}"
OUT="${OUTPUT_DIR:-$W/out}"
CACHE="${CACHE_DIR:-$W/cache}"
EPOCHS="${EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
ACCUM="${ACCUMULATION_STEPS:-4}"
IS_PREFORMATTED="${IS_PREFORMATTED:-0}"
TRAIN_ROWS="${TRAIN_ROWS:-128}"
EXPECTED_CORPUS_SHA256="${EXPECTED_CORPUS_SHA256:-2824835f81288541eaa6a97362cd7e308e6f7f80c001d8a871860506f15f1bde}"
STOP_FLOOR="${VAST_STOP_FLOOR:-4.00}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "FATAL: $*" >&2; exit 1; }
need_file() { [[ -s "$1" ]] || die "missing required file: $1"; }

need_file "$SF/scripts/train_dflash.py"
need_file "$CONFIG"
need_file "$CORPUS"
need_file "$TARGET/config.json"
need_file "$TARGET/tokenizer.json"
need_file "$TARGET/model.safetensors.index.json"
need_file "$SCRIPT_DIR/validate-deepseek-dflash2-offline.py"
need_file "$SCRIPT_DIR/validate-deepseek-dflash2-checkpoint.py"
[[ -d "$HIDDEN" ]] || die "offline hidden directory is absent: $HIDDEN"
case "$IS_PREFORMATTED" in
  0) FORMAT_ARGS=() ;;
  1) FORMAT_ARGS=(--is-preformatted) ;;
  *) die "IS_PREFORMATTED must be 0 or 1" ;;
esac
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] \
  || die "PREFLIGHT_ONLY must be 0 or 1"
[[ "$TRAIN_ROWS" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_ROWS must be a positive integer"
(( TRAIN_ROWS >= 128 )) || die "TRAIN_ROWS must be at least 128"
[[ "$EXPECTED_CORPUS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "EXPECTED_CORPUS_SHA256 must be a lowercase SHA-256 digest"
ACTUAL_CORPUS_SHA256="$(sha256sum "$CORPUS" | awk '{print $1}')"
[[ "$ACTUAL_CORPUS_SHA256" == "$EXPECTED_CORPUS_SHA256" ]] \
  || die "corpus SHA-256 mismatch: got $ACTUAL_CORPUS_SHA256 expected $EXPECTED_CORPUS_SHA256"

python3 - "$CONFIG" "$TARGET" "$HIDDEN" "$TRAIN_ROWS" <<'PY'
import json, pathlib, sys
from safetensors import safe_open

config_path, target_dir, hidden_dir = map(pathlib.Path, sys.argv[1:4])
train_rows = int(sys.argv[4])
cfg = json.loads(config_path.read_text())
assert cfg["hidden_size"] == 4096
assert cfg["vocab_size"] == 129280
assert cfg["num_target_layers"] == 43
assert cfg["block_size"] == 16
assert cfg["dflash_config"]["target_layer_ids"] == [1, 11, 22, 32, 43]
assert cfg["dflash_config"]["mask_token_id"] == 129000

index = json.loads((target_dir / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
for key, shape in (("embed.weight", [129280, 4096]), ("head.weight", [129280, 4096])):
    shard = target_dir / weight_map[key]
    if not shard.is_file():
        raise RuntimeError(f"missing target component shard {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(key)
        actual = list(tensor.get_shape())
        dtype = str(tensor.get_dtype())
    if actual != shape or dtype != "BF16":
        raise RuntimeError(f"{key} is shape={actual} dtype={dtype}, expected {shape} BF16")

hidden = list(hidden_dir.glob("*.pt"))
if len(hidden) < train_rows:
    raise RuntimeError(
        f"only {len(hidden)} offline hidden rows; need at least {train_rows}"
    )
print(f"preflight OK: {len(hidden)} hidden rows")
PY

python3 "$SCRIPT_DIR/validate-deepseek-dflash2-offline.py" \
  --specforge-dir "$SF" --draft-config "$CONFIG" \
  --target-components "$TARGET" --corpus "$CORPUS" \
  --hidden-dir "$HIDDEN" --cache-dir "$CACHE" \
  --max-length "$MAX_LENGTH" --min-rows "$TRAIN_ROWS" \
  --limit "$TRAIN_ROWS" \
  --chat-template deepseek-v3 "${FORMAT_ARGS[@]}"

# The patched training driver must auto-detect Atlas's embed.weight/head.weight
# keys. Refuse an unpatched Qwen-only driver before allocating the GPU.
grep -q 'SPECFORGE_TARGET_EMBED_KEY' "$SF/scripts/train_dflash.py" \
  || die "SpecForge driver lacks configurable target component key support"
grep -q -- '--max-train-rows' "$SF/scripts/train_dflash.py" \
  || die "SpecForge driver lacks deterministic training-row selection"
grep -q 'deepseek-dflash2-preprocess-v2' "$SF/scripts/train_dflash.py" \
  || die "SpecForge driver lacks content-addressed preprocessing cache"
grep -q 'ATLAS_SAFE_SINGLE_GPU_TEARDOWN' "$SF/specforge/distributed.py" \
  || die "SpecForge lacks safe single-GPU process-group teardown"

mkdir -p "$OUT" "$CACHE"
cat > "$OUT/run-contract.json" <<EOF
{"epochs":$EPOCHS,"max_length":$MAX_LENGTH,"train_rows":$TRAIN_ROWS,"corpus_sha256":"$ACTUAL_CORPUS_SHA256","accumulation_steps":$ACCUM,"is_preformatted":$IS_PREFORMATTED,"stop_floor":$STOP_FLOOR,"target_layers":[1,11,22,32,43],"block_size":16}
EOF

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPECFORGE_PAD_TO="$MAX_LENGTH"
export SPECFORGE_SKIP_MISSING_HIDDEN=0
export SPECFORGE_TARGET_EMBED_KEY=embed.weight
export SPECFORGE_TARGET_LM_HEAD_KEY=head.weight

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "CPU preflight complete; PREFLIGHT_ONLY=1, refusing to start torchrun"
  exit 0
fi

[[ "${CREDIT_GUARD_CONFIRM:-0}" == 1 ]] \
  || die "credit guard is not confirmed; run scripts/vast-credit-guard.py --arm and set CREDIT_GUARD_CONFIRM=1"

torchrun --standalone --nproc_per_node 1 "$SF/scripts/train_dflash.py" \
  --target-model-path "$TARGET" --target-model-backend hf \
  --offline-hidden-dir "$HIDDEN" --draft-config-path "$CONFIG" \
  --block-size 16 --num-draft-layers 3 --mask-token-id 129000 \
  --attention-backend flex_attention --trust-remote-code \
  --num-anchors 128 --loss-decay-gamma 7.0 \
  --train-data-path "$CORPUS" --chat-template deepseek-v3 "${FORMAT_ARGS[@]}" \
  --max-train-rows "$TRAIN_ROWS" \
  --num-epochs "$EPOCHS" --batch-size 1 --learning-rate 1e-4 \
  --max-length "$MAX_LENGTH" --warmup-ratio 0.04 --max-grad-norm 1.0 \
  --accumulation-steps "$ACCUM" --seed 42 --dataloader-num-workers 2 \
  --output-dir "$OUT" --cache-dir "$CACHE" --save-interval 500 \
  --tp-size 1 --report-to none

LATEST="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*_step_*' -print \
  | sort -V | tail -n 1)"
[[ -n "$LATEST" ]] || die "training returned success but produced no checkpoint"
python3 "$SCRIPT_DIR/validate-deepseek-dflash2-checkpoint.py" "$LATEST" \
  --report "$OUT/final-checkpoint-abi.json"
echo "training and checkpoint ABI validation complete: $LATEST"
