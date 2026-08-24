#!/usr/bin/env bash
# Train the Atlas-native DeepSeek-V4 DFlash2 head on the retained Vast H200.
#
# This launcher intentionally handles only the paid training phase. Target
# hidden states must already have been captured by Atlas and converted to the
# SpecForge keyed offline cache. It refuses to start an online HF teacher: the
# serving checkpoint uses Atlas-packed names/kernels and is not an HF teacher.
set -euo pipefail

W="${DFLASH_WORKSPACE:-/workspace/deepseek-dflash2}"
SF="${SPECFORGE_DIR:-/workspace/SpecForge}"
TARGET="${TARGET_COMPONENTS_DIR:-$W/target-components}"
CONFIG="${DRAFT_CONFIG:-$W/deepseek-v4-dflash2.json}"
CORPUS="${TRAIN_CORPUS:-$W/corpus.jsonl}"
HIDDEN="${OFFLINE_HIDDEN_DIR:-$W/hidden}"
OUT="${OUTPUT_DIR:-$W/out}"
CACHE="${CACHE_DIR:-$W/cache}"
EPOCHS="${EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
ACCUM="${ACCUMULATION_STEPS:-4}"
STOP_FLOOR="${VAST_STOP_FLOOR:-4.00}"

die() { echo "FATAL: $*" >&2; exit 1; }
need_file() { [[ -s "$1" ]] || die "missing required file: $1"; }

need_file "$SF/scripts/train_dflash.py"
need_file "$CONFIG"
need_file "$CORPUS"
need_file "$TARGET/config.json"
need_file "$TARGET/tokenizer.json"
need_file "$TARGET/model.safetensors.index.json"
[[ -d "$HIDDEN" ]] || die "offline hidden directory is absent: $HIDDEN"

python3 - "$CONFIG" "$TARGET" "$HIDDEN" <<'PY'
import json, pathlib, sys
from safetensors import safe_open

config_path, target_dir, hidden_dir = map(pathlib.Path, sys.argv[1:])
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
        actual = list(handle.get_slice(key).get_shape())
    if actual != shape:
        raise RuntimeError(f"{key} is {actual}, expected {shape}")

hidden = list(hidden_dir.glob("*.pt"))
if len(hidden) < 128:
    raise RuntimeError(
        f"only {len(hidden)} offline hidden rows; refuse paid training below 128"
    )
print(f"preflight OK: {len(hidden)} hidden rows")
PY

# The patched training driver must auto-detect Atlas's embed.weight/head.weight
# keys. Refuse an unpatched Qwen-only driver before allocating the GPU.
grep -q 'SPECFORGE_TARGET_EMBED_KEY' "$SF/scripts/train_dflash.py" \
  || die "SpecForge driver lacks configurable target component key support"

mkdir -p "$OUT" "$CACHE"
cat > "$OUT/run-contract.json" <<EOF
{"epochs":$EPOCHS,"max_length":$MAX_LENGTH,"accumulation_steps":$ACCUM,"stop_floor":$STOP_FLOOR,"target_layers":[1,11,22,32,43],"block_size":16}
EOF

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPECFORGE_PAD_TO="$MAX_LENGTH"
export SPECFORGE_SKIP_MISSING_HIDDEN=0
export SPECFORGE_TARGET_EMBED_KEY=embed.weight
export SPECFORGE_TARGET_LM_HEAD_KEY=head.weight

exec torchrun --standalone --nproc_per_node 1 "$SF/scripts/train_dflash.py" \
  --target-model-path "$TARGET" --target-model-backend hf \
  --offline-hidden-dir "$HIDDEN" --draft-config-path "$CONFIG" \
  --block-size 16 --num-draft-layers 3 --mask-token-id 129000 \
  --attention-backend flex_attention --trust-remote-code \
  --num-anchors 128 --loss-decay-gamma 7.0 \
  --train-data-path "$CORPUS" --chat-template deepseek --is-preformatted \
  --num-epochs "$EPOCHS" --batch-size 1 --learning-rate 1e-4 \
  --max-length "$MAX_LENGTH" --warmup-ratio 0.04 --max-grad-norm 1.0 \
  --accumulation-steps "$ACCUM" --seed 42 --dataloader-num-workers 2 \
  --output-dir "$OUT" --cache-dir "$CACHE" --save-interval 500 \
  --tp-size 1 --report-to none
