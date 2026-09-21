#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANONICAL="$ROOT/bench/qwen38-gb10/serve-dflash2-official.sh"
TMP="$(mktemp -d)"
cleanup() {
  find "$TMP" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf -- "$TMP"
}
trap cleanup EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

expect_fail() {
  local needle="$1"
  shift
  local output
  if output="$("$@" 2>&1)"; then
    fail "command unexpectedly succeeded: $*"
  fi
  [[ "$output" == *"$needle"* ]] || fail "missing failure '$needle': $output"
}

sha() { sha256sum "$1" | awk '{print $1}'; }

pin() {
  local file="$1" old="$2" new="$3"
  python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1:]
data = open(path, encoding="utf-8").read()
if data.count(old) != 1:
    raise SystemExit(f"expected one pin occurrence for {old!r}, got {data.count(old)}")
with open(path, "w", encoding="utf-8") as handle:
    handle.write(data.replace(old, new))
PY
}

TARGET="$TMP/target"
DRAFT="$TMP/draft"
mkdir -p "$TARGET" "$DRAFT" "$TMP/sessions"
printf '%s\n' '{"target":"fixture"}' >"$TARGET/config.json"
printf '%s\n' '{"tokenizer":"fixture"}' >"$TARGET/tokenizer.json"
printf '%s\n' '{"tokenizer_config":"fixture"}' >"$TARGET/tokenizer_config.json"
printf '%s\n' 'fixture chat template' >"$TARGET/chat_template.jinja"
printf '%s\n' '{"eos_token_id":248044}' >"$TARGET/generation_config.json"
printf '%s\n' '{"quantization":"fixture"}' >"$TARGET/hf_quant_config.json"
TARGET_WEIGHTS=(
  model-00001-of-00006.safetensors
  model-00002-of-00006.safetensors
  model-00003-of-00006.safetensors
  model-00004-of-00006.safetensors
  model-00005-of-00006.safetensors
  model-00006-of-00006.safetensors
  nvfp4_experts_mtp.safetensors
)
for weight in "${TARGET_WEIGHTS[@]}"; do
  printf '%s\n' "$weight" >"$TARGET/$weight"
done
python3 - "$TARGET/model.safetensors.index.json" "${TARGET_WEIGHTS[@]}" <<'PY'
import json
import sys
names = sys.argv[2:]
value = {
    "metadata": {"total_size": sum(len(name) + 1 for name in names)},
    "weight_map": {f"fixture.{index}": name for index, name in enumerate(names)},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
PY

python3 - "$DRAFT/config.json" "$DRAFT/model.safetensors" <<'PY'
import json
import struct
import sys
config = {
    "architectures": ["DFlash2DraftModel"],
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
    "dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 262144,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "sliding_window": 2048,
    "use_sliding_window": True,
    "vocab_size": 248320,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, sort_keys=True, separators=(",", ":"))
header = {}
offset = 0
for index in range(81):
    header[f"fixture.tensor.{index:02d}"] = {
        "dtype": "BF16",
        "shape": [1],
        "data_offsets": [offset, offset + 2],
    }
    offset += 2
raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
with open(sys.argv[2], "wb") as handle:
    handle.write(struct.pack("<Q", len(raw)))
    handle.write(raw)
    handle.write(b"\0" * offset)
PY

FAKE="$TMP/fake-spark"
CAPTURE_ARGS="$TMP/captured-args.json"
CAPTURE_ENV="$TMP/captured-env.json"
python3 - "$FAKE" "$CAPTURE_ARGS" "$CAPTURE_ENV" <<'PY'
import os
import sys
body = f'''#!/usr/bin/python3 -I
import json
import os
import sys
with open({sys.argv[2]!r}, "w", encoding="utf-8") as handle:
    json.dump(sys.argv, handle, sort_keys=True, separators=(",", ":"))
with open({sys.argv[3]!r}, "w", encoding="utf-8") as handle:
    json.dump(dict(os.environ), handle, sort_keys=True, separators=(",", ":"))
'''
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(body)
os.chmod(sys.argv[1], 0o555)
PY

for file in "$TARGET"/* "$DRAFT"/*; do chmod 0444 "$file"; done
PTX_SHA="71d95d7815d36cc0df61070d125b1598464a67aaaac44425f5f4a0ba6d6e61c2"
ABI="$TMP/abi.json"

normalized_launcher_sha() {
  python3 - "$1" <<'PY'
import hashlib
import re
import sys
raw = open(sys.argv[1], "rb").read()
pattern = rb'(?m)^EXPECTED_ABI_ATTESTATION_SHA256 = "[^"]+"$'
replacement = b'EXPECTED_ABI_ATTESTATION_SHA256 = "UNRELEASED_POST_OFFICIAL_DFLASH2_FIX_ATTESTATION"'
normalized, count = re.subn(pattern, replacement, raw)
assert count == 1
print(hashlib.sha256(normalized).hexdigest())
PY
}

write_abi() {
  local host_count="$1" launcher_sha="$2"
  [[ ! -e "$ABI" ]] || chmod 0644 "$ABI"
  python3 - "$ABI" "$FAKE" "$host_count" "$PTX_SHA" "$launcher_sha" <<'PY'
import hashlib
import json
import os
import sys
binary_sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
value = {
    "binary_sha256": binary_sha,
    "host_argument_count": int(sys.argv[3]),
    "launcher_normalized_sha256": sys.argv[5],
    "ptx_parameter_count": 13,
    "ptx_sha256": sys.argv[4],
    "schema": "atlas-qwen38-attention-abi-v2",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
os.chmod(sys.argv[1], 0o444)
PY
}

HEADER_LENGTH="$(python3 - "$DRAFT/model.safetensors" <<'PY'
import struct
import sys
with open(sys.argv[1], "rb") as handle:
    print(struct.unpack("<Q", handle.read(8))[0])
PY
)"
HEADER_SHA="$(python3 - "$DRAFT/model.safetensors" <<'PY'
import hashlib
import struct
import sys
with open(sys.argv[1], "rb") as handle:
    length = struct.unpack("<Q", handle.read(8))[0]
    print(hashlib.sha256(handle.read(length)).hexdigest())
PY
)"

make_profile() {
  local session_template="$1" host_count="${2:-13}"
  PROFILE="$TMP/profile-$RANDOM-$RANDOM.sh"
  cp "$CANONICAL" "$PROFILE"
  chmod 0644 "$PROFILE"
  pin "$PROFILE" '/home/flocka/atlas/qwen38/dflash2-official/spark' "$FAKE"
  pin "$PROFILE" '/home/flocka/atlas/qwen38/optimized-qwen' "$TARGET"
  pin "$PROFILE" '/home/flocka/atlas/qwen38/drafter-dflash2-incoai' "$DRAFT"
  pin "$PROFILE" '/home/flocka/atlas/qwen38/dflash2-official/abi-v2.json' "$ABI"
  pin "$PROFILE" '/run/user/{uid}/atlas-qwen38-dflash2-official-v2' "$session_template"
  pin "$PROFILE" 'UNRELEASED_POST_OFFICIAL_DFLASH2_FIX_BINARY' "$(sha "$FAKE")"
  pin "$PROFILE" '267be2125ee2ec272555748c87cc636b25a96107946f05491bd6043151c7fe4e' "$(sha "$TARGET/config.json")"
  pin "$PROFILE" 'f9ba0436d933e2362fb1bfa0508931c28b68f1fddbd5b94e6d50712e36a0636c' "$(sha "$TARGET/model.safetensors.index.json")"
  pin "$PROFILE" '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3' "$(sha "$TARGET/tokenizer.json")"
  pin "$PROFILE" '5a205aa76328f59df93a9091d0496aeda4615c6b0c495cbd37e14c79c0cd0f94' "$(sha "$TARGET/tokenizer_config.json")"
  pin "$PROFILE" 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041' "$(sha "$TARGET/chat_template.jinja")"
  pin "$PROFILE" 'd0dbf670c6a372817b2ff92d5d47e3130d35de9c3a7164ba455fd7a88255b362' "$(sha "$TARGET/generation_config.json")"
  pin "$PROFILE" '0c1004d622f835eef17ba193d8e966ecfd2d5218cbc8d25b7effff8cb4011893' "$(sha "$TARGET/hf_quant_config.json")"
  pin "$PROFILE" '873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980' "$(sha "$DRAFT/config.json")"
  pin "$PROFILE" '67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c' "$(sha "$DRAFT/model.safetensors")"
  pin "$PROFILE" '3848817896' "$(stat -c %s "$DRAFT/model.safetensors")"
  pin "$PROFILE" '8928' "$HEADER_LENGTH"
  pin "$PROFILE" '0c2c70601b30f8d1ca7d5794b817779ba2dcf1956cfc7d4f83e87091e1ab7c8c' "$HEADER_SHA"
  pin "$PROFILE" '1924404480' '81'
  local index=0
  local hashes=(
    4b71c5d9d5027c88d1df9f53b93ad77b8d5e428d5d6f75ae42063704d94930b9
    d0ceb042ef003ec63e00203823eb70d4815c9fc0430ba04a5442b6b8bde19093
    b633a26c1d3b97bf061585363b99d4c01650b4e3f6d0cbb18e7c7e03ca2de1e4
    485ea312626e2d1f3a820d9f7545809cef1e2e77414e3977a61d32dacd370bd0
    2aa61bab1ed252fe3cdb479286b5b7add7e8ad567501add5f6eeef3987b03d31
    8214cd21a46832380c06be552c42d9514c6cc9f27d98e09e2a09e36a8bd6e6ce
    d97834985f2442cd8b7e48e8426636e2ef8836ad1c7c5fba4ad879a0fcc66685
  )
  local sizes=(4013662656 4104675536 4028901296 4003421984 4000515080 5380672368 849400408)
  for weight in "${TARGET_WEIGHTS[@]}"; do
    pin "$PROFILE" "${hashes[$index]}" "$(sha "$TARGET/$weight")"
    pin "$PROFILE" "${sizes[$index]}" "$(stat -c %s "$TARGET/$weight")"
    index=$((index + 1))
  done
  chmod 0555 "$PROFILE"
  write_abi "$host_count" "$(normalized_launcher_sha "$PROFILE")"
  chmod 0644 "$PROFILE"
  pin "$PROFILE" \
    'EXPECTED_ABI_ATTESTATION_SHA256 = "UNRELEASED_POST_OFFICIAL_DFLASH2_FIX_ATTESTATION"' \
    "EXPECTED_ABI_ATTESTATION_SHA256 = \"$(sha "$ABI")\""
  chmod 0555 "$PROFILE"
}

expect_fail 'official binary identity is unreleased' \
  /usr/bin/env -i PATH=/attacker BASH_ENV="$TMP/hostile-bash-env" \
  PYTHONPATH="$TMP/hostile-python" ATLAS_DFLASH_ECHO=1 "$CANONICAL"
expect_fail 'CLI arguments are forbidden' "$CANONICAL" --dflash-gamma 15

chmod 0644 "$DRAFT/config.json"
cp "$DRAFT/config.json" "$TMP/config.good"
python3 - "$DRAFT/config.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
value["dflash_config"]["block_size"] = 7
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
PY
chmod 0444 "$DRAFT/config.json"
make_profile "$TMP/sessions/bad-block-{uid}"
expect_fail 'draft config field dflash_config.block_size' "$PROFILE"
chmod 0644 "$DRAFT/config.json"
cp "$TMP/config.good" "$DRAFT/config.json"
chmod 0444 "$DRAFT/config.json"

make_profile "$TMP/sessions/drift-weight-{uid}"
chmod 0644 "$TARGET/model-00003-of-00006.safetensors"
printf 'drift\n' >>"$TARGET/model-00003-of-00006.safetensors"
chmod 0444 "$TARGET/model-00003-of-00006.safetensors"
expect_fail 'target_weight:model-00003-of-00006.safetensors size mismatch' "$PROFILE"
chmod 0644 "$TARGET/model-00003-of-00006.safetensors"
printf '%s\n' 'model-00003-of-00006.safetensors' \
  >"$TARGET/model-00003-of-00006.safetensors"
chmod 0444 "$TARGET/model-00003-of-00006.safetensors"

make_profile "$TMP/sessions/writable-binary-{uid}"
chmod 0755 "$FAKE"
expect_fail 'binary must be immutable' "$PROFILE"
chmod 0555 "$FAKE"

make_profile "$TMP/sessions/bad-abi-{uid}" 12
expect_fail 'ABI field host_argument_count' "$PROFILE"

cp "$DRAFT/model.safetensors" "$TMP/model.good"
chmod 0644 "$DRAFT/model.safetensors"
python3 - "$DRAFT/model.safetensors" <<'PY'
import sys
with open(sys.argv[1], "r+b") as handle:
    handle.seek(9)
    byte = handle.read(1)
    handle.seek(9)
    handle.write(bytes([byte[0] ^ 1]))
PY
chmod 0444 "$DRAFT/model.safetensors"
make_profile "$TMP/sessions/bad-header-{uid}"
expect_fail 'draft safetensors header sha256 mismatch' "$PROFILE"
chmod 0644 "$DRAFT/model.safetensors"
cp "$TMP/model.good" "$DRAFT/model.safetensors"
chmod 0444 "$DRAFT/model.safetensors"

make_profile "$TMP/sessions/source-drift-{uid}"
chmod 0755 "$PROFILE"
printf '\n# source drift\n' >>"$PROFILE"
chmod 0555 "$PROFILE"
expect_fail 'ABI field launcher_normalized_sha256' "$PROFILE"

mkdir -p "$TMP/real-session-parent"
ln -s "$TMP/real-session-parent" "$TMP/session-parent-link"
make_profile "$TMP/session-parent-link/session-{uid}"
expect_fail 'session path component is not a real directory' "$PROFILE"

make_profile "$TMP/sessions/official-{uid}"
UID_VALUE="$(id -u)"
SESSION_ROOT="${TMP}/sessions/official-${UID_VALUE}"
RECEIPT="$SESSION_ROOT/launch-receipt.json"
ALTERNATE="$TMP/attacker-selected-receipt.json"
/usr/bin/env -i PATH=/attacker BASH_ENV="$TMP/hostile-bash-env" \
  PYTHONHOME="$TMP/hostile-python" ATLAS_DFLASH_ECHO=1 \
  ATLAS_DDTREE_MAX_NODES=999 ATLAS_DFLASH2_LAUNCH_RECEIPT="$ALTERNATE" \
  "$PROFILE"
[[ ! -e "$ALTERNATE" ]] || fail "caller-selected alternate receipt was honored"

python3 - "$RECEIPT" "$CAPTURE_ARGS" "$CAPTURE_ENV" "$SESSION_ROOT" "$PROFILE" <<'PY'
import json
import os
import stat
import sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
captured_args = json.load(open(sys.argv[2], encoding="utf-8"))
captured_env = json.load(open(sys.argv[3], encoding="utf-8"))
assert receipt["schema"] == "atlas-qwen38-dflash2-official-launch-v2"
assert receipt["status"] == "preflight-passed-launch-pending"
assert receipt["transport"] == "retained-fd-exec-and-model-view"
assert receipt["environment"] == captured_env
assert receipt["argv"][1:] == captured_args[1:]
assert receipt["argv"][3].startswith("/proc/self/fd/")
assert receipt["argv"][receipt["argv"].index("--draft-model") + 1].startswith("/proc/self/fd/")
assert receipt["geometry"] == {
    "block_size": 8,
    "context_window": 2047,
    "gamma": 7,
    "k": 8,
    "sliding_window": 2048,
    "target_max_seq_len": 262144,
    "vocab_size": 248320,
}
assert receipt["abi"]["host_argument_count"] == 13
assert receipt["abi"]["ptx_parameter_count"] == 13
assert receipt["abi"]["launcher_normalized_sha256"] == receipt["launcher_normalized_sha256"]
assert len([key for key in receipt["identity"] if key.startswith("target_weight:")]) == 7
assert stat.S_IMODE(os.stat(sys.argv[1]).st_mode) == 0o444
assert stat.S_IMODE(os.stat(sys.argv[4]).st_mode) == 0o500
source = open(sys.argv[5], encoding="utf-8").read()
assert source.count("EXEC_ENV = {") == 1
assert source.count('"environment": EXEC_ENV') == 1
assert source.count("os.execve(binary_fd, argv, EXEC_ENV)") == 1
assert "identity(script_path, \"launcher\", sha256_file(script_path))" not in source
assert source.startswith("#!/usr/bin/env -S -i /usr/bin/python3 -I\n")
assert "os.open(component, DIRECTORY_OPEN_FLAGS" in source
assert 'f"/proc/self/fd/{target_view_fd}"' in source
assert 'f"/proc/self/fd/{draft_view_fd}"' in source
PY

expect_fail 'fixed launch session already exists' "$PROFILE"
printf 'PASS serve-dflash2-official dependency-free tests\n'
