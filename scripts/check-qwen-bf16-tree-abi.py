# SPDX-License-Identifier: AGPL-3.0-only

"""Static contracts for the Qwen-only BF16 DDTree attention ABI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


common_cuda = (ROOT / "kernels/gb10/common/paged_decode_attn.cu").read_text()
common_kernel = section(
    common_cuda,
    'extern "C" __global__ void paged_decode_attn(',
    ') {',
)
assert "sliding_window" in common_kernel
assert "kv_indir" not in common_kernel

common_host = (
    ROOT / "crates/spark-model/src/layers/ops/prefill_attn_a.rs"
).read_text()
common_launch = section(
    common_host,
    "pub fn paged_decode_attn_bf16(",
    "pub fn paged_decode_attn_fp8(",
)
assert "sliding_window" in common_launch
assert "kv_indir" not in common_launch

tree_cuda = (
    ROOT
    / "kernels/gb10/qwen3.8-27b/nvfp4/paged_decode_attn_bf16_qwen_tree.cu"
).read_text()
tree_kernel = section(
    tree_cuda,
    'extern "C" __global__ void paged_decode_attn_bf16_qwen_tree(',
    ") {",
)
for required in (
    "paged_decode_attn_bf16_qwen_tree",
    "kv_indirection",
    "kv_indir_base_ptr",
    "kv_indir_stride",
    "actual_pos = base +",
    "#define BC 4",
    "if (logical_pos < base)",
):
    assert required in tree_cuda
assert "sliding_window" not in tree_kernel

target_toml = (
    ROOT / "kernels/gb10/qwen3.8-27b/nvfp4/KERNEL.toml"
).read_text()
assert (
    'paged_decode_attn_bf16_qwen_tree = "paged_decode_bf16_qwen_tree"'
    in target_toml
)

dispatch = (
    ROOT
    / "crates/spark-model/src/layers/qwen3_attention/decode/run_paged_decode.rs"
).read_text()
tree_dispatch = section(
    dispatch,
    "KvCacheDtype::Bf16 => {",
    "// FP8 paged decode",
)
assert "if kv_indirection != DevicePtr::NULL" in tree_dispatch
assert "paged_decode_attn_bf16_qwen_tree" in tree_dispatch
assert "paged_decode_attn_bf16(" in tree_dispatch

certificate = (
    ROOT / "crates/spark-model/src/layers/qwen3_attention/trait_impl.rs"
).read_text()
assert "KvCacheDtype::Bf16 => qwen_bf16_tree_kernel_loaded" in certificate
assert "self.paged_decode_bf16_qwen_tree_k.0 != 0" in certificate

print("PASS: Qwen BF16 tree ABI is dedicated and fail-closed")
