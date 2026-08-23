// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen-only paged-attention launch contracts for DDTree verification.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Launch the dedicated Qwen BF16 ancestor-indirection kernel.
///
/// This ABI must remain separate from `paged_decode_attn_bf16`, whose final
/// argument is Gemma-4's sliding window and whose symbol is shared by every
/// BF16 model target.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16_qwen_tree(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    kv_indirection: DevicePtr,
    kv_indir_base_ptr: DevicePtr,
    kv_indir_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_ptr(kv_indirection)
        .arg_ptr(kv_indir_base_ptr)
        .arg_u32(kv_indir_stride)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn qwen_bf16_tree_launches_one_dedicated_kernel() {
        let gpu = MockGpuBackend::new();
        paged_decode_attn_bf16_qwen_tree(
            &gpu,
            KernelHandle(7),
            DevicePtr(1),
            DevicePtr(2),
            DevicePtr(3),
            DevicePtr(4),
            DevicePtr(5),
            DevicePtr(6),
            128,
            17,
            24,
            4,
            256,
            16,
            0.0625,
            24 * 256,
            DevicePtr(7),
            DevicePtr(8),
            32,
            0,
        )
        .unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
