// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM53_KDA_CONV_FUSED=1`: KDA conv stage fed by the EXL3 combined
//! projection.
//!
//! This lives beside `glm53_kda_conv.rs` rather than inside it because that
//! file's source TEXT is pinned by
//! `glm53_kda_conv_tests::exact_abi_address_and_barrier_contract_rejects_compile_plausible_mutants`,
//! which slices it at `pub fn launch_stage` / `pub fn launch_commit` and counts
//! `.arg_` calls in each half. Any extra launcher added to that file -- before,
//! between or after those two -- lands inside one of the slices and breaks the
//! contract. The duplicated finalize launch below is the price of leaving the
//! pinned ABI alone, and it is deliberate: the two launchers must be compared
//! by reading them, not by sharing a helper that could be edited once.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;
use super::glm53_kda_conv::{
    CHANNELS, Glm53KdaConvBuffers, Glm53KdaConvKernel, Glm53KdaConvPlan, STREAMS, THREADS,
    validate_buffers,
};

impl Glm53KdaConvKernel {
    /// Run the conv stage directly off the combined `[tokens][3][8192]` EXL3
    /// projection, so `atlas_glm53_kda_split_qkv` need not materialise three
    /// dense planes first. The conv arithmetic is unchanged, so the q/k/v
    /// outputs are bit-identical to `launch_stage` on split inputs.
    pub fn launch_stage_combined(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaConvPlan,
        combined_bf16: GgmlIqBuffer,
        buffers: Glm53KdaConvBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        anyhow::ensure!(
            combined_bf16.bytes == plan.stream_bytes * STREAMS as usize,
            "GLM KDA fused conv combined extent mismatch"
        );
        let fused = gpu.kernel(
            "glm53_kda_conv_fused",
            "atlas_glm53_kda_conv_f32_stage_fused",
        )?;
        gpu.memset_async(
            buffers.published_nonces_u64.ptr,
            0,
            plan.nonce_bytes,
            stream,
        )?;
        gpu.memset_async(buffers.published_ends_u32.ptr, 0, plan.end_bytes, stream)?;
        KernelLaunch::new(gpu, fused)
            .grid([CHANNELS.div_ceil(THREADS), plan.batch, STREAMS])
            .block([THREADS, 1, 1])
            .arg_ptr(combined_bf16.ptr)
            .arg_ptr(buffers.q_weight_f32.ptr)
            .arg_ptr(buffers.k_weight_f32.ptr)
            .arg_ptr(buffers.v_weight_f32.ptr)
            .arg_ptr(buffers.persistent_state_f32.ptr)
            .arg_ptr(buffers.staged_state_f32.ptr)
            .arg_ptr(buffers.q_output_bf16.ptr)
            .arg_ptr(buffers.k_output_bf16.ptr)
            .arg_ptr(buffers.v_output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .launch(stream)?;
        // Same publish half as launch_stage; see the module note above.
        KernelLaunch::new(gpu, self.finalize)
            .grid([1, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.published_ends_u32.ptr)
            .arg_ptr(buffers.published_nonces_u64.ptr)
            .arg_ptr(buffers.logical_lengths_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.capacity)
            .arg_u32(plan.start_position)
            .arg_u32(plan.end_position)
            .arg_u64(plan.transaction_nonce)
            .launch(stream)
    }
}
