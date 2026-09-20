// SPDX-License-Identifier: AGPL-3.0-only

//! Expert-major launch schedule; GEMM variants remain shared with legacy.
use super::*;

impl Glm53Exl3MoeKernels {
    pub(super) fn launch_staged(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Exl3MoePlan,
        buffers: Glm53Exl3MoeBuffers,
        stream: u64,
    ) -> Result<()> {
        let staged = self
            .staged
            .as_ref()
            .context("GLM EXL3 staged MoE kernels were not loaded")?;
        let p = buffers.pointers;
        let mut build = KernelLaunch::new(gpu, staged.build_chunks)
            .grid([1, 1, 1])
            .block([320, 1, 1])
            .arg_ptr(buffers.expert_count_i64.ptr)
            .arg_ptr(buffers.pair_expert_u32.ptr)
            .arg_ptr(buffers.chunk_expert_u32.ptr)
            .arg_ptr(buffers.chunk_start_u32.ptr)
            .arg_ptr(buffers.chunk_rows_u32.ptr)
            .arg_ptr(buffers.chunk_count_u32.ptr)
            .arg_ptr(buffers.route_status_u32.ptr)
            .arg_u32(EXPERTS)
            .arg_u32(plan.max_chunks);
        if staged.private {
            build = build.arg_u32(plan.pairs);
        }
        build.launch(stream)?;
        let mut gather = KernelLaunch::new(gpu, staged.gather)
            .grid([plan.pairs, 1, 1])
            .block([STAGED_BLOCK_THREADS, 1, 1])
            .arg_ptr(buffers.input_f16.ptr)
            .arg_ptr(buffers.temp_state_g_f16.ptr)
            .arg_ptr(buffers.temp_state_u_f16.ptr)
            .arg_ptr(p.gate_suh.ptr)
            .arg_ptr(p.up_suh.ptr)
            .arg_ptr(buffers.token_sorted_i64.ptr)
            .arg_ptr(buffers.pair_expert_u32.ptr)
            .arg_u32(plan.pairs);
        if staged.private {
            gather = gather.arg_ptr(buffers.route_status_u32.ptr);
        }
        gather.launch(stream)?;
        KernelLaunch::new(gpu, staged.gate_up)
            .grid([INTERMEDIATE / STAGED_TILE_N, plan.max_chunks, 2])
            .block([STAGED_BLOCK_THREADS, 1, 1])
            .shared_mem(STAGED_SHARED_MEMORY_BYTES)
            .arg_ptr(buffers.temp_state_g_f16.ptr)
            .arg_ptr(buffers.temp_state_u_f16.ptr)
            .arg_ptr(buffers.temp_intermediate_g_f16.ptr)
            .arg_ptr(buffers.temp_intermediate_u_f16.ptr)
            .arg_ptr(p.gate_trellis.ptr)
            .arg_ptr(p.up_trellis.ptr)
            .arg_ptr(buffers.chunk_expert_u32.ptr)
            .arg_ptr(buffers.chunk_start_u32.ptr)
            .arg_ptr(buffers.chunk_rows_u32.ptr)
            .arg_ptr(buffers.chunk_count_u32.ptr)
            .arg_i32(HIDDEN as i32)
            .arg_i32(INTERMEDIATE as i32)
            .arg_i32(128)
            .arg_ptr(buffers.locks_i32.ptr)
            .launch(stream)?;
        let mut activate = KernelLaunch::new(gpu, staged.activate)
            .grid([plan.pairs, 1, 1])
            .block([STAGED_BLOCK_THREADS, 1, 1])
            .arg_ptr(buffers.temp_intermediate_g_f16.ptr)
            .arg_ptr(buffers.temp_intermediate_u_f16.ptr)
            .arg_ptr(p.gate_svh.ptr)
            .arg_ptr(p.up_svh.ptr)
            .arg_ptr(p.down_suh.ptr)
            .arg_ptr(buffers.pair_expert_u32.ptr)
            .arg_u32(plan.pairs);
        if staged.private {
            activate = activate.arg_ptr(buffers.route_status_u32.ptr);
        }
        activate.launch(stream)?;
        KernelLaunch::new(gpu, staged.down)
            .grid([HIDDEN / STAGED_TILE_N, plan.max_chunks, 1])
            .block([STAGED_BLOCK_THREADS, 1, 1])
            .shared_mem(STAGED_SHARED_MEMORY_BYTES)
            .arg_ptr(buffers.temp_intermediate_g_f16.ptr)
            .arg_ptr(buffers.temp_state_g_f16.ptr)
            .arg_ptr(p.down_trellis.ptr)
            .arg_ptr(buffers.chunk_expert_u32.ptr)
            .arg_ptr(buffers.chunk_start_u32.ptr)
            .arg_ptr(buffers.chunk_rows_u32.ptr)
            .arg_ptr(buffers.chunk_count_u32.ptr)
            .arg_i32(256)
            .arg_ptr(buffers.locks_i32.ptr)
            .launch(stream)?;
        let direct_combine = direct_combine_scope(plan.rows, self.direct_combine);
        if direct_combine {
            let kernel = staged
                .direct_combine
                .context("GLM direct prefill combine kernel was not loaded")?;
            return KernelLaunch::new(gpu, kernel)
                .grid([plan.rows, HIDDEN / 128, 1])
                .block([32, 1, 1])
                .shared_mem(DIRECT_COMBINE_SHARED_MEMORY_BYTES)
                .arg_ptr(buffers.temp_state_g_f16.ptr)
                .arg_ptr(buffers.output_f32.ptr)
                .arg_ptr(p.down_svh.ptr)
                .arg_ptr(buffers.weight_sorted_f16.ptr)
                .arg_ptr(buffers.pair_expert_u32.ptr)
                .arg_ptr(buffers.route_private_f32.ptr)
                .arg_ptr(buffers.route_status_u32.ptr)
                .arg_u32(plan.rows)
                .arg_u32(plan.pairs)
                .launch(stream);
        }
        let routed = if staged.private {
            buffers.route_private_f32
        } else {
            buffers.output_f32
        };
        let mut scatter = KernelLaunch::new(gpu, staged.scatter)
            .grid([plan.pairs, 1, 1])
            .block([STAGED_BLOCK_THREADS, 1, 1])
            .shared_mem(STAGED_SCATTER_SHARED_MEMORY_BYTES)
            .arg_ptr(buffers.temp_state_g_f16.ptr)
            .arg_ptr(routed.ptr)
            .arg_ptr(p.down_svh.ptr)
            .arg_ptr(buffers.token_sorted_i64.ptr)
            .arg_ptr(buffers.weight_sorted_f16.ptr)
            .arg_ptr(buffers.pair_expert_u32.ptr)
            .arg_u32(plan.pairs);
        if staged.private {
            scatter = scatter.arg_ptr(buffers.route_status_u32.ptr);
        }
        scatter.launch(stream)
    }
}
