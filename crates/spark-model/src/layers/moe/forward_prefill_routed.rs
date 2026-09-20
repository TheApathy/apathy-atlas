// SPDX-License-Identifier: AGPL-3.0-only

//! Routed grouped-GEMM phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_routed_grouped_gemm`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring. Covers steps 4-6 of the prefill
//! pipeline: grid sizing, grouped gate+up GEMM, SiLU, grouped down GEMM.

use super::*;

impl MoeLayer {
    /// Routed-expert grouped-GEMM path: upper-bound grid sizing → grouped
    /// gate+up GEMM → SiLU+mul → grouped down GEMM.
    ///
    /// Writes the routed expert outputs into `ctx.buffers.expert_down_out()`.
    /// `t0` carries the running profile timer so per-step timing output
    /// matches the original inline pipeline exactly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_grouped_gemm(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        num_tokens: usize,
        ne: usize,
        t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    *t0 = Some(std::time::Instant::now());
                }
            };
        }

        // 4. Upper-bound max_m_tiles — sized for the absolute worst case
        // (one expert eats all tokens) so the kernel never silently truncates
        // a heavily-loaded expert's rows. The previous `avg*2` heuristic was
        // wrong: real learned MoE routers concentrate experts ~7× the average
        // (observed on Qwen3.6-35B-A3B at chunk=4097: avg=129, max=929 for
        // expert 227 → kernel covered 320 rows but needed 929 → 609 rows
        // silently dropped → systematic ~-14% under-count in routed-MoE
        // output). The Poisson(avg) assumption in the old comment doesn't
        // hold for trained routers — they're sparse + concentrated.
        //
        // Cost: extra empty tiles for under-utilized experts; each early-
        // exits on `m_idx >= M_expert` so overhead is low vs the correctness
        // bug.
        //
        // Mirrors the FP8 path (see forward_prefill_fp8.rs).
        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        let mut max_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        // Diagnostic bridge for the GPU compact-worklist implementation:
        // the correctness-first upper bound above launches
        // num_experts*ceil(N*top_k/64) row tiles even though nearly all of
        // them immediately exit. On 512-expert Flash-Next this is millions
        // of empty CTAs per layer. Read the already-built 2 KiB offset table
        // to prove the exact-grid ceiling before replacing this host sync
        // with a device-generated compact tile list.
        if std::env::var("ATLAS_MOE_EXACT_PREFILL_GRID")
            .ok()
            .as_deref()
            == Some("1")
        {
            ctx.gpu.synchronize(stream)?;
            let mut bytes = vec![0u8; (ne + 1) * 4];
            ctx.gpu.copy_d2h(expert_offsets, &mut bytes)?;
            let offsets: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let max_rows = offsets
                .windows(2)
                .map(|w| w[1].saturating_sub(w[0]))
                .max()
                .unwrap_or(0);
            max_m_tiles = max_rows.div_ceil(64).max(1);
        }
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );
        prof_step!("grid_setup");

        let total_expanded = n * top_k;

        // 5. Grouped gate+up GEMM — cp.async pipelined FP8-MMA K64 (transposed).
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // Zero expert buffers unconditionally before the grouped GEMMs.
        // Even with worst-case `max_m_tiles` (above), some kernel paths only
        // write rows where `m_idx < M_expert` per expert — rows past the
        // expert's actual count keep stale data from the previous prefill
        // (or uninit memory on first prefill), which then propagates
        // through unpermute_reduce as spurious contributions. Previously
        // guarded by `ctx.comm.is_some()` (EP only), now unconditional.
        // Mirrors the FP8 path fix (commit 34626d3).
        {
            let gate_bytes = total_expanded as usize * inter as usize * 2;
            let up_bytes = gate_bytes;
            let down_bytes = total_expanded as usize * h as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, up_bytes, stream)?;
            ctx.gpu
                .memset_async(ctx.buffers.expert_down_out(), 0, down_bytes, stream)?;
        }
        if self.qwen4_compact.is_some() {
            anyhow::ensure!(
                n as usize == num_tokens
                    && h == 2560
                    && inter == 640
                    && num_experts == 512
                    && ne == 512
                    && top_k == 10,
                "Qwen4 compact routed dispatch geometry mismatch"
            );
            self.run_qwen4_compact(
                expert_input,
                expert_offsets,
                sorted_token_ids,
                num_tokens,
                ctx,
                stream,
            )?;
            prof_step!("grouped_compact_checked");
            return Ok(());
        }
        if max_m_tiles > 0 {
            if self.nvfp4_moe_worklist {
                anyhow::ensure!(
                    h.is_multiple_of(64) && inter.is_multiple_of(64),
                    "NVFP4 compact work-list K64 kernels require H/intermediate divisible by 64 (H={h}, intermediate={inter})"
                );
                let (gp, up) = match (&self.gate_ptrs_t, &self.up_ptrs_t) {
                    (Some(gp), Some(up)) => (gp, up),
                    _ => anyhow::bail!(
                        "ATLAS_NVFP4_MOE_WORKLIST=1 requires transposed gate/up expert pointer tables"
                    ),
                };
                let n_tiles = (2 * inter).div_ceil(ops::NVFP4_WORKLIST_N_TILE);
                let max_tiles = ops::nvfp4_worklist_capacity_items(
                    total_expanded as usize,
                    ne,
                    n_tiles,
                    ops::NVFP4_WORKLIST_GATE_UP_M_TILE,
                )?;
                anyhow::ensure!(
                    max_tiles as usize * 2 * std::mem::size_of::<u32>()
                        <= ctx.buffers.sizes().moe_worklist,
                    "NVFP4 gate/up work-list bound exceeds persistent arena allocation"
                );
                ops::moe_w4a16_build_tile_worklist(
                    ctx.gpu,
                    self.moe_build_nvfp4_worklist_k,
                    expert_offsets,
                    gp.packed_ptrs,
                    ctx.buffers.moe_worklist(),
                    ctx.buffers.moe_worklist_total(),
                    num_experts,
                    n_tiles,
                    ops::NVFP4_WORKLIST_GATE_UP_M_TILE,
                    stream,
                )?;
                ops::moe_w4a16_fused_gate_up_k64_worklist(
                    ctx.gpu,
                    self.moe_fused_gate_up_t_k64_worklist_k,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_gate_out,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    ctx.buffers.moe_worklist(),
                    ctx.buffers.moe_worklist_total(),
                    max_tiles,
                    stream,
                )?;
            } else if let (Some(gp), Some(up)) = (&self.gate_ptrs_t, &self.up_ptrs_t) {
                // Block D #3 dispatch: M=128 path needs the env var on AND
                // the kernel actually loaded (try_kernel returns 0 on
                // models that don't ship it). max_m_tiles_m128 = ceil(...
                // /128) instead of /64; reuse the same upper bound by
                // halving (each m128 tile covers 2 m64 tiles).
                let use_m128 = self.nvfp4_gate_up_m128 && self.moe_fused_gate_up_t_k64_m128.0 != 0;
                if use_m128 {
                    let max_m_tiles_m128 = max_m_tiles.div_ceil(2).max(1);
                    ops::moe_w4a16_fused_gate_up_k64_m128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_m128,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles_m128,
                        stream,
                    )?;
                } else {
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                }
            } else {
                let (gp, up) = (&self.gate_ptrs, &self.up_ptrs);
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_gate_up");

        // 6. Activation+mul for routed experts + grouped down GEMM (K64 pipelined).
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            if self.nvfp4_moe_worklist {
                let dp = self.down_ptrs_t.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_NVFP4_MOE_WORKLIST=1 requires a transposed down expert pointer table"
                    )
                })?;
                let n_tiles = h.div_ceil(ops::NVFP4_WORKLIST_N_TILE);
                let max_tiles = ops::nvfp4_worklist_capacity_items(
                    total_expanded as usize,
                    ne,
                    n_tiles,
                    ops::NVFP4_WORKLIST_DOWN_M_TILE,
                )?;
                anyhow::ensure!(
                    max_tiles as usize * 2 * std::mem::size_of::<u32>()
                        <= ctx.buffers.sizes().moe_worklist,
                    "NVFP4 down work-list bound exceeds persistent arena allocation"
                );
                ops::moe_w4a16_build_tile_worklist(
                    ctx.gpu,
                    self.moe_build_nvfp4_worklist_k,
                    expert_offsets,
                    dp.packed_ptrs,
                    ctx.buffers.moe_worklist(),
                    ctx.buffers.moe_worklist_total(),
                    num_experts,
                    n_tiles,
                    ops::NVFP4_WORKLIST_DOWN_M_TILE,
                    stream,
                )?;
                ops::moe_w4a16_grouped_gemm_ptrtable_k64_worklist(
                    ctx.gpu,
                    self.moe_grouped_gemm_t_k64_worklist_k,
                    expert_gate_out,
                    dp.packed_ptrs,
                    dp.scale_ptrs,
                    dp.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    ctx.buffers.moe_worklist(),
                    ctx.buffers.moe_worklist_total(),
                    max_tiles,
                    stream,
                )?;
                static WORKLIST_ENGAGED: std::sync::Once = std::sync::Once::new();
                WORKLIST_ENGAGED.call_once(|| {
                    tracing::info!("ENGAGED ATLAS_NVFP4_MOE_WORKLIST: compact-m64-gate-up-down");
                });
            } else if let Some(dp) = &self.down_ptrs_t {
                ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                    ctx.gpu,
                    self.moe_grouped_gemm_t_k64,
                    expert_gate_out,
                    dp.packed_ptrs,
                    dp.scale_ptrs,
                    dp.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            } else {
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_gate_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_silu_down");

        Ok(())
    }
}
