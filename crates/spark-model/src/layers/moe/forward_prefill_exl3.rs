// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis (K2/K3, 2.0/3.0 bpw) routed-expert PREFILL (M>1) — plan §3 "P1".
//!
//! The trellis tiles are not per-(k,n) addressable, so the grouped
//! tensor-core GEMMs cannot read them directly. Bring-up path (option (a)
//! of the P1 design — rotations on the ACTIVATIONS, scratch holds the raw
//! decoded weights):
//!
//!   1. `exl3_h128_pre_rows`: expand the token-major input into the sorted
//!      layout with the per-expert input rotation applied per row:
//!      `A_rot[r] = H128(diag(suh_e) · A[token_r]) / √128`. suh differs per
//!      expert, so a token routed to k experts gets k distinct rows — which
//!      is exactly the sorted-layout expansion the grouped GEMM indexes.
//!   2. `exl3_dequant_chunk_bf16`: per expert CHUNK (default 8 ≈ 134 MB
//!      scratch), decode the trellis to raw BF16 `[N, K]` slots.
//!   3. `moe_bf16_grouped_gemm` launched over the chunk SUB-RANGE
//!      (`weight_ptrs = slot_tab`, `expert_offsets + e0`, `num_experts =
//!      chunk_len`, `sorted_token_ids = NULL`): `expert_offsets` values are
//!      absolute rows, so sub-range launches read/write the right global
//!      rows of the sorted buffers.
//!   4. `exl3_h128_post_rows`: in-place output rotation
//!      `Y[r] = diag(svh_e) · H128(Y[r]) / √128`.
//!
//! Rotation composition verified against the f64 CPU oracle in
//! `examples/exl3_gemv_microtest.rs` (the same composition the M=1 GEMV
//! applies): `y = diag(svh) · H128( W_dec · H128( diag(suh) · x ) ) / 128`.
//!
//! Buffer aliasing (no new activation buffers): `A_rot` lives in
//! `expert_down_out` (`[total_expanded, h]` — exactly the needed shape); it
//! is dead by the time the down GEMM writes that buffer. The down-input
//! rotation runs IN PLACE over the post-SiLU `expert_gate_out` (the pre/post
//! kernels are warp-private per 128-chunk, so in-place is safe with the
//! identity gather).
//!
//! P1 costs and the opt-in direct P2 path are documented in
//! docs/kernels/exl3-gemv.md §§7/9. Persistent P2 sizes its work entirely
//! from device offsets and avoids the P1/exact-grid host synchronization.
//! This prefill path is not yet enabled during CUDA graph capture.

use anyhow::{Context, Result, ensure};
use spark_runtime::kernel_args::KernelLaunch;

use super::exl3_decode::Exl3ProjTable;
use super::*;

/// Threads per block of the H128 row kernels (8 warps × one 128-chunk each).
const H128_BLOCK: u32 = 256;
const H128_COLS_PER_BLOCK: u32 = 1024;
const EXL3_DIRECT_M_TILE: u32 = 64;
const EXL3_DIRECT_M128_TILE: u32 = 128;

#[allow(clippy::too_many_arguments)]
fn launch_h128_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr, // [num_tokens, k] token-major (or sorted when gather is identity)
    sorted_token_ids: DevicePtr, // NULL → identity gather (required for in-place)
    sorted_expert_ids: DevicePtr,
    suh_tab: DevicePtr,
    a_out: DevicePtr, // [rows, k] sorted layout
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, k.div_ceil(H128_COLS_PER_BLOCK), 1])
        .block([H128_BLOCK, 1, 1])
        .arg_ptr(a)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(suh_tab)
        .arg_ptr(a_out)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_h128_pre_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    gate_suh_tab: DevicePtr,
    up_suh_tab: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 4, 1])
        .block([H128_BLOCK, 1, 1])
        .arg_ptr(a)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(gate_suh_tab)
        .arg_ptr(up_suh_tab)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_u32(k)
        .arg_u32(rows)
        .launch(stream)
}

fn launch_h128_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    y: DevicePtr, // [rows, n] sorted layout, in place
    sorted_expert_ids: DevicePtr,
    svh_tab: DevicePtr,
    rows: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n.div_ceil(H128_COLS_PER_BLOCK), 1])
        .block([H128_BLOCK, 1, 1])
        .arg_ptr(y)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(svh_tab)
        .arg_u32(n)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_h128_post_silu_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    sorted_expert_ids: DevicePtr,
    gate_svh_tab: DevicePtr,
    up_svh_tab: DevicePtr,
    down_suh_tab: DevicePtr,
    rows: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n.div_ceil(H128_COLS_PER_BLOCK), 1])
        .block([H128_BLOCK, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(gate_svh_tab)
        .arg_ptr(up_svh_tab)
        .arg_ptr(down_suh_tab)
        .arg_u32(n)
        .launch(stream)
}

impl MoeLayer {
    /// EXL3 replacement for the routed grouped-GEMM phase of
    /// `forward_prefill` (steps 5–6): dequant-to-scratch chunks + H128
    /// activation rotations around the BF16 grouped GEMM. Writes the routed
    /// outputs into `ctx.buffers.expert_down_out()` in the sorted layout the
    /// downstream unpermute expects.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_grouped_gemm_exl3(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        sorted_expert_ids: DevicePtr,
        h: u32,
        inter: u32,
        num_experts: u32,
        total_expanded: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let st = self
            .exl3
            .as_ref()
            .expect("run_routed_grouped_gemm_exl3 without EXL3 state");
        let pf = &st.prefill;
        let direct = pf.direct;
        let persistent = pf.persistent;
        ensure!(
            !ctx.graph_capture,
            "EXL3 prefill is not yet enabled under CUDA graph capture"
        );
        ensure!(
            direct || self.moe_bf16_grouped_gemm_k.0 != 0,
            "EXL3 P1 prefill needs the moe_bf16_grouped_gemm kernel module"
        );
        ensure!(
            st.gate.n == inter && st.gate.k == h && st.down.n == h && st.down.k == inter,
            "EXL3 prefill dims mismatch: gate [{}x{}] down [{}x{}] vs h={h} inter={inter}",
            st.gate.n,
            st.gate.k,
            st.down.n,
            st.down.k
        );
        if total_expanded == 0 {
            return Ok(());
        }
        if self.try_run_exl3_w2a8_prefill(
            expert_input,
            expert_offsets,
            sorted_token_ids,
            sorted_expert_ids,
            num_experts,
            total_expanded,
            top_k,
            ctx,
            stream,
        )? {
            // The W2A8 core ends at the raw BF16 down projection. Preserve
            // the incumbent down SVH/H128 boundary unless the outer
            // post-unpermute fusion owns that transform.
            if !pf.fused_unpermute {
                launch_h128_post(
                    ctx.gpu,
                    pf.h128_post_k,
                    ctx.buffers.expert_down_out(),
                    sorted_expert_ids,
                    st.down.svh_tab,
                    total_expanded,
                    h,
                    stream,
                )?;
            }
            return Ok(());
        }
        let gpu = ctx.gpu;
        // P1 and the exact-grid P2 fallback need a host histogram. Persistent
        // P2 assigns exact (expert, N64 strip) work on-device and walks each
        // expert's live M tiles, avoiding this stream-draining D2H without a
        // rectangular empty-task universe or a router load cap.
        let ne = num_experts as usize;
        let needs_host_offsets = !direct || !persistent;
        let offs = if needs_host_offsets {
            let mut raw = vec![0u8; (ne + 1) * 4];
            gpu.copy_d2h_on_stream(expert_offsets, &mut raw, stream)
                .context("EXL3 prefill: expert_offsets D2H")?;
            Some(
                raw.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // A_rot aliases expert_down_out: same shape [total_expanded, h], and
        // it is dead before the down GEMM writes the buffer (see module doc).
        let a_rot = expert_down_out;

        // Chunked dequant + sub-range grouped GEMM over one projection.
        let run_proj = |tab: &Exl3ProjTable, a: DevicePtr, out: DevicePtr| -> Result<()> {
            if direct {
                let direct_m_tile = if pf.direct_m128 {
                    EXL3_DIRECT_M128_TILE
                } else {
                    EXL3_DIRECT_M_TILE
                };
                let direct_n_tile = if pf.direct_n256 {
                    256
                } else if pf.direct_n128 {
                    128
                } else {
                    64
                };
                let direct_k_step = if pf.direct_k64 { 64 } else { 16 };
                let direct_block_threads = (direct_m_tile / 64) * (direct_n_tile / 16) * 32;
                ensure!(
                    tab.n.is_multiple_of(direct_n_tile) && tab.k.is_multiple_of(direct_k_step),
                    "EXL3 direct prefill requires N divisible by {direct_n_tile} and K by {direct_k_step}, got N={} K={}",
                    tab.n,
                    tab.k
                );
                // P2: decode each trellis tile straight into BF16 B-fragment
                // registers and consume it with m16n8k16 MMA. `a` is already sorted and
                // H128-rotated, so sorted_token_ids is NULL (identity rows).
                let max_m_tiles = (!persistent).then(|| {
                    offs.as_ref()
                        .expect("exact-grid P2 requires host offsets")
                        .windows(2)
                        .map(|pair| pair[1].saturating_sub(pair[0]))
                        .max()
                        .unwrap_or(0)
                        .div_ceil(direct_m_tile)
                        .max(1)
                });
                let grid = if persistent {
                    let strips = num_experts
                        .checked_mul(tab.n / direct_n_tile)
                        .context("EXL3 persistent prefill grid size overflow")?;
                    // One CTA per exact (expert, N-tile) strip. The kernel still
                    // supports grid-striding for parity tests, but production
                    // exposes all independent work to the hardware scheduler.
                    [strips.max(1), 1, 1]
                } else {
                    [
                        tab.n / 64,
                        max_m_tiles.expect("exact-grid P2 max M tiles"),
                        num_experts,
                    ]
                };
                let fixed_gu = tab.n == 2048 && tab.k == 4096;
                let fixed_down = tab.n == 4096 && tab.k == 2048;
                let fixed_k2_shape = pf.fixed_shape && tab.bits == 2;
                let kernel = if fixed_k2_shape && fixed_gu && pf.direct_n256 {
                    pf.grouped_direct_k64_n256_k2_gu_k
                } else if fixed_k2_shape && fixed_down && pf.direct_n256 {
                    pf.grouped_direct_k64_n256_k2_down_k
                } else if fixed_k2_shape && fixed_gu && pf.direct_n128 {
                    pf.grouped_direct_k64_n128_k2_gu_k
                } else if fixed_k2_shape && fixed_down && pf.direct_n128 {
                    pf.grouped_direct_k64_n128_k2_down_k
                } else if fixed_k2_shape && fixed_gu && pf.direct_k64 {
                    pf.grouped_direct_k64_k2_gu_k
                } else if fixed_k2_shape && fixed_down && pf.direct_k64 {
                    pf.grouped_direct_k64_k2_down_k
                } else if fixed_k2_shape && fixed_gu {
                    pf.grouped_direct_k2_gu_k
                } else if fixed_k2_shape && fixed_down {
                    pf.grouped_direct_k2_down_k
                } else if pf.direct_m128 {
                    pf.grouped_direct_m128_k
                } else if pf.direct_k64 && pf.fixed_k2 && tab.bits == 2 {
                    pf.grouped_direct_k64_k2_k
                } else if pf.direct_k64 {
                    pf.grouped_direct_k64_k
                } else if pf.fixed_k2 && tab.bits == 2 {
                    pf.grouped_direct_k2_k
                } else {
                    pf.grouped_direct_k
                };
                return KernelLaunch::new(gpu, kernel)
                    .grid(grid)
                    .block([direct_block_threads, 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(tab.trellis_tab)
                    .arg_ptr(out)
                    .arg_ptr(expert_offsets)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(num_experts)
                    .arg_u32(tab.n)
                    .arg_u32(tab.k)
                    .arg_u32(tab.bits)
                    .arg_u32(u32::from(persistent))
                    .launch(stream);
            }
            let offs = offs.as_ref().expect("P1 requires host offsets");
            for e0 in (0..ne).step_by(pf.chunk as usize) {
                let cnt = (ne - e0).min(pf.chunk as usize) as u32;
                let max_rows = (e0..e0 + cnt as usize)
                    .map(|e| offs[e + 1] - offs[e])
                    .max()
                    .unwrap_or(0);
                if max_rows == 0 {
                    continue; // no routed rows in this chunk
                }
                KernelLaunch::new(gpu, pf.dequant_chunk_k)
                    .grid([tab.n / 16, tab.k / 16, cnt])
                    .block([32, 1, 1])
                    .arg_ptr(tab.trellis_tab)
                    .arg_u32(e0 as u32)
                    .arg_u32(cnt)
                    .arg_ptr(pf.scratch)
                    .arg_u32(tab.n)
                    .arg_u32(tab.k)
                    .arg_u32(tab.bits)
                    .launch(stream)?;
                // Sub-range grouped GEMM: offsets are absolute rows, so the
                // chunk's outputs land at their global sorted positions.
                // sorted_token_ids = NULL — `a` is already the expanded,
                // per-expert-rotated sorted layout.
                ops::moe_bf16_grouped_gemm(
                    gpu,
                    self.moe_bf16_grouped_gemm_k,
                    a,
                    pf.slot_tab,
                    out,
                    expert_offsets.offset(e0 * 4),
                    DevicePtr(0),
                    cnt,
                    tab.n,
                    tab.k,
                    max_rows.div_ceil(64).max(1),
                    stream,
                )?;
            }
            Ok(())
        };

        let up_projected = if pf.dual_pre {
            ensure!(h == 4096, "EXL3 dual pre-rotation requires H=4096, got {h}");
            let rotated_up_bytes = (total_expanded as usize)
                .checked_mul(h as usize)
                .and_then(|elements| elements.checked_mul(2))
                .context("EXL3 dual pre-rotation capacity overflow")?;
            ensure!(
                ctx.buffers.expert_up_out_bytes() >= rotated_up_bytes,
                "EXL3 dual pre-rotation needs {rotated_up_bytes} expert_up_out bytes, arena has {}",
                ctx.buffers.expert_up_out_bytes()
            );
            launch_h128_pre_dual(
                gpu,
                pf.h128_pre_dual_h4096_k,
                expert_input,
                sorted_token_ids,
                sorted_expert_ids,
                st.gate.suh_tab,
                st.up.suh_tab,
                a_rot,
                expert_up_out,
                total_expanded,
                h,
                stream,
            )?;
            run_proj(&st.gate, a_rot, expert_gate_out)?;
            // Gate's H4096 rotation is dead; up safely writes H2048 here.
            run_proj(&st.up, expert_up_out, expert_down_out)?;
            let up_projected = expert_down_out;
            up_projected
        } else {
            // ── gate (w1) ──
            launch_h128_pre(
                gpu,
                pf.h128_pre_k,
                expert_input,
                sorted_token_ids,
                sorted_expert_ids,
                st.gate.suh_tab,
                a_rot,
                total_expanded,
                h,
                stream,
            )?;
            run_proj(&st.gate, a_rot, expert_gate_out)?;
            if !pf.fused_post {
                launch_h128_post(
                    gpu,
                    pf.h128_post_k,
                    expert_gate_out,
                    sorted_expert_ids,
                    st.gate.svh_tab,
                    total_expanded,
                    inter,
                    stream,
                )?;
            }

            // ── up (w3), reusing A_rot ──
            launch_h128_pre(
                gpu,
                pf.h128_pre_k,
                expert_input,
                sorted_token_ids,
                sorted_expert_ids,
                st.up.suh_tab,
                a_rot,
                total_expanded,
                h,
                stream,
            )?;
            run_proj(&st.up, a_rot, expert_up_out)?;
            expert_up_out
        };

        if pf.fused_post {
            launch_h128_post_silu_pre(
                gpu,
                pf.h128_post_silu_pre_k,
                expert_gate_out,
                up_projected,
                sorted_expert_ids,
                st.gate.svh_tab,
                st.up.svh_tab,
                st.down.suh_tab,
                total_expanded,
                inter,
                stream,
            )?;
        } else {
            launch_h128_post(
                gpu,
                pf.h128_post_k,
                up_projected,
                sorted_expert_ids,
                st.up.svh_tab,
                total_expanded,
                inter,
                stream,
            )?;

            // Clamped SwiGLU (same kernel the NVFP4 prefill path uses).
            ops::silu_mul(
                gpu,
                self.moe_act_mul,
                expert_gate_out,
                up_projected,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
        }

        // ── down (w2): in-place pre-rotate (identity gather) → GEMM → post ──
        if !pf.fused_post {
            launch_h128_pre(
                gpu,
                pf.h128_pre_k,
                expert_gate_out,
                DevicePtr(0),
                sorted_expert_ids,
                st.down.suh_tab,
                expert_gate_out,
                total_expanded,
                inter,
                stream,
            )?;
        }
        run_proj(&st.down, expert_gate_out, expert_down_out)?;
        if !pf.fused_unpermute {
            launch_h128_post(
                gpu,
                pf.h128_post_k,
                expert_down_out,
                sorted_expert_ids,
                st.down.svh_tab,
                total_expanded,
                h,
                stream,
            )?;
        }
        Ok(())
    }
}
