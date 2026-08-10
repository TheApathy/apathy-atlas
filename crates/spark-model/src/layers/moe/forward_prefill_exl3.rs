// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis (3.0 bpw) routed-expert PREFILL (M>1) — plan §3 "P1".
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
//! Costs (bring-up, honest arithmetic — see docs/kernels/exl3-gemv.md §7):
//! per layer at full 2410-token prefill all 216 experts are routed, so the
//! dequant writes 10.87 GB and the GEMM re-reads it 1–2× — a few seconds per
//! prefill pass total. Acceptable for smoke; P2 (grouped trellis GEMM) is
//! the fix. NOT graph-capture-legal (host-driven chunk loop + D2H of
//! `expert_offsets`) — prefill never captures.

use anyhow::{Context, Result, ensure};
use spark_runtime::kernel_args::KernelLaunch;

use super::*;
use super::exl3_decode::Exl3ProjTable;

/// Threads per block of the H128 row kernels (8 warps × one 128-chunk each).
const H128_BLOCK: u32 = 256;
/// K-columns covered per block of the H128 row kernels.
const H128_COLS_PER_BLOCK: u32 = 1024;

#[allow(clippy::too_many_arguments)]
fn launch_h128_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,                // [num_tokens, k] token-major (or sorted when gather is identity)
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
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let st = self
            .exl3
            .as_ref()
            .expect("run_routed_grouped_gemm_exl3 without EXL3 state");
        ensure!(
            !ctx.graph_capture,
            "EXL3 prefill is host-driven (expert-chunk loop + offsets D2H) — \
             not legal under CUDA graph capture"
        );
        ensure!(
            self.moe_bf16_grouped_gemm_k.0 != 0,
            "EXL3 prefill needs the moe_bf16_grouped_gemm kernel module"
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
        let gpu = ctx.gpu;
        let pf = &st.prefill;

        // Host per-expert histogram: exact per-chunk m-tiles + empty-chunk
        // skip. Prefill-only D2H — the same pattern (and cost) as the
        // exact-tiles grid sizing in `run_routed_grouped_gemm`.
        let ne = num_experts as usize;
        let mut offs_raw = vec![0u8; (ne + 1) * 4];
        gpu.copy_d2h_on_stream(expert_offsets, &mut offs_raw, stream)
            .context("EXL3 prefill: expert_offsets D2H")?;
        let offs: Vec<u32> = offs_raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // A_rot aliases expert_down_out: same shape [total_expanded, h], and
        // it is dead before the down GEMM writes the buffer (see module doc).
        let a_rot = expert_down_out;

        // Chunked dequant + sub-range grouped GEMM over one projection.
        let run_proj = |tab: &Exl3ProjTable, a: DevicePtr, out: DevicePtr| -> Result<()> {
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

        // ── gate (w1): pre-rotate → grouped GEMM → post-rotate ──
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

        // ── up (w3): same, with up's rotation pair (A_rot reused) ──
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
        launch_h128_post(
            gpu,
            pf.h128_post_k,
            expert_up_out,
            sorted_expert_ids,
            st.up.svh_tab,
            total_expanded,
            inter,
            stream,
        )?;

        // ── clamped SwiGLU (same kernel the NVFP4 prefill path uses) ──
        ops::silu_mul(
            gpu,
            self.moe_act_mul,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            total_expanded * inter,
            stream,
        )?;

        // ── down (w2): in-place pre-rotate (identity gather) → GEMM → post ──
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
        run_proj(&st.down, expert_gate_out, expert_down_out)?;
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
        Ok(())
    }
}
