// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 NVFP4 MoE path that re-uses the prefill-style fused gate+up GEMM.
//!
//! The closed Atlas Alpha leaderboard hits ~202 tok/s on
//! Qwen3.6-35B-A3B-NVFP4 partly because it routes the K=3 MTP verify
//! step through `moe_w4a16_fused_gate_up_t_k64` — one expert weight
//! read per launch covering BOTH gate and up projections. OSS only
//! wires that kernel into the prefill path; K=3 decode stays on the
//! warp-reduction `moe_expert_gate_up_shared_batch3` GEMV which does
//! two separate launches (~100-104 tok/s ceiling).
//!
//! This module bridges the gap. Pipeline (mirrors `forward_prefill`
//! with N=3):
//!
//! ```text
//! sort_by_expert(indices)          // 1 single-block launch (cheap)
//!   → moe_w4a16_fused_gate_up_k64_n128(A=[3,H], B_t=gate∥up)
//!   → silu_mul(gate, up)
//!   → moe_w4a16_grouped_gemm_ptrtable_n128(B_t=down)
//!   → moe_unpermute_reduce_indexed → output
//! run_shared_expert_prefill (N=3)  // separate GEMVs on default stream
//!   → moe_batched_blend            // adds sigmoid(input·gate_w)·shared
//! ```
//!
//! Gated behind `ATLAS_MOE_K3_FUSED_GATE_UP=1` in `init.rs`. Dispatch
//! site in `forward_k3.rs` additionally requires:
//!   - `gate_ptrs_t`, `up_ptrs_t`, `down_ptrs_t` all `Some` (full
//!     persistent transpose; `down_t_scratch_packed.is_none()`).
//!   - `moe_fused_gate_up_t_k64 != KernelHandle(0)`.
//!   - `correction_bias_dev.is_none()` (softmax routing only).
//!   - `fp8_gate_weight_ptrs.is_none()` (NVFP4 experts only).
//!   - EP world_size == 1 (single-GPU).
//!
//! Falls through to the existing batch3 path on any mismatch.

use super::*;

impl MoeLayer {
    /// Eligibility predicate for the fused K=3 gate+up path.
    ///
    /// Resolves a single env var (cached in `k3_fused_gate_up`) AND the
    /// runtime preconditions for safely launching
    /// `moe_w4a16_fused_gate_up_t_k64` with a 3-token input. EP and FP8
    /// must be checked at the dispatch site (we don't have `ctx.comm`
    /// here).
    #[inline]
    pub(crate) fn k3_fused_gate_up_eligible(&self) -> bool {
        self.k3_fused_gate_up
            && self.moe_fused_gate_up_t_k64.0 != 0
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
            && self.correction_bias_dev.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
    }

    /// Fused NVFP4 K=3 expert dispatch.
    ///
    /// Caller has already produced:
    ///   - `gate_logits`: BF16 [3, num_experts] (gate GEMM output) —
    ///     reused as sort scratch (sorted_token_ids ∥ sorted_expert_ids
    ///     ∥ expert_offsets ∥ token_to_perm).
    ///   - `indices_dev`: [3*top_k] u32 (top-k expert indices, expanded).
    ///   - `weights_dev`: [3*top_k] f32 (top-k routing weights).
    ///
    /// Produces final `output` = sum(weights * expert_down_out) +
    /// sigmoid(input·shared_gate_w) * shared_down_out.
    ///
    /// EP single-GPU only — the dispatch site bails to fallback when
    /// `world_size > 1`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_k3_fused_gate_up(
        &self,
        input: DevicePtr,       // [3, H] BF16 — raw MoE input (post-norm by caller)
        router_in: DevicePtr,   // [3, H] BF16 — router input (Gemma-4 pre-norm, else == input)
        gate_logits: DevicePtr, // [3, num_experts] BF16 — gate GEMM output (reused as sort scratch)
        indices_dev: DevicePtr, // [3*top_k] u32
        weights_dev: DevicePtr, // [3*top_k] f32
        expert_gate_out: DevicePtr, // [3*top_k, inter] BF16 (sorted-order writes)
        expert_up_out: DevicePtr, // [3*top_k, inter] BF16 (sorted-order writes)
        expert_down_out: DevicePtr, // [3*top_k, H] BF16 (sorted-order writes)
        shared_down_out: DevicePtr, // [3, H] BF16
        output: DevicePtr,      // [3, H] BF16 — final MoE output
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Mirror the prefill router_input handling (Gemma-4 pre-norm
        // already produced `router_in`; quiet `unused` until we wire
        // a use for it).
        let _ = router_in;

        let n = 3u32;
        let total_expanded = (n * top_k) as usize;
        let ne = num_experts as usize;

        // ── 1. Sort by expert ─────────────────────────────────────────
        //
        // gate_logits buffer is `[3, num_experts]` BF16 = 1536 B for
        // Qwen3.6 (256 experts). Sort needs:
        //   sorted_token_ids  : te * 4   = 96 B
        //   sorted_expert_ids : te * 4   = 96 B
        //   expert_offsets    : (ne+1)*4 = 1028 B  (257 entries)
        //   token_to_perm     : te * 4   = 96 B
        // Total: 1316 B  ≤ 1536 B — fits inside the gate_logits region.
        // (The sort kernel runs after gate_logits has been consumed by
        // topK, so reuse is safe.)
        let te = total_expanded as u32;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te as usize * 4);
        let expert_offsets = gate_logits.offset(te as usize * 4 * 2);
        let token_to_perm = gate_logits.offset(te as usize * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            te,
            num_experts,
            top_k,
            stream,
        )?;

        // ── 2. Fused gate+up GEMM (transposed NVFP4) ─────────────────
        //
        // M_TILE=64, N_TILE_LG=128. With te=24 tokens spread across 256
        // experts, no expert is touched by more than 1-2 tokens at K=3,
        // so max_m_tiles = ceil(2/64) = 1. Grid covers `2*inter/128`
        // N-tiles (gate=lower half, up=upper half) per expert.
        // `expert_offsets[e+1] - expert_offsets[e] == 0` ⇒ kernel
        // early-exits for that expert (`M_expert <= 0`).
        let max_m_tiles = 1u32;
        let gp = self
            .gate_ptrs_t
            .as_ref()
            .expect("k3_fused_gate_up_eligible() guarantees gate_ptrs_t");
        let up = self
            .up_ptrs_t
            .as_ref()
            .expect("k3_fused_gate_up_eligible() guarantees up_ptrs_t");
        let dp = self
            .down_ptrs_t
            .as_ref()
            .expect("k3_fused_gate_up_eligible() guarantees down_ptrs_t");
        ops::moe_w4a16_fused_gate_up_k64_n128(
            ctx.gpu,
            self.moe_fused_gate_up_t_k64,
            input,
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

        // ── 3. SiLU(gate) * up → expert_gate_out (in-place) ──────────
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            te * inter,
            stream,
        )?;

        // ── 4. Grouped down GEMM (transposed NVFP4) ──────────────────
        //
        // `moe_w4a16_grouped_gemm_ptrtable_n128` reads sorted-order
        // activations directly (sorted_token_ids = DevicePtr(0) →
        // identity mapping over the sorted output of step 2). N_OUT=h.
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

        // ── 5. Unpermute + weighted reduce → output ──────────────────
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_down_out,
            output,
            token_to_perm,
            weights_dev,
            h,
            n,
            top_k,
            stream,
        )?;

        // ── 6. Shared expert (if present) on default stream ──────────
        //
        // We deliberately run the shared expert sequentially on the
        // default stream (same as `forward_prefill` with overlap
        // disabled). At N=3 the cross-stream overhead dwarfs the
        // potential overlap win.
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let has_shared = shared_inter > 0;
        if has_shared {
            self.run_shared_expert_prefill(input, n, h, shared_inter, stream, stream, false, ctx)?;
            // ── 7. Blend shared expert into output ───────────────────
            //
            // moe_batched_blend computes
            //   out[t] += sigmoid(dot(input[t], gate_w)) * shared[t]
            // for each of the 3 tokens. Skip the gate when the model
            // has no shared_expert_gate (gate weight ptr == 0) — drop
            // back to plain residual_add.
            if self.weights.shared_expert_gate.weight.0 == 0 {
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add,
                    output,
                    shared_down_out,
                    n * h,
                    stream,
                )?;
            } else {
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    output,
                    shared_down_out,
                    input,
                    self.weights.shared_expert_gate.weight,
                    h,
                    n,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
