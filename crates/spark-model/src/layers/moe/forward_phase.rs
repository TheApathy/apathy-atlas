// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 8a unified-layout decode dispatch — hoisted from `forward.rs`
//! to keep that file under the 500 LoC cap.
//!
//! Single helper `dispatch_unified_t_decode` runs the gate+up and silu+down
//! kernels against transposed expert weight tables (gate_t / up_t / down_t
//! plus shared_*_t). Mirrors the inline `else if self.use_t_layout_for_decode()`
//! branch 1:1.

use anyhow::Result;

use super::*;

/// k-split the unified-`_t` decode path uses when eligible. Must not exceed
/// `spark_runtime::buffers::MOE_DECODE_MAX_SPLIT`, which sizes the partial
/// buffer, and must match the `_v{VEC}s{SPLIT}` entry points registered in
/// `init.rs`.
const T_SPLIT: u32 = 4;

impl MoeLayer {
    /// Split factor for this decode call, or `None` to use the non-split
    /// kernels.
    ///
    /// Split-K reassociates each output's dot product — the sum is over a fixed
    /// block order so decode stays bit-reproducible run to run, but it is not
    /// bit-equal to the single-sweep kernels. `ATLAS_MOE_SPLITK=0` turns it off
    /// to A/B that, or to reproduce a reference hash captured before it landed.
    fn unified_t_split_k(
        &self,
        ctx: &ForwardContext,
        h: u32,
        inter: u32,
        top_k: u32,
    ) -> Option<u32> {
        if std::env::var("ATLAS_MOE_SPLITK").as_deref() == Ok("0") {
            return None;
        }
        let split = T_SPLIT;
        let vec = ops::T_SPLIT_VEC;
        // The vector load/store serves whole VEC groups only, and each block
        // must cover a whole number of scale groups. Routed MXFP4 is per-32,
        // NVFP4 per-16, so `32 * split` covers both.
        let widths_ok = inter % (ops::T_BLOCK * vec) == 0 && h % (ops::T_BLOCK * vec) == 0;
        let depths_ok = h % (32 * split) == 0 && inter % (32 * split) == 0;
        let handles_ok = self.moe_gate_up_partial_finalize_k.0 != 0
            && self.moe_down_partial_finalize_k.0 != 0
            && self
                .e8m0_or_opt(
                    self.moe_expert_gate_up_shared_t_splitk_k,
                    self.moe_expert_gate_up_shared_t_e8m0_splitk_k,
                )
                .is_some()
            && self
                .e8m0_or_opt(
                    self.moe_expert_silu_down_shared_t_splitk_k,
                    self.moe_expert_silu_down_shared_t_e8m0_splitk_k,
                )
                .is_some();
        let need = ops::moe_splitk_partial_bytes(split, inter, h, top_k);
        let space_ok = ctx.buffers.moe_splitk_partials_bytes() >= need;
        if widths_ok && depths_ok && handles_ok && space_ok {
            Some(split)
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_unified_t_decode(
        &self,
        ctx: &ForwardContext,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr,
        h: u32,
        inter: u32,
        top_k: u32,
        stream: u64,
    ) -> Result<()> {
        // Phase 8a unified-layout decode path: transposed weight tables
        // for all three projections. Only fires when ATLAS_UNIFIED_MOE_LAYOUT=1
        // AND the weight loader has built persistent transposed copies for
        // gate / up / down (no lazy-scratch path).
        let gate_t = self
            .gate_ptrs_t
            .as_ref()
            .expect("gate_ptrs_t under unified_t");
        let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under unified_t");
        let down_t = self
            .down_ptrs_t
            .as_ref()
            .expect("down_ptrs_t under unified_t");
        let null_qw = QuantizedWeight::null();
        let sh_gate_t = self.shared_gate_t.as_ref().unwrap_or(&null_qw);
        let sh_up_t = self.shared_up_t.as_ref().unwrap_or(&null_qw);
        let sh_down_t = self.shared_down_t.as_ref().unwrap_or(&null_qw);
        // ARM-2 Phase-K RIDER A1: _e8m0 fused decode assumes NVFP4 shared expert.
        if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            self.shared_experts_scale_kind.expect(
                crate::weight_map::WeightQuantFormat::Nvfp4,
                "decode fused _e8m0 kernel assumes an NVFP4 shared expert",
            );
        }
        if let Some(split) = self.unified_t_split_k(ctx, h, inter, top_k) {
            let partials = ctx.buffers.moe_splitk_partials();
            let down_partials = partials.offset(ops::moe_splitk_down_offset(split, inter, top_k));
            ops::moe_expert_gate_up_shared_t_splitk(
                ctx.gpu,
                self.e8m0_or(
                    self.moe_expert_gate_up_shared_t_splitk_k,
                    self.moe_expert_gate_up_shared_t_e8m0_splitk_k,
                    "decode gate_up_shared_t split-K (unified_t)",
                ),
                self.moe_gate_up_partial_finalize_k,
                expert_input,
                gate_t.packed_ptrs,
                gate_t.scale_ptrs,
                gate_t.scale2_vals,
                expert_gate_out,
                up_t.packed_ptrs,
                up_t.scale_ptrs,
                up_t.scale2_vals,
                expert_up_out,
                indices_dev,
                sh_gate_t,
                shared_gate_scratch,
                sh_up_t,
                shared_up_scratch,
                partials,
                split,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_t_splitk(
                ctx.gpu,
                self.e8m0_or(
                    self.moe_expert_silu_down_shared_t_splitk_k,
                    self.moe_expert_silu_down_shared_t_e8m0_splitk_k,
                    "decode silu_down_shared_t split-K (unified_t)",
                ),
                self.moe_down_partial_finalize_k,
                expert_gate_out,
                expert_up_out,
                down_t.packed_ptrs,
                down_t.scale_ptrs,
                down_t.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                sh_down_t,
                shared_out,
                down_partials,
                split,
                h,
                inter,
                top_k,
                stream,
            )?;
            return Ok(());
        }
        ops::moe_expert_gate_up_shared_t(
            ctx.gpu,
            self.e8m0_or(
                self.moe_expert_gate_up_shared_t_k,
                self.moe_expert_gate_up_shared_t_e8m0_k,
                "decode gate_up_shared_t (unified_t)",
            ),
            expert_input,
            gate_t.packed_ptrs,
            gate_t.scale_ptrs,
            gate_t.scale2_vals,
            expert_gate_out,
            up_t.packed_ptrs,
            up_t.scale_ptrs,
            up_t.scale2_vals,
            expert_up_out,
            indices_dev,
            sh_gate_t,
            shared_gate_scratch,
            sh_up_t,
            shared_up_scratch,
            inter,
            h,
            top_k,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_t(
            ctx.gpu,
            self.e8m0_or(
                self.moe_expert_silu_down_shared_t_k,
                self.moe_expert_silu_down_shared_t_e8m0_k,
                "decode silu_down_shared_t (unified_t)",
            ),
            expert_gate_out,
            expert_up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            sh_down_t,
            shared_out,
            h,
            inter,
            top_k,
            stream,
        )?;
        Ok(())
    }
}
