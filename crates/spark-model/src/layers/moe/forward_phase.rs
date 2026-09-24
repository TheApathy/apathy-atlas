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

/// `ATLAS_MOE_T_LANES=1` routes unified-layout decode through the
/// lane-parallel `_t_lanes` kernels, which match the untransposed decode
/// kernels bit for bit (the default `_t` kernels do not).
fn t_lanes_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_MOE_T_LANES").ok().as_deref() == Some("1");
        if on {
            tracing::info!("ENGAGED ATLAS_MOE_T_LANES: exact lane-parallel transposed decode MoE");
        }
        on
    })
}

impl MoeLayer {
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
        let lanes = t_lanes_selected();
        if lanes {
            anyhow::ensure!(
                self.moe_expert_gate_up_shared_t_lanes_k.0 != 0
                    && self.moe_expert_silu_down_shared_t_lanes_k.0 != 0,
                "ATLAS_MOE_T_LANES=1 but this target's moe_shared_expert_fused_t module has no _t_lanes kernels"
            );
        }
        let (gate_up_k, silu_down_k, out_block, threads) = if lanes {
            (
                self.moe_expert_gate_up_shared_t_lanes_k,
                self.moe_expert_silu_down_shared_t_lanes_k,
                ops::T_LANES_OUT_BLOCK,
                ops::T_LANES_THREADS,
            )
        } else {
            (
                self.moe_expert_gate_up_shared_t_k,
                self.moe_expert_silu_down_shared_t_k,
                ops::T_BLOCK,
                ops::T_BLOCK,
            )
        };
        ops::moe_expert_gate_up_shared_t_shape(
            ctx.gpu,
            gate_up_k,
            out_block,
            threads,
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
        ops::moe_expert_silu_down_shared_t_shape(
            ctx.gpu,
            silu_down_k,
            out_block,
            threads,
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
