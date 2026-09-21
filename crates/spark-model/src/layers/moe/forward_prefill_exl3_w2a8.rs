// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off exact DeepSeek K2 W2A8 routed-prefill chain.
//!
//! Eligibility and every dynamic arena bound are decided before the first
//! producer write. Once mutation starts, launch failures propagate; falling
//! back into the incumbent alias schedule would consume overwritten buffers.

use anyhow::ensure;
use spark_runtime::kernel_args::KernelLaunch;

use super::*;

const W2A8_GROUP_K: usize = 128;
const W2A8_GATE_UP_N: u32 = 2048;
const W2A8_GATE_UP_K: u32 = 4096;
const W2A8_DOWN_N: u32 = 4096;
const W2A8_DOWN_K: u32 = 2048;
const W2A8_TOP_K: u32 = 6;
const W2A8_TARGET_TOTAL_EXPANDED: u32 = 2_410 * W2A8_TOP_K;

#[derive(Clone, Copy)]
struct SidecarLayout {
    fp8_bytes: usize,
    total_bytes: usize,
}

fn checked_sidecar_layout(rows: u32, k: u32) -> Option<SidecarLayout> {
    let rows = rows as usize;
    let k = k as usize;
    let fp8_bytes = rows.checked_mul(k)?;
    let scale_bytes = rows
        .checked_mul(k.checked_div(W2A8_GROUP_K)?)?
        .checked_mul(4)?;
    Some(SidecarLayout {
        fp8_bytes,
        total_bytes: fp8_bytes.checked_add(scale_bytes)?,
    })
}

fn checked_bf16_bytes(rows: u32, columns: u32) -> Option<usize> {
    (rows as usize)
        .checked_mul(columns as usize)?
        .checked_mul(2)
}

fn checked_offset(base: DevicePtr, bytes: usize) -> Option<DevicePtr> {
    Some(DevicePtr(base.0.checked_add(u64::try_from(bytes).ok()?)?))
}

fn checked_range(base: DevicePtr, bytes: usize) -> Option<(u64, u64)> {
    let end = base.0.checked_add(u64::try_from(bytes).ok()?)?;
    (!base.is_null() && end > base.0).then_some((base.0, end))
}

fn ranges_disjoint(left: (u64, u64), right: (u64, u64)) -> bool {
    left.1 <= right.0 || right.1 <= left.0
}

fn pointer_aligned(pointer: DevicePtr, alignment: u64) -> bool {
    !pointer.is_null() && pointer.0.is_multiple_of(alignment)
}

#[derive(Clone, Copy)]
struct W2a8Plan {
    gate_fp8: DevicePtr,
    gate_scale: DevicePtr,
    up_fp8: DevicePtr,
    up_scale: DevicePtr,
    down_fp8: DevicePtr,
    down_scale: DevicePtr,
    expert_gate_out: DevicePtr,
    expert_down_out: DevicePtr,
    gu_grid: u32,
    fused_gu_grid: u32,
    fused_gu_n256_grid: u32,
    down_grid: u32,
    n256_down_grid: u32,
    use_fused_gu: bool,
    use_fused_gu_n256: bool,
    use_n256_down: bool,
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_run_exl3_w2a8_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        sorted_expert_ids: DevicePtr,
        num_experts: u32,
        total_expanded: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(st) = self.exl3.as_ref() else {
            return Ok(false);
        };
        let pf = &st.prefill;
        // This flag is used only by the single-pass qualification launcher.
        // Treat any prefill shape drift as a failed receipt, including a
        // scheduler/request mismatch that no longer yields N=2410.
        let require_exact_max = pf.prefill_max_require_arms;
        macro_rules! decline {
            ($reason:literal) => {{
                ensure!(
                    !require_exact_max,
                    "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 exact N=2410 request: base W2A8 arm cannot engage ({})",
                    $reason
                );
                return Ok(false);
            }};
        }
        ensure!(
            !require_exact_max
                || (total_expanded == W2A8_TARGET_TOTAL_EXPANDED
                    && num_experts == 256
                    && top_k == W2A8_TOP_K),
            "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 qualification shape drift: expected n_tokens=2410 total_expanded=14460 experts=256 top_k=6; got total_expanded={total_expanded} experts={num_experts} top_k={top_k}"
        );
        let handles_present = pf.w2a8_pre_dual_emit_k.0 != 0
            && pf.w2a8_post_silu_pre_emit_k.0 != 0
            && pf.w2a8_grouped_gu_k.0 != 0
            && pf.w2a8_grouped_down_k.0 != 0;
        let exact_shape = st.gate.bits == 2
            && st.up.bits == 2
            && st.down.bits == 2
            && st.gate.n == W2A8_GATE_UP_N
            && st.gate.k == W2A8_GATE_UP_K
            && st.up.n == W2A8_GATE_UP_N
            && st.up.k == W2A8_GATE_UP_K
            && st.down.n == W2A8_DOWN_N
            && st.down.k == W2A8_DOWN_K;
        let topology_supported = top_k == W2A8_TOP_K
            && ctx.config.tp_world_size.max(1) == 1
            && ctx.config.ep_world_size.max(1) == 1
            && ctx.comm.is_none();
        let pointers_present = !expert_input.is_null()
            && !expert_offsets.is_null()
            && !sorted_token_ids.is_null()
            && !sorted_expert_ids.is_null()
            && !st.gate.trellis_tab.is_null()
            && !st.gate.suh_tab.is_null()
            && !st.gate.svh_tab.is_null()
            && !st.up.trellis_tab.is_null()
            && !st.up.suh_tab.is_null()
            && !st.up.svh_tab.is_null()
            && !st.down.trellis_tab.is_null()
            && !st.down.suh_tab.is_null()
            && !st.down.svh_tab.is_null();
        if !pf.w2a8_requested
            || !pf.direct
            || !pf.persistent
            || !pf.fixed_shape
            || !pf.fixed_k2
            || pf.direct_m128
            || pf.direct_n128
            || pf.direct_n256
            || !pf.dual_pre
            || !pf.fused_post
            || !exact_shape
            || !topology_supported
            || ctx.graph_capture
            || !handles_present
            || !pointers_present
            || total_expanded == 0
            || num_experts == 0
        {
            decline!("configuration, topology, handle, or pointer eligibility");
        }

        let Some(gu) = checked_sidecar_layout(total_expanded, W2A8_GATE_UP_K) else {
            decline!("gate/up sidecar size overflow");
        };
        let Some(down) = checked_sidecar_layout(total_expanded, W2A8_DOWN_K) else {
            decline!("down sidecar size overflow");
        };
        let Some(gu_output_bytes) = checked_bf16_bytes(total_expanded, W2A8_GATE_UP_N) else {
            decline!("gate/up output size overflow");
        };
        let Some(down_output_bytes) = checked_bf16_bytes(total_expanded, W2A8_DOWN_N) else {
            decline!("down output size overflow");
        };
        let sizes = ctx.buffers.sizes();
        if sizes.expert_gate_out < gu_output_bytes
            || sizes.expert_up_out < gu.total_bytes.max(down.total_bytes)
            || sizes.expert_down_out < gu.total_bytes.max(gu_output_bytes).max(down_output_bytes)
        {
            decline!("arena capacity");
        }

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        if expert_gate_out.is_null() || expert_up_out.is_null() || expert_down_out.is_null() {
            decline!("arena pointers");
        }
        let gate_fp8 = expert_down_out;
        let up_fp8 = expert_up_out;
        let Some(gate_scale) = checked_offset(gate_fp8, gu.fp8_bytes) else {
            decline!("gate scale pointer overflow");
        };
        let Some(up_scale) = checked_offset(up_fp8, gu.fp8_bytes) else {
            decline!("up scale pointer overflow");
        };
        let incumbent_down_fp8 = expert_up_out;
        let Some(incumbent_down_scale) = checked_offset(incumbent_down_fp8, down.fp8_bytes) else {
            decline!("incumbent down scale pointer overflow");
        };
        let fused_down_fp8 = expert_gate_out;
        let Some(fused_down_scale) = checked_offset(fused_down_fp8, down.fp8_bytes) else {
            decline!("fused down scale pointer overflow");
        };
        let fused_ranges = (
            checked_range(gate_fp8, gu.total_bytes),
            checked_range(up_fp8, gu.total_bytes),
            checked_range(fused_down_fp8, down.total_bytes),
        );
        let fused_ranges_disjoint = match fused_ranges {
            (Some(gate), Some(up), Some(down)) => {
                ranges_disjoint(gate, up)
                    && ranges_disjoint(gate, down)
                    && ranges_disjoint(up, down)
            }
            _ => false,
        };
        let fused_pointers_aligned = [gate_fp8, up_fp8, fused_down_fp8]
            .into_iter()
            .all(|pointer| pointer_aligned(pointer, 16))
            && [gate_scale, up_scale, fused_down_scale, expert_offsets]
                .into_iter()
                .all(|pointer| pointer_aligned(pointer, 4))
            && [
                st.gate.trellis_tab,
                st.up.trellis_tab,
                st.gate.svh_tab,
                st.up.svh_tab,
                st.down.suh_tab,
            ]
            .into_iter()
            .all(|pointer| pointer_aligned(pointer, 8));
        let use_fused_gu = pf.w2a8_fused_gu_down_requested
            && pf.w2a8_fused_gu_down_emit_n128_k.0 != 0
            && num_experts == 256
            && fused_ranges_disjoint
            && fused_pointers_aligned;
        let use_fused_gu_n256 = pf.w2a8_fused_gu_down_n256_requested
            && pf.w2a8_fused_gu_down_emit_n256_k.0 != 0
            && total_expanded == W2A8_TARGET_TOTAL_EXPANDED
            && num_experts == 256
            && fused_ranges_disjoint
            && fused_pointers_aligned;
        let use_any_fused_gu = use_fused_gu_n256 || use_fused_gu;
        let (down_fp8, down_scale) = if use_any_fused_gu {
            (fused_down_fp8, fused_down_scale)
        } else {
            (incumbent_down_fp8, incumbent_down_scale)
        };
        let n256_ranges_disjoint = match (
            checked_range(down_fp8, down.total_bytes),
            checked_range(expert_down_out, down_output_bytes),
        ) {
            (Some(input), Some(output)) => ranges_disjoint(input, output),
            _ => false,
        };
        let n256_pointers_aligned = pointer_aligned(down_fp8, 16)
            && pointer_aligned(down_scale, 4)
            && pointer_aligned(st.down.trellis_tab, 8)
            && pointer_aligned(expert_down_out, 4)
            && pointer_aligned(expert_offsets, 4);
        let use_n256_down = pf.w2a8_n256_down_requested
            && pf.w2a8_grouped_n256_down_k.0 != 0
            && num_experts == 256
            && n256_ranges_disjoint
            && n256_pointers_aligned;
        let Some(gu_grid) = num_experts.checked_mul(W2A8_GATE_UP_N / 64) else {
            decline!("gate/up grid overflow");
        };
        let Some(fused_gu_grid) = num_experts.checked_mul(W2A8_GATE_UP_N / 128) else {
            decline!("fused gate/up N128 grid overflow");
        };
        let Some(fused_gu_n256_grid) = num_experts.checked_mul(W2A8_GATE_UP_N / 256) else {
            decline!("fused gate/up N256 grid overflow");
        };
        let Some(down_grid) = num_experts.checked_mul(W2A8_DOWN_N / 64) else {
            decline!("down grid overflow");
        };
        let Some(n256_down_grid) = num_experts.checked_mul(W2A8_DOWN_N / 256) else {
            decline!("N256 down grid overflow");
        };
        let plan = W2a8Plan {
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            down_fp8,
            down_scale,
            expert_gate_out,
            expert_down_out,
            gu_grid,
            fused_gu_grid,
            fused_gu_n256_grid,
            down_grid,
            n256_down_grid,
            use_fused_gu,
            use_fused_gu_n256,
            use_n256_down,
        };
        if require_exact_max {
            ensure!(
                plan.use_fused_gu && !plan.use_fused_gu_n256,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 exact N=2410 request: fused GU N128 arm cannot engage"
            );
            ensure!(
                plan.use_n256_down,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 exact N=2410 request: N256 down arm cannot engage"
            );
            ensure!(
                !pf.fused_blend_requested
                    || (ctx.config.shared_expert_intermediate_size > 0 && !super::dump::enabled()),
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 exact N=2410 request: requested fused blend tail cannot engage"
            );
        }

        // MUTATION START: every decline and checked calculation is above.
        KernelLaunch::new(ctx.gpu, pf.w2a8_pre_dual_emit_k)
            .grid([total_expanded, 4, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(sorted_token_ids)
            .arg_ptr(sorted_expert_ids)
            .arg_ptr(st.gate.suh_tab)
            .arg_ptr(st.up.suh_tab)
            .arg_ptr(plan.gate_fp8)
            .arg_ptr(plan.gate_scale)
            .arg_ptr(plan.up_fp8)
            .arg_ptr(plan.up_scale)
            .arg_u32(W2A8_GATE_UP_K)
            .arg_u32(total_expanded)
            .launch(stream)?;
        if plan.use_fused_gu_n256 {
            KernelLaunch::new(ctx.gpu, pf.w2a8_fused_gu_down_emit_n256_k)
                .grid([plan.fused_gu_n256_grid, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(plan.gate_fp8)
                .arg_ptr(plan.gate_scale)
                .arg_ptr(plan.up_fp8)
                .arg_ptr(plan.up_scale)
                .arg_ptr(st.gate.trellis_tab)
                .arg_ptr(st.up.trellis_tab)
                .arg_ptr(st.gate.svh_tab)
                .arg_ptr(st.up.svh_tab)
                .arg_ptr(st.down.suh_tab)
                .arg_ptr(plan.down_fp8)
                .arg_ptr(plan.down_scale)
                .arg_ptr(expert_offsets)
                .arg_u32(num_experts)
                .arg_u32(total_expanded)
                .arg_u32(W2A8_GATE_UP_N)
                .arg_u32(W2A8_GATE_UP_K)
                .arg_u32(2)
                .arg_u32(1)
                .launch(stream)?;
        } else if plan.use_fused_gu {
            KernelLaunch::new(ctx.gpu, pf.w2a8_fused_gu_down_emit_n128_k)
                .grid([plan.fused_gu_grid, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(plan.gate_fp8)
                .arg_ptr(plan.gate_scale)
                .arg_ptr(plan.up_fp8)
                .arg_ptr(plan.up_scale)
                .arg_ptr(st.gate.trellis_tab)
                .arg_ptr(st.up.trellis_tab)
                .arg_ptr(st.gate.svh_tab)
                .arg_ptr(st.up.svh_tab)
                .arg_ptr(st.down.suh_tab)
                .arg_ptr(plan.down_fp8)
                .arg_ptr(plan.down_scale)
                .arg_ptr(expert_offsets)
                .arg_u32(num_experts)
                .arg_u32(total_expanded)
                .arg_u32(W2A8_GATE_UP_N)
                .arg_u32(W2A8_GATE_UP_K)
                .arg_u32(2)
                .arg_u32(1)
                .launch(stream)?;
        } else {
            KernelLaunch::new(ctx.gpu, pf.w2a8_grouped_gu_k)
                .grid([plan.gu_grid, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(plan.gate_fp8)
                .arg_ptr(plan.gate_scale)
                .arg_ptr(st.gate.trellis_tab)
                .arg_ptr(plan.expert_gate_out)
                .arg_ptr(expert_offsets)
                .arg_u32(num_experts)
                .arg_u32(W2A8_GATE_UP_N)
                .arg_u32(W2A8_GATE_UP_K)
                .arg_u32(2)
                .arg_u32(1)
                .launch(stream)?;
            KernelLaunch::new(ctx.gpu, pf.w2a8_grouped_gu_k)
                .grid([plan.gu_grid, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(plan.up_fp8)
                .arg_ptr(plan.up_scale)
                .arg_ptr(st.up.trellis_tab)
                .arg_ptr(plan.expert_down_out)
                .arg_ptr(expert_offsets)
                .arg_u32(num_experts)
                .arg_u32(W2A8_GATE_UP_N)
                .arg_u32(W2A8_GATE_UP_K)
                .arg_u32(2)
                .arg_u32(1)
                .launch(stream)?;
            KernelLaunch::new(ctx.gpu, pf.w2a8_post_silu_pre_emit_k)
                .grid([total_expanded, 2, 1])
                .block([256, 1, 1])
                .arg_ptr(plan.expert_gate_out)
                .arg_ptr(plan.expert_down_out)
                .arg_ptr(sorted_expert_ids)
                .arg_ptr(st.gate.svh_tab)
                .arg_ptr(st.up.svh_tab)
                .arg_ptr(st.down.suh_tab)
                .arg_ptr(plan.down_fp8)
                .arg_ptr(plan.down_scale)
                .arg_u32(W2A8_DOWN_K)
                .arg_u32(total_expanded)
                .launch(stream)?;
        }
        if plan.use_n256_down {
            KernelLaunch::new(ctx.gpu, pf.w2a8_grouped_n256_down_k)
                .grid([plan.n256_down_grid, 1, 1])
                .block([512, 1, 1])
                .arg_ptr(plan.down_fp8)
                .arg_ptr(plan.down_scale)
                .arg_ptr(st.down.trellis_tab)
                .arg_ptr(plan.expert_down_out)
                .arg_ptr(expert_offsets)
                .arg_u32(num_experts)
                .arg_u32(total_expanded)
                .arg_u32(W2A8_DOWN_N)
                .arg_u32(W2A8_DOWN_K)
                .arg_u32(2)
                .arg_u32(1)
                .launch(stream)?;
        } else {
            KernelLaunch::new(ctx.gpu, pf.w2a8_grouped_down_k)
                .grid([plan.down_grid, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(plan.down_fp8)
                .arg_ptr(plan.down_scale)
                .arg_ptr(st.down.trellis_tab)
                .arg_ptr(plan.expert_down_out)
                .arg_ptr(expert_offsets)
                .arg_u32(num_experts)
                .arg_u32(W2A8_DOWN_N)
                .arg_u32(W2A8_DOWN_K)
                .arg_u32(2)
                .arg_u32(1)
                .launch(stream)?;
        }
        if require_exact_max {
            static RECEIPT: std::sync::Once = std::sync::Once::new();
            if plan.use_fused_gu_n256 {
                RECEIPT.call_once(|| {
                    tracing::info!(
                        "ATLAS_PREFILL_MAX_ARMS_RECEIPT core=w2a8 fused_gu=n256 down=n256 n_tokens=2410 total_expanded=14460 experts=256 top_k=6"
                    );
                });
            } else {
                RECEIPT.call_once(|| {
                    tracing::info!(
                        "ATLAS_PREFILL_MAX_ARMS_RECEIPT core=w2a8 fused_gu=n128 down=n256 n_tokens=2410 total_expanded=14460 experts=256 top_k=6"
                    );
                });
            }
        }
        Ok(true)
    }
}
