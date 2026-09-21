// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 prefill output tails. The combined shared-expert path is deliberately
//! exact-shape and opt-in until its GPU byte-parity/timing promotion gate runs.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::*;

const H128_BLOCK: u32 = 256;
const H128_COLS_PER_BLOCK: u32 = 1024;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_exl3_fused_post_unpermute(
        &self,
        expert_output: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        topk_weights: DevicePtr,
        sorted_expert_ids: DevicePtr,
        hidden_size: u32,
        num_tokens: u32,
        topk: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(st) = self.exl3.as_ref() else {
            return Ok(false);
        };
        if !st.prefill.fused_unpermute {
            return Ok(false);
        }
        ensure!(
            !st.prefill.prefill_max_require_arms
                || (st.prefill.hrow_fixed_shape
                    && st.down.bits == 2
                    && hidden_size == 4096
                    && num_tokens == 2410
                    && topk == 6
                    && !ctx.graph_capture
                    && st.prefill.h128_post_unpermute_h4096_k.0 != 0
                    && !expert_output.is_null()
                    && !output.is_null()
                    && !token_to_perm.is_null()
                    && !topk_weights.is_null()
                    && !sorted_expert_ids.is_null()
                    && !st.down.svh_tab.is_null()),
            "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requested fused unpermute tail cannot engage"
        );
        ensure!(
            hidden_size.is_multiple_of(128),
            "EXL3 fused post-unpermute requires H divisible by 128, got {hidden_size}"
        );
        let fixed_h4096 = st.prefill.hrow_fixed_shape && st.down.bits == 2 && hidden_size == 4096;
        let kernel = if fixed_h4096 {
            st.prefill.h128_post_unpermute_h4096_k
        } else {
            st.prefill.h128_post_unpermute_k
        };
        let grid = if fixed_h4096 {
            [num_tokens, 4, 1]
        } else {
            [num_tokens, hidden_size.div_ceil(H128_COLS_PER_BLOCK), 1]
        };
        KernelLaunch::new(ctx.gpu, kernel)
            .grid(grid)
            .block([H128_BLOCK, 1, 1])
            .arg_ptr(expert_output)
            .arg_ptr(output)
            .arg_ptr(token_to_perm)
            .arg_ptr(topk_weights)
            .arg_ptr(sorted_expert_ids)
            .arg_ptr(st.down.svh_tab)
            .arg_u32(hidden_size)
            .arg_u32(num_tokens)
            .arg_u32(topk)
            .launch(stream)?;
        if st.prefill.prefill_max_require_arms {
            static RECEIPT: std::sync::Once = std::sync::Once::new();
            RECEIPT.call_once(|| {
                tracing::info!(
                    "ATLAS_PREFILL_MAX_ARMS_RECEIPT tail=fused_unpermute n_tokens={num_tokens} hidden={hidden_size} top_k={topk}"
                );
            });
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_exl3_fused_post_unpermute_blend(
        &self,
        expert_output: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        topk_weights: DevicePtr,
        sorted_expert_ids: DevicePtr,
        shared_out: DevicePtr,
        normed: DevicePtr,
        gate_weight: DevicePtr,
        hidden_size: u32,
        num_tokens: u32,
        topk: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(st) = self.exl3.as_ref() else {
            return Ok(false);
        };
        let pf = &st.prefill;
        let unavailable = !pf.fused_blend
            || !pf.fixed_shape
            || !pf.hrow_fixed_shape
            || st.down.bits != 2
            || hidden_size != 4096
            || num_tokens == 0
            || topk != 6
            || ctx.graph_capture
            || pf.h128_post_unpermute_blend_h4096_k.0 == 0
            || expert_output.is_null()
            || output.is_null()
            || token_to_perm.is_null()
            || topk_weights.is_null()
            || sorted_expert_ids.is_null()
            || st.down.svh_tab.is_null()
            || shared_out.is_null()
            || normed.is_null();
        ensure!(
            !(pf.prefill_max_require_arms && pf.fused_blend_requested)
                || (!unavailable && num_tokens == 2410),
            "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requested fused blend tail cannot engage"
        );
        if unavailable {
            return Ok(false);
        }

        KernelLaunch::new(ctx.gpu, pf.h128_post_unpermute_blend_h4096_k)
            .grid([num_tokens, 1, 1])
            .block([H128_BLOCK, 1, 1])
            .arg_ptr(expert_output)
            .arg_ptr(output)
            .arg_ptr(token_to_perm)
            .arg_ptr(topk_weights)
            .arg_ptr(sorted_expert_ids)
            .arg_ptr(st.down.svh_tab)
            .arg_ptr(shared_out)
            .arg_ptr(normed)
            .arg_ptr(gate_weight)
            .arg_u32(hidden_size)
            .arg_u32(num_tokens)
            .arg_u32(topk)
            .launch(stream)?;
        if pf.prefill_max_require_arms {
            static RECEIPT: std::sync::Once = std::sync::Once::new();
            RECEIPT.call_once(|| {
                tracing::info!(
                    "ATLAS_PREFILL_MAX_ARMS_RECEIPT tail=fused_blend n_tokens={num_tokens} hidden={hidden_size} top_k={topk}"
                );
            });
        }
        Ok(true)
    }
}
