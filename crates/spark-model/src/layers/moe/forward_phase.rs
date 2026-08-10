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

/// `ATLAS_MOE_GEMV_V2=1` wide-load decode tier: VEC=4 (128-byte warp requests
/// on the weight stream) with SPLIT=8, which lands on exactly the same CTA
/// count as the v2s4 default — grid.x halves, grid.z doubles — so the wider
/// request costs no occupancy. Default OFF until serve-validated.
const T_SPLIT_WIDE: u32 = 8;
const T_SPLIT_VEC_WIDE: u32 = 4;

/// `ATLAS_MOE_SPLITK_V2=1` wide-load tier for the multi-row (`_m`) verify
/// kernels: VEC=4 at the SAME SPLIT=4 (`_m{2,6,8}v4s4`), plus smem-staged
/// activations on the gate_up side. Unlike the single-row v4s8 tier this one
/// keeps the split points, so it is BIT-IDENTICAL to the v2s4 incumbent — the
/// multi-row grid has `num_tokens*top_k + 1` y-slots (37 at the 6-row verify vs
/// the single-row 7), so the CTAs VEC=4 gives up don't starve the SMs and no
/// SPLIT bump is needed to win them back. Default OFF until serve-validated.
const T_SPLIT_M_VEC_WIDE: u32 = 4;

static MOE_MROW_PARTITION_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static MOE_GEMV_V2_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static MOE_SPLITK_M_V2_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Kernel selection for the single-row unified-`_t` split-K decode.
pub(super) struct UnifiedTSplitCfg {
    pub split: u32,
    pub vec: u32,
    pub gate_up: KernelHandle,
    pub silu_down: KernelHandle,
}

/// Kernel selection for the multi-row (`_m`) dedup'd split-K verify.
struct SplitkMCfg {
    split: u32,
    /// Compile-time VEC of the selected entries — sizes grid.x, and marks the
    /// V2 tier (== [`T_SPLIT_M_VEC_WIDE`]), whose gate_up entries need the
    /// staged-activation smem.
    vec: u32,
    gate_up: KernelHandle,
    silu_down: KernelHandle,
    /// MROW of the compiled entry point (sets smem strides), NOT `num_tokens`.
    mrow: u32,
    partition: Option<SplitkMPartitionHandles>,
}

impl MoeLayer {
    /// Split-K config for this decode call, or `None` to use the non-split
    /// kernels.
    ///
    /// Split-K reassociates each output's dot product — the sum is over a fixed
    /// block order so decode stays bit-reproducible run to run, but it is not
    /// bit-equal to the single-sweep kernels. `ATLAS_MOE_SPLITK=0` turns it off
    /// to A/B that, or to reproduce a reference hash captured before it landed.
    ///
    /// `ATLAS_MOE_GEMV_V2=1` selects the `_v4s8` wide-load entries instead of
    /// `_v2s4` when the shape and target allow (falls back silently when not):
    /// same warp count, 128-byte weight requests instead of 64. SPLIT=8 cuts
    /// each dot product at different points than SPLIT=4, so the two tiers are
    /// reassociation-equivalent, not bit-equal; both stay bit-reproducible.
    fn unified_t_split_k(
        &self,
        ctx: &ForwardContext,
        h: u32,
        inter: u32,
        top_k: u32,
    ) -> Option<UnifiedTSplitCfg> {
        if std::env::var("ATLAS_MOE_SPLITK").as_deref() == Ok("0") {
            return None;
        }
        let base_finalize_ok = self.moe_gate_up_partial_finalize_k.0 != 0
            && self.moe_down_partial_finalize_k.0 != 0;
        if !base_finalize_ok {
            return None;
        }
        let eligible = |split: u32, vec: u32, gu: KernelHandle, dn: KernelHandle| {
            // The vector load/store serves whole VEC groups only, and each
            // block must cover a whole number of scale groups. Routed MXFP4 is
            // per-32, NVFP4 per-16, so `32 * split` covers both.
            let widths_ok = inter.is_multiple_of(ops::t_block() * vec)
                && h.is_multiple_of(ops::t_block() * vec);
            let depths_ok = h.is_multiple_of(32 * split) && inter.is_multiple_of(32 * split);
            let need = ops::moe_splitk_partial_bytes(split, inter, h, top_k);
            let space_ok = ctx.buffers.moe_splitk_partials_bytes() >= need;
            (widths_ok && depths_ok && space_ok && gu.0 != 0 && dn.0 != 0).then_some(
                UnifiedTSplitCfg {
                    split,
                    vec,
                    gate_up: gu,
                    silu_down: dn,
                },
            )
        };
        let wide = *MOE_GEMV_V2_ENABLED
            .get_or_init(|| std::env::var("ATLAS_MOE_GEMV_V2").as_deref() == Ok("1"));
        if wide
            && let (Some(gu), Some(dn)) = (
                self.e8m0_or_opt(
                    self.moe_expert_gate_up_shared_t_splitk8_k,
                    self.moe_expert_gate_up_shared_t_e8m0_splitk8_k,
                ),
                self.e8m0_or_opt(
                    self.moe_expert_silu_down_shared_t_splitk8_k,
                    self.moe_expert_silu_down_shared_t_e8m0_splitk8_k,
                ),
            )
            && let Some(cfg) = eligible(T_SPLIT_WIDE, T_SPLIT_VEC_WIDE, gu, dn)
        {
            return Some(cfg);
        }
        let gu = self.e8m0_or_opt(
            self.moe_expert_gate_up_shared_t_splitk_k,
            self.moe_expert_gate_up_shared_t_e8m0_splitk_k,
        )?;
        let dn = self.e8m0_or_opt(
            self.moe_expert_silu_down_shared_t_splitk_k,
            self.moe_expert_silu_down_shared_t_e8m0_splitk_k,
        )?;
        eligible(T_SPLIT, ops::T_SPLIT_VEC, gu, dn)
    }

    /// Split factor for a `num_tokens`-row dedup'd `_t` decode, or `None` when
    /// this shape/target can't take that path.
    ///
    /// Same shape and space rules as [`Self::unified_t_split_k`] — only the
    /// partial-buffer sizing differs, because the multi-row kernels keep one
    /// accumulator row per (slot, token) rather than per slot. Shares the
    /// `ATLAS_MOE_SPLITK=0` kill switch: that flag exists to A/B split-K
    /// reassociation, and the multi-row kernels reassociate the same way.
    fn unified_t_split_k_m(
        &self,
        ctx: &ForwardContext,
        h: u32,
        inter: u32,
        top_k: u32,
        num_tokens: u32,
    ) -> Option<SplitkMCfg> {
        if std::env::var("ATLAS_MOE_SPLITK").as_deref() == Ok("0") {
            return None;
        }
        // `ATLAS_MOE_SPLITK_M=0` turns off only the multi-row rewrite, so the
        // K=2 verify can be measured against the batch2_t fallback on one
        // binary without also disabling split-K for plain decode.
        if std::env::var("ATLAS_MOE_SPLITK_M").as_deref() == Ok("0") {
            return None;
        }
        let split = T_SPLIT;
        let widths_ok = |vec: u32| {
            inter.is_multiple_of(ops::t_block() * vec) && h.is_multiple_of(ops::t_block() * vec)
        };
        let depths_ok = h.is_multiple_of(32 * split) && inter.is_multiple_of(32 * split);
        if !(widths_ok(ops::T_SPLIT_VEC) && depths_ok) || num_tokens < 2 {
            return None;
        }
        // MROW is baked into the entry point, and a kernel whose MROW is below
        // `num_tokens` would silently drop the rows past it. `splitk_m_t_handles`
        // returns the narrowest compiled entry that covers this row count, or
        // `None` when nothing does.
        let partition_enabled = *MOE_MROW_PARTITION_ENABLED
            .get_or_init(|| std::env::var("ATLAS_MOE_MROW_PARTITION").as_deref() == Ok("1"));
        let v2_enabled = *MOE_SPLITK_M_V2_ENABLED
            .get_or_init(|| std::env::var("ATLAS_MOE_SPLITK_V2").as_deref() == Ok("1"));
        let (gate_up, silu_down, mrow, vec, partition) = if partition_enabled
            && let Some(handles) = self.splitk_m_t_partition_handles(num_tokens)
        {
            (
                handles.gate_unique,
                handles.down_unique,
                1,
                ops::T_SPLIT_VEC,
                Some(handles),
            )
        } else if v2_enabled
            && widths_ok(T_SPLIT_M_VEC_WIDE)
            && let Some((gu, sd, mrow)) = self.splitk_m_t_v2_handles(num_tokens)
        {
            // Same SPLIT, so the partial buffer sizing and both finalize
            // kernels are the incumbent's — only the GEMV entries and the
            // gate_up smem contract change.
            (gu, sd, mrow, T_SPLIT_M_VEC_WIDE, None)
        } else {
            let regular = self.splitk_m_t_handles(num_tokens)?;
            (regular.0, regular.1, regular.2, ops::T_SPLIT_VEC, None)
        };
        if self.moe_gate_up_partial_finalize_m_k.0 == 0 || self.moe_down_partial_finalize_m_k.0 == 0
        {
            return None;
        }
        let need = ops::moe_splitk_m_partial_bytes(split, inter, h, top_k, num_tokens);
        if ctx.buffers.moe_splitk_partials_bytes() < need {
            return None;
        }
        Some(SplitkMCfg {
            split,
            vec,
            gate_up,
            silu_down,
            mrow,
            partition,
        })
    }

    /// True when a 2-row verify would take the MROW=2 dedup'd split-K `_t`
    /// kernels rather than falling back to `batch2_t`.
    ///
    /// Callers use this to decide whether batching the verify MoE is worth it
    /// at all: `batch2_t` is the pre-split-K kernel shape and measured 17.0 vs
    /// 19.8 tok/s against the per-row loop, so "batch when we can" is only the
    /// right default while the fast path is the one that fires.
    pub fn k2_verify_ffn_is_batched(&self, ctx: &ForwardContext) -> bool {
        self.verify_ffn_is_batched(ctx, 2)
    }

    /// [`Self::k2_verify_ffn_is_batched`] for an arbitrary verify width.
    pub fn verify_ffn_is_batched(&self, ctx: &ForwardContext, num_tokens: u32) -> bool {
        if !self.use_t_layout_for_decode() {
            return false;
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        self.unified_t_split_k_m(ctx, h, inter, top_k, num_tokens)
            .is_some()
    }

    /// Run a `num_tokens`-row verify MoE through the dedup'd split-K `_t`
    /// kernels.
    ///
    /// Returns `Ok(false)` without touching any buffer when the path is not
    /// eligible, so the caller can fall back. On `Ok(true)` the outputs are in
    /// exactly the layout `batch2_t` would have produced — routed slots flat in
    /// `expert_gate_out` / `expert_up_out` / `expert_down_out`, one shared row
    /// per token in the shared scratch buffers — so the blend downstream is
    /// unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_splitk_m_t(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr,
        gate_t: &ExpertPtrTable,
        up_t: &ExpertPtrTable,
        down_t: &ExpertPtrTable,
        sh_gate_t: &QuantizedWeight,
        sh_up_t: &QuantizedWeight,
        sh_down_t: &QuantizedWeight,
        h: u32,
        inter: u32,
        top_k: u32,
        num_tokens: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(SplitkMCfg {
            split,
            vec,
            gate_up: gate_up_k,
            silu_down: silu_down_k,
            mrow,
            partition,
        }) = self.unified_t_split_k_m(ctx, h, inter, top_k, num_tokens)
        else {
            return Ok(false);
        };
        // V2 (`_v4s4`) gate_up entries stage the gathered rows' activation
        // slices in dynamic smem — MROW slices of K/SPLIT bf16 each. The
        // incumbent and the partition arms take none.
        let gate_stage_mrow = if vec == T_SPLIT_M_VEC_WIDE { mrow } else { 0 };
        // Same RIDER A1 precondition as the single-row unified_t path: the
        // `_e8m0` variants compute an NVFP4 shared expert alongside E8M0 routed
        // weights, so a native-MXFP4 shared expert would be misread.
        if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            self.shared_experts_scale_kind.expect(
                crate::weight_map::WeightQuantFormat::Nvfp4,
                "multi-row verify fused _e8m0 kernel assumes an NVFP4 shared expert",
            );
        }
        let partials = ctx.buffers.moe_splitk_partials();
        let down_partials = partials.offset(ops::moe_splitk_m_down_offset(
            split, inter, top_k, num_tokens,
        ));
        let gate_finalize = if partition.is_some() {
            self.moe_gate_up_partial_finalize_m_act_k
        } else {
            self.moe_gate_up_partial_finalize_m_k
        };
        ops::moe_expert_gate_up_shared_t_splitk_m(
            ctx.gpu,
            gate_up_k,
            gate_finalize,
            input,
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
            partition.map(|handles| handles.gate_duplicated),
            split,
            vec,
            gate_stage_mrow,
            inter,
            h,
            top_k,
            num_tokens,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_t_splitk_m(
            ctx.gpu,
            silu_down_k,
            self.moe_down_partial_finalize_m_k,
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
            partials,
            down_partials,
            partition.map(|handles| handles.down_buckets),
            split,
            vec,
            h,
            inter,
            top_k,
            num_tokens,
            // MROW of the compiled entry point, which sets the s_act stride.
            mrow,
            stream,
        )?;
        Ok(true)
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
        if let Some(cfg) = self.unified_t_split_k(ctx, h, inter, top_k) {
            let split = cfg.split;
            let partials = ctx.buffers.moe_splitk_partials();
            let down_partials = partials.offset(ops::moe_splitk_down_offset(split, inter, top_k));
            ops::moe_expert_gate_up_shared_t_splitk(
                ctx.gpu,
                cfg.gate_up,
                self.moe_gate_up_partial_finalize_k,
                cfg.vec,
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
                cfg.silu_down,
                self.moe_down_partial_finalize_k,
                cfg.vec,
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
