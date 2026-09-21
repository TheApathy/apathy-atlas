// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-shape DeepSeek-V4 prefill HC-finish + vanilla-RMSNorm arm.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::KvCacheDtype;

use super::super::super::{HcSiteWeights, Qwen3AttentionLayer};
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::weight_map::DenseWeight;

const TARGET_TOKENS: usize = 2_410;
const TARGET_HIDDEN: usize = 4_096;
const TARGET_HC: usize = 4;
const TARGET_MIX: usize = (2 + TARGET_HC) * TARGET_HC;
const TARGET_LAYERS: usize = 43;
const TARGET_SINKHORN_ITERS: usize = 20;

fn v4_hc_pre_finish_rms_fused_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_V4_PREFILL_HC_RMS_FUSED").as_deref() == Ok("1"))
}

fn v4_prefill_max_require_arms() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_PREFILL_MAX_REQUIRE_ARMS").as_deref() == Ok("1"))
}

pub(super) fn hc_tiled_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_HC_TILED").as_deref() != Ok("0"))
}

fn checked_bytes(extents: &[usize]) -> Option<u64> {
    extents.iter().try_fold(1_u64, |bytes, &extent| {
        bytes.checked_mul(u64::try_from(extent).ok()?)
    })
}

fn checked_range(pointer: DevicePtr, bytes: u64) -> Option<(u64, u64)> {
    if pointer.is_null() || bytes == 0 {
        return None;
    }
    Some((pointer.0, pointer.0.checked_add(bytes)?))
}

fn ranges_disjoint(left: (u64, u64), right: (u64, u64)) -> bool {
    left.1 <= right.0 || right.1 <= left.0
}

fn all_ranges_disjoint(writable: &[(u64, u64)], read_only: &[(u64, u64)]) -> bool {
    writable.iter().enumerate().all(|(index, &left)| {
        writable[index + 1..]
            .iter()
            .all(|&right| ranges_disjoint(left, right))
            && read_only.iter().all(|&right| ranges_disjoint(left, right))
    })
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_v4_hc_pre_finish_rms_fused(
        &self,
        site: &HcSiteWeights,
        norm_weight: &DenseWeight,
        streams: DevicePtr,
        hidden_out: DevicePtr,
        normed_out: DevicePtr,
        post_out: DevicePtr,
        comb_out: DevicePtr,
        mix_scratch: DevicePtr,
        num_tokens: usize,
        hidden_size: usize,
        hc_mult: usize,
        sinkhorn_iters: usize,
        hc_norm_eps: f32,
        hc_eps: f32,
        diagnostic: bool,
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
    ) -> Result<bool> {
        let requested = v4_hc_pre_finish_rms_fused_requested();
        let sizes = ctx.buffers.sizes();
        let stream_bytes = checked_bytes(&[TARGET_TOKENS, TARGET_HC, TARGET_HIDDEN, 4]);
        let fn_bytes = checked_bytes(&[TARGET_MIX, TARGET_HC, TARGET_HIDDEN, 4]);
        let mix_bytes = checked_bytes(&[TARGET_TOKENS, TARGET_MIX + 1, 4]);
        let hidden_bytes = checked_bytes(&[TARGET_TOKENS, TARGET_HIDDEN, 2]);
        let post_bytes = checked_bytes(&[TARGET_TOKENS, TARGET_HC, 4]);
        let comb_bytes = checked_bytes(&[TARGET_TOKENS, TARGET_HC, TARGET_HC, 4]);
        let scale_bytes = checked_bytes(&[3, 4]);
        let base_bytes = checked_bytes(&[TARGET_MIX, 4]);
        let weight_bytes = checked_bytes(&[TARGET_HIDDEN, 2]);

        let pointers_nonzero = [
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            norm_weight.weight,
            hidden_out,
            normed_out,
            post_out,
            comb_out,
            mix_scratch,
        ]
        .into_iter()
        .all(|pointer| !pointer.is_null());
        let pointers_aligned = [
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            norm_weight.weight,
            hidden_out,
            normed_out,
            post_out,
            comb_out,
            mix_scratch,
        ]
        .into_iter()
        .all(|pointer| pointer.0 & 3 == 0);
        let canonical_pointers = streams == ctx.buffers.hc_streams()
            && hidden_out == ctx.buffers.hidden_states()
            && normed_out == ctx.buffers.norm_output()
            && post_out == ctx.buffers.hc_post()
            && comb_out == ctx.buffers.hc_comb()
            && mix_scratch == ctx.buffers.expert_up_out();
        let ranges = stream_bytes
            .zip(fn_bytes)
            .zip(mix_bytes)
            .zip(hidden_bytes)
            .zip(post_bytes)
            .zip(comb_bytes)
            .zip(scale_bytes)
            .zip(base_bytes)
            .zip(weight_bytes)
            .and_then(
                |((((((((stream, function), mix), hidden), post), comb), scale), base), weight)| {
                    Some((
                        [
                            checked_range(mix_scratch, mix)?,
                            checked_range(hidden_out, hidden)?,
                            checked_range(normed_out, hidden)?,
                            checked_range(post_out, post)?,
                            checked_range(comb_out, comb)?,
                        ],
                        [
                            checked_range(streams, stream)?,
                            checked_range(site.hc_fn, function)?,
                            checked_range(site.hc_scale, scale)?,
                            checked_range(site.hc_base, base)?,
                            checked_range(norm_weight.weight, weight)?,
                        ],
                    ))
                },
            );
        let buffers_disjoint = ranges
            .as_ref()
            .is_some_and(|(writable, read_only)| all_ranges_disjoint(writable, read_only));
        let arena_fits = stream_bytes
            .zip(mix_bytes)
            .zip(hidden_bytes)
            .zip(post_bytes)
            .zip(comb_bytes)
            .is_some_and(|((((stream, mix), hidden), post), comb)| {
                u64::try_from(sizes.hc_streams).is_ok_and(|size| size >= stream)
                    && u64::try_from(sizes.expert_up_out).is_ok_and(|size| size >= mix)
                    && u64::try_from(sizes.hidden_states).is_ok_and(|size| size >= hidden)
                    && u64::try_from(sizes.norm_output).is_ok_and(|size| size >= hidden)
                    && u64::try_from(sizes.hc_post).is_ok_and(|size| size >= post)
                    && u64::try_from(sizes.hc_comb).is_ok_and(|size| size >= comb)
            });

        let engaged = requested
            && ctx.config.model_type == "deepseek_v4"
            && ctx.config.num_hidden_layers == TARGET_LAYERS
            && ctx.config.hc_mult == TARGET_HC
            && ctx.config.hc_sinkhorn_iters == TARGET_SINKHORN_ITERS
            && self.kv_dtype == KvCacheDtype::Fp8
            && self.norm_vanilla
            && num_tokens == TARGET_TOKENS
            && hidden_size == TARGET_HIDDEN
            && hc_mult == TARGET_HC
            && sinkhorn_iters == TARGET_SINKHORN_ITERS
            && ctx.config.tp_world_size.max(1) == 1
            && ctx.config.ep_world_size.max(1) == 1
            && ctx.comm.is_none()
            && !ctx.graph_capture
            && !diagnostic
            && !ctx.profile
            && batched_meta.is_none()
            && hc_tiled_enabled()
            && self.v4_hc_pre_finish_rms_fused_k.0 != 0
            && self.hc_pre_mix_tiled_k.0 != 0
            && hc_norm_eps.is_finite()
            && hc_norm_eps > 0.0
            && hc_norm_eps == (ctx.config.rms_norm_eps as f32)
            && hc_eps.is_finite()
            && hc_eps == 1.0e-6
            && (ctx.config.rms_norm_eps as f32).is_finite()
            && (ctx.config.rms_norm_eps as f32) > 0.0
            && pointers_nonzero
            && pointers_aligned
            && canonical_pointers
            && buffers_disjoint
            && arena_fits;
        if v4_prefill_max_require_arms() && requested {
            anyhow::ensure!(
                engaged,
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1 requires requested V4 arm hc_pre_finish_rms_fused to engage; refusing incumbent fallback"
            );
        }
        Ok(engaged)
    }

    // BEGIN V4 HC-finish/RMS mutation
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_v4_hc_pre_finish_rms_fused(
        &self,
        site: &HcSiteWeights,
        norm_weight: &DenseWeight,
        streams: DevicePtr,
        hidden_out: DevicePtr,
        normed_out: DevicePtr,
        post_out: DevicePtr,
        comb_out: DevicePtr,
        mix_scratch: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        hc_mult: u32,
        sinkhorn_iters: u32,
        hc_norm_eps: f32,
        hc_eps: f32,
        rms_eps: f32,
        site_name: &'static str,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        static V4_HC_RMS_ATTN_ENGAGED_LOGGED: std::sync::Once = std::sync::Once::new();
        static V4_HC_RMS_FFN_ENGAGED_LOGGED: std::sync::Once = std::sync::Once::new();
        let logged = match site_name {
            "attention" => &V4_HC_RMS_ATTN_ENGAGED_LOGGED,
            "ffn" => &V4_HC_RMS_FFN_ENGAGED_LOGGED,
            _ => anyhow::bail!("invalid V4 HC-finish/RMS site receipt: {site_name}"),
        };
        KernelLaunch::new(ctx.gpu, self.hc_pre_mix_tiled_k)
            .grid([num_tokens.div_ceil(32), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(site.hc_fn)
            .arg_ptr(mix_scratch)
            .arg_u32(num_tokens)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .launch(stream)?;
        KernelLaunch::new(ctx.gpu, self.v4_hc_pre_finish_rms_fused_k)
            .grid([num_tokens, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(mix_scratch)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(norm_weight.weight)
            .arg_ptr(hidden_out)
            .arg_ptr(normed_out)
            .arg_ptr(post_out)
            .arg_ptr(comb_out)
            .arg_u32(num_tokens)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(sinkhorn_iters)
            .arg_f32(hc_norm_eps)
            .arg_f32(hc_eps)
            .arg_f32(rms_eps)
            .launch(stream)?;

        logged.call_once(|| {
            tracing::info!(
                "V4_PREFILL_MAX_ARM_ENGAGED arm=hc_pre_finish_rms_fused site={} layer={} n={}",
                site_name,
                self.attn_layer_idx,
                num_tokens
            );
        });
        Ok(())
    }
    // END V4 HC-finish/RMS mutation
}
