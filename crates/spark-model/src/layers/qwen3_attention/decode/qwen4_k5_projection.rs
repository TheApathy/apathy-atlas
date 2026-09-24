// SPDX-License-Identifier: AGPL-3.0-only

//! Exact K1-order Qwen4 K=5 attention projection staging.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

const QWEN4_K16_EXACT_ENV: &str = "ATLAS_QWEN4_K16_EXACT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen4ExactQkvRowsRoute {
    LegacyK5K9,
    K16,
    K32,
}

const QWEN4_K16_EXACT_RECEIPT: &str = "ENGAGED ATLAS_QWEN4_K16_EXACT: M16 qg+dual_kv+qk_norm";
const QWEN4_K32_EXACT_RECEIPT: &str =
    "ENGAGED ATLAS_QWEN4_PREFILL_ATTN_CORE32: two ordered M16 qg+dual_kv projections";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen4QkvProjectionExtents {
    q_dim: u32,
    q_proj_dim: u32,
    kv_dim: u32,
    q_proj_bytes: usize,
    kv_bytes: usize,
    row_stride_bf16: u32,
    row_bytes: usize,
    total_bytes: usize,
}

const fn qwen4_k16_target_attention_geometry(
    is_qwen4_exp: bool,
    hidden_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> bool {
    is_qwen4_exp && hidden_size == 2560 && num_q_heads == 24 && num_kv_heads == 2 && head_dim == 256
}

fn qwen4_qkv_projection_extents(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rows: usize,
) -> Option<Qwen4QkvProjectionExtents> {
    let q_dim = num_q_heads.checked_mul(head_dim)?;
    let q_proj_dim = q_dim.checked_mul(2)?;
    let kv_dim = num_kv_heads.checked_mul(head_dim)?;
    let row_stride_bf16 = q_proj_dim.checked_add(kv_dim.checked_mul(2)?)?;
    let q_proj_bytes = (q_proj_dim as usize).checked_mul(2)?;
    let kv_bytes = (kv_dim as usize).checked_mul(2)?;
    let row_bytes = (row_stride_bf16 as usize).checked_mul(2)?;
    let total_bytes = rows.checked_mul(row_bytes)?;
    Some(Qwen4QkvProjectionExtents {
        q_dim,
        q_proj_dim,
        kv_dim,
        q_proj_bytes,
        kv_bytes,
        row_stride_bf16,
        row_bytes,
        total_bytes,
    })
}

const fn qwen4_exact_qkv_receipt(route: Qwen4ExactQkvRowsRoute) -> Option<&'static str> {
    match route {
        Qwen4ExactQkvRowsRoute::LegacyK5K9 => None,
        Qwen4ExactQkvRowsRoute::K16 => Some(QWEN4_K16_EXACT_RECEIPT),
        Qwen4ExactQkvRowsRoute::K32 => Some(QWEN4_K32_EXACT_RECEIPT),
    }
}

/// Keep the already-qualified K5/K9 switch independent from the experimental
/// gamma-15 route. In particular, enabling either switch must not widen the
/// other one's admitted row set.
const fn qwen4_exact_qkv_rows_route(
    rows: usize,
    legacy_k5_k9_enabled: bool,
    k16_enabled: bool,
) -> Option<Qwen4ExactQkvRowsRoute> {
    match rows {
        5 | 9 if legacy_k5_k9_enabled => Some(Qwen4ExactQkvRowsRoute::LegacyK5K9),
        16 if k16_enabled => Some(Qwen4ExactQkvRowsRoute::K16),
        _ => None,
    }
}

fn parse_exact_k16_env_value(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{QWEN4_K16_EXACT_ENV} must be exactly 0 or 1, got {other:?}"),
    }
}

fn qwen4_k16_exact_requested() -> Result<bool> {
    match std::env::var(QWEN4_K16_EXACT_ENV) {
        Ok(value) => parse_exact_k16_env_value(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_exact_k16_env_value(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{QWEN4_K16_EXACT_ENV} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

impl Qwen3AttentionLayer {
    /// Projects an admitted set of contiguous Qwen4 attention inputs into a strided
    /// `[Q|gate|K|V]` arena. The projection kernels preserve the ordinary K1
    /// reduction order; Q/K normalization remains the same per-head K1 call.
    ///
    /// Returns `None` unless the explicit Qwen4 K5/K9 flag or the independent
    /// strict K16 flag is enabled. A requested route fails closed when the
    /// layer geometry or exact kernels do not match.
    pub(in super::super) fn qwen4_k5_project_qkv_exact(
        &self,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, usize)>> {
        let legacy_k5_k9_enabled = std::env::var("ATLAS_QWEN4_K5_BATCH_ATTN_QKV")
            .ok()
            .as_deref()
            == Some("1");
        let route = if rows == 32 && crate::layers::qwen4_prefill_moe::attn16::core32_selected()? {
            Some(Qwen4ExactQkvRowsRoute::K32)
        } else if rows == 4 && legacy_k5_k9_enabled && crate::layers::qwen4_hybrid_rows(4) {
            // The K4 hybrid shares the K5/K9 exact-rows projection family.
            Some(Qwen4ExactQkvRowsRoute::LegacyK5K9)
        } else {
            qwen4_exact_qkv_rows_route(rows, legacy_k5_k9_enabled, qwen4_k16_exact_requested()?)
        };
        let Some(route) = route else {
            return Ok(None);
        };

        ensure!(self.gated, "Qwen4 exact QKV requires gated attention");
        ensure!(
            self.attn.q_norm_full.is_none()
                && self.attn.k_norm_full.is_none()
                && self.v_norm_weight.is_none(),
            "Qwen4 exact QKV requires per-head Q/K norms and no V norm"
        );
        let q_weight = self
            .q_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact Q requires ordinary NVFP4"))?;
        let k_weight = self
            .k_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact K requires ordinary NVFP4"))?;
        let v_weight = self
            .v_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact V requires ordinary NVFP4"))?;

        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        if matches!(
            route,
            Qwen4ExactQkvRowsRoute::K16 | Qwen4ExactQkvRowsRoute::K32
        ) {
            ensure!(
                qwen4_k16_target_attention_geometry(ctx.config.is_qwen4_exp(), h, nq, nkv, hd),
                "{QWEN4_K16_EXACT_ENV} requires exact Flash-Next attention geometry"
            );
        }
        let extents = qwen4_qkv_projection_extents(nq, nkv, hd, rows)
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact QKV extent overflow"))?;
        let qkv = ctx.buffers.qkv_output();
        ensure!(
            extents.total_bytes <= ctx.buffers.sizes().qkv_output,
            "Qwen4 exact QKV staging exceeds QKV arena"
        );
        let projection_rows = if route == Qwen4ExactQkvRowsRoute::K32 {
            16
        } else {
            rows
        };
        let qg_kernel = self.w4a16_exact_qkv_kernels.qg_for_rows(projection_rows);
        let dual_kv_kernel = self
            .w4a16_exact_qkv_kernels
            .dual_kv_for_rows(projection_rows);
        ensure!(
            qg_kernel.0 != 0 && dual_kv_kernel.0 != 0,
            "Qwen4 exact QKV kernels are unavailable"
        );

        for row_start in (0..rows).step_by(projection_rows) {
            let chunk_input = normed.offset(row_start * h as usize * 2);
            let chunk_output = qkv.offset(row_start * extents.row_bytes);
            ops::w4a16_gemv_qg_exact(
                ctx.gpu,
                qg_kernel,
                chunk_input,
                q_weight,
                chunk_output,
                projection_rows as u32,
                extents.q_proj_dim,
                h,
                nq,
                hd,
                extents.row_stride_bf16,
                stream,
            )?;
            let k_base = chunk_output.offset(extents.q_proj_bytes);
            let v_base = k_base.offset(extents.kv_bytes);
            ops::w4a16_gemv_dual_kv_exact(
                ctx.gpu,
                dual_kv_kernel,
                chunk_input,
                k_weight,
                k_base,
                v_weight,
                v_base,
                projection_rows as u32,
                extents.kv_dim,
                h,
                extents.row_stride_bf16,
                stream,
            )?;
        }

        let eps = ctx.config.rms_norm_eps as f32;
        for row in 0..rows {
            let q_out = qkv.offset(row * extents.row_bytes);
            let k_out = q_out.offset(extents.q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    q_out,
                    &self.attn.q_norm,
                    q_out,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    k_out,
                    &self.attn.k_norm,
                    k_out,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        if let Some(receipt) = qwen4_exact_qkv_receipt(route) {
            crate::model::k16_route_receipt::mark_qkv_layer()?;
            static K16_EXACT_ENGAGED: std::sync::Once = std::sync::Once::new();
            K16_EXACT_ENGAGED.call_once(|| tracing::warn!("{receipt}"));
        }
        Ok(Some((qkv, extents.row_bytes)))
    }
}

#[cfg(test)]
#[path = "qwen4_k16_projection_tests.rs"]
mod qwen4_k16_projection_tests;
