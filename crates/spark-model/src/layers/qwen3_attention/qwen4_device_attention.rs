// SPDX-License-Identifier: AGPL-3.0-only
//! Strict candidate-eager contract for graph-safe Qwen4 K5 attention.

use anyhow::{Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH";
pub(crate) const FULL_ATTENTION_LAYERS: [usize; 12] =
    [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Qwen4DeviceAttentionRoute {
    DensePaged { num_splits: u32 },
    SparseQsa,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4DeviceAttentionRow {
    pub qsa_pool_endpoint: bool,
    pub qsa_visible_groups: u32,
    pub qsa_score_grid_x: u32,
    pub route: Qwen4DeviceAttentionRoute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4K5DeviceAttentionPlan {
    rows: [Qwen4DeviceAttentionRow; 5],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Qwen4K5DeviceAttentionTopology {
    pool_endpoint_mask: u8,
    score_grids: [u32; 5],
    routes: [Qwen4DeviceAttentionRoute; 5],
}

fn parse_exact_bool(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be absent, 0, or 1; got {other:?}"),
    }
}

fn exact_env(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse_exact_bool(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_exact_bool(name, None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{name} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

fn exact_layer_map(config: &ModelConfig) -> bool {
    if config.num_hidden_layers != 48 {
        return false;
    }
    (0..48).all(|layer| {
        let expected = if FULL_ATTENTION_LAYERS.contains(&layer) {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        };
        config.layer_type(layer) == expected
    })
}

fn classify(position: usize) -> Result<Qwen4DeviceAttentionRow> {
    let sequence_length = position
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Qwen4 K5 device-attention position overflow"))?;
    let visible_groups = sequence_length / 4;
    let score_grid = if visible_groups > 512 {
        u32::try_from(visible_groups.div_ceil(8))?
    } else {
        0
    };
    Ok(Qwen4DeviceAttentionRow {
        qsa_pool_endpoint: sequence_length.is_multiple_of(4),
        qsa_visible_groups: u32::try_from(visible_groups)?,
        qsa_score_grid_x: score_grid,
        route: if position >= 2_048 {
            Qwen4DeviceAttentionRoute::SparseQsa
        } else {
            Qwen4DeviceAttentionRoute::DensePaged { num_splits: 2 }
        },
    })
}

impl Qwen4K5DeviceAttentionPlan {
    pub(crate) fn from_host(
        config: &ModelConfig,
        num_tokens: usize,
        seq_len: usize,
        metadata: AttnMetadataDev,
    ) -> Result<Option<Self>> {
        if !exact_env(SELECTOR)? {
            return Ok(None);
        }
        ensure!(num_tokens == 5, "{SELECTOR} requires exact physical K5");
        ensure!(
            exact_env("ATLAS_QWEN4_K5_HYBRID")? && exact_env("ATLAS_QWEN4_K5_BATCH_ATTN_QKV")?,
            "{SELECTOR} requires the qualified exact K5 attention route"
        );
        ensure!(
            !exact_env("ATLAS_QWEN4_K5_BATCH_HYPER")? && !exact_env("ATLAS_PAGED_DECODE_SPLITK")?,
            "{SELECTOR} requires row-exact hyper and legacy two-way paged splits"
        );
        ensure!(
            config.is_qwen4_exp()
                && config.hidden_size == 2_560
                && config.num_attention_heads == 24
                && config.num_key_value_heads == 2
                && config.head_dim == 256
                && exact_layer_map(config),
            "{SELECTOR} requires exact Qwen3.8-Flash-Next target geometry"
        );
        ensure!(
            metadata.qwen4_qsa_required
                && metadata.num_seqs == 5
                && metadata.max_blocks_per_seq > 0,
            "{SELECTOR} requires complete K5 QSA metadata"
        );
        let pointers = [
            metadata.positions.0,
            metadata.slot.0,
            metadata.seq_len.0,
            metadata.block_table.0,
        ];
        ensure!(
            !pointers.contains(&0)
                && pointers
                    .iter()
                    .enumerate()
                    .all(|(i, p)| !pointers[..i].contains(p)),
            "{SELECTOR} requires stable non-aliasing metadata pointers"
        );
        let mut rows = [classify(seq_len)?; 5];
        for (row, item) in rows.iter_mut().enumerate() {
            *item = classify(seq_len.checked_add(row).ok_or_else(|| {
                anyhow::anyhow!("Qwen4 K5 device-attention row position overflow")
            })?)?;
        }
        Ok(Some(Self { rows }))
    }

    pub(crate) fn row(self, row: usize) -> Result<Qwen4DeviceAttentionRow> {
        self.rows
            .get(row)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 K5 device-attention row out of range"))
    }

    fn max_visible_groups(self) -> u32 {
        self.rows
            .iter()
            .map(|row| row.qsa_visible_groups)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn topology(self) -> Qwen4K5DeviceAttentionTopology {
        let mut pool_endpoint_mask = 0;
        let mut score_grids = [0; 5];
        let mut routes = [Qwen4DeviceAttentionRoute::SparseQsa; 5];
        for (index, row) in self.rows.iter().enumerate() {
            pool_endpoint_mask |= u8::from(row.qsa_pool_endpoint) << index;
            score_grids[index] = row.qsa_score_grid_x;
            routes[index] = row.route;
        }
        Qwen4K5DeviceAttentionTopology {
            pool_endpoint_mask,
            score_grids,
            routes,
        }
    }
}

impl Qwen3AttentionLayer {
    /// Resolve every optional candidate dependency before hyperconnection or
    /// model-state kernels execute. A requested route may fail, but it must
    /// never fall through to the ordinary host-scalar attention path.
    pub(super) fn prepare_qwen4_k5_device_attention(
        &self,
        plan: Qwen4K5DeviceAttentionPlan,
        kv_cache: &PagedKvCache,
        metadata: AttnMetadataDev,
        ctx: &ForwardContext,
    ) -> Result<()> {
        ensure!(
            self.attn_layer_idx < FULL_ATTENTION_LAYERS.len(),
            "{SELECTOR} reached an invalid attention-layer ordinal"
        );
        ensure!(
            self.gated
                && self.mla.is_none()
                && self.num_q_heads_override.is_none()
                && self.num_kv_heads_override.is_none()
                && self.head_dim_override.is_none(),
            "{SELECTOR} requires ordinary gated Qwen4 attention geometry"
        );
        ensure!(
            self.attn.q_norm_full.is_none()
                && self.attn.k_norm_full.is_none()
                && self.v_norm_weight.is_none(),
            "{SELECTOR} requires per-head Q/K norms and no V norm"
        );
        ensure!(
            self.q_weight
                .as_ref()
                .and_then(|weight| weight.as_nvfp4())
                .is_some()
                && self
                    .k_weight
                    .as_ref()
                    .and_then(|weight| weight.as_nvfp4())
                    .is_some()
                && self
                    .v_weight
                    .as_ref()
                    .and_then(|weight| weight.as_nvfp4())
                    .is_some(),
            "{SELECTOR} requires ordinary NVFP4 Q/K/V weights"
        );
        ensure!(
            self.w4a16_exact_qkv_kernels.qg_for_rows(5).0 != 0
                && self.w4a16_exact_qkv_kernels.dual_kv_for_rows(5).0 != 0,
            "{SELECTOR} exact K5 QKV kernels are unavailable"
        );
        let qkv_bytes = 5usize
            .checked_mul((24 * 256 * 2 + 2 * 256 * 2) * 2)
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR} QKV extent overflow"))?;
        ensure!(
            qkv_bytes <= ctx.buffers.sizes().qkv_output,
            "{SELECTOR} exact K5 QKV staging exceeds the QKV arena"
        );
        ensure!(
            kv_cache.config().cache_blocks_per_seq.is_none(),
            "{SELECTOR} is incompatible with high-speed-swap cache windows"
        );
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR} requires QSA on every attention layer"))?;
        let _stable_qsa_pointers =
            qsa.prepare_device_attention(kv_cache, metadata, plan.max_visible_groups(), ctx.gpu)?;
        Ok(())
    }
}
