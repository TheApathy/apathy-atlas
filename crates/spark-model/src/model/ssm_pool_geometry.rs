// SPDX-License-Identifier: AGPL-3.0-only

//! Checked single source of truth for persistent SSM active-pool geometry.

use anyhow::{Result, anyhow, bail};
use atlas_core::config::ModelConfig;

#[path = "ssm_state_geometry.rs"]
mod state_geometry;
use state_geometry::checked_ssm_state_bytes;
pub const DDTREE_KERNEL_KMAX: usize = 32;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsmSpeculativeGeometry {
    pub dflash_verify_width: usize,
    pub ddtree_capacity: usize,
    pub num_intermediates: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct SsmPoolGeometryInput {
    pub max_slots: usize,
    pub num_ssm_layers: usize,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub has_mtp: bool,
    pub num_intermediates: usize,
    pub lazy_commit: bool,
    pub num_key_heads: usize,
    pub key_head_dim: usize,
    pub num_value_heads: usize,
    pub value_head_dim: usize,
}
impl SsmPoolGeometryInput {
    pub fn from_config(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        lazy_commit: bool,
    ) -> Result<Self> {
        let num_ssm_layers = config.num_ssm_layers();
        let (h_bytes, conv_bytes) = if num_ssm_layers == 0 {
            (0, 0)
        } else {
            checked_ssm_state_bytes(config)?
        };
        Ok(Self {
            max_slots,
            num_ssm_layers,
            h_bytes,
            conv_bytes,
            has_mtp,
            num_intermediates,
            lazy_commit,
            num_key_heads: config.linear_num_key_heads,
            key_head_dim: config.linear_key_head_dim,
            num_value_heads: config.linear_num_value_heads,
            value_head_dim: config.linear_value_head_dim,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsmPoolGeometry {
    pub total_slots: usize,
    pub state_copies: usize,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub h_state_allocation_bytes: usize,
    pub conv_state_allocation_bytes: usize,
    pub h_intermediate_allocation_bytes: usize,
    pub conv_intermediate_allocation_bytes: usize,
    pub kv_retain_bytes: usize,
    pub gate_retain_bytes: usize,
    pub kv_retain_allocation_bytes: usize,
    pub gate_retain_allocation_bytes: usize,
    pub bytes_per_layer_slot: usize,
    pub total_bytes: usize,
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("SSM pool geometry overflow: {label} ({left} + {right})"))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("SSM pool geometry overflow: {label} ({left} * {right})"))
}

pub fn checked_ssm_speculative_geometry(
    has_mtp: bool,
    dflash_enabled: bool,
    num_drafts: usize,
    requested_ddtree_capacity: Option<usize>,
) -> Result<SsmSpeculativeGeometry> {
    let dflash_verify_width = if dflash_enabled {
        let width = checked_add(num_drafts, 1, "DFlash verify width")?;
        if width > DDTREE_KERNEL_KMAX {
            bail!(
                "SSM pool geometry: DFlash verify width {width} exceeds tree-WY kernel maximum {DDTREE_KERNEL_KMAX}"
            );
        }
        width
    } else {
        0
    };
    let ddtree_capacity = if dflash_enabled {
        requested_ddtree_capacity
            .unwrap_or(dflash_verify_width)
            .clamp(dflash_verify_width, DDTREE_KERNEL_KMAX)
    } else {
        0
    };
    let num_intermediates = if has_mtp {
        let flat = checked_add(num_drafts, 2, "drafts + 2 intermediates")?;
        let tree = checked_add(ddtree_capacity, 1, "DDTree capacity + 1 intermediates")?;
        flat.max(tree)
    } else {
        0
    };
    Ok(SsmSpeculativeGeometry {
        dflash_verify_width,
        ddtree_capacity,
        num_intermediates,
    })
}

pub fn checked_ssm_pool_geometry(input: SsmPoolGeometryInput) -> Result<SsmPoolGeometry> {
    if input.num_ssm_layers == 0 {
        return Ok(SsmPoolGeometry::default());
    }
    if input.h_bytes == 0 {
        bail!("SSM pool geometry: enabled h state bytes must be positive");
    }
    if input.conv_bytes == 0 {
        bail!("SSM pool geometry: enabled conv state bytes must be positive");
    }
    if input.has_mtp && input.num_intermediates == 0 {
        bail!("SSM pool geometry: enabled intermediate count must be positive");
    }
    if !input.has_mtp && input.num_intermediates != 0 {
        bail!("SSM pool geometry: intermediates require speculative state");
    }

    let total_slots = checked_add(input.max_slots, 1, "dummy-inclusive slots")?;
    let state_bytes = checked_add(input.h_bytes, input.conv_bytes, "state bytes per copy")?;
    let state_copies = if input.has_mtp {
        checked_add(
            input.num_intermediates,
            2,
            "base + intermediates + checkpoint state copies",
        )?
    } else {
        1
    };
    let state_bytes_per_slot = checked_mul(state_bytes, state_copies, "state bytes per slot")?;

    let (kv_retain_bytes, gate_retain_bytes) = if input.has_mtp && input.lazy_commit {
        let key_width = checked_mul(
            input.num_key_heads,
            input.key_head_dim,
            "retention key width",
        )?;
        let doubled_keys = checked_mul(key_width, 2, "retention doubled key width")?;
        let value_width = checked_mul(
            input.num_value_heads,
            input.value_head_dim,
            "retention value width",
        )?;
        let conv_width = checked_add(doubled_keys, value_width, "retention conv width")?;
        if conv_width == 0 || input.num_value_heads == 0 {
            bail!("SSM pool geometry: enabled lazy retention dimensions must be positive");
        }
        let kv_elements =
            checked_mul(input.num_intermediates, conv_width, "KV retention elements")?;
        let kv_bytes = checked_mul(kv_elements, 2, "KV retention bytes")?;
        let gate_width = checked_mul(input.num_value_heads, 2, "gate width")?;
        let gate_elements = checked_mul(
            input.num_intermediates,
            gate_width,
            "gate retention elements",
        )?;
        let gate_bytes = checked_mul(gate_elements, 4, "gate retention bytes")?;
        (kv_bytes, gate_bytes)
    } else {
        (0, 0)
    };
    let retention_bytes_per_slot = checked_add(
        kv_retain_bytes,
        gate_retain_bytes,
        "retention bytes per slot",
    )?;
    let bytes_per_layer_slot = checked_add(
        state_bytes_per_slot,
        retention_bytes_per_slot,
        "state + retention bytes per layer slot",
    )?;
    let bytes_per_layer = checked_mul(
        total_slots,
        bytes_per_layer_slot,
        "dummy-inclusive bytes per layer",
    )?;
    let total_bytes = checked_mul(input.num_ssm_layers, bytes_per_layer, "all SSM layers")?;

    let h_state_allocation_bytes = checked_mul(total_slots, input.h_bytes, "h state allocation")?;
    let conv_state_allocation_bytes =
        checked_mul(total_slots, input.conv_bytes, "conv state allocation")?;
    let (h_intermediate_allocation_bytes, conv_intermediate_allocation_bytes) = if input.has_mtp {
        let intermediate_slots = checked_mul(
            total_slots,
            input.num_intermediates,
            "intermediate allocation slots",
        )?;
        (
            checked_mul(
                intermediate_slots,
                input.h_bytes,
                "h intermediate allocation",
            )?,
            checked_mul(
                intermediate_slots,
                input.conv_bytes,
                "conv intermediate allocation",
            )?,
        )
    } else {
        (0, 0)
    };
    let kv_retain_allocation_bytes =
        checked_mul(total_slots, kv_retain_bytes, "KV retention allocation")?;
    let gate_retain_allocation_bytes =
        checked_mul(total_slots, gate_retain_bytes, "gate retention allocation")?;

    Ok(SsmPoolGeometry {
        total_slots,
        state_copies,
        h_bytes: input.h_bytes,
        conv_bytes: input.conv_bytes,
        h_state_allocation_bytes,
        conv_state_allocation_bytes,
        h_intermediate_allocation_bytes,
        conv_intermediate_allocation_bytes,
        kv_retain_bytes,
        gate_retain_bytes,
        kv_retain_allocation_bytes,
        gate_retain_allocation_bytes,
        bytes_per_layer_slot,
        total_bytes,
    })
}
