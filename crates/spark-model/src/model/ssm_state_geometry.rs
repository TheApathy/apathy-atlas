// SPDX-License-Identifier: AGPL-3.0-only

//! Checked projection from raw model dimensions to per-layer SSM state bytes.

use anyhow::Result;
use atlas_core::config::ModelConfig;

use super::{checked_add, checked_mul};

pub(super) fn checked_ssm_state_bytes(config: &ModelConfig) -> Result<(usize, usize)> {
    if config.mamba_num_heads > 0 && config.mamba_head_dim > 0 {
        let inner = checked_mul(
            config.mamba_num_heads,
            config.mamba_head_dim,
            "Mamba inner width",
        )?;
        let h_elements = checked_mul(inner, config.ssm_state_size, "Mamba h state elements")?;
        let h_bytes = checked_mul(h_elements, 4, "Mamba h FP32 bytes")?;
        let groups = checked_mul(
            config.n_groups,
            config.ssm_state_size,
            "Mamba grouped state width",
        )?;
        let groups = checked_mul(groups, 2, "Mamba doubled grouped state width")?;
        let conv_width = checked_add(inner, groups, "Mamba conv input width")?;
        let conv_elements = checked_mul(
            conv_width,
            config.linear_conv_kernel_dim,
            "Mamba conv kernel elements",
        )?;
        Ok((
            h_bytes,
            checked_mul(conv_elements, 4, "Mamba conv FP32 bytes")?,
        ))
    } else {
        let values = checked_mul(
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            "GDN value width",
        )?;
        let h_elements = checked_mul(values, config.linear_key_head_dim, "GDN h state elements")?;
        let h_bytes = checked_mul(h_elements, 4, "GDN h FP32 bytes")?;
        let keys = checked_mul(
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            "GDN key width",
        )?;
        let keys = checked_mul(keys, 2, "GDN doubled key width")?;
        let conv_width = checked_add(keys, values, "GDN conv width")?;
        let conv_elements = checked_mul(
            conv_width,
            config.linear_conv_kernel_dim,
            "GDN conv kernel elements",
        )?;
        Ok((
            h_bytes,
            checked_mul(conv_elements, 4, "GDN conv FP32 bytes")?,
        ))
    }
}
