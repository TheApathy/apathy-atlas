// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactFlatSsmRoute {
    Existing,
    ExactSequence,
    SerialK1,
}

pub(super) const fn exact_flat_ssm_route(
    rows: usize,
    tree_active: bool,
    ba_exact: bool,
    conv_exact: bool,
    gdn_exact: bool,
    norm_exact: bool,
) -> ExactFlatSsmRoute {
    if tree_active || rows < 4 || rows > 32 {
        return ExactFlatSsmRoute::Existing;
    }
    if ba_exact && conv_exact && gdn_exact && norm_exact {
        ExactFlatSsmRoute::ExactSequence
    } else {
        ExactFlatSsmRoute::SerialK1
    }
}

pub(super) fn contiguous_intermediate_base(
    ptrs: &[DevicePtr],
    rows: usize,
    stride_bytes: usize,
    label: &str,
) -> Result<DevicePtr> {
    let Some(base) = ptrs.first().copied() else {
        bail!("exact flat SSM {label} intermediates are empty");
    };
    if ptrs.len() < rows {
        bail!(
            "exact flat SSM {label} intermediates too short: have {}, need {rows}",
            ptrs.len()
        );
    }
    for (row, ptr) in ptrs.iter().copied().take(rows).enumerate() {
        let expected = base
            .0
            .checked_add((row * stride_bytes) as u64)
            .ok_or_else(|| {
                anyhow::anyhow!("exact flat SSM {label} intermediate address overflow")
            })?;
        if ptr.0 != expected {
            bail!("exact flat SSM {label} intermediates are not contiguous at row {row}");
        }
    }
    Ok(base)
}

#[cfg(test)]
#[path = "exact_flat_route_tests.rs"]
mod tests;
