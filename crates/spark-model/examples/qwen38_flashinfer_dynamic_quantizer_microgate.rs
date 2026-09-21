// SPDX-License-Identifier: AGPL-3.0-only

//! Raw gate for the default-unrouted Qwen3.8 SSM projection dynamic quantizer.
//! It compares the device-resident chain with the old synchronized host-scale
//! boundary and does not launch a GEMM or alter a production route.

use anyhow::{Context, Result, bail};

#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/contract.rs"]
mod contract;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/fixtures.rs"]
mod fixtures;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/gate.rs"]
mod gate;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/guarded.rs"]
mod guarded;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/invalid.rs"]
mod invalid;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/launch.rs"]
mod launch;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/provenance.rs"]
mod provenance;
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/valid.rs"]
mod valid;

use contract::INVALID_CHILD_ENV;
use gate::run_gate;
use invalid::invalid_trap_child;

fn main() -> Result<()> {
    match std::env::var(INVALID_CHILD_ENV) {
        Ok(child) => {
            let (kind, cols) = child
                .split_once(':')
                .context("invalid child must be KIND:K")?;
            invalid_trap_child(kind, cols.parse().context("invalid child K")?)
        }
        Err(std::env::VarError::NotPresent) => run_gate(),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{INVALID_CHILD_ENV} must be UTF-8"),
    }
}

#[cfg(test)]
#[path = "qwen38_flashinfer_dynamic_quantizer_microgate/tests.rs"]
mod tests;
