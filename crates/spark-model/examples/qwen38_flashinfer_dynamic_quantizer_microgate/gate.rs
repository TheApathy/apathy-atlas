// SPDX-License-Identifier: AGPL-3.0-only

use std::process::Command;

use anyhow::{Context, Result, ensure};
use spark_model::layers::ops::nvfp4_dynamic_scale::Nvfp4DynamicScaleKernels;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

use super::contract::{COLS, Fixture, INVALID_CHILD_ENV, ROWS};
use super::invalid::invalid_status_case;
use super::provenance::{exact_bundle, function_resources, require_gb10};
use super::valid::valid_case;

pub(super) fn run_gate() -> Result<()> {
    let modules = exact_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    require_gb10()?;
    let stream = gpu.create_stream()?;
    let kernels = Nvfp4DynamicScaleKernels::load(gpu)?;
    let static_quantizer = gpu.kernel(
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4",
    )?;
    let quant_resources = function_resources(kernels.quantize_from_absmax)?;
    let alpha_resources = function_resources(kernels.combined_alpha)?;
    ensure!(
        quant_resources.max_threads >= 256
            && quant_resources.shared_bytes == 0
            && quant_resources.local_bytes == 0
            && quant_resources.registers <= 128
            && alpha_resources.max_threads >= 1
            && alpha_resources.shared_bytes == 0
            && alpha_resources.local_bytes == 0
            && alpha_resources.registers <= 32,
        "resource gate failed: quantizer={quant_resources:?} alpha={alpha_resources:?}"
    );
    println!("RESOURCES quantizer={quant_resources:?} alpha={alpha_resources:?} gate=PASS");

    for cols in COLS {
        for rows in ROWS {
            for fixture in Fixture::ALL {
                valid_case(gpu, kernels, static_quantizer, rows, cols, fixture, stream)?;
            }
        }
    }
    for cols in COLS {
        for kind in ["nan", "posinf", "neginf"] {
            invalid_status_case(gpu, kernels, kind, cols, stream)?;
        }
    }

    let executable = std::env::current_exe()?;
    for cols in COLS {
        for kind in ["nan", "posinf", "neginf"] {
            let child = format!("{kind}:{cols}");
            let status = Command::new(&executable)
                .env(INVALID_CHILD_ENV, &child)
                .status()
                .with_context(|| format!("spawn invalid trap child {child}"))?;
            ensure!(
                status.success(),
                "invalid trap child {child} failed: {status}"
            );
        }
    }
    println!(
        "FINAL verdict=PASS shapes=4 fixtures_per_shape=7 parity_arms=28 invalid=6 packed=EXACT scales_128x4=EXACT scale2=EXACT alpha=EXACT no_production_route=true"
    );
    Ok(())
}
