// SPDX-License-Identifier: AGPL-3.0-only

//! Raw GB10 qualification for the physical-layout fused SwiGLU quantizer.
//! Parity, guards, determinism, and a synthetic real-kernel down projection
//! precede balanced timing; a receipt is emitted only when every gate passes.

use anyhow::{Context, Result, ensure};
use serde_json::json;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "qwen38_flashinfer_fused_silu_physical_microgate/contract.rs"]
mod contract;
#[path = "qwen38_flashinfer_fused_silu_physical_microgate/guarded.rs"]
mod guarded;
#[path = "qwen38_flashinfer_fused_silu_physical_microgate/launch.rs"]
mod launch;
#[path = "qwen38_flashinfer_fused_silu_physical_microgate/provenance.rs"]
mod provenance;
#[path = "qwen38_flashinfer_fused_silu_physical_microgate/runtime.rs"]
mod runtime;
#[path = "qwen38_flashinfer_fused_silu_physical_microgate/timing.rs"]
mod timing;

use contract::*;
use guarded::Guarded;
use provenance::*;
use runtime::run;

fn main() -> Result<()> {
    let attest_only = strict_switch("ATLAS_FUSED_SILU_PHYSICAL_ATTEST_ONLY")?;
    let binary = running_binary()?;
    let sources = require_sources()?;
    let (target, modules) = exact_bundle()?;
    let bundle = bundle_identity(&target, &modules)?;
    if attest_only {
        let end = running_binary()?;
        stable_binary(&binary, &end, false)?;
        println!(
            "{}",
            serde_json::to_string(
                &json!({"schema":"qwen38-fused-silu-physical-build-attestation-v1","status":"non-authorizing","performance_claim":false,"running_executable":{"path":binary.path,"sha256":binary.sha256,"profile":binary.profile},"source_sha256":sources,"embedded_bundle":{"target":bundle.target,"module_count":bundle.module_count,"sha256":bundle.sha256,"direct_modules":bundle.direct_modules}})
            )?
        );
        return Ok(());
    }
    ensure!(
        binary.profile == "release",
        "qualification requires cargo run --release"
    );
    ensure!(
        std::env::var("ATLAS_FUSED_SILU_PHYSICAL_TIMING").as_deref() == Ok("1"),
        "requires ATLAS_FUSED_SILU_PHYSICAL_TIMING=1"
    );
    let nonce = std::env::var("ATLAS_FUSED_SILU_PHYSICAL_NONCE").context("nonce is required")?;
    ensure!(
        nonce.len() >= 32 && nonce.is_ascii() && !nonce.chars().any(char::is_whitespace),
        "nonce must be >=32 non-whitespace ASCII bytes"
    );
    let plans = [
        Plan::checked(2_079, K, 22, SCALE2)?,
        Plan::checked(8_192, K, 32, SCALE2)?,
    ];
    let backend = AtlasCudaBackend::new(0, &modules)?;
    require_gb10()?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    ensure!(stream != 0, "qualification requires a nondefault stream");
    let kernels = Kernels {
        silu: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
        parent: gpu.kernel(
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        )?,
        fused: gpu.kernel(
            "quantize_nvfp4",
            "quantize_silu_mul_bf16_to_nvfp4_atlas_128x4",
        )?,
        down: gpu.kernel("nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")?,
    };
    ensure!(
        [kernels.silu, kernels.parent, kernels.fused, kernels.down]
            .iter()
            .all(|handle| handle.0 != 0),
        "required production handle missing"
    );
    let weight = Guarded::input(gpu, vec![0x11; (K / 2 * DOWN_N) as usize], 12)?;
    let weight_sf = Guarded::input(gpu, vec![0x38; (K / 16 * DOWN_N) as usize], 13)?;
    let receipts = plans
        .into_iter()
        .map(|plan| {
            run(
                gpu, stream, kernels, plan, &weight, &weight_sf, &nonce, &binary, &bundle, &sources,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    weight.free(gpu)?;
    weight_sf.free(gpu)?;
    stable_binary(&binary, &running_binary()?, true)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"schema":"qwen38-fused-silu-physical-raw-bundle-v1","status":"qualified","running_executable":{"path":binary.path,"sha256":binary.sha256,"profile":binary.profile},"source_sha256":sources,"embedded_bundle":{"target":bundle.target,"module_count":bundle.module_count,"sha256":bundle.sha256,"direct_modules":bundle.direct_modules},"receipts":receipts})
        )?
    );
    Ok(())
}

#[cfg(test)]
#[path = "qwen38_flashinfer_fused_silu_physical_microgate/tests.rs"]
mod tests;
