// SPDX-License-Identifier: AGPL-3.0-only

//! Raw GB10 parity and timing gate for direct merged-row SwiGLU quantization.
//! The incumbent split-plus-quantize chain and the candidate consume the same
//! immutable merged FlashInfer FFN projection on one nondefault stream.

use serde_json::json;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "qwen38_flashinfer_direct_merged_silu_microgate/authority.rs"]
mod authority;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/contract.rs"]
mod contract;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/guarded.rs"]
mod guarded;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/launch.rs"]
mod launch;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/owner.rs"]
mod owner;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/provenance.rs"]
mod provenance;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/runtime.rs"]
mod runtime;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/scheduler_authority.rs"]
mod scheduler_authority;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/scheduler_build.rs"]
mod scheduler_build;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/scheduler_trust.rs"]
mod scheduler_trust;
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/timing.rs"]
mod timing;

use authority::QualificationAuthority;
use contract::*;
use owner::RetainedGpuFailure;
use provenance::*;
use runtime::run;
use scheduler_authority::SchedulerSession;

enum GateFailure {
    Ordinary(anyhow::Error),
    Retained(RetainedGpuFailure),
}

impl std::fmt::Debug for GateFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(error) => formatter.debug_tuple("Ordinary").field(error).finish(),
            Self::Retained(error) => formatter.debug_tuple("Retained").field(error).finish(),
        }
    }
}

impl From<anyhow::Error> for GateFailure {
    fn from(error: anyhow::Error) -> Self {
        Self::Ordinary(error)
    }
}

impl From<RetainedGpuFailure> for GateFailure {
    fn from(error: RetainedGpuFailure) -> Self {
        Self::Retained(error)
    }
}

impl From<serde_json::Error> for GateFailure {
    fn from(error: serde_json::Error) -> Self {
        Self::Ordinary(error.into())
    }
}

impl From<std::str::Utf8Error> for GateFailure {
    fn from(error: std::str::Utf8Error) -> Self {
        Self::Ordinary(error.into())
    }
}

fn main() -> std::result::Result<(), GateFailure> {
    let attest_only = strict_switch("ATLAS_DIRECT_MERGED_SILU_ATTEST_ONLY")?;
    let (target, modules) = exact_bundle()?;
    let bundle = bundle_identity(&target, &modules)?;
    let authority = QualificationAuthority::seal()?;
    let binary = authority.binary();
    let sources = authority.sources();
    if attest_only {
        authority.verify_unchanged()?;
        println!(
            "{}",
            serde_json::to_string(
                &json!({"schema":"qwen38-direct-merged-silu-build-attestation-v1","status":"non-authorizing","performance_claim":false,"running_executable":{"path":binary.path,"sha256":binary.sha256,"profile":binary.profile},"source_sha256":sources,"embedded_bundle":{"target":bundle.target,"module_count":bundle.module_count,"sha256":bundle.sha256,"direct_modules":bundle.direct_modules}})
            )?
        );
        return Ok(());
    }
    require(
        binary.profile == "release",
        "qualification requires cargo run --release",
    )?;
    require(
        std::env::var("ATLAS_DIRECT_MERGED_SILU_TIMING").as_deref() == Ok("1"),
        "requires ATLAS_DIRECT_MERGED_SILU_TIMING=1",
    )?;
    let scheduler = SchedulerSession::claim(&binary, &bundle, &sources)?;
    let session_id = scheduler.session_id().to_owned();
    let scheduler_evidence = scheduler.evidence().clone();
    let plans = [
        Plan::checked(2_079, K, 22, SCALE2)?,
        Plan::checked(8_192, K, 32, SCALE2)?,
    ];
    let backend = AtlasCudaBackend::new(0, &modules)?;
    require_gb10()?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    require(stream != 0, "qualification requires a nondefault stream")?;
    let kernels = Kernels {
        split: gpu.kernel(
            "flashinfer_projection_split",
            "flashinfer_projection_split_ffn_gate_up",
        )?,
        parent: gpu.kernel(
            "quantize_nvfp4",
            "quantize_silu_mul_bf16_to_nvfp4_atlas_128x4",
        )?,
        candidate: gpu.kernel(
            "quantize_nvfp4",
            "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
        )?,
        down: gpu.kernel("nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")?,
    };
    require(
        [
            kernels.split,
            kernels.parent,
            kernels.candidate,
            kernels.down,
        ]
        .iter()
        .all(|handle| handle.0 != 0),
        "required production handle missing",
    )?;
    let mut receipts = Vec::new();
    for plan in plans {
        match run(
            gpu,
            stream,
            kernels,
            plan,
            &session_id,
            &binary,
            &bundle,
            &sources,
        ) {
            Ok(receipt) => receipts.push(receipt),
            Err(mut failure) => {
                for _ in 0..3 {
                    if !failure.retains_cleanup_authority() || failure.retry_cleanup(gpu).is_ok() {
                        break;
                    }
                }
                return Err(failure.into());
            }
        }
    }
    authority.verify_unchanged()?;
    let final_receipt = serde_json::to_vec(
        &json!({"schema":"qwen38-direct-merged-silu-raw-bundle-v4","status":"qualified","performance_claim":true,"one_shot":true,"scheduler_session_id":session_id,"scheduler_authority":scheduler_evidence,"running_executable":{"path":binary.path,"sha256":binary.sha256,"profile":binary.profile},"source_sha256":sources,"embedded_bundle":{"target":bundle.target,"module_count":bundle.module_count,"sha256":bundle.sha256,"direct_modules":bundle.direct_modules},"receipts":receipts}),
    )?;
    let _published_path = scheduler.publish(&final_receipt)?;
    println!("{}", std::str::from_utf8(&final_receipt)?);
    Ok(())
}

fn require(condition: bool, message: &str) -> anyhow::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(anyhow::anyhow!(message.to_owned()))
    }
}

#[cfg(test)]
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/hostile_tests.rs"]
mod hostile_tests;
#[cfg(test)]
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/owner_tests.rs"]
mod owner_tests;
#[cfg(test)]
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/scheduler_authority_tests.rs"]
mod scheduler_authority_tests;
#[cfg(test)]
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/scheduler_build_tests.rs"]
mod scheduler_build_tests;
#[cfg(test)]
#[path = "qwen38_flashinfer_direct_merged_silu_microgate/tests.rs"]
mod tests;
