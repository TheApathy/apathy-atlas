// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed raw gate for the NVFP4 paged-attention BR128 shadow.
//!
//! The default invocation is a non-qualifying smoke. Full continuation parity
//! and timing require explicit strict switches and a separately reserved GB10.

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "nvfp4_paged_attn_br128_microgate/contract.rs"]
mod contract;
#[path = "nvfp4_paged_attn_br128_microgate/fixtures.rs"]
mod fixtures;
#[path = "nvfp4_paged_attn_br128_microgate/guarded.rs"]
mod guarded;
#[path = "nvfp4_paged_attn_br128_microgate/runtime.rs"]
mod runtime;
#[path = "nvfp4_paged_attn_br128_microgate/timing.rs"]
mod timing;

use contract::{FULL_CASES, exact_modules, function_resources, require_gb10, require_target};

fn main() -> Result<()> {
    require_target()?;
    let full = contract::strict_switch("ATLAS_BR128_MICROGATE_FULL")?;
    let timing = contract::strict_switch("ATLAS_BR128_MICROGATE_TIMING")?;
    ensure!(
        !timing || full,
        "timing requires ATLAS_BR128_MICROGATE_FULL=1"
    );

    let modules = exact_modules()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    require_gb10()?;
    let stream = gpu.create_stream()?;
    ensure!(stream != 0, "qualification requires a nondefault stream");
    let parent = gpu.kernel("prefill_paged_nvfp4", "inferspark_prefill_paged_nvfp4_64")?;
    let candidate = gpu.kernel("prefill_paged_nvfp4", "inferspark_prefill_paged_nvfp4_128")?;
    // The Driver classifies the PTX's compiler-reserved 1 KiB as neither
    // user static shared nor local storage. Its LOCAL attribute includes the
    // separately attested 64-byte stack frame. Bind those runtime semantics
    // and explicitly prove the >48 KiB dynamic-shared opt-in before effects.
    let resources = function_resources(candidate)?;
    ensure!(
        resources.max_threads >= 512
            && resources.registers <= 128
            && resources.shared_bytes == 0
            && resources.local_bytes == 64
            && resources.max_dynamic_shared_bytes == 95_808,
        "SM121 resource gate failed: {resources:?}"
    );
    println!("BR128_RESOURCES {resources:?} dynamic_shared=95808 gate=PASS");

    let cases = if full {
        FULL_CASES.as_slice()
    } else {
        contract::SMOKE_CASES.as_slice()
    };
    for case in cases.iter().copied() {
        runtime::run_case(gpu, stream, parent, candidate, case)?;
    }

    if timing {
        for case in contract::CONTINUATION_CASES {
            timing::time_case(gpu, stream, parent, candidate, case)?;
        }
        println!(
            "BR128_QUALIFIED parity_cases={} timing_cases=3 pairs_per_case={} continuation_only=true exact=true",
            FULL_CASES.len(),
            contract::TIMING_PAIRS
        );
    } else if full {
        println!(
            "BR128_FULL_PARITY_PASS cases={} performance_qualified=false timing_required=true",
            FULL_CASES.len()
        );
    } else {
        println!("BR128_SMOKE_PASS cases=2 qualified=false performance_claim=false");
    }
    Ok(())
}
