// SPDX-License-Identifier: AGPL-3.0-only

use super::support::*;
use serde_json::json;
use spark_model::layers::ops::flashinfer_sm121::FlashInferSm121;
use spark_model::weight_map::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
};

#[path = "buffers.rs"]
mod buffers;
#[path = "evidence.rs"]
mod evidence;
#[path = "producer.rs"]
mod producer;
#[path = "producer_io.rs"]
mod producer_io;
#[path = "runtime.rs"]
mod runtime;
use evidence::*;

fn common_receipt(
    checkpoint: &Checkpoint,
    data: &ProjectionData,
    layer: usize,
    sources: std::collections::BTreeMap<String, String>,
    executable: &serde_json::Value,
    bundle: serde_json::Value,
) -> Result<serde_json::Value> {
    let index_path = checkpoint.root.join("model.safetensors.index.json");
    let shards = checkpoint_shards(checkpoint, layer)?;
    ensure!(
        data.manifest.get("checkpoint_source_shards") == Some(&serde_json::to_value(&shards)?),
        "generated-weight manifest shard provenance mismatch"
    );
    Ok(json!({
        "schema":SCHEMA,"production_authorized":false,"independent_gpu_review_required":true,
        "source_sha256":sources,"executable":executable,"embedded_bundle":bundle,
        "checkpoint":{"root":checkpoint.root,"index_sha256":sha256_file(&index_path)?,
            "shards":shards,"layer":layer,
            "projection":"ssm-qkvz","prefix":data.prefix,"convention":format!("{:?}", data.convention),
            "packed_sha256":sha256_bytes(&data.packed.bytes)?,
            "logical_scale_sha256":sha256_bytes(&data.scales.bytes)?,
            "weight_scale2_bits":data.weight_scale2.to_bits(),"generated_weight_manifest":&data.manifest},
    }))
}

pub(crate) fn run() -> Result<()> {
    let produce = strict_switch("ATLAS_SSM_QKVZ_PRODUCE")?;
    let attest_only = strict_switch("ATLAS_SSM_QKVZ_ATTEST_ONLY")?;
    let timing = strict_switch("ATLAS_SSM_QKVZ_TIMING")?;
    ensure!(
        usize::from(produce) + usize::from(attest_only) + usize::from(timing) == 1,
        "select exactly one of produce, attest-only, or timing"
    );
    if produce {
        return producer::run();
    }
    let sources = attest_sources()?;
    let executable = executable_identity(timing)?;
    let checkpoint_dir = PathBuf::from(required_env("ATLAS_FI_CHECKPOINT_DIR")?);
    let layer = parse_usize("ATLAS_FI_LAYER")?;
    let checkpoint = Checkpoint::open(&checkpoint_dir)?;
    let (modules, bundle) = bundle_identity()?;
    let manifest =
        producer_io::verify_consumer_inputs(&checkpoint, layer, &executable, &sources, &bundle)?;
    let data = load_ssm_qkvz(layer, manifest)?;
    ensure!(
        data.packed.bytes.len() == N * K / 2,
        "QKVZ packed extent drift"
    );
    ensure!(
        data.scales.bytes.len() == N * K / 16,
        "QKVZ scale extent drift"
    );
    let physical_weight_scales = interleave_nvfp4_scales_128x4(
        &data.scales.bytes,
        &[N, K / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(
            &physical_weight_scales,
            &[N, K / NVFP4_GROUP_SIZE],
            NVFP4_GROUP_SIZE,
        )? == data.scales.bytes,
        "weight physical-layout roundtrip failed"
    );

    if attest_only {
        let common = common_receipt(&checkpoint, &data, layer, sources, &executable, bundle)?;
        recheck_executable(&executable)?;
        println!(
            "{}",
            serde_json::to_string(
                &json!({"attestation":common,"verdict":"ATTEST_PASS","gpu_initialized":false,"timing":false})
            )?
        );
        return Ok(());
    }

    let nonce = required_env("ATLAS_SSM_QKVZ_NONCE")?;
    validate_nonce(&nonce)?;
    let library_path = PathBuf::from(required_env("ATLAS_FI_CABI_LIB")?);
    ensure!(
        sha256_file(&library_path)? == CABI_SHA256,
        "C ABI source hash drift"
    );
    let library = FlashInferSm121::open_with_sha256(&library_path, CABI_SHA256_BYTES)?;
    ensure!(
        library.sha256_hex() == CABI_SHA256,
        "retained C ABI hash drift"
    );
    let common = common_receipt(&checkpoint, &data, layer, sources, &executable, bundle)?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let device = require_gb10()?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    ensure!(stream != 0, "runtime requires a nondefault stream");
    let mut cases = Vec::new();
    for (rows, reps) in SHAPES {
        cases.push(runtime::run_shape(
            gpu,
            &library,
            &data,
            &physical_weight_scales,
            rows,
            reps,
            stream,
        )?);
    }
    let receipt = json!({
        "common":common,"nonce":nonce,"device":device,"cabi":{"path":library.path(),"sha256":library.sha256_hex(),
            "device_inode":library.file_identity()},"cases":cases,"verdict":"PASS",
        "raw_gate_only":true,"production_authorized":false,
    });
    recheck_executable(&executable)?;
    write_receipt(&receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_model::layers::ops::nvfp4_dynamic_scale::{
        Nvfp4DynamicScaleBuffers, Nvfp4DynamicScaleKernels,
        validate_qwen38_ssm_projection_dynamic_scale,
    };
    use spark_runtime::gpu::{DevicePtr, KernelHandle};

    fn kernels() -> Nvfp4DynamicScaleKernels {
        Nvfp4DynamicScaleKernels {
            absmax: KernelHandle(1),
            quantize_from_absmax: KernelHandle(2),
            combined_alpha: KernelHandle(3),
        }
    }
    fn buffers() -> Nvfp4DynamicScaleBuffers {
        Nvfp4DynamicScaleBuffers {
            input_bf16: DevicePtr(0x10_0000),
            packed_e2m1: DevicePtr(0x3000_0000),
            scales_e4m3_128x4: DevicePtr(0x4000_0000),
            global_max_f32: DevicePtr(0x5000_0000),
            scale2_a_f32: DevicePtr(0x5000_0010),
            status_u32: DevicePtr(0x5000_0020),
            combined_alpha_f32: DevicePtr(0x5000_0030),
        }
    }
    #[test]
    fn hostile_preflight_rejects_before_runtime() {
        let good = buffers();
        assert!(
            validate_qwen38_ssm_projection_dynamic_scale(kernels(), good, 2_079, K as u32, 0.75)
                .is_ok()
        );
        let mut alias = good;
        alias.packed_e2m1 = alias.input_bf16;
        assert!(
            validate_qwen38_ssm_projection_dynamic_scale(kernels(), alias, 2_079, K as u32, 0.75)
                .is_err()
        );
        assert!(
            validate_qwen38_ssm_projection_dynamic_scale(
                kernels(),
                good,
                2_079,
                K as u32,
                f32::NAN
            )
            .is_err()
        );
        let mut absent = kernels();
        absent.combined_alpha = KernelHandle(0);
        assert!(
            validate_qwen38_ssm_projection_dynamic_scale(absent, good, 2_079, K as u32, 0.75)
                .is_err()
        );
    }
    #[test]
    fn source_contract_is_fail_closed() {
        let driver = include_str!("driver.rs");
        for marker in [
            "usize::from(produce) + usize::from(attest_only) + usize::from(timing) == 1",
            "producer_io::verify_consumer_inputs(",
            "validate_nonce(&nonce)",
            "open_with_sha256",
            "stream != 0",
            "recheck_executable(&executable)",
            "write_receipt(&receipt)",
        ] {
            assert!(driver.contains(marker));
        }
        let runtime = include_str!("runtime.rs");
        for marker in [
            "prepare_borrowed_zero_workspace",
            "for_flashinfer(stream + 1)",
            "physical scale tail is not zero",
            "cm < pm && cp < pp && dm > 0.0",
        ] {
            assert!(runtime.contains(marker));
        }
        let entry = include_str!("../qwen38_flashinfer_ssm_qkvz_dynamic_microgate.rs");
        for marker in [
            "qwen38-ssm-qkvz-generated-weight-v1",
            "checkpoint_source_shards",
            "ATLAS_SSM_QKVZ_WEIGHT_PACKED",
            "major == 12 && minor == 1 && sms == 48",
        ] {
            assert!(entry.contains(marker));
        }
        let handoff = include_str!("producer_io.rs");
        for marker in [
            "producer_executable_sha256",
            "producer receipt hash mismatch",
        ] {
            assert!(handoff.contains(marker));
        }
    }
}
