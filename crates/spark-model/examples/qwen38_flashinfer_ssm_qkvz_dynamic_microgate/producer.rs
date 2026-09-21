// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use producer_io::{
    OutputPaths, bind_tensor_shards, load_qkvz_bf16, recheck_tensor_sources,
    stable_checkpoint_index, write_immutable, write_json_immutable,
};
use serde_json::{Value, json};

const RECEIPT_SCHEMA: &str = "qwen38-ssm-qkvz-generated-weight-producer-receipt-v1";
const MANIFEST_SCHEMA: &str = "qwen38-ssm-qkvz-generated-weight-v1";

fn executable_field<'a>(identity: &'a Value, name: &str) -> Result<&'a str> {
    identity
        .get(name)
        .and_then(Value::as_str)
        .with_context(|| format!("executable lacks {name}"))
}

pub(super) fn run() -> Result<()> {
    // Everything through output-path admission is CPU-only and precedes CUDA initialization.
    let sources = attest_sources()?;
    let executable = executable_identity(true)?;
    let checkpoint_dir = PathBuf::from(required_env("ATLAS_FI_CHECKPOINT_DIR")?);
    let layer = parse_usize("ATLAS_FI_LAYER")?;
    let checkpoint = Checkpoint::open(&checkpoint_dir)?;
    let (input_bytes, tensor_sources) = load_qkvz_bf16(&checkpoint, layer)?;
    let outputs = OutputPaths::from_env()?;
    let (index_sha256, index_identity) = stable_checkpoint_index(&checkpoint)?;
    let shards = checkpoint_shards(&checkpoint, layer)?;
    bind_tensor_shards(&tensor_sources, &shards)?;

    let (modules, bundle) = bundle_identity()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let device = require_gb10()?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    ensure!(stream != 0, "producer requires a nondefault stream");
    let input = Guarded::input(gpu, stream, input_bytes, 0x71)?;
    let packed = Guarded::output(gpu, stream, N * K / 2, 0x72)?;
    let scales = Guarded::output(gpu, stream, N * K / 16, 0x73)?;
    let maximum = Guarded::output(gpu, stream, 4, 0x74)?;
    let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;

    // This is byte-for-byte the load-time quantize_to_nvfp4 sequence.
    gpu.memset(maximum.ptr(), 0, 4)?;
    ops::nvfp4_global_absmax(
        gpu,
        absmax,
        input.ptr(),
        maximum.ptr(),
        (N * K) as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let max_bytes = maximum.payload(gpu, "producer-maximum")?;
    let global_max = f32::from_le_bytes(max_bytes.try_into().unwrap());
    ensure!(
        global_max.is_finite() && global_max >= 0.0,
        "invalid global maximum"
    );
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };
    ops::quantize_bf16_to_nvfp4(
        gpu,
        quantize,
        input.ptr(),
        packed.ptr(),
        scales.ptr(),
        scale2,
        N as u32,
        K as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let packed_bytes = packed.payload(gpu, "producer-packed")?;
    let scale_bytes = scales.payload(gpu, "producer-scales")?;
    input.check_immutable(gpu, "producer-input")?;
    ensure!(
        packed_bytes.len() == N * K / 2 && scale_bytes.len() == N * K / 16,
        "generated weight extent mismatch"
    );
    for buffer in [&input, &packed, &scales, &maximum] {
        buffer.free(gpu)?;
    }

    // Freeze the same canonical executable identity again after every GPU effect.
    recheck_executable(&executable)?;
    recheck_tensor_sources(&checkpoint, layer, &tensor_sources)?;
    ensure!(
        checkpoint_shards(&checkpoint, layer)? == shards,
        "checkpoint shard provenance drift"
    );
    ensure!(
        stable_checkpoint_index(&checkpoint)? == (index_sha256.clone(), index_identity.clone()),
        "checkpoint index provenance drift"
    );
    let packed_sha256 = sha256_bytes(&packed_bytes)?;
    let scales_sha256 = sha256_bytes(&scale_bytes)?;
    let receipt = json!({
        "schema":RECEIPT_SCHEMA,"production_authorized":false,"independent_gpu_review_required":true,
        "source_sha256":sources,"executable":executable,"embedded_bundle":bundle,"device":device,
        "checkpoint_root":checkpoint.root,"checkpoint_index_sha256":index_sha256,"checkpoint_index_identity":index_identity,
        "checkpoint_source_shards":shards,"checkpoint_tensor_sources":tensor_sources,"layer":layer,"n":N,"k":K,
        "packed_path":outputs.packed,"packed_sha256":packed_sha256,
        "logical_scales_path":outputs.scales,"logical_scales_sha256":scales_sha256,
        "weight_scale2_bits":scale2.to_bits(),
        "production_contract":{"loader":"load_ssm_qwen35/dequant_nvfp4_to_bf16",
            "concat":"gpu_concat_rows(qkv,z)","requantizer":"quantize_to_nvfp4",
            "e2m1":"production LUT","e4m3":"production E4M3FN LUT","bf16":"truncate"}
    });
    write_immutable(&outputs.packed, &packed_bytes)?;
    write_immutable(&outputs.scales, &scale_bytes)?;
    let receipt_bytes = write_json_immutable(&outputs.receipt, &receipt)?;
    let receipt_sha256 = sha256_bytes(&receipt_bytes)?;
    recheck_executable(&receipt["executable"])?;
    let manifest = json!({
        "schema":MANIFEST_SCHEMA,"production_authorized":false,"independent_gpu_review_required":true,
        "checkpoint_root":receipt["checkpoint_root"],"checkpoint_index_sha256":receipt["checkpoint_index_sha256"],
        "checkpoint_index_identity":receipt["checkpoint_index_identity"],
        "checkpoint_source_shards":receipt["checkpoint_source_shards"],"checkpoint_tensor_sources":receipt["checkpoint_tensor_sources"],
        "layer":layer,"n":N,"k":K,
        "packed_path":receipt["packed_path"],"packed_sha256":receipt["packed_sha256"],
        "logical_scales_path":receipt["logical_scales_path"],"logical_scales_sha256":receipt["logical_scales_sha256"],
        "weight_scale2_bits":receipt["weight_scale2_bits"],"executable":receipt["executable"],
        "producer_executable_path":executable_field(&receipt["executable"], "path")?,
        "producer_executable_sha256":executable_field(&receipt["executable"], "sha256")?,
        "producer_receipt_path":outputs.receipt,"producer_receipt_sha256":receipt_sha256
    });
    write_json_immutable(&outputs.manifest, &manifest)?;
    println!(
        "{}",
        serde_json::to_string(&json!({"verdict":"PRODUCER_PASS",
        "production_authorized":false,"manifest":outputs.manifest,"manifest_sha256":sha256_file(&outputs.manifest)?}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_contract_is_fail_closed() {
        let source = include_str!("producer.rs");
        let production = source.split("#[cfg(test)]").next().unwrap();
        for marker in [
            "executable_identity(true)",
            "OutputPaths::from_env()",
            "gpu.memset(maximum.ptr(), 0, 4)",
            "input.check_immutable",
            "recheck_executable(&executable)",
            "recheck_tensor_sources(&checkpoint, layer, &tensor_sources)",
            "write_json_immutable(&outputs.manifest",
        ] {
            assert!(production.contains(marker));
        }
        assert!(
            production.find("OutputPaths::from_env()").unwrap()
                < production.find("AtlasCudaBackend::new").unwrap()
        );
        assert!(
            production.find("recheck_executable(&executable)?").unwrap()
                < production.find("write_immutable(&outputs.packed").unwrap()
        );
        assert!(
            production
                .find("recheck_executable(&receipt[\"executable\"])?")
                .unwrap()
                < production
                    .find("write_json_immutable(&outputs.manifest")
                    .unwrap()
        );
    }
}
