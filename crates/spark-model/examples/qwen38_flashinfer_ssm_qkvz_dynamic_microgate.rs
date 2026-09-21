// SPDX-License-Identifier: AGPL-3.0-only

//! Provenance-closed raw gate for the complete Qwen3.8 SSM QKVZ projection.
//!
//! CPU-only attestation does not initialize CUDA. Runtime is impossible unless
//! both the exact timing switch and a fresh nonce are supplied, and still
//! requires an independently reviewed release binary and reserved GB10.

#[rustfmt::skip]
mod support {
    pub(crate) use anyhow::{Context, Result, bail, ensure};
    pub(crate) use half::bf16;
    pub(crate) use spark_model::layers::ops;
    pub(crate) use spark_runtime::cuda_backend::AtlasCudaBackend;
    pub(crate) use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
    pub(crate) use std::path::{Path, PathBuf};
    pub(crate) use std::time::Instant;
    use serde_json::Value;
    use std::ffi::{CStr, c_char};
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::unix::fs::MetadataExt;

    const REDZONE: usize = 4 * 1_024;
    const ATTR_SMS: u32 = 16;
    const ATTR_MAJOR: u32 = 75;
    const ATTR_MINOR: u32 = 76;
    unsafe extern "C" {
        fn cuCtxGetDevice(device: *mut i32) -> i32;
        fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
        fn cuDeviceGetName(name: *mut c_char, length: i32, device: i32) -> i32;
    }

    pub(crate) struct TensorBytes { pub bytes: Vec<u8> }
    pub(crate) struct ProjectionData {
        pub prefix: String, pub packed: TensorBytes, pub scales: TensorBytes,
        pub weight_scale2: f32, pub convention: &'static str, pub manifest: Value,
    }
    pub(crate) struct Checkpoint {
        pub root: PathBuf, pub index: serde_json::Map<String, Value>,
    }
    impl Checkpoint {
        pub fn open(root: &Path) -> Result<Self> {
            ensure!(root.is_absolute(), "checkpoint path must be absolute");
            let root = root.canonicalize()?;
            let value: Value = serde_json::from_reader(File::open(root.join("model.safetensors.index.json"))?)?;
            let index = value.get("weight_map").and_then(Value::as_object).context("missing weight_map")?.clone();
            Ok(Self { root, index })
        }
    }
    fn digest_bytes(bytes: &[u8]) -> Result<String> {
        let mut child = std::process::Command::new("/usr/bin/sha256sum").stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn()?;
        child.stdin.as_mut().context("sha256sum stdin")?.write_all(bytes)?;
        let output = child.wait_with_output()?; ensure!(output.status.success(), "sha256sum bytes failed");
        Ok(String::from_utf8(output.stdout)?.split_whitespace().next().context("missing digest")?.to_owned())
    }
    fn artifact(name: &str, expected_len: usize, expected_hash: &str) -> Result<Vec<u8>> {
        let path = PathBuf::from(required_env(name)?);
        ensure!(path.is_absolute(), "{name} must be absolute"); let path = path.canonicalize()?;
        let file = File::open(path)?; let metadata = file.metadata()?;
        ensure!(metadata.is_file() && metadata.len() == expected_len as u64 && metadata.mode() & 0o777 == 0o444, "{name} extent/mode mismatch");
        let mut bytes = Vec::with_capacity(expected_len); file.take(expected_len as u64 + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() == expected_len && digest_bytes(&bytes)? == expected_hash, "{name} content mismatch"); Ok(bytes)
    }
    pub(crate) fn load_ssm_qkvz(layer: usize, manifest: Value) -> Result<ProjectionData> {
        ensure!(layer < 64, "layer must be in 0..64");
        ensure!(manifest.get("schema").and_then(Value::as_str) == Some("qwen38-ssm-qkvz-generated-weight-v1"), "weight manifest schema mismatch");
        ensure!(manifest.get("layer").and_then(Value::as_u64) == Some(layer as u64)
            && manifest.get("n").and_then(Value::as_u64) == Some(16_384)
            && manifest.get("k").and_then(Value::as_u64) == Some(5_120), "weight manifest geometry mismatch");
        let field = |name| manifest.get(name).and_then(Value::as_str).with_context(|| format!("manifest missing {name}"));
        let packed = artifact("ATLAS_SSM_QKVZ_WEIGHT_PACKED", 16_384 * 2_560, field("packed_sha256")?)?;
        let scales = artifact("ATLAS_SSM_QKVZ_WEIGHT_SCALES", 16_384 * 320, field("logical_scales_sha256")?)?;
        let bits = u32::try_from(manifest.get("weight_scale2_bits").and_then(Value::as_u64).context("missing scale2 bits")?)?;
        let weight_scale2 = f32::from_bits(bits);
        ensure!(weight_scale2.is_finite() && weight_scale2 > 0.0, "invalid generated weight scale2");
        ensure!(manifest.get("checkpoint_source_shards").and_then(Value::as_object).is_some_and(|v| !v.is_empty()), "missing checkpoint shard provenance");
        Ok(ProjectionData { prefix: format!("generated-runtime-qkvz-layer-{layer}"),
            packed: TensorBytes { bytes: packed }, scales: TensorBytes { bytes: scales },
            weight_scale2, convention: "production-generated-requantized", manifest })
    }

    pub(crate) fn required_env(name: &str) -> Result<String> {
        let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
        ensure!(!value.is_empty(), "{name} must not be empty"); Ok(value)
    }
    pub(crate) fn strict_switch(name: &str) -> Result<bool> {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => Ok(false), Ok(v) if v == "0" => Ok(false),
            Ok(v) if v == "1" => Ok(true), Ok(_) => bail!("{name} must be exactly 0 or 1"),
            Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be UTF-8"),
        }
    }
    pub(crate) fn parse_usize(name: &str) -> Result<usize> { required_env(name)?.parse().with_context(|| format!("{name} must be unsigned")) }
    pub(crate) fn transpose(input: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut output = vec![0; input.len()];
        for row in 0..rows { for col in 0..cols { output[col * rows + row] = input[row * cols + col]; } } output
    }
    pub(crate) fn median(values: &[f64]) -> f64 {
        let mut values = values.to_vec(); values.sort_by(f64::total_cmp); values[values.len() / 2]
    }
    pub(crate) fn require_gb10() -> Result<Value> {
        let mut device = -1; ensure!(unsafe { cuCtxGetDevice(&mut device) } == 0, "cuCtxGetDevice failed");
        let attribute = |kind| -> Result<i32> { let mut value = -1;
            ensure!(unsafe { cuDeviceGetAttribute(&mut value, kind, device) } == 0, "device attribute failed"); Ok(value) };
        let (major, minor, sms) = (attribute(ATTR_MAJOR)?, attribute(ATTR_MINOR)?, attribute(ATTR_SMS)?);
        let mut raw_name = [0 as c_char; 256];
        ensure!(unsafe { cuDeviceGetName(raw_name.as_mut_ptr(), raw_name.len() as i32, device) } == 0, "device name failed");
        let name = unsafe { CStr::from_ptr(raw_name.as_ptr()) }.to_string_lossy().into_owned();
        ensure!(major == 12 && minor == 1 && sms == 48, "requires GB10 SM12.1/48SM; got {name} SM{major}.{minor}/{sms}SM");
        Ok(serde_json::json!({"ordinal":device,"name":name,"compute_major":major,"compute_minor":minor,"sm_count":sms}))
    }

    pub(crate) struct Guarded {
        allocation: DevicePtr, payload_len: usize, prefix: Vec<u8>, suffix: Vec<u8>, immutable: Option<Vec<u8>>,
    }
    impl Guarded {
        fn create(gpu: &dyn GpuBackend, stream: u64, payload: Vec<u8>, salt: u8, immutable: bool) -> Result<Self> {
            let prefix = vec![0xa5 ^ salt; REDZONE]; let suffix = vec![0x5a ^ salt; REDZONE];
            let mut image = Vec::with_capacity(REDZONE + payload.len() + REDZONE);
            image.extend_from_slice(&prefix); image.extend_from_slice(&payload); image.extend_from_slice(&suffix);
            let allocation = gpu.alloc(image.len())?; gpu.synchronize(stream)?; gpu.copy_h2d(&image, allocation)?;
            Ok(Self { allocation, payload_len: payload.len(), prefix, suffix, immutable: immutable.then_some(payload) })
        }
        pub fn input(gpu: &dyn GpuBackend, stream: u64, payload: Vec<u8>, salt: u8) -> Result<Self> { Self::create(gpu, stream, payload, salt, true) }
        pub fn output(gpu: &dyn GpuBackend, stream: u64, bytes: usize, salt: u8) -> Result<Self> {
            let mut payload = vec![0; bytes]; for pair in payload.chunks_exact_mut(2) { pair.copy_from_slice(&0x7f81u16.to_le_bytes()); }
            Self::create(gpu, stream, payload, salt, false)
        }
        pub fn ptr(&self) -> DevicePtr { self.allocation.offset(REDZONE) }
        pub fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
            let mut prefix = vec![0; REDZONE]; let mut suffix = vec![0; REDZONE];
            gpu.copy_d2h(self.allocation, &mut prefix)?; gpu.copy_d2h(self.ptr().offset(self.payload_len), &mut suffix)?;
            ensure!(prefix == self.prefix && suffix == self.suffix, "{label}: redzone changed");
            let mut payload = vec![0; self.payload_len]; gpu.copy_d2h(self.ptr(), &mut payload)?; Ok(payload)
        }
        pub fn check_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
            ensure!(self.payload(gpu, label)? == *self.immutable.as_ref().context("not immutable")?, "{label}: input changed"); Ok(())
        }
        pub fn free(&self, gpu: &dyn GpuBackend) -> Result<()> { gpu.free(self.allocation) }
    }
    pub(crate) fn exact_bundle() -> Result<Vec<(&'static str, &'static str)>> {
        ensure!(std::env::var("ATLAS_TARGET_MODEL").as_deref() == Ok("qwen3.8-27b"), "wrong target model");
        ensure!(std::env::var("ATLAS_TARGET_QUANT").as_deref() == Ok("nvfp4"), "wrong target quant");
        let mut found: Vec<_> = atlas_kernels::available_targets().into_iter().filter(|set|
            set.target.arch == "sm_121" && set.target.model == "qwen3.8-27b" && set.target.quant == "nvfp4").collect();
        ensure!(found.len() == 1, "expected one exact embedded target"); Ok(found.pop().unwrap().modules)
    }
}

#[allow(dead_code)]
#[path = "qwen38_flashinfer_ssm_qkvz_dynamic_microgate/driver.rs"]
mod gate;

fn main() -> anyhow::Result<()> {
    gate::run()
}
