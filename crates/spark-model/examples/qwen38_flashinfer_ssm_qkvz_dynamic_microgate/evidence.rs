// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
pub(super) const N: usize = 16_384;
pub(super) const K: usize = 5_120;
pub(super) const SHAPES: [(usize, usize); 2] = [(2_079, 22), (8_192, 32)];
pub(super) const SCHEMA: &str = "qwen38-flashinfer-ssm-qkvz-dynamic-raw-v1";
pub(super) const MAX_RELATIVE_RMS: f64 = 0.025;
pub(super) const MIN_COSINE: f64 = 0.999;
pub(super) const CABI_SHA256: &str =
    "a007a82566ca3d3115c8cc0e73e2bbfc0bd1c76b6342313fba7ee5483e49c020";
pub(super) const CABI_SHA256_BYTES: [u8; 32] = [
    0xa0, 0x07, 0xa8, 0x25, 0x66, 0xca, 0x3d, 0x31, 0x15, 0xc8, 0xcc, 0x0e, 0x73, 0xe2, 0xbb, 0xfc,
    0x0b, 0xd1, 0xc7, 0x6b, 0x63, 0x42, 0x31, 0x3f, 0xba, 0x7e, 0xe5, 0x48, 0x3e, 0x49, 0xc0, 0x20,
];
#[rustfmt::skip]
const SOURCES: [(&str, &str); 13] = [
    ("crates/spark-model/src/layers/ops/nvfp4_dynamic_scale.rs", "4c4ff85a92a48b68ead2759f6f113c81559f3bd87b43420e450799098f432311"),
    ("crates/spark-model/src/layers/ops/flashinfer_sm121.rs", "eabf7498d1928bf1699e2a9eca5ffa639dcb51b57283a6e8e9928b0ce7061cb2"),
    ("crates/spark-model/src/layers/ops/gemm_dense.rs", "9dfb6b64da3dd21cd83fe03d8308c3df486b50f3f0d8aa9f533dd9631c8ce7aa"),
    ("crates/spark-model/src/weight_map/cutlass_scale_layout.rs", "ab6c8b7fa565c4b4e926b5c89343b92e477c9c0c736cf923848087c7b4366524"),
    ("crates/spark-model/src/weight_map/quantized.rs", "087be281d8b80c4ff09ea9c19b350ee7c331e434ccc5966a8769d9d22fb29355"),
    ("crates/spark-model/src/weight_loader/qwen35_dense.rs", "fc8c5e51db96cde3c30e8284202ad20c293bb5e58c561cb5176da783696cbd18"),
    ("crates/spark-model/src/layers/qwen3_ssm/trait_prefill.rs", "4211102b90012f19eaffcd15f218f7f7bc69c1782b7622a3862fbb3db7a933e4"),
    ("crates/spark-model/src/weight_map/fp8_lut.rs", "762e4cbc7f2f401db00149f7d0e978a038cff19f4492b8f976c0fcd9789051c4"),
    ("crates/spark-model/src/weight_map/loaders_fp8.rs", "e295fc317da09c90811bb5c8978745b9be7b3035a7b1625dd7242923248b2c4e"),
    ("crates/spark-model/src/weight_map/ssm_qwen35.rs", "c634f0a200f84043ea995cbf62b101f85f99ad2c26850c3dd6005f577f142d9c"),
    ("kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu", "80a4d6a3e55f47747545ccf16d5b89e854430b00ef5a21920e263efc47b5ec31"),
    ("kernels/gb10/common/quantize_bf16_to_nvfp4.cu", "2c6edf17f5c0fe22a9eb5b248bc5a0da21044143cf4ca9c3ccc67697c86011cf"),
    ("kernels/gb10/qwen3.8-27b/nvfp4/quantize_bf16_to_nvfp4_cutlass.cu", "506cf6498f7f6ef2dd16d20e99995e7b7fe05737c634fc2b85e51bc808ca4779"),
];
#[rustfmt::skip]
const OWN_SOURCES: [(&str, &str); 7] = [
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate.rs", include_str!("../qwen38_flashinfer_ssm_qkvz_dynamic_microgate.rs")),
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate/evidence.rs", include_str!("evidence.rs")),
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate/buffers.rs", include_str!("buffers.rs")),
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate/runtime.rs", include_str!("runtime.rs")),
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate/driver.rs", include_str!("driver.rs")),
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate/producer.rs", include_str!("producer.rs")),
    ("crates/spark-model/examples/qwen38_flashinfer_ssm_qkvz_dynamic_microgate/producer_io.rs", include_str!("producer_io.rs")),
];
#[rustfmt::skip]
fn root() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../..") }

#[rustfmt::skip]
pub(super) fn valid_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

pub(super) fn sha256_file(path: &Path) -> Result<String> {
    let output = std::process::Command::new("/usr/bin/sha256sum")
        .arg(path)
        .output()
        .with_context(|| format!("hash {}", path.display()))?;
    ensure!(
        output.status.success(),
        "sha256sum failed for {}",
        path.display()
    );
    let text = String::from_utf8(output.stdout)?;
    let digest = text.split_whitespace().next().context("missing SHA256")?;
    ensure!(valid_hex(digest), "invalid SHA256 for {}", path.display());
    Ok(digest.to_owned())
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> Result<String> {
    use std::io::Write;
    let mut child = std::process::Command::new("/usr/bin/sha256sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("sha256sum stdin")?
        .write_all(bytes)?;
    let output = child.wait_with_output()?;
    ensure!(output.status.success(), "sha256sum bytes failed");
    let text = String::from_utf8(output.stdout)?;
    let digest = text
        .split_whitespace()
        .next()
        .context("missing byte SHA256")?;
    ensure!(valid_hex(digest), "invalid byte SHA256");
    Ok(digest.to_owned())
}

pub(super) fn attest_sources() -> Result<BTreeMap<String, String>> {
    let root = root();
    let mut result = BTreeMap::new();
    for (relative, expected) in SOURCES {
        let actual = sha256_file(&root.join(relative))?;
        ensure!(
            actual == expected,
            "source drift: {relative}: {actual} != {expected}"
        );
        result.insert(relative.to_owned(), actual);
    }
    for (own, embedded) in OWN_SOURCES {
        let actual = sha256_file(&root.join(own))?;
        ensure!(
            actual == sha256_bytes(embedded.as_bytes())?,
            "built/source drift: {own}"
        );
        result.insert(own.to_owned(), actual);
    }
    Ok(result)
}

#[rustfmt::skip]
pub(super) fn executable_identity(runtime: bool) -> Result<Value> {
    let path = std::env::current_exe()?.canonicalize()?; let metadata = path.metadata()?;
    ensure!(metadata.is_file(), "executable is not a regular file"); let hash = sha256_file(&path)?; let after = path.metadata()?;
    ensure!(metadata.dev() == after.dev() && metadata.ino() == after.ino() && metadata.len() == after.len()
        && metadata.mode() == after.mode(), "executable changed during capture");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    if runtime { ensure!(!cfg!(debug_assertions), "runtime requires a release build");
        ensure!(metadata.mode() & 0o777 == 0o555, "runtime executable must be mode 0555"); }
    Ok(json!({"path":path,"sha256":hash,"profile":profile,"mode":metadata.mode() & 0o777,
        "device":metadata.dev(),"inode":metadata.ino(),"bytes":metadata.len()}))
}

#[rustfmt::skip]
pub(super) fn recheck_executable(identity: &Value) -> Result<()> {
    let path = PathBuf::from(identity.get("path").and_then(Value::as_str).context("identity path")?); let metadata = path.metadata()?;
    ensure!(metadata.is_file(), "executable ceased to be regular"); let hash = sha256_file(&path)?; let after = path.metadata()?;
    ensure!(metadata.dev() == after.dev() && metadata.ino() == after.ino() && metadata.len() == after.len()
        && metadata.mode() == after.mode(), "executable changed during recheck");
    ensure!(identity.get("sha256").and_then(Value::as_str) == Some(&hash), "executable hash drift");
    ensure!(identity.get("device").and_then(Value::as_u64) == Some(metadata.dev())
        && identity.get("inode").and_then(Value::as_u64) == Some(metadata.ino())
        && identity.get("bytes").and_then(Value::as_u64) == Some(metadata.len())
        && identity.get("mode").and_then(Value::as_u64) == Some(u64::from(metadata.mode() & 0o777)), "executable identity drift"); Ok(())
}

pub(super) fn bundle_identity() -> Result<(Vec<(&'static str, &'static str)>, Value)> {
    let modules = exact_bundle()?;
    let mut direct = BTreeMap::new();
    let mut manifest = Vec::new();
    for (name, ptx) in &modules {
        let digest = sha256_bytes(ptx.as_bytes())?;
        manifest.extend_from_slice(name.as_bytes());
        manifest.push(0);
        manifest.extend_from_slice(digest.as_bytes());
        manifest.push(b'\n');
        if matches!(
            *name,
            "w4a16" | "quantize_nvfp4" | "quantize_bf16_to_nvfp4_cutlass"
        ) {
            direct.insert((*name).to_owned(), digest);
        }
    }
    ensure!(
        direct.len() == 3,
        "embedded bundle lacks an executed module"
    );
    let value = json!({"target":"sm_121/qwen3.8-27b/nvfp4","module_count":modules.len(),"sha256":sha256_bytes(&manifest)?,"direct_modules":direct});
    Ok((modules, value))
}

pub(super) fn checkpoint_shards(
    checkpoint: &Checkpoint,
    layer: usize,
) -> Result<BTreeMap<String, String>> {
    let base = format!("model.language_model.layers.{layer}.linear_attn");
    let mut names = BTreeSet::new();
    for part in ["in_proj_qkv", "in_proj_z"] {
        let prefix = format!("{base}.{part}");
        for suffix in [
            "weight",
            "weight_packed",
            "weight_scale",
            "input_scale",
            "input_global_scale",
            "weight_scale_2",
            "weight_global_scale",
        ] {
            let key = format!("{prefix}.{suffix}");
            if let Some(shard) = checkpoint.index.get(&key).and_then(Value::as_str) {
                names.insert(shard.to_owned());
            }
        }
    }
    ensure!(!names.is_empty(), "no QKVZ checkpoint shards found");
    names
        .into_iter()
        .map(|name| {
            let path = checkpoint.root.join(&name).canonicalize()?;
            ensure!(
                path.starts_with(&checkpoint.root),
                "checkpoint shard escapes root"
            );
            Ok((name, sha256_file(&path)?))
        })
        .collect()
}

pub(super) fn validate_nonce(value: &str) -> Result<()> {
    ensure!(valid_hex(value), "nonce must be fresh lowercase 64-hex");
    ensure!(
        value.bytes().collect::<BTreeSet<_>>().len() >= 8,
        "nonce lacks diversity"
    );
    Ok(())
}

pub(super) fn write_receipt(value: &Value) -> Result<()> {
    use std::io::Write;
    let path = PathBuf::from(required_env("ATLAS_SSM_QKVZ_RECEIPT")?);
    ensure!(
        path.is_absolute() && !path.exists(),
        "receipt must be a new absolute path"
    );
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o444))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn source_catalog_and_nonce_are_strict() {
        assert!(SOURCES.iter().all(|(_, hash)| valid_hex(hash)));
        assert!(attest_sources().is_ok());
        assert!(validate_nonce(&"0123456789abcdef".repeat(4)).is_ok());
        for bad in ["a".repeat(64), "A".repeat(64), "0".repeat(63)] {
            assert!(validate_nonce(&bad).is_err());
        }
        let mut wrong = executable_identity(false).unwrap();
        wrong["sha256"] = Value::String("0".repeat(64));
        assert!(recheck_executable(&wrong).is_err());
        let mut replaced = executable_identity(false).unwrap();
        replaced["inode"] = Value::from(0u64);
        assert!(recheck_executable(&replaced).is_err());
        if cfg!(debug_assertions) { assert!(executable_identity(true).is_err()); }
    }
}
