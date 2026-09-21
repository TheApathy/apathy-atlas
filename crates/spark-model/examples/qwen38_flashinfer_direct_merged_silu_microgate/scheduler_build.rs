// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;

use anyhow::{Result, ensure};
use serde_json::{Value, json};

use super::authority::HeldFile;
use super::scheduler_authority::{field, parse_manifest};
use super::scheduler_trust::{trusted_root_file, validate_build_recipe};
use super::{BinaryIdentity, BundleIdentity};

pub(super) const GATE_SOURCE_PATHS: [&str; 18] = [
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/authority.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/contract.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/guarded.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/hostile_tests.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/launch.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/owner.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/owner_tests.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/provenance.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/runtime.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/runtime_helpers.inc.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/scheduler_authority.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/scheduler_authority_tests.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/scheduler_build.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/scheduler_build_tests.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/scheduler_trust.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/tests.rs",
    "crates/spark-model/examples/qwen38_flashinfer_direct_merged_silu_microgate/timing.rs",
];
pub(super) const TOOL_NAMES: [&str; 7] =
    ["sha256sum", "cargo", "rustc", "nvcc", "ptxas", "cxx", "ld"];
pub(super) const RUNTIME_ENVIRONMENT: &str = "ATLAS_DIRECT_MERGED_SILU_ATTEST_ONLY=0;ATLAS_DIRECT_MERGED_SILU_TIMING=1;ATLAS_TARGET_MODEL=qwen3.8-27b;ATLAS_TARGET_QUANT=nvfp4;LC_ALL=C";

pub(super) struct BuildAuthority {
    held: Vec<HeldFile>,
    evidence: Value,
}

impl BuildAuthority {
    pub(super) fn evidence(&self) -> &Value {
        &self.evidence
    }

    pub(super) fn verify_unchanged(&self) -> Result<()> {
        for held in &self.held {
            held.verify_unchanged()?;
        }
        Ok(())
    }
}

pub(super) fn manifest_field_count(source_count: usize, direct_count: usize) -> usize {
    19 + source_count + direct_count + GATE_SOURCE_PATHS.len() + TOOL_NAMES.len() * 2
}

pub(super) fn validate_build_manifest(manifest: &BTreeMap<&str, &str>) -> Result<()> {
    ensure!(
        field(manifest, "runtime_environment")? == RUNTIME_ENVIRONMENT,
        "runtime environment authority changed"
    );
    validate_build_recipe(
        field(manifest, "build_argv")?,
        field(manifest, "build_environment")?,
        field(manifest, "tool:cargo:path")?,
    )?;
    ensure!(
        Path::new(field(manifest, "build_receipt_path")?).is_absolute()
            && valid_sha256(field(manifest, "build_receipt_sha256")?),
        "build receipt authority is not canonical"
    );
    for path in GATE_SOURCE_PATHS {
        ensure!(
            valid_sha256(field(manifest, &format!("gate_source:{path}"))?),
            "gate source authority is not canonical"
        );
    }
    for name in TOOL_NAMES {
        ensure!(
            Path::new(field(manifest, &format!("tool:{name}:path"))?).is_absolute()
                && valid_sha256(field(manifest, &format!("tool:{name}:sha256"))?),
            "tool authority is not canonical"
        );
    }
    Ok(())
}

pub(super) fn open_build_authority(
    manifest: &BTreeMap<&str, &str>,
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> Result<BuildAuthority> {
    validate_build_manifest(manifest)?;
    exact_runtime_environment(manifest)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut held = Vec::new();
    let mut gate_sources = BTreeMap::new();
    for path in GATE_SOURCE_PATHS {
        let key = format!("gate_source:{path}");
        let expected = field(manifest, &key)?;
        held.push(HeldFile::open(&root.join(path), Some(expected))?);
        gate_sources.insert(path, expected);
    }
    let mut tools = BTreeMap::new();
    for name in TOOL_NAMES {
        let path = field(manifest, &format!("tool:{name}:path"))?;
        let sha256 = field(manifest, &format!("tool:{name}:sha256"))?;
        ensure!(Path::new(path).is_absolute(), "tool path is not absolute");
        held.push(HeldFile::open(Path::new(path), Some(sha256))?);
        tools.insert(name, json!({"path":path,"sha256":sha256}));
    }
    let receipt_path = Path::new(field(manifest, "build_receipt_path")?);
    trusted_root_file(receipt_path, 0o444, "build receipt")?;
    let receipt = HeldFile::open(receipt_path, Some(field(manifest, "build_receipt_sha256")?))?;
    let receipt_text = receipt.read_text()?;
    let receipt_fields = parse_manifest(&receipt_text)?;
    bind_build_receipt(&receipt_fields, manifest, binary, bundle, sources)?;
    let evidence = json!({
        "build_receipt":{"path":receipt_path.display().to_string(),"sha256":field(manifest,"build_receipt_sha256")?},
        "build_argv":field(manifest,"build_argv")?,
        "build_environment":field(manifest,"build_environment")?,
        "runtime_environment":RUNTIME_ENVIRONMENT,
        "gate_source_sha256":gate_sources,
        "tools":tools,
    });
    held.push(receipt);
    Ok(BuildAuthority { held, evidence })
}

pub(super) fn bind_build_receipt(
    receipt: &BTreeMap<&str, &str>,
    manifest: &BTreeMap<&str, &str>,
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> Result<()> {
    let mut expected = BTreeMap::from([
        (
            "schema".to_owned(),
            "qwen38-direct-merged-silu-build-v1".to_owned(),
        ),
        ("binary_sha256".to_owned(), binary.sha256.clone()),
        ("bundle_sha256".to_owned(), bundle.sha256.clone()),
        ("bundle_target".to_owned(), bundle.target.clone()),
        (
            "bundle_module_count".to_owned(),
            bundle.module_count.to_string(),
        ),
        (
            "build_argv".to_owned(),
            field(manifest, "build_argv")?.to_owned(),
        ),
        (
            "build_environment".to_owned(),
            field(manifest, "build_environment")?.to_owned(),
        ),
    ]);
    for (path, sha256) in sources {
        expected.insert(format!("source:{path}"), (*sha256).to_owned());
    }
    for (name, sha256) in &bundle.direct_modules {
        expected.insert(format!("ptx:{name}"), sha256.clone());
    }
    for path in GATE_SOURCE_PATHS {
        let key = format!("gate_source:{path}");
        expected.insert(key.clone(), field(manifest, &key)?.to_owned());
    }
    for name in TOOL_NAMES {
        for suffix in ["path", "sha256"] {
            let key = format!("tool:{name}:{suffix}");
            expected.insert(key.clone(), field(manifest, &key)?.to_owned());
        }
    }
    ensure!(
        receipt.len() == expected.len()
            && expected.iter().all(
                |(key, value)| field(receipt, key).is_ok_and(|actual| actual == value.as_str())
            ),
        "build receipt binding changed"
    );
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn exact_runtime_environment(manifest: &BTreeMap<&str, &str>) -> Result<()> {
    ensure!(
        field(manifest, "runtime_environment")? == RUNTIME_ENVIRONMENT,
        "runtime environment authority changed"
    );
    let expected = [
        ("ATLAS_DIRECT_MERGED_SILU_ATTEST_ONLY", "0"),
        ("ATLAS_DIRECT_MERGED_SILU_TIMING", "1"),
        ("ATLAS_TARGET_MODEL", "qwen3.8-27b"),
        ("ATLAS_TARGET_QUANT", "nvfp4"),
        ("LC_ALL", "C"),
    ]
    .into_iter()
    .map(|(key, value)| (OsString::from(key), OsString::from(value)))
    .collect::<BTreeMap<_, _>>();
    ensure!(
        std::env::vars_os().collect::<BTreeMap<_, _>>() == expected,
        "qualification runtime environment is not sanitized"
    );
    Ok(())
}
