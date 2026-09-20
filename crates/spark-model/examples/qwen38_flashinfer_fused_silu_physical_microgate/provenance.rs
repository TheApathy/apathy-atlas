// SPDX-License-Identifier: AGPL-3.0-only

use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};

use super::{
    BinaryIdentity, BundleIdentity, CUDA_SHA, DIRECT_MODULES, DOWN_SHA, PHYSICAL_SHA, ROUTE_SHA,
    SILU_SHA, WRAPPER_SHA,
};

unsafe extern "C" {
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
}

pub(super) fn exact_bundle() -> Result<(String, Vec<(&'static str, &'static str)>)> {
    ensure!(
        std::env::var("ATLAS_TARGET_MODEL").as_deref() == Ok("qwen3.8-27b"),
        "requires ATLAS_TARGET_MODEL=qwen3.8-27b"
    );
    ensure!(
        std::env::var("ATLAS_TARGET_QUANT").as_deref() == Ok("nvfp4"),
        "requires ATLAS_TARGET_QUANT=nvfp4"
    );
    let mut sets: Vec<_> = atlas_kernels::available_targets()
        .into_iter()
        .filter(|set| {
            set.target.arch == "sm_121"
                && set.target.model == "qwen3.8-27b"
                && set.target.quant == "nvfp4"
        })
        .collect();
    ensure!(
        sets.len() == 1,
        "requires exactly one embedded SM121 Qwen3.8 NVFP4 bundle"
    );
    let set = sets.pop().context("bundle disappeared")?;
    Ok((set.target.to_string(), set.modules))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(super) fn check_hash(label: &str, actual: &str, expected: &str) -> Result<()> {
    ensure!(
        valid_sha256(actual) && actual == expected,
        "SHA256 mismatch for {label}"
    );
    Ok(())
}

fn output_sha256(output: std::process::Output, label: &str) -> Result<String> {
    ensure!(output.status.success(), "sha256sum failed for {label}");
    let hash = std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .context("sha256sum omitted digest")?
        .to_owned();
    ensure!(
        valid_sha256(&hash),
        "sha256sum emitted invalid digest for {label}"
    );
    Ok(hash)
}

fn sha256_file(path: &std::path::Path) -> Result<String> {
    output_sha256(
        Command::new("/usr/bin/sha256sum").arg(path).output()?,
        &path.display().to_string(),
    )
}

fn sha256_bytes(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("/usr/bin/sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .context("sha256sum stdin missing")?
        .write_all(bytes)?;
    output_sha256(child.wait_with_output()?, "embedded bytes")
}

pub(super) fn running_binary() -> Result<BinaryIdentity> {
    let path = std::fs::canonicalize(std::env::current_exe()?)?;
    let sha256 = sha256_file(&path)?;
    Ok(BinaryIdentity {
        path: path.to_string_lossy().into_owned(),
        sha256,
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    })
}

pub(super) fn stable_binary(
    start: &BinaryIdentity,
    end: &BinaryIdentity,
    require_release: bool,
) -> Result<()> {
    ensure!(
        !require_release || start.profile == "release",
        "qualification requires a release-profile executable"
    );
    ensure!(
        start == end,
        "running executable identity changed during qualification"
    );
    Ok(())
}

pub(super) fn bundle_identity(target: &str, modules: &[(&str, &str)]) -> Result<BundleIdentity> {
    ensure!(!modules.is_empty(), "embedded kernel bundle is empty");
    let mut identities = modules
        .iter()
        .map(|(name, ptx)| Ok(((*name).to_owned(), sha256_bytes(ptx.as_bytes())?)))
        .collect::<Result<Vec<_>>>()?;
    identities.sort();
    ensure!(
        identities.windows(2).all(|pair| pair[0].0 != pair[1].0),
        "embedded kernel bundle has duplicate module names"
    );
    let direct_modules = DIRECT_MODULES
        .iter()
        .map(|required| {
            identities
                .iter()
                .find(|(name, _)| name == required)
                .cloned()
                .with_context(|| format!("embedded bundle omitted direct module {required}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut manifest = format!("target={target}\nmodules={}\n", identities.len()).into_bytes();
    for (name, hash) in &identities {
        manifest.extend_from_slice(format!("{name}={hash}\n").as_bytes());
    }
    Ok(BundleIdentity {
        target: target.to_owned(),
        module_count: identities.len(),
        sha256: sha256_bytes(&manifest)?,
        direct_modules,
    })
}

pub(super) fn require_sources() -> Result<Vec<(&'static str, &'static str)>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sources = vec![
        ("kernels/gb10/common/quantize_bf16_to_nvfp4.cu", CUDA_SHA),
        ("kernels/gb10/common/moe_silu_mul.cu", SILU_SHA),
        (
            "kernels/gb10/qwen3.8-27b/nvfp4/quantize_bf16_to_nvfp4_cutlass.cu",
            PHYSICAL_SHA,
        ),
        (
            "kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu",
            DOWN_SHA,
        ),
        (
            "crates/spark-model/src/layers/ops/gemm_dense.rs",
            WRAPPER_SHA,
        ),
        ("crates/spark-model/src/layers/dense_ffn.rs", ROUTE_SHA),
    ];
    for (path, expected) in &sources {
        check_hash(path, &sha256_file(&root.join(path))?, expected)?;
    }
    Ok(sources)
}

pub(super) fn require_gb10() -> Result<()> {
    let mut device = -1;
    ensure!(
        unsafe { cuCtxGetDevice(&mut device) } == 0 && device >= 0,
        "CUDA context/device unavailable"
    );
    let attribute = |kind| -> Result<i32> {
        let mut value = -1;
        ensure!(
            unsafe { cuDeviceGetAttribute(&mut value, kind, device) } == 0,
            "CUDA attribute {kind} failed"
        );
        Ok(value)
    };
    ensure!(
        attribute(75)? == 12 && attribute(76)? == 1 && attribute(16)? == 48,
        "requires GB10 SM121 with 48 SMs"
    );
    Ok(())
}

pub(super) fn strict_switch(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("{name} must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be UTF-8"),
    }
}
