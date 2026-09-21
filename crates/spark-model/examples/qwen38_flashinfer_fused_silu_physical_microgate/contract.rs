// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::KernelHandle;

pub(super) const K: u32 = 17_408;
pub(super) const DOWN_N: u32 = 128;
pub(super) const SCALE2: f32 = 1.0 / 896.0;
pub(super) const REDZONE: usize = 4096;
pub(super) const SCHEMA: &str = "qwen38-fused-silu-physical-raw-v1";
pub(super) const CUDA_SHA: &str =
    "2c6edf17f5c0fe22a9eb5b248bc5a0da21044143cf4ca9c3ccc67697c86011cf";
pub(super) const WRAPPER_SHA: &str =
    "9dfb6b64da3dd21cd83fe03d8308c3df486b50f3f0d8aa9f533dd9631c8ce7aa";
pub(super) const ROUTE_SHA: &str =
    "aae9aec6f202142ad75da031305f72819689ae882d940a69001dc810058c9da3";
pub(super) const SILU_SHA: &str =
    "4b82914f131e84d1aea37f7c70152cec0ae12dc63bd9e975c046c6806788bdfc";
pub(super) const PHYSICAL_SHA: &str =
    "506cf6498f7f6ef2dd16d20e99995e7b7fe05737c634fc2b85e51bc808ca4779";
pub(super) const DOWN_SHA: &str =
    "da1f5beb05a99e546aa622f4c9ec40bbc90d2864dc144f64b25de4966111c760";
pub(super) const DIRECT_MODULES: [&str; 4] = [
    "moe_silu_mul",
    "quantize_bf16_to_nvfp4_cutlass",
    "quantize_nvfp4",
    "nvfp4_cutlass",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BinaryIdentity {
    pub(super) path: String,
    pub(super) sha256: String,
    pub(super) profile: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BundleIdentity {
    pub(super) target: String,
    pub(super) module_count: usize,
    pub(super) sha256: String,
    pub(super) direct_modules: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub(super) m: u32,
    pub(super) padded: u32,
    pub(super) elements: usize,
    pub(super) packed: usize,
    pub(super) scales: usize,
    pub(super) tail: usize,
    pub(super) reps: usize,
}

impl Plan {
    pub(super) fn checked(m: u32, k: u32, reps: usize, scale2: f32) -> Result<Self> {
        ensure!(
            matches!(m, 2_079 | 8_192) && k == K,
            "requires exact M=2079|8192,K=17408"
        );
        ensure!(
            scale2.is_finite() && scale2 > 0.0,
            "scale2 must be finite and positive"
        );
        let minimum = if m == 2_079 { 21 } else { 31 };
        ensure!(
            reps >= minimum,
            "M={m} requires at least {minimum} timing pairs"
        );
        let padded = m.checked_add(127).context("row padding overflow")? / 128 * 128;
        let elements = usize::try_from(m)?
            .checked_mul(usize::try_from(k)?)
            .context("element overflow")?;
        let scales = usize::try_from(padded)?
            .checked_mul(usize::try_from(k / 16)?)
            .context("scale overflow")?;
        let tail = usize::try_from(padded - m)? * usize::try_from(k / 16)?;
        ensure!(
            m != 2_079 || tail == 105_536,
            "M2079 physical tail extent changed"
        );
        ensure!(m != 8_192 || tail == 0, "M8192 must not have a padded tail");
        Ok(Self {
            m,
            padded,
            elements,
            packed: elements / 2,
            scales,
            tail,
            reps,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub(super) silu: KernelHandle,
    pub(super) parent: KernelHandle,
    pub(super) fused: KernelHandle,
    pub(super) down: KernelHandle,
}
