// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::KernelHandle;

pub(super) const K: u32 = 17_408;
pub(super) const DOWN_N: u32 = 128;
pub(super) const SCALE2: f32 = 1.0 / 896.0;
pub(super) const REDZONE: usize = 4096;
pub(super) const SCHEMA: &str = "qwen38-direct-merged-silu-raw-v1";
pub(super) const QUANT_CUDA_SHA: &str =
    "8e17d307c551a0a999b16f43c2bcde7bb83a7357c95e52f024b2c142055593d7";
pub(super) const SPLIT_CUDA_SHA: &str =
    "cb8772e4962f487a5b27a159212d0662997123f94425fdd641f009d6e0a60a15";
pub(super) const DOWN_CUDA_SHA: &str =
    "da1f5beb05a99e546aa622f4c9ec40bbc90d2864dc144f64b25de4966111c760";
pub(super) const WRAPPER_SHA: &str =
    "9677b1778ab4bb22974e6242f3c1bdd83c7dceaf1dfdd911fe6e482c55a76c40";
pub(super) const ROUTE_SHA: &str =
    "9070abc77602e3baeec2ed7e5905c289ade39ceeae17bd11c16b32caaa94b6ce";
pub(super) const MANIFEST_SHA: &str =
    "e15f5c2fc6741ae14229812f6de1b2ffaed08b0030d659a442c0daf5cbcb1e17";
pub(super) const DIRECT_MODULES: [&str; 3] = [
    "flashinfer_projection_split",
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
    pub(super) merged_bytes: usize,
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
        let merged_bytes = elements.checked_mul(4).context("merged byte overflow")?;
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
            merged_bytes,
            packed: elements / 2,
            scales,
            tail,
            reps,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub(super) split: KernelHandle,
    pub(super) parent: KernelHandle,
    pub(super) candidate: KernelHandle,
    pub(super) down: KernelHandle,
}
