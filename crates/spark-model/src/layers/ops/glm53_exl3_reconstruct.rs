// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-scope EXL3 projection via fused reconstruction + F16 GEMM.
//!
//! The pinned cooperative trellis GEMM re-decodes the whole weight matrix for
//! every 16-row M tile (`while (size_m_ > 0) { ... size_m_ -= 16; }`), so a
//! 2047-row prompt decodes each dense projection 128 times. Upstream ExLlamaV3
//! switches to "reconstruct, then hgemm" above `AUTO_RECONSTRUCT_THRESHOLD`
//! (144 rows) for exactly this reason. This module ports that policy: the
//! fused `reconstruct_had` kernel emits the ORIGINAL-basis F16 weight once per
//! projection per layer, and cuBLASLt multiplies the raw F16 activation by it.
//!
//! Numerics: not bit-identical to the trellis GEMM (Hadamards are folded into
//! the weight; summation order differs). Opt-in only, and only inside the
//! explicit layer-major prompt scope; decode/verify paths are untouched.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

use super::glm53_exl3_projection::{
    Glm53Exl3ProjectionBuffers, Glm53Exl3ProjectionPlan, Glm53Exl3ProjectionScratch,
    glm53_layer_major_prefill_active,
};
use super::{Glm53Exl3CastKernels, Glm53Exl3CastPlan};
use crate::weight_loader::Glm53Exl3Linear;

const MODULE: &str = "glm53_exl3_reconstruct";
const THREADS: u32 = 256;
const TILE: u32 = 128;
const DEFAULT_MIN_ROWS: u32 = 144;
const ENABLE_ENV: &str = "ATLAS_GLM53_EXL3_PREFILL_RECONSTRUCT";
const MIN_ROWS_ENV: &str = "ATLAS_GLM53_EXL3_RECONSTRUCT_MIN_ROWS";
const DTYPE_ENV: &str = "ATLAS_GLM53_EXL3_RECONSTRUCT_DTYPE";
const PRECISION_ENV: &str = "ATLAS_GLM53_EXL3_RECONSTRUCT_PRECISION";
const OVERLAP_ENV: &str = "ATLAS_GLM53_EXL3_RECONSTRUCT_OVERLAP";

/// `ATLAS_GLM53_EXL3_RECONSTRUCT_OVERLAP=1`: reconstruct on a side stream into
/// alternating staging slots so reconstruct(i+1) overlaps GEMM(i). Pure
/// scheduling; the arithmetic is unchanged.
fn overlap_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var(OVERLAP_ENV) {
            Ok(v) if v == "1" => Ok(true),
            Ok(v) if v == "0" => Ok(false),
            Ok(other) => Err(format!("{OVERLAP_ENV} must be 0 or 1, got {other:?}")),
            Err(std::env::VarError::NotPresent) => Ok(false),
            Err(e) => Err(format!("{OVERLAP_ENV}: {e}")),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Side stream + per-slot events: `recon_done[slot]` (side -> main) and
/// `gemm_done[slot]` (main -> side, slot free). Process-global, single GPU.
struct Overlap {
    side_stream: u64,
    recon_done: [u64; 2],
    gemm_done: [u64; 2],
    next_slot: std::sync::atomic::AtomicUsize,
}

fn overlap(gpu: &dyn GpuBackend) -> Result<&'static Overlap> {
    static OVERLAP: OnceLock<std::result::Result<Overlap, String>> = OnceLock::new();
    OVERLAP
        .get_or_init(|| {
            let make = || -> Result<Overlap> {
                Ok(Overlap {
                    side_stream: gpu.create_stream()?,
                    recon_done: [gpu.create_event()?, gpu.create_event()?],
                    gemm_done: [gpu.create_event()?, gpu.create_event()?],
                    next_slot: std::sync::atomic::AtomicUsize::new(0),
                })
            };
            make().map_err(|e| format!("{e:#}"))
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}
const TILE_F32_BYTES: u32 = 128 * 128 * 4;

/// `f32` (default): Hadamard butterflies in f32, one rounding at the store
/// (64 KB dynamic shared tile). `f16`: upstream's fp16 butterflies.
fn butterfly_f32() -> Result<bool> {
    static PRECISION: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    PRECISION
        .get_or_init(|| match std::env::var(PRECISION_ENV) {
            Ok(value) if value == "f32" => Ok(true),
            Ok(value) if value == "f16" => Ok(false),
            Ok(other) => Err(format!("{PRECISION_ENV} must be f32 or f16, got {other:?}")),
            Err(std::env::VarError::NotPresent) => Ok(true),
            Err(error) => Err(format!("{PRECISION_ENV}: {error}")),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagingDtype {
    /// F16 weight, F16 cuBLASLt GEMM, bf16<->f16 casts around it (upstream form).
    F16,
    /// BF16 weight (same F16 value path, final store rounded to BF16), BF16 GEMM on
    /// the raw activation: no casts. Adds one BF16 rounding per weight.
    Bf16,
}

fn staging_dtype() -> Result<StagingDtype> {
    static DTYPE: OnceLock<std::result::Result<StagingDtype, String>> = OnceLock::new();
    DTYPE
        .get_or_init(|| match std::env::var(DTYPE_ENV) {
            Ok(value) if value == "f16" => Ok(StagingDtype::F16),
            Ok(value) if value == "bf16" => Ok(StagingDtype::Bf16),
            Ok(other) => Err(format!("{DTYPE_ENV} must be f16 or bf16, got {other:?}")),
            Err(std::env::VarError::NotPresent) => Ok(StagingDtype::F16),
            Err(error) => Err(format!("{DTYPE_ENV}: {error}")),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// `Some(min_rows)` when the opt-in is set; parsed once per process.
fn policy() -> Result<Option<u32>> {
    static POLICY: OnceLock<std::result::Result<Option<u32>, String>> = OnceLock::new();
    POLICY
        .get_or_init(|| {
            let enabled = match std::env::var(ENABLE_ENV) {
                Ok(value) if value == "1" => true,
                Ok(value) if value == "0" => false,
                Ok(other) => return Err(format!("{ENABLE_ENV} must be 0 or 1, got {other:?}")),
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => return Err(format!("{ENABLE_ENV}: {error}")),
            };
            if !enabled {
                return Ok(None);
            }
            let min_rows = match std::env::var(MIN_ROWS_ENV) {
                Ok(value) => value
                    .parse::<u32>()
                    .map_err(|e| format!("{MIN_ROWS_ENV}: {e}"))?,
                Err(std::env::VarError::NotPresent) => DEFAULT_MIN_ROWS,
                Err(error) => return Err(format!("{MIN_ROWS_ENV}: {error}")),
            };
            if min_rows < 2 {
                return Err(format!("{MIN_ROWS_ENV} must be >= 2"));
            }
            Ok(Some(min_rows))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// True when this projection call should take the reconstruct path.
pub fn reconstruct_prefill_rows(rows: u32) -> Result<bool> {
    if !glm53_layer_major_prefill_active() {
        return Ok(false);
    }
    Ok(policy()?.is_some_and(|min_rows| rows >= min_rows))
}

pub struct Glm53Exl3ReconstructKernels;

impl Glm53Exl3ReconstructKernels {
    fn symbol(bits: u8, dtype: StagingDtype, f32_butterfly: bool) -> Result<String> {
        ensure!(
            matches!(bits, 2..=5),
            "GLM EXL3 reconstruct admits K2..K5, got K{bits}"
        );
        let precision = if f32_butterfly { "_f32" } else { "" };
        let suffix = match dtype {
            StagingDtype::F16 => "",
            StagingDtype::Bf16 => "_bf16",
        };
        Ok(format!(
            "atlas_glm53_exl3_reconstruct_had_k{bits}_cb2{precision}{suffix}"
        ))
    }

    /// Dynamic shared bytes of the f32-butterfly kernel: packed tile + f32 tile.
    fn f32_shared_bytes(bits: u8) -> u32 {
        8 * 8 * (256 * u32::from(bits) / 16) * 2 + TILE_F32_BYTES
    }
}

/// Reconstruct `linear` into `scratch.reconstruct_f16` and multiply.
///
/// Returns `Ok(false)` (caller falls back to the trellis GEMM) only when the
/// staging buffer cannot hold this matrix; every other failure is an error.
pub fn launch_reconstructed(
    gpu: &dyn GpuBackend,
    linear: &Glm53Exl3Linear,
    plan: Glm53Exl3ProjectionPlan,
    buffers: Glm53Exl3ProjectionBuffers,
    scratch: Glm53Exl3ProjectionScratch,
    stream: u64,
) -> Result<bool> {
    let k = linear.size_k();
    let n = linear.size_n();
    ensure!(
        plan.compressed && plan.input == k && plan.output == n,
        "GLM EXL3 reconstruct plan/linear geometry drift"
    );
    ensure!(
        k % TILE == 0 && n % TILE == 0,
        "GLM EXL3 reconstruct needs 128-divisible dimensions"
    );
    let weight_bytes = usize::try_from(k)?
        .checked_mul(usize::try_from(n)?)
        .and_then(|e| e.checked_mul(2))
        .context("GLM EXL3 reconstruct weight byte overflow")?;
    let use_overlap = overlap_enabled()? && !scratch.reconstruct_f16_b.ptr.is_null();
    let (staging, slot, ov) = if use_overlap {
        let ov = overlap(gpu)?;
        let slot = ov.next_slot.fetch_xor(1, std::sync::atomic::Ordering::Relaxed);
        (
            if slot == 0 {
                scratch.reconstruct_f16
            } else {
                scratch.reconstruct_f16_b
            },
            slot,
            Some(ov),
        )
    } else {
        (scratch.reconstruct_f16, 0, None)
    };
    if staging.ptr.is_null() || staging.bytes < weight_bytes {
        static WARNED: OnceLock<()> = OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                k, n, staging_bytes = staging.bytes,
                "GLM EXL3 reconstruct staging too small; projection falls back to trellis GEMM"
            );
        });
        return Ok(false);
    }
    let trellis = linear.trellis_buffer();
    let suh = linear.scale_in_buffer();
    let svh = linear.scale_out_buffer();
    ensure!(
        trellis.bytes == plan_trellis_bytes(k, n, linear.bits())?
            && suh.bytes == usize::try_from(k)? * 2
            && svh.bytes == usize::try_from(n)? * 2,
        "GLM EXL3 reconstruct operand extents drift"
    );
    let dtype = staging_dtype()?;
    let casts = Glm53Exl3CastKernels::load(gpu)?;
    if dtype == StagingDtype::F16 {
        casts.bf16_to_f16(
            gpu,
            Glm53Exl3CastPlan::new(plan.rows, k)?,
            buffers.input_bf16,
            scratch.input_f16,
            stream,
        )?;
    }
    let f32_butterfly = butterfly_f32()?;
    let kernel = gpu.kernel(
        MODULE,
        &Glm53Exl3ReconstructKernels::symbol(linear.bits(), dtype, f32_butterfly)?,
    )?;
    let shared = if f32_butterfly {
        let bytes = Glm53Exl3ReconstructKernels::f32_shared_bytes(linear.bits());
        gpu.set_kernel_max_dynamic_shared_memory(kernel, bytes)?;
        bytes
    } else {
        0
    };
    // Overlap: the side stream must not overwrite a slot until the GEMM that
    // read it has finished; the main stream must not start the GEMM before the
    // reconstruct has landed.
    let recon_stream = if let Some(ov) = ov {
        gpu.stream_wait_event(ov.side_stream, ov.gemm_done[slot])?;
        ov.side_stream
    } else {
        stream
    };
    KernelLaunch::new(gpu, kernel)
        .grid([n / TILE, k / TILE, 1])
        .block([THREADS, 1, 1])
        .shared_mem(shared)
        .arg_ptr(staging.ptr)
        .arg_ptr(trellis.ptr)
        .arg_ptr(suh.ptr)
        .arg_ptr(svh.ptr)
        .arg_i32(i32::try_from(n / 16)?)
        .arg_i32(0)
        .launch(recon_stream)?;
    if let Some(ov) = ov {
        gpu.record_event(ov.recon_done[slot], ov.side_stream)?;
        gpu.stream_wait_event(stream, ov.recon_done[slot])?;
    }
    // Numerics oracle for the FP8-reconstruct question: capture ONE reconstructed
    // dense weight exactly as it leaves the reconstruct kernel, so the e4m3
    // round-trip error can be measured offline against this bf16/f16 baseline
    // instead of argued from the format. Fires once per process; free when unset.
    crate::model::glm53::oracle_dump::dump_site(
        gpu,
        recon_stream,
        "reconstruct_weight",
        &[crate::model::glm53::oracle_dump::t(
            "weight",
            staging.ptr,
            weight_bytes,
            match dtype {
                StagingDtype::F16 => "f16",
                StagingDtype::Bf16 => "bf16",
            },
            &[usize::try_from(k)?, usize::try_from(n)?],
        )],
        &[
            ("k", k.to_string()),
            ("n", n.to_string()),
            ("bits", linear.bits().to_string()),
            ("staging_dtype", format!("\"{:?}\"", dtype)),
        ],
    )?;
    match dtype {
        StagingDtype::F16 => {
            for (first, rows) in super::glm53_gemm_row_slices(plan.rows) {
                spark_runtime::cublaslt::f16_gemm_act_weight(
                    scratch.input_f16.ptr.0 + u64::from(first) * u64::from(k) * 2,
                    staging.ptr.0,
                    scratch.output_f16.ptr.0 + u64::from(first) * u64::from(n) * 2,
                    rows,
                    n,
                    k,
                    stream,
                )?;
            }
            casts.f16_to_bf16(
                gpu,
                Glm53Exl3CastPlan::new(plan.rows, n)?,
                scratch.output_f16,
                buffers.output_bf16,
                stream,
            )?;
        }
        StagingDtype::Bf16 => {
            for (first, rows) in super::glm53_gemm_row_slices(plan.rows) {
                spark_runtime::cublaslt::bf16_gemm_act_weight(
                    buffers.input_bf16.ptr.0 + u64::from(first) * u64::from(k) * 2,
                    staging.ptr.0,
                    buffers.output_bf16.ptr.0 + u64::from(first) * u64::from(n) * 2,
                    rows,
                    n,
                    k,
                    stream,
                )?;
            }
        }
    }
    if let Some(ov) = ov {
        gpu.record_event(ov.gemm_done[slot], stream)?;
    }
    Ok(true)
}

fn plan_trellis_bytes(k: u32, n: u32, bits: u8) -> Result<usize> {
    if !matches!(bits, 2..=5) {
        bail!("GLM EXL3 reconstruct admits K2..K5");
    }
    Ok(usize::try_from(u64::from(k) * u64::from(n))? * usize::from(bits) / 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_are_pinned_per_bit_width() {
        assert_eq!(
            Glm53Exl3ReconstructKernels::symbol(2, StagingDtype::F16, false).unwrap(),
            "atlas_glm53_exl3_reconstruct_had_k2_cb2"
        );
        assert_eq!(
            Glm53Exl3ReconstructKernels::symbol(3, StagingDtype::Bf16, false).unwrap(),
            "atlas_glm53_exl3_reconstruct_had_k3_cb2_bf16"
        );
        assert_eq!(
            Glm53Exl3ReconstructKernels::symbol(4, StagingDtype::F16, true).unwrap(),
            "atlas_glm53_exl3_reconstruct_had_k4_cb2_f32"
        );
        assert_eq!(Glm53Exl3ReconstructKernels::f32_shared_bytes(4), 8 * 8 * 64 * 2 + 65_536);
        assert!(Glm53Exl3ReconstructKernels::symbol(6, StagingDtype::F16, false).is_err());
    }

    #[test]
    fn outside_the_layer_major_scope_the_path_is_off() {
        assert!(!reconstruct_prefill_rows(2_047).unwrap());
    }
}
