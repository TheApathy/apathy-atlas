// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off numerical prefill projections for Flash-Next: NVFP4 weights are
//! dequantized to a BF16 scratch (`w4a16_dequant_bf16`, same per-weight BF16
//! rounding as the W4A16 tensor-core GEMM) and multiplied with cuBLASLt
//! (BF16 x BF16, FP32 accumulation). NOT bit-exact versus the exact GEMV
//! family: accumulation order differs. Selected per family by
//! `ATLAS_QWEN4_PREFILL_FAST_PROJ=<list>` where the list is comma separated
//! from {ssm, hc, attn} or `all`; absent/0 keeps every exact path unchanged.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FAST_PROJ";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Selection {
    pub ssm: bool,
    pub hc: bool,
    pub attn: bool,
}

pub(crate) fn parse(value: Option<&str>) -> Result<Selection> {
    let mut sel = Selection::default();
    let Some(value) = value else { return Ok(sel) };
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return Ok(sel);
    }
    for item in value.split(',') {
        match item.trim() {
            "all" => sel = Selection { ssm: true, hc: true, attn: true },
            "ssm" => sel.ssm = true,
            "hc" => sel.hc = true,
            "attn" => sel.attn = true,
            other => bail!("{SELECTOR}: unknown family {other:?} (expected ssm, hc, attn, all)"),
        }
    }
    Ok(sel)
}

/// Parsed once per process; a malformed selector fails the first prefill.
pub(crate) fn selection() -> Result<Selection> {
    static SEL: OnceLock<Result<Selection, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let raw = std::env::var(SELECTOR).ok();
        parse(raw.as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("w4a16_dequant_bf16", "w4a16_dequant_bf16")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: dequant kernel unavailable: {e}"))
}

/// Grow-only BF16 weight scratch shared by every fast projection. All
/// callers run on the single prefill stream, so reuse is stream ordered.
fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static S: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
    let mut guard = S.lock().unwrap();
    if let Some((ptr, cap)) = *guard
        && cap >= bytes
    {
        return Ok(ptr);
    }
    // Round up so the SSM QKVZ (16384x2560) and attention QG (12288x2560)
    // shapes share one allocation; the old block is intentionally leaked
    // (at most one grow step in practice) to avoid freeing in-flight memory.
    let cap = bytes.max(16384 * 2560 * 2).next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{SELECTOR}: alloc {cap} B dequant scratch"))?;
    tracing::info!("QWEN4_FAST_PROJ scratch allocated bytes={cap}");
    *guard = Some((ptr, cap));
    Ok(ptr)
}

fn fold_scale2() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("ATLAS_QWEN4_FAST_PROJ_FOLD_SCALE2").ok().as_deref() == Some("1"))
}

fn tuned() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("ATLAS_CUBLAS_TUNED").ok().as_deref() == Some("1"))
}

/// `out[m, n] = act[m, k] @ dequant(weight)[n, k]^T`, all BF16 row-major.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_gemm(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    dequant_gemm_ex(gpu, act, weight, out, m, n, k, n, None, stream)
}

fn qk_norm_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_qk_norm_strided", "qwen4_qk_norm_strided")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: qk-norm kernel unavailable: {e}"))
}

/// In-place per-head RMS norm (`x * rsqrt(mean(x^2)+eps) * (1+w)`) over
/// `rows x heads` heads of `head_dim` inside rows of `row_stride` elements.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_strided(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    weight: DevicePtr,
    rows: usize,
    heads: usize,
    head_dim: usize,
    row_stride: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let kh = qk_norm_kernel(gpu)?;
    KernelLaunch::new(gpu, kh)
        .grid([(rows * heads) as u32, 1, 1])
        .block([head_dim.min(1024) as u32, 1, 1])
        .arg_ptr(x)
        .arg_ptr(weight)
        .arg_u32(heads as u32)
        .arg_u32(head_dim as u32)
        .arg_u32(row_stride as u32)
        .arg_f32(eps)
        .launch(stream)
}

/// General form: output row stride `ldc` elements and an optional gated-Q
/// weight-row permutation `(heads, head_dim)` applied during dequantization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_gemm_ex(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n > 0 && k > 0 && k % 16 == 0,
        "{SELECTOR}: unsupported GEMM shape m={m} n={n} k={k}"
    );
    anyhow::ensure!(
        !weight.weight.is_null() && !weight.weight_scale.is_null() && weight.weight_scale_2.is_finite(),
        "{SELECTOR}: invalid NVFP4 weight"
    );
    let kh = kernel(gpu)?;
    let w = scratch(gpu, n * k * 2)?;
    let groups = (n * k / 16) as u32;
    // LUT[nib] * fp8 has <= 7 significant bits, so it is EXACT in BF16; the
    // per-tensor scale2 is applied as the GEMM alpha in FP32 instead of being
    // folded into the rounded weight (default). ATLAS_QWEN4_FAST_PROJ_FOLD_SCALE2=1
    // restores the W4A16-GEMM-style bf16(LUT*fp8*scale2) operand.
    let fold = fold_scale2();
    let (dq_scale, alpha) = if fold { (weight.weight_scale_2, 1.0f32) } else { (1.0f32, weight.weight_scale_2) };
    KernelLaunch::new(gpu, kh)
        .grid([div_ceil(groups, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(dq_scale)
        .arg_ptr(w)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(qg.map_or(0, |(h, _)| h) as u32)
        .arg_u32(qg.map_or(0, |(_, d)| d) as u32)
        .launch(stream)?;
    if let Some((heads, head_dim)) = qg {
        anyhow::ensure!(heads * head_dim * 2 == n, "{SELECTOR}: gated-Q permutation shape mismatch");
    }
    if tuned() {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_tuned_ex(
            act.0, w.0, out.0, m as u32, n as u32, k as u32, ldc as u32, alpha, stream,
        )
    } else {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_ldc(
            act.0, w.0, out.0, m as u32, n as u32, k as u32, ldc as u32, alpha, stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_selector_lists() {
        assert_eq!(parse(None).unwrap(), Selection::default());
        assert_eq!(parse(Some("0")).unwrap(), Selection::default());
        assert_eq!(parse(Some("ssm")).unwrap(), Selection { ssm: true, hc: false, attn: false });
        assert_eq!(parse(Some("ssm,hc")).unwrap(), Selection { ssm: true, hc: true, attn: false });
        assert_eq!(parse(Some("all")).unwrap(), Selection { ssm: true, hc: true, attn: true });
        assert!(parse(Some("1")).is_err());
        assert!(parse(Some("moe")).is_err());
    }
}
