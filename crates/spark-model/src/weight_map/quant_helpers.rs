// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Shared CPU-side FP8 E4M3 → BF16 conversion.
pub(super) fn dequant_fp8_bytes_to_bf16(fp8_buf: &[u8], scale: f32) -> Vec<u8> {
    fp8_buf
        .iter()
        .flat_map(|&byte| {
            let val = fp8_e4m3_to_f32(byte) * scale;
            f32_to_bf16(val).to_le_bytes()
        })
        .collect()
}

/// Resolve a safetensors FP8 scale shape into its logical scale grid and the
/// number of weight rows/columns covered by each scale element.
///
/// Compressed-tensors mixed-precision checkpoints use `[N, 1]` for
/// per-output-channel scales, while native FP8 checkpoints normally use a
/// two-dimensional block grid such as `[N/128, K/128]`. Scalar scales are
/// also accepted for the ModelOpt per-tensor convention.
pub(super) fn fp8_scale_geometry(
    n: usize,
    k: usize,
    shape: &[usize],
) -> Result<(usize, usize, usize, usize)> {
    ensure!(n > 0 && k > 0, "FP8 weight dimensions must be non-zero");
    let (sn, sk) = match shape {
        [] => (1, 1),
        [sn] => (*sn, 1),
        [sn, sk] => (*sn, *sk),
        _ => bail!("FP8 scale must be scalar, 1-D, or 2-D; got shape {shape:?}"),
    };
    ensure!(sn > 0 && sk > 0, "FP8 scale dimensions must be non-zero");
    ensure!(
        n.is_multiple_of(sn) && k.is_multiple_of(sk),
        "FP8 scale shape {shape:?} does not evenly tile weight shape [{n}, {k}]"
    );
    Ok((sn, sk, n / sn, k / sk))
}

/// Validate and narrow the arguments consumed by the GPU FP8 dequant kernel.
///
/// The kernel accepts the same logical scale grids as the CPU reference:
/// scalar, one-dimensional per-row, two-dimensional per-row (`[N,1]`), and
/// native block grids. Unsupported dtypes and non-tiling shapes fail before
/// any allocation or launch.
pub(super) fn fp8_gpu_dequant_args(
    n: usize,
    k: usize,
    scale_shape: &[usize],
    scale_dtype: WeightDtype,
) -> Result<(u32, u32, u32, u32, u32, u32)> {
    ensure!(
        scale_dtype == WeightDtype::BF16 || scale_dtype == WeightDtype::FP32,
        "GPU FP8 dequant scale must be BF16 or FP32, got {scale_dtype:?}"
    );
    let (_sn, sk, block_n, block_k) = fp8_scale_geometry(n, k, scale_shape)?;
    Ok((
        u32::try_from(n).context("FP8 weight row count exceeds u32")?,
        u32::try_from(k).context("FP8 weight column count exceeds u32")?,
        u32::try_from(block_n).context("FP8 scale row block exceeds u32")?,
        u32::try_from(block_k).context("FP8 scale column block exceeds u32")?,
        u32::try_from(sk).context("FP8 scale column count exceeds u32")?,
        u32::from(scale_dtype == WeightDtype::FP32),
    ))
}

/// Dequantize FP8 E4M3 block/per-row/per-tensor-scaled weight → BF16.
///
/// Block-scaled FP8 (e.g. `quant_method: "fp8"` with `weight_block_size: [128, 128]`):
///   - `{prefix}.weight`: FP8E4M3 tensor of shape `[N, K]`
///   - `{prefix}.weight_scale_inv`: BF16 tensor of shape `[N/block, K/block]`, or
///   - `{prefix}.weight_scale`: BF16/FP32 tensor shaped as a block grid,
///     `[N,1]` per-row scales, or a scalar.
///   - Dequant: `bf16[i,j] = fp8[i,j] * scale[i/block_n, j/block_k]`
///
/// Returns a BF16 DenseWeight on GPU.
pub(crate) fn dequant_fp8_blockscaled_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let total = n
        .checked_mul(k)
        .context("FP8 weight element count overflows usize")?;
    let bf16_byte_size = total
        .checked_mul(2)
        .context("FP8 dequant BF16 allocation size overflows usize")?;
    let byte_size = w.byte_size();
    tracing::debug!(
        "FP8 blockscaled dequant: {prefix} shape=[{n},{k}] total={total} byte_size={byte_size} ptr={}",
        w.ptr.0,
    );
    ensure!(
        total == byte_size,
        "FP8 size mismatch: total={total} byte_size={byte_size}"
    );

    // Native FP8 uses `weight_scale_inv`; compressed-tensors float-quantized
    // groups use `weight_scale` (notably `[N,1]` in unsloth Qwen3.8). Both
    // store the direct multiplier consumed by the dequant equation above.
    let scale_inv_key = format!("{prefix}.weight_scale_inv");
    let scale_key = format!("{prefix}.weight_scale");
    let (selected_scale_key, s) = if let Ok(s) = store.get(&scale_inv_key) {
        (scale_inv_key, s)
    } else if let Ok(s) = store.get(&scale_key) {
        (scale_key, s)
    } else {
        bail!("FP8 tensor {prefix}: no .weight_scale_inv or .weight_scale found for dequant");
    };
    ensure!(
        s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
        "Expected BF16 or FP32 for {selected_scale_key}, got {:?}",
        s.dtype,
    );
    let (sn, sk, block_n, block_k) = fp8_scale_geometry(n, k, &s.shape)?;
    let scale_is_f32 = s.dtype == WeightDtype::FP32;

    // The CUDA kernel consumes the same multiplier grid as the CPU reference.
    // Qwen3.8's `[N,1]` scale becomes block_n=1, block_k=K, sk=1. Keep the
    // lookup optional so non-CUDA backends retain the exact CPU behavior, but
    // fail closed after a successful lookup: a bad launch/sync is not silently
    // converted into a benchmark-contaminating fallback.
    let (n_u32, k_u32, block_n_u32, block_k_u32, sk_u32, scale_is_f32_u32) =
        fp8_gpu_dequant_args(n, k, &s.shape, s.dtype)?;
    let stream = gpu.default_stream();
    match gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    ) {
        Ok(kernel) => {
            use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

            let out = gpu.alloc(bf16_byte_size)?;
            let launched = KernelLaunch::new(gpu, kernel)
                .grid([div_ceil(k_u32, 64), div_ceil(n_u32, 4), 1])
                .block([64, 4, 1])
                .arg_ptr(w.ptr)
                .arg_ptr(s.ptr)
                .arg_ptr(out)
                .arg_u32(n_u32)
                .arg_u32(k_u32)
                .arg_u32(block_n_u32)
                .arg_u32(block_k_u32)
                .arg_u32(sk_u32)
                .arg_u32(scale_is_f32_u32)
                .launch(stream)
                .and_then(|()| gpu.synchronize(stream));
            if let Err(error) = launched {
                let _ = gpu.free(out);
                return Err(error).with_context(|| {
                    format!(
                        "GPU FP8 dequant failed for {prefix}: weight=[{n},{k}] scale={:?}",
                        s.shape
                    )
                });
            }
            tracing::debug!(
                "GPU-dequanted FP8 {prefix}: [{n},{k}] scale=[{sn},{sk}] block=[{block_n},{block_k}] → BF16"
            );
            return Ok(DenseWeight { weight: out });
        }
        Err(error) => {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "GPU FP8 dequant kernel unavailable ({error:#}); using the numerically-identical CPU fallback"
                );
            }
        }
    }

    // Compatibility fallback for backends without the CUDA module. This is
    // intentionally after all metadata validation, so malformed layouts never
    // hide behind the fallback.
    gpu.synchronize(stream)?;
    let mut fp8_buf = vec![0u8; byte_size];
    gpu.copy_d2h(w.ptr, &mut fp8_buf).with_context(|| {
        let free = gpu.free_memory().unwrap_or(0);
        format!(
            "D2H failed for {prefix}.weight: ptr={}, size={byte_size}, free={:.1} GB",
            w.ptr.0,
            free as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    })?;

    let scale_bytes_per = if scale_is_f32 { 4 } else { 2 };
    let mut scale_buf = vec![0u8; sn * sk * scale_bytes_per];
    gpu.copy_d2h(s.ptr, &mut scale_buf).with_context(|| {
        format!(
            "D2H failed for {selected_scale_key}: ptr={}, size={}",
            s.ptr.0,
            sn * sk * scale_bytes_per
        )
    })?;

    // CPU dequant: bf16_out[i,j] = fp8[i,j] * scale[i/block_n, j/block_k]
    let mut bf16_out = vec![0u8; bf16_byte_size];
    for row in 0..n {
        let scale_row = row / block_n;
        for col in 0..k {
            let scale_col = col / block_k;
            let scale_idx = scale_row * sk + scale_col;
            let scale_f32 = if scale_is_f32 {
                let b = [
                    scale_buf[scale_idx * 4],
                    scale_buf[scale_idx * 4 + 1],
                    scale_buf[scale_idx * 4 + 2],
                    scale_buf[scale_idx * 4 + 3],
                ];
                f32::from_le_bytes(b)
            } else {
                let b = [scale_buf[scale_idx * 2], scale_buf[scale_idx * 2 + 1]];
                bf16_bytes_to_f32(b)
            };

            let fp8_byte = fp8_buf[row * k + col];
            let val = fp8_e4m3_to_f32(fp8_byte) * scale_f32;
            let bf16_val = f32_to_bf16(val);

            let out_idx = (row * k + col) * 2;
            let [lo, hi] = bf16_val.to_le_bytes();
            bf16_out[out_idx] = lo;
            bf16_out[out_idx + 1] = hi;
        }
    }

    // Diagnostic: print weight statistics for first few dequants
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DIAG_COUNT: AtomicUsize = AtomicUsize::new(0);
        let count = DIAG_COUNT.fetch_add(1, Ordering::Relaxed);
        if count < 3 {
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            let mut sum = 0.0f64;
            let mut zeros = 0usize;
            for i in 0..total {
                let lo = bf16_out[i * 2];
                let hi = bf16_out[i * 2 + 1];
                let v = bf16_bytes_to_f32([lo, hi]);
                if v == 0.0 {
                    zeros += 1;
                }
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
                sum += v as f64;
            }
            let mean = sum / total as f64;
            tracing::info!(
                "FP8 dequant stats for {prefix}: min={min_val:.6}, max={max_val:.6}, mean={mean:.6}, zeros={zeros}/{total}"
            );
            // First 8 values
            let vals: Vec<f32> = (0..8.min(total))
                .map(|i| bf16_bytes_to_f32([bf16_out[i * 2], bf16_out[i * 2 + 1]]))
                .collect();
            tracing::info!("  First 8 BF16 values: {:?}", vals);
        }
    }

    let ptr = gpu.alloc(bf16_out.len())?;
    gpu.copy_h2d(&bf16_out, ptr)?;

    // Diagnostic: readback first 8 BF16 values from GPU and compare with CPU
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static VERIFY_COUNT: AtomicUsize = AtomicUsize::new(0);
        if VERIFY_COUNT.fetch_add(1, Ordering::Relaxed) < 3 {
            let check_len = 16.min(bf16_out.len());
            let mut readback = vec![0u8; check_len];
            if gpu.copy_d2h(ptr, &mut readback).is_ok() {
                let match_ok = readback[..check_len] == bf16_out[..check_len];
                if !match_ok {
                    tracing::error!(
                        "BF16 GPU readback MISMATCH for {prefix}: cpu={:?} gpu={:?}",
                        &bf16_out[..check_len],
                        &readback[..check_len],
                    );
                } else {
                    tracing::info!("BF16 GPU readback verified OK for {prefix}");
                }
            }
        }
    }

    tracing::debug!(
        "Dequanted FP8 blockscaled {prefix}: [{n}, {k}] block=[{block_n}, {block_k}] → BF16",
    );
    Ok(DenseWeight { weight: ptr })
}

/// Convert BF16 bytes (little-endian) to f32.
pub(super) fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

/// Load a dense weight, auto-detecting FP8 scaled vs BF16.
///
/// FP8 tensors may use native `weight_scale_inv`, compressed-tensors
/// `weight_scale [N,1]`, or a scalar `weight_scale`; all are dequantized to
/// BF16. Non-FP8 tensors retain their existing pointer-alias behavior.
pub(crate) fn dense_auto(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    if w.dtype == WeightDtype::FP8E4M3 {
        // Derive prefix: "foo.q_proj.weight" → "foo.q_proj"
        let prefix = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow::anyhow!("FP8 tensor {name} doesn't end with .weight"))?;
        dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)
    } else {
        Ok(DenseWeight { weight: w.ptr })
    }
}

/// Build a QuantizedWeight from Sehyo/compressed-tensors NVFP4 naming convention.
///
/// Sehyo quantization uses: weight_packed, weight_scale, weight_global_scale, input_global_scale
/// (vs standard: weight, weight_scale, weight_scale_2, input_scale).
///
/// **Scale convention difference**: compressed-tensors stores `weight_global_scale`
/// as the reciprocal of Atlas/TRT-LLM's `scale2`. Verified empirically:
///   - nvidia 80B `weight_scale_2` ≈ 7.01e-5 (small)
///   - Sehyo 35B `weight_global_scale` = 29568 → `1/29568` ≈ 3.38e-5 (same order)
///
/// Atlas GEMV dequant: `w = E2M1_val * fp8_scale * scale2` requires the small value.
pub(crate) fn quantized_v2(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let raw_global_scale = scalar_f32(store, &format!("{prefix}.weight_global_scale"), gpu)?;
    // Guard against degenerate / corrupted checkpoints where
    // weight_global_scale is 0 — the unconditional 1/x would store
    // +inf into weight_scale_2 and silently NaN every dequant. Treat
    // it as a hard load error so the operator notices.
    if !raw_global_scale.is_finite() || raw_global_scale.abs() < f32::MIN_POSITIVE {
        anyhow::bail!(
            "{prefix}.weight_global_scale is non-finite or zero ({raw_global_scale}); \
             checkpoint likely corrupted"
        );
    }
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight_packed"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: 1.0 / raw_global_scale,
        input_scale: ptr(store, &format!("{prefix}.input_global_scale"))?,
    })
}
