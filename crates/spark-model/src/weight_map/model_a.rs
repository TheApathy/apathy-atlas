// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// All model weights organized by layer.
pub struct ModelWeights {
    /// Embedding table: [vocab_size, hidden_size] BF16.
    pub embed_tokens: DenseWeight,
    /// Final RMS norm: `[hidden_size]` BF16.
    pub final_norm: DenseWeight,
    /// LM head: [hidden_size, vocab_size] BF16.
    pub lm_head: DenseWeight,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
}

/// Extract a DevicePtr from the weight store by name.
pub(crate) fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store.get(name)?.ptr)
}

/// Extract a scalar f32 from a single-element weight tensor via D2H copy.
pub(crate) fn scalar_f32(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    ensure!(
        w.num_elements() == 1,
        "Expected scalar for {name}, got {} elements",
        w.num_elements()
    );
    let mut buf = [0u8; 4];
    gpu.copy_d2h(w.ptr, &mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Load FP8 KV cache quantization scales from a checkpoint.
///
/// Searches for `{attn_prefix}.k_proj.k_scale` and `{attn_prefix}.v_proj.v_scale`
/// tensors in the weight store. When found, returns the scalar f32 values from the
/// checkpoint (calibrated per-tensor scales for FP8 KV cache quantization).
///
/// When not found, returns (1.0, 1.0) with a debug-level log. This fallback is
/// correct for BF16 KV cache (scales are unused) and acceptable as uncalibrated
/// default for FP8 KV cache (Workstream 4C will add proper calibration).
pub(crate) fn load_kv_scales(
    store: &WeightStore,
    attn_prefix: &str,
    gpu: &dyn GpuBackend,
) -> (f32, f32) {
    let k_key = format!("{attn_prefix}.k_proj.k_scale");
    let v_key = format!("{attn_prefix}.v_proj.v_scale");

    let k_scale = if store.contains(&k_key) {
        match scalar_f32(store, &k_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded k_scale={v:.6} from {k_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {k_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {k_key} in checkpoint, using k_scale=1.0");
        1.0
    };

    let v_scale = if store.contains(&v_key) {
        match scalar_f32(store, &v_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded v_scale={v:.6} from {v_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {v_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {v_key} in checkpoint, using v_scale=1.0");
        1.0
    };

    (k_scale, v_scale)
}

/// Build a QuantizedWeight from the store using the standard NVFP4 naming.
///
/// `weight_scale_2` is a single FP32 scalar — extracted from GPU via D2H copy.
pub(crate) fn quantized(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let input_scale_key = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&input_scale_key) {
        let tensor = store.get(&input_scale_key)?;
        validate_modelopt_input_scale(&input_scale_key, tensor.dtype, &tensor.shape)?;
        tensor.ptr
    } else {
        DevicePtr::NULL
    };
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: scalar_f32(store, &format!("{prefix}.weight_scale_2"), gpu)?,
        input_scale,
    })
}

/// Load a ModelOpt NVFP4 projection while retaining the caller's exact
/// logical geometry as an admission boundary.
///
/// The ordinary pointer-only helper predates consumers whose native ABI reads
/// the complete packed and scale matrices directly.  Such consumers must not
/// infer allocation lengths from configuration after accepting a merely
/// one-element scalar, so this path validates every checkpoint tensor rank,
/// dtype, and extent before exposing its pointers.
pub(crate) fn quantized_modelopt_with_shape(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    ensure!(n > 0, "ModelOpt NVFP4 output dimension must be non-zero");
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "ModelOpt NVFP4 input dimension must be a positive multiple of 16"
    );

    let weight_name = format!("{prefix}.weight");
    let weight = store.get(&weight_name)?;
    ensure!(
        weight.dtype == WeightDtype::UInt8,
        "Expected packed UInt8 ModelOpt weight for {weight_name}, got {:?}",
        weight.dtype
    );
    ensure!(
        weight.shape.as_slice() == [n, k / 2],
        "Expected ModelOpt weight {weight_name} shape [{n}, {}], got {:?}",
        k / 2,
        weight.shape
    );

    let scale_name = format!("{prefix}.weight_scale");
    let weight_scale = store.get(&scale_name)?;
    ensure!(
        weight_scale.dtype == WeightDtype::FP8E4M3,
        "Expected FP8 E4M3 ModelOpt block scales for {scale_name}, got {:?}",
        weight_scale.dtype
    );
    ensure!(
        weight_scale.shape.as_slice() == [n, k / 16],
        "Expected ModelOpt scale {scale_name} shape [{n}, {}], got {:?}",
        k / 16,
        weight_scale.shape
    );

    let scale2_name = format!("{prefix}.weight_scale_2");
    let scale2 = store.get(&scale2_name)?;
    ensure!(
        scale2.dtype == WeightDtype::FP32 && scale2.shape.is_empty(),
        "Expected rank-0 FP32 ModelOpt second-level scale for {scale2_name}, got {:?} {:?}",
        scale2.dtype,
        scale2.shape
    );

    let input_scale_name = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&input_scale_name) {
        let tensor = store.get(&input_scale_name)?;
        validate_modelopt_input_scale(&input_scale_name, tensor.dtype, &tensor.shape)?;
        tensor.ptr
    } else {
        DevicePtr::NULL
    };

    Ok(QuantizedWeight {
        weight: weight.ptr,
        weight_scale: weight_scale.ptr,
        weight_scale_2: scalar_f32(store, &scale2_name, gpu)?,
        input_scale,
    })
}

fn validate_modelopt_input_scale(name: &str, dtype: WeightDtype, shape: &[usize]) -> Result<()> {
    ensure!(
        dtype == WeightDtype::FP32,
        "Expected FP32 ModelOpt input scale for {name}, got {dtype:?}"
    );
    ensure!(
        shape.is_empty(),
        "Expected rank-0 ModelOpt input scale for {name}, got shape {shape:?}"
    );
    Ok(())
}

pub(crate) fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    let w = store.get(name)?;
    Ok(DenseWeight { weight: w.ptr })
}

/// Load a weight, auto-dequanting FP8 block-scaled to BF16 when needed.
///
/// Used for models with mixed-precision layers — Qwen3.6's ViT, for
/// example, keeps the first four blocks in BF16 but stores the rest as
/// FP8. Callers pass a prefix (e.g. `"model.visual.blocks.5.attn.qkv"`)
/// and get back a BF16 GPU buffer either way.
pub(crate) fn dense_auto_fp8_or_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP8E4M3 => dequant_fp8_blockscaled_to_bf16(store, prefix, gpu),
        other => anyhow::bail!(
            "dense_auto_fp8_or_bf16: unsupported dtype {:?} for {prefix}.weight",
            other
        ),
    }
}

/// Load a dense weight, converting FP32 → BF16 if needed. Used for norm weights
/// that may be FP32 in some checkpoints (e.g. Qwen FP8 `linear_attn.norm.weight`).
pub(crate) fn dense_f32_safe(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    if w.dtype == WeightDtype::FP32 {
        tracing::info!(
            "dense_f32_safe: converting {name} from FP32 to BF16 ({:?})",
            w.shape
        );
        let n = w.num_elements();
        let mut f32_buf = vec![0u8; n * 4];
        gpu.copy_d2h(w.ptr, &mut f32_buf)?;
        // Truncate f32 → bf16: take upper 16 bits of each f32
        let bf16_buf: Vec<u8> = f32_buf
            .chunks_exact(4)
            .flat_map(|c| [c[2], c[3]]) // upper 2 bytes = BF16
            .collect();
        let ptr = gpu.alloc(bf16_buf.len())?;
        gpu.copy_h2d(&bf16_buf, ptr)?;
        Ok(DenseWeight { weight: ptr })
    } else {
        Ok(DenseWeight { weight: w.ptr })
    }
}

/// Load a weight and ensure it's FP32 on GPU, regardless of source dtype.
///
/// Used for SSM gate parameters (A_log, dt_bias) where BF16 precision
/// causes exponential error amplification in the recurrent state at
/// long context (8k+ tokens). A 1-ULP BF16 error in the decay gate
/// produces (g_correct/g_error)^8000 ≈ 3000x magnitude divergence.
///
/// - FP32 in safetensors: keep as-is (no conversion)
/// - BF16 in safetensors: convert BF16 → FP32 via zero-extension
pub(crate) fn dense_keep_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::FP32 => {
            // Already FP32 — use directly, no conversion needed
            Ok(DenseWeight { weight: w.ptr })
        }
        WeightDtype::BF16 => {
            // Convert BF16 → FP32 to preserve precision
            tracing::info!(
                "dense_keep_f32: promoting {name} from BF16 to FP32 ({:?})",
                w.shape
            );
            let n = w.num_elements();
            let mut bf16_buf = vec![0u8; n * 2];
            gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
            let f32_buf: Vec<u8> = bf16_buf
                .chunks_exact(2)
                .flat_map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let f32_bits = (bits as u32) << 16;
                    f32_bits.to_le_bytes()
                })
                .collect();
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(DenseWeight { weight: ptr })
        }
        other => {
            bail!("dense_keep_f32: unsupported dtype {:?} for {name}", other);
        }
    }
}

/// Load a BF16 tensor and convert to F32 on-device via CPU roundtrip.
///
/// Used for Nemotron-H SSM parameters (A_log, D, dt_bias, conv1d.bias)
/// which are stored as BF16 in safetensors but consumed as F32 by CUDA kernels.
pub(crate) fn dense_bf16_as_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::BF16,
        "Expected BF16 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut bf16_buf = vec![0u8; n * 2];
    gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
    let f32_buf: Vec<u8> = bf16_buf
        .chunks_exact(2)
        .flat_map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let f32_bits = (bits as u32) << 16;
            f32_bits.to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(f32_buf.len())?;
    gpu.copy_h2d(&f32_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Load an F32 tensor and convert to BF16 on-device via CPU roundtrip.
///
/// Used for Nemotron-H gate weights (F32 in safetensors, consumed as BF16).
pub(crate) fn dense_f32_as_bf16(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut f32_buf = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut f32_buf)?;
    let bf16_buf: Vec<u8> = f32_buf
        .chunks_exact(4)
        .flat_map(|c| {
            let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            f32_to_bf16(val).to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Dequantize FP8 E4M3 + per-tensor scale → BF16, returning a DenseWeight.
///
/// Allocates a new GPU buffer for the result. Use `dequant_fp8_to_bf16_into` to
/// write into a pre-allocated scratch buffer instead (avoids gpu.free on GB10 UVM).
pub(crate) fn dequant_fp8_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let n_bytes = w.num_elements();
    let mut fp8_buf = vec![0u8; n_bytes];
    gpu.copy_d2h(w.ptr, &mut fp8_buf)?;
    let scale = scalar_f32(store, &format!("{prefix}.weight_scale"), gpu)?;
    let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Dequantize FP8 E4M3 → BF16, writing into a pre-allocated destination buffer.
///
/// Avoids gpu.alloc/free for the intermediate BF16 data. The caller provides a
/// reusable scratch buffer that is overwritten each call. Safe to reuse after
/// `quantize_to_nvfp4` returns (it synchronizes internally).
pub(crate) fn dequant_fp8_to_bf16_into(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    dest: DevicePtr,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let n_bytes = w.num_elements();
    let mut fp8_buf = vec![0u8; n_bytes];
    gpu.copy_d2h(w.ptr, &mut fp8_buf)?;
    let scale = scalar_f32(store, &format!("{prefix}.weight_scale"), gpu)?;
    let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
    gpu.copy_h2d(&bf16_buf, dest)?;
    Ok(DenseWeight { weight: dest })
}

#[cfg(test)]
mod modelopt_input_scale_tests {
    use super::validate_modelopt_input_scale;
    use spark_runtime::weights::WeightDtype;

    #[test]
    fn requires_one_fp32_modelopt_input_scale() {
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[]).is_ok()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::BF16, &[])
                .is_err()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[0])
                .is_err()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[1])
                .is_err()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[1, 1])
                .is_err()
        );
    }
}
