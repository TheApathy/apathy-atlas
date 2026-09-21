// SPDX-License-Identifier: AGPL-3.0-only

//! Strict admission for actual DeepSeek Vision routing tensors.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

pub(super) fn load_bias(
    store: &WeightStore,
    prefix: &str,
    suffix: &str,
    experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let keys = [
        format!("{prefix}.ffn.gate.{suffix}"),
        format!("{prefix}.mlp.gate.{suffix}"),
    ];
    let tensor = keys
        .iter()
        .find_map(|key| store.get(key).ok())
        .with_context(|| format!("DeepSeek Vision {prefix}: missing required {suffix}"))?;
    ensure!(
        tensor.shape == [experts],
        "DeepSeek Vision {prefix}.{suffix}: expected [{experts}], got {:?}",
        tensor.shape
    );
    ensure!(
        matches!(tensor.dtype, WeightDtype::FP32 | WeightDtype::BF16),
        "DeepSeek Vision {prefix}.{suffix}: requires F32/BF16, got {:?}",
        tensor.dtype
    );
    let mut bytes = vec![0; experts * tensor.dtype.byte_size()];
    gpu.copy_d2h(tensor.ptr, &mut bytes)?;
    let widened = decode_bias(&bytes, tensor.dtype)?;
    if tensor.dtype == WeightDtype::FP32 {
        return Ok(tensor.ptr);
    }
    let ptr = gpu.alloc(widened.len())?;
    gpu.copy_h2d(&widened, ptr)?;
    Ok(ptr)
}

fn decode_bias(bytes: &[u8], dtype: WeightDtype) -> Result<Vec<u8>> {
    ensure!(
        matches!(dtype, WeightDtype::FP32 | WeightDtype::BF16),
        "visual bias dtype is not F32/BF16"
    );
    let width = dtype.byte_size();
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(width),
        "invalid visual bias byte extent"
    );
    let mut output = Vec::with_capacity(bytes.len() / width * 4);
    for element in bytes.chunks_exact(width) {
        let bits = if width == 2 {
            u32::from(u16::from_le_bytes([element[0], element[1]])) << 16
        } else {
            u32::from_le_bytes(element.try_into().expect("four-byte chunk"))
        };
        ensure!(
            f32::from_bits(bits).is_finite(),
            "nonfinite DeepSeek visual routing bias"
        );
        output.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(output)
}

pub(super) fn validate_hash_table(
    store: &WeightStore,
    prefix: &str,
    vocab: usize,
    top_k: usize,
    experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let table = store.get(&format!("{prefix}.ffn.gate.tid2eid"))?;
    ensure!(
        table.shape == [vocab, top_k],
        "DeepSeek Vision {prefix}: hash table shape mismatch"
    );
    // Pinned Vision K2 uses I64, as does the serving hash-kernel ABI. Do not
    // reinterpret a future official/native I32 checkpoint as this ABI.
    ensure!(
        table.dtype == WeightDtype::Int64,
        "DeepSeek Vision {prefix}: hash table must be I64"
    );
    let bytes = vocab
        .checked_mul(top_k)
        .and_then(|n| n.checked_mul(8))
        .context("DeepSeek Vision hash table byte overflow")?;
    let mut source = vec![0; bytes];
    gpu.copy_d2h(table.ptr, &mut source)?;
    validate_hash_bytes(&source, experts)
}

fn validate_hash_bytes(bytes: &[u8], experts: usize) -> Result<()> {
    ensure!(
        experts > 0 && !bytes.is_empty() && bytes.len().is_multiple_of(8),
        "invalid hash table byte extent"
    );
    for row in bytes.chunks_exact(8) {
        let expert = i64::from_le_bytes(row.try_into().expect("eight-byte chunk"));
        ensure!(
            expert >= 0 && (expert as u64) < experts as u64,
            "DeepSeek Vision hash expert {expert} out of range 0..{experts}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biases_preserve_f32_and_widen_bf16_exactly() {
        let values = [0.0f32, -1.5, 4.0];
        let f32_bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bf16_bytes: Vec<_> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(
            decode_bias(&f32_bytes, WeightDtype::FP32).unwrap(),
            f32_bytes
        );
        assert_eq!(
            decode_bias(&bf16_bytes, WeightDtype::BF16).unwrap(),
            f32_bytes
        );
    }

    #[test]
    fn malformed_biases_fail_closed() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(decode_bias(&value.to_le_bytes(), WeightDtype::FP32).is_err());
        }
        assert!(decode_bias(&[0; 3], WeightDtype::FP32).is_err());
        assert!(decode_bias(&[], WeightDtype::BF16).is_err());
        assert!(decode_bias(&[0; 8], WeightDtype::Int64).is_err());
    }

    #[test]
    fn hash_experts_never_clamp_invalid_ids_to_zero() {
        for expert in [-1i64, 256, i64::MAX] {
            assert!(validate_hash_bytes(&expert.to_le_bytes(), 256).is_err());
        }
        assert!(validate_hash_bytes(&255i64.to_le_bytes(), 256).is_ok());
        assert!(validate_hash_bytes(&[0; 7], 256).is_err());
    }
}
