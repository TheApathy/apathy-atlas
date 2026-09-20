// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed admission and retention of ModelOpt checkpoint input scales.
//!
//! ModelOpt records each projection's input scale as a rank-0 FP32 tensor.  A
//! FlashInfer route must retain both the exact scalar bits and the checkpoint
//! tensor identity: substituting a default, a numerically close value, or a
//! scale from another layer changes the quantization contract.

use anyhow::{Result, ensure};
use spark_runtime::{gpu::DevicePtr, weights::WeightDtype};

/// Projection owning a ModelOpt input-scale tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModeloptScaleProjection {
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    FfnGate,
    FfnUp,
    FfnDown,
    SsmInput,
    SsmOutput,
}

/// Layer-local identity of the checkpoint tensor that supplied a scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModeloptScaleSource {
    pub layer: usize,
    pub projection: ModeloptScaleProjection,
}

/// A validated scale retaining its source allocation and exact FP32 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdmittedModeloptScale {
    source: ModeloptScaleSource,
    device_ptr: DevicePtr,
    value_bits: u32,
}

impl AdmittedModeloptScale {
    pub fn source(self) -> ModeloptScaleSource {
        self.source
    }

    pub fn device_ptr(self) -> DevicePtr {
        self.device_ptr
    }

    pub fn value_bits(self) -> u32 {
        self.value_bits
    }

    pub fn value(self) -> f32 {
        f32::from_bits(self.value_bits)
    }

    /// Exact scalar equality. This intentionally does not compare rounded values.
    pub fn has_same_value(self, other: Self) -> bool {
        self.value_bits == other.value_bits
    }
}

/// Admit one checkpoint scale without a default or fallback path.
pub fn admit_modelopt_checkpoint_scale(
    source: ModeloptScaleSource,
    dtype: WeightDtype,
    shape: &[usize],
    device_ptr: DevicePtr,
    value_le_bytes: [u8; 4],
) -> Result<AdmittedModeloptScale> {
    ensure!(
        dtype == WeightDtype::FP32,
        "ModelOpt input scale must be FP32"
    );
    ensure!(shape.is_empty(), "ModelOpt input scale must have rank 0");
    ensure!(
        !device_ptr.is_null(),
        "ModelOpt input scale must retain a non-NULL checkpoint allocation"
    );

    let value = f32::from_le_bytes(value_le_bytes);
    ensure!(
        value.is_finite() && value > 0.0,
        "ModelOpt input scale must be finite and positive"
    );

    Ok(AdmittedModeloptScale {
        source,
        device_ptr,
        value_bits: u32::from_le_bytes(value_le_bytes),
    })
}

/// An admitted shared Q/K/V scale retaining all three checkpoint sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdmittedSharedQkvScale {
    layer: usize,
    scales: [AdmittedModeloptScale; 3],
}

impl AdmittedSharedQkvScale {
    pub fn layer(self) -> usize {
        self.layer
    }

    pub fn value_bits(self) -> u32 {
        self.scales[0].value_bits()
    }

    pub fn value(self) -> f32 {
        self.scales[0].value()
    }

    pub fn scales(self) -> [AdmittedModeloptScale; 3] {
        self.scales
    }
}

/// Admit scale reuse only for exact, correctly attributed Q/K/V peers.
pub fn admit_shared_qkv_scale(
    requested_layer: usize,
    query: AdmittedModeloptScale,
    key: AdmittedModeloptScale,
    value: AdmittedModeloptScale,
) -> Result<AdmittedSharedQkvScale> {
    let expected = [
        ModeloptScaleProjection::AttentionQuery,
        ModeloptScaleProjection::AttentionKey,
        ModeloptScaleProjection::AttentionValue,
    ];
    let scales = [query, key, value];

    for (scale, projection) in scales.iter().zip(expected) {
        ensure!(
            scale.source.layer == requested_layer,
            "shared Q/K/V scale source must belong to the requested layer"
        );
        ensure!(
            scale.source.projection == projection,
            "shared Q/K/V scale source has the wrong projection identity"
        );
    }
    ensure!(
        query.has_same_value(key) && query.has_same_value(value),
        "shared Q/K/V scales must be bit-identical"
    );

    Ok(AdmittedSharedQkvScale {
        layer: requested_layer,
        scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(layer: usize, projection: ModeloptScaleProjection) -> ModeloptScaleSource {
        ModeloptScaleSource { layer, projection }
    }

    fn admitted(
        layer: usize,
        projection: ModeloptScaleProjection,
        ptr: u64,
        bits: u32,
    ) -> AdmittedModeloptScale {
        admit_modelopt_checkpoint_scale(
            source(layer, projection),
            WeightDtype::FP32,
            &[],
            DevicePtr(ptr),
            bits.to_le_bytes(),
        )
        .unwrap()
    }

    #[test]
    fn admits_and_retains_exact_rank_zero_fp32_source() {
        let scale = admitted(
            7,
            ModeloptScaleProjection::FfnGate,
            0x1234,
            1.25f32.to_bits(),
        );
        assert_eq!(scale.source(), source(7, ModeloptScaleProjection::FfnGate));
        assert_eq!(scale.device_ptr(), DevicePtr(0x1234));
        assert_eq!(scale.value_bits(), 1.25f32.to_bits());
        assert_eq!(scale.value(), 1.25);
    }

    #[test]
    fn rejects_wrong_dtype_rank_or_null_attribution() {
        let src = source(0, ModeloptScaleProjection::FfnUp);
        for dtype in [WeightDtype::BF16, WeightDtype::FP8E4M3, WeightDtype::UInt8] {
            assert!(
                admit_modelopt_checkpoint_scale(
                    src,
                    dtype,
                    &[],
                    DevicePtr(1),
                    1.0f32.to_le_bytes()
                )
                .is_err()
            );
        }
        for shape in [&[1usize][..], &[1usize, 1][..]] {
            assert!(
                admit_modelopt_checkpoint_scale(
                    src,
                    WeightDtype::FP32,
                    shape,
                    DevicePtr(1),
                    1.0f32.to_le_bytes()
                )
                .is_err()
            );
        }
        assert!(
            admit_modelopt_checkpoint_scale(
                src,
                WeightDtype::FP32,
                &[],
                DevicePtr::NULL,
                1.0f32.to_le_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_non_positive_or_non_finite_values() {
        let src = source(0, ModeloptScaleProjection::FfnDown);
        for value in [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(
                admit_modelopt_checkpoint_scale(
                    src,
                    WeightDtype::FP32,
                    &[],
                    DevicePtr(1),
                    value.to_le_bytes()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn identity_and_value_equality_are_deliberately_distinct() {
        let a = admitted(3, ModeloptScaleProjection::FfnGate, 11, 1.0f32.to_bits());
        let same = admitted(3, ModeloptScaleProjection::FfnGate, 11, 1.0f32.to_bits());
        let other_ptr = admitted(3, ModeloptScaleProjection::FfnGate, 12, 1.0f32.to_bits());
        let other_source = admitted(4, ModeloptScaleProjection::FfnGate, 11, 1.0f32.to_bits());
        let next_value = admitted(
            3,
            ModeloptScaleProjection::FfnGate,
            11,
            1.0f32.to_bits() + 1,
        );

        assert_eq!(a, same);
        assert_ne!(a, other_ptr);
        assert_ne!(a, other_source);
        assert!(a.has_same_value(other_ptr));
        assert!(!a.has_same_value(next_value));
    }

    #[test]
    fn admits_bit_identical_layer_local_qkv_and_retains_all_sources() {
        let bits = 0.125f32.to_bits();
        let q = admitted(9, ModeloptScaleProjection::AttentionQuery, 21, bits);
        let k = admitted(9, ModeloptScaleProjection::AttentionKey, 22, bits);
        let v = admitted(9, ModeloptScaleProjection::AttentionValue, 23, bits);
        let shared = admit_shared_qkv_scale(9, q, k, v).unwrap();

        assert_eq!(shared.layer(), 9);
        assert_eq!(shared.value_bits(), bits);
        assert_eq!(shared.value(), 0.125);
        assert_eq!(shared.scales(), [q, k, v]);
    }

    #[test]
    fn rejects_qkv_value_layer_and_projection_misattribution() {
        let bits = 1.0f32.to_bits();
        let q = admitted(2, ModeloptScaleProjection::AttentionQuery, 31, bits);
        let k = admitted(2, ModeloptScaleProjection::AttentionKey, 32, bits);
        let v = admitted(2, ModeloptScaleProjection::AttentionValue, 33, bits);
        let changed = admitted(2, ModeloptScaleProjection::AttentionValue, 33, bits + 1);
        let other_layer = admitted(3, ModeloptScaleProjection::AttentionValue, 33, bits);
        let wrong_projection = admitted(2, ModeloptScaleProjection::AttentionKey, 33, bits);

        assert!(admit_shared_qkv_scale(2, q, k, changed).is_err());
        assert!(admit_shared_qkv_scale(2, q, k, other_layer).is_err());
        assert!(admit_shared_qkv_scale(2, q, k, wrong_projection).is_err());
        assert!(admit_shared_qkv_scale(3, q, k, v).is_err());
    }
}
