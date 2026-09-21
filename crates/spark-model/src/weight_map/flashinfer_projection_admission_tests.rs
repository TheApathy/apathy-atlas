// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::weights::WeightDtype;

fn scale(
    layer: usize,
    projection: Qwen38Projection,
    ptr: u64,
    value: f32,
) -> AdmittedModeloptScale {
    let modelopt_projection = match projection {
        Qwen38Projection::AttentionQueryGate => ModeloptScaleProjection::AttentionQuery,
        Qwen38Projection::AttentionKey => ModeloptScaleProjection::AttentionKey,
        Qwen38Projection::AttentionValue => ModeloptScaleProjection::AttentionValue,
        Qwen38Projection::AttentionOutput => ModeloptScaleProjection::AttentionOutput,
        Qwen38Projection::SsmQkvz => ModeloptScaleProjection::SsmInput,
        Qwen38Projection::SsmOutput => ModeloptScaleProjection::SsmOutput,
    };
    super::super::modelopt_scale_admission::admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        DevicePtr(ptr),
        value.to_le_bytes(),
    )
    .unwrap()
}

fn material(projection: Qwen38Projection) -> (Vec<u8>, usize) {
    let (n, k) = projection_shape(projection);
    let len = n * (k / NVFP4_GROUP_SIZE);
    (
        (0..len)
            .map(|i| (i.wrapping_mul(131) & 0xff) as u8)
            .collect(),
        n * k / 2,
    )
}

fn admit(layer: usize, projection: Qwen38Projection) -> AdmittedQwen38Projection {
    let (logical, packed_weight_bytes) = material(projection);
    admit_qwen38_projection(
        layer,
        projection,
        Qwen38CheckpointProjection {
            source: Qwen38ProjectionSource { layer, projection },
            packed_weight: DevicePtr(0x1000),
            packed_weight_bytes,
            logical_weight_scales: &logical,
            input_scale: scale(layer, projection, 0x2000, 0.5),
            weight_scale_2_le_bytes: 0.25f32.to_le_bytes(),
        },
    )
    .unwrap()
}

#[test]
fn admits_every_exact_projection_shape_and_retains_checkpoint_weight() {
    let cases = [
        (
            Qwen38Projection::AttentionQueryGate,
            QWEN38_QG,
            QWEN38_HIDDEN,
        ),
        (Qwen38Projection::AttentionKey, QWEN38_KV, QWEN38_HIDDEN),
        (Qwen38Projection::AttentionValue, QWEN38_KV, QWEN38_HIDDEN),
        (
            Qwen38Projection::AttentionOutput,
            QWEN38_HIDDEN,
            QWEN38_ATTN_VALUE,
        ),
        (Qwen38Projection::SsmQkvz, QWEN38_SSM_QKVZ, QWEN38_HIDDEN),
        (Qwen38Projection::SsmOutput, QWEN38_HIDDEN, QWEN38_SSM_VALUE),
    ];
    for (projection, n, k) in cases {
        let admitted = admit(7, projection);
        assert_eq!((admitted.n(), admitted.k()), (n, k));
        assert_eq!(admitted.packed_weight(), DevicePtr(0x1000));
        assert_eq!(admitted.packed_weight_bytes(), n * k / 2);
        assert_eq!(admitted.physical_weight_scales().len(), n * k / 16);
        assert_eq!(admitted.alpha().value_bits(), 0.125f32.to_bits());
    }
}

#[test]
fn ssm_geometry_matches_qwen38_config_and_checkpoint_components() {
    // Qwen3.8 text_config: 16 key heads * 128 for Q and K, 48 value
    // heads * 128 for V and Z. The checkpoint stores QKV and Z separately.
    let key = 16 * 128;
    let value = 48 * 128;
    assert_eq!(QWEN38_SSM_QKV, key * 2 + value);
    assert_eq!(QWEN38_SSM_Z, value);
    assert_eq!(QWEN38_SSM_QKVZ, 16_384);
    assert_eq!(QWEN38_SSM_VALUE, value);

    let qkvz = admit(0, Qwen38Projection::SsmQkvz);
    assert_eq!(qkvz.packed_weight_bytes(), 16_384 * 5_120 / 2);
    let output = admit(0, Qwen38Projection::SsmOutput);
    assert_eq!(output.packed_weight_bytes(), 5_120 * 6_144 / 2);
    assert_eq!(output.physical_weight_scales().len(), 5_120 * 6_144 / 16);
}

#[test]
fn rejects_source_scale_pointer_and_extent_forgery() {
    let projection = Qwen38Projection::AttentionKey;
    let (logical, packed_weight_bytes) = material(projection);
    let make = |source, packed_weight, bytes, input_scale| Qwen38CheckpointProjection {
        source,
        packed_weight,
        packed_weight_bytes: bytes,
        logical_weight_scales: &logical,
        input_scale,
        weight_scale_2_le_bytes: 0.25f32.to_le_bytes(),
    };
    let source = Qwen38ProjectionSource {
        layer: 2,
        projection,
    };
    assert!(
        admit_qwen38_projection(
            1,
            projection,
            make(
                source,
                DevicePtr(0x1000),
                packed_weight_bytes,
                scale(2, projection, 0x2000, 1.0)
            )
        )
        .is_err()
    );
    assert!(
        admit_qwen38_projection(
            2,
            projection,
            make(
                source,
                DevicePtr::NULL,
                packed_weight_bytes,
                scale(2, projection, 0x2000, 1.0)
            )
        )
        .is_err()
    );
    assert!(
        admit_qwen38_projection(
            2,
            projection,
            make(
                source,
                DevicePtr(0x1001),
                packed_weight_bytes,
                scale(2, projection, 0x2000, 1.0)
            )
        )
        .is_err()
    );
    assert!(
        admit_qwen38_projection(
            2,
            projection,
            make(
                source,
                DevicePtr(0x1000),
                packed_weight_bytes - 1,
                scale(2, projection, 0x2000, 1.0)
            )
        )
        .is_err()
    );
    assert!(
        admit_qwen38_projection(
            2,
            projection,
            make(
                source,
                DevicePtr(0x1000),
                packed_weight_bytes,
                scale(3, projection, 0x2000, 1.0)
            )
        )
        .is_err()
    );
}

#[test]
fn rejects_scale_length_nonfinite_scalar_and_alpha_overflow() {
    let projection = Qwen38Projection::SsmOutput;
    let (mut logical, packed_weight_bytes) = material(projection);
    logical.pop();
    let source = Qwen38ProjectionSource {
        layer: 1,
        projection,
    };
    assert!(
        admit_qwen38_projection(
            1,
            projection,
            Qwen38CheckpointProjection {
                source,
                packed_weight: DevicePtr(0x1000),
                packed_weight_bytes,
                logical_weight_scales: &logical,
                input_scale: scale(1, projection, 0x2000, 1.0),
                weight_scale_2_le_bytes: 1.0f32.to_le_bytes(),
            }
        )
        .is_err()
    );
    let (logical, _) = material(projection);
    for value in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(
            admit_qwen38_projection(
                1,
                projection,
                Qwen38CheckpointProjection {
                    source,
                    packed_weight: DevicePtr(0x1000),
                    packed_weight_bytes,
                    logical_weight_scales: &logical,
                    input_scale: scale(1, projection, 0x2000, 1.0),
                    weight_scale_2_le_bytes: value.to_le_bytes(),
                }
            )
            .is_err()
        );
    }
    assert!(
        admit_qwen38_projection(
            1,
            projection,
            Qwen38CheckpointProjection {
                source,
                packed_weight: DevicePtr(0x1000),
                packed_weight_bytes,
                logical_weight_scales: &logical,
                input_scale: scale(1, projection, 0x2000, f32::MAX),
                weight_scale_2_le_bytes: 2.0f32.to_le_bytes(),
            }
        )
        .is_err()
    );
}

#[test]
fn cache_requires_exact_uploaded_alpha_and_non_aliasing_aligned_addresses() {
    let admitted = admit(4, Qwen38Projection::AttentionOutput);
    let bits = admitted.alpha().value_bits();
    let cache =
        finalize_qwen38_projection_cache(&admitted, DevicePtr(0x3000), DevicePtr(0x4000), bits)
            .unwrap();
    assert_eq!(cache.source(), admitted.source());
    assert_eq!(cache.shape(), (admitted.n(), admitted.k()));
    assert_eq!(cache.packed_weight(), DevicePtr(0x1000));
    assert_eq!(cache.input_scale(), DevicePtr(0x2000));
    assert_eq!(cache.alpha_bits(), bits);
    assert!(
        finalize_qwen38_projection_cache(&admitted, DevicePtr::NULL, DevicePtr(0x4000), bits)
            .is_err()
    );
    assert!(
        finalize_qwen38_projection_cache(&admitted, DevicePtr(0x3001), DevicePtr(0x4000), bits)
            .is_err()
    );
    assert!(
        finalize_qwen38_projection_cache(&admitted, DevicePtr(0x1000), DevicePtr(0x4000), bits)
            .is_err()
    );
    assert!(
        finalize_qwen38_projection_cache(&admitted, DevicePtr(0x3000), DevicePtr(0x4001), bits)
            .is_err()
    );
    assert!(
        finalize_qwen38_projection_cache(&admitted, DevicePtr(0x3000), DevicePtr(0x4000), bits ^ 1)
            .is_err()
    );
}

#[test]
fn qkv_activation_reuse_requires_exact_layer_projection_and_value_bits() {
    fn cache(
        layer: usize,
        projection: Qwen38Projection,
        input_ptr: u64,
        input_value: f32,
    ) -> Qwen38ProjectionOperandCache {
        let (logical, packed_weight_bytes) = material(projection);
        let admitted = admit_qwen38_projection(
            layer,
            projection,
            Qwen38CheckpointProjection {
                source: Qwen38ProjectionSource { layer, projection },
                packed_weight: DevicePtr(0x1000 + projection as u64 * 0x100),
                packed_weight_bytes,
                logical_weight_scales: &logical,
                input_scale: scale(layer, projection, input_ptr, input_value),
                weight_scale_2_le_bytes: 0.25f32.to_le_bytes(),
            },
        )
        .unwrap();
        finalize_qwen38_projection_cache(
            &admitted,
            DevicePtr(0x3000 + projection as u64 * 0x100),
            DevicePtr(0x4000 + projection as u64 * 4),
            admitted.alpha().value_bits(),
        )
        .unwrap()
    }
    let q = cache(3, Qwen38Projection::AttentionQueryGate, 0x2000, 0.5);
    let k = cache(3, Qwen38Projection::AttentionKey, 0x2100, 0.5);
    let v = cache(3, Qwen38Projection::AttentionValue, 0x2200, 0.5);
    let group = admit_qwen38_attention_qkv_cache_group(3, q, k, v).unwrap();
    assert_eq!(group.layer(), 3);
    assert_eq!(group.caches(), [q, k, v]);
    assert!(admit_qwen38_attention_qkv_cache_group(4, q, k, v).is_err());
    assert!(admit_qwen38_attention_qkv_cache_group(3, q, v, k).is_err());
    assert!(
        admit_qwen38_attention_qkv_cache_group(
            3,
            q,
            k,
            cache(3, Qwen38Projection::AttentionValue, 0x2300, 0.50000006)
        )
        .is_err()
    );
}

#[test]
fn merged_qgkv_requires_exact_scalars_and_preserves_q_k_v_order() {
    fn operand(
        layer: usize,
        projection: Qwen38Projection,
        weight_scale_2: f32,
    ) -> (AdmittedQwen38Projection, Vec<u8>) {
        let (logical, packed_weight_bytes) = material(projection);
        let packed = vec![projection as u8 + 1; packed_weight_bytes];
        let admitted = admit_qwen38_projection(
            layer,
            projection,
            Qwen38CheckpointProjection {
                source: Qwen38ProjectionSource { layer, projection },
                packed_weight: DevicePtr(0x1000 + projection as u64 * 0x100),
                packed_weight_bytes,
                logical_weight_scales: &logical,
                input_scale: scale(layer, projection, 0x2000, 0.5),
                weight_scale_2_le_bytes: weight_scale_2.to_le_bytes(),
            },
        )
        .unwrap();
        (admitted, packed)
    }

    let (q, q_bytes) = operand(9, Qwen38Projection::AttentionQueryGate, 0.25);
    let (k, k_bytes) = operand(9, Qwen38Projection::AttentionKey, 0.25);
    let (v, v_bytes) = operand(9, Qwen38Projection::AttentionValue, 0.25);
    let merged = merge_qwen38_attention_qkv(9, &q, &q_bytes, &k, &k_bytes, &v, &v_bytes).unwrap();
    assert_eq!(merged.layer(), 9);
    assert_eq!(
        merged.packed_weight().len(),
        QWEN38_ATTN_QGKV * QWEN38_HIDDEN / 2
    );
    assert_eq!(
        merged.physical_weight_scales().len(),
        QWEN38_ATTN_QGKV * QWEN38_HIDDEN / NVFP4_GROUP_SIZE
    );
    assert!(
        merged.packed_weight()[..q_bytes.len()]
            .iter()
            .all(|&byte| byte == 1)
    );
    assert!(
        merged.packed_weight()[q_bytes.len()..q_bytes.len() + k_bytes.len()]
            .iter()
            .all(|&byte| byte == 2)
    );
    assert!(
        merged.packed_weight()[q_bytes.len() + k_bytes.len()..]
            .iter()
            .all(|&byte| byte == 3)
    );

    let cache = finalize_qwen38_attention_qkv_merged_cache(
        &merged,
        DevicePtr(0x5000),
        DevicePtr(0x6000),
        DevicePtr(0x7000),
        merged.alpha_bits(),
    )
    .unwrap();
    assert_eq!(cache.layer(), 9);
    assert_eq!(cache.shape(), (QWEN38_ATTN_QGKV, QWEN38_HIDDEN));
    assert_eq!(cache.packed_weight(), DevicePtr(0x5000));
    assert_eq!(cache.input_scale_bits(), 0.5f32.to_bits());
    assert_eq!(cache.alpha_bits(), 0.125f32.to_bits());
    assert!(
        finalize_qwen38_attention_qkv_merged_cache(
            &merged,
            DevicePtr(0x5000),
            DevicePtr(0x6000),
            DevicePtr(0x7000),
            merged.alpha_bits() ^ 1,
        )
        .is_err()
    );

    let (different_alpha, different_alpha_bytes) =
        operand(9, Qwen38Projection::AttentionValue, 0.5);
    assert!(
        merge_qwen38_attention_qkv(
            9,
            &q,
            &q_bytes,
            &k,
            &k_bytes,
            &different_alpha,
            &different_alpha_bytes,
        )
        .is_err()
    );
}

#[test]
fn frozen_attention_real_checkpoint_receipts_select_only_exact_m() {
    let qgkv = select_qwen38_attention_projection_launch(
        Qwen38AttentionProjectionFamily::MergedQkv,
        2_079,
    )
    .unwrap();
    assert_eq!((qgkv.n, qgkv.k, qgkv.tactic), (14_336, 5_120, 4));
    assert_eq!(qgkv.workspace_bytes, 0);
    assert!(qgkv.real_checkpoint_qualified);

    let output =
        select_qwen38_attention_projection_launch(Qwen38AttentionProjectionFamily::Output, 2_079)
            .unwrap();
    assert_eq!((output.n, output.k, output.tactic), (5_120, 6_144, 0));
    assert_eq!(output.workspace_bytes, 0);
    assert!(output.real_checkpoint_qualified);

    for m in [0, 2_048, 2_080, 8_191, 8_192, 8_193] {
        assert!(
            select_qwen38_attention_projection_launch(
                Qwen38AttentionProjectionFamily::MergedQkv,
                m,
            )
            .is_err()
        );
        assert!(
            select_qwen38_attention_projection_launch(Qwen38AttentionProjectionFamily::Output, m,)
                .is_err()
        );
    }
}

#[test]
fn synthetic_projection_receipts_never_select_for_production() {
    for projection in [
        Qwen38Projection::AttentionOutput,
        Qwen38Projection::SsmQkvz,
        Qwen38Projection::SsmOutput,
    ] {
        for m in [2_079, 8_192] {
            let plan = qwen38_projection_launch_candidate(projection, m).unwrap();
            assert_eq!(plan.workspace_bytes, 0);
            assert!(!plan.real_checkpoint_qualified);
            assert!(select_qwen38_projection_launch(projection, m).is_err());
        }
    }
    for projection in [
        Qwen38Projection::AttentionQueryGate,
        Qwen38Projection::AttentionKey,
        Qwen38Projection::AttentionValue,
    ] {
        for m in [2_079, 8_192] {
            assert!(qwen38_projection_launch_candidate(projection, m).is_err());
        }
    }
    assert!(qwen38_projection_launch_candidate(Qwen38Projection::SsmQkvz, 2_048).is_err());
}
