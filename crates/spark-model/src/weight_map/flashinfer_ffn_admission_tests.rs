// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::{gpu::DevicePtr, weights::WeightDtype};

use super::*;
use crate::weight_map::modelopt_scale_admission::admit_modelopt_checkpoint_scale;

fn bytes(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|index| (index as u8).wrapping_mul(131).wrapping_add(salt))
        .collect()
}

fn input_scale(
    layer: usize,
    projection: ModeloptScaleProjection,
    ptr: u64,
    bits: u32,
) -> AdmittedModeloptScale {
    admit_modelopt_checkpoint_scale(
        ModeloptScaleSource { layer, projection },
        WeightDtype::FP32,
        &[],
        DevicePtr(ptr),
        bits.to_le_bytes(),
    )
    .unwrap()
}

struct ProjectionFixture {
    packed: Vec<u8>,
    logical_scales: Vec<u8>,
    input_scale: AdmittedModeloptScale,
    weight_scale_2: [u8; 4],
    source: Qwen38FfnSource,
}

impl ProjectionFixture {
    fn new(layer: usize, projection: Qwen38FfnProjection, salt: u8) -> Self {
        let modelopt_projection = match projection {
            Qwen38FfnProjection::Gate => ModeloptScaleProjection::FfnGate,
            Qwen38FfnProjection::Up => ModeloptScaleProjection::FfnUp,
            Qwen38FfnProjection::Down => ModeloptScaleProjection::FfnDown,
        };
        let (n, k) = projection_shape(projection);
        let packed = bytes(n * k / 2, salt);
        let logical_scales = bytes(n * k / NVFP4_GROUP_SIZE, salt.wrapping_add(17));
        Self {
            packed,
            logical_scales,
            input_scale: input_scale(
                layer,
                modelopt_projection,
                salt as u64 + 1,
                0.5f32.to_bits(),
            ),
            weight_scale_2: 0.25f32.to_le_bytes(),
            source: Qwen38FfnSource { layer, projection },
        }
    }

    fn view(&self) -> Qwen38FfnCheckpointProjection<'_> {
        Qwen38FfnCheckpointProjection {
            source: self.source,
            packed_weight: &self.packed,
            logical_weight_scales: &self.logical_scales,
            input_scale: self.input_scale,
            weight_scale_2_le_bytes: self.weight_scale_2,
        }
    }
}

#[test]
fn full_qwen_gate_up_merge_preserves_exact_order_and_scale_bytes() {
    let gate = ProjectionFixture::new(5, Qwen38FfnProjection::Gate, 11);
    let up = ProjectionFixture::new(5, Qwen38FfnProjection::Up, 29);
    let admitted = admit_qwen38_merged_gate_up(5, gate.view(), up.view()).unwrap();

    assert_eq!(admitted.layer(), 5);
    assert_eq!(
        admitted.packed_weight().len(),
        2 * QWEN38_INTERMEDIATE * QWEN38_HIDDEN / 2
    );
    assert_eq!(
        &admitted.packed_weight()[..gate.packed.len()],
        gate.packed.as_slice()
    );
    assert_eq!(
        &admitted.packed_weight()[gate.packed.len()..],
        up.packed.as_slice()
    );

    let merged_logical = deinterleave_nvfp4_scales_128x4(
        admitted.physical_weight_scales(),
        &[2 * QWEN38_INTERMEDIATE, QWEN38_HIDDEN / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )
    .unwrap();
    assert_eq!(
        &merged_logical[..gate.logical_scales.len()],
        gate.logical_scales
    );
    assert_eq!(
        &merged_logical[gate.logical_scales.len()..],
        up.logical_scales
    );

    assert_eq!(admitted.input_scales(), [gate.input_scale, up.input_scale]);
    assert_eq!(admitted.weight_scale_2()[0].source(), gate.source);
    assert_eq!(admitted.weight_scale_2()[1].source(), up.source);
    assert_eq!(admitted.weight_scale_2()[0].value_bits(), 0.25f32.to_bits());
    assert_eq!(admitted.shared_alpha_bits(), 0.125f32.to_bits());
    assert_eq!(admitted.alpha()[0].source(), gate.source);
    assert_eq!(admitted.alpha()[1].source(), up.source);
}

#[test]
fn merged_operand_retained_memory_accounting_is_exact() {
    assert_eq!(qwen38_merged_gate_up_retained_bytes(), 100_270_084);
    assert_eq!(qwen38_merged_gate_up_additional_bytes(), 89_128_956);
    assert_eq!(qwen38_merged_gate_up_additional_bytes() * 64, 5_704_253_184);
}

#[test]
fn separate_projection_admission_is_primary_and_owns_one_exact_scale_cache() {
    let gate = ProjectionFixture::new(4, Qwen38FfnProjection::Gate, 23);
    let admitted = admit_qwen38_ffn_projection(4, Qwen38FfnProjection::Gate, gate.view()).unwrap();
    assert_eq!(admitted.source(), gate.source);
    assert_eq!(
        (admitted.n(), admitted.k()),
        (QWEN38_INTERMEDIATE, QWEN38_HIDDEN)
    );
    assert_eq!(admitted.packed_weight().as_ptr(), gate.packed.as_ptr());
    assert_eq!(admitted.packed_weight(), gate.packed);
    assert_eq!(
        deinterleave_nvfp4_scales_128x4(
            admitted.physical_weight_scales(),
            &[QWEN38_INTERMEDIATE, QWEN38_HIDDEN / NVFP4_GROUP_SIZE],
            NVFP4_GROUP_SIZE,
        )
        .unwrap(),
        gate.logical_scales
    );
    assert_eq!(admitted.input_scale(), gate.input_scale);
    assert_eq!(admitted.weight_scale_2().source(), gate.source);
    assert_eq!(admitted.weight_scale_2().value_bits(), 0.25f32.to_bits());
    assert_eq!(admitted.alpha().source(), gate.source);
    assert_eq!(admitted.alpha().value_bits(), 0.125f32.to_bits());
}

#[test]
fn down_admission_uses_h_by_i_geometry_and_exact_source() {
    let down = ProjectionFixture::new(12, Qwen38FfnProjection::Down, 37);
    let admitted = admit_qwen38_ffn_projection(12, Qwen38FfnProjection::Down, down.view()).unwrap();

    assert_eq!(admitted.source(), down.source);
    assert_eq!(
        (admitted.n(), admitted.k()),
        (QWEN38_HIDDEN, QWEN38_INTERMEDIATE)
    );
    assert_eq!(admitted.packed_weight().as_ptr(), down.packed.as_ptr());
    assert_eq!(admitted.packed_weight(), down.packed);
    assert_eq!(admitted.input_scale(), down.input_scale);
    assert_eq!(admitted.weight_scale_2().source(), down.source);
    assert_eq!(admitted.alpha().source(), down.source);
    let recovered = deinterleave_nvfp4_scales_128x4(
        admitted.physical_weight_scales(),
        &[QWEN38_HIDDEN, QWEN38_INTERMEDIATE / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )
    .unwrap();
    assert_eq!(recovered, down.logical_scales);

    // The same bytes under gate/up geometry produce a different permutation;
    // exact N/K retention prevents a caller from silently reinterpreting it.
    let wrong_physical = interleave_nvfp4_scales_128x4(
        &down.logical_scales,
        &[QWEN38_INTERMEDIATE, QWEN38_HIDDEN / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )
    .unwrap();
    assert_ne!(admitted.physical_weight_scales(), wrong_physical);
}

#[test]
fn down_admission_rejects_gate_identity_and_wrong_scale_extent() {
    let mut down = ProjectionFixture::new(13, Qwen38FfnProjection::Down, 43);
    down.input_scale = input_scale(13, ModeloptScaleProjection::FfnGate, 44, 0.5f32.to_bits());
    assert!(admit_qwen38_ffn_projection(13, Qwen38FfnProjection::Down, down.view()).is_err());
    down.input_scale = input_scale(13, ModeloptScaleProjection::FfnDown, 44, 0.5f32.to_bits());
    down.source.projection = Qwen38FfnProjection::Gate;
    assert!(admit_qwen38_ffn_projection(13, Qwen38FfnProjection::Down, down.view()).is_err());
    down.source.projection = Qwen38FfnProjection::Down;
    down.logical_scales.pop();
    assert!(admit_qwen38_ffn_projection(13, Qwen38FfnProjection::Down, down.view()).is_err());
}

#[test]
fn merge_rejects_scalar_source_shape_and_length_drift() {
    let gate = ProjectionFixture::new(6, Qwen38FfnProjection::Gate, 7);
    let mut up = ProjectionFixture::new(6, Qwen38FfnProjection::Up, 13);

    up.input_scale = input_scale(6, ModeloptScaleProjection::FfnUp, 99, 0.5f32.to_bits() + 1);
    assert!(admit_qwen38_merged_gate_up(6, gate.view(), up.view()).is_err());
    up.input_scale = input_scale(6, ModeloptScaleProjection::FfnUp, 99, 0.5f32.to_bits());
    up.weight_scale_2 = 0.5f32.to_le_bytes();
    assert!(admit_qwen38_merged_gate_up(6, gate.view(), up.view()).is_err());
    up.weight_scale_2 = 0.25f32.to_le_bytes();
    up.source.layer = 7;
    assert!(admit_qwen38_merged_gate_up(6, gate.view(), up.view()).is_err());
    up.source.layer = 6;
    up.input_scale = input_scale(6, ModeloptScaleProjection::FfnGate, 99, 0.5f32.to_bits());
    assert!(admit_qwen38_merged_gate_up(6, gate.view(), up.view()).is_err());
    up.input_scale = input_scale(6, ModeloptScaleProjection::FfnUp, 99, 0.5f32.to_bits());
    up.packed.pop();
    assert!(admit_qwen38_merged_gate_up(6, gate.view(), up.view()).is_err());
    up.packed.push(0);
    up.logical_scales.pop();
    assert!(admit_qwen38_merged_gate_up(6, gate.view(), up.view()).is_err());
}

#[test]
fn merge_rejects_invalid_weight_and_combined_scalars() {
    let gate = ProjectionFixture::new(8, Qwen38FfnProjection::Gate, 3);
    let mut up = ProjectionFixture::new(8, Qwen38FfnProjection::Up, 5);
    for value in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
        up.weight_scale_2 = value.to_le_bytes();
        assert!(admit_qwen38_merged_gate_up(8, gate.view(), up.view()).is_err());
    }
    up.weight_scale_2 = f32::MAX.to_le_bytes();
    up.input_scale = input_scale(8, ModeloptScaleProjection::FfnUp, 6, f32::MAX.to_bits());
    assert!(admit_qwen38_merged_gate_up(8, gate.view(), up.view()).is_err());
}

#[test]
fn exact_qwen_shapes_and_measured_tactics_fail_closed() {
    let expected = [
        (Qwen38FfnOperation::Gate, 2_079, 17_408, 5_120, 4),
        (Qwen38FfnOperation::Up, 2_079, 17_408, 5_120, 4),
        (Qwen38FfnOperation::Gate, 8_192, 17_408, 5_120, 2),
        (Qwen38FfnOperation::Up, 8_192, 17_408, 5_120, 2),
        (Qwen38FfnOperation::MergedGateUp, 2_079, 34_816, 5_120, 4),
        (Qwen38FfnOperation::MergedGateUp, 8_192, 34_816, 5_120, 2),
        (Qwen38FfnOperation::Down, 2_079, 5_120, 17_408, 4),
        (Qwen38FfnOperation::Down, 8_192, 5_120, 17_408, 4),
    ];
    for (operation, m, n, k, tactic) in expected {
        assert_eq!(
            select_qwen38_ffn_launch(operation, m).unwrap(),
            Qwen38FfnLaunchPlan {
                m,
                n,
                k,
                tactic,
                workspace_bytes: 0,
                performance_qualified: true,
            }
        );
    }
    for m in [0, 1, 2_048, 2_080, 8_191, 8_193, usize::MAX] {
        assert!(select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, m).is_err());
        assert!(select_qwen38_ffn_launch(Qwen38FfnOperation::Down, m).is_err());
    }
}

#[test]
fn activation_scales_allocate_explicit_zero_padded_128x4_rows() {
    for operation in [Qwen38FfnOperation::Gate, Qwen38FfnOperation::MergedGateUp] {
        let launch = select_qwen38_ffn_launch(operation, 2_079).unwrap();
        let groups = launch.k / NVFP4_GROUP_SIZE;
        let logical = bytes(launch.m * groups, 41);
        let admitted = allocate_qwen38_activation_scales(launch, &logical).unwrap();
        assert_eq!(admitted.logical_rows, 2_079);
        assert_eq!(admitted.padded_rows, 2_176);
        assert_eq!(admitted.groups, groups);
        assert_eq!(&admitted.logical_padded()[..logical.len()], logical);
        assert!(
            admitted.logical_padded()[logical.len()..]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            deinterleave_nvfp4_scales_128x4(
                admitted.physical(),
                &[admitted.padded_rows, groups],
                NVFP4_GROUP_SIZE,
            )
            .unwrap(),
            admitted.logical_padded()
        );
    }

    let launch = select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, 8_192).unwrap();
    let logical = vec![1; launch.m * launch.k / NVFP4_GROUP_SIZE];
    let admitted = allocate_qwen38_activation_scales(launch, &logical).unwrap();
    assert_eq!(admitted.logical_rows, admitted.padded_rows);
    assert_eq!(admitted.logical_padded(), logical);
}

#[test]
fn activation_scale_allocation_rejects_wrong_bytes_or_forged_plan() {
    let launch = select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, 2_079).unwrap();
    let expected = launch.m * launch.k / NVFP4_GROUP_SIZE;
    assert!(allocate_qwen38_activation_scales(launch, &vec![0; expected - 1]).is_err());
    assert!(allocate_qwen38_activation_scales(launch, &vec![0; expected + 1]).is_err());
    assert!(
        allocate_qwen38_activation_scales(
            Qwen38FfnLaunchPlan {
                tactic: 5,
                ..launch
            },
            &vec![0; expected],
        )
        .is_err()
    );
    assert!(
        allocate_qwen38_activation_scales(
            Qwen38FfnLaunchPlan { n: 1, ..launch },
            &vec![0; expected],
        )
        .is_err()
    );
}
