// SPDX-License-Identifier: AGPL-3.0-only
// Standalone std-only RED gate: root can compile this without the broken P1 lib.
#[path = "../src/host_angles.rs"]
mod host_angles;

#[test]
fn only_retained_historical_grids_are_admitted_with_fixed_geometry() {
    for (h, w) in [(3, 3), (4, 5), (54, 54)] {
        let values = host_angles::legacy_angles(h, w).unwrap();
        assert_eq!(values.len(), h * w * 64);
        assert!(values.len() <= 3456 * 64);
        assert!(values.iter().all(|v| v.is_finite()));
    }
    for (h, w) in [
        (0, 3),
        (3, 0),
        (1, 1),
        (3, 4),
        (54, 55),
        (usize::MAX, 3),
        (3, usize::MAX),
        (usize::MAX, usize::MAX),
    ] {
        assert!(host_angles::legacy_angles(h, w).is_none());
    }
}

#[test]
fn historical_layout_is_patch_major_height_then_width_cos_then_sin() {
    let values = host_angles::legacy_angles(3, 3).unwrap();
    assert!(values[..32].iter().all(|v| v.to_bits() == 1.0f32.to_bits()));
    assert!(values[32..64].iter().all(|v| v.to_bits() == 0));
    // Patch1 is (h0,w1), patch3 is (h1,w0). The corresponding axes exchange.
    assert_eq!(&values[64 + 16..64 + 32], &values[3 * 64..3 * 64 + 16]);
    assert_eq!(&values[64 + 48..64 + 64], &values[3 * 64 + 32..3 * 64 + 48]);
    assert!(
        values[64..64 + 16]
            .iter()
            .all(|v| v.to_bits() == 1.0f32.to_bits())
    );
    assert!(values[64 + 32..64 + 48].iter().all(|v| v.to_bits() == 0));
}

#[test]
fn historical_fp32_libm_bits_are_not_replaced_with_cuda_angle_bits() {
    // Retained grid-3x3-host-angles.f32 SHA9681be8c68566e2c46d174ec4fa17c38f3f70a8d770330a1e4996145e9273fbf.
    // These cells were read from the retained buffer, not regenerated as a teacher.
    let values = host_angles::legacy_angles(3, 3).unwrap();
    for (index, bits) in [
        (112, 0x3f576aa4),
        (144, 0xbed51133),
        (146, 0x3f4e7bec),
        (147, 0x3f6ffaa5),
        (176, 0x3f68c7b7),
    ] {
        assert_eq!(values[index].to_bits(), bits, "historical cell {index}");
    }
    // The proven CUDA candidate differs at these FP32 cells, despite matching
    // some BF16 Q/K outputs. This helper must remain the historical CONTROL.
    assert_ne!(values[112].to_bits(), 0x3f576aa5);
    assert_ne!(values[144].to_bits(), 0xbed51132);
}

#[test]
fn bench_owns_legacy_control_and_receipts_without_restoring_production_host_path() {
    let lib = include_str!("../src/lib.rs");
    let rope = include_str!("../src/rope.rs");
    let pins = include_str!("../src/pins.rs");
    assert!(lib.contains("mod host_angles;"));
    assert!(!lib.contains("native_geometry"));
    assert!(!rope.contains("native_geometry"));
    assert_eq!(
        rope.matches("crate::host_angles::legacy_angles(").count(),
        2
    );
    assert!(pins.contains("\"src/host_angles.rs\""));
    assert!(pins.contains("\"tests/host_angles.rs\""));
    let geometry =
        include_str!("../../../crates/spark-model/src/layers/deepseek_vision/geometry.rs");
    let compact: String = geometry.split_whitespace().collect();
    assert!(compact.contains("#[cfg(test)]pub(super)fnrope_angles("));
    let forward = include_str!("../../../crates/spark-model/src/layers/deepseek_vision/forward.rs");
    assert!(!forward.contains("rope_angles("));
    assert!(forward.contains("self.kernels.angles"));
}
