// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only tests of the production F32-highway/BF16-history address planner.

#[path = "../src/model/dspark_capture.rs"]
mod capture;

use capture::DsparkCaptureLayout;

#[test]
fn long_prefill_keeps_the_newest_rows_using_f32_source_stride() {
    let plan = DsparkCaptureLayout::new(256, true)
        .addresses(0, 2048, 2, 3, 2048, 4096, 4)
        .unwrap();
    assert_eq!(plan.len(), 1);
    assert_eq!(plan[0].source_offset, 1792 * 4 * 4096 * 4);
    assert_eq!(plan[0].source_bytes, 256 * 4 * 4096 * 4);
    assert_eq!(plan[0].destination_offset, 2 * 256 * 4096 * 2);
    assert_eq!(plan[0].destination_bytes, 256 * 4096 * 2);
    assert_eq!(plan[0].rows, 256);
}

#[test]
fn wrapped_span_uses_two_distinct_f32_source_rows() {
    let plan = DsparkCaptureLayout::new(8, true)
        .addresses(6, 4, 1, 3, 4, 2, 2)
        .unwrap();
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0].source_offset, 0);
    assert_eq!(plan[1].source_offset, 2 * 2 * 2 * 4);
    assert_eq!(plan[0].destination_offset, (8 + 6) * 2 * 2);
    assert_eq!(plan[1].destination_offset, 8 * 2 * 2);
    // Distinct row/lane tags expose the former half-byte-stride addressing.
    let source: Vec<u8> = (0..4)
        .flat_map(|row| (0..4).flat_map(move |lane| ((row * 100 + lane) as f32).to_le_bytes()))
        .collect();
    let read = |offset| f32::from_le_bytes(source[offset..offset + 4].try_into().unwrap());
    assert_eq!(read(plan[1].source_offset), 200.0);
    assert_eq!(read(plan[1].source_offset / 2), 100.0);
}

#[test]
fn linear_and_zero_source_offsets_preserve_existing_layout() {
    let plan = DsparkCaptureLayout::new(8, false)
        .addresses(6, 4, 0, 1, 4, 4096, 4)
        .unwrap();
    assert_eq!(plan.len(), 1);
    assert_eq!(plan[0].source_offset, 0);
    assert_eq!(plan[0].destination_offset, 6 * 4096 * 2);
    assert_eq!(plan[0].rows, 2);
}

#[test]
fn invalid_extents_fail_before_a_capture_can_be_launched() {
    let ring = DsparkCaptureLayout::new(8, true);
    assert!(ring.addresses(0, 9, 0, 1, 8, 4096, 4).is_err());
    assert!(ring.addresses(0, 4, 1, 1, 4, 4096, 4).is_err());
    assert!(ring.addresses(0, 4, 0, 0, 4, 4096, 4).is_err());
    assert!(ring.addresses(0, 4, 0, 1, 4, 0, 4).is_err());
    assert!(ring.addresses(0, 4, 0, 1, 4, 4096, 0).is_err());
    assert!(ring.addresses(usize::MAX, 4, 0, 1, 4, 4096, 4).is_err());
    assert!(ring.addresses(0, 4, 0, usize::MAX, 4, 4096, 4).is_err());
    assert!(ring.addresses(0, 4, 0, 1, usize::MAX, 4096, 4).is_err());
    assert!(ring.addresses(0, 4, 0, 1, 4, usize::MAX, 4).is_err());
}

#[test]
fn production_eager_capture_consumes_checked_offsets_before_launch() {
    let source = include_str!("../src/model/impl_b3.rs");
    let capture = source
        .split("pub(super) fn try_dspark_capture(")
        .nth(1)
        .unwrap()
        .split("pub(super) fn dspark_capture_commit(")
        .next()
        .unwrap();
    assert!(
        capture.find(".addresses(").unwrap()
            < capture.find("crate::layers::ops::hc_mean(").unwrap()
    );
    assert!(capture.contains(".offset(address.source_offset)"));
    assert!(capture.contains(".offset(address.destination_offset)"));
    assert!(!capture.contains("span.src_row * self.config.hc_mult * h * 2"));
}
