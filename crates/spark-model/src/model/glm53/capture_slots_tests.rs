// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const BASE: u64 = 0x7_0000_0000;

fn slots() -> Glm53CaptureSlots {
    Glm53CaptureSlots::bind(DevicePtr(BASE), GLM53_DFLASH_CAPTURE_BYTES).unwrap()
}

#[test]
fn slots_tile_the_reserved_region_at_eight_rows_each() {
    let s = slots();
    let mut spans = Vec::new();
    for slot in 0..5 {
        let buffer = s.slot_rows(slot, GLM53_CAPTURE_MAX_ROWS).unwrap();
        assert_eq!(
            buffer.bytes as u64,
            GLM53_CAPTURE_SLOT_BYTES * u64::from(GLM53_CAPTURE_MAX_ROWS)
        );
        spans.push((buffer.ptr.0, buffer.ptr.0 + buffer.bytes as u64));
    }
    assert_eq!(spans[0].0, BASE);
    assert_eq!(spans[4].1, BASE + GLM53_DFLASH_CAPTURE_BYTES);
    for pair in spans.windows(2) {
        assert_eq!(pair[0].1, pair[1].0, "capture slots must tile without gaps");
    }
    assert!(s.slot(5).is_err());
}

/// Each tap layer owns exactly one slot, in schedule order.
#[test]
fn every_tap_layer_maps_to_its_own_slot_and_non_taps_are_refused() {
    for (index, layer) in GLM53_CAPTURE_LAYERS.iter().enumerate() {
        assert_eq!(
            Glm53CaptureSlots::slot_for_layer(*layer).unwrap(),
            index as u32
        );
    }
    // Layers adjacent to taps must not be capturable.
    for layer in [0, 3, 5, 12, 14, 42, 44] {
        assert!(
            Glm53CaptureSlots::slot_for_layer(layer).is_err(),
            "layer {layer} is not a tap"
        );
    }
}

/// **The load-bearing check.** A layer written into another layer's slot is
/// silent corruption of the drafter's input, not a crash.
#[test]
fn a_layer_written_into_the_wrong_slot_is_refused() {
    let s = slots();
    // Zero-based layer 13 (checkpoint ID 14) belongs in slot 1.
    assert!(s.contracted_slot(13, 1).is_ok());
    for wrong in [0, 2, 3, 4] {
        let error = s
            .contracted_slot(13, wrong)
            .expect_err("wrong slot must be refused");
        assert!(format!("{error:#}").contains("belongs in slot 1"));
    }
}

#[test]
fn contracted_slots_are_exact_bf16_hidden_vectors() {
    let s = slots();
    for (slot, layer) in GLM53_CAPTURE_LAYERS.iter().copied().enumerate() {
        let buffer = s.contracted_slot(layer, slot as u32).unwrap();
        assert_eq!(buffer.bytes, 4096 * 2);
        assert_eq!(
            buffer.ptr.0,
            BASE + slot as u64 * u64::from(GLM53_CAPTURE_MAX_ROWS) * 4096 * 2
        );
        let last = s.slot_row(slot as u32, 7).unwrap();
        assert_eq!(last.ptr.0, buffer.ptr.0 + 7 * 4096 * 2);
    }
    assert!(s.slot_rows(0, 0).is_err());
    assert!(s.slot_rows(0, 9).is_err());
    assert!(s.slot_row(0, 8).is_err());
}

#[test]
fn bind_refuses_null_misaligned_or_undersized_regions() {
    assert!(Glm53CaptureSlots::bind(DevicePtr(0), GLM53_DFLASH_CAPTURE_BYTES).is_err());
    assert!(Glm53CaptureSlots::bind(DevicePtr(BASE + 1), GLM53_DFLASH_CAPTURE_BYTES).is_err());
    assert!(Glm53CaptureSlots::bind(DevicePtr(BASE), GLM53_DFLASH_CAPTURE_BYTES - 1).is_err());
}
