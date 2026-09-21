// SPDX-License-Identifier: AGPL-3.0-only

use super::encode_position_ids;

fn rows(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

#[test]
fn target_image_delta_must_not_shift_dflash_chronological_context_or_noise() {
    // The target's 13-token multimodal prompt has logical tail 8. This
    // qwen3/default-RoPE drafter still addresses physical context 9..13
    // and anchor/noise 13..21; target delta -5 is NOT its position space.
    let mut bytes = [0xA5; 52];
    let used = encode_position_ids(&mut bytes, 13, 4, 8).unwrap();
    assert_eq!(used, 48);
    assert_eq!(rows(&bytes[..used]), (9..21).collect::<Vec<_>>());
    assert_ne!(rows(&bytes[..used]), (4..16).collect::<Vec<_>>());
    assert_eq!(&bytes[used..], &[0xA5; 4]);
}

#[test]
fn context_before_physical_origin_rejects_without_partial_staging() {
    let mut bytes = [0xA5; 24];
    assert!(encode_position_ids(&mut bytes, 2, 4, 2).is_err());
    assert_eq!(bytes, [0xA5; 24]);
}

#[test]
fn last_row_overflow_rejects_before_any_pinned_byte_changes() {
    let mut bytes = [0xA5; 12];
    assert!(encode_position_ids(&mut bytes, i32::MAX as usize, 1, 2).is_err());
    assert_eq!(bytes, [0xA5; 12]);
}
