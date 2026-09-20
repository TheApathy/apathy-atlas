// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    CORE32_SELECTOR, DEVICE_SELECTOR, HC_SELECTOR, TILE_ROWS, admit_surface, parse, parse_core32,
    parse_device, parse_hc, plan, plan_for_tile,
};

#[test]
fn selector_is_exact_and_default_off() {
    assert!(!parse(None).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("1")).unwrap());
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(parse(Some(bad)).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn device_selector_is_exact_default_off_and_nested() {
    assert!(!parse_device(None, false).unwrap());
    assert!(!parse_device(Some("0"), false).unwrap());
    assert!(parse_device(Some("1"), true).unwrap());
    assert!(parse_device(Some("1"), false).is_err());
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(parse_device(Some(bad), true).is_err(), "accepted {bad:?}");
    }
    assert_eq!(DEVICE_SELECTOR, "ATLAS_QWEN4_PREFILL_ATTN_DEVICE16");
}

#[test]
fn hc_selector_is_exact_default_off_and_nested_under_f37() {
    assert!(!parse_hc(None, false, false).unwrap());
    assert!(!parse_hc(Some("0"), true, true).unwrap());
    assert!(parse_hc(Some("1"), true, true).unwrap());
    assert!(parse_hc(Some("1"), false, true).is_err());
    assert!(parse_hc(Some("1"), true, false).is_err());
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(parse_hc(Some(bad), true, true).is_err(), "accepted {bad:?}");
    }
    assert_eq!(HC_SELECTOR, "ATLAS_QWEN4_PREFILL_ATTN_HC16");
}

#[test]
fn core32_selector_is_exact_default_off_and_fully_nested() {
    assert!(!parse_core32(None, true, true, true).unwrap());
    assert!(!parse_core32(Some("0"), true, true, true).unwrap());
    assert!(parse_core32(Some("1"), true, true, true).unwrap());
    for deps in [
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        assert!(parse_core32(Some("1"), deps.0, deps.1, deps.2).is_err());
    }
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(
            parse_core32(Some(bad), true, true, true).is_err(),
            "accepted {bad:?}"
        );
    }
    assert_eq!(CORE32_SELECTOR, "ATLAS_QWEN4_PREFILL_ATTN_CORE32");
}

#[test]
fn planner_covers_full_tiles_and_scalar_tails() {
    for (rows, tiles, tail) in [
        (16, 1, 0),
        (17, 1, 1),
        (31, 1, 15),
        (32, 2, 0),
        (2048, 128, 0),
    ] {
        let p = plan(rows, 0, 9, 4096).unwrap();
        assert_eq!((p.full_tiles, p.tail_rows), (tiles, tail));
        assert_eq!(p.block_table_words, TILE_ROWS * 9);
        assert_eq!(p.seq_lens_offset % 8, 0);
        assert!(p.scratch_bytes <= 4096);
    }
}

#[test]
fn planner_fails_before_unsafe_geometry() {
    for rows in [0, 1, 15, 2049, usize::MAX] {
        assert!(plan(rows, 0, 9, usize::MAX).is_err());
    }
    assert!(plan(16, 1, 9, 4096).is_err());
    assert!(plan(16, 0, 0, 4096).is_err());
    let needed = plan(16, 0, 9, usize::MAX).unwrap().scratch_bytes;
    assert!(plan(16, 0, 9, needed - 1).is_err());
}

#[test]
fn planner32_halves_complete_tiles_and_sizes_causal_metadata() {
    let p = plan_for_tile(2048, 0, 9, 8192, 32).unwrap();
    assert_eq!((p.full_tiles, p.tail_rows), (64, 0));
    assert_eq!(p.block_table_words, 32 * 9);
    assert_eq!(p.seq_lens_offset % 8, 0);
    assert!(p.scratch_bytes <= 8192);
    for tile in [0, 1, 15, 17, 31, 33, 64, usize::MAX] {
        if tile != 16 && tile != 32 {
            assert!(plan_for_tile(2048, 0, 9, usize::MAX, tile).is_err());
        }
    }
}

#[test]
fn surface_is_cold_text_single_chunk_without_swap() {
    assert!(admit_surface(false, 0, 0, 2048, 2048, false).is_ok());
    for args in [
        (true, 0, 0, 2048, 2048, false),
        (false, 1, 0, 2048, 2048, false),
        (false, 0, 1, 2047, 2048, false),
        (false, 0, 0, 1024, 2048, false),
        (false, 0, 0, 2048, 2048, true),
    ] {
        assert!(admit_surface(args.0, args.1, args.2, args.3, args.4, args.5).is_err());
    }
}
