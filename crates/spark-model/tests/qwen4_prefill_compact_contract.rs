// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/moe/qwen4_compact_contract.rs"]
mod contract;

#[test]
fn selector_is_explicit_and_strict() {
    for v in [None, Some("0")] {
        assert_eq!(contract::parse(v), Ok(false));
    }
    assert_eq!(contract::parse(Some("1")), Ok(true));
    for v in ["", "true", "01", " 1", "1\n", "2"] {
        assert!(contract::parse(Some(v)).is_err());
    }
}

#[test]
fn contract_encoding_has_exact_cuda_field_offsets_and_bounds() {
    assert_eq!(contract::ARENA_BYTES, 14_448);
    for rows in [0, 1, 2049, 8192, usize::MAX] {
        assert!(contract::contract(rows).is_err());
    }
    for rows in [2, 3, 63, 64, 65, 2013, 2048] {
        let bytes = contract::contract(rows).unwrap();
        assert_eq!(bytes.len(), 64);
        assert_eq!(&bytes[..8], &0x4f52494749363430u64.to_le_bytes());
        let fields: Vec<u32> = bytes[8..]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            fields,
            [
                1,
                rows as u32,
                2560,
                640,
                512,
                10,
                rows as u32 * 10,
                64,
                64,
                contract::GRID,
                contract::MAX_ITEMS,
                14_384,
                0,
                0
            ]
        );
        assert!(rows.div_ceil(64) * 10 + 512 <= contract::MAX_ITEMS as usize);
    }
}

#[test]
fn pending_building_and_all_errors_fail_closed() {
    assert!(contract::check_status(1).is_ok());
    for status in [-2, -1, 0, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, i32::MAX] {
        assert!(contract::check_status(status).is_err());
    }
}

#[test]
fn missing_partial_and_postload_selector_changes_fail() {
    assert!(contract::bundle_matches(false, None));
    assert!(contract::bundle_matches(true, Some((1, 2))));
    assert!(!contract::bundle_matches(true, None));
    assert!(!contract::bundle_matches(false, Some((1, 2))));
    for handles in [(0, 0), (0, 2), (1, 0)] {
        assert!(!contract::bundle_matches(true, Some(handles)));
    }
}

#[test]
fn pointer_extent_and_alias_guards_include_adjacency() {
    let a = contract::region(4096, contract::WORKSPACE_BYTES, 16).unwrap();
    let b = contract::region(a.1, contract::CONTRACT_BYTES, 16).unwrap();
    assert!(contract::disjoint(a, b));
    assert!(!contract::disjoint(a, (a.1 - 1, b.1)));
    assert!(!contract::disjoint(a, a));
    assert!(contract::region(16, 16, 0).is_err());
    for (ptr, bytes) in [(0, 16), (16, 0), (17, 16), (u64::MAX - 15, 16)] {
        assert!(contract::region(ptr, bytes, 16).is_err());
    }
}

#[test]
fn skewed_and_tail_tiles_cover_each_expanded_row_exactly_once() {
    for rows in [2usize, 3, 63, 64, 65, 2013, 2048] {
        for active in [1usize, 2, 31, 512] {
            let expanded = rows * 10;
            let counts: Vec<usize> = (0..512)
                .map(|e| {
                    if e < active {
                        expanded / active + usize::from(e < expanded % active)
                    } else {
                        0
                    }
                })
                .collect();
            let mut offset = 0;
            let mut items = 0;
            let mut coverage = vec![0u8; expanded];
            for count in counts {
                for tile in 0..count.div_ceil(64) {
                    items += 1;
                    for row in tile * 64..((tile + 1) * 64).min(count) {
                        coverage[offset + row] += 1;
                    }
                }
                offset += count;
            }
            assert_eq!(offset, expanded);
            assert!(items > 0 && items <= contract::MAX_ITEMS as usize);
            assert!(coverage.into_iter().all(|visits| visits == 1));
        }
    }
}
