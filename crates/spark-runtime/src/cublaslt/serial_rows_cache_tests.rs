// SPDX-License-Identifier: AGPL-3.0-only

use super::serial_rows_cache_ffi::{
    BoundedKeys, CacheSlot, MAX_RESOURCE_SETS, ResourceKey, parse_setting,
};
use super::{ByteSpan, Orientation, SerialRowsRequest};

fn request(rows: u32, n: u32, k: u32, orientation: Orientation) -> SerialRowsRequest {
    SerialRowsRequest {
        rows,
        n,
        k,
        orientation,
        act: ByteSpan {
            address: 0x1000,
            bytes: rows as usize * k as usize * 2,
        },
        weight: ByteSpan {
            address: 0x20_0000,
            bytes: n as usize * k as usize * 2,
        },
        out: ByteSpan {
            address: 0x40_0000,
            bytes: rows as usize * n as usize * 2,
        },
        stream: 0x8000,
    }
}

#[test]
fn selector_is_strict_and_defaults_off() {
    assert!(!parse_setting(None).unwrap());
    assert!(!parse_setting(Some("0")).unwrap());
    assert!(parse_setting(Some("1")).unwrap());
    for invalid in ["", "true", "2", " 1"] {
        assert!(parse_setting(Some(invalid)).is_err());
    }
}

#[test]
fn key_separates_rows_dimensions_and_orientation() {
    let base = ResourceKey::new(request(6, 15, 20, Orientation::Nk)).unwrap();
    assert_ne!(
        base,
        ResourceKey::new(request(7, 15, 20, Orientation::Nk)).unwrap()
    );
    assert_ne!(
        base,
        ResourceKey::new(request(6, 16, 20, Orientation::Nk)).unwrap()
    );
    assert_ne!(
        base,
        ResourceKey::new(request(6, 15, 21, Orientation::Nk)).unwrap()
    );
    assert_ne!(
        base,
        ResourceKey::new(request(6, 15, 20, Orientation::Kn)).unwrap()
    );
    assert!(ResourceKey::new(request(1, 15, 20, Orientation::Nk)).is_err());
}

#[test]
fn bounded_index_reuses_and_fails_closed() {
    let mut keys = BoundedKeys::default();
    let first = ResourceKey::new(request(2, 1, 1, Orientation::Nk)).unwrap();
    assert_eq!(keys.locate(first).unwrap(), CacheSlot::Vacant);
    keys.record(first).unwrap();
    assert_eq!(keys.locate(first).unwrap(), CacheSlot::Existing);

    for n in 2..=MAX_RESOURCE_SETS as u32 {
        keys.record(ResourceKey::new(request(2, n, 1, Orientation::Nk)).unwrap())
            .unwrap();
    }
    assert_eq!(keys.len(), MAX_RESOURCE_SETS);
    assert!(
        keys.locate(ResourceKey::new(request(3, 1, 1, Orientation::Nk)).unwrap())
            .is_err()
    );
}
