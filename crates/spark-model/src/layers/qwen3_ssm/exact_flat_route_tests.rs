// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn route_is_exact_only_for_flat_dflash_windows_with_all_handles() {
    for rows in 4..=32 {
        assert_eq!(
            exact_flat_ssm_route(rows, false, true, true, true, true),
            ExactFlatSsmRoute::ExactSequence
        );
    }
    for rows in [0, 1, 2, 3, 33, 64] {
        assert_eq!(
            exact_flat_ssm_route(rows, false, true, true, true, true),
            ExactFlatSsmRoute::Existing
        );
    }
    assert_eq!(
        exact_flat_ssm_route(5, true, true, true, true, true),
        ExactFlatSsmRoute::Existing
    );
}

#[test]
fn missing_exact_handle_falls_back_to_serial_k1() {
    for missing in 0..4 {
        let mut handles = [true; 4];
        handles[missing] = false;
        assert_eq!(
            exact_flat_ssm_route(4, false, handles[0], handles[1], handles[2], handles[3]),
            ExactFlatSsmRoute::SerialK1
        );
    }
}

#[test]
fn contiguous_intermediates_fail_closed() {
    let good = [DevicePtr(100), DevicePtr(116), DevicePtr(132)];
    assert_eq!(
        contiguous_intermediate_base(&good, 3, 16, "test").unwrap(),
        DevicePtr(100)
    );
    let bad = [DevicePtr(100), DevicePtr(117), DevicePtr(132)];
    assert!(contiguous_intermediate_base(&bad, 3, 16, "test").is_err());
    assert!(contiguous_intermediate_base(&good, 4, 16, "test").is_err());
}
