// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    TC_VERIFY_MAX_ROWS, TC_VERIFY_MIN_ROWS, decode_tc_parity_engaged, rows_in_tc_verify_window,
};

#[test]
fn window_is_four_to_thirty_two_rows_inclusive() {
    assert!(!rows_in_tc_verify_window(TC_VERIFY_MIN_ROWS - 1));
    assert!(rows_in_tc_verify_window(TC_VERIFY_MIN_ROWS));
    assert!(rows_in_tc_verify_window(TC_VERIFY_MAX_ROWS));
    assert!(!rows_in_tc_verify_window(TC_VERIFY_MAX_ROWS + 1));
}

#[test]
fn parity_engages_only_when_requested_and_the_verify_width_is_in_window() {
    // gamma 8 -> 9 rows, gamma 15 -> 16 rows: engaged.
    assert!(decode_tc_parity_engaged(true, Some(9)));
    assert!(decode_tc_parity_engaged(true, Some(16)));
    // gamma <= 2 (K2/K3 fused kernels) and gamma > 31 (wide m128 FFN): fail closed.
    assert!(!decode_tc_parity_engaged(true, Some(3)));
    assert!(!decode_tc_parity_engaged(true, Some(33)));
    // No drafter, or not requested: off.
    assert!(!decode_tc_parity_engaged(true, None));
    assert!(!decode_tc_parity_engaged(false, Some(9)));
}
