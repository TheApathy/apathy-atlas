// SPDX-License-Identifier: AGPL-3.0-only

use super::bf16_plan_cache::GemmKey;
use super::classic_bf16::{BatchedPlan, GemmPlan, parse_setting};
use super::{ByteSpan, Orientation, SerialRowsRequest};

#[test]
fn selector_is_strict_and_default_off() {
    assert!(!parse_setting(None).unwrap());
    assert!(!parse_setting(Some("0")).unwrap());
    assert!(parse_setting(Some("1")).unwrap());
    assert!(parse_setting(Some("true")).is_err());
}

#[test]
fn row_major_weight_transpose_maps_to_exact_column_major_contract() {
    let plan = GemmPlan::new(GemmKey::new(6, 154_880, 4_096, true).unwrap()).unwrap();
    assert_eq!(plan.trans_a, 1);
    assert_eq!((plan.m, plan.n, plan.k), (154_880, 6, 4_096));
    assert_eq!((plan.lda, plan.ldb, plan.ldc), (4_096, 4_096, 154_880));
}

#[test]
fn row_major_act_weight_maps_without_transposing_the_weight() {
    let plan = GemmPlan::new(GemmKey::new(8, 4_096, 512, false).unwrap()).unwrap();
    assert_eq!(plan.trans_a, 0);
    assert_eq!((plan.m, plan.n, plan.k), (4_096, 8, 512));
    assert_eq!((plan.lda, plan.ldb, plan.ldc), (4_096, 512, 4_096));
}

#[test]
fn native_rows_use_one_exact_strided_m1_batch() {
    let plan = BatchedPlan::new(SerialRowsRequest {
        rows: 6,
        n: 1_024,
        k: 4_096,
        orientation: Orientation::Nk,
        act: ByteSpan {
            address: 0x1000,
            bytes: 6 * 4_096 * 2,
        },
        weight: ByteSpan {
            address: 0x1_0000,
            bytes: 1_024 * 4_096 * 2,
        },
        out: ByteSpan {
            address: 0x1_000_000,
            bytes: 6 * 1_024 * 2,
        },
        stream: 0,
    })
    .unwrap();
    assert_eq!((plan.gemm.m, plan.gemm.n, plan.gemm.k), (1_024, 1, 4_096));
    assert_eq!(plan.gemm.trans_a, 1);
    assert_eq!(
        (plan.stride_weight, plan.stride_act, plan.stride_out),
        (0, 4_096, 1_024)
    );
    assert_eq!(plan.batch_count, 6);
}
