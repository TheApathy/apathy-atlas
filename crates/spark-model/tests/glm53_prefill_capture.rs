// SPDX-License-Identifier: AGPL-3.0-only
//! CPU contracts for the bounded layer-major capture bank; no GPU or model load.
// STALE, not compiled: this test (and tests/support/glm53_prefill_capture_copying.rs)
// predates the P57 capture's Copies8/Gather128 transfer-mode API — `bind` now
// takes a `CaptureTransferMode` and `CaptureSlice` has no `copies` field — and
// fails to compile on the perf/glm-5.3-flash-prefill-20260920 tip as well.
// The in-crate model/glm53/prefill_capture_tile_tests.rs covers the current
// API. Re-enable by porting it to that API.
#![cfg(any())]

mod model {
    pub mod glm53 {
        pub use spark_model::model::glm53::GLM53_CAPTURE_LAYERS;
    }
}
mod layers {
    pub use spark_model::layers::Glm53TargetGeometry;
}
#[path = "support/glm53_prefill_capture_copying.rs"]
mod copying;
#[path = "../src/model/glm53/prefill_capture_ingest.rs"]
#[allow(dead_code)] // Dispatcher entry points are covered by the separate tap target.
mod prefill_capture_ingest;
#[path = "../src/model/glm53/prefill_capture_plan.rs"]
mod prefill_capture_plan;

use prefill_capture_ingest::CaptureReceipt;
use prefill_capture_plan::{BoundCapturePlan, DeviceSpan, PrefillCapturePlan};

const BANK: u64 = 0x1000_0000;
const STAGING: u64 = 0x2000_0000;
const PROJECTED: u64 = 0x3000_0000;
const LIMIT: u32 = 2047;

fn plan() -> PrefillCapturePlan {
    PrefillCapturePlan::new(32, 17, 11, 11, LIMIT, 8192).unwrap()
}

fn spans(plan: &PrefillCapturePlan) -> [DeviceSpan; 3] {
    [
        DeviceSpan {
            address: BANK,
            bytes: plan.bank_bytes(),
        },
        DeviceSpan {
            address: STAGING,
            bytes: 8 * 5 * plan.row_bytes(),
        },
        DeviceSpan {
            address: PROJECTED,
            bytes: LIMIT as usize * plan.row_bytes(),
        },
    ]
}

fn bound() -> BoundCapturePlan {
    let plan = plan();
    let [bank, staging, projected] = spans(&plan);
    plan.bind(bank, staging, projected).unwrap()
}

fn complete_receipt() -> CaptureReceipt {
    let mut receipt = CaptureReceipt::new(bound());
    for (slot, layer) in model::glm53::GLM53_CAPTURE_LAYERS.into_iter().enumerate() {
        receipt.record_tap(layer, slot).unwrap();
    }
    receipt
}

#[test]
fn geometry_and_eighty_mib_ceiling_are_checked() {
    let plan = plan();
    assert_eq!(plan.row_bytes(), 8192);
    assert_eq!(plan.rows(), 17);
    assert_eq!(plan.end_position(), 28);
    assert_eq!(plan.slot_stride_bytes(), 32 * 8192);
    assert_eq!(plan.bank_bytes(), 5 * 32 * 8192);
    let max = PrefillCapturePlan::new(2048, 2047, 0, 0, LIMIT, 8192).unwrap();
    assert_eq!(max.bank_bytes(), 80 * 1024 * 1024);
    for args in [
        (0, 1, 0, 0, LIMIT, 8192),
        (2049, 1, 0, 0, LIMIT, 8192),
        (32, 0, 0, 0, LIMIT, 8192),
        (32, 33, 0, 0, LIMIT, 8192),
        (32, 17, 11, 10, LIMIT, 8192),
        (32, 17, 2031, 2031, LIMIT, 8192),
        (32, 17, 11, 11, LIMIT, 27),
        (32, 17, 0, 0, 0, 8192),
        (32, 17, u32::MAX - 8, u32::MAX - 8, u32::MAX, u32::MAX),
    ] {
        assert!(PrefillCapturePlan::new(args.0, args.1, args.2, args.3, args.4, args.5).is_err());
    }
}

#[test]
fn spans_reject_null_alignment_extent_overflow_and_aliases() {
    for slot in 0..3 {
        for invalid in [
            DeviceSpan {
                address: 0,
                bytes: usize::MAX,
            },
            DeviceSpan {
                address: BANK + 1,
                bytes: usize::MAX,
            },
            DeviceSpan {
                address: u64::MAX - 255,
                bytes: usize::MAX,
            },
            DeviceSpan {
                address: BANK,
                bytes: 1,
            },
        ] {
            let p = plan();
            let mut buffers = spans(&p);
            buffers[slot] = invalid;
            assert!(p.bind(buffers[0], buffers[1], buffers[2]).is_err());
        }
    }
    for (source, destination) in [(0, 1), (0, 2), (1, 2)] {
        let p = plan();
        let mut buffers = spans(&p);
        buffers[destination].address = buffers[source].address;
        assert!(p.bind(buffers[0], buffers[1], buffers[2]).is_err());
    }
}

#[test]
fn canonical_taps_and_nonzero_row_gathers_are_exact() {
    let bound = bound();
    for (slot, layer) in model::glm53::GLM53_CAPTURE_LAYERS.into_iter().enumerate() {
        let destination = bound.tap_destination(layer, slot).unwrap();
        assert_eq!(destination.address, BANK + (slot * 32 * 8192) as u64);
        assert_eq!(destination.bytes, 17 * 8192);
        assert!(bound.tap_destination(layer + 1, slot).is_err());
    }
    assert!(bound.tap_destination(4, 5).is_err());
    for (first, rows) in [(0, 8), (8, 8), (16, 1)] {
        let slice = bound.slice(first, rows).unwrap();
        assert_eq!(slice.copies.len(), rows as usize * 5);
        for (index, copy) in slice.copies.iter().enumerate() {
            let row = index / 5;
            let slot = index % 5;
            assert_eq!(
                copy.source,
                BANK + ((slot * 32 + first as usize + row) * 8192) as u64
            );
            assert_eq!(copy.destination, STAGING + (index * 8192) as u64);
            assert_eq!(copy.bytes, 8192);
        }
        assert_eq!(slice.projection.input.address, STAGING);
        assert_eq!(slice.projection.input.bytes, rows as usize * 5 * 8192);
        assert_eq!(
            slice.projection.output.address,
            PROJECTED + ((11 + first) * 8192) as u64
        );
        assert_eq!(slice.projection.output.bytes, rows as usize * 8192);
        assert_eq!(slice.projection.rows, rows);
    }
    for (first, rows) in [(0, 0), (0, 9), (16, 2), (17, 1), (u32::MAX, 8)] {
        assert!(bound.slice(first, rows).is_err());
    }
}

#[test]
fn tap_receipt_rejects_missing_duplicate_wrong_and_out_of_order_taps() {
    let taps = model::glm53::GLM53_CAPTURE_LAYERS;
    let mut receipt = CaptureReceipt::new(bound());
    assert!(receipt.record_tap(taps[1], 1).is_err());
    assert!(receipt.record_tap(taps[0] + 1, 0).is_err());
    receipt.record_tap(taps[0], 0).unwrap();
    assert!(receipt.record_tap(taps[0], 0).is_err());
    let mut io = copying::MemoryIo::new();
    assert!(receipt.ingest(&mut io, 28, 11, 7).is_err());
    assert_eq!(io.calls(), 0);
    assert_eq!(receipt.published_position(), None);
    assert!(!receipt.failed());
    for (slot, layer) in taps.into_iter().enumerate().skip(1) {
        receipt.record_tap(layer, slot).unwrap();
    }
    assert!(receipt.record_tap(taps[4], 4).is_err());
    assert!(receipt.ingest(&mut io, 27, 11, 7).is_err());
    assert!(receipt.ingest(&mut io, 28, 12, 7).is_err());
    assert_eq!(io.calls(), 0);
}
