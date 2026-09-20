// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::OsStr;

use anyhow::{Result, bail};

use super::GLM53_CAPTURE_LAYERS;
use super::dflash2_runtime::QUERY_TOKENS;
use super::prefill_capture_ingest::{CaptureReceipt, PrefillCaptureIo};
use super::prefill_capture_plan::{
    CaptureTransfer, CaptureTransferMode, CopyStep, DeviceSpan, GatherStep, PrefillCapturePlan,
    ProjectionStep, TILED_CAPTURE_ROWS, TILED_CAPTURE_STAGING_BYTES,
};

const BANK: DeviceSpan = DeviceSpan {
    address: 0x1000_0000,
    bytes: 80 * 1024 * 1024,
};
const STAGING: DeviceSpan = DeviceSpan {
    address: 0x2000_0000,
    bytes: TILED_CAPTURE_STAGING_BYTES,
};
const PROJECTED: DeviceSpan = DeviceSpan {
    address: 0x3000_0000,
    bytes: 2047 * 4096 * 2,
};

fn plan(rows: u32) -> PrefillCapturePlan {
    PrefillCapturePlan::new(2048, rows, 0, 0, 2047, 2048).unwrap()
}

fn slices(rows: u32, mode: CaptureTransferMode) -> Vec<super::prefill_capture_plan::CaptureSlice> {
    let bound = plan(rows).bind(BANK, STAGING, PROJECTED, mode).unwrap();
    let tile = mode.rows();
    (0..rows)
        .step_by(tile as usize)
        .map(|first| bound.slice(first, tile.min(rows - first)).unwrap())
        .collect()
}

#[test]
fn tile_selector_is_strict_default_off_and_query_width_is_unchanged() {
    assert_eq!(
        CaptureTransferMode::parse(None).unwrap(),
        CaptureTransferMode::Copies8
    );
    assert_eq!(
        CaptureTransferMode::parse(Some(OsStr::new("0"))).unwrap(),
        CaptureTransferMode::Copies8
    );
    assert_eq!(
        CaptureTransferMode::parse(Some(OsStr::new("1"))).unwrap(),
        CaptureTransferMode::Gather128
    );
    for invalid in ["", "2", "01", "true", " 1", "1 "] {
        assert!(CaptureTransferMode::parse(Some(OsStr::new(invalid))).is_err());
    }
    assert_eq!(QUERY_TOKENS, 8);
    assert_eq!(TILED_CAPTURE_ROWS, 128);
    assert_eq!(TILED_CAPTURE_STAGING_BYTES, 5 * 128 * 4096 * 2);
    assert!(TILED_CAPTURE_STAGING_BYTES <= 6 * 1024 * 1024);
}

#[test]
fn tiled_tail_partition_is_exact_for_every_boundary() {
    for (rows, expected) in [
        (1, vec![1]),
        (7, vec![7]),
        (8, vec![8]),
        (9, vec![9]),
        (127, vec![127]),
        (128, vec![128]),
        (129, vec![128, 1]),
        (2047, vec![128; 15].into_iter().chain([127]).collect()),
    ] {
        let got = slices(rows, CaptureTransferMode::Gather128)
            .into_iter()
            .map(|slice| slice.projection.rows)
            .collect::<Vec<_>>();
        assert_eq!(got, expected, "rows={rows}");
    }
}

#[test]
fn gather_descriptor_is_order_equivalent_to_legacy_row_tap_copies() {
    let slice = slices(129, CaptureTransferMode::Gather128).remove(0);
    let CaptureTransfer::Gather(gather) = slice.transfer else {
        panic!("tile selector did not produce gather")
    };
    let legacy = slices(8, CaptureTransferMode::Copies8).remove(0);
    let CaptureTransfer::Copies(copies) = legacy.transfer else {
        panic!("default selector did not produce copies")
    };
    assert_eq!(copies.len(), 8 * 5);
    assert_eq!(gather.rows, 128);
    assert_eq!(gather.first_row, 0);
    assert_eq!(gather.row_bytes, 4096 * 2);
    assert_eq!(gather.slot_stride_bytes, 2048 * 4096 * 2);
    for row in [0, 1, 63, 127] {
        for slot in 0..5 {
            for column in [0, 1, 2047, 4095] {
                let (source, destination) = gather.addresses(row, slot, column).unwrap();
                let row = row as usize;
                let expected_source =
                    BANK.address + (slot * 2048 * 4096 * 2 + row * 4096 * 2 + column * 2) as u64;
                let expected_destination =
                    STAGING.address + ((row * 5 + slot) * 4096 * 2 + column * 2) as u64;
                assert_eq!(
                    (source, destination),
                    (expected_source, expected_destination)
                );
                if row < 8 {
                    let copy = copies[row * 5 + slot];
                    assert_eq!(source, copy.source + (column * 2) as u64);
                    assert_eq!(destination, copy.destination + (column * 2) as u64);
                    assert_eq!(copy.bytes, 4096 * 2);
                }
            }
        }
    }
    assert!(gather.addresses(128, 0, 0).is_err());
    assert!(gather.addresses(0, 5, 0).is_err());
    assert!(gather.addresses(0, 0, 4096).is_err());
}

#[test]
fn bind_rejects_short_overlapping_and_overflowed_staging() {
    let p = plan(129);
    assert!(
        p.clone()
            .bind(
                BANK,
                DeviceSpan {
                    bytes: TILED_CAPTURE_STAGING_BYTES - 1,
                    ..STAGING
                },
                PROJECTED,
                CaptureTransferMode::Gather128,
            )
            .is_err()
    );
    assert!(
        p.clone()
            .bind(BANK, BANK, PROJECTED, CaptureTransferMode::Gather128)
            .is_err()
    );
    assert!(
        p.bind(
            BANK,
            DeviceSpan {
                address: u64::MAX - 255,
                bytes: TILED_CAPTURE_STAGING_BYTES,
            },
            PROJECTED,
            CaptureTransferMode::Gather128,
        )
        .is_err()
    );
}

#[derive(Default)]
struct RecordingIo {
    events: Vec<&'static str>,
    fail_gather: bool,
    fail_project: bool,
}

impl PrefillCaptureIo for RecordingIo {
    fn copy(&mut self, _step: CopyStep, _stream: u64) -> Result<()> {
        self.events.push("copy");
        Ok(())
    }

    fn gather(&mut self, _step: GatherStep, _stream: u64) -> Result<()> {
        self.events.push("gather");
        if self.fail_gather {
            bail!("injected gather failure")
        }
        Ok(())
    }

    fn project_norm(&mut self, _step: ProjectionStep, _stream: u64) -> Result<()> {
        self.events.push("project");
        if self.fail_project {
            bail!("injected project failure")
        }
        Ok(())
    }

    fn synchronize(&mut self, _stream: u64) -> Result<()> {
        self.events.push("sync");
        Ok(())
    }
}

fn receipt(rows: u32, mode: CaptureTransferMode) -> CaptureReceipt {
    let bound = plan(rows).bind(BANK, STAGING, PROJECTED, mode).unwrap();
    let mut receipt = CaptureReceipt::new(bound);
    for (slot, layer) in GLM53_CAPTURE_LAYERS.iter().copied().enumerate() {
        receipt.record_tap(layer, slot).unwrap();
    }
    receipt
}

#[test]
fn tiled_ingest_orders_one_gather_before_each_projection_and_fences_last() {
    let mut receipt = receipt(129, CaptureTransferMode::Gather128);
    let mut io = RecordingIo::default();
    assert_eq!(receipt.ingest(&mut io, 129, 0, 77).unwrap(), 129);
    assert_eq!(
        io.events,
        ["gather", "project", "gather", "project", "sync"]
    );
    assert_eq!(receipt.published_position(), Some(129));
}

#[test]
fn tiled_ingest_poison_is_terminal_without_copy_fallback() {
    for fail_project in [false, true] {
        let mut receipt = receipt(9, CaptureTransferMode::Gather128);
        let mut io = RecordingIo {
            fail_gather: !fail_project,
            fail_project,
            ..RecordingIo::default()
        };
        assert!(receipt.ingest(&mut io, 9, 0, 1).is_err());
        assert!(receipt.failed());
        assert!(!io.events.contains(&"copy"));
        assert!(receipt.ingest(&mut io, 9, 0, 1).is_err());
    }
}

#[test]
fn cuda_source_pins_checked_slot_major_to_token_tap_gather() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/iq3/glm53_dflash2_capture.cu"
    ));
    assert!(source.contains("atlas_glm53_dflash2_gather_capture_tile"));
    assert!(source.contains("rows > 128U"));
    assert!(source.contains("slot_stride_elements"));
    assert!(source.contains("(row * GLM53_DFLASH2_CAPTURES + slot)"));
}
