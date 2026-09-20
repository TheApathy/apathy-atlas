// SPDX-License-Identifier: AGPL-3.0-only

#[allow(dead_code)]
#[path = "../src/model/glm53/verify_trace.rs"]
mod verify_trace;

use anyhow::{Result, bail};
use base64::Engine;
use verify_trace::{TraceIo, TracePhase, TracePoint, TraceRecord, observe, parse_position};

const VOCAB: usize = 154_880;
const ROW_BYTES: usize = VOCAB * 2;

#[derive(Default)]
struct MemoryTrace {
    row: Vec<u8>,
    copies: Vec<(u64, usize, u64)>,
    records: Vec<TraceRecord>,
    fail_copy: bool,
    fail_publish: bool,
}

impl TraceIo for MemoryTrace {
    fn copy_row(&mut self, source: u64, destination: &mut [u8], stream: u64) -> Result<()> {
        self.copies.push((source, destination.len(), stream));
        if self.fail_copy {
            bail!("injected trace copy failure");
        }
        assert_eq!(destination.len(), self.row.len());
        destination.copy_from_slice(&self.row);
        Ok(())
    }

    fn publish(&mut self, record: TraceRecord) -> Result<()> {
        if self.fail_publish {
            bail!("injected trace publish failure");
        }
        self.records.push(record);
        Ok(())
    }
}

fn point(phase: TracePhase) -> TracePoint {
    TracePoint {
        phase,
        start: 209,
        rows: 8,
        anchor: 6504,
        selected_oracle: match phase {
            TracePhase::Wide => Some(198),
            TracePhase::Replay => None,
        },
        device_selector: true,
        stream: 17,
    }
}

fn set(row: &mut [u8], token: usize, value: f32) {
    let bits = (value.to_bits() >> 16) as u16;
    row[token * 2..token * 2 + 2].copy_from_slice(&bits.to_le_bytes());
}

fn memory() -> MemoryTrace {
    let mut row = vec![0; ROW_BYTES];
    set(&mut row, 198, 19.375);
    set(&mut row, 271, 19.5);
    MemoryTrace {
        row,
        ..Default::default()
    }
}

#[test]
fn trace_selector_is_absent_or_one_bounded_canonical_position() {
    assert_eq!(parse_position(None).unwrap(), None);
    for position in [0, 209, 2047] {
        assert_eq!(
            parse_position(Some(&position.to_string())).unwrap(),
            Some(position)
        );
    }
    for invalid in [
        "",
        " ",
        " 209",
        "209 ",
        "0209",
        "+209",
        "-1",
        "2048",
        "4294967296",
        "true",
    ] {
        assert!(parse_position(Some(invalid)).is_err(), "{invalid:?}");
    }
}

#[test]
fn disabled_or_other_position_has_zero_io() {
    let mut io = MemoryTrace::default();
    observe(&mut io, None, point(TracePhase::Wide), 0).unwrap();
    observe(&mut io, Some(208), point(TracePhase::Wide), 0).unwrap();
    assert!(io.copies.is_empty());
    assert!(io.records.is_empty());
}

#[test]
fn unique_max_is_reported_without_replacing_the_selected_oracle() {
    let mut io = memory();
    observe(&mut io, Some(209), point(TracePhase::Wide), 0x8000).unwrap();
    assert_eq!(io.copies, [(0x8000, ROW_BYTES, 17)]);
    let record = &io.records[0];
    assert_eq!((record.start, record.rows, record.anchor), (209, 8, 6504));
    assert_eq!(record.selected_oracle, Some(198));
    assert_eq!(record.cpu_first_argmax, Some(271));
    assert_eq!(record.cpu_last_argmax, Some(271));
    assert_eq!(record.newline_bits, (19.375f32.to_bits() >> 16) as u16);
    assert_eq!(record.double_newline_bits, (19.5f32.to_bits() >> 16) as u16);
    assert_eq!(record.maximum_bits, Some((19.5f32.to_bits() >> 16) as u16));
    assert_eq!(record.nonfinite_values, 0);
    assert_eq!(record.raw_bf16_fnv1a64, atlas_tier::hash::fnv1a_64(&io.row));
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(&record.raw_bf16_base64)
            .unwrap(),
        io.row
    );
}

#[test]
fn tied_scores_report_both_policies_and_replay_is_labeled_separately() {
    let mut io = memory();
    set(&mut io.row, 198, 19.5);
    observe(&mut io, Some(209), point(TracePhase::Wide), 0x8000).unwrap();
    observe(&mut io, Some(209), point(TracePhase::Replay), 0x8000).unwrap();
    assert_eq!(io.records.len(), 2);
    for record in &io.records {
        assert_eq!(record.cpu_first_argmax, Some(198));
        assert_eq!(record.cpu_last_argmax, Some(271));
    }
    assert_eq!(io.records[0].phase, TracePhase::Wide);
    assert_eq!(io.records[1].phase, TracePhase::Replay);
    assert_eq!(io.records[1].selected_oracle, None);
}

#[test]
fn nonfinite_payload_is_retained_as_diagnostic_data() {
    let mut io = memory();
    set(&mut io.row, 12, f32::NAN);
    observe(&mut io, Some(209), point(TracePhase::Wide), 0x8000).unwrap();
    assert_eq!(io.records[0].nonfinite_values, 1);
    assert_eq!(io.records[0].cpu_first_argmax, Some(271));
    let raw = base64::engine::general_purpose::STANDARD
        .decode(&io.records[0].raw_bf16_base64)
        .unwrap();
    assert_eq!(raw, io.row);
}

#[test]
fn malformed_trace_geometry_is_rejected_before_copy() {
    let mut io = memory();
    for rows in [0, 1, 9, usize::MAX] {
        let mut invalid = point(TracePhase::Wide);
        invalid.rows = rows;
        assert!(observe(&mut io, Some(209), invalid, 0x8000).is_err());
    }
    let mut invalid = point(TracePhase::Wide);
    invalid.anchor = VOCAB as u32;
    assert!(observe(&mut io, Some(209), invalid, 0x8000).is_err());
    assert!(observe(&mut io, Some(209), point(TracePhase::Wide), 0).is_err());
    assert!(io.copies.is_empty());
}

#[test]
fn copy_and_publish_errors_are_propagated() {
    let mut io = memory();
    io.fail_copy = true;
    assert!(
        observe(&mut io, Some(209), point(TracePhase::Wide), 0x8000)
            .unwrap_err()
            .to_string()
            .contains("copy failure")
    );
    assert!(io.records.is_empty());
    io.fail_copy = false;
    io.fail_publish = true;
    assert!(
        observe(&mut io, Some(209), point(TracePhase::Wide), 0x8000)
            .unwrap_err()
            .to_string()
            .contains("publish failure")
    );
    assert!(io.records.is_empty());
}
