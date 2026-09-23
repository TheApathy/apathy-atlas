// SPDX-License-Identifier: AGPL-3.0-only

use super::{OwnerIo, PreparedOwner};
use crate::model::glm53::prefill_input_exl3::{InputRegion, PreparedRows};

const ADDRESS: u64 = 0x3_0000_0000;
const STREAM: u64 = 17;
const ROW_BYTES: usize = 4096 * 2;

fn prepared() -> PreparedRows {
    PreparedRows {
        region: InputRegion {
            address: ADDRESS,
            bytes: 2 * ROW_BYTES,
        },
        rows: 2,
    }
}

#[derive(Debug, PartialEq)]
enum Event {
    Drain(u64),
    Free(u64),
}

struct ByteOwnerIo {
    allocation: Option<Vec<u8>>,
    pending: bool,
    copied: Vec<u8>,
    fail_drain: bool,
    fail_free: bool,
    events: Vec<Event>,
}

impl ByteOwnerIo {
    fn new() -> Self {
        Self {
            allocation: Some((0..2 * ROW_BYTES).map(|i| (i % 251) as u8).collect()),
            pending: false,
            copied: Vec::new(),
            fail_drain: false,
            fail_free: false,
            events: Vec::new(),
        }
    }
}

impl OwnerIo for ByteOwnerIo {
    fn drain(&mut self, stream: u64) -> anyhow::Result<()> {
        self.events.push(Event::Drain(stream));
        anyhow::ensure!(stream == STREAM, "wrong stream");
        anyhow::ensure!(!self.fail_drain, "injected drain failure");
        if self.pending {
            self.copied = self
                .allocation
                .as_ref()
                .expect("owner freed before drain")
                .clone();
            self.pending = false;
        }
        Ok(())
    }
    fn free(&mut self, address: u64) -> anyhow::Result<()> {
        self.events.push(Event::Free(address));
        anyhow::ensure!(address == ADDRESS && !self.pending, "unsafe free");
        anyhow::ensure!(!self.fail_free, "injected free failure");
        assert!(self.allocation.take().is_some(), "double free");
        Ok(())
    }
}

fn active() -> (PreparedOwner, ByteOwnerIo) {
    let mut owner = PreparedOwner::new();
    owner.publish(prepared()).unwrap();
    let rows = owner.begin().unwrap().unwrap();
    assert_eq!(
        (rows.region.address, rows.region.bytes, rows.rows),
        (ADDRESS, 2 * ROW_BYTES, 2)
    );
    assert!(owner.has_owner());
    (owner, ByteOwnerIo::new())
}

#[test]
fn empty_text_owner_has_no_io_and_ready_owner_is_not_overwritten() {
    let mut owner = PreparedOwner::new();
    let mut io = ByteOwnerIo::new();
    assert!(owner.begin().unwrap().is_none());
    owner.finish(&mut io).unwrap();
    owner.retry_quarantine(&mut io).unwrap();
    owner.clear(&mut io).unwrap();
    assert!(io.events.is_empty());
    owner.publish(prepared()).unwrap();
    let mut replacement = prepared();
    replacement.region.address += 2 * ROW_BYTES as u64;
    assert!(owner.publish(replacement).is_err());
    assert_eq!(owner.begin().unwrap().unwrap().region.address, ADDRESS);
    assert!(owner.begin().is_err());
}

#[test]
fn ready_or_unarmed_active_owner_can_be_released_without_fabricated_stream_work() {
    let mut owner = PreparedOwner::new();
    let mut io = ByteOwnerIo::new();
    owner.publish(prepared()).unwrap();
    owner.clear(&mut io).unwrap();
    assert_eq!(io.events, [Event::Free(ADDRESS)]);
    assert!(!owner.has_owner());
    let (mut owner, mut io) = active();
    owner.finish(&mut io).unwrap();
    assert_eq!(io.events, [Event::Free(ADDRESS)]);
    assert!(!owner.has_owner());
}

#[test]
fn active_owner_rejects_prepare_clear_and_stream_switch_without_io() {
    let (mut owner, mut io) = active();
    assert!(owner.clear(&mut io).is_err());
    assert!(owner.publish(prepared()).is_err());
    owner.retry_quarantine(&mut io).unwrap(); // Reset must retain this request's owner.
    owner.arm(STREAM).unwrap();
    owner.arm(STREAM).unwrap();
    assert!(owner.arm(STREAM + 1).is_err());
    assert!(io.events.is_empty());
    assert!(owner.has_owner());
}

#[test]
fn completion_drains_while_source_bytes_live_then_frees_exactly_once() {
    let (mut owner, mut io) = active();
    let expected = io.allocation.clone().unwrap();
    owner.arm(STREAM).unwrap();
    io.pending = true;
    owner.finish(&mut io).unwrap();
    assert_eq!(io.copied, expected);
    assert_eq!(io.events, [Event::Drain(STREAM), Event::Free(ADDRESS)]);
    assert!(!owner.has_owner());
    owner.clear(&mut io).unwrap();
    assert_eq!(io.events.len(), 2);
}

#[test]
fn drain_failure_quarantines_live_bytes_and_blocks_reuse_until_retry_finishes() {
    let (mut owner, mut io) = active();
    let expected = io.allocation.clone().unwrap();
    owner.arm(STREAM).unwrap();
    io.pending = true;
    io.fail_drain = true;
    assert!(owner.finish(&mut io).is_err());
    assert!(owner.is_quarantined() && owner.has_owner());
    assert_eq!(io.events, [Event::Drain(STREAM)]);
    assert_eq!(io.allocation.as_ref().unwrap(), &expected);
    assert!(owner.begin().is_err());
    assert!(owner.publish(prepared()).is_err());
    assert!(owner.arm(STREAM).is_err());
    assert!(owner.clear(&mut io).is_err());
    assert!(
        io.events
            .iter()
            .all(|event| matches!(event, Event::Drain(STREAM)))
    );
    io.fail_drain = false;
    owner.retry_quarantine(&mut io).unwrap();
    assert_eq!(io.copied, expected);
    assert_eq!(io.events.last(), Some(&Event::Free(ADDRESS)));
    assert!(!owner.has_owner() && !owner.is_quarantined());
    assert!(owner.begin().unwrap().is_none());
}

#[test]
fn free_failure_retains_drained_owner_and_retry_does_not_replay_completed_work() {
    let (mut owner, mut io) = active();
    owner.arm(STREAM).unwrap();
    io.pending = true;
    io.fail_free = true;
    assert!(owner.finish(&mut io).is_err());
    assert!(owner.is_quarantined() && owner.has_owner());
    assert!(io.allocation.is_some() && !io.pending);
    assert!(owner.begin().is_err());
    io.fail_free = false;
    owner.retry_quarantine(&mut io).unwrap();
    assert_eq!(
        io.events,
        [
            Event::Drain(STREAM),
            Event::Free(ADDRESS),
            Event::Free(ADDRESS)
        ]
    );
    assert!(!owner.has_owner());
}

#[test]
fn malformed_descriptors_never_become_prepared_owners() {
    for bad in [
        PreparedRows {
            rows: 0,
            ..prepared()
        },
        PreparedRows {
            rows: usize::MAX,
            ..prepared()
        },
        PreparedRows {
            region: InputRegion {
                address: 0,
                bytes: 2 * ROW_BYTES,
            },
            ..prepared()
        },
        PreparedRows {
            region: InputRegion {
                address: ADDRESS + 1,
                bytes: 2 * ROW_BYTES,
            },
            ..prepared()
        },
        PreparedRows {
            region: InputRegion {
                address: ADDRESS,
                bytes: ROW_BYTES,
            },
            ..prepared()
        },
        PreparedRows {
            region: InputRegion {
                address: u64::MAX - 1,
                bytes: 2 * ROW_BYTES,
            },
            ..prepared()
        },
    ] {
        let mut owner = PreparedOwner::new();
        assert!(owner.publish(bad).is_err());
        assert!(!owner.has_owner());
    }
}

#[test]
fn preparation_is_only_published_ready_after_its_stream_drains() {
    let (mut owner, mut io) = active();
    owner.arm(STREAM).unwrap();
    io.pending = true;
    owner.complete_preparation(&mut io).unwrap();
    assert_eq!(io.events, [Event::Drain(STREAM)]);
    assert!(owner.has_owner() && !owner.is_quarantined());
    assert!(io.allocation.is_some());
    assert_eq!(owner.begin().unwrap().unwrap().region.address, ADDRESS);
    owner.finish(&mut io).unwrap();
    assert_eq!(io.events, [Event::Drain(STREAM), Event::Free(ADDRESS)]);
}

#[test]
fn failed_preparation_drain_cannot_be_used_as_ready_input() {
    let (mut owner, mut io) = active();
    owner.arm(STREAM).unwrap();
    io.pending = true;
    io.fail_drain = true;
    assert!(owner.complete_preparation(&mut io).is_err());
    assert!(owner.is_quarantined());
    assert!(owner.begin().is_err());
    assert!(io.allocation.is_some());
    io.fail_drain = false;
    owner.retry_quarantine(&mut io).unwrap();
    assert!(io.allocation.is_none());
}
