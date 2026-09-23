// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::DevicePtr;

use crate::layers::ops::GgmlIqBuffer;
use crate::layers::{Glm53Dflash2RuntimeGeometry, Glm53Dflash2ScratchPlan};

use std::ffi::OsStr;

use super::dflash2_runtime::readback::{ProposalReadback, coalesced_readback_span};

fn buffer(arena: DevicePtr, offset: usize, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: DevicePtr(arena.0 + offset as u64),
        bytes,
    }
}

#[test]
fn coalesced_span_covers_aligned_status_gap_and_complete_path() {
    let plan =
        Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 2047, 8)).unwrap();
    let arena = DevicePtr(0x1000_0000);
    let status = buffer(
        arena,
        plan.selector_status.offset,
        plan.selector_status.bytes,
    );
    let path = buffer(
        arena,
        plan.chosen_ids.offset,
        7 * std::mem::size_of::<u32>(),
    );

    let span = coalesced_readback_span(status, path).unwrap();
    assert_eq!(span.source, status.ptr);
    assert_eq!(span.status_offset, 0);
    assert_eq!(
        span.path_offset,
        plan.chosen_ids.offset - plan.selector_status.offset
    );
    assert_eq!(
        span.bytes,
        span.path_offset + 7 * std::mem::size_of::<u32>()
    );
}

#[test]
fn coalesced_span_rejects_reverse_or_overlapping_layouts() {
    let status = GgmlIqBuffer {
        ptr: DevicePtr(0x2000),
        bytes: 4,
    };
    let reverse = GgmlIqBuffer {
        ptr: DevicePtr(0x1000),
        bytes: 28,
    };
    let overlap = GgmlIqBuffer {
        ptr: DevicePtr(0x2002),
        bytes: 28,
    };

    assert!(coalesced_readback_span(status, reverse).is_err());
    assert!(coalesced_readback_span(status, overlap).is_err());
}

#[test]
fn coalesced_selector_is_strict_and_default_off() {
    assert_eq!(
        ProposalReadback::parse(None).unwrap(),
        ProposalReadback::Separate
    );
    assert_eq!(
        ProposalReadback::parse(Some(OsStr::new("0"))).unwrap(),
        ProposalReadback::Separate
    );
    assert_eq!(
        ProposalReadback::parse(Some(OsStr::new("1"))).unwrap(),
        ProposalReadback::Coalesced
    );
    assert!(ProposalReadback::parse(Some(OsStr::new("true"))).is_err());
    assert!(ProposalReadback::parse(Some(OsStr::new("2"))).is_err());
}
