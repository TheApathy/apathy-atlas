// SPDX-License-Identifier: AGPL-3.0-only

mod layers {
    pub use spark_model::layers::glm53_dsa_t1_transaction;
    pub use spark_model::layers::ops;
}
#[allow(dead_code)] // Capture admission is exercised by the dedicated plan target.
#[path = "../src/model/glm53/dsa_verify_plan.rs"]
mod dsa_verify_plan;
#[path = "../src/model/glm53/dsa_verify_transaction.rs"]
mod dsa_verify_transaction;

use anyhow::{Result, bail};
use dsa_verify_plan::{DsaVerifyPlan, MAX_BACKUP_BYTES};
use dsa_verify_transaction::{CopyIo, DeviceRegion, VerifySnapshot};
use std::collections::BTreeMap;

#[derive(Default)]
struct MemoryIo {
    bytes: BTreeMap<u64, Vec<u8>>,
    copies: usize,
    drains: usize,
    fail_copy: Option<usize>,
    fail_drain: bool,
    streams: Vec<u64>,
}

impl MemoryIo {
    fn allocate(&mut self, count: usize, tag: u8) -> DeviceRegion {
        let address = (self.bytes.len() as u64 + 1) * 0x10_0000;
        self.bytes.insert(address, vec![tag; count]);
        DeviceRegion {
            address,
            bytes: count,
        }
    }

    fn location(&self, address: u64, count: usize) -> Result<(u64, usize)> {
        for (&base, bytes) in &self.bytes {
            if let Some(offset) = address
                .checked_sub(base)
                .and_then(|v| usize::try_from(v).ok())
            {
                if offset
                    .checked_add(count)
                    .is_some_and(|end| end <= bytes.len())
                {
                    return Ok((base, offset));
                }
            }
        }
        bail!("out-of-bounds test I/O")
    }
}

impl CopyIo for MemoryIo {
    fn copy(&mut self, source: u64, destination: u64, count: usize, stream: u64) -> Result<()> {
        self.copies += 1;
        self.streams.push(stream);
        if self.fail_copy == Some(self.copies) {
            bail!("injected copy failure");
        }
        let (base, offset) = self.location(source, count)?;
        let saved = self.bytes[&base][offset..offset + count].to_vec();
        let (base, offset) = self.location(destination, count)?;
        self.bytes.get_mut(&base).unwrap()[offset..offset + count].copy_from_slice(&saved);
        Ok(())
    }

    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.drains += 1;
        self.streams.push(stream);
        if self.fail_drain {
            bail!("injected stream failure");
        }
        Ok(())
    }
}

fn fixture() -> (MemoryIo, Vec<[DeviceRegion; 6]>, DeviceRegion) {
    let mut io = MemoryIo::default();
    let mut layers = Vec::new();
    for layer in 0..11 {
        layers
            .push([32 * 1024, 8 * 256, 8, 768, 768, 3].map(|bytes| io.allocate(bytes, layer + 1)));
    }
    let backup = io.allocate(MAX_BACKUP_BYTES, 0xe7);
    (io, layers, backup)
}

#[test]
fn actual_copy_restore_preserves_every_byte_at_all_widths_and_residues() {
    for position in 0..8 {
        for rows in 1..=8 {
            let (mut io, layers, backup) = fixture();
            let before = io.bytes.clone();
            let plan = DsaVerifyPlan::new(position, rows, 32).unwrap();
            let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 73).unwrap();
            snapshot.save(&mut io).unwrap();
            for layer in &layers {
                for (region, span) in layer.iter().zip(plan.source_regions()) {
                    io.bytes.get_mut(&region.address).unwrap()
                        [span.offset..span.offset + span.bytes]
                        .fill(0x99);
                }
            }
            snapshot.restore(&mut io).unwrap();
            for layer in &layers {
                for region in layer {
                    assert_eq!(io.bytes[&region.address], before[&region.address]);
                }
            }
            assert!(!snapshot.needs_reset());
            assert!(io.streams.iter().all(|&stream| stream == 73));
            assert_eq!(io.drains, 1);
            assert!(snapshot.restore(&mut io).is_err());
        }
    }
}

#[test]
fn partial_save_failure_drains_without_touching_target_or_claiming_a_snapshot() {
    for failure in 1..=66 {
        let (mut io, layers, backup) = fixture();
        let before = io.bytes.clone();
        let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
        let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 17).unwrap();
        io.fail_copy = Some(failure);
        assert!(snapshot.save(&mut io).is_err());
        assert_eq!(io.drains, 1);
        assert!(!snapshot.needs_reset());
        let count = io.copies;
        assert!(snapshot.restore(&mut io).is_err());
        assert_eq!(io.copies, count);
        for layer in &layers {
            for region in layer {
                assert_eq!(io.bytes[&region.address], before[&region.address]);
            }
        }
    }
}

#[test]
fn restore_or_drain_failure_poison_and_never_allow_commit() {
    for failure in 1..=66 {
        let (mut io, layers, backup) = fixture();
        let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
        let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 9).unwrap();
        snapshot.save(&mut io).unwrap();
        io.fail_copy = Some(io.copies + failure);
        assert!(snapshot.restore(&mut io).is_err());
        assert!(snapshot.needs_reset());
        assert!(snapshot.begin_commit().is_err());
        assert_eq!(io.drains, 1);
    }
    let (mut io, layers, backup) = fixture();
    let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
    let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 9).unwrap();
    io.fail_copy = Some(2);
    io.fail_drain = true;
    assert!(snapshot.save(&mut io).is_err());
    assert!(snapshot.needs_reset());
}

#[test]
fn persistent_commit_is_irreversible_and_requires_final_stream_success() {
    let (mut io, layers, backup) = fixture();
    let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
    let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 5).unwrap();
    assert!(snapshot.begin_commit().is_err());
    snapshot.save(&mut io).unwrap();
    snapshot.begin_commit().unwrap();
    assert!(snapshot.restore(&mut io).is_err());
    io.fail_drain = true;
    assert!(snapshot.finish(&mut io).is_err());
    assert!(snapshot.needs_reset());
}

#[test]
fn full_and_replayed_commit_complete_once_but_explicit_commit_failure_is_sticky() {
    for replay in [false, true] {
        let (mut io, layers, backup) = fixture();
        let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
        let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 5).unwrap();
        snapshot.save(&mut io).unwrap();
        if replay {
            snapshot.restore(&mut io).unwrap();
        }
        snapshot.begin_commit().unwrap();
        snapshot.finish(&mut io).unwrap();
        assert!(!snapshot.needs_reset());
        assert!(snapshot.begin_commit().is_err());
    }
    let (mut io, layers, backup) = fixture();
    let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
    let mut snapshot = VerifySnapshot::bind(&plan, &layers, backup, 5).unwrap();
    snapshot.save(&mut io).unwrap();
    snapshot.begin_commit().unwrap();
    snapshot.fail_commit(&mut io).unwrap();
    assert!(snapshot.needs_reset());
    assert_eq!(io.drains, 1);
    assert!(snapshot.finish(&mut io).is_err());
}

#[test]
fn malformed_bindings_fail_before_any_copy() {
    let (io, layers, backup) = fixture();
    let plan = DsaVerifyPlan::new(3, 8, 32).unwrap();
    assert!(VerifySnapshot::bind(&plan, &layers[..10], backup, 1).is_err());
    for bad in [
        DeviceRegion {
            address: 0,
            ..backup
        },
        DeviceRegion { bytes: 1, ..backup },
        DeviceRegion {
            address: u64::MAX - 2,
            ..backup
        },
        layers[0][0],
    ] {
        assert!(VerifySnapshot::bind(&plan, &layers, bad, 1).is_err());
    }
    let mut bad = layers.clone();
    bad[10][0].bytes = 1024;
    assert!(VerifySnapshot::bind(&plan, &bad, backup, 1).is_err());
    bad = layers.clone();
    bad[10][5] = bad[0][5];
    assert!(VerifySnapshot::bind(&plan, &bad, backup, 1).is_err());
    assert_eq!(io.copies, 0);
}
