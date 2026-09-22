// SPDX-License-Identifier: AGPL-3.0-only
//! Real allocating CPU backend for the single lazy capture-bank owner.

mod model {
    pub mod glm53 {
        pub use spark_model::model::glm53::GLM53_CAPTURE_LAYERS;
    }
}
mod layers {
    pub use spark_model::layers::Glm53TargetGeometry;
}
#[path = "../src/model/glm53/prefill_capture_owner.rs"]
mod prefill_capture_owner;
#[path = "../src/model/glm53/prefill_capture_plan.rs"]
#[allow(dead_code)] // This target exercises allocation; binding/ingestion have separate targets.
mod prefill_capture_plan;

use anyhow::{Result, bail, ensure};
use prefill_capture_owner::{CaptureBankIo, CaptureBankOwner};
use prefill_capture_plan::PrefillCapturePlan;
use std::collections::BTreeMap;

struct MemoryIo {
    live: BTreeMap<u64, Vec<u8>>,
    events: Vec<&'static str>,
    next: u64,
    peak: usize,
    fail_allocate: bool,
    fail_drain: bool,
    fail_free: bool,
}

impl MemoryIo {
    fn new() -> Self {
        Self {
            live: BTreeMap::new(),
            events: Vec::new(),
            next: 0x1000_0000,
            peak: 0,
            fail_allocate: false,
            fail_drain: false,
            fail_free: false,
        }
    }
}

impl CaptureBankIo for MemoryIo {
    fn allocate(&mut self, bytes: usize) -> Result<u64> {
        self.events.push("allocate");
        if self.fail_allocate {
            bail!("allocation failed");
        }
        ensure!(bytes > 0 && bytes <= 80 * 1024 * 1024);
        ensure!(self.live.is_empty(), "must never own two capture banks");
        let address = self.next;
        self.next += 0x1000_0000;
        self.live.insert(address, vec![0; bytes]);
        self.peak = self.peak.max(self.live.values().map(Vec::len).sum());
        Ok(address)
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.events.push("drain");
        ensure!(stream == 7, "wrong capture stream");
        ensure!(!self.fail_drain, "drain failed");
        Ok(())
    }
    fn free(&mut self, address: u64) -> Result<()> {
        self.events.push("free");
        ensure!(!self.fail_free, "free failed");
        ensure!(
            self.live.remove(&address).is_some(),
            "unknown or duplicate free"
        );
        Ok(())
    }
}

fn plan(capacity: u32) -> PrefillCapturePlan {
    PrefillCapturePlan::new(capacity, capacity.min(2047), 0, 0, 2047, 8192).unwrap()
}

#[test]
fn allocation_is_lazy_and_reused_only_after_actual_completion() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    assert!(!owner.has_owner());
    owner.release(&mut io).unwrap();
    assert!(io.events.is_empty());
    let bank = owner.begin(&mut io, &plan(32), 7).unwrap();
    assert_eq!(bank.bytes, 32 * 5 * 8192);
    io.live.get_mut(&bank.address).unwrap()[0] = 0xa7;
    assert!(owner.begin(&mut io, &plan(16), 7).is_err());
    assert_eq!(io.events, ["allocate"]);
    owner.complete(&mut io).unwrap();
    let reused = owner.begin(&mut io, &plan(16), 7).unwrap();
    assert_eq!(reused, bank);
    assert_eq!(io.live[&bank.address][0], 0xa7);
    owner.complete(&mut io).unwrap();
    owner.release(&mut io).unwrap();
    assert_eq!(io.events, ["allocate", "drain", "drain", "free"]);
    assert!(!owner.has_owner());
}

#[test]
fn growth_releases_old_storage_before_allocating_and_never_exceeds_eighty_mib() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    let first = owner.begin(&mut io, &plan(16), 7).unwrap();
    owner.complete(&mut io).unwrap();
    let max = owner.begin(&mut io, &plan(2048), 7).unwrap();
    assert_ne!(first.address, max.address);
    assert_eq!(max.bytes, 80 * 1024 * 1024);
    assert_eq!(io.peak, 80 * 1024 * 1024);
    assert_eq!(io.events, ["allocate", "drain", "free", "allocate"]);
    owner.complete(&mut io).unwrap();
    owner.release(&mut io).unwrap();
}

#[test]
fn failed_completion_and_failed_release_drain_keep_the_same_owner() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    let bank = owner.begin(&mut io, &plan(32), 7).unwrap();
    io.fail_drain = true;
    assert!(owner.complete(&mut io).is_err());
    assert!(owner.is_poisoned());
    assert!(owner.has_owner());
    assert!(owner.begin(&mut io, &plan(16), 7).is_err());
    assert!(owner.release(&mut io).is_err());
    assert!(io.live.contains_key(&bank.address));
    assert!(!io.events.contains(&"free"));
    io.fail_drain = false;
    owner.release(&mut io).unwrap();
    assert_eq!(&io.events[io.events.len() - 2..], ["drain", "free"]);
    assert!(!owner.has_owner());
}

#[test]
fn abort_drains_but_stays_poisoned_and_does_not_claim_byte_rollback() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    let bank = owner.begin(&mut io, &plan(32), 7).unwrap();
    io.live.get_mut(&bank.address).unwrap()[512] = 0x5a;
    owner.abort(&mut io).unwrap();
    assert!(owner.is_poisoned());
    assert_eq!(io.live[&bank.address][512], 0x5a);
    let before = io.events.len();
    assert!(owner.complete(&mut io).is_err());
    assert!(owner.begin(&mut io, &plan(32), 7).is_err());
    assert_eq!(io.events.len(), before);
    owner.release(&mut io).unwrap();
    assert!(io.live.is_empty());
}

#[test]
fn failed_free_retains_owner_and_retries_without_repeating_a_successful_fence() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    let bank = owner.begin(&mut io, &plan(16), 7).unwrap();
    io.fail_free = true;
    assert!(owner.release(&mut io).is_err());
    assert!(owner.is_poisoned());
    assert!(io.live.contains_key(&bank.address));
    let drains = io.events.iter().filter(|event| **event == "drain").count();
    io.fail_free = false;
    owner.release(&mut io).unwrap();
    assert_eq!(
        io.events.iter().filter(|event| **event == "drain").count(),
        drains
    );
    assert!(io.live.is_empty());
    owner.release(&mut io).unwrap(); // No duplicate free.
}

#[test]
fn failed_growth_cannot_lose_old_owner_or_allocate_a_second_bank() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    let bank = owner.begin(&mut io, &plan(16), 7).unwrap();
    owner.complete(&mut io).unwrap();
    io.fail_free = true;
    assert!(owner.begin(&mut io, &plan(32), 7).is_err());
    assert!(owner.is_poisoned());
    assert!(io.live.contains_key(&bank.address));
    assert_eq!(
        io.events
            .iter()
            .filter(|event| **event == "allocate")
            .count(),
        1
    );
}

#[test]
fn failed_allocation_is_empty_but_invalid_returned_allocation_remains_owned() {
    let mut owner = CaptureBankOwner::new();
    let mut io = MemoryIo::new();
    io.fail_allocate = true;
    assert!(owner.begin(&mut io, &plan(16), 7).is_err());
    assert!(!owner.has_owner());
    io.fail_allocate = false;
    io.next += 1; // Allocator returns owned but misaligned storage.
    assert!(owner.begin(&mut io, &plan(16), 7).is_err());
    assert!(owner.has_owner());
    assert!(owner.is_poisoned());
    assert_eq!(io.live.len(), 1);
    owner.release(&mut io).unwrap();
    assert!(io.live.is_empty());
}
