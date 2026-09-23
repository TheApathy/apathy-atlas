// SPDX-License-Identifier: AGPL-3.0-only
//! Real host-owner and consuming-release failure boundaries; no CUDA runtime.

#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
// The complete cursor API is exercised by the separate prefix integration test;
// this executable only exercises the shared production host-release owner.
#[allow(dead_code)]
mod kv_prefix;
#[path = "../src/model/glm53/dflash2_kv_prefix_owner.rs"]
mod owner;
#[path = "../src/model/glm53/dflash2_projection_contract.rs"]
#[allow(dead_code)]
mod projection_contract;

use anyhow::{Result, bail};
use kv_prefix::{KvPrefix, KvPrefixIo, KvTail};
use owner::{KvPrefixState, retain_until_drained};
use projection_contract::ProjectionFamily;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Mutex;

struct DropOwner {
    drops: Rc<Cell<usize>>,
    backend_lease: Rc<()>,
    bytes: Box<[u8; 36]>,
}
impl Drop for DropOwner {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}
fn owned(drops: &Rc<Cell<usize>>) -> DropOwner {
    DropOwner {
        drops: drops.clone(),
        backend_lease: Rc::new(()),
        bytes: Box::new([73; 36]),
    }
}

#[test]
fn completed_drain_returns_exact_owner_and_only_then_allows_drop() {
    let drops = Rc::new(Cell::new(0));
    let initial = owned(&drops);
    let address = initial.bytes.as_ptr();
    let owner = retain_until_drained(initial, |value| {
        assert_eq!(drops.get(), 0);
        assert_eq!(value.bytes.as_ptr(), address);
        assert_eq!(value.bytes.as_ref(), &[73; 36]);
        Ok(())
    })
    .unwrap();
    assert_eq!(owner.bytes.as_ptr(), address);
    assert_eq!(drops.get(), 0);
    drop(owner);
    assert_eq!(drops.get(), 1);
}

#[test]
fn failed_drain_quarantines_host_and_backend_lease_without_false_success() {
    let drops = Rc::new(Cell::new(0));
    let initial = owned(&drops);
    let lease = Rc::downgrade(&initial.backend_lease);
    let error = match retain_until_drained(initial, |_| bail!("injected fence failure")) {
        Ok(_) => panic!("failed drain returned a releasable owner"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("injected fence failure"));
    assert!(format!("{error:#}").contains("quarantined"));
    assert_eq!(drops.get(), 0);
    assert_eq!(lease.strong_count(), 1);
}

#[test]
fn panicking_drain_cannot_unwind_through_the_consuming_owner() {
    let drops = Rc::new(Cell::new(0));
    let initial = owned(&drops);
    let lease = Rc::downgrade(&initial.backend_lease);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        retain_until_drained(initial, |_| panic!("injected completion panic"))
    }));
    let error = match result.expect("release boundary must catch completion panic") {
        Ok(_) => panic!("panicking drain returned owner"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("panicked"));
    assert_eq!(drops.get(), 0);
    assert_eq!(lease.strong_count(), 1);
}

struct HostCompletion {
    path: *mut u8,
    fail: bool,
    streams: Vec<u64>,
}
impl KvPrefixIo for HostCompletion {
    fn enqueue_layer(&mut self, _: usize, _: KvTail, _: u64) -> Result<()> {
        Ok(())
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.streams.push(stream);
        if self.fail {
            bail!("deferred host copy still pending");
        }
        // The test retains the actual production heap owner across the failed
        // fence and poisoned mutex. No write occurs until a successful fence.
        unsafe {
            std::ptr::copy_nonoverlapping([9u8; 28].as_ptr(), self.path, 28);
        }
        Ok(())
    }
}

#[test]
fn caught_io_panic_keeps_stable_host_bytes_through_failed_then_successful_reset() {
    let state = KvPrefixState::new(KvPrefix::new(1, 2047).unwrap());
    let mutex = Mutex::new(state);
    let address = {
        let mut state = mutex.lock().unwrap();
        state.admit_projection(ProjectionFamily::StableTc).unwrap();
        state.select_projection(ProjectionFamily::StableTc).unwrap();
        assert!(state.admit_projection(ProjectionFamily::Original).is_err());
        let host = state.host.as_mut().unwrap();
        host.anchor = 31u32.to_le_bytes();
        host.path.fill(7);
        host.status = 0u32.to_le_bytes();
        host.path.as_mut_ptr()
    };
    let mut io = HostCompletion {
        path: address,
        fail: true,
        streams: Vec::new(),
    };
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut state = mutex.lock().unwrap();
        state.prefix.begin(31, 31, 0, false).unwrap();
        state.prefix.enqueue_layer(0, &mut io).unwrap();
        panic!("copy enqueued before panic");
    }));
    assert!(panic.is_err());
    let mut state = mutex.lock().unwrap_or_else(|e| e.into_inner());
    assert!(state.prefix.pending());
    assert!(state.prefix.begin(31, 31, 0, false).is_err());
    assert!(state.reset(&mut io).is_err());
    assert!(state.admit_projection(ProjectionFamily::Original).is_err());
    assert_eq!(
        state.host.as_ref().unwrap().path.as_ptr(),
        address as *const u8
    );
    assert_eq!(state.host.as_ref().unwrap().path, [7; 28]);
    io.fail = false;
    state.reset(&mut io).unwrap();
    state.admit_projection(ProjectionFamily::Original).unwrap();
    assert_eq!(io.streams, [0, 0]);
    assert_eq!(state.host.as_ref().unwrap().path, [9; 28]);
    assert_eq!(state.host.as_ref().unwrap().anchor, 31u32.to_le_bytes());
    assert_eq!(state.host.as_ref().unwrap().status, [0; 4]);
    assert!(!state.prefix.pending());
    assert_eq!(state.prefix.completed_rows(0).unwrap(), 0);
}
