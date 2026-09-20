// SPDX-License-Identifier: AGPL-3.0-only
//! Owned host readback lifecycle; no CUDA, model weights, or foreign execution.
#[path = "../src/model/glm53/owned_verify_readback.rs"]
mod owned_verify_readback;
use anyhow::{Result, bail};
use owned_verify_readback::{OwnedReadback, ReadbackIo};
use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// Watch deallocation without dereferencing a possibly freed pointer. Only the
// final test sets WATCHED; other tests cannot share its still-live allocation.
struct AllocationWatch;
static WATCHED: AtomicUsize = AtomicUsize::new(0);
static WAS_FREED: AtomicBool = AtomicBool::new(false);
#[global_allocator]
static ALLOCATOR: AllocationWatch = AllocationWatch;
// SAFETY: all allocation operations delegate unchanged to System; the observer
// uses allocation-free atomics and never reads or writes allocation contents.
unsafe impl GlobalAlloc for AllocationWatch {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: preserve GlobalAlloc's caller-provided allocation contract.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if WATCHED.load(Ordering::SeqCst) == ptr as usize {
            WAS_FREED.store(true, Ordering::SeqCst);
        }
        // SAFETY: delegate the unchanged pointer and layout to its allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[derive(Default)]
struct Io {
    calls: Vec<(&'static str, u64, usize)>,
    copy_error: bool,
    drain_error: bool,
    panic_copy: bool,
    panic_drain: bool,
    last_ptr: usize,
    generation: u8,
}
impl ReadbackIo for Io {
    fn copy(&mut self, dst: &mut [u8], stream: u64) -> Result<()> {
        self.calls.push(("copy", stream, dst.len()));
        self.last_ptr = dst.as_ptr() as usize;
        for (i, byte) in dst.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(self.generation);
        }
        assert!(!self.panic_copy, "copy panicked after submission");
        if self.copy_error {
            bail!("copy failed after submission");
        }
        Ok(())
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.calls.push(("drain", stream, 0));
        assert!(!self.panic_drain, "drain panicked");
        if self.drain_error {
            bail!("stream completion failed");
        }
        Ok(())
    }
}

#[test]
fn invalid_capacity_and_read_bounds_fail_before_io() {
    for max in [0, 1, 3, usize::MAX, usize::MAX - 1] {
        assert!(OwnedReadback::new(max).is_err(), "invalid capacity {max}");
    }
    let mut owner = OwnedReadback::new(64).unwrap();
    let mut io = Io::default();
    for bytes in [0, 1, 3, 65, 66, usize::MAX] {
        assert!(owner.read(bytes, 19, &mut io).is_err());
        assert!(!owner.pending());
    }
    assert!(io.calls.is_empty());
}

#[test]
fn copies_every_full_row_then_exact_single_row_and_scalar_extents() {
    let mut owner = OwnedReadback::new(64).unwrap();
    let mut io = Io::default();
    // Four distinct BF16 rows with vocabulary7; catching an accidental first-row copy.
    let full = owner.read(4 * 7 * 2, 91, &mut io).unwrap();
    assert_eq!(full, &(0u8..56).collect::<Vec<_>>());
    assert_eq!(&full[42..56], &(42u8..56).collect::<Vec<_>>());
    assert!(!owner.pending());
    io.generation = 100;
    let row = owner.read(7 * 2, 92, &mut io).unwrap();
    assert_eq!(row, &(100u8..114).collect::<Vec<_>>());
    let scalar = owner.read(4, 0, &mut io).unwrap();
    assert_eq!(scalar, [100, 101, 102, 103]);
    assert_eq!(
        io.calls,
        [
            ("copy", 91, 56),
            ("drain", 91, 0),
            ("copy", 92, 14),
            ("drain", 92, 0),
            ("copy", 0, 4),
            ("drain", 0, 0),
        ]
    );
}

#[test]
fn failed_copy_still_drains_and_cannot_publish_success() {
    let mut owner = OwnedReadback::new(32).unwrap();
    let mut io = Io {
        copy_error: true,
        ..Io::default()
    };
    let error = owner.read(16, 77, &mut io).unwrap_err();
    assert!(error.to_string().contains("copy failed after submission"));
    assert_eq!(io.calls, [("copy", 77, 16), ("drain", 77, 0)]);
    assert!(!owner.pending());
    io.copy_error = false;
    io.generation = 20;
    assert_eq!(
        owner.read(8, 78, &mut io).unwrap(),
        &(20u8..28).collect::<Vec<_>>()
    );
}

#[test]
fn failed_completion_retains_buffer_and_blocks_all_read_reuse() {
    let mut owner = OwnedReadback::new(64).unwrap();
    let mut io = Io {
        drain_error: true,
        ..Io::default()
    };
    let error = owner.read(16, 123, &mut io).unwrap_err();
    assert!(error.to_string().contains("stream completion failed"));
    assert!(owner.pending());
    let prior = io.calls.clone();
    for (bytes, stream) in [(16, 123), (64, 999), (0, 0)] {
        assert!(owner.read(bytes, stream, &mut io).is_err());
    }
    assert_eq!(io.calls, prior);
    assert!(owner.pending());
    io.drain_error = false;
    owner.drain(&mut io).unwrap();
    assert_eq!(io.calls.last(), Some(&("drain", 123, 0)));
    assert!(!owner.pending());
}

#[test]
fn both_failures_remain_visible_and_recovery_drains_only_the_original_stream() {
    let mut owner = OwnedReadback::new(32).unwrap();
    let mut io = Io {
        copy_error: true,
        drain_error: true,
        ..Io::default()
    };
    let error = format!("{:#}", owner.read(8, 456, &mut io).unwrap_err());
    assert!(error.contains("copy failed after submission"));
    assert!(error.contains("stream completion failed"));
    assert!(owner.drain(&mut io).is_err());
    assert!(owner.pending());
    assert_eq!(
        io.calls,
        [("copy", 456, 8), ("drain", 456, 0), ("drain", 456, 0)]
    );
    io.copy_error = false;
    io.drain_error = false;
    owner.drain(&mut io).unwrap();
    assert!(!owner.pending());
    assert_eq!(owner.read(2, 789, &mut io).unwrap(), [0, 1]);
    assert_eq!(
        &io.calls[3..],
        [("drain", 456, 0), ("copy", 789, 2), ("drain", 789, 0)]
    );
}

#[test]
fn caught_copy_panic_leaves_pending_before_any_possible_submission() {
    let mut owner = OwnedReadback::new(32).unwrap();
    let mut io = Io {
        panic_copy: true,
        ..Io::default()
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = owner.read(16, 321, &mut io);
        }))
        .is_err()
    );
    assert!(owner.pending());
    let prior = io.calls.clone();
    assert!(owner.read(32, 654, &mut io).is_err());
    assert_eq!(io.calls, prior);
    io.panic_copy = false;
    owner.drain(&mut io).unwrap();
    assert_eq!(io.calls.last(), Some(&("drain", 321, 0)));
    assert!(!owner.pending());
    assert_eq!(owner.read(2, 654, &mut io).unwrap(), [0, 1]);
}

#[test]
fn caught_drain_panic_also_preserves_pending_for_explicit_recovery() {
    let mut owner = OwnedReadback::new(8).unwrap();
    let mut io = Io {
        panic_drain: true,
        ..Io::default()
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = owner.read(8, 222, &mut io);
        }))
        .is_err()
    );
    assert!(owner.pending());
    io.panic_drain = false;
    owner.drain(&mut io).unwrap();
    assert_eq!(
        io.calls,
        [("copy", 222, 8), ("drain", 222, 0), ("drain", 222, 0)]
    );
    assert!(!owner.pending());
}

#[test]
fn draining_idle_owner_is_a_noop() {
    let mut owner = OwnedReadback::new(8).unwrap();
    let mut io = Io::default();
    owner.drain(&mut io).unwrap();
    assert!(io.calls.is_empty());
    owner.read(2, 10, &mut io).unwrap();
    let prior = io.calls.clone();
    owner.drain(&mut io).unwrap();
    assert_eq!(io.calls, prior);
}

#[test]
fn drop_frees_completed_storage_but_retains_pending_allocation() {
    for pending in [false, true] {
        let mut owner = OwnedReadback::new(32).unwrap();
        let mut io = Io {
            drain_error: pending,
            ..Io::default()
        };
        assert_eq!(owner.read(16, 99, &mut io).is_err(), pending);
        assert_ne!(io.last_ptr, 0);
        WAS_FREED.store(false, Ordering::SeqCst);
        WATCHED.store(io.last_ptr, Ordering::SeqCst);
        drop(owner);
        let freed = WAS_FREED.load(Ordering::SeqCst);
        WATCHED.store(0, Ordering::SeqCst);
        assert_eq!(
            freed, !pending,
            "Drop must retain only the uncertain allocation"
        );
    }
}
