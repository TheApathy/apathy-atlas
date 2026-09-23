// SPDX-License-Identifier: AGPL-3.0-only
//! New family exercises the real shared cursor/publication contract.
#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
#[allow(dead_code)]
mod kv_prefix;
#[path = "../src/model/glm53/dflash2_projection_contract.rs"]
#[allow(dead_code)]
mod projection_contract;
use anyhow::{Result, bail};
use kv_prefix::{KvPrefix, KvPrefixIo, KvTail};
use projection_contract::{ProjectionBinding, ProjectionFamily as Family};

fn families() -> [Family; 3] {
    [Family::Original, Family::StableTc, Family::StableGemv]
}
#[derive(Default)]
struct Completion {
    drains: Vec<u64>,
    fail_layer: Option<usize>,
    fail_drain: bool,
    panic_drain: bool,
}
impl KvPrefixIo for Completion {
    fn enqueue_layer(&mut self, layer: usize, _: KvTail, _: u64) -> Result<()> {
        if self.fail_layer == Some(layer) {
            bail!("submitted projection failure");
        }
        Ok(())
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.drains.push(stream);
        assert!(!self.panic_drain, "injected completion panic");
        if self.fail_drain {
            bail!("injected completion failure");
        }
        Ok(())
    }
}
fn complete(prefix: &mut KvPrefix, rows: u32, io: &mut Completion) {
    prefix.begin(rows, rows, 37, false).unwrap();
    for layer in 0..5 {
        prefix.enqueue_layer(layer, io).unwrap();
    }
    prefix.finish(io).unwrap();
}

#[test]
fn every_pair_of_families_requires_reset_and_admission_does_not_bind() {
    for first in families() {
        for next in families() {
            let mut p = KvPrefix::new(5, 2047).unwrap();
            let mut binding = ProjectionBinding::new();
            let mut io = Completion::default();
            binding.admit(first, &p).unwrap();
            assert_eq!(binding.family(), None);
            binding.select(first, &p).unwrap();
            complete(&mut p, 17, &mut io);
            assert_eq!(binding.admit(next, &p).is_ok(), first == next);
            assert_eq!(binding.select(next, &p).is_ok(), first == next);
            assert_eq!(binding.family(), Some(first));
            binding.reset(&mut p, &mut io).unwrap();
            assert_eq!(binding.family(), None);
            assert!(!p.has_completed_rows());
            binding.select(next, &p).unwrap();
            assert_eq!(binding.family(), Some(next));
        }
    }
}

#[test]
fn unbound_or_pending_prefix_cannot_claim_any_gemv_family_authority() {
    let mut p = KvPrefix::new(5, 2047).unwrap();
    let mut io = Completion::default();
    complete(&mut p, 8, &mut io);
    let mut binding = ProjectionBinding::new();
    for family in families() {
        assert!(binding.select(family, &p).is_err());
    }
    binding.reset(&mut p, &mut io).unwrap();
    binding.select(Family::StableGemv, &p).unwrap();
    p.begin(16, 16, 37, false).unwrap();
    for family in families() {
        assert!(binding.admit(family, &p).is_err());
    }
}

#[test]
fn failed_layer_and_failed_fence_retain_previous_cursor_family_and_stream() {
    let mut p = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    let mut io = Completion::default();
    binding.select(Family::StableGemv, &p).unwrap();
    complete(&mut p, 8, &mut io);
    p.begin(16, 16, 37, false).unwrap();
    p.enqueue_layer(0, &mut io).unwrap();
    io.fail_layer = Some(1);
    assert!(p.enqueue_layer(1, &mut io).is_err());
    assert!(p.finish(&mut io).is_err());
    io.fail_drain = true;
    assert!(binding.reset(&mut p, &mut io).is_err());
    assert_eq!(binding.family(), Some(Family::StableGemv));
    assert_eq!(p.pending_stream(), Some(37));
    assert!(p.poisoned());
    for layer in 0..5 {
        assert_eq!(p.completed_rows(layer).unwrap(), 8);
    }
    for family in families() {
        assert!(binding.select(family, &p).is_err());
    }
    io.fail_drain = false;
    binding.reset(&mut p, &mut io).unwrap();
    assert_eq!(binding.family(), None);
    assert!(!p.pending() && !p.poisoned() && !p.has_completed_rows());
    assert_eq!(io.drains, [37, 37, 37]);
}

#[test]
fn reset_panic_retains_gemv_binding_and_incomplete_layers_never_publish() {
    let mut p = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    let mut io = Completion::default();
    binding.select(Family::StableGemv, &p).unwrap();
    p.begin(2047, 2047, 0, false).unwrap();
    for layer in 0..4 {
        p.enqueue_layer(layer, &mut io).unwrap();
    }
    assert!(p.finish(&mut io).is_err());
    assert!(!p.has_completed_rows());
    io.panic_drain = true;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || binding.reset(&mut p, &mut io)
        ))
        .is_err()
    );
    assert_eq!(binding.family(), Some(Family::StableGemv));
    assert_eq!(p.pending_stream(), Some(0));
    io.panic_drain = false;
    binding.reset(&mut p, &mut io).unwrap();
    assert_eq!(binding.family(), None);
    assert_eq!(io.drains, [0, 0]);
}
