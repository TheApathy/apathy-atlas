// SPDX-License-Identifier: AGPL-3.0-only
//! P12 RED: family authority shares the real prefix owner and completion API.
#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
#[allow(dead_code)]
mod kv_prefix;
#[path = "../src/model/glm53/dflash2_projection_contract.rs"]
#[allow(dead_code)]
mod projection;
use anyhow::{Result, bail};
use kv_prefix::{KvPrefix, KvPrefixIo, KvTail};
use projection::{ProjectionBinding, ProjectionFamily};

#[derive(Default)]
struct Completion {
    enqueues: Vec<(usize, u32, u64)>,
    drains: Vec<u64>,
    fail_enqueue: bool,
    fail_drain: bool,
    panic_drain: bool,
}
impl KvPrefixIo for Completion {
    fn enqueue_layer(&mut self, layer: usize, tail: KvTail, stream: u64) -> Result<()> {
        self.enqueues.push((layer, tail.committed_end(), stream));
        if self.fail_enqueue {
            bail!("submitted before failure");
        }
        Ok(())
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.drains.push(stream);
        if self.panic_drain {
            panic!("completion panic");
        }
        if self.fail_drain {
            bail!("completion failure");
        }
        Ok(())
    }
}
fn complete(prefix: &mut KvPrefix, context: u32, stream: u64, io: &mut Completion) {
    prefix.begin(context, context, stream, false).unwrap();
    for layer in 0..5 {
        prefix.enqueue_layer(layer, io).unwrap();
    }
    prefix.finish(io).unwrap();
}

#[test]
fn explicit_selection_is_idempotent_and_switch_requires_successful_reset() {
    for (first, other) in [
        (ProjectionFamily::Original, ProjectionFamily::StableTc),
        (ProjectionFamily::StableTc, ProjectionFamily::Original),
    ] {
        let mut prefix = KvPrefix::new(5, 2047).unwrap();
        let mut binding = ProjectionBinding::new();
        assert_eq!(binding.family(), None);
        binding.select(first, &prefix).unwrap();
        binding.select(first, &prefix).unwrap();
        assert!(binding.select(other, &prefix).is_err());
        assert_eq!(binding.family(), Some(first));
        let mut io = Completion::default();
        complete(&mut prefix, 1, 0, &mut io);
        binding.select(first, &prefix).unwrap();
        assert!(binding.select(other, &prefix).is_err());
        binding.reset(&mut prefix, &mut io).unwrap();
        assert_eq!(binding.family(), None);
        for layer in 0..5 {
            assert_eq!(prefix.completed_rows(layer).unwrap(), 0);
        }
        binding.select(other, &prefix).unwrap();
        assert_eq!(binding.family(), Some(other));
    }
}

#[test]
fn unbound_completed_prefix_cannot_be_relabelled_as_either_arithmetic_family() {
    let mut prefix = KvPrefix::new(5, 2047).unwrap();
    let mut io = Completion::default();
    complete(&mut prefix, 17, 3, &mut io);
    let mut binding = ProjectionBinding::new();
    for family in [ProjectionFamily::Original, ProjectionFamily::StableTc] {
        assert!(binding.select(family, &prefix).is_err());
        assert_eq!(binding.family(), None);
    }
    binding.reset(&mut prefix, &mut io).unwrap();
    binding.select(ProjectionFamily::StableTc, &prefix).unwrap();
}

#[test]
fn pending_and_poisoned_prefix_cannot_reselect_even_the_same_family() {
    let mut prefix = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    binding.select(ProjectionFamily::StableTc, &prefix).unwrap();
    prefix.begin(2, 2, 19, false).unwrap();
    assert!(binding.select(ProjectionFamily::StableTc, &prefix).is_err());
    let mut io = Completion {
        fail_enqueue: true,
        ..Completion::default()
    };
    assert!(prefix.enqueue_layer(0, &mut io).is_err());
    assert!(prefix.poisoned());
    prefix.abort(&mut io).unwrap();
    assert!(!prefix.pending());
    assert!(prefix.poisoned());
    assert!(binding.select(ProjectionFamily::StableTc, &prefix).is_err());
    assert_eq!(binding.family(), Some(ProjectionFamily::StableTc));
    binding.reset(&mut prefix, &mut io).unwrap();
    binding.select(ProjectionFamily::Original, &prefix).unwrap();
}

#[test]
fn failed_reset_retains_original_family_pending_stream_and_prior_cursors() {
    let mut prefix = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    binding.select(ProjectionFamily::Original, &prefix).unwrap();
    let mut io = Completion::default();
    complete(&mut prefix, 8, 37, &mut io);
    prefix.begin(16, 16, 37, false).unwrap();
    prefix.enqueue_layer(0, &mut io).unwrap();
    io.fail_drain = true;
    assert!(binding.reset(&mut prefix, &mut io).is_err());
    assert_eq!(binding.family(), Some(ProjectionFamily::Original));
    assert_eq!(prefix.pending_stream(), Some(37));
    assert!(prefix.poisoned());
    for layer in 0..5 {
        assert_eq!(prefix.completed_rows(layer).unwrap(), 8);
    }
    assert!(binding.select(ProjectionFamily::StableTc, &prefix).is_err());
    io.fail_drain = false;
    binding.reset(&mut prefix, &mut io).unwrap();
    assert_eq!(binding.family(), None);
    assert!(!prefix.pending());
    assert_eq!(io.drains, [37, 37, 37]);
    binding.select(ProjectionFamily::StableTc, &prefix).unwrap();
}

#[test]
fn caught_reset_panic_never_clears_the_binding_or_original_stream_authority() {
    let mut prefix = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    binding.select(ProjectionFamily::StableTc, &prefix).unwrap();
    prefix.begin(2047, 2047, 0, false).unwrap();
    let mut io = Completion {
        panic_drain: true,
        ..Completion::default()
    };
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        binding.reset(&mut prefix, &mut io)
    }));
    assert!(failed.is_err());
    assert_eq!(binding.family(), Some(ProjectionFamily::StableTc));
    assert_eq!(prefix.pending_stream(), Some(0));
    assert!(binding.select(ProjectionFamily::Original, &prefix).is_err());
    io.panic_drain = false;
    binding.reset(&mut prefix, &mut io).unwrap();
    assert_eq!(binding.family(), None);
    assert_eq!(io.drains, [0, 0]);
}

#[test]
fn all_layer_completion_remains_required_and_reset_does_not_publish_noise() {
    let mut prefix = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    binding.select(ProjectionFamily::StableTc, &prefix).unwrap();
    let mut io = Completion::default();
    prefix.begin(2047, 2047, 8, false).unwrap();
    for layer in 0..4 {
        prefix.enqueue_layer(layer, &mut io).unwrap();
    }
    assert!(prefix.finish(&mut io).is_err());
    for layer in 0..5 {
        assert_eq!(prefix.completed_rows(layer).unwrap(), 0);
    }
    assert!(binding.select(ProjectionFamily::StableTc, &prefix).is_err());
    binding.reset(&mut prefix, &mut io).unwrap();
    assert_eq!(binding.family(), None);
    assert_eq!(
        io.enqueues,
        [(0, 2047, 8), (1, 2047, 8), (2, 2047, 8), (3, 2047, 8)]
    );
    assert_eq!(io.drains, [8]);
}
