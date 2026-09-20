// SPDX-License-Identifier: AGPL-3.0-only
//! Startup selection is policy; completed KV authority remains in its real owner.
#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
#[allow(dead_code)]
mod kv_prefix;
#[path = "../src/model/glm53/dflash2_projection_contract.rs"]
#[allow(dead_code)]
mod projection_contract;
#[path = "../src/model/glm53/dflash2_serving_contract.rs"]
#[allow(dead_code)]
mod serving_contract;

use anyhow::{Result, bail};
use kv_prefix::{KvPrefix, KvPrefixIo, KvTail};
use projection_contract::{ProjectionBinding, ProjectionFamily};
use serving_contract::ServingProjection;
use std::ffi::OsStr;

fn value(text: &str) -> Option<&OsStr> {
    Some(OsStr::new(text))
}

#[test]
fn original_default_and_explicit_original_preserve_both_existing_prefix_choices() {
    for prefix in [false, true] {
        for spelling in [None, value("original")] {
            let choice = ServingProjection::parse(spelling, prefix).unwrap();
            assert_eq!(choice, ServingProjection::Original);
            assert_eq!(choice.family(), ProjectionFamily::Original);
            assert_eq!(choice.as_str(), "original");
        }
    }
}

#[test]
fn stable_gemv_is_explicit_requires_prefix_and_maps_to_the_existing_family() {
    assert!(ServingProjection::parse(value("stable-gemv"), false).is_err());
    let choice = ServingProjection::parse(value("stable-gemv"), true).unwrap();
    assert_eq!(choice, ServingProjection::StableGemv);
    assert_eq!(choice.family(), ProjectionFamily::StableGemv);
    assert_eq!(choice.as_str(), "stable-gemv");
}

#[test]
fn malformed_values_never_silently_choose_original_or_a_diagnostic_family() {
    for spelling in [
        "",
        "0",
        "1",
        "gemv",
        "stable_gemv",
        "StableGemv",
        "tc",
        "stable-tc",
        " original",
        "original ",
        "stable-gemv\n",
        "original\0",
    ] {
        for prefix in [false, true] {
            assert!(
                ServingProjection::parse(value(spelling), prefix).is_err(),
                "{spelling:?}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_input_is_rejected_without_mutating_process_environment() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let malformed = OsString::from_vec(vec![0xff, b'g']);
    for prefix in [false, true] {
        assert!(ServingProjection::parse(Some(&malformed), prefix).is_err());
        assert!(
            ServingProjection::Original
                .admit_current(Some(&malformed), prefix)
                .is_err()
        );
        assert!(
            ServingProjection::StableGemv
                .admit_current(Some(&malformed), prefix)
                .is_err()
        );
    }
}

#[test]
fn startup_choice_refuses_later_family_change_instead_of_reselecting() {
    let original = ServingProjection::parse(None, false).unwrap();
    for prefix in [false, true] {
        original.admit_current(None, prefix).unwrap();
        original.admit_current(value("original"), prefix).unwrap();
        assert!(
            original
                .admit_current(value("stable-gemv"), prefix)
                .is_err()
        );
    }
    let gemv = ServingProjection::parse(value("stable-gemv"), true).unwrap();
    gemv.admit_current(value("stable-gemv"), true).unwrap();
    for prefix in [false, true] {
        assert!(gemv.admit_current(None, prefix).is_err());
        assert!(gemv.admit_current(value("original"), prefix).is_err());
    }
    assert!(gemv.admit_current(value("stable-gemv"), false).is_err());
    assert_eq!(gemv, ServingProjection::StableGemv);
}

#[test]
fn gemv_and_cached_original_reject_capture_but_legacy_uncached_admission_is_unchanged() {
    let original = ServingProjection::Original;
    original.admit_proposal(false, false).unwrap();
    original.admit_proposal(false, true).unwrap();
    original.admit_proposal(true, false).unwrap();
    assert!(original.admit_proposal(true, true).is_err());
    let gemv = ServingProjection::StableGemv;
    gemv.admit_proposal(true, false).unwrap();
    assert!(gemv.admit_proposal(true, true).is_err());
    assert!(gemv.admit_proposal(false, false).is_err());
    assert!(gemv.admit_proposal(false, true).is_err());
}

#[test]
fn graph_probe_rejects_gemv_even_when_the_startup_prefix_requirement_was_met() {
    let gemv = ServingProjection::parse(value("stable-gemv"), true).unwrap();
    assert!(gemv.admit_graph(true).is_err());
    assert!(gemv.admit_graph(false).is_err());
    ServingProjection::Original.admit_graph(false).unwrap();
    assert!(ServingProjection::Original.admit_graph(true).is_err());
}

#[derive(Default)]
struct Completion {
    streams: Vec<u64>,
    fail: bool,
    panic: bool,
}
impl KvPrefixIo for Completion {
    fn enqueue_layer(&mut self, _: usize, _: KvTail, _: u64) -> Result<()> {
        Ok(())
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.streams.push(stream);
        assert!(!self.panic, "injected submitted-work completion panic");
        if self.fail {
            bail!("injected completion failure");
        }
        Ok(())
    }
}

#[test]
fn successful_request_reset_invalidates_cache_not_the_startup_serving_choice() {
    let choice = ServingProjection::parse(value("stable-gemv"), true).unwrap();
    let mut prefix = KvPrefix::new(5, 2047).unwrap();
    let mut binding = ProjectionBinding::new();
    let mut io = Completion::default();
    binding.select(choice.family(), &prefix).unwrap();
    prefix.begin(17, 17, 41, false).unwrap();
    for layer in 0..5 {
        prefix.enqueue_layer(layer, &mut io).unwrap();
    }
    prefix.finish(&mut io).unwrap();
    assert!(prefix.has_completed_rows());
    binding.reset(&mut prefix, &mut io).unwrap();
    assert_eq!(binding.family(), None);
    assert!(!prefix.has_completed_rows());
    assert_eq!(choice, ServingProjection::StableGemv);
    assert!(choice.admit_current(value("original"), true).is_err());
    choice.admit_current(value("stable-gemv"), true).unwrap();
    binding.select(choice.family(), &prefix).unwrap();
    assert_eq!(binding.family(), Some(ProjectionFamily::StableGemv));
}

#[test]
fn failed_or_panicking_reset_retains_real_family_pending_stream_and_startup_choice() {
    for panic in [false, true] {
        let choice = ServingProjection::parse(value("stable-gemv"), true).unwrap();
        let mut prefix = KvPrefix::new(5, 2047).unwrap();
        let mut binding = ProjectionBinding::new();
        let mut io = Completion::default();
        binding.select(choice.family(), &prefix).unwrap();
        prefix.begin(8, 8, 73, false).unwrap();
        prefix.enqueue_layer(0, &mut io).unwrap();
        io.fail = !panic;
        io.panic = panic;
        let reset = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            binding.reset(&mut prefix, &mut io)
        }));
        assert!(matches!(reset, Err(_) | Ok(Err(_))));
        assert_eq!(prefix.pending_stream(), Some(73));
        assert!(prefix.poisoned());
        assert_eq!(binding.family(), Some(choice.family()));
        assert!(binding.admit(choice.family(), &prefix).is_err());
        assert_eq!(choice, ServingProjection::StableGemv);
        io.fail = false;
        io.panic = false;
        binding.reset(&mut prefix, &mut io).unwrap();
        assert_eq!(io.streams, [73, 73]);
        assert!(!prefix.pending() && !prefix.has_completed_rows());
        assert_eq!(binding.family(), None);
        assert!(choice.admit_current(None, true).is_err());
    }
}
