// SPDX-License-Identifier: AGPL-3.0-only
//! Real deferred host bytes test ownership/completion, not encoder arithmetic.
#[path = "../src/layers/deepseek_vision/linear_diagnostic_completion.rs"]
mod completion;
use completion::{CompletionIo, OwnerIo, release_owned, with_completion};

struct Session {
    calls: Vec<&'static str>,
    owners: [Option<Vec<u8>>; 4], // weights, encoder arena, workspace, handle receipt
    pending: Option<Vec<u8>>,
    fail_fence: bool,
    poisoned: bool,
}
impl Session {
    fn new() -> Self {
        Self {
            calls: vec![],
            owners: std::array::from_fn(|i| Some(vec![i as u8; 8])),
            pending: None,
            fail_fence: false,
            poisoned: false,
        }
    }
    fn upload(&mut self) {
        self.calls.push("upload");
        self.pending = Some(vec![0x3f; 8]);
    }
    fn submit(&mut self) {
        self.calls.push("submit");
        self.pending.as_mut().unwrap()[0] = 0x40;
    }
    fn observe(&mut self) {
        self.calls.push("observe");
    }
    fn retained(&self) -> bool {
        self.owners.iter().all(Option::is_some)
    }
}
impl CompletionIo for Session {
    fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    fn synchronize(&mut self) -> Result<(), String> {
        self.calls.push("fence");
        if self.fail_fence {
            return Err("fence failed".into());
        }
        if let Some(bytes) = self.pending.take() {
            self.owners[1].as_mut().unwrap().copy_from_slice(&bytes);
        }
        Ok(())
    }
    fn quarantine(&mut self) {
        self.calls.push("quarantine");
        self.poisoned = true;
    }
}
impl OwnerIo for Session {
    fn release_owners(&mut self) -> Result<(), String> {
        assert!(self.pending.is_none(), "free before completion");
        self.calls.push("release");
        for owner in &mut self.owners {
            *owner = None;
        }
        Ok(())
    }
}

#[test]
fn success_returns_only_after_completion_and_release_is_separately_fenced() {
    let mut s = Session::new();
    let value = with_completion(&mut s, |s| {
        s.upload();
        s.submit();
        s.observe();
        Ok(17)
    })
    .unwrap();
    assert_eq!(value, 17);
    assert_eq!(s.calls, ["upload", "submit", "observe", "fence"]);
    assert_eq!(s.owners[1].as_ref().unwrap()[0], 0x40);
    assert!(s.retained());
    release_owned(&mut s).unwrap();
    assert_eq!(&s.calls[4..], ["fence", "release"]);
    assert!(s.owners.iter().all(Option::is_none));
}

#[test]
fn upload_submit_and_observer_errors_each_drain_without_publishing() {
    for stop in 0..3 {
        let mut s = Session::new();
        let error = with_completion(&mut s, |s| -> Result<(), String> {
            s.upload();
            if stop == 0 {
                return Err("upload failed after submission".into());
            }
            s.submit();
            if stop == 1 {
                return Err("linear submission failed".into());
            }
            s.observe();
            Err("observer failed".into())
        })
        .unwrap_err();
        assert!(error.contains(
            [
                "upload failed",
                "linear submission failed",
                "observer failed"
            ][stop]
        ));
        assert_eq!(s.calls.last(), Some(&"fence"));
        assert!(s.pending.is_none());
        assert!(s.retained());
        assert!(!s.poisoned);
        release_owned(&mut s).unwrap();
    }
}

#[test]
fn failed_fence_retains_every_owner_and_preserves_both_errors() {
    for fail_operation in [false, true] {
        let mut s = Session::new();
        s.fail_fence = true;
        let error = with_completion(&mut s, |s| {
            s.upload();
            s.submit();
            if fail_operation {
                Err("linear submission failed".into())
            } else {
                Ok(17)
            }
        })
        .unwrap_err();
        assert!(error.contains("fence failed"));
        if fail_operation {
            assert!(error.contains("linear submission failed"));
        }
        assert!(s.poisoned && s.retained());
        assert!(s.pending.is_some());
        assert!(!s.calls.contains(&"release"));
        let before = s.calls.clone();
        assert!(
            with_completion(&mut s, |_| -> Result<(), String> {
                panic!("reused poisoned session")
            })
            .is_err()
        );
        assert!(release_owned(&mut s).is_err());
        assert_eq!(
            s.calls, before,
            "quarantine must prevent reuse and optimistic free"
        );
    }
}

#[test]
fn teardown_fence_failure_does_not_destroy_handle_or_free_weights() {
    let mut s = Session::new();
    with_completion(&mut s, |s| {
        s.upload();
        Ok(())
    })
    .unwrap();
    s.fail_fence = true;
    assert!(release_owned(&mut s).is_err());
    assert!(s.retained() && s.poisoned);
    assert!(!s.calls.contains(&"release"));
}
