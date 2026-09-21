// SPDX-License-Identifier: AGPL-3.0-only
//! Execute the production owning capsule, not a test-owned quarantine flag.
#[path = "../src/layers/deepseek_vision/linear_diagnostic_completion.rs"]
mod linear_diagnostic_completion;
#[path = "../src/layers/deepseek_vision/owned_session.rs"]
mod owned_session;
#[path = "deepseek_vision_owned_fixture.rs"]
mod fixture;
use fixture::*;
use owned_session::OwnedSession;

#[test]
fn moved_upload_keeps_its_allocation_and_completion_precedes_publication() {
    let log = events();
    let upload = Tracked::new("upload", &log);
    let pointer = upload.bytes.as_ptr() as usize;
    let mut session = OwnedSession::new(Resources::new(&log));
    let receipt = session.run(|r| {
        r.enqueue_upload(upload, pointer);
        r.enqueue_readback();
        Ok(17u64)
    }, Resources::fence).unwrap();
    assert_eq!(receipt, 17);
    assert!(saw(&log, "readback:complete"));
    assert!(saw(&log, "drop:upload") && saw(&log, "drop:readback"));
    assert!(!saw(&log, "drop:arena"));
    assert!(!session.is_poisoned() && !session.is_released());
    session.try_release(Resources::fence, Resources::free).unwrap();
    assert!(session.is_released());
    for name in ["handle", "workspace", "arena", "weights", "backend"] {
        assert!(saw(&log, &format!("drop:{name}")));
    }
    let prior = log.borrow().clone();
    drop(session);
    assert_eq!(*log.borrow(), prior, "successful cleanup must not run twice");
}

#[test]
fn upload_error_after_enqueue_and_failed_fence_retains_actual_payload_on_drop() {
    let log = events();
    let upload = Tracked::new("upload", &log);
    let pointer = upload.bytes.as_ptr() as usize;
    let mut resources = Resources::new(&log);
    resources.fail_fence = true;
    let mut session = OwnedSession::new(resources);
    let error = session.run(|r| -> Result<(), String> {
        r.enqueue_upload(upload, pointer);
        Err("copy_h2d failed after enqueue".into())
    }, Resources::fence).unwrap_err();
    assert!(error.contains("copy_h2d failed after enqueue"));
    assert!(error.contains("injected completion failure"));
    assert!(session.is_poisoned() && !session.is_released());
    assert_no_drop(&log);
    drop(session);
    assert_no_drop(&log);
}

#[test]
fn readback_error_retains_destination_and_all_five_resource_owners() {
    let log = events();
    let mut resources = Resources::new(&log);
    resources.fail_fence = true;
    let mut session = OwnedSession::new(resources);
    assert!(session.run(|r| -> Result<(), String> {
        r.enqueue_readback();
        Err("copy_d2h failed after enqueue".into())
    }, Resources::fence).is_err());
    let prior = log.borrow().clone();
    assert!(session.run(|_| -> Result<(), String> {
        panic!("poisoned resource reuse");
    }, |_| panic!("automatic poisoned retry fence")).is_err());
    assert!(session.try_release(|_| panic!("automatic poisoned recovery"),
        |_| panic!("free after failed completion")).is_err());
    assert_eq!(*log.borrow(), prior);
    drop(session);
    assert_no_drop(&log);
}

#[test]
fn upload_submission_and_observer_errors_with_good_fence_remain_cleanup_safe() {
    for message in ["upload error", "submission error", "observer error"] {
        let log = events();
        let upload = Tracked::new("upload", &log);
        let pointer = upload.bytes.as_ptr() as usize;
        let mut session = OwnedSession::new(Resources::new(&log));
        let error = session.run(|r| -> Result<(), String> {
            r.enqueue_upload(upload, pointer);
            r.enqueue_readback();
            Err(message.into())
        }, Resources::fence).unwrap_err();
        assert!(error.contains(message));
        assert!(saw(&log, "readback:complete"));
        assert!(!session.is_poisoned());
        session.try_release(Resources::fence, Resources::free).unwrap();
        assert!(session.is_released());
    }
}

#[test]
fn teardown_fence_failure_preserves_nonconsumed_session_and_owners() {
    let log = events();
    let mut resources = Resources::new(&log);
    resources.fail_fence = true;
    let mut session = OwnedSession::new(resources);
    let error = session.try_release(Resources::fence, Resources::free).unwrap_err();
    assert!(error.contains("injected completion failure"));
    assert!(session.is_poisoned() && !session.is_released());
    assert!(!saw(&log, "free:0"));
    drop(session);
    assert_no_drop(&log);
}

#[test]
fn partial_cleanup_keeps_failed_and_later_owners_without_double_release() {
    let log = events();
    let mut resources = Resources::new(&log);
    resources.fail_free = Some(2);
    let mut session = OwnedSession::new(resources);
    assert!(session.try_release(Resources::fence, Resources::free).is_err());
    assert!(saw(&log, "drop:handle") && saw(&log, "drop:workspace"));
    for name in ["arena", "weights", "backend"] {
        assert!(!saw(&log, &format!("drop:{name}")));
    }
    assert!(session.is_poisoned() && !session.is_released());
    let prior = log.borrow().clone();
    assert!(session.try_release(Resources::fence, Resources::free).is_err());
    drop(session);
    assert_eq!(*log.borrow(), prior);
}

#[test]
fn unreleased_owner_does_not_run_implicit_device_or_ffi_cleanup_on_drop() {
    let log = events();
    drop(OwnedSession::new(Resources::new(&log)));
    assert!(log.borrow().is_empty());
}

#[test]
fn released_owner_rejects_new_work_and_repeated_cleanup_without_callbacks() {
    let log = events();
    let mut session = OwnedSession::new(Resources::new(&log));
    session.try_release(Resources::fence, Resources::free).unwrap();
    let prior = log.borrow().clone();
    assert!(session.run(|_| -> Result<(), String> { panic!("released reuse"); },
        |_| panic!("released fence")).is_err());
    assert!(session.try_release(|_| panic!("repeated release fence"),
        |_| panic!("repeated release")).is_err());
    assert_eq!(*log.borrow(), prior);
}
