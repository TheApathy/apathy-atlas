// SPDX-License-Identifier: AGPL-3.0-only

use std::any::Any;

use anyhow::{Context, Result, bail};

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

pub(super) fn join_scheduler(handle: std::thread::JoinHandle<()>) -> Result<()> {
    if let Err(payload) = handle.join() {
        bail!("scheduler thread panicked: {}", panic_message(&*payload));
    }
    Ok(())
}

pub(super) fn resolve_serve_result(
    serve_result: Result<()>,
    scheduler_join: std::result::Result<Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    match serve_result {
        Err(primary) => {
            match scheduler_join {
                Ok(Err(error)) => {
                    tracing::error!("scheduler shutdown also failed after server error: {error:#}")
                }
                Err(error) => {
                    tracing::error!("scheduler join task also failed after server error: {error:#}")
                }
                Ok(Ok(())) => {}
            }
            Err(primary)
        }
        Ok(()) => {
            scheduler_join.context("scheduler join task failed")??;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};

    use super::{join_scheduler, resolve_serve_result};

    struct DropReceipt(Arc<AtomicBool>);

    impl Drop for DropReceipt {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn join_waits_for_scheduler_owned_drop() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (release_tx, release_rx) = mpsc::channel();
        let receipt = DropReceipt(dropped.clone());
        let scheduler = std::thread::spawn(move || {
            let _receipt = receipt;
            release_rx.recv().unwrap();
        });
        let releaser = std::thread::spawn(move || release_tx.send(()).unwrap());

        join_scheduler(scheduler).unwrap();
        releaser.join().unwrap();
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn closing_the_last_request_sender_lets_scheduler_drop_before_join_returns() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (request_tx, request_rx) = mpsc::channel::<()>();
        let receipt = DropReceipt(dropped.clone());
        let scheduler = std::thread::spawn(move || {
            let _receipt = receipt;
            while request_rx.recv().is_ok() {}
        });

        assert!(!dropped.load(Ordering::SeqCst));
        drop(request_tx);
        join_scheduler(scheduler).unwrap();
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn join_reports_scheduler_panic() {
        let scheduler = std::thread::spawn(|| panic!("intentional scheduler panic"));
        let error = join_scheduler(scheduler).unwrap_err().to_string();
        assert!(error.contains("scheduler thread panicked"));
        assert!(error.contains("intentional scheduler panic"));
    }

    #[test]
    fn join_reports_non_string_scheduler_panic() {
        let scheduler = std::thread::spawn(|| std::panic::panic_any(7_u32));
        let error = join_scheduler(scheduler).unwrap_err().to_string();
        assert!(error.contains("scheduler thread panicked"));
        assert!(error.contains("non-string panic payload"));
    }

    #[test]
    fn server_error_remains_primary_when_scheduler_also_fails() {
        let error = resolve_serve_result(
            Err(anyhow::anyhow!("primary server failure")),
            Ok(Err(anyhow::anyhow!("secondary scheduler failure"))),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(error, "primary server failure");
    }

    #[test]
    fn scheduler_error_propagates_after_clean_server_shutdown() {
        let error = resolve_serve_result(
            Ok(()),
            Ok(Err(anyhow::anyhow!("scheduler shutdown failure"))),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(error, "scheduler shutdown failure");
    }

    #[test]
    fn scheduler_spawn_is_after_all_fallible_startup_sites() {
        let source = include_str!("serve.rs");
        let spawn = source
            .find("let scheduler_thread = std::thread::spawn")
            .unwrap();
        for fallible in [
            "serve_phases::resolve_tool_call_parser",
            "let auth = build_auth_config(&args)?",
            "signal(SignalKind::user_defined1())",
            "signal(SignalKind::user_defined2())",
        ] {
            assert!(source.find(fallible).unwrap() < spawn, "{fallible}");
        }
        assert!(spawn < source.find("Ok(Some((").unwrap());
    }
}
