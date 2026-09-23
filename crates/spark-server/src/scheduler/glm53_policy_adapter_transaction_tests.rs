// SPDX-License-Identifier: AGPL-3.0-only

//! Drives the real model transaction with the proposed real server policy.
//! The bridge is test-only while the cross-crate production trait is held.

#[path = "../../../spark-model/src/model/glm53/verify_policy_transaction.rs"]
mod transaction;

use super::fixture::{MODEL, fingerprint, grammar, row, sequence};
use super::{OrdinaryVerifyPolicy, context, options};
use crate::scheduler::ResponseSink;
use anyhow::{Result, bail};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use transaction::{
    CommittedVerify, LogitsIo, PolicyAdvance, VerifyCommitIo, VerifyOutcome, VerifyPolicy,
    VerifyRequest, run_verify_policy_transaction,
};

fn committed(outcome: VerifyOutcome) -> CommittedVerify {
    match outcome {
        VerifyOutcome::Committed(receipt) => receipt,
        VerifyOutcome::RestoredForOrdinaryReplay(_) => panic!("unexpected ordinary replay"),
    }
}

struct Source {
    bytes: Vec<u8>,
    short: bool,
}
impl LogitsIo for Source {
    fn copy_logits(&mut self, dst: &mut [u8]) -> Result<usize> {
        assert_eq!(dst.len(), self.bytes.len());
        dst.copy_from_slice(&self.bytes);
        Ok(dst.len() - usize::from(self.short))
    }
}

struct Bridge<'a> {
    inner: OrdinaryVerifyPolicy<'a>,
    before: Vec<String>,
    restored: Rc<Cell<bool>>,
    visits: Rc<RefCell<Vec<usize>>>,
    fail_restore: bool,
}
impl VerifyPolicy for Bridge<'_> {
    fn checkpoint(&mut self) -> Result<()> {
        self.inner.checkpoint()
    }
    fn pick(&mut self, index: usize, bytes: &[u8]) -> Result<u32> {
        self.visits.borrow_mut().push(index);
        self.inner.pick(index, bytes)
    }
    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        match self.inner.advance(token) {
            Ok(true) => Ok(PolicyAdvance::Continue),
            Ok(false) => Ok(PolicyAdvance::Terminal),
            // Classify the callback's typed effect before policy/target restore.
            // A later restoration error is never inspected for replay authority.
            Err(error) if error.is::<super::adapter::NeedOrdinaryReplay>() => {
                Ok(PolicyAdvance::OrdinaryReplay)
            }
            Err(error) => Err(error),
        }
    }
    fn restore(&mut self) -> Result<()> {
        self.inner.restore()?;
        assert_eq!(fingerprint(self.inner.sequence()), self.before);
        if self.fail_restore {
            bail!("injected policy restoration failure");
        }
        self.restored.set(true);
        Ok(())
    }
}

struct Target {
    restored: Rc<Cell<bool>>,
    events: Vec<&'static str>,
    fail_commit: bool,
    committed_rows: usize,
}
impl VerifyCommitIo for Target {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        assert!(
            self.restored.get(),
            "target commit cannot precede live policy restoration"
        );
        self.events.push("commit");
        self.committed_rows = rows;
        if self.fail_commit {
            bail!("injected persistent commit failure");
        }
        Ok(request.start() + rows)
    }
    fn abort_staged(&mut self) -> Result<()> {
        self.events.push("abort");
        Ok(())
    }
    fn poison(&mut self) {
        self.events.push("poison");
    }
}

#[test]
fn every_budget_and_eos_terminal_row_commits_before_exposing_emission() {
    for rows in 2..=8 {
        for stop_row in 0..rows {
            for eos in [false, true] {
                let mut live = sequence();
                live.remaining = if eos { 20 } else { stop_row + 1 };
                if eos {
                    live.eos_tokens = vec![3];
                }
                let (tx, mut rx) = tokio::sync::mpsc::channel(16);
                live.sink = ResponseSink::Streaming(tx);
                let before = fingerprint(&live);
                let ctx = context();
                let restored = Rc::new(Cell::new(false));
                let visits = Rc::new(RefCell::new(vec![]));
                let picks: Vec<u32> = (0..rows)
                    .map(|i| if eos && i == stop_row { 3 } else { 1 })
                    .collect();
                let mut inputs = vec![0];
                inputs.extend_from_slice(&picks[..rows - 1]);
                let request = VerifyRequest::new(2, 64, 5, &inputs).unwrap();
                let bytes = picks
                    .iter()
                    .flat_map(|&token| {
                        let mut scores = [0.0; 5];
                        scores[token as usize] = 20.0;
                        row(&scores)
                    })
                    .collect();
                let mut source = Source {
                    bytes,
                    short: false,
                };
                let mut target = Target {
                    restored: restored.clone(),
                    events: vec![],
                    fail_commit: false,
                    committed_rows: 0,
                };
                let receipt = {
                    let inner =
                        OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
                    let mut policy = Bridge {
                        inner,
                        before: before.clone(),
                        restored,
                        visits: visits.clone(),
                        fail_restore: false,
                    };
                    let receipt = committed(
                        run_verify_policy_transaction(
                            &request,
                            &mut source,
                            &mut policy,
                            &mut target,
                        )
                        .unwrap(),
                    );
                    assert!(
                        rx.try_recv().is_err(),
                        "neither pick nor advance may call the live stream"
                    );
                    receipt
                };
                assert_eq!(fingerprint(&live), before);
                assert_eq!(*visits.borrow(), (0..=stop_row).collect::<Vec<_>>());
                assert_eq!(target.events, ["commit"]);
                assert_eq!(target.committed_rows, (stop_row + 2).min(rows));
                let published = receipt
                    .publish(
                        &mut live.seq.tokens,
                        &mut live.seq.seq_len,
                        &mut live.seq.kv_valid_tokens,
                    )
                    .unwrap();
                assert!(published.terminal());
                assert_eq!(published.emitted_tokens(), &picks[..=stop_row]);
                assert_eq!(live.seq.tokens.len(), live.seq.seq_len);
                assert_eq!(live.seq.seq_len, live.seq.kv_valid_tokens);
                assert_eq!(live.seq.seq_len, 2 + target.committed_rows);
                assert!(rx.try_recv().is_err(), "host publication is not streaming");
            }
        }
    }
}

#[test]
fn real_grammar_rejects_draft_then_bonus_without_visiting_or_committing_suffix() {
    let mut live = sequence();
    live.grammar_state = Some(grammar());
    live.eos_tokens = vec![3];
    let before = fingerprint(&live);
    let ctx = context();
    let restored = Rc::new(Cell::new(false));
    let visits = Rc::new(RefCell::new(vec![]));
    let request = VerifyRequest::new(2, 64, 5, &[4, 0, 2, 2]).unwrap();
    let mut source = Source {
        bytes: [
            row(&[20.0, 1.0, 30.0, 0.0, 0.0]),
            row(&[0.0, 1.0, 30.0, 0.0, 0.0]),
            row(&[0.0, 0.0, 30.0, 0.0, 0.0]),
            row(&[0.0, 0.0, 30.0, 0.0, 0.0]),
        ]
        .concat(),
        short: false,
    };
    let mut target = Target {
        restored: restored.clone(),
        events: vec![],
        fail_commit: false,
        committed_rows: 0,
    };
    let receipt = {
        let inner = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
        let mut policy = Bridge {
            inner,
            before: before.clone(),
            restored,
            visits: visits.clone(),
            fail_restore: false,
        };
        committed(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
        )
    };
    assert_eq!(*visits.borrow(), [0, 1]);
    assert_eq!(fingerprint(&live), before);
    assert_eq!(receipt.accepted_drafts(), 1);
    assert_eq!(target.committed_rows, 2);
    let published = receipt
        .publish(
            &mut live.seq.tokens,
            &mut live.seq.seq_len,
            &mut live.seq.kv_valid_tokens,
        )
        .unwrap();
    assert_eq!(published.emitted_tokens(), [0, 1]);
    assert!(!published.terminal());
    assert_eq!(live.seq.tokens, [0, 0, 4, 0]);
    assert_eq!(live.grammar_state.as_ref().unwrap().num_history_steps(), 0);
}

#[test]
fn copy_restore_and_commit_failure_leave_stream_silent_and_policy_restored() {
    for failure in ["copy", "restore", "commit"] {
        let mut live = sequence();
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        live.sink = ResponseSink::Streaming(tx);
        let before = fingerprint(&live);
        let ctx = context();
        let restored = Rc::new(Cell::new(false));
        let visits = Rc::new(RefCell::new(vec![]));
        let request = VerifyRequest::new(2, 64, 5, &[0, 1]).unwrap();
        let mut source = Source {
            bytes: [
                row(&[0.0, 20.0, 0.0, 0.0, 0.0]),
                row(&[0.0, 0.0, 20.0, 0.0, 0.0]),
            ]
            .concat(),
            short: failure == "copy",
        };
        let mut target = Target {
            restored: restored.clone(),
            events: vec![],
            fail_commit: failure == "commit",
            committed_rows: 0,
        };
        {
            let inner = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
            let mut policy = Bridge {
                inner,
                before: before.clone(),
                restored,
                visits: visits.clone(),
                fail_restore: failure == "restore",
            };
            assert!(
                run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target)
                    .is_err()
            );
        }
        assert_eq!(fingerprint(&live), before);
        assert!(rx.try_recv().is_err());
        match failure {
            "copy" => {
                assert!(visits.borrow().is_empty());
                assert_eq!(target.events, ["abort"]);
            }
            "restore" => {
                assert!(!target.events.contains(&"commit"));
                assert!(target.events.contains(&"poison"));
            }
            "commit" => assert_eq!(target.events, ["commit", "poison"]),
            _ => unreachable!(),
        }
    }
}

#[test]
fn simple_content_can_only_reach_existing_sink_after_successful_publication() {
    let mut live = sequence();
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    live.sink = ResponseSink::Streaming(tx);
    let before = fingerprint(&live);
    let ctx = context();
    let restored = Rc::new(Cell::new(false));
    let request = VerifyRequest::new(2, 64, 5, &[0, 1]).unwrap();
    let mut source = Source {
        bytes: [
            row(&[0.0, 20.0, 0.0, 0.0, 0.0]),
            row(&[0.0, 0.0, 20.0, 0.0, 0.0]),
        ]
        .concat(),
        short: false,
    };
    let mut target = Target {
        restored: restored.clone(),
        events: vec![],
        fail_commit: false,
        committed_rows: 0,
    };
    let receipt = {
        let inner = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
        let mut policy = Bridge {
            inner,
            before,
            restored,
            visits: Rc::new(RefCell::new(vec![])),
            fail_restore: false,
        };
        committed(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
        )
    };
    assert!(rx.try_recv().is_err());
    assert_eq!(target.events, ["commit"]);
    let published = receipt
        .publish(
            &mut live.seq.tokens,
            &mut live.seq.seq_len,
            &mut live.seq.kv_valid_tokens,
        )
        .unwrap();
    // Only a simple content sink-order check; emit_token is NOT our ordinary
    // thinking/grammar transition oracle or the proposed production adapter.
    for &token in published.emitted_tokens() {
        crate::scheduler::emit_step::emit_token(&mut live, token, None);
    }
    assert!(matches!(
        rx.try_recv().unwrap(),
        crate::api::StreamEvent::Token(1)
    ));
    assert!(matches!(
        rx.try_recv().unwrap(),
        crate::api::StreamEvent::Token(2)
    ));
    assert!(rx.try_recv().is_err());
}

#[test]
fn real_unsupported_ordinary_effect_returns_replay_only_after_both_restores() {
    for fail_restore in [false, true] {
        let mut live = sequence();
        live.tool_request = true;
        live.prose_tokens_since_last_tool = u32::MAX;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        live.sink = ResponseSink::Streaming(tx);
        let before = fingerprint(&live);
        let ctx = context();
        let restored = Rc::new(Cell::new(false));
        let request = VerifyRequest::new(2, 64, 5, &[0, 1]).unwrap();
        let mut source = Source {
            bytes: [row(&[0., 20., 0., 0., 0.]), row(&[0., 20., 0., 0., 0.])].concat(),
            short: false,
        };
        let mut target = Target {
            restored: restored.clone(),
            events: vec![],
            fail_commit: false,
            committed_rows: 0,
        };
        let outcome = {
            let inner = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
            let mut policy = Bridge {
                inner,
                before: before.clone(),
                restored: restored.clone(),
                visits: Rc::new(RefCell::new(vec![])),
                fail_restore,
            };
            let outcome =
                run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target);
            assert!(
                policy.inner.take_prepared().is_err(),
                "no retained emission after replay request"
            );
            outcome
        };
        assert_eq!(fingerprint(&live), before);
        assert!(rx.try_recv().is_err());
        assert_eq!(target.committed_rows, 0);
        if fail_restore {
            assert!(outcome.is_err());
            assert_eq!(target.events, ["poison", "abort"]);
        } else {
            let VerifyOutcome::RestoredForOrdinaryReplay(receipt) = outcome.unwrap() else {
                panic!("a watchdog re-steer cannot commit or terminate the request");
            };
            assert_eq!((receipt.start(), receipt.anchor()), (2, 0));
            assert!(restored.get());
            assert_eq!(target.events, ["abort"]);
        }
    }
}
