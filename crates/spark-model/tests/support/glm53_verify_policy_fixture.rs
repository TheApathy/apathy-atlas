// SPDX-License-Identifier: AGPL-3.0-only

//! Injectable I/O only; selection uses the shipping runtime penalty/argmax code.

use crate::verify_policy_transaction::{
    CommittedVerify, LogitsIo, PolicyAdvance, VerifyCommitIo, VerifyOutcome, VerifyPolicy,
    VerifyRequest,
};
use anyhow::{Result, bail};
use spark_runtime::sampler::{SamplingParams, apply_penalties_and_bias, argmax_first_wins_f32};
use std::{cell::RefCell, rc::Rc};

pub type Events = Rc<RefCell<Vec<&'static str>>>;

pub fn encoded(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

pub struct Source {
    pub bytes: Rc<RefCell<Vec<u8>>>,
    pub events: Events,
    pub short: bool,
    pub fail: bool,
}

impl LogitsIo for Source {
    fn copy_logits(&mut self, destination: &mut [u8]) -> Result<usize> {
        self.events.borrow_mut().push("copy");
        if self.fail {
            bail!("injected D2H failure");
        }
        let bytes = self.bytes.borrow();
        let count = if self.short {
            destination.len() / 2
        } else {
            destination.len()
        };
        destination[..count].copy_from_slice(&bytes[..count]);
        Ok(count)
    }
}

pub struct Policy {
    pub params: SamplingParams,
    pub history: Vec<u32>,
    pub counters: [u32; 3],
    pub observed_history: Vec<Vec<u32>>,
    pub observed_rows: Vec<Vec<u8>>,
    pub masks: Vec<Vec<u32>>,
    pub terminal: Option<u32>,
    pub fail_pick: Option<usize>,
    pub fail_advance: bool,
    pub fail_restore: bool,
    pub invalid_pick: Option<u32>,
    pub events: Events,
    saved: Option<(Vec<u32>, [u32; 3])>,
}

impl Policy {
    pub fn new(events: Events) -> Self {
        Self {
            params: SamplingParams::greedy(32),
            history: vec![],
            counters: [4, 5, 6],
            observed_history: vec![],
            observed_rows: vec![],
            masks: vec![],
            terminal: None,
            fail_pick: None,
            fail_advance: false,
            fail_restore: false,
            invalid_pick: None,
            events,
            saved: None,
        }
    }
}

impl VerifyPolicy for Policy {
    fn checkpoint(&mut self) -> Result<()> {
        self.events.borrow_mut().push("policy_checkpoint");
        self.saved = Some((self.history.clone(), self.counters));
        Ok(())
    }

    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32> {
        self.events.borrow_mut().push("pick");
        self.observed_history.push(self.history.clone());
        self.observed_rows.push(logits.to_vec());
        self.counters.iter_mut().for_each(|v| *v += 1);
        if self.fail_pick == Some(row) {
            bail!("injected policy failure");
        }
        if let Some(token) = self.invalid_pick {
            return Ok(token);
        }
        let mut scores: Vec<f32> = logits
            .chunks_exact(2)
            .map(|v| half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect();
        if let Some(forbidden) = self.masks.get(row) {
            for &token in forbidden {
                scores[token as usize] = f32::NEG_INFINITY;
            }
        }
        apply_penalties_and_bias(&mut scores, &self.params, &self.history);
        Ok(argmax_first_wins_f32(&scores))
    }

    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        self.events.borrow_mut().push("policy_advance");
        self.history.push(token);
        if self.fail_advance {
            bail!("injected grammar advance failure");
        }
        Ok(if self.terminal == Some(token) {
            PolicyAdvance::Terminal
        } else {
            PolicyAdvance::Continue
        })
    }

    fn restore(&mut self) -> Result<()> {
        self.events.borrow_mut().push("policy_restore");
        if self.fail_restore {
            bail!("injected policy rollback failure");
        }
        let (history, counters) = self.saved.take().expect("checkpoint precedes restore");
        self.history = history;
        self.counters = counters;
        Ok(())
    }
}

pub struct Target {
    pub position: usize,
    pub commits: Vec<usize>,
    pub aborts: usize,
    pub poisoned: bool,
    pub fail_commit: bool,
    pub fail_abort: bool,
    pub wrong_position: bool,
    pub device_logits: Rc<RefCell<Vec<u8>>>,
    pub events: Events,
}

impl VerifyCommitIo for Target {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        self.events.borrow_mut().push("commit");
        self.commits.push(rows);
        self.position = request.start() + rows;
        self.device_logits.borrow_mut().fill(0); // Serial replay overwrites the GPU buffer.
        if self.fail_commit {
            bail!("injected irreversible commit failure");
        }
        Ok(self.position + usize::from(self.wrong_position))
    }

    fn abort_staged(&mut self) -> Result<()> {
        self.events.borrow_mut().push("abort");
        self.aborts += 1;
        if self.fail_abort {
            bail!("injected DSA restore failure");
        }
        Ok(())
    }

    fn poison(&mut self) {
        self.events.borrow_mut().push("poison");
        self.poisoned = true;
    }
}

pub fn fixture(rows: &[Vec<f32>]) -> (Source, Policy, Target, Events) {
    let events = Rc::new(RefCell::new(vec![]));
    let bytes = Rc::new(RefCell::new(encoded(rows)));
    let source = Source {
        bytes: bytes.clone(),
        events: events.clone(),
        short: false,
        fail: false,
    };
    let policy = Policy::new(events.clone());
    let target = Target {
        position: 4,
        commits: vec![],
        aborts: 0,
        poisoned: false,
        fail_commit: false,
        fail_abort: false,
        wrong_position: false,
        device_logits: bytes,
        events: events.clone(),
    };
    (source, policy, target, events)
}

#[allow(dead_code)] // Shared fixture also serves failure-only integration targets.
pub fn committed(outcome: VerifyOutcome) -> CommittedVerify {
    match outcome {
        VerifyOutcome::Committed(receipt) => receipt,
        VerifyOutcome::RestoredForOrdinaryReplay(_) => panic!("unexpected ordinary replay"),
    }
}
