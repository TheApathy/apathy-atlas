// SPDX-License-Identifier: AGPL-3.0-only

//! Compact greedy selection must never materialize full vocabulary rows.

#[path = "../src/model/glm53/verify_policy_transaction.rs"]
#[allow(dead_code)]
mod transaction;

use anyhow::{Result, bail};
use transaction::{
    LogitsIo, PolicyAdvance, VerifyCommitIo, VerifyOutcome, VerifyPolicy, VerifyRequest,
    run_verify_policy_transaction,
};

struct Source {
    compact: Vec<u32>,
    full_calls: usize,
    compact_calls: usize,
}

impl LogitsIo for Source {
    fn copy_logits(&mut self, _: &mut [u8]) -> Result<usize> {
        self.full_calls += 1;
        bail!("full logits must not be copied for admitted compact greedy")
    }

    fn copy_argmax(&mut self, destination: &mut [u32], excluded: [u32; 2]) -> Result<usize> {
        assert_eq!(excluded, [u32::MAX; 2]);
        self.compact_calls += 1;
        destination.copy_from_slice(&self.compact);
        Ok(destination.len())
    }
}

struct Policy {
    picked: Vec<u32>,
}

impl VerifyPolicy for Policy {
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }
    fn compact_argmax_exclusions(&self) -> Option<[u32; 2]> {
        Some([u32::MAX; 2])
    }
    fn pick(&mut self, _: usize, _: &[u8]) -> Result<u32> {
        bail!("full row policy must not run for admitted compact greedy")
    }
    fn pick_argmax(&mut self, row: usize, token: u32) -> Result<u32> {
        assert_eq!(row, self.picked.len());
        self.picked.push(token);
        Ok(token)
    }
    fn advance(&mut self, _: u32) -> Result<PolicyAdvance> {
        Ok(PolicyAdvance::Continue)
    }
    fn restore(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Target {
    rows: usize,
}

impl VerifyCommitIo for Target {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        self.rows = rows;
        Ok(request.start() + rows)
    }
    fn abort_staged(&mut self) -> Result<()> {
        Ok(())
    }
    fn poison(&mut self) {}
}

#[test]
fn admitted_compact_argmax_copies_one_token_per_row_and_preserves_selection_order() {
    let request = VerifyRequest::new(7, 64, 5, &[0, 1, 2]).unwrap();
    let mut source = Source {
        compact: vec![1, 2, 3],
        full_calls: 0,
        compact_calls: 0,
    };
    let mut policy = Policy { picked: vec![] };
    let mut target = Target { rows: 0 };
    let outcome =
        run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap();
    let VerifyOutcome::Committed(receipt) = outcome else {
        panic!("compact greedy unexpectedly requested ordinary replay")
    };
    assert_eq!(receipt.accepted_drafts(), 2);
    assert_eq!(source.full_calls, 0);
    assert_eq!(source.compact_calls, 1);
    assert_eq!(policy.picked, [1, 2, 3]);
    assert_eq!(target.rows, 3);
}
