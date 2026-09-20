// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit fixed-prefix commit-path probe; never a natural acceptance score.
use anyhow::{Result, ensure};
use spark_model::model::glm53::verify_policy_transaction::{PolicyAdvance, VerifyPolicy};

const VOCAB: u32 = 154_880;
pub struct ForcedPrefixPolicy {
    inputs: Vec<u32>,
    rows: usize,
    cursor: usize,
    pending: Option<u32>,
    checkpoint: Option<(usize, Option<u32>)>,
}
impl ForcedPrefixPolicy {
    pub fn new(inputs: Vec<u32>, rows: usize) -> Result<Self> {
        ensure!(
            (2..=8).contains(&inputs.len()) && (1..=inputs.len()).contains(&rows),
            "forced policy requires an admitted committed prefix"
        );
        ensure!(
            inputs.iter().all(|&id| id < VOCAB),
            "forced input outside vocabulary"
        );
        Ok(Self {
            inputs,
            rows,
            cursor: 0,
            pending: None,
            checkpoint: None,
        })
    }

    pub fn publication_matches(&self, emitted: &[u32], terminal: bool) -> bool {
        !terminal
            && emitted.len() == self.rows
            && emitted[..self.rows - 1] == self.inputs[1..self.rows]
            && emitted[self.rows - 1] == self.selected_token(self.rows - 1)
    }

    fn selected_token(&self, row: usize) -> u32 {
        if row + 1 < self.rows {
            self.inputs[row + 1]
        } else if self.rows < self.inputs.len() {
            (self.inputs[row + 1] + 1) % VOCAB
        } else {
            0
        }
    }
}
impl VerifyPolicy for ForcedPrefixPolicy {
    fn checkpoint(&mut self) -> Result<()> {
        self.checkpoint = Some((self.cursor, self.pending));
        Ok(())
    }
    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32> {
        ensure!(
            self.checkpoint.is_some()
                && self.pending.is_none()
                && row == self.cursor
                && row < self.rows
                && logits.len() == VOCAB as usize * 2,
            "forced policy row/logits/ordering mismatch"
        );
        let token = self.selected_token(row);
        self.pending = Some(token);
        Ok(token)
    }
    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        ensure!(
            self.pending == Some(token),
            "forced policy advance differs from selected token"
        );
        self.cursor += 1;
        self.pending = None;
        Ok(PolicyAdvance::Continue)
    }
    fn restore(&mut self) -> Result<()> {
        let (cursor, pending) = self
            .checkpoint
            .ok_or_else(|| anyhow::anyhow!("missing forced policy checkpoint"))?;
        self.cursor = cursor;
        self.pending = pending;
        Ok(())
    }
}
