// SPDX-License-Identifier: AGPL-3.0-only

//! Qualification-only target-chain fixture for every flat K16 commit branch.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, ensure};

use super::*;

#[path = "verify_dflash_k16_selector.rs"]
mod selector;

use selector::select_from_env;

const K16: usize = 16;
const DRAFTS: usize = K16 - 1;
const LOGICAL_VOCAB: usize = 248_077;
const PHYSICAL_VOCAB: usize = 248_320;

static CONSUMED: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
pub(super) struct FixtureFrame {
    accepted: usize,
    observed: bool,
}

#[derive(Debug)]
pub(super) struct PreparedFixture {
    pub(super) frame: FixtureFrame,
    pub(super) drafts: Vec<u32>,
}

pub(super) struct EvidenceGuard<'a> {
    model: &'a dyn Model,
    armed: bool,
}

impl<'a> EvidenceGuard<'a> {
    pub(super) fn begin(
        model: &'a dyn Model,
        pre_verify_len: usize,
        inputs: &[u32],
    ) -> Result<Self> {
        model.k16_fixture_evidence_begin(pre_verify_len, inputs)?;
        Ok(Self { model, armed: true })
    }

    fn confirm(&mut self, pre_verify_len: usize, inputs: &[u32]) -> Result<()> {
        self.model
            .k16_fixture_evidence_finish(pre_verify_len, inputs)?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for EvidenceGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.model.k16_fixture_evidence_abort();
        }
    }
}

fn forced_drafts(chain: &[u32], accepted: usize, vocab: usize) -> Result<Vec<u32>> {
    ensure!(
        chain.len() == DRAFTS,
        "K16 target chain must have 15 drafts"
    );
    ensure!(
        accepted <= DRAFTS && vocab > 1,
        "invalid K16 fixture geometry"
    );
    ensure!(
        chain.iter().all(|&token| (token as usize) < vocab),
        "target token outside vocab"
    );
    let mut drafts = chain.to_vec();
    if accepted < DRAFTS {
        let mismatch = (usize::try_from(chain[accepted])? + 1) % vocab;
        drafts[accepted] = u32::try_from(mismatch)?;
        ensure!(
            drafts[accepted] != chain[accepted],
            "fixture mismatch construction failed"
        );
    }
    Ok(drafts)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    model: &dyn Model,
    seq: &mut SequenceState,
    last_token: u32,
    draft_len: usize,
    num_drafts: usize,
    flat_frame: bool,
    c1_diagnostic: bool,
    neutral_greedy: bool,
) -> Result<Option<PreparedFixture>> {
    let Some(selector) = select_from_env(seq.seq_len, CONSUMED.load(Ordering::Acquire))? else {
        return Ok(None);
    };
    ensure!(
        !CONSUMED.swap(true, Ordering::AcqRel),
        "K16 acceptance fixture already consumed"
    );
    ensure!(
        flat_frame && c1_diagnostic && neutral_greedy,
        "K16 fixture requires neutral-greedy C1 flat mode"
    );
    ensure!(!model.is_ep(), "K16 fixture forbids expert parallelism");
    ensure!(
        model.proposer_is_dflash(),
        "K16 fixture requires the native DFlash proposer"
    );
    ensure!(
        model.dflash_verify_capacity_k() == Some(K16),
        "K16 fixture requires physical K16"
    );
    ensure!(
        draft_len == DRAFTS && num_drafts == DRAFTS,
        "K16 fixture requires gamma15"
    );
    ensure!(
        matches!(model.vocab_size(), LOGICAL_VOCAB | PHYSICAL_VOCAB),
        "K16 fixture requires exact Flash-Next target vocabulary"
    );
    ensure!(
        (last_token as usize) < LOGICAL_VOCAB,
        "K16 fixture input must be inside the canonical logical vocabulary"
    );
    ensure!(
        !model.decode_logits_fp32(),
        "K16 fixture requires BF16 full-vocabulary diagnostics"
    );
    ensure!(
        seq.tokens.len() == seq.seq_len,
        "K16 fixture sequence extent mismatch"
    );

    let pre_verify_len = seq.seq_len;
    model.pre_verify_copy_async(seq)?;
    let chain: Result<Vec<u32>> = (|| {
        let mut input = last_token;
        let mut chain = Vec::with_capacity(DRAFTS);
        for _ in 0..DRAFTS {
            let logits = model.decode(input, seq, 0)?;
            input = model.argmax_on_device(logits, 0)?;
            chain.push(input);
        }
        Ok(chain)
    })();
    seq.seq_len = pre_verify_len;
    seq.tokens.truncate(pre_verify_len);
    model.pre_verify_copy_async(seq)?;
    let chain = chain?;
    let drafts = forced_drafts(&chain, selector.accepted, LOGICAL_VOCAB)?;
    let inputs: Vec<u32> = std::iter::once(last_token)
        .chain(drafts.iter().copied())
        .collect();
    ensure!(
        inputs == selector.expected_inputs,
        "K16 fixture generated inputs do not match stage/commit selectors; generated={}",
        inputs
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(Some(PreparedFixture {
        frame: FixtureFrame {
            accepted: selector.accepted,
            observed: false,
        },
        drafts,
    }))
}

impl FixtureFrame {
    pub(super) fn observe(&mut self, drafts: &[u32], verified: &[u32], serial: bool) -> Result<()> {
        ensure!(serial, "K16 fixture requires the target serial oracle");
        ensure!(
            drafts.len() == DRAFTS && verified.len() == K16,
            "K16 fixture row mismatch"
        );
        let observed = drafts
            .iter()
            .zip(verified)
            .take_while(|(draft, target)| draft == target)
            .count();
        ensure!(
            observed == self.accepted,
            "K16 fixture accepted {observed}, expected {}",
            self.accepted
        );
        self.observed = true;
        Ok(())
    }

    pub(super) fn finish(
        &self,
        mut evidence: EvidenceGuard<'_>,
        num_accepted: usize,
        pre_verify_len: usize,
        inputs: &[u32],
    ) -> Result<()> {
        ensure!(
            self.observed && num_accepted == self.accepted,
            "K16 fixture commit mismatch"
        );
        ensure!(inputs.len() == K16, "K16 fixture committed non-K16 frame");
        evidence.confirm(pre_verify_len, inputs)?;
        tracing::info!(
            accepted_drafts = self.accepted,
            total_accepted = self.accepted + 1,
            inputs = %inputs.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
            target_authoritative = true,
            stage_count = 51,
            terminal_stage = "logits",
            full_logits_compared = true,
            batched_entries = 1,
            exact_qkv_layers = 12,
            exact_ssm_layers = 36,
            committed_buffers_required = 146,
            performance_claim_allowed = false,
            "DFLASH_K16_ACCEPTANCE_FIXTURE match=true"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "verify_dflash_k16_fixture_tests.rs"]
mod tests;
