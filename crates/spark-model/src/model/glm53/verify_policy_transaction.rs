// SPDX-License-Identifier: AGPL-3.0-only

//! Policy-before-commit contract for the GLM EXL3 verifier.
//!
//! This helper owns logits, not GPU state. Its adapter must bind the current
//! model/request/stream and the saved DSA transaction before calling it. No
//! callback may emit, publish host sequence state, or retain borrowed rows.
//! Production registration and the ordinary scheduler policy adapter are
//! deliberately separate qualification gates.

use anyhow::{Context, Error, Result, ensure};
use std::sync::Arc;

#[path = "verify_policy_staging_pool.rs"]
mod staging_pool;

const MAX_ROWS: usize = 8;
const MAX_VOCAB: usize = 154_880;
const BF16_BYTES: usize = 2;

pub struct VerifyRequest {
    start: usize,
    vocab: usize,
    inputs: Vec<u32>,
    row_bytes: usize,
    total_bytes: usize,
    host_prefix: Option<Arc<[u32]>>,
}

impl VerifyRequest {
    pub fn new(start: usize, capacity: usize, vocab: usize, inputs: &[u32]) -> Result<Self> {
        ensure!(
            (2..=MAX_ROWS).contains(&inputs.len()),
            "GLM policy verify needs 2..=8 rows"
        );
        ensure!(
            (1..=MAX_VOCAB).contains(&vocab),
            "GLM policy vocabulary exceeds its bounded extent"
        );
        ensure!(
            inputs.iter().all(|&token| (token as usize) < vocab),
            "GLM policy input token is outside vocabulary"
        );
        let end = start
            .checked_add(inputs.len())
            .context("GLM policy position overflow")?;
        ensure!(end <= capacity, "GLM policy verify exceeds target capacity");
        let row_bytes = vocab
            .checked_mul(BF16_BYTES)
            .context("GLM policy row size overflow")?;
        let total_bytes = row_bytes
            .checked_mul(inputs.len())
            .context("GLM policy logits size overflow")?;
        Ok(Self {
            start,
            vocab,
            inputs: inputs.to_vec(),
            row_bytes,
            total_bytes,
            host_prefix: None,
        })
    }

    pub fn start(&self) -> usize {
        self.start
    }

    /// Anchor followed by drafts; immutable throughout selection and commit.
    pub fn inputs(&self) -> &[u32] {
        &self.inputs
    }

    pub(super) fn bind_host_prefix(&mut self, prefix: &[u32]) {
        self.host_prefix = Some(Arc::from(prefix));
    }

    pub fn host_prefix(&self) -> Option<&[u32]> {
        self.host_prefix.as_deref()
    }
}

pub trait LogitsIo {
    /// Copy the full staged BF16 extent and complete stream-ordered D2H before
    /// returning the actual initialized byte count. Short success is an error.
    /// The GPU adapter must validate its source extent, not use the legacy
    /// single-row `copy_logits` implementation with a larger destination.
    fn copy_logits(&mut self, destination: &mut [u8]) -> Result<usize>;

    /// Copy one already-selected token per staged row. The default refuses so
    /// no policy can silently replace full logits without an explicit source.
    fn copy_argmax(&mut self, _destination: &mut [u32], _excluded: [u32; 2]) -> Result<usize> {
        anyhow::bail!("GLM policy compact argmax source is unavailable")
    }
}

pub struct StagedLogits {
    bytes: staging_pool::StagingBytes,
    row_bytes: usize,
    rows: usize,
}

impl StagedLogits {
    pub fn read(request: &VerifyRequest, io: &mut dyn LogitsIo) -> Result<Self> {
        let mut bytes = staging_pool::StagingBytes::acquire(request.total_bytes)?;
        let copied = io
            .copy_logits(bytes.as_mut_slice())
            .context("GLM policy logits copy failed")?;
        ensure!(
            copied == bytes.len(),
            "GLM policy logits copy initialized {copied} of {} bytes",
            bytes.len()
        );
        Ok(Self {
            bytes,
            row_bytes: request.row_bytes,
            rows: request.inputs().len(),
        })
    }

    pub fn row(&self, index: usize) -> Result<&[u8]> {
        ensure!(
            index < self.rows,
            "GLM policy logits row is outside the captured extent"
        );
        let start = index
            .checked_mul(self.row_bytes)
            .context("GLM policy row offset overflow")?;
        let end = start
            .checked_add(self.row_bytes)
            .context("GLM policy row end overflow")?;
        self.bytes
            .as_slice()
            .get(start..end)
            .context("GLM policy row exceeds owned logits")
    }
}

/// A policy callback classifies unsupported persistent effects here, before
/// restoration can add unrelated failures. Errors never grant replay authority.
pub enum PolicyAdvance {
    Continue,
    Terminal,
    OrdinaryReplay,
}

pub trait VerifyPolicy {
    /// Establish an atomic checkpoint; failure must leave policy unchanged.
    /// This includes counters/history and the grammar's actual history depth.
    fn checkpoint(&mut self) -> Result<()>;

    /// True only when the ordinary policy proved raw BF16 argmax equivalent.
    fn compact_argmax_exclusions(&self) -> Option<[u32; 2]> {
        None
    }

    /// Select using the ordinary request policy at this accepted-prefix state.
    /// Caller bias, sampling settings, masks and penalties belong in the
    /// adapter, not a second sampler in this model helper.
    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32>;

    /// Consume one device-selected token after compact admission. The default
    /// refuses so existing policy implementations remain on full logits.
    fn pick_argmax(&mut self, _row: usize, _token: u32) -> Result<u32> {
        anyhow::bail!("GLM policy compact argmax was not admitted")
    }

    /// Advance only the reversible policy view, never real emission. Return
    /// Terminal when this pick ends generation. OrdinaryReplay requests a real
    /// ordinary round only after both owners restore. Grammar refusal is an
    /// error, not permission to drop the matcher or continue unconstrained.
    fn advance(&mut self, token: u32) -> Result<PolicyAdvance>;

    /// Restore every speculative mutation after a successful checkpoint,
    /// including on failed pick/advance. Production must check restored depth;
    /// stop/terminated tokens need not advance xgrammar history.
    fn restore(&mut self) -> Result<()>;
}

pub trait VerifyCommitIo {
    /// Commit exactly anchor + accepted drafts and their proposer captures.
    /// Any needed DSA restore/replay is internal to this operation. Return only
    /// after the stream is complete and the actual target position is known.
    /// An error can follow irreversible KDA mutation and must poison the model.
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize>;

    /// Restore/drain the saved staged DSA transaction before irreversible commit.
    fn abort_staged(&mut self) -> Result<()>;

    /// Fail closed until the caller resets/releases this sequence.
    fn poison(&mut self);
}

struct Selection {
    accepted_drafts: usize,
    emitted: Vec<u32>,
    terminal: bool,
}

enum SelectionOutcome {
    Selected(Selection),
    OrdinaryReplay,
}

enum PolicyRows {
    Full(StagedLogits),
    Compact {
        tokens: [u32; MAX_ROWS],
        rows: usize,
    },
}

impl PolicyRows {
    fn read(
        request: &VerifyRequest,
        source: &mut dyn LogitsIo,
        excluded: Option<[u32; 2]>,
    ) -> Result<Self> {
        let Some(excluded) = excluded else {
            return StagedLogits::read(request, source).map(Self::Full);
        };
        let rows = request.inputs().len();
        let mut tokens = [0u32; MAX_ROWS];
        let copied = source
            .copy_argmax(&mut tokens[..rows], excluded)
            .context("GLM policy compact argmax copy failed")?;
        ensure!(
            copied == rows,
            "GLM policy compact argmax row count mismatch"
        );
        ensure!(
            tokens[..rows]
                .iter()
                .all(|&token| (token as usize) < request.vocab),
            "GLM policy compact argmax token is outside vocabulary"
        );
        Ok(Self::Compact { tokens, rows })
    }

    fn pick(&self, row: usize, policy: &mut dyn VerifyPolicy) -> Result<u32> {
        match self {
            Self::Full(logits) => policy.pick(row, logits.row(row)?),
            Self::Compact { tokens, rows } => {
                ensure!(row < *rows, "GLM policy compact row is outside extent");
                policy.pick_argmax(row, tokens[row])
            }
        }
    }
}

fn select(
    request: &VerifyRequest,
    rows: &PolicyRows,
    policy: &mut dyn VerifyPolicy,
) -> Result<SelectionOutcome> {
    let mut selected = Selection {
        accepted_drafts: 0,
        emitted: Vec::with_capacity(request.inputs().len()),
        terminal: false,
    };
    for row in 0..request.inputs().len() {
        let token = rows.pick(row, policy)?;
        ensure!(
            (token as usize) < request.vocab,
            "GLM policy picked a token outside vocabulary"
        );
        let continues = match policy.advance(token)? {
            PolicyAdvance::Continue => true,
            PolicyAdvance::Terminal => false,
            PolicyAdvance::OrdinaryReplay => return Ok(SelectionOutcome::OrdinaryReplay),
        };
        selected.emitted.push(token);
        let accepted = request.inputs().get(row + 1) == Some(&token);
        if accepted {
            selected.accepted_drafts += 1;
        }
        if !continues {
            selected.terminal = true;
            break;
        }
        // First mismatch or the final bonus: later logits describe a prefix
        // that is no longer authoritative and must not affect policy state.
        if !accepted {
            break;
        }
    }
    Ok(SelectionOutcome::Selected(selected))
}

fn abort_before_commit(io: &mut dyn VerifyCommitIo, error: Error) -> Error {
    match io.abort_staged() {
        Ok(()) => error.context("GLM policy verification aborted before persistent commit"),
        Err(restore) => {
            io.poison();
            error.context(format!(
                "GLM staged restore failed; sequence poisoned: {restore:#}"
            ))
        }
    }
}

/// Produces no publishable receipt until policy is restored and target commit
/// succeeds. The adapter must validate host/model prefix identity before any
/// staged GPU effects, and must treat a later publication error as fatal.
pub fn run_verify_policy_transaction(
    request: &VerifyRequest,
    source: &mut dyn LogitsIo,
    policy: &mut dyn VerifyPolicy,
    target: &mut dyn VerifyCommitIo,
) -> Result<VerifyOutcome> {
    let rows = match PolicyRows::read(request, source, policy.compact_argmax_exclusions()) {
        Ok(rows) => rows,
        Err(error) => return Err(abort_before_commit(target, error)),
    };
    if let Err(error) = policy.checkpoint() {
        return Err(abort_before_commit(
            target,
            error.context("GLM policy checkpoint failed"),
        ));
    }
    let selection = select(request, &rows, policy);
    if let Err(restore) = policy.restore() {
        target.poison();
        let error = match selection {
            Ok(_) => restore.context("GLM speculative policy restore failed"),
            Err(error) => error.context(format!(
                "GLM speculative policy restore also failed: {restore:#}"
            )),
        };
        return Err(abort_before_commit(target, error));
    }
    let selection = match selection {
        Ok(SelectionOutcome::Selected(selection)) => selection,
        Ok(SelectionOutcome::OrdinaryReplay) => {
            return match target.abort_staged() {
                Ok(()) => Ok(VerifyOutcome::RestoredForOrdinaryReplay(RestoredVerify {
                    start: request.start(),
                    anchor: request.inputs()[0],
                    host_prefix: request.host_prefix.clone(),
                })),
                Err(error) => {
                    target.poison();
                    Err(error.context("GLM ordinary replay DSA restore failed; sequence poisoned"))
                }
            };
        }
        Err(error) => return Err(abort_before_commit(target, error)),
    };
    let rows = selection.accepted_drafts + 1;
    // `rows <= inputs.len()` by selection; the full end was checked at admission.
    let end = request.start() + rows;
    let committed_inputs = request.inputs()[..rows].to_vec();
    match target.commit_prefix(request, rows) {
        Ok(actual) if actual == end => {}
        Ok(actual) => {
            target.poison();
            anyhow::bail!(
                "GLM policy commit reached position {actual}, expected {end}; sequence poisoned"
            );
        }
        Err(error) => {
            target.poison();
            // A DSA-only abort cannot undo this operation's persistent KDA.
            return Err(error.context("GLM policy persistent commit failed; sequence poisoned"));
        }
    }
    Ok(VerifyOutcome::Committed(CommittedVerify {
        start: request.start(),
        end,
        committed_inputs,
        accepted_drafts: selection.accepted_drafts,
        emitted: selection.emitted,
        terminal: selection.terminal,
        host_prefix: request.host_prefix.clone(),
    }))
}

#[must_use = "only an explicit restored outcome permits ordinary replay"]
pub enum VerifyOutcome {
    Committed(CommittedVerify),
    RestoredForOrdinaryReplay(RestoredVerify),
}

/// No commit or emission authority: both policy and staged target are restored.
/// The bound caller must revalidate the start/anchor before one ordinary round.
pub struct RestoredVerify {
    start: usize,
    anchor: u32,
    host_prefix: Option<Arc<[u32]>>,
}

impl RestoredVerify {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn anchor(&self) -> u32 {
        self.anchor
    }

    pub fn validate_host_prefix(
        &self,
        tokens: &[u32],
        seq_len: usize,
        kv_valid: usize,
    ) -> Result<()> {
        validate_prefix(
            self.start,
            self.host_prefix.as_deref(),
            tokens,
            seq_len,
            kv_valid,
        )
    }
}

fn validate_prefix(
    start: usize,
    expected: Option<&[u32]>,
    tokens: &[u32],
    seq_len: usize,
    kv_valid: usize,
) -> Result<()> {
    ensure!(
        tokens.len() == start && seq_len == start && kv_valid == start,
        "GLM committed receipt does not match the host prefix"
    );
    if let Some(expected) = expected {
        ensure!(
            tokens == expected,
            "GLM receipt host prefix tokens changed after binding"
        );
    }
    Ok(())
}

/// Non-cloneable, single-use authority to publish the successfully committed
/// prefix. Emission tokens are accessible only after successful publication.
#[must_use = "publish the committed prefix or fail/reset the sequence without emission"]
pub struct CommittedVerify {
    start: usize,
    end: usize,
    committed_inputs: Vec<u32>,
    accepted_drafts: usize,
    emitted: Vec<u32>,
    terminal: bool,
    host_prefix: Option<Arc<[u32]>>,
}

impl CommittedVerify {
    pub fn accepted_drafts(&self) -> usize {
        self.accepted_drafts
    }

    /// Caller must poison/reset on error because the GPU commit already ran.
    /// Validate all three host extents before modifying any of their contents.
    pub fn publish(
        self,
        tokens: &mut Vec<u32>,
        seq_len: &mut usize,
        kv_valid: &mut usize,
    ) -> Result<PublishedVerify> {
        validate_prefix(
            self.start,
            self.host_prefix.as_deref(),
            tokens,
            *seq_len,
            *kv_valid,
        )?;
        tokens
            .try_reserve(self.committed_inputs.len())
            .context("GLM host prefix allocation failed after commit")?;
        tokens.extend_from_slice(&self.committed_inputs);
        *seq_len = self.end;
        *kv_valid = self.end;
        Ok(PublishedVerify {
            emitted: self.emitted,
            terminal: self.terminal,
        })
    }
}

#[must_use = "apply the qualified policy/emission transition after host publication"]
pub struct PublishedVerify {
    emitted: Vec<u32>,
    terminal: bool,
}

impl PublishedVerify {
    /// These are selected tokens, including suppressed terminal/control tokens,
    /// not a promise that every token belongs in the user-visible text stream.
    pub fn emitted_tokens(&self) -> &[u32] {
        &self.emitted
    }

    pub fn terminal(&self) -> bool {
        self.terminal
    }
}
