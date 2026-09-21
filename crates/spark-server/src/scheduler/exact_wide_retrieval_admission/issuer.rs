// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, ensure};

use super::{ExactWideFrameKey, ExactWideIdentities, MAX_VERIFY_K, MIN_VERIFY_K};

static NEXT_ISSUER_INSTANCE_NONCE: AtomicU64 = AtomicU64::new(1);

fn mint_issuer_instance_nonce() -> Result<u64> {
    NEXT_ISSUER_INSTANCE_NONCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |nonce| {
            nonce.checked_add(1)
        })
        .map_err(|_| anyhow::anyhow!("exact-wide issuer-instance nonce exhausted"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IssuerPhase {
    Idle,
    Published(ExactWideFrameKey),
    Verifying(ExactWideFrameKey),
    RawTargetSealed {
        key: ExactWideFrameKey,
        digest: [u8; 32],
    },
    AwaitingCommit(ExactWideFrameKey),
    CommitSealed {
        key: ExactWideFrameKey,
        next_epoch: u64,
        next_state: [u8; 32],
    },
}

#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub(in super::super) struct ExactWideRawTargetReceipt {
    key: ExactWideFrameKey,
    raw_target_tokens: Vec<u32>,
}

impl ExactWideRawTargetReceipt {
    pub(super) fn key(&self) -> ExactWideFrameKey {
        self.key
    }

    pub(super) fn raw_target_tokens(&self) -> &[u32] {
        &self.raw_target_tokens
    }

    #[cfg(test)]
    pub(super) fn duplicate_for_test(&self) -> Self {
        Self {
            key: self.key,
            raw_target_tokens: self.raw_target_tokens.clone(),
        }
    }

    #[cfg(test)]
    pub(super) fn forge_token_for_test(mut self) -> Self {
        self.raw_target_tokens[0] ^= 1;
        self
    }
}

#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub(super) struct ExactWideTargetStateReceipt {
    key: ExactWideFrameKey,
    next_epoch: u64,
    next_state: [u8; 32],
}

impl ExactWideTargetStateReceipt {
    #[cfg(test)]
    pub(super) fn duplicate_for_test(&self) -> Self {
        Self {
            key: self.key,
            next_epoch: self.next_epoch,
            next_state: self.next_state,
        }
    }

    #[cfg(test)]
    pub(super) fn forge_next_state_for_test(mut self) -> Self {
        self.next_state[0] ^= 1;
        self
    }
}

/// Scheduler-owned, non-clone mint and one-frame lifecycle authority.
#[derive(Debug)]
pub(in super::super) struct ExactWideReceiptIssuer {
    pub(super) session_nonce: u64,
    pub(super) instance_nonce: u64,
    pub(super) max_context: usize,
    pub(super) physical_verify_k: usize,
    pub(super) target_commit_epoch: u64,
    pub(super) next_receipt_nonce: u64,
    pub(super) next_proposal_epoch: u64,
    pub(super) identities: ExactWideIdentities,
    phase: IssuerPhase,
}

impl ExactWideReceiptIssuer {
    pub(super) fn new(
        session_nonce: u64,
        max_context: usize,
        physical_verify_k: usize,
        target_commit_epoch: u64,
        last_proposal_epoch: u64,
        identities: ExactWideIdentities,
    ) -> Result<Self> {
        ensure!(session_nonce != 0, "zero exact-wide session nonce");
        ensure!(max_context != 0, "zero exact-wide max context");
        ensure!(
            (MIN_VERIFY_K..=MAX_VERIFY_K).contains(&physical_verify_k),
            "physical verify K is outside exact-wide K17..32"
        );
        ensure!(target_commit_epoch != 0, "zero target commit epoch");
        identities.validate()?;
        let next_proposal_epoch = last_proposal_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("proposal epoch exhausted"))?;
        Ok(Self {
            session_nonce,
            instance_nonce: mint_issuer_instance_nonce()?,
            max_context,
            physical_verify_k,
            target_commit_epoch,
            next_receipt_nonce: 1,
            next_proposal_epoch,
            identities,
            phase: IssuerPhase::Idle,
        })
    }

    pub(super) fn require_idle(&self) -> Result<()> {
        ensure!(
            matches!(self.phase, IssuerPhase::Idle),
            "exact-wide receipt active"
        );
        Ok(())
    }

    pub(super) fn publish(&mut self, key: ExactWideFrameKey) -> Result<()> {
        self.require_idle()?;
        self.phase = IssuerPhase::Published(key);
        Ok(())
    }

    pub(super) fn begin_verify(&mut self, key: ExactWideFrameKey) -> Result<()> {
        ensure!(
            matches!(self.phase, IssuerPhase::Published(published) if published == key),
            "stale or cross-issuer exact-wide receipt"
        );
        self.phase = IssuerPhase::Verifying(key);
        Ok(())
    }

    pub(super) fn seal_raw_target(
        &mut self,
        key: ExactWideFrameKey,
        raw_target_tokens: &[u32],
    ) -> Result<ExactWideRawTargetReceipt> {
        ensure!(
            matches!(self.phase, IssuerPhase::Verifying(verifying) if verifying == key),
            "stale or out-of-order exact-wide target permit"
        );
        key.validate()?;
        ensure!(
            raw_target_tokens.len() == key.verify_k,
            "raw target row count mismatch"
        );
        ensure!(
            raw_target_tokens
                .iter()
                .all(|token| *token < key.vocab_size),
            "raw target token exceeds vocabulary"
        );
        let digest = raw_target_digest(raw_target_tokens)?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(raw_target_tokens.len())
            .map_err(|_| anyhow::anyhow!("raw target receipt allocation failed"))?;
        owned.extend_from_slice(raw_target_tokens);
        self.phase = IssuerPhase::RawTargetSealed { key, digest };
        Ok(ExactWideRawTargetReceipt {
            key,
            raw_target_tokens: owned,
        })
    }

    pub(super) fn consume_raw_target(
        &mut self,
        key: ExactWideFrameKey,
        receipt: ExactWideRawTargetReceipt,
    ) -> Result<()> {
        ensure!(receipt.key == key, "raw target receipt frame drift");
        let digest = raw_target_digest(&receipt.raw_target_tokens)?;
        ensure!(
            matches!(
                self.phase,
                IssuerPhase::RawTargetSealed {
                    key: sealed,
                    digest: expected,
                } if sealed == key && expected == digest
            ),
            "forged, replayed, or cross-frame raw target receipt"
        );
        self.phase = IssuerPhase::AwaitingCommit(key);
        Ok(())
    }

    pub(super) fn seal_target_state_commit(
        &mut self,
        next_state: [u8; 32],
    ) -> Result<ExactWideTargetStateReceipt> {
        let key = match self.phase {
            IssuerPhase::AwaitingCommit(key) => key,
            _ => anyhow::bail!("no exact-wide frame awaits target commit"),
        };
        ensure!(
            key.target_commit_epoch == self.target_commit_epoch,
            "frame target epoch drift"
        );
        ensure!(
            key.identities == self.identities,
            "frame target identity drift"
        );
        ensure!(
            next_state != [0; 32] && next_state != self.identities.target_state,
            "bad next state identity"
        );
        let next_epoch = self
            .target_commit_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("target commit epoch exhausted"))?;
        self.phase = IssuerPhase::CommitSealed {
            key,
            next_epoch,
            next_state,
        };
        Ok(ExactWideTargetStateReceipt {
            key,
            next_epoch,
            next_state,
        })
    }

    pub(super) fn record_target_commit(
        &mut self,
        receipt: ExactWideTargetStateReceipt,
    ) -> Result<()> {
        ensure!(
            matches!(
                self.phase,
                IssuerPhase::CommitSealed {
                    key,
                    next_epoch,
                    next_state,
                } if key == receipt.key
                    && next_epoch == receipt.next_epoch
                    && next_state == receipt.next_state
            ),
            "forged, replayed, or cross-frame target state receipt"
        );
        ensure!(
            receipt.key.target_commit_epoch == self.target_commit_epoch
                && receipt.key.identities == self.identities,
            "stale target state receipt"
        );
        ensure!(
            receipt.next_epoch
                == self
                    .target_commit_epoch
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("target commit epoch exhausted"))?,
            "out-of-order target commit epoch"
        );
        ensure!(
            receipt.next_state != [0; 32] && receipt.next_state != self.identities.target_state,
            "bad next state identity"
        );
        self.target_commit_epoch = receipt.next_epoch;
        self.identities.target_state = receipt.next_state;
        self.phase = IssuerPhase::Idle;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn instance_nonce_for_test(&self) -> u64 {
        self.instance_nonce
    }
}

const SHA256_INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_ROUNDS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub(super) fn canonical_prefix_digest(tokens: &[u32]) -> Result<[u8; 32]> {
    token_digest(b"atlas/exact-wide/canonical-prefix/v1\0", tokens)
}

fn raw_target_digest(tokens: &[u32]) -> Result<[u8; 32]> {
    token_digest(b"atlas/exact-wide/raw-target/v1\0", tokens)
}

fn token_digest(domain: &[u8], tokens: &[u32]) -> Result<[u8; 32]> {
    let token_bytes = tokens
        .len()
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("token digest extent overflow"))?;
    let capacity = domain
        .len()
        .checked_add(8)
        .and_then(|prefix| prefix.checked_add(token_bytes))
        .ok_or_else(|| anyhow::anyhow!("token digest extent overflow"))?;
    let bit_len = u64::try_from(capacity)
        .ok()
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or_else(|| anyhow::anyhow!("token digest bit length overflow"))?;
    let token_count = u64::try_from(tokens.len())
        .map_err(|_| anyhow::anyhow!("token count exceeds digest format"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| anyhow::anyhow!("token digest allocation failed"))?;
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&token_count.to_be_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_be_bytes());
    }
    debug_assert_eq!(bytes.len(), capacity);
    Ok(sha256(&bytes, bit_len))
}

fn sha256(bytes: &[u8], bit_len: u64) -> [u8; 32] {
    let mut state = SHA256_INITIAL;
    let mut chunks = bytes.chunks_exact(64);
    for chunk in &mut chunks {
        sha256_compress(&mut state, chunk.try_into().expect("exact SHA-256 block"));
    }
    let remainder = chunks.remainder();
    let mut tail = [0_u8; 128];
    tail[..remainder.len()].copy_from_slice(remainder);
    tail[remainder.len()] = 0x80;
    let padded_len = if remainder.len() < 56 { 64 } else { 128 };
    tail[padded_len - 8..padded_len].copy_from_slice(&bit_len.to_be_bytes());
    for chunk in tail[..padded_len].chunks_exact(64) {
        sha256_compress(&mut state, chunk.try_into().expect("exact SHA-256 tail"));
    }
    let mut digest = [0_u8; 32];
    for (word, output) in state.iter().zip(digest.chunks_exact_mut(4)) {
        output.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn sha256_compress(state: &mut [u32; 8], chunk: &[u8; 64]) {
    let mut schedule = [0_u32; 64];
    for (word, input) in schedule[..16].iter_mut().zip(chunk.chunks_exact(4)) {
        *word = u32::from_be_bytes(input.try_into().expect("exact SHA-256 word"));
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(SHA256_ROUNDS[index])
            .wrapping_add(schedule[index]);
        let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}
