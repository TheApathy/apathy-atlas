// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheCap {
    pub block_size: usize,
    pub max_seq_len: usize,
    pub max_batch_size: usize,
    pub cap_tokens: Option<usize>,
    pub high_speed_swap: bool,
}

impl KvCacheCap {
    /// Validate an explicit total KV-token cap and return its exact block cap.
    /// `None` preserves the budget-driven pool without inventing a default.
    pub fn validated_blocks(self) -> Result<Option<usize>> {
        let Some(cap_tokens) = self.cap_tokens else {
            return Ok(None);
        };
        if self.high_speed_swap {
            bail!("--kv-cache-cap-tokens cannot be combined with --high-speed-swap");
        }
        if self.block_size == 0 {
            bail!("KV cache block size must be greater than zero");
        }
        if cap_tokens == 0 || !cap_tokens.is_multiple_of(self.block_size) {
            bail!(
                "--kv-cache-cap-tokens must be a positive multiple of --block-size={} (got {cap_tokens})",
                self.block_size,
            );
        }
        let required_blocks = self
            .max_seq_len
            .div_ceil(self.block_size)
            .checked_mul(self.max_batch_size)
            .ok_or_else(|| anyhow::anyhow!("KV active-sequence block requirement overflow"))?;
        let cap_blocks = cap_tokens / self.block_size;
        if cap_blocks < required_blocks {
            bail!(
                "--kv-cache-cap-tokens={cap_tokens} provides {cap_blocks} blocks, but \
                 --max-seq-len={} x --max-batch-size={} requires {required_blocks} blocks",
                self.max_seq_len,
                self.max_batch_size,
            );
        }
        Ok(Some(cap_blocks))
    }

    pub(crate) fn resolve_num_blocks(self, budget_blocks: usize) -> Result<usize> {
        Ok(self
            .validated_blocks()?
            .map_or(budget_blocks, |cap| budget_blocks.min(cap)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(cap_tokens: Option<usize>) -> KvCacheCap {
        KvCacheCap {
            block_size: 16,
            max_seq_len: 1_024,
            max_batch_size: 1,
            cap_tokens,
            high_speed_swap: false,
        }
    }

    #[test]
    fn leaves_budget_driven_pool_unchanged_without_an_explicit_cap() {
        assert_eq!(cap(None).resolve_num_blocks(7_720).unwrap(), 7_720);
    }

    #[test]
    fn caps_a_single_stream_pool_at_the_explicit_token_capacity() {
        assert_eq!(cap(Some(1_024)).resolve_num_blocks(7_720).unwrap(), 64);
    }

    #[test]
    fn rejects_a_cap_that_cannot_hold_every_active_sequence() {
        let mut value = cap(Some(1_024));
        value.max_batch_size = 2;
        let error = value.validated_blocks().unwrap_err().to_string();
        assert!(error.contains("128 blocks"));
        assert!(error.contains("64 blocks"));
    }

    #[test]
    fn rejects_zero_and_non_block_aligned_caps() {
        assert!(cap(Some(0)).validated_blocks().is_err());
        assert!(cap(Some(1_023)).validated_blocks().is_err());
    }

    #[test]
    fn rejects_combining_the_cap_with_high_speed_swap() {
        let mut value = cap(Some(1_024));
        value.high_speed_swap = true;
        assert!(value.validated_blocks().is_err());
    }

    #[test]
    fn a_cap_is_a_maximum_not_a_requested_allocation() {
        assert_eq!(cap(Some(2_048)).resolve_num_blocks(96).unwrap(), 96);
    }
}
