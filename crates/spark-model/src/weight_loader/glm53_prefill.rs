// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded CPU schedule for streaming GLM-5.3-Flash target prefill.
//!
//! The schedule contains at most 17 small descriptors. It allocates no token,
//! hidden-state, KDA, or DSA payload and never retains a whole-prompt hidden
//! buffer; execution and GPU qualification remain separate work.

use anyhow::{Context, Result, bail};

use super::glm53_context::GLM53_MAX_CONTEXT_TOKENS;

pub const GLM53_PREFILL_BLOCK_TOKENS: u32 = 16;
pub const GLM53_PREFILL_MAX_GRID_Y: u32 = 4_095;
pub const GLM53_PREFILL_MAX_CHUNK_TOKENS: u32 =
    GLM53_PREFILL_BLOCK_TOKENS * GLM53_PREFILL_MAX_GRID_Y;
pub const GLM53_PREFILL_MAX_CHUNKS: usize =
    ((GLM53_MAX_CONTEXT_TOKENS + GLM53_PREFILL_MAX_CHUNK_TOKENS - 1)
        / GLM53_PREFILL_MAX_CHUNK_TOKENS) as usize;

const INDEX_KPOOL: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53PrefillIndexState {
    pub complete_pools: u32,
    pub tail_tokens: u32,
}

impl Glm53PrefillIndexState {
    fn at_position(position: u32) -> Self {
        Self {
            complete_pools: position / INDEX_KPOOL,
            tail_tokens: position % INDEX_KPOOL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53PrefillContracts {
    pub kda_carry_in: bool,
    pub kda_carry_out: bool,
    pub dsa_latent_append: bool,
    pub dsa_index_append: bool,
    pub capture_streaming: bool,
    pub whole_prompt_hidden_buffer: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53PrefillChunk {
    /// Absolute token interval `[start, end)`.
    pub start: u32,
    pub end: u32,
    pub final_position: u32,
    pub token_count: u32,
    pub grid_y_blocks: u32,
    pub padded_tokens: u32,
    pub index_before: Glm53PrefillIndexState,
    pub index_after: Glm53PrefillIndexState,
    pub newly_completed_index_pools: u32,
    pub contracts: Glm53PrefillContracts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53PrefillSchedule {
    requested_tokens: u32,
    chunks: Vec<Glm53PrefillChunk>,
}

impl Glm53PrefillSchedule {
    pub fn new(requested_tokens: u32) -> Result<Self> {
        if requested_tokens == 0 || requested_tokens > GLM53_MAX_CONTEXT_TOKENS {
            bail!("GLM-5.3 prefill tokens must be in 1..={GLM53_MAX_CONTEXT_TOKENS}");
        }
        let chunk_count = requested_tokens.div_ceil(GLM53_PREFILL_MAX_CHUNK_TOKENS);
        let chunk_count =
            usize::try_from(chunk_count).context("GLM-5.3 prefill chunk count overflows usize")?;
        if chunk_count == 0 || chunk_count > GLM53_PREFILL_MAX_CHUNKS {
            bail!("GLM-5.3 prefill chunk count exceeds the bounded schedule");
        }

        let mut chunks = Vec::with_capacity(chunk_count);
        let mut start = 0u32;
        while start < requested_tokens {
            let remaining = requested_tokens
                .checked_sub(start)
                .context("GLM-5.3 prefill cursor exceeded requested length")?;
            let token_count = remaining.min(GLM53_PREFILL_MAX_CHUNK_TOKENS);
            let chunk = make_chunk(start, token_count)?;
            start = chunk.end;
            chunks.push(chunk);
        }
        let schedule = Self {
            requested_tokens,
            chunks,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn requested_tokens(&self) -> u32 {
        self.requested_tokens
    }

    pub fn chunks(&self) -> &[Glm53PrefillChunk] {
        &self.chunks
    }

    pub fn validate(&self) -> Result<()> {
        if self.requested_tokens == 0 || self.requested_tokens > GLM53_MAX_CONTEXT_TOKENS {
            bail!("GLM-5.3 prefill schedule has an invalid requested length");
        }
        if self.chunks.is_empty() || self.chunks.len() > GLM53_PREFILL_MAX_CHUNKS {
            bail!("GLM-5.3 prefill schedule has an invalid chunk count");
        }

        let mut expected_start = 0u32;
        let mut sum = 0u64;
        for (index, chunk) in self.chunks.iter().copied().enumerate() {
            if chunk.start != expected_start {
                bail!("GLM-5.3 prefill chunk {index} is not monotonic");
            }
            let expected = make_chunk(chunk.start, chunk.token_count)?;
            if chunk != expected {
                bail!("GLM-5.3 prefill chunk {index} violates its sealed contract");
            }
            expected_start = chunk.end;
            sum = sum
                .checked_add(u64::from(chunk.token_count))
                .context("GLM-5.3 prefill token sum overflow")?;
        }
        if expected_start != self.requested_tokens || sum != u64::from(self.requested_tokens) {
            bail!("GLM-5.3 prefill chunks do not exactly cover the request");
        }
        Ok(())
    }
}

fn make_chunk(start: u32, token_count: u32) -> Result<Glm53PrefillChunk> {
    if token_count == 0 || token_count > GLM53_PREFILL_MAX_CHUNK_TOKENS {
        bail!("GLM-5.3 prefill chunk token count is outside the safe range");
    }
    let end = start
        .checked_add(token_count)
        .context("GLM-5.3 prefill absolute end overflow")?;
    if end > GLM53_MAX_CONTEXT_TOKENS {
        bail!("GLM-5.3 prefill chunk exceeds the target context");
    }
    let final_position = end
        .checked_sub(1)
        .context("GLM-5.3 prefill chunk has no final position")?;
    let grid_y_blocks = token_count.div_ceil(GLM53_PREFILL_BLOCK_TOKENS);
    if grid_y_blocks == 0 || grid_y_blocks > GLM53_PREFILL_MAX_GRID_Y {
        bail!("GLM-5.3 prefill chunk exceeds the CUDA grid-Y contract");
    }
    let padded_tokens = grid_y_blocks
        .checked_mul(GLM53_PREFILL_BLOCK_TOKENS)
        .context("GLM-5.3 prefill padded token count overflow")?;
    let index_before = Glm53PrefillIndexState::at_position(start);
    let index_after = Glm53PrefillIndexState::at_position(end);
    let newly_completed_index_pools = index_after
        .complete_pools
        .checked_sub(index_before.complete_pools)
        .context("GLM-5.3 prefill index pool count regressed")?;

    Ok(Glm53PrefillChunk {
        start,
        end,
        final_position,
        token_count,
        grid_y_blocks,
        padded_tokens,
        index_before,
        index_after,
        newly_completed_index_pools,
        contracts: Glm53PrefillContracts {
            kda_carry_in: start != 0,
            kda_carry_out: true,
            dsa_latent_append: true,
            dsa_index_append: true,
            capture_streaming: true,
            whole_prompt_hidden_buffer: false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_full_context_has_seventeen_monotonic_chunks() {
        let schedule = Glm53PrefillSchedule::new(GLM53_MAX_CONTEXT_TOKENS).unwrap();
        assert_eq!(GLM53_PREFILL_MAX_CHUNKS, 17);
        assert_eq!(schedule.chunks().len(), 17);
        assert_eq!(schedule.requested_tokens(), 1_048_576);
        let mut sum = 0u64;
        let mut grid_blocks = 0u64;
        let mut expected_start = 0u32;
        for (index, chunk) in schedule.chunks().iter().enumerate() {
            assert_eq!(chunk.start, expected_start);
            assert!((1..=65_520).contains(&chunk.token_count));
            assert!(chunk.grid_y_blocks <= 4_095);
            assert_eq!(chunk.contracts.kda_carry_in, index != 0);
            assert!(chunk.contracts.kda_carry_out);
            assert!(chunk.contracts.dsa_latent_append);
            assert!(chunk.contracts.dsa_index_append);
            assert!(chunk.contracts.capture_streaming);
            assert!(!chunk.contracts.whole_prompt_hidden_buffer);
            expected_start = chunk.end;
            sum += u64::from(chunk.token_count);
            grid_blocks += u64::from(chunk.grid_y_blocks);
        }
        assert_eq!(sum, 1_048_576);
        assert_eq!(grid_blocks, 65_536);
        let last = schedule.chunks().last().unwrap();
        assert_eq!(last.token_count, 256);
        assert_eq!(last.end, 1_048_576);
        assert_eq!(last.final_position, 1_048_575);
        schedule.validate().unwrap();
    }

    #[test]
    fn maximum_chunk_and_partial_block_are_exact() {
        let exact = Glm53PrefillSchedule::new(GLM53_PREFILL_MAX_CHUNK_TOKENS).unwrap();
        assert_eq!(exact.chunks().len(), 1);
        assert_eq!(exact.chunks()[0].grid_y_blocks, 4_095);
        assert_eq!(exact.chunks()[0].padded_tokens, 65_520);

        let split = Glm53PrefillSchedule::new(GLM53_PREFILL_MAX_CHUNK_TOKENS + 1).unwrap();
        assert_eq!(split.chunks().len(), 2);
        assert_eq!(split.chunks()[1].token_count, 1);
        assert_eq!(split.chunks()[1].grid_y_blocks, 1);
        assert_eq!(split.chunks()[1].padded_tokens, 16);
    }

    #[test]
    fn kpool_transitions_preserve_complete_groups_and_tail() {
        let crossing = make_chunk(3, 2).unwrap();
        assert_eq!(
            crossing.index_before,
            Glm53PrefillIndexState {
                complete_pools: 0,
                tail_tokens: 3,
            }
        );
        assert_eq!(
            crossing.index_after,
            Glm53PrefillIndexState {
                complete_pools: 1,
                tail_tokens: 1,
            }
        );
        assert_eq!(crossing.newly_completed_index_pools, 1);

        let full = Glm53PrefillSchedule::new(GLM53_MAX_CONTEXT_TOKENS).unwrap();
        let last = full.chunks().last().unwrap();
        assert_eq!(last.index_after.complete_pools, 262_144);
        assert_eq!(last.index_after.tail_tokens, 0);
    }

    #[test]
    fn zero_overflow_and_out_of_range_chunks_fail_closed() {
        assert!(Glm53PrefillSchedule::new(0).is_err());
        assert!(Glm53PrefillSchedule::new(GLM53_MAX_CONTEXT_TOKENS + 1).is_err());
        assert!(make_chunk(0, 0).is_err());
        assert!(make_chunk(0, GLM53_PREFILL_MAX_CHUNK_TOKENS + 1).is_err());
        assert!(make_chunk(GLM53_MAX_CONTEXT_TOKENS, 1).is_err());
        assert!(make_chunk(u32::MAX, 1).is_err());
    }
}
