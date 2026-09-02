// SPDX-License-Identifier: AGPL-3.0-only

//! Exact scratch-arena planning for the GLM-5.3 DFlash2 drafter.
//!
//! The five draft layers run sequentially, so their transient attention,
//! convolution, and MLP buffers are reused rather than multiplied by five.
//! Selector anchors remain an external input from the verified target token.
//! Target context and absolute positions remain external and span the target's
//! full 1,048,576-token domain (positions 0..=1,048,575). The 2,048-token bound
//! here is only the DFlash2 drafter's locally visible context chunk and
//! persistent K/V ring/window, not a target-context limit.

use anyhow::{Context, Result, bail};

const ALIGNMENT: usize = 256;
const BF16_BYTES: u64 = 2;
const U32_BYTES: u64 = 4;
const F32_BYTES: u64 = 4;
const PHASES: u32 = 2;
const CONV_KERNEL: u32 = 2;
const CONV_GROUPS: u32 = 256;
const SLIDING_WINDOW: u32 = 2048;
const MAX_DRAFT_TOKENS: u32 = 8;
const MAX_COMBINED_TOKENS: u32 = SLIDING_WINDOW + MAX_DRAFT_TOKENS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2RuntimeGeometry {
    pub batch: u32,
    /// Current DFlash2-local visible chunk, not total target context length.
    pub context_chunk_tokens: u32,
    pub draft_tokens: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_target_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
}

impl Glm53Dflash2RuntimeGeometry {
    pub fn exact(batch: u32, context_chunk_tokens: u32, draft_tokens: u32) -> Self {
        Self {
            batch,
            context_chunk_tokens,
            draft_tokens,
            hidden_size: 4096,
            intermediate_size: 12288,
            num_layers: 5,
            num_target_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 154880,
            selector_rank: 256,
            selector_top_k: 16,
        }
    }

    fn validate(self) -> Result<()> {
        if self.batch == 0
            || self.context_chunk_tokens == 0
            || self.context_chunk_tokens > SLIDING_WINDOW
            || self.draft_tokens == 0
            || self.draft_tokens > MAX_DRAFT_TOKENS
        {
            bail!(
                "GLM DFlash2 runtime requires batch>0, local-context-chunk1..2048, and draft1..8"
            );
        }
        let combined_tokens = self
            .context_chunk_tokens
            .checked_add(self.draft_tokens)
            .context("GLM DFlash2 context-plus-draft token overflow")?;
        if combined_tokens > MAX_COMBINED_TOKENS {
            bail!("GLM DFlash2 context-plus-draft window exceeds 2056 tokens");
        }
        if (
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_target_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.selector_rank,
            self.selector_top_k,
        ) != (4096, 12288, 5, 5, 32, 8, 128, 154880, 256, 16)
        {
            bail!("GLM DFlash2 runtime geometry drift");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2ScratchRegion {
    pub offset: usize,
    pub bytes: usize,
}

impl Glm53Dflash2ScratchRegion {
    pub fn end(self) -> Result<usize> {
        self.offset
            .checked_add(self.bytes)
            .context("GLM DFlash2 scratch-region end overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2ScratchPlan {
    pub geometry: Glm53Dflash2RuntimeGeometry,
    pub arena_bytes: usize,
    pub target_capture_bytes: usize,
    /// External five-layer, 2,048-token DFlash2 K/V ring/window allocation.
    pub persistent_draft_kv_bytes: usize,
    pub stream_a: Glm53Dflash2ScratchRegion,
    pub stream_b: Glm53Dflash2ScratchRegion,
    pub norm: Glm53Dflash2ScratchRegion,
    pub projected_target: Glm53Dflash2ScratchRegion,
    pub query: Glm53Dflash2ScratchRegion,
    pub key: Glm53Dflash2ScratchRegion,
    pub value: Glm53Dflash2ScratchRegion,
    pub attention: Glm53Dflash2ScratchRegion,
    pub dynamic_conv: Glm53Dflash2ScratchRegion,
    pub mlp_gate: Glm53Dflash2ScratchRegion,
    pub mlp_up: Glm53Dflash2ScratchRegion,
    pub mlp_intermediate: Glm53Dflash2ScratchRegion,
    pub logits: Glm53Dflash2ScratchRegion,
    pub selector_hidden: Glm53Dflash2ScratchRegion,
    pub candidate_ids: Glm53Dflash2ScratchRegion,
    pub candidate_scores: Glm53Dflash2ScratchRegion,
    pub topk_status: Glm53Dflash2ScratchRegion,
    pub selector_status: Glm53Dflash2ScratchRegion,
    pub chosen_ids: Glm53Dflash2ScratchRegion,
}

impl Glm53Dflash2ScratchPlan {
    pub const CHECKPOINT_PAYLOAD_BYTES: usize = 2_342_160_896;

    pub fn new(geometry: Glm53Dflash2RuntimeGeometry) -> Result<Self> {
        geometry.validate()?;
        let draft_rows = u64::from(geometry.batch)
            .checked_mul(u64::from(geometry.draft_tokens))
            .context("GLM DFlash2 draft row overflow")?;
        let context_rows = u64::from(geometry.batch)
            .checked_mul(u64::from(geometry.context_chunk_tokens))
            .context("GLM DFlash2 context row overflow")?;
        let combined_tokens = geometry
            .context_chunk_tokens
            .checked_add(geometry.draft_tokens)
            .context("GLM DFlash2 combined token overflow")?;
        let kv_rows = u64::from(geometry.batch)
            .checked_mul(u64::from(combined_tokens))
            .context("GLM DFlash2 K/V row overflow")?;
        let hidden = bytes(draft_rows, u64::from(geometry.hidden_size), BF16_BYTES)?;
        let projected_target = bytes(context_rows, u64::from(geometry.hidden_size), BF16_BYTES)?;
        let kv_width = u64::from(geometry.num_key_value_heads)
            .checked_mul(u64::from(geometry.head_dim))
            .context("GLM DFlash2 KV width overflow")?;
        let kv = bytes(kv_rows, kv_width, BF16_BYTES)?;
        let dynamic_width = u64::from(PHASES)
            .checked_mul(u64::from(CONV_KERNEL))
            .and_then(|value| value.checked_mul(u64::from(CONV_GROUPS)))
            .context("GLM DFlash2 dynamic-convolution width overflow")?;
        let dynamic_conv = bytes(draft_rows, dynamic_width, BF16_BYTES)?;
        let mlp = bytes(
            draft_rows,
            u64::from(geometry.intermediate_size),
            BF16_BYTES,
        )?;
        // T is capacity. A proposal may activate only T-1 predicted rows.
        let logits = bytes(draft_rows, u64::from(geometry.vocab_size), BF16_BYTES)?;
        let selector_hidden = bytes(draft_rows, u64::from(geometry.selector_rank), BF16_BYTES)?;
        let candidates = bytes(draft_rows, u64::from(geometry.selector_top_k), U32_BYTES)?;
        let scores = bytes(draft_rows, u64::from(geometry.selector_top_k), F32_BYTES)?;
        let topk_status_bytes = bytes(draft_rows, 1, U32_BYTES)?;
        let selector_status_bytes = bytes(u64::from(geometry.batch), 1, U32_BYTES)?;
        let chosen = bytes(draft_rows, 1, U32_BYTES)?;
        let target_width = u64::from(geometry.num_target_layers)
            .checked_mul(u64::from(geometry.hidden_size))
            .context("GLM DFlash2 target-capture width overflow")?;
        let target_capture_bytes = bytes(context_rows, target_width, BF16_BYTES)?;
        let persistent_kv_width = u64::from(geometry.num_layers)
            .checked_mul(2)
            .and_then(|value| value.checked_mul(u64::from(SLIDING_WINDOW)))
            .and_then(|value| value.checked_mul(kv_width))
            .context("GLM DFlash2 persistent K/V width overflow")?;
        let persistent_draft_kv_bytes =
            bytes(u64::from(geometry.batch), persistent_kv_width, BF16_BYTES)?;

        let mut cursor = 0usize;
        let stream_a = place(&mut cursor, hidden)?;
        let stream_b = place(&mut cursor, hidden)?;
        let norm = place(&mut cursor, hidden)?;
        let projected_target = place(&mut cursor, projected_target)?;
        let query = place(&mut cursor, hidden)?;
        let key = place(&mut cursor, kv)?;
        let value = place(&mut cursor, kv)?;
        let attention = place(&mut cursor, hidden)?;
        let dynamic_conv = place(&mut cursor, dynamic_conv)?;
        let mlp_gate = place(&mut cursor, mlp)?;
        let mlp_up = place(&mut cursor, mlp)?;
        let mlp_intermediate = place(&mut cursor, mlp)?;
        let logits = place(&mut cursor, logits)?;
        let selector_hidden = place(&mut cursor, selector_hidden)?;
        let candidate_ids = place(&mut cursor, candidates)?;
        let candidate_scores = place(&mut cursor, scores)?;
        let topk_status = place(&mut cursor, topk_status_bytes)?;
        let selector_status = place(&mut cursor, selector_status_bytes)?;
        let chosen_ids = place(&mut cursor, chosen)?;
        let arena_bytes = align_up(cursor)?;

        let plan = Self {
            geometry,
            arena_bytes,
            target_capture_bytes,
            persistent_draft_kv_bytes,
            stream_a,
            stream_b,
            norm,
            projected_target,
            query,
            key,
            value,
            attention,
            dynamic_conv,
            mlp_gate,
            mlp_up,
            mlp_intermediate,
            logits,
            selector_hidden,
            candidate_ids,
            candidate_scores,
            topk_status,
            selector_status,
            chosen_ids,
        };
        plan.validate_layout()?;
        Ok(plan)
    }

    pub fn regions(self) -> [(&'static str, Glm53Dflash2ScratchRegion); 19] {
        [
            ("stream_a", self.stream_a),
            ("stream_b", self.stream_b),
            ("norm", self.norm),
            ("projected_target", self.projected_target),
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("attention", self.attention),
            ("dynamic_conv", self.dynamic_conv),
            ("mlp_gate", self.mlp_gate),
            ("mlp_up", self.mlp_up),
            ("mlp_intermediate", self.mlp_intermediate),
            ("logits", self.logits),
            ("selector_hidden", self.selector_hidden),
            ("candidate_ids", self.candidate_ids),
            ("candidate_scores", self.candidate_scores),
            ("topk_status", self.topk_status),
            ("selector_status", self.selector_status),
            ("chosen_ids", self.chosen_ids),
        ]
    }

    pub fn validate_layout(self) -> Result<()> {
        let mut previous_end = 0usize;
        for (name, region) in self.regions() {
            let expected = align_up(previous_end)?;
            if region.offset != expected || region.offset % ALIGNMENT != 0 || region.bytes == 0 {
                bail!("GLM DFlash2 scratch region {name} has invalid alignment or extent");
            }
            previous_end = region.end()?;
        }
        if align_up(previous_end)? != self.arena_bytes {
            bail!("GLM DFlash2 scratch arena terminal extent drift");
        }
        Ok(())
    }
}

fn bytes(rows: u64, width: u64, element_bytes: u64) -> Result<usize> {
    let value = rows
        .checked_mul(width)
        .and_then(|count| count.checked_mul(element_bytes))
        .context("GLM DFlash2 scratch byte overflow")?;
    usize::try_from(value).context("GLM DFlash2 scratch exceeds host address space")
}

fn align_up(value: usize) -> Result<usize> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|end| end & !(ALIGNMENT - 1))
        .context("GLM DFlash2 scratch alignment overflow")
}

fn place(cursor: &mut usize, bytes: usize) -> Result<Glm53Dflash2ScratchRegion> {
    let offset = align_up(*cursor)?;
    *cursor = offset
        .checked_add(bytes)
        .context("GLM DFlash2 scratch placement overflow")?;
    Ok(Glm53Dflash2ScratchRegion { offset, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context8_draft8_single_spark_working_set_is_exact_and_packed() {
        let plan =
            Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 8, 8)).unwrap();
        assert_eq!(plan.arena_bytes, 3_548_928);
        assert_eq!(plan.target_capture_bytes, 327_680);
        assert_eq!(plan.persistent_draft_kv_bytes, 41_943_040);
        assert_eq!(plan.projected_target.bytes, 65_536);
        assert_eq!(plan.key.bytes, 32_768);
        assert_eq!(plan.value.bytes, 32_768);
        assert_eq!(plan.logits.bytes, 2_478_080);
        assert_eq!(plan.dynamic_conv.bytes, 16_384);
        assert_eq!(plan.mlp_intermediate.bytes, 196_608);
        assert_eq!(plan.topk_status.bytes, 8 * 4);
        assert_eq!(plan.selector_status.bytes, 4);
        assert!(plan.topk_status.end().unwrap() <= plan.selector_status.offset);
        assert!(plan.selector_status.end().unwrap() <= plan.chosen_ids.offset);
        assert_eq!(
            Glm53Dflash2ScratchPlan::CHECKPOINT_PAYLOAD_BYTES,
            2_342_160_896
        );
        plan.validate_layout().unwrap();
    }

    #[test]
    fn full_draft_visible_window_chunk_keeps_external_state_separate() {
        let plan =
            Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 2048, 8)).unwrap();
        assert_eq!(plan.arena_bytes, 28_616_448);
        assert_eq!(plan.target_capture_bytes, 83_886_080);
        assert_eq!(plan.persistent_draft_kv_bytes, 41_943_040);
        assert_eq!(plan.projected_target.bytes, 16_777_216);
        assert_eq!(plan.key.bytes, 4_210_688);
        assert_eq!(plan.value.bytes, 4_210_688);
    }

    #[test]
    fn invalid_block_and_architecture_drift_fail_closed() {
        assert!(Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(0, 8, 8)).is_err());
        assert!(Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 0, 8)).is_err());
        assert!(Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 8, 0)).is_err());
        assert!(
            Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 2049, 8)).is_err()
        );
        assert!(Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 8, 9)).is_err());
        let mut drift = Glm53Dflash2RuntimeGeometry::exact(1, 8, 8);
        drift.hidden_size = 2048;
        assert!(Glm53Dflash2ScratchPlan::new(drift).is_err());
    }

    #[test]
    fn layout_validator_rejects_overlap_and_unaligned_offsets() {
        let plan =
            Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(1, 8, 8)).unwrap();
        let mut overlap = plan;
        overlap.stream_b.offset = overlap.stream_a.offset;
        assert!(overlap.validate_layout().is_err());
        let mut unaligned = plan;
        unaligned.query.offset += 1;
        assert!(unaligned.validate_layout().is_err());
    }
}
