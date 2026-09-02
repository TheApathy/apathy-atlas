// SPDX-License-Identifier: AGPL-3.0-only

//! Dormant, CPU-only contract for the GLM-5.3 DFlash2 proposer.
//!
//! This module is intentionally not registered yet. Effectful target or drafter
//! code must obtain [`Glm53Dflash2Admission`] before allocating, capturing, or
//! launching anything.

use anyhow::{Result, ensure};

#[path = "glm53_dflash2_state.rs"]
mod state;

pub use state::{CaptureAppendPlan, Glm53Dflash2SequenceState};

pub const GLM53_DFLASH2_BLOCK_TOKENS: u8 = 8;
pub const GLM53_DFLASH2_RETURNED_DRAFTS: u8 = 7;
pub const GLM53_DFLASH2_WINDOW: u16 = 2_048;
pub const GLM53_DFLASH2_CAPTURE_LAYERS: [u8; 5] = [5, 14, 24, 33, 42];
pub const GLM53_MAX_POSITION_EXCLUSIVE: u64 = 1_048_576;
pub const GLM53_MAX_POSITION_INCLUSIVE: u64 = GLM53_MAX_POSITION_EXCLUSIVE - 1;

pub const GLM53_DFLASH2_CAPTURE_BYTES: u64 = 83_886_080;
pub const GLM53_DFLASH2_KV_BYTES: u64 = 41_943_040;
pub const GLM53_DFLASH2_ARENA_BYTES: u64 = 28_616_448;
pub const GLM53_DFLASH2_MMQ_SCRATCH_BYTES: u64 = 36_864;
pub const GLM53_DFLASH2_SEQUENCE_BYTES: u64 = GLM53_DFLASH2_CAPTURE_BYTES
    + GLM53_DFLASH2_KV_BYTES
    + GLM53_DFLASH2_ARENA_BYTES
    + GLM53_DFLASH2_MMQ_SCRATCH_BYTES;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingCursor {
    pub(crate) absolute_end: u64,
    pub(crate) head: u16,
    pub(crate) retained: u16,
}

impl RingCursor {
    pub const fn absolute_end(&self) -> u64 {
        self.absolute_end
    }

    pub const fn head(&self) -> u16 {
        self.head
    }

    pub const fn retained(&self) -> u16 {
        self.retained
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureSegment {
    pub(crate) destination_row: u16,
    pub(crate) rows: u16,
}

impl CaptureSegment {
    pub const fn destination_row(&self) -> u16 {
        self.destination_row
    }

    pub const fn rows(&self) -> u16 {
        self.rows
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureReadSegment {
    pub(crate) source_row: u16,
    pub(crate) rows: u16,
}

impl CaptureReadSegment {
    pub const fn source_row(&self) -> u16 {
        self.source_row
    }

    pub const fn rows(&self) -> u16 {
        self.rows
    }

    pub const fn source_end(&self) -> u16 {
        self.source_row + self.rows
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransactionId {
    pub(crate) generation: u64,
    pub(crate) nonce: u64,
}

impl TransactionId {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn nonce(&self) -> u64 {
        self.nonce
    }

    #[cfg(test)]
    pub(crate) const fn forged_for_test(generation: u64, nonce: u64) -> Self {
        Self { generation, nonce }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProposalPlan {
    pub(crate) txn: TransactionId,
    pub(crate) anchor_position: u64,
    pub(crate) logical_new_context_rows: u32,
    pub(crate) attention_new_context_rows: u16,
    pub(crate) absolute_context_end: u64,
    pub(crate) capture_oldest_position: u64,
    pub(crate) capture_retained_rows: u16,
    pub(crate) target_source_start_position: u64,
    pub(crate) capture_source_offset_rows: u16,
    pub(crate) capture_source_segments: [CaptureReadSegment; 2],
    pub(crate) kept_past_rows: u16,
    pub(crate) past_drop_rows: u16,
    pub(crate) local_context_rows: u16,
    pub(crate) route: Glm53Dflash2VerifyRoute,
}

impl ProposalPlan {
    pub const fn transaction(&self) -> TransactionId {
        self.txn
    }

    pub const fn anchor_position(&self) -> u64 {
        self.anchor_position
    }

    pub const fn logical_new_context_rows(&self) -> u32 {
        self.logical_new_context_rows
    }

    /// Row count supplied to attention after gathering the ring segments.
    pub const fn attention_new_context_rows(&self) -> u16 {
        self.attention_new_context_rows
    }

    /// The gathered attention input already begins at the selected tail.
    pub const fn attention_source_skip_rows(&self) -> u16 {
        0
    }

    pub const fn absolute_context_end(&self) -> u64 {
        self.absolute_context_end
    }

    pub const fn capture_oldest_position(&self) -> u64 {
        self.capture_oldest_position
    }

    pub const fn capture_retained_rows(&self) -> u16 {
        self.capture_retained_rows
    }

    pub const fn target_source_start_position(&self) -> u64 {
        self.target_source_start_position
    }

    pub const fn capture_source_offset_rows(&self) -> u16 {
        self.capture_source_offset_rows
    }

    pub const fn capture_source_segments(&self) -> [CaptureReadSegment; 2] {
        self.capture_source_segments
    }

    pub const fn target_tail_rows(&self) -> u16 {
        self.attention_new_context_rows
    }

    pub const fn kept_past_rows(&self) -> u16 {
        self.kept_past_rows
    }

    pub const fn past_drop_rows(&self) -> u16 {
        self.past_drop_rows
    }

    pub const fn local_context_rows(&self) -> u16 {
        self.local_context_rows
    }

    pub const fn local_context_start_position(&self) -> u64 {
        self.absolute_context_end - self.local_context_rows as u64
    }

    pub const fn noise_rows(&self) -> u8 {
        8
    }

    pub const fn returned_drafts(&self) -> u8 {
        GLM53_DFLASH2_RETURNED_DRAFTS
    }

    pub const fn route(&self) -> Glm53Dflash2VerifyRoute {
        self.route
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReleaseReceipt {
    pub(crate) lease_id: u64,
    pub(crate) generation: u64,
}

impl ReleaseReceipt {
    pub const fn lease_id(&self) -> u64 {
        self.lease_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn allocation_bytes(&self) -> u64 {
        GLM53_DFLASH2_SEQUENCE_BYTES
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53Dflash2VerifyRoute {
    /// The only legal route for the seven returned GLM DFlash2 drafts.
    DflashGamma8,
}

/// Fail-closed scheduler discriminator.
///
/// This check must run before generic K4 dispatch. A GLM DFlash2 length of
/// seven always uses `step_verify_dflash`; this API deliberately has no K4
/// variant.
pub fn verify_route_for_returned_drafts(returned_drafts: u8) -> Result<Glm53Dflash2VerifyRoute> {
    ensure!(
        returned_drafts == GLM53_DFLASH2_RETURNED_DRAFTS,
        "GLM DFlash2 requires exactly seven returned drafts"
    );
    Ok(Glm53Dflash2VerifyRoute::DflashGamma8)
}

#[derive(Clone, Copy, Debug)]
pub struct Glm53Dflash2Request {
    pub temperature: f32,
    pub anchor_position: u64,
    pub block_tokens: u8,
    pub returned_drafts: u8,
    pub window: u16,
}

impl Glm53Dflash2Request {
    pub const fn greedy(anchor_position: u64) -> Self {
        Self {
            temperature: 0.0,
            anchor_position,
            block_tokens: GLM53_DFLASH2_BLOCK_TOKENS,
            returned_drafts: GLM53_DFLASH2_RETURNED_DRAFTS,
            window: GLM53_DFLASH2_WINDOW,
        }
    }
}

/// Exact, sealed GLM DFlash2 policy. There is no configurable approximation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Glm53Dflash2Policy {
    _sealed: (),
}

impl Glm53Dflash2Policy {
    pub const fn exact() -> Self {
        Self { _sealed: () }
    }

    pub const fn capture_layers(&self) -> [u8; 5] {
        GLM53_DFLASH2_CAPTURE_LAYERS
    }

    pub fn admit_target_position(&self, position: u64) -> Result<()> {
        ensure!(
            position <= GLM53_MAX_POSITION_INCLUSIVE,
            "target position exceeds GLM-5.3 context"
        );
        Ok(())
    }

    /// Pure admission. Callers must not perform target or DFlash effects until
    /// this returns its unforgeable token.
    pub fn admit(&self, request: Glm53Dflash2Request) -> Result<Glm53Dflash2Admission> {
        ensure!(
            request.temperature.is_finite() && request.temperature == 0.0,
            "GLM DFlash2 is greedy-only"
        );
        ensure!(
            request.block_tokens == GLM53_DFLASH2_BLOCK_TOKENS,
            "GLM DFlash2 requires gamma eight"
        );
        let route = verify_route_for_returned_drafts(request.returned_drafts)?;
        ensure!(
            request.window == GLM53_DFLASH2_WINDOW,
            "GLM DFlash2 requires a 2048-token window"
        );
        let final_position = request
            .anchor_position
            .checked_add(u64::from(GLM53_DFLASH2_BLOCK_TOKENS - 1))
            .ok_or_else(|| anyhow::anyhow!("proposal position overflow"))?;
        self.admit_target_position(final_position)?;
        Ok(Glm53Dflash2Admission {
            anchor_position: request.anchor_position,
            route,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Glm53Dflash2Admission {
    anchor_position: u64,
    route: Glm53Dflash2VerifyRoute,
}

impl Glm53Dflash2Admission {
    pub const fn anchor_position(&self) -> u64 {
        self.anchor_position
    }

    pub const fn route(&self) -> Glm53Dflash2VerifyRoute {
        self.route
    }
}

#[cfg(test)]
#[path = "glm53_dflash2_proposer_tests.rs"]
mod tests;
