// SPDX-License-Identifier: AGPL-3.0-only

//! The five contracted DFlash2 capture slots in the arena.
//!
//! The GLM DFlash2 drafter taps the widened mHC streams after layers
//! zero-based layers `[4, 13, 23, 32, 41]` (`GLM53_CAPTURE_LAYERS`), which
//! correspond to the checkpoint's one-based IDs `[5, 14, 24, 33, 42]`.
//! DFlash2 consumes the five
//! post-layer mHC states only after `hc_post` has been contracted from four F32
//! streams to one `[4096]` BF16 hidden vector. Its `fc.weight` is therefore
//! `[4096, 5 * 4096]`, not `[4096, 5 * 4 * 4096]`.
//!
//! The 327,680-byte arena reservation is exactly large enough to retain eight
//! verifier rows for every tap (`5 * 8 * 4096 * sizeof(BF16)`). Serial decode
//! uses row zero of each slot; K8 verification uses the complete slot.
//!
//! The dispatcher obtains the exact destination from this object and launches
//! the same ordered mHC contraction used by the target head. Ordering remains
//! load-bearing: a capture dispatched before that layer's `PostFfn` would
//! record the wrong residual.
//!
//! **Not yet transaction-bound.** The executor's seam map attributes this event
//! to a T1 capture that proves `CompleteT1Scratch`, and the copy here does not
//! yet publish or verify a completion receipt. The capability gate stays closed,
//! so this cannot be mistaken for a discharged capability.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layers::ops::GgmlIqBuffer;

use super::arena::GLM53_DFLASH_CAPTURE_BYTES;
use super::forward_one::GLM53_CAPTURE_LAYERS;

/// One target layer's contracted hidden state: 4096 BF16 values.
pub const GLM53_CAPTURE_SLOT_BYTES: u64 = 4096 * 2;
pub const GLM53_CAPTURE_PAYLOAD_BYTES: u64 = GLM53_CAPTURE_SLOT_BYTES * SLOTS as u64;
pub const GLM53_CAPTURE_MAX_ROWS: u32 = 8;
const GLM53_CAPTURE_SLOT_STRIDE_BYTES: u64 =
    GLM53_CAPTURE_SLOT_BYTES * GLM53_CAPTURE_MAX_ROWS as u64;

const SLOTS: usize = GLM53_CAPTURE_LAYERS.len();

const _: () = {
    assert!(SLOTS == 5);
    assert!(GLM53_CAPTURE_SLOT_BYTES == 8_192);
    assert!(GLM53_CAPTURE_PAYLOAD_BYTES == 40_960);
    assert!(GLM53_CAPTURE_SLOT_STRIDE_BYTES * SLOTS as u64 == GLM53_DFLASH_CAPTURE_BYTES);
};

/// The capture region, addressed by slot.
#[derive(Debug, Clone, Copy)]
pub struct Glm53CaptureSlots {
    base: DevicePtr,
}

impl Glm53CaptureSlots {
    pub fn bind(base: DevicePtr, region_bytes: u64) -> Result<Self> {
        ensure!(!base.is_null(), "GLM capture region base is NULL");
        ensure!(
            base.0.is_multiple_of(256),
            "GLM capture region base is 256-byte misaligned"
        );
        ensure!(
            region_bytes >= GLM53_DFLASH_CAPTURE_BYTES,
            "GLM capture region is {region_bytes} bytes, need {GLM53_DFLASH_CAPTURE_BYTES}"
        );
        Ok(Self { base })
    }

    /// The device buffer for one capture slot.
    pub fn slot(&self, slot: u32) -> Result<GgmlIqBuffer> {
        self.slot_rows(slot, 1)
    }

    /// Contiguous `[rows,4096]` BF16 capture storage for one target tap.
    pub fn slot_rows(&self, slot: u32, rows: u32) -> Result<GgmlIqBuffer> {
        let index = usize::try_from(slot)?;
        ensure!(
            index < SLOTS,
            "GLM capture slot {slot} is outside 0..{SLOTS}"
        );
        ensure!(
            (1..=GLM53_CAPTURE_MAX_ROWS).contains(&rows),
            "GLM capture rows {rows} are outside 1..={GLM53_CAPTURE_MAX_ROWS}"
        );
        let offset = (index as u64)
            .checked_mul(GLM53_CAPTURE_SLOT_STRIDE_BYTES)
            .context("GLM capture slot offset overflow")?;
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(
                self.base
                    .0
                    .checked_add(offset)
                    .context("GLM capture slot address overflow")?,
            ),
            bytes: usize::try_from(GLM53_CAPTURE_SLOT_BYTES * u64::from(rows))?,
        })
    }

    /// One retained verifier row from one target tap.
    pub fn slot_row(&self, slot: u32, row: u32) -> Result<GgmlIqBuffer> {
        ensure!(
            row < GLM53_CAPTURE_MAX_ROWS,
            "GLM capture row {row} is outside 0..{GLM53_CAPTURE_MAX_ROWS}"
        );
        let slot = self.slot_rows(slot, GLM53_CAPTURE_MAX_ROWS)?;
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(slot.ptr.0 + u64::from(row) * GLM53_CAPTURE_SLOT_BYTES),
            bytes: usize::try_from(GLM53_CAPTURE_SLOT_BYTES)?,
        })
    }

    /// The slot a given target layer captures into.
    ///
    /// Returns `Err` for a layer that is not a tap: capturing an unscheduled
    /// layer would overwrite another tap's state with the wrong residual.
    pub fn slot_for_layer(layer: u32) -> Result<u32> {
        GLM53_CAPTURE_LAYERS
            .iter()
            .position(|&tap| tap == layer)
            .map(|index| index as u32)
            .with_context(|| format!("GLM layer {layer} is not a DFlash2 capture tap"))
    }

    /// Resolve the packed BF16 destination for one scheduled target tap.
    pub fn contracted_slot(&self, layer: u32, slot: u32) -> Result<GgmlIqBuffer> {
        self.contracted_slot_rows(layer, slot, 1)
    }

    pub fn contracted_slot_rows(&self, layer: u32, slot: u32, rows: u32) -> Result<GgmlIqBuffer> {
        // The schedule's slot and the layer's own tap index must agree, or the
        // walk is writing a layer's state into another layer's slot.
        let expected = Self::slot_for_layer(layer)?;
        ensure!(
            expected == slot,
            "GLM capture layer {layer} belongs in slot {expected}, not {slot}"
        );
        self.slot_rows(slot, rows)
    }
}

#[cfg(test)]
#[path = "capture_slots_tests.rs"]
mod tests;
