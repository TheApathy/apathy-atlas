// SPDX-License-Identifier: AGPL-3.0-only

//! The five DFlash2 capture slots in the arena.
//!
//! The GLM DFlash2 drafter taps the widened mHC streams after layers
//! `[5, 14, 24, 33, 42]` (`GLM53_CAPTURE_LAYERS`). The arena reserves
//! `GLM53_DFLASH_CAPTURE_BYTES` = 327,680 bytes for this, which is exactly
//! `5 x 65,536` — one token's `[4 streams][4096 hidden]` F32 widened state per
//! slot. That the reservation tiles the capture set exactly is asserted here, so
//! a geometry change cannot leave the last tap writing past its region.
//!
//! Capture is a pure device-to-device copy of the post-layer stream state. It
//! reads the walk's `widened_hc` at the moment the schedule emits the event, so
//! ordering is the caller's responsibility: a capture dispatched before that
//! layer's `PostFfn` would record the wrong residual.
//!
//! **Not yet transaction-bound.** The executor's seam map attributes this event
//! to a T1 capture that proves `CompleteT1Scratch`, and the copy here does not
//! yet publish or verify a completion receipt. The capability gate stays closed,
//! so this cannot be mistaken for a discharged capability.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::GgmlIqBuffer;

use super::arena::GLM53_DFLASH_CAPTURE_BYTES;
use super::forward_one::GLM53_CAPTURE_LAYERS;

/// One token's widened mHC state: 4 streams x 4096 hidden, F32.
///
/// This tracks the workspace's `widened_hc` element width exactly, because
/// `capture` refuses a source whose extent differs from the slot's.
pub const GLM53_CAPTURE_SLOT_BYTES: u64 = 4 * 4096 * 4;

const SLOTS: usize = GLM53_CAPTURE_LAYERS.len();

const _: () = {
    assert!(SLOTS == 5);
    assert!(GLM53_CAPTURE_SLOT_BYTES == 65_536);
    assert!(GLM53_CAPTURE_SLOT_BYTES * SLOTS as u64 == GLM53_DFLASH_CAPTURE_BYTES);
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
        let index = usize::try_from(slot)?;
        ensure!(
            index < SLOTS,
            "GLM capture slot {slot} is outside 0..{SLOTS}"
        );
        let offset = (index as u64)
            .checked_mul(GLM53_CAPTURE_SLOT_BYTES)
            .context("GLM capture slot offset overflow")?;
        Ok(GgmlIqBuffer {
            ptr: DevicePtr(
                self.base
                    .0
                    .checked_add(offset)
                    .context("GLM capture slot address overflow")?,
            ),
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

    /// Copy the widened stream state into a slot.
    pub fn capture(
        &self,
        gpu: &dyn GpuBackend,
        layer: u32,
        slot: u32,
        widened_hc: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        // The schedule's slot and the layer's own tap index must agree, or the
        // walk is writing a layer's state into another layer's slot.
        let expected = Self::slot_for_layer(layer)?;
        ensure!(
            expected == slot,
            "GLM capture layer {layer} belongs in slot {expected}, not {slot}"
        );
        let destination = self.slot(slot)?;
        ensure!(
            widened_hc.bytes == destination.bytes,
            "GLM capture source is {} bytes, slot holds {}",
            widened_hc.bytes,
            destination.bytes
        );
        ensure!(!widened_hc.ptr.is_null(), "GLM capture source is NULL");
        // Stream-ordered: `widened_hc` is written by the mHC fold on `stream`.
        gpu.copy_d2d_async(widened_hc.ptr, destination.ptr, destination.bytes, stream)
    }
}

#[cfg(test)]
#[path = "capture_slots_tests.rs"]
mod tests;
