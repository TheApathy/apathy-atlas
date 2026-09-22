// SPDX-License-Identifier: AGPL-3.0-only

//! Borrowed larger-capture dispatch; ordinary capture operands remain unchanged.
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm53Dispatcher;
use crate::layers::Glm53TargetEvent;
use crate::layers::ops::{GgmlIqBuffer, glm53_layer_major_prefill_active};
use crate::model::glm53::prefill_capture_ingest::CaptureReceipt;

impl Glm53Dispatcher<'_> {
    /// Borrow a checked receipt for this event only. Legacy verifier/scalar
    /// callers retain the original capture ABI and ordinary dispatch path.
    pub(crate) fn dispatch_with_prefill_capture(
        &self,
        gpu: &dyn GpuBackend,
        event: &Glm53TargetEvent,
        stream: u64,
        capture: Option<&mut CaptureReceipt>,
    ) -> Result<()> {
        if let (Some(receipt), Glm53TargetEvent::CaptureWidenedMhc { layer, slot }) =
            (capture, event)
        {
            ensure!(
                glm53_layer_major_prefill_active(),
                "large capture dispatch outside prefill scope"
            );
            return receipt.enqueue_tap(*layer, usize::try_from(*slot)?, |destination| {
                self.hyper.mean(
                    gpu,
                    self.plan,
                    self.bound.widened_hc,
                    GgmlIqBuffer {
                        ptr: DevicePtr(destination.address),
                        bytes: destination.bytes,
                    },
                    stream,
                )
            });
        }
        self.dispatch(gpu, event, stream)
    }
}
