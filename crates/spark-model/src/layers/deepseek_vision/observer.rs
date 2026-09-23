// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit diagnostic observation; production forward never enables it.
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionStageDtype {
    Bf16,
    F32,
}

/// The pointer is borrowed until the callback returns. Copy it there if needed.
/// Callbacks run only after the default stream is synchronized.
pub type VisionObserver<'a> =
    dyn FnMut(&str, DevicePtr, [usize; 2], VisionStageDtype) -> Result<()> + 'a;

pub(super) fn observe(
    gpu: &dyn GpuBackend,
    observer: &mut Option<&mut VisionObserver<'_>>,
    name: &str,
    pointer: DevicePtr,
    shape: [usize; 2],
    dtype: VisionStageDtype,
) -> Result<()> {
    if let Some(callback) = observer.as_deref_mut() {
        gpu.synchronize(gpu.default_stream())?;
        callback(name, pointer, shape, dtype)?;
    }
    Ok(())
}
