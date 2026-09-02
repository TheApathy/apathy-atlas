// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

/// Records the exact effect order so a test can assert that the marker is
/// enqueued after every payload copy and that a rejected decision enqueues
/// nothing at all.
#[derive(Debug, PartialEq, Eq, Clone)]
enum Effect {
    Memset { ptr: u64, bytes: usize },
    Copy { src: u64, dst: u64, bytes: usize },
    Launch,
    D2h { ptr: u64, bytes: usize },
}

struct OrderGpu {
    inner: MockGpuBackend,
    effects: Mutex<Vec<Effect>>,
}

impl OrderGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            effects: Mutex::new(Vec::new()),
        }
    }
    fn effects(&self) -> Vec<Effect> {
        self.effects.lock().unwrap().clone()
    }
}

impl GpuBackend for OrderGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], _stream: u64) -> Result<()> {
        self.effects.lock().unwrap().push(Effect::D2h {
            ptr: src.0,
            bytes: dst.len(),
        });
        Ok(())
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.effects.lock().unwrap().push(Effect::Copy {
            src: src.0,
            dst: dst.0,
            bytes,
        });
        Ok(())
    }
    fn memset(&self, ptr: DevicePtr, _value: u8, bytes: usize) -> Result<()> {
        self.effects
            .lock()
            .unwrap()
            .push(Effect::Memset { ptr: ptr.0, bytes });
        Ok(())
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memset(ptr, value, bytes)
    }
    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        self.effects.lock().unwrap().push(Effect::Launch);
        Ok(())
    }
    fn stream_is_capturing(&self, _stream: u64) -> bool {
        false
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _module: &str, _name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(7))
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
}

const ORDINAL: u64 = GLM53_KDA_RECURRENT_ORDINAL_BYTES;
const TOTAL: u64 = ORDINAL * GLM53_KDA_RECURRENT_ORDINALS as u64;
const SLAB: u64 = (GLM53_COMPLETION_SLOTS * GLM53_COMPLETION_SLOT_BYTES) as u64;

/// The layout must keep the 34x4 MiB relationship this phase copies against.
#[test]
fn t1_recurrent_payload_is_exactly_thirty_four_four_mib_ordinals() {
    let layout = Glm53T1StateLayout::exact().unwrap();
    assert_eq!(layout.kda_recurrent_f32.payload_bytes, TOTAL);
    assert_eq!(TOTAL, 142_606_336);
    assert_eq!(GLM53_KDA_RECURRENT_ORDINALS, 34);
}

/// The commit slot sits immediately after the 34 conv slots and inside the slab.
#[test]
fn recurrent_commit_slot_follows_the_thirty_four_conv_slots() {
    assert_eq!(GLM53_KDA_RECURRENT_COMMIT_SLOT, GLM53_KDA_T1_LAYERS.len());
    assert!(GLM53_KDA_RECURRENT_COMMIT_SLOT < GLM53_COMPLETION_SLOTS);
}

/// A rejected decision must not clear, copy, launch or read back — persistent
/// recurrent state has to be byte-identical after a rejected speculative step.
#[test]
fn rejected_decision_enqueues_no_effect_at_all() {
    let gpu = OrderGpu::new();
    let kernel = Glm53KdaRecurrentCommitKernel::load(&gpu).unwrap();
    let executor = Glm53KdaRecurrentCommitExecutor::new(kernel, 19).unwrap();
    executor.retire_rejected(0).unwrap();
    assert!(gpu.effects().is_empty());
}

/// Retiring with an accepted decision is a programming error, not a silent
/// no-op: the accepted path must go through `commit`.
#[test]
fn retire_rejects_an_accepted_decision() {
    let gpu = OrderGpu::new();
    let kernel = Glm53KdaRecurrentCommitKernel::load(&gpu).unwrap();
    let executor = Glm53KdaRecurrentCommitExecutor::new(kernel, 19).unwrap();
    assert!(executor.retire_rejected(1).is_err());
}

/// A zero stream is never the caller's exclusive stream.
#[test]
fn zero_stream_is_refused_before_any_effect() {
    let gpu = OrderGpu::new();
    let kernel = Glm53KdaRecurrentCommitKernel::load(&gpu).unwrap();
    assert!(Glm53KdaRecurrentCommitExecutor::new(kernel, 0).is_err());
    assert!(gpu.effects().is_empty());
}
