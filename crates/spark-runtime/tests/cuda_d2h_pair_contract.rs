// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only behavioral and production-caller contracts for coalesced D2H.

use std::sync::Mutex;

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const GPU_TRAIT: &str = include_str!("../src/gpu.rs");
const GPU_IMPL: &str = include_str!("../src/cuda_backend/gpu_impl.rs");
const DFLASH: &str = include_str!("../../spark-model/src/layers/dflash_head/forward_block.rs");
const HSS: &str =
    include_str!("../../spark-model/src/layers/qwen3_attention/decode/high_speed_swap.rs");

#[derive(Clone, Copy)]
enum Failure {
    None,
    Sync,
    Copy(usize),
}

struct RecordingGpu {
    events: Mutex<Vec<String>>,
    failure: Failure,
}

impl RecordingGpu {
    fn new(failure: Failure) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            failure,
        }
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl GpuBackend for RecordingGpu {
    fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
        bail!("unused")
    }

    fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
        bail!("unused")
    }

    fn free(&self, _ptr: DevicePtr) -> Result<()> {
        bail!("unused")
    }

    fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
        bail!("unused")
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let ordinal = self
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with("copy:"))
            .count()
            + 1;
        self.events
            .lock()
            .unwrap()
            .push(format!("copy:{}:{}", src.0, dst.len()));
        if matches!(self.failure, Failure::Copy(failed) if failed == ordinal) {
            bail!("copy {ordinal} failed")
        }
        dst.fill(src.0 as u8);
        Ok(())
    }

    fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
        bail!("unused")
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        bail!("unused")
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.events.lock().unwrap().push(format!("sync:{stream}"));
        if matches!(self.failure, Failure::Sync) {
            bail!("sync failed")
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        bail!("unused")
    }

    fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
        bail!("unused")
    }

    fn memset_async(&self, _ptr: DevicePtr, _value: u8, _bytes: usize, _stream: u64) -> Result<()> {
        bail!("unused")
    }

    fn total_memory(&self) -> Result<usize> {
        bail!("unused")
    }

    fn free_memory(&self) -> Result<usize> {
        bail!("unused")
    }
}

#[test]
fn default_pair_is_exactly_one_sync_then_two_copies() {
    let gpu = RecordingGpu::new(Failure::None);
    let mut first = [0_u8; 3];
    let mut second = [0_u8; 5];
    gpu.copy_d2h_pair_on_stream(DevicePtr(17), &mut first, DevicePtr(29), &mut second, 41)
        .unwrap();
    assert_eq!(gpu.events(), ["sync:41", "copy:17:3", "copy:29:5"]);
    assert_eq!(first, [17; 3]);
    assert_eq!(second, [29; 5]);
}

#[test]
fn pair_fails_closed_in_operation_order() {
    for (failure, expected) in [
        (Failure::Sync, vec!["sync:41"]),
        (Failure::Copy(1), vec!["sync:41", "copy:17:3"]),
        (Failure::Copy(2), vec!["sync:41", "copy:17:3", "copy:29:5"]),
    ] {
        let gpu = RecordingGpu::new(failure);
        let mut first = [0_u8; 3];
        let mut second = [0_u8; 5];
        assert!(
            gpu.copy_d2h_pair_on_stream(DevicePtr(17), &mut first, DevicePtr(29), &mut second, 41,)
                .is_err()
        );
        assert_eq!(gpu.events(), expected);
    }
}

#[test]
fn production_has_six_pairs_and_no_async_d2h_slice_surface() {
    assert_eq!(DFLASH.matches("copy_d2h_pair_on_stream(").count(), 1);
    assert_eq!(HSS.matches("copy_d2h_pair_on_stream(").count(), 5);
    for source in [GPU_TRAIT, GPU_IMPL, DFLASH, HSS] {
        assert!(!source.contains("copy_d2h_async_on_stream"));
        assert!(!source.contains("cuMemcpyDtoHAsync_v2("));
    }
}
