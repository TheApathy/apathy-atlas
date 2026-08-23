// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailingCopyGpu {
    fail_at: Option<usize>,
    copies: AtomicUsize,
}

impl FailingCopyGpu {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            fail_at,
            copies: AtomicUsize::new(0),
        }
    }

    fn copy_count(&self) -> usize {
        self.copies.load(Ordering::SeqCst)
    }
}

impl GpuBackend for FailingCopyGpu {
    fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
        Ok(DevicePtr(0x1000))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }

    fn free(&self, _ptr: DevicePtr) -> Result<()> {
        Ok(())
    }

    fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
        Ok(())
    }

    fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        Ok(())
    }

    fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        _src: DevicePtr,
        _dst: DevicePtr,
        _bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        let call = self.copies.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_at == Some(call) {
            bail!("injected D2D failure at copy {call}");
        }
        Ok(())
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        Ok(())
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(0))
    }

    fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn memset_async(&self, _ptr: DevicePtr, _value: u8, _bytes: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(1 << 30)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(1 << 30)
    }
}

fn state_pool() -> SsmStatePool {
    SsmStatePool {
        h_state_pools: vec![DevicePtr(0x1000), DevicePtr(0x2000)],
        conv_state_pools: vec![DevicePtr(0x3000), DevicePtr(0x4000)],
        h_intermediate_pools: Vec::new(),
        conv_intermediate_pools: Vec::new(),
        h_checkpoint_pools: Vec::new(),
        conv_checkpoint_pools: Vec::new(),
        wy17_kv_retain_pools: Vec::new(),
        wy17_gate_retain_pools: Vec::new(),
        kv_retain_bytes: 0,
        gate_retain_bytes: 0,
        h_bytes: 8,
        conv_bytes: 4,
        max_slots: 1,
        num_ssm_layers: 2,
        has_mtp: false,
        num_intermediates: 0,
        free_slots: Mutex::new(Vec::new()),
    }
}

fn snapshot_pool(free_slots: Vec<usize>) -> SsmSnapshotPool {
    SsmSnapshotPool {
        h_snapshots: vec![DevicePtr(0x5000), DevicePtr(0x6000)],
        conv_snapshots: vec![DevicePtr(0x7000), DevicePtr(0x8000)],
        free_slots: Mutex::new(free_slots),
        num_slots: 1,
        h_bytes: 8,
        conv_bytes: 4,
        num_ssm_layers: 2,
        // Matches the disabled-path value production uses in `SsmSnapshotPool::new`.
        // Only read when `h_is_f16` is true, which these tests never set.
        h_f16_to_f32_k: KernelHandle(0),
        session_tags: Mutex::new(HashMap::new()),
        decode_h_snapshots: Vec::new(),
        decode_conv_snapshots: Vec::new(),
        decode_ring_slots: 0,
        decode_max_seqs: 0,
    }
}

#[test]
fn every_copy_failure_restores_slot_and_clears_owner() {
    let main = state_pool();
    for fail_at in 1..=4 {
        let gpu = FailingCopyGpu::new(Some(fail_at));
        let snapshots = snapshot_pool(vec![0]);
        snapshots.session_tags.lock().insert(0, 0xAABB);

        let error = snapshots
            .save(0, 0xCCDD, false, &main, &gpu, 7)
            .unwrap_err();

        assert!(error.to_string().contains(&format!("copy {fail_at}")));
        assert_eq!(gpu.copy_count(), fail_at);
        assert_eq!(*snapshots.free_slots.lock(), vec![0]);
        assert!(!snapshots.session_tags.lock().contains_key(&0));
    }
}

#[test]
fn successful_save_and_free_keep_existing_semantics() {
    let main = state_pool();
    let gpu = FailingCopyGpu::new(None);
    let snapshots = snapshot_pool(vec![0]);

    assert_eq!(
        snapshots.save(0, 0xCCDD, false, &main, &gpu, 7).unwrap(),
        Some(0)
    );
    assert_eq!(gpu.copy_count(), 4);
    assert!(snapshots.free_slots.lock().is_empty());
    assert_eq!(snapshots.session_tags.lock().get(&0), Some(&0xCCDD));

    snapshots.free(0);
    assert_eq!(*snapshots.free_slots.lock(), vec![0]);
    assert!(!snapshots.session_tags.lock().contains_key(&0));
}

#[test]
fn disabled_or_exhausted_pool_still_returns_none_without_copying() {
    let main = state_pool();
    let gpu = FailingCopyGpu::new(Some(1));
    let mut disabled = snapshot_pool(vec![0]);
    disabled.num_slots = 0;
    assert_eq!(disabled.save(0, 1, false, &main, &gpu, 7).unwrap(), None);

    let exhausted = snapshot_pool(Vec::new());
    assert_eq!(exhausted.save(0, 1, false, &main, &gpu, 7).unwrap(), None);
    assert_eq!(gpu.copy_count(), 0);
}
