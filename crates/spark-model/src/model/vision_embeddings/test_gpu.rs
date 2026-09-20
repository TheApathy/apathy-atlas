// SPDX-License-Identifier: AGPL-3.0-only

//! Byte-bearing I/O backend: unlike the generic mock, D2D copies execute.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(super) struct MemoryGpu {
    allocs: Mutex<BTreeMap<u64, Vec<u8>>>,
    next: AtomicUsize,
    copies: AtomicUsize,
    pub fail_copy: AtomicBool,
    pub fail_sync: AtomicBool,
    pub events: Mutex<Vec<(char, u64)>>,
}

impl MemoryGpu {
    pub fn new() -> Self {
        Self {
            allocs: Mutex::new(BTreeMap::new()),
            next: AtomicUsize::new(4096),
            copies: AtomicUsize::new(0),
            fail_copy: AtomicBool::new(false),
            fail_sync: AtomicBool::new(false),
            events: Mutex::new(Vec::new()),
        }
    }
    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().unwrap().len()
    }
    pub fn copy_count(&self) -> usize {
        self.copies.load(Ordering::SeqCst)
    }
    pub fn bytes(&self, ptr: DevicePtr) -> Vec<u8> {
        self.allocs.lock().unwrap()[&ptr.0].clone()
    }
}

impl GpuBackend for MemoryGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        ensure!(bytes > 0 && bytes <= 1024 * 1024, "test allocation bound");
        let ptr = self.next.fetch_add(bytes + 256, Ordering::SeqCst) as u64;
        self.allocs.lock().unwrap().insert(ptr, vec![0xdd; bytes]);
        Ok(DevicePtr(ptr))
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        ensure!(
            self.allocs.lock().unwrap().remove(&ptr.0).is_some(),
            "double free"
        );
        self.events.lock().unwrap().push(('f', ptr.0));
        Ok(())
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut allocs = self.allocs.lock().unwrap();
        let (&base, data) = allocs
            .range_mut(..=dst.0)
            .next_back()
            .ok_or_else(|| anyhow::anyhow!("unknown destination"))?;
        let start = (dst.0 - base) as usize;
        ensure!(
            start
                .checked_add(src.len())
                .is_some_and(|end| end <= data.len()),
            "write outside allocation"
        );
        data[start..start + src.len()].copy_from_slice(src);
        Ok(())
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let allocs = self.allocs.lock().unwrap();
        let (&base, data) = allocs
            .range(..=src.0)
            .next_back()
            .ok_or_else(|| anyhow::anyhow!("unknown source"))?;
        let start = (src.0 - base) as usize;
        ensure!(
            start
                .checked_add(dst.len())
                .is_some_and(|end| end <= data.len()),
            "read outside allocation"
        );
        dst.copy_from_slice(&data[start..start + dst.len()]);
        Ok(())
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        ensure!(
            !self.fail_copy.swap(false, Ordering::SeqCst),
            "injected copy failure"
        );
        let mut data = vec![0; bytes];
        self.copy_d2h(src, &mut data)?;
        self.copy_h2d(&data, dst)?;
        self.copies.fetch_add(1, Ordering::SeqCst);
        self.events.lock().unwrap().push(('c', dst.0));
        Ok(())
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("unexpected kernel")
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.events.lock().unwrap().push(('s', stream));
        ensure!(
            !self.fail_sync.swap(false, Ordering::SeqCst),
            "injected sync failure"
        );
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        7
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        anyhow::bail!("unexpected kernel lookup")
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.copy_h2d(&vec![value; bytes], ptr)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _: u64) -> Result<()> {
        self.memset(ptr, value, bytes)
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(1024 * 1024)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(1024 * 1024)
    }
}
