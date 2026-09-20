// SPDX-License-Identifier: AGPL-3.0-only
//! Retain every host/device/backend owner across failed copies and fences.
use anyhow::{Result, bail};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::mem::ManuallyDrop;
pub struct Session {
    pub gpu: ManuallyDrop<AtlasCudaBackend>,
    pub stream: u64,
    pub ptrs: Vec<DevicePtr>,
    hosts: Vec<Vec<u8>>,
    readback: Vec<u8>,
    poison: Vec<u8>,
    closed: bool,
}
impl Session {
    pub fn new() -> Result<Self> {
        let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
        let stream = gpu.default_stream();
        Ok(Self {
            gpu: ManuallyDrop::new(gpu),
            stream,
            ptrs: Vec::new(),
            hosts: Vec::new(),
            readback: vec![0; 4096],
            poison: [0xc0, 0x7f].repeat(2048),
            closed: false,
        })
    }
    pub fn upload(&mut self, bytes: Vec<u8>) -> Result<DevicePtr> {
        let ptr = self.gpu.alloc(bytes.len())?;
        self.ptrs.push(ptr);
        self.hosts.push(bytes);
        self.gpu.copy_h2d(self.hosts.last().unwrap(), ptr)?;
        Ok(ptr)
    }
    pub fn output(&mut self) -> Result<DevicePtr> {
        let ptr = self.gpu.alloc(self.readback.len())?;
        self.ptrs.push(ptr);
        Ok(ptr)
    }
    /// Nonfinite BF16 makes every active missing write fail the raw gate.
    /// Storage remains owned across copy/fence failure just like input uploads.
    pub fn poison_output(&self, ptr: DevicePtr) -> Result<()> {
        self.gpu.copy_h2d(&self.poison, ptr)?;
        self.gpu.synchronize(self.stream)
    }
    pub fn read(&mut self, ptr: DevicePtr) -> Result<Vec<u8>> {
        self.gpu.copy_d2h(ptr, &mut self.readback)?;
        Ok(self.readback.clone())
    }
    pub fn close(mut self) -> Result<()> {
        let fence = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.gpu.synchronize(self.stream)
        }));
        if !matches!(fence, Ok(Ok(()))) {
            std::mem::forget(self);
            bail!("operator owners quarantined until process teardown after failed fence");
        }
        for &ptr in &self.ptrs {
            self.gpu.free(ptr)?;
        }
        self.closed = true;
        unsafe {
            ManuallyDrop::drop(&mut self.gpu);
        }
        Ok(())
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if !self.closed {
            std::mem::forget(std::mem::take(&mut self.hosts));
            std::mem::forget(std::mem::take(&mut self.readback));
            std::mem::forget(std::mem::take(&mut self.poison));
            eprintln!("operator host/device/backend owners retained until process teardown");
        }
    }
}
