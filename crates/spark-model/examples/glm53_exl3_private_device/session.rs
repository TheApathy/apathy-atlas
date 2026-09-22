// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded diagnostic device and transfer ownership.
use super::{DEVICE_CAP, GUARD};
use anyhow::{Context, Result, anyhow, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::mem::ManuallyDrop;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Buffer {
    base: DevicePtr,
    pub(super) ptr: DevicePtr,
    bytes: usize,
}
pub(super) struct Session {
    pub(super) gpu: ManuallyDrop<AtlasCudaBackend>,
    pub(super) stream: u64,
    buffers: Vec<Buffer>,
    total: usize,
    pending_host: Option<Vec<u8>>,
    closed: bool,
}
impl Session {
    pub(super) fn new() -> Result<Self> {
        let buffers = Vec::with_capacity(8);
        let gpu = ManuallyDrop::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
        let stream = gpu.default_stream();
        Ok(Self {
            gpu,
            stream,
            buffers,
            total: 0,
            pending_host: None,
            closed: false,
        })
    }
    pub(super) fn fence(&self) -> Result<()> {
        self.gpu.synchronize(self.stream)
    }
    pub(super) fn alloc(&mut self, bytes: usize) -> Result<Buffer> {
        ensure!(
            bytes > 0 && self.buffers.len() < 8,
            "allocation geometry/count"
        );
        let total = bytes
            .checked_add(2 * GUARD)
            .context("allocation overflow")?;
        ensure!(self.total + total <= DEVICE_CAP, "device cap exceeded");
        self.fence()?;
        let base = self.gpu.alloc(total)?;
        // Record ownership before any fallible pointer validation or upload.
        let b = Buffer {
            base,
            ptr: base,
            bytes,
        };
        self.buffers.push(b);
        self.total += total;
        ensure!(
            base.0 != 0 && base.0.is_multiple_of(256),
            "allocation alignment"
        );
        base.0
            .checked_add(total as u64)
            .context("allocation address overflow")?;
        let b = Buffer {
            ptr: base.offset(GUARD),
            ..b
        };
        *self.buffers.last_mut().expect("owned allocation") = b;
        self.write_raw(base, vec![0xa5; total])?;
        Ok(b)
    }
    fn write_raw(&mut self, ptr: DevicePtr, bytes: Vec<u8>) -> Result<()> {
        ensure!(
            self.pending_host.is_none(),
            "pending transfer cannot be replaced"
        );
        self.fence()?;
        self.pending_host = Some(bytes);
        self.gpu
            .copy_h2d(self.pending_host.as_ref().expect("owned upload"), ptr)?;
        self.fence()?;
        self.pending_host.take();
        Ok(())
    }
    pub(super) fn write(&mut self, b: Buffer, bytes: Vec<u8>) -> Result<()> {
        ensure!(
            self.buffers.contains(&b) && bytes.len() == b.bytes,
            "write extent/owner"
        );
        self.write_raw(b.ptr, bytes)
    }
    pub(super) fn poison(&mut self, b: Buffer) -> Result<()> {
        self.write(b, vec![0xcd; b.bytes])
    }
    fn read_raw(&mut self, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        ensure!(
            self.pending_host.is_none() && bytes <= DEVICE_CAP,
            "readback state/size"
        );
        self.fence()?;
        self.pending_host = Some(vec![0; bytes]);
        self.gpu
            .copy_d2h(ptr, self.pending_host.as_mut().expect("owned readback"))?;
        self.fence()?;
        Ok(self.pending_host.take().expect("completed readback"))
    }
    pub(super) fn read(&mut self, b: Buffer) -> Result<Vec<u8>> {
        ensure!(self.buffers.contains(&b), "read owner");
        self.read_raw(b.ptr, b.bytes)
    }
    pub(super) fn word(&mut self, b: Buffer) -> Result<u32> {
        ensure!(b.bytes == 4, "word extent");
        Ok(u32::from_le_bytes(
            self.read(b)?
                .try_into()
                .map_err(|_| anyhow!("word width"))?,
        ))
    }
    pub(super) fn guards(&mut self) -> Result<()> {
        for b in self.buffers.clone() {
            let raw = self.read_raw(b.base, b.bytes + 2 * GUARD)?;
            ensure!(
                raw[..GUARD]
                    .iter()
                    .chain(&raw[GUARD + b.bytes..])
                    .all(|x| *x == 0xa5),
                "allocation guard changed, body_bytes={}",
                b.bytes
            );
        }
        Ok(())
    }
    pub(super) fn close(mut self) -> Result<()> {
        self.fence().context("close fence; no frees on failure")?;
        // Pending synchronous transfer storage is released only after the fence.
        self.pending_host.take();
        self.guards()?;
        while let Some(b) = self.buffers.pop() {
            if let Err(error) = self.gpu.free(b.base) {
                self.buffers.push(b);
                return Err(error.context("free failed; remaining owners quarantined"));
            }
        }
        unsafe {
            ManuallyDrop::drop(&mut self.gpu);
        }
        self.closed = true;
        Ok(())
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if !self.closed {
            if let Some(host) = self.pending_host.take() {
                std::mem::forget(host);
            }
            eprintln!(
                "PRIVATE_DEVICE_QUARANTINE: backend and unfreed device owners retained until process exit"
            );
        }
    }
}
