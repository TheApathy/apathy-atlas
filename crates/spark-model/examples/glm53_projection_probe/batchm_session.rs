// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded diagnostic owners. Failed/unwound completion quarantines everything.
use super::batchm_contract::{INPUT_ROW_BYTES, MAX_INPUT_ROWS, MAX_ROWS, OUTPUT_ROW_BYTES};
use super::gemv_contract::Span;
use anyhow::{Context, Result, bail, ensure};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::mem::ManuallyDrop;

pub const GUARD_BYTES: usize = 256;
pub const DEVICE_CAP_BYTES: usize = 64 * 1024 * 1024;
const MAX_BODY: usize = MAX_INPUT_ROWS as usize * INPUT_ROW_BYTES;
pub const OUTPUT_BYTES: usize = MAX_ROWS as usize * OUTPUT_ROW_BYTES;
#[derive(Clone, Copy)]
struct Buffer {
    base: DevicePtr,
    span: Span,
    total: usize,
    host: usize,
    immutable: bool,
}
pub struct Session {
    pub gpu: ManuallyDrop<AtlasCudaBackend>,
    pub stream: u64,
    bases: Vec<DevicePtr>,
    buffers: Vec<Buffer>,
    hosts: Vec<Vec<u8>>,
    readback: Vec<u8>,
    poison: Vec<u8>,
    total: usize,
    closed: bool,
}
impl Session {
    pub fn new() -> Result<Self> {
        // Host allocation precedes CUDA construction. All post-construction
        // effects are inside the caller's catch_unwind/consuming close envelope.
        let mut readback = Vec::new();
        readback.try_reserve_exact(MAX_BODY + 2 * GUARD_BYTES)?;
        readback.resize(MAX_BODY + 2 * GUARD_BYTES, 0);
        let poison = [0xc0, 0x7f].repeat(OUTPUT_BYTES / 2);
        let bases = Vec::with_capacity(4);
        let buffers = Vec::with_capacity(4);
        let hosts = Vec::with_capacity(4);
        let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
        let stream = gpu.default_stream();
        Ok(Self {
            gpu: ManuallyDrop::new(gpu),
            stream,
            bases,
            buffers,
            hosts,
            readback,
            poison,
            total: 0,
            closed: false,
        })
    }
    pub fn fence(&self) -> Result<()> {
        self.gpu.synchronize(self.stream)
    }
    fn allocate(&mut self, body: Vec<u8>, immutable: bool) -> Result<Span> {
        ensure!(
            !self.closed && self.bases.len() < 4,
            "batchM owner count/closed state"
        );
        ensure!(
            !body.is_empty() && body.len() <= MAX_BODY && body.len() % 2 == 0,
            "batchM allocation size"
        );
        let bytes = body.len();
        let total = bytes
            .checked_add(2 * GUARD_BYTES)
            .context("guard size overflow")?;
        let owned = self
            .total
            .checked_add(total)
            .context("owned bytes overflow")?;
        ensure!(owned <= DEVICE_CAP_BYTES, "batchM device cap exceeded");
        self.fence()?;
        let mut initial = Vec::new();
        initial.try_reserve_exact(total)?;
        initial.resize(total, 0xa5);
        initial[GUARD_BYTES..GUARD_BYTES + bytes].copy_from_slice(&body);
        let host = self.hosts.len();
        self.hosts.push(initial);
        let base = self.gpu.alloc(total)?;
        // Pre-reserved slots: no fallible pointer arithmetic before raw ownership.
        self.bases.push(base);
        self.total = owned;
        ensure!(
            base.0 != 0 && base.0 % 16 == 0,
            "batchM allocation alignment"
        );
        base.0
            .checked_add(u64::try_from(total)?)
            .context("allocation end overflow")?;
        let span = Span {
            ptr: base.0 + GUARD_BYTES as u64,
            bytes,
        };
        self.buffers.push(Buffer {
            base,
            span,
            total,
            host,
            immutable,
        });
        self.gpu.copy_h2d(&self.hosts[host], base)?;
        self.fence()?;
        Ok(span)
    }
    pub fn upload(&mut self, body: Vec<u8>) -> Result<Span> {
        self.allocate(body, true)
    }
    pub fn output(&mut self) -> Result<Span> {
        self.allocate(vec![0; OUTPUT_BYTES], false)
    }
    fn owned(&self, span: Span) -> Result<Buffer> {
        self.buffers
            .iter()
            .copied()
            .find(|b| b.span == span)
            .context("unowned exact span")
    }
    pub fn poison_output(&mut self, span: Span) -> Result<()> {
        let b = self.owned(span)?;
        ensure!(
            !b.immutable && span.bytes == self.poison.len(),
            "invalid poison destination"
        );
        self.fence()?;
        self.gpu.copy_h2d(&self.poison, DevicePtr(span.ptr))?;
        self.fence()
    }
    fn read_span(&mut self, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes > 0 && bytes <= self.readback.len(), "readback extent");
        let end = ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .context("readback end overflow")?;
        ensure!(
            self.buffers
                .iter()
                .any(|b| ptr.0 >= b.base.0 && end <= b.base.0 + b.total as u64),
            "readback outside owned allocation"
        );
        self.fence()?;
        // The destination never moves/resizes; even a copy error may have queued
        // DMA. Caller stops on error; close/Drop retain it until completion.
        self.gpu
            .copy_d2h_on_stream(ptr, &mut self.readback[..bytes], self.stream)?;
        self.fence()?;
        Ok(self.readback[..bytes].to_vec())
    }
    pub fn read_output(&mut self, span: Span) -> Result<Vec<u8>> {
        ensure!(
            !self.owned(span)?.immutable,
            "read_output requires output owner"
        );
        self.read_span(DevicePtr(span.ptr), span.bytes)
    }
    pub fn guard_snapshot(&mut self) -> Result<Vec<u8>> {
        let mut raw = Vec::with_capacity(self.buffers.len() * 2 * GUARD_BYTES);
        for b in self.buffers.clone() {
            raw.extend(self.read_span(b.base, GUARD_BYTES)?);
            raw.extend(self.read_span(DevicePtr(b.span.ptr + b.span.bytes as u64), GUARD_BYTES)?);
        }
        Ok(raw)
    }
    pub fn check_guards(&self, raw: &[u8]) -> Result<()> {
        ensure!(
            raw.len() == self.buffers.len() * 2 * GUARD_BYTES && raw.iter().all(|b| *b == 0xa5),
            "batchM guard corruption"
        );
        Ok(())
    }
    pub fn immutable_snapshots(&mut self) -> Result<Vec<(usize, Vec<u8>, bool)>> {
        let mut result = Vec::new();
        for b in self.buffers.clone().into_iter().filter(|b| b.immutable) {
            let raw = self.read_span(DevicePtr(b.span.ptr), b.span.bytes)?;
            let exact = raw == self.hosts[b.host][GUARD_BYTES..GUARD_BYTES + b.span.bytes];
            result.push((b.host, raw, exact));
        }
        Ok(result)
    }
    pub fn owned_bytes(&self) -> usize {
        self.total
    }
    pub fn close(mut self) -> Result<()> {
        let release = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.fence()?;
            for base in &mut self.bases {
                self.gpu.free(*base)?;
                *base = DevicePtr::NULL;
            }
            Ok(())
        }));
        if !matches!(release, Ok(Ok(()))) {
            let reason = match release {
                Ok(Err(e)) => format!("{e:#}"),
                _ => "completion/release panicked".into(),
            };
            std::mem::forget(self);
            bail!("batchM owners quarantined until process teardown: {reason}");
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
            eprintln!("batchM host/device/backend owners retained until process teardown");
        }
    }
}
