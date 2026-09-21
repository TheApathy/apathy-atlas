// SPDX-License-Identifier: AGPL-3.0-only
//! One owned context/stream. Default-stream memset and pageable copies receive
//! an explicit context completion fence before host data or device consumers
//! can be released. Draining only the nonblocking kernel stream is insufficient.
//! No CPU API loads CUDA.
use crate::{
    contract::{Arg, LaunchSpec},
    cuda_abi::{Api, Handle},
};
use anyhow::{Result, ensure};
use libloading::Library;
use serde_json::{Value, json};
use std::{
    ffi::{CString, c_char, c_void},
    path::Path,
    ptr,
};
const GUARD: usize = 256;
const CAP: usize = 128 * 1024 * 1024;
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    base: u64,
    pub ptr: u64,
    pub bytes: usize,
}
pub fn guarded_bytes(bytes: usize, current: usize) -> Result<(usize, usize)> {
    ensure!(bytes > 0, "zero allocation");
    let size = bytes
        .checked_add(2 * GUARD)
        .ok_or_else(|| anyhow::anyhow!("allocation size overflow"))?;
    let total = current
        .checked_add(size)
        .ok_or_else(|| anyhow::anyhow!("allocation total overflow"))?;
    ensure!(total <= CAP, "owned CUDA allocation cap exceeded");
    Ok((size, total))
}
pub fn guarded_span(base: u64, bytes: usize) -> Result<(u64, u64, u64)> {
    guarded_bytes(bytes, 0)?;
    ensure!(base != 0, "null allocation base");
    let guard = u64::try_from(GUARD)?;
    let bytes = u64::try_from(bytes)?;
    let body = base
        .checked_add(guard)
        .ok_or_else(|| anyhow::anyhow!("leading guard pointer overflow"))?;
    let tail = body
        .checked_add(bytes)
        .ok_or_else(|| anyhow::anyhow!("payload end pointer overflow"))?;
    let end = tail
        .checked_add(guard)
        .ok_or_else(|| anyhow::anyhow!("trailing guard pointer overflow"))?;
    Ok((body, tail, end))
}
pub struct Driver {
    api: Api,
    context: Handle,
    pub stream: Handle,
    modules: Vec<Handle>,
    allocations: Vec<Buffer>,
    owned_bases: Vec<u64>,
    pub peak_bytes: usize,
    pub identity: Value,
    _library: Library,
}
fn check(status: i32, name: &str) -> Result<()> {
    ensure!(status == 0, "CUDA {name} failed: {status}");
    Ok(())
}
pub fn complete_default_stream<T>(
    operation: impl FnOnce() -> Result<T>,
    completion: impl FnOnce() -> Result<()>,
) -> Result<T> {
    // An API error may surface after enqueueing work. Always attempt completion
    // while borrowed host memory remains live, and retain both errors if needed.
    let result = operation();
    let fence = completion();
    match (result, fence) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(operation), Err(fence)) => {
            anyhow::bail!("operation: {operation:#}; completion: {fence:#}")
        }
    }
}
impl Driver {
    pub fn open(library_path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(library_path) }?;
        let api = unsafe { Api::load(&library) }?;
        let mut driver = Self {
            api,
            context: ptr::null_mut(),
            stream: ptr::null_mut(),
            modules: vec![],
            allocations: vec![],
            owned_bases: vec![],
            peak_bytes: 0,
            identity: Value::Null,
            _library: library,
        };
        unsafe {
            check((driver.api.init)(0), "init")?;
            let mut device = 0;
            check((driver.api.device_get)(&mut device, 0), "device0")?;
            let (mut major, mut minor) = (0, 0);
            check(
                (driver.api.device_attribute)(&mut major, 75, device),
                "SM major",
            )?;
            check(
                (driver.api.device_attribute)(&mut minor, 76, device),
                "SM minor",
            )?;
            ensure!((major, minor) == (12, 1), "probe requires GB10 SM12.1");
            let mut name = [0 as c_char; 256];
            let mut version = 0;
            check(
                (driver.api.device_name)(name.as_mut_ptr(), 256, device),
                "name",
            )?;
            check((driver.api.version)(&mut version), "version")?;
            let bytes: Vec<_> = name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            driver.identity = json!({"device_ordinal":0,"device_name":String::from_utf8(bytes)?,
                "driver_version":version,"compute_capability":[major,minor],"context":"owned fresh","stream":"owned nonblocking",
                "copy_contract":"owned-stream drain before copies; cuCtxSynchronize after every default-stream memset/copy before return",
                "initialization_completion":"cuCtxSynchronize","cleanup_completion":"cuCtxSynchronize before resource release"});
            check(
                (driver.api.context_create)(&mut driver.context, 0, device),
                "context create",
            )?;
            check(
                (driver.api.stream_create)(&mut driver.stream, 1),
                "stream create",
            )?;
        }
        Ok(driver)
    }
    pub fn sync(&self) -> Result<()> {
        ensure!(
            !self.context.is_null() && !self.stream.is_null(),
            "CUDA owner is not open"
        );
        unsafe { check((self.api.stream_sync)(self.stream), "stream drain") }
    }
    fn context_complete(&self) -> Result<()> {
        ensure!(!self.context.is_null(), "CUDA context is not open");
        unsafe { check((self.api.context_sync)(), "context completion") }
    }
    pub fn allocate(&mut self, bytes: usize, fill: u8) -> Result<Buffer> {
        let current = self.allocations.iter().try_fold(0usize, |total, b| {
            guarded_bytes(b.bytes, total).map(|(_, n)| n)
        })?;
        let (size, total) = guarded_bytes(bytes, current)?;
        self.sync()?;
        let mut base = 0;
        unsafe {
            check((self.api.alloc)(&mut base, size), "allocation")?;
        }
        // Register raw ownership immediately: even pointer validation or memset
        // failure must leave this allocation available to explicit cleanup.
        self.owned_bases.push(base);
        self.peak_bytes = self.peak_bytes.max(total);
        let (body, _, _) = guarded_span(base, bytes)?;
        let b = Buffer {
            base,
            ptr: body,
            bytes,
        };
        self.allocations.push(b);
        complete_default_stream(
            || unsafe {
                check((self.api.memset)(base, 0xa5, size), "guards")?;
                check((self.api.memset)(b.ptr, fill, bytes), "body fill")
            },
            || self.context_complete(),
        )?;
        Ok(b)
    }
    pub fn upload(&mut self, raw: &[u8]) -> Result<Buffer> {
        let b = self.allocate(raw.len(), 0)?;
        self.write(b, raw)?;
        Ok(b)
    }
    pub fn write(&self, b: Buffer, raw: &[u8]) -> Result<()> {
        ensure!(
            self.allocations.contains(&b),
            "H2D buffer is not an owned exact span"
        );
        ensure!(b.bytes == raw.len(), "H2D size mismatch");
        self.sync()?;
        complete_default_stream(
            || unsafe { check((self.api.h2d)(b.ptr, raw.as_ptr().cast(), raw.len()), "H2D") },
            || self.context_complete(),
        )
    }
    fn read_span(&self, p: u64, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes > 0 && bytes <= CAP, "D2H byte bound");
        let end = p
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow::anyhow!("D2H pointer overflow"))?;
        ensure!(
            self.allocations
                .iter()
                .any(|b| guarded_span(b.base, b.bytes)
                    .is_ok_and(|(_, _, limit)| p >= b.base && end <= limit)),
            "D2H outside owned allocation"
        );
        self.sync()?;
        let mut raw = vec![0; bytes];
        complete_default_stream(
            || unsafe { check((self.api.d2h)(raw.as_mut_ptr().cast(), p, bytes), "D2H") },
            || self.context_complete(),
        )?;
        Ok(raw)
    }
    pub fn read(&self, b: Buffer) -> Result<Vec<u8>> {
        ensure!(
            self.allocations.contains(&b),
            "D2H buffer is not an owned exact span"
        );
        self.read_span(b.ptr, b.bytes)
    }
    pub fn guards(&self) -> Result<()> {
        for b in &self.allocations {
            let (_, tail, _) = guarded_span(b.base, b.bytes)?;
            ensure!(
                self.read_span(b.base, GUARD)?.iter().all(|&v| v == 0xa5)
                    && self.read_span(tail, GUARD)?.iter().all(|&v| v == 0xa5),
                "device redzone changed"
            );
        }
        Ok(())
    }
    pub fn function(&mut self, ptx: &[u8], name: &str) -> Result<Handle> {
        self.sync()?;
        let text = CString::new(ptx)?;
        let name = CString::new(name)?;
        let mut module = ptr::null_mut();
        unsafe {
            check(
                (self.api.module_load)(&mut module, text.as_ptr().cast()),
                "module load",
            )?;
        }
        self.modules.push(module);
        let mut function = ptr::null_mut();
        unsafe {
            check(
                (self.api.function)(&mut function, module, name.as_ptr()),
                "function",
            )?;
        }
        Ok(function)
    }
    pub fn launch(&self, k: Handle, mut spec: LaunchSpec) -> Result<()> {
        ensure!(
            !self.context.is_null() && !self.stream.is_null(),
            "CUDA owner is not open"
        );
        ensure!(
            !k.is_null() && spec.grid.iter().all(|&n| n > 0) && spec.block.iter().all(|&n| n > 0),
            "invalid launch geometry"
        );
        for arg in &spec.args {
            if let Arg::Ptr(p) = arg {
                ensure!(
                    *p == 0 || self.allocations.iter().any(|b| b.ptr == *p),
                    "kernel pointer outside owned buffers"
                );
            }
        }
        let mut params: Vec<*mut c_void> = spec
            .args
            .iter_mut()
            .map(|a| match a {
                Arg::Ptr(p) => (p as *mut u64).cast(),
                Arg::U32(n) => (n as *mut u32).cast(),
            })
            .collect();
        unsafe {
            check(
                (self.api.launch)(
                    k,
                    spec.grid[0],
                    spec.grid[1],
                    spec.grid[2],
                    spec.block[0],
                    spec.block[1],
                    spec.block[2],
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "launch",
            )
        }
    }
    pub fn close(&mut self) -> Result<()> {
        let mut errors = Vec::new();
        macro_rules! record {
            ($call:expr,$name:literal) => {
                if let Err(e) = check(unsafe { $call }, $name) {
                    errors.push(e.to_string());
                }
            };
        }
        if !self.stream.is_null() {
            record!((self.api.stream_sync)(self.stream), "cleanup drain");
        }
        if !self.context.is_null() {
            // Also drain default-stream initialization/copies before releasing
            // modules, allocations, or the nonblocking kernel stream.
            record!((self.api.context_sync)(), "cleanup context completion");
        }
        for module in self.modules.drain(..).rev() {
            record!((self.api.module_unload)(module), "module unload");
        }
        self.allocations.clear();
        for base in self.owned_bases.drain(..).rev() {
            record!((self.api.free)(base), "free");
        }
        if !self.stream.is_null() {
            record!((self.api.stream_destroy)(self.stream), "stream destroy");
            self.stream = ptr::null_mut();
        }
        if !self.context.is_null() {
            record!((self.api.context_destroy)(self.context), "context destroy");
            self.context = ptr::null_mut();
        }
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        if !self.context.is_null() {
            if let Err(e) = self.close() {
                eprintln!("CUDA emergency cleanup failed: {e}");
            }
        }
    }
}
