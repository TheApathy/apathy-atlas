// SPDX-License-Identifier: AGPL-3.0-only

//! Env-gated oracle capture for prefill kernels.
//!
//! `ATLAS_QWEN4_ORACLE_DUMP=<dir>` makes the hot prefill kernels write the exact
//! device-side buffers they were launched with (plus a `meta.json` describing the
//! scalar arguments, shapes and byte counts) into `<dir>/<tag>/`. Each tag is
//! captured ONCE per process, on the first invocation, so a normal 2048-token
//! prefill produces one complete recording per hot kernel.
//!
//! The point is that a standalone harness can replay a kernel bit-for-bit from
//! those files without loading the 106 GB model. Nothing here runs unless the
//! variable is set, and nothing here changes kernel arguments or ordering.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Tags already captured in this process (one recording per tag).
static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Capture root, or `None` when `ATLAS_QWEN4_ORACLE_DUMP` is unset/empty.
pub(crate) fn root() -> Option<PathBuf> {
    match std::env::var("ATLAS_QWEN4_ORACLE_DUMP") {
        Ok(d) if !d.is_empty() => Some(PathBuf::from(d)),
        _ => None,
    }
}

/// One recording. Created by [`Sink::open`]; buffers are appended and the
/// manifest is written by [`Sink::finish`].
pub(crate) struct Sink {
    dir: PathBuf,
    meta: Vec<String>,
}

impl Sink {
    /// Open a recording for `tag`, or return `None` when capture is off or this
    /// tag was already recorded in this process.
    pub(crate) fn open(tag: &str) -> Option<Sink> {
        let root = root()?;
        {
            let mut seen = SEEN.lock().ok()?;
            if seen.iter().any(|t| t == tag) {
                return None;
            }
            seen.push(tag.to_string());
        }
        let dir = root.join(tag);
        std::fs::create_dir_all(&dir).ok()?;
        Some(Sink {
            dir,
            meta: Vec::new(),
        })
    }

    /// Record a scalar argument (any Display value) in the manifest.
    pub(crate) fn scalar(&mut self, name: &str, value: impl std::fmt::Display) {
        self.meta
            .push(format!("    \"{name}\": {value}"));
    }

    /// Copy `bytes` from `ptr` to `<dir>/<name>.bin` and note it in the manifest.
    ///
    /// The caller must have synchronised `stream` (see [`Sink::sync`]); a
    /// zero/­null pointer or a zero length is recorded as absent rather than
    /// being copied, so optional kernel arguments are safe to pass.
    pub(crate) fn buf(
        &mut self,
        gpu: &dyn GpuBackend,
        name: &str,
        ptr: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        if ptr.0 == 0 || bytes == 0 {
            self.meta.push(format!("    \"{name}_bytes\": 0"));
            return Ok(());
        }
        let mut host = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut host)?;
        let path = self.dir.join(format!("{name}.bin"));
        std::fs::write(&path, &host)?;
        self.meta.push(format!("    \"{name}_bytes\": {bytes}"));
        Ok(())
    }

    /// Read a device array of `count` u64 device pointers back to the host.
    ///
    /// Used to follow the MoE expert pointer tables so each expert's weight blob
    /// can be dumped alongside the table itself.
    pub(crate) fn read_ptr_table(
        &self,
        gpu: &dyn GpuBackend,
        ptr: DevicePtr,
        count: usize,
    ) -> Result<Vec<u64>> {
        let mut host = vec![0u8; count * 8];
        gpu.copy_d2h(ptr, &mut host)?;
        Ok(host
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect())
    }

    /// Dump `count` equally sized blobs addressed by a pointer table into one
    /// concatenated file (`<name>.bin`, blob `i` at offset `i * blob_bytes`).
    pub(crate) fn blobs(
        &mut self,
        gpu: &dyn GpuBackend,
        name: &str,
        ptrs: &[u64],
        blob_bytes: usize,
    ) -> Result<()> {
        let path = self.dir.join(format!("{name}.bin"));
        let file = std::fs::File::create(&path)?;
        let mut out = std::io::BufWriter::new(file);
        let mut host = vec![0u8; blob_bytes];
        let mut written = 0usize;
        for &p in ptrs {
            if p == 0 {
                host.iter_mut().for_each(|b| *b = 0);
            } else {
                gpu.copy_d2h(DevicePtr(p), &mut host)?;
            }
            out.write_all(&host)?;
            written += blob_bytes;
        }
        out.flush()?;
        self.meta.push(format!(
            "    \"{name}_bytes\": {written}, \"{name}_count\": {}, \"{name}_blob_bytes\": {blob_bytes}",
            ptrs.len()
        ));
        Ok(())
    }

    /// Write the manifest. Call after every `buf`/`blobs`/`scalar`.
    pub(crate) fn finish(self) {
        let body = self.meta.join(",\n");
        let json = format!("{{\n{body}\n}}\n");
        std::fs::write(self.dir.join("meta.json"), json).ok();
        tracing::info!(
            "ATLAS_QWEN4_ORACLE_DUMP: wrote {}",
            self.dir.display()
        );
    }
}
