// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-3 production storage backend: `io_uring` (IORING_SETUP_SQPOLL +
// IORING_REGISTER_BUFFERS) + per-buffer `CudaEvent` for safe reuse across
// async H→D DMAs. Per-buffer events let us keep QD≥8 in flight without the
// per-op `cuStreamSynchronize` that throttled the POSIX backend to QD=1.

use anyhow::{Context, Result, bail};
use io_uring::{IoUring, opcode, types};
use std::os::fd::RawFd;

use super::{ReadRequest, StorageBackend};
use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_pinned_async, stream_sync};
use crate::group::GroupKey;
use crate::layout::Layout;

pub struct IoUringBackend {
    layout: Layout,
    ring: IoUring,
    buffers: Vec<PinnedBuffer>,
    pending: Vec<Option<PendingTransfer>>, // transfer per buffer, None = idle
    qd: usize,
    poisoned: bool,
}

struct PendingTransfer {
    event: Option<CudaEvent>,
    stream: u64,
}

impl PendingTransfer {
    fn wait(&self) -> Result<()> {
        match &self.event {
            Some(event) => event.sync().or_else(|_| stream_sync(self.stream)),
            None => stream_sync(self.stream),
        }
    }
}

impl IoUringBackend {
    pub fn new(layout: Layout, qd: usize) -> Result<Self> {
        if qd == 0 {
            bail!("queue depth must be ≥ 1");
        }
        // SQPOLL: kernel polls SQ; idle 2s before parking.
        let ring = IoUring::builder()
            .setup_sqpoll(2_000)
            .build(qd as u32)
            .context("io_uring build")?;

        let group_bytes = layout.group_bytes() as usize;
        let mut buffers = Vec::with_capacity(qd);
        for _ in 0..qd {
            buffers.push(PinnedBuffer::new(group_bytes)?);
        }
        // Register the pinned host buffers with io_uring for zero-copy
        // direct-IO. After this, ReadFixed at index `i` lands in `buffers[i]`.
        let iovecs: Vec<libc::iovec> = buffers
            .iter_mut()
            .map(|b| libc::iovec {
                iov_base: b.as_mut_ptr(),
                iov_len: b.len(),
            })
            .collect();
        unsafe {
            ring.submitter()
                .register_buffers(&iovecs)
                .context("register_buffers")?;
        }
        let pending: Vec<Option<PendingTransfer>> = (0..qd).map(|_| None).collect();
        Ok(Self {
            layout,
            ring,
            buffers,
            pending,
            qd,
            poisoned: false,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Test helper: drop the page cache for the layer files so subsequent
    /// reads actually hit NVMe.
    pub fn drop_pagecache(&self) {
        for layer in 0..self.layout.spec.num_layers {
            let fd = self.layout.fd(layer);
            unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// Wait for the previous DMA out of `buf_idx` to complete (if any) so
    /// we can reuse the buffer for a new io_uring read.
    fn wait_buffer_free(&mut self, buf_idx: usize) -> Result<()> {
        if let Some(transfer) = &self.pending[buf_idx] {
            transfer.wait()?;
        }
        self.pending[buf_idx] = None;
        Ok(())
    }

    /// Submit one read request into `buf_idx` and return its user_data tag.
    fn submit_read(
        &mut self,
        fd: RawFd,
        offset: u64,
        bytes: u32,
        buf_idx: u16,
        user_data: u64,
    ) -> Result<()> {
        let buf_ptr = self.buffers[buf_idx as usize].as_mut_ptr().cast::<u8>();
        let read_e = opcode::ReadFixed::new(types::Fd(fd), buf_ptr, bytes, buf_idx)
            .offset(offset)
            .build()
            .user_data(user_data);
        unsafe {
            self.ring
                .submission()
                .push(&read_e)
                .map_err(|_| anyhow::anyhow!("io_uring SQ full"))?;
        }
        Ok(())
    }
}

impl StorageBackend for IoUringBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        if self.poisoned {
            bail!("io_uring backend is poisoned by an earlier incomplete batch");
        }
        // user_data layout: high 16 bits = req index, low 16 bits = buf index.
        // (We never submit > 65k requests in one batch.)
        if requests.len() > u16::MAX as usize {
            bail!("io_uring batch too large: {}", requests.len());
        }

        let result = (|| {
            let bytes = self.layout.group_bytes() as u32;
            let mut next_submit = 0;
            let mut completed = 0;
            // Buffer ownership: free buffers form a stack; busy ones are claimed
            // by an in-flight read until its CQE arrives.
            let mut free_bufs: Vec<u16> = (0..self.qd as u16).rev().collect();

            while completed < requests.len() {
                // Submit while we have a free buffer and pending requests.
                while next_submit < requests.len() {
                    let Some(&buf_idx) = free_bufs.last() else {
                        break;
                    };
                    self.wait_buffer_free(buf_idx as usize)?;
                    free_bufs.pop();
                    let req = &requests[next_submit];
                    let fd = self.layout.fd(req.group.layer);
                    let off = self.layout.offset(req.group);
                    let user = ((next_submit as u64) << 16) | (buf_idx as u64);
                    self.submit_read(fd, off, bytes, buf_idx, user)?;
                    next_submit += 1;
                }
                // Submit and wait for at least one completion.
                self.ring
                    .submit_and_wait(1)
                    .context("io_uring submit_and_wait")?;
                // Drain everything that's ready.
                let cq = self.ring.completion();
                for cqe in cq {
                    let user = cqe.user_data();
                    let buf_idx = (user & 0xFFFF) as u16;
                    let req_idx = (user >> 16) as usize;
                    let result = cqe.result();
                    if result < 0 {
                        bail!("io_uring read failed for req {req_idx}: errno {}", -result);
                    }
                    if result as u32 != bytes {
                        bail!("io_uring short read: req {req_idx} got {result}, expected {bytes}");
                    }
                    let req = &requests[req_idx];
                    let buf_idx = buf_idx as usize;
                    let ev = CudaEvent::new()?;
                    // Install the stream-level fallback before submission. CUDA
                    // may surface deferred work through an error return, so every
                    // post-submission exit retains a completion path.
                    self.pending[buf_idx] = Some(PendingTransfer {
                        event: None,
                        stream,
                    });
                    let buf = &mut self.buffers[buf_idx];
                    // SAFETY: `self.buffers[buf_idx]` stays allocated and is not
                    // reused until the recorded event (or fallback stream drain)
                    // completes; Drop drains every pending transfer as well.
                    unsafe {
                        copy_h_to_d_pinned_async(
                            req.dst_dev_ptr,
                            buf.pinned_slice(bytes as usize)?,
                            stream,
                        )?;
                    }
                    if let Err(error) = ev.record(stream) {
                        // The copy was accepted but no event can prove completion.
                        return Err(error).context("record H2D completion event");
                    }
                    self.pending[buf_idx] = Some(PendingTransfer {
                        event: Some(ev),
                        stream,
                    });
                    free_bufs.push(buf_idx as u16);
                    completed += 1;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        if self.poisoned {
            bail!("io_uring backend is poisoned by an earlier incomplete batch");
        }
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        let result = (|| {
            // Stage through buffer 0 — pinned + page-aligned for O_DIRECT.
            self.wait_buffer_free(0)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    self.buffers[0].as_mut_ptr().cast::<u8>(),
                    bytes,
                );
            }
            let fd = self.layout.fd(key.layer);
            let off = self.layout.offset(key) as i64;
            let n = unsafe { libc::pwrite(fd, self.buffers[0].as_mut_ptr(), bytes, off) };
            if n != bytes as isize {
                bail!(
                    "pwrite {bytes}@{off} returned {n}, errno {}",
                    std::io::Error::last_os_error()
                );
            }
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

impl Drop for IoUringBackend {
    fn drop(&mut self) {
        // Page-locked buffers are fields of `self`; drain every outstanding
        // DMA before Rust drops those owners. If CUDA cannot prove completion,
        // abort rather than permit either host or device transfer endpoints to
        // be destroyed while work may still target them.
        let drained = self
            .pending
            .iter()
            .flatten()
            .all(|transfer| transfer.wait().is_ok());
        if !drained {
            // Neither the event nor the owning stream could prove DMA
            // completion. Continuing would let external device destinations
            // be freed while a transfer may still target them.
            eprintln!("fatal: unable to prove io_uring CUDA H2D completion during Drop");
            std::process::abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h};
    use crate::group::{GroupKey, GroupLayout, KvKind};
    use std::path::PathBuf;

    fn tempdir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("atlas-iouring-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("rt");
        let spec = GroupLayout::new(1, 4, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = IoUringBackend::new(layout, 4).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        // Three different patterns at three different keys to exercise SQ depth.
        let patterns: Vec<(GroupKey, Vec<u8>)> = (0..4u32)
            .map(|b| {
                let k = GroupKey::new(0, b, 0, KvKind::K);
                let pat: Vec<u8> = (0..bytes)
                    .map(|i| ((i + b as usize) & 0xFF) as u8)
                    .collect();
                (k, pat)
            })
            .collect();
        for (k, p) in &patterns {
            backend.write_from_host(*k, p).unwrap();
        }
        backend.drop_pagecache();
        let dev: Vec<DeviceBuffer> = patterns
            .iter()
            .map(|_| DeviceBuffer::new(bytes).unwrap())
            .collect();
        let reqs: Vec<ReadRequest> = patterns
            .iter()
            .zip(&dev)
            .map(|((k, _), d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.ptr,
            })
            .collect();
        backend.read(&reqs, _ctx.stream).unwrap();
        for ((_, expected), d) in patterns.iter().zip(&dev) {
            let mut got = vec![0_u8; bytes];
            copy_d_to_h(&mut got, d.ptr, _ctx.stream).unwrap();
            assert_eq!(&got, expected);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
