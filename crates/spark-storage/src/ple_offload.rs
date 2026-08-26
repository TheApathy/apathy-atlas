// SPDX-License-Identifier: AGPL-3.0-only

//! Sparse O_DIRECT reader for Qwen3.8-Flash-Next PLE n-gram rows.
//!
//! The table has 320M rows but inference selects only 16 rows per token.
//! Sidecars page-pack 91 NVFP4 records into 8 KiB, so every selected row is
//! fulfilled by exactly one aligned read with only two padding bytes/page.

use anyhow::{Context, Result, bail, ensure};
use io_uring::{IoUring, opcode, types};
use serde::Deserialize;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::os::fd::{FromRawFd, RawFd};
use std::path::{Path, PathBuf};

const EXPECTED_FORMAT: &str = "qwen38-flash-next-ple-nvfp4-direct-v1";
const EXPECTED_PAGE_BYTES: usize = 8192;
const EXPECTED_RECORD_BYTES: usize = 90;
const EXPECTED_RECORDS_PER_PAGE: usize = 91;

#[derive(Debug, Deserialize)]
struct Manifest {
    format: String,
    page_bytes: usize,
    record_bytes: usize,
    records_per_page: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    group_size: usize,
    entries: Vec<ManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    shard: usize,
    file: PathBuf,
    rows: usize,
    width: usize,
    scale2: f32,
    bytes: usize,
}

struct AlignedPage {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the allocation is uniquely owned and this type deliberately does
// not implement Sync. Moving ownership to an I/O worker is safe.
unsafe impl Send for AlignedPage {}

impl AlignedPage {
    fn new() -> Self {
        let layout = Layout::from_size_align(EXPECTED_PAGE_BYTES, EXPECTED_PAGE_BYTES)
            .expect("valid PLE direct-I/O page layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout }
    }

    fn record(&self, slot: usize) -> [u8; EXPECTED_RECORD_BYTES] {
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.add(slot * EXPECTED_RECORD_BYTES),
                record.as_mut_ptr(),
                EXPECTED_RECORD_BYTES,
            );
        }
        record
    }
}

impl Drop for AlignedPage {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

struct Shard {
    file: File,
    rows: usize,
    scale2: f32,
}

/// One packed PLE row: 80 E2M1 bytes, 10 E4M3 group scales, and its global scale.
#[derive(Clone, Debug)]
pub struct PleNvfp4Row {
    pub record: [u8; EXPECTED_RECORD_BYTES],
    pub scale2: f32,
}

/// Batched sparse PLE reader. A decode request normally contains 16 rows.
pub struct PleOffloadReader {
    ring: IoUring,
    pages: Vec<AlignedPage>,
    shards: Vec<Shard>,
    queue_depth: usize,
    cache: HashMap<(usize, usize), CachedPage>,
    cache_order: VecDeque<((usize, usize), u64)>,
    cache_capacity_pages: usize,
    cache_clock: u64,
}

struct CachedPage {
    bytes: Box<[u8; EXPECTED_PAGE_BYTES]>,
    generation: u64,
}

impl PleOffloadReader {
    pub fn open(manifest_path: &Path, queue_depth: usize, cache_bytes: usize) -> Result<Self> {
        ensure!(
            queue_depth >= 16,
            "PLE offload queue depth must be at least 16"
        );
        ensure!(
            queue_depth <= u16::MAX as usize,
            "PLE offload queue depth exceeds io_uring fixed-buffer index range"
        );
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(manifest_path)
                .with_context(|| format!("read PLE manifest {}", manifest_path.display()))?,
        )?;
        ensure!(
            manifest.format == EXPECTED_FORMAT,
            "unsupported PLE offload format {}",
            manifest.format
        );
        ensure!(
            manifest.page_bytes == EXPECTED_PAGE_BYTES,
            "PLE page size mismatch"
        );
        ensure!(
            manifest.record_bytes == EXPECTED_RECORD_BYTES,
            "PLE record size mismatch"
        );
        ensure!(
            manifest.records_per_page == EXPECTED_RECORDS_PER_PAGE,
            "PLE records/page mismatch"
        );
        ensure!(
            manifest.packed_bytes == 80 && manifest.scale_bytes == 10,
            "PLE field geometry mismatch"
        );
        ensure!(manifest.group_size == 16, "PLE NVFP4 group size must be 16");
        ensure!(!manifest.entries.is_empty(), "PLE manifest has no shards");

        let base = manifest_path
            .parent()
            .context("PLE manifest has no parent directory")?;
        let mut ordered = manifest.entries;
        ordered.sort_by_key(|entry| entry.shard);
        let mut shards = Vec::with_capacity(ordered.len());
        for (expected, entry) in ordered.into_iter().enumerate() {
            ensure!(
                entry.shard == expected,
                "PLE shards must be contiguous: expected {expected}, got {}",
                entry.shard
            );
            ensure!(
                entry.width == 160,
                "PLE shard {expected} width {} != 160",
                entry.width
            );
            ensure!(
                entry.scale2.is_finite() && entry.scale2 > 0.0,
                "PLE shard {expected} has invalid scale2"
            );
            let pages = entry.rows.div_ceil(EXPECTED_RECORDS_PER_PAGE);
            ensure!(
                entry.bytes == pages * EXPECTED_PAGE_BYTES,
                "PLE shard {expected} byte count mismatch"
            );
            let path = base.join(entry.file);
            let file =
                open_direct(&path).with_context(|| format!("open PLE shard {}", path.display()))?;
            ensure!(
                file.metadata()?.len() as usize == entry.bytes,
                "PLE shard {expected} file size mismatch"
            );
            shards.push(Shard {
                file,
                rows: entry.rows,
                scale2: entry.scale2,
            });
        }

        let ring = IoUring::new(queue_depth as u32).context("create PLE io_uring")?;
        let mut pages: Vec<AlignedPage> = (0..queue_depth).map(|_| AlignedPage::new()).collect();
        let iovecs: Vec<libc::iovec> = pages
            .iter_mut()
            .map(|page| libc::iovec {
                iov_base: page.ptr.cast(),
                iov_len: EXPECTED_PAGE_BYTES,
            })
            .collect();
        unsafe { ring.submitter().register_buffers(&iovecs) }
            .context("register PLE io_uring buffers")?;
        Ok(Self {
            ring,
            pages,
            shards,
            queue_depth,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_capacity_pages: cache_bytes / EXPECTED_PAGE_BYTES,
            cache_clock: 0,
        })
    }

    fn cached_record(
        &mut self,
        key: (usize, usize),
        slot: usize,
    ) -> Option<[u8; EXPECTED_RECORD_BYTES]> {
        let cached = self.cache.get_mut(&key)?;
        self.cache_clock = self.cache_clock.wrapping_add(1);
        cached.generation = self.cache_clock;
        self.cache_order.push_back((key, self.cache_clock));
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        let start = slot * EXPECTED_RECORD_BYTES;
        record.copy_from_slice(&cached.bytes[start..start + EXPECTED_RECORD_BYTES]);
        Some(record)
    }

    /// Read selected `(shard, row)` pairs in caller order. Identical pages in
    /// a batch are coalesced before submission.
    pub fn read_rows(&mut self, selections: &[(usize, usize)]) -> Result<Vec<PleNvfp4Row>> {
        ensure!(!selections.is_empty(), "PLE offload selection is empty");
        let mut unique = Vec::<(usize, usize)>::new();
        let mut lookup = HashMap::<(usize, usize), usize>::new();
        let mut requested = Vec::with_capacity(selections.len());
        let mut output = vec![None; selections.len()];
        for (request_index, &(shard_idx, row)) in selections.iter().enumerate() {
            let (shard_rows, scale2) = self
                .shards
                .get(shard_idx)
                .map(|shard| (shard.rows, shard.scale2))
                .with_context(|| format!("PLE shard {shard_idx} out of range"))?;
            ensure!(
                row < shard_rows,
                "PLE row {row} out of range for shard {shard_idx}"
            );
            let page = row / EXPECTED_RECORDS_PER_PAGE;
            let key = (shard_idx, page);
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            if let Some(record) = self.cached_record(key, slot) {
                output[request_index] = Some(PleNvfp4Row { record, scale2 });
                continue;
            }
            let index = match lookup.get(&key) {
                Some(&index) => index,
                None => {
                    let index = unique.len();
                    lookup.insert(key, index);
                    unique.push(key);
                    index
                }
            };
            requested.push((request_index, index, slot, scale2));
        }
        if unique.is_empty() {
            return output
                .into_iter()
                .map(|row| row.context("PLE cache lookup left an empty row"))
                .collect();
        }
        ensure!(
            unique.len() <= self.queue_depth,
            "PLE request needs {} unique pages but queue depth is {}",
            unique.len(),
            self.queue_depth
        );

        for (index, &(shard_idx, page)) in unique.iter().enumerate() {
            let fd = raw_fd(&self.shards[shard_idx].file);
            let entry = opcode::ReadFixed::new(
                types::Fd(fd),
                self.pages[index].ptr,
                EXPECTED_PAGE_BYTES as u32,
                index as u16,
            )
            .offset((page * EXPECTED_PAGE_BYTES) as u64)
            .build()
            .user_data(index as u64);
            unsafe { self.ring.submission().push(&entry) }
                .map_err(|_| anyhow::anyhow!("PLE io_uring submission queue full"))?;
        }
        self.ring
            .submit_and_wait(unique.len())
            .context("submit PLE reads")?;
        let mut completed = vec![false; unique.len()];
        for entry in self.ring.completion() {
            let index = entry.user_data() as usize;
            if index >= completed.len() {
                bail!("PLE io_uring returned invalid tag {index}");
            }
            if entry.result() != EXPECTED_PAGE_BYTES as i32 {
                bail!(
                    "PLE read {index} returned {}, expected {EXPECTED_PAGE_BYTES}",
                    entry.result()
                );
            }
            completed[index] = true;
        }
        ensure!(
            completed.iter().all(|done| *done),
            "PLE io_uring batch completed partially"
        );
        for (page_index, &key) in unique.iter().enumerate() {
            // Copying a cold 8 KiB page is bounded by the configured cache;
            // cache_bytes=0 keeps the pure O_DIRECT path allocation-free.
            let source_ptr = self.pages[page_index].ptr;
            let source = AlignedPageRef { ptr: source_ptr };
            self.cache_page_ref(key, source);
        }
        for (request_index, page, slot, scale2) in requested {
            output[request_index] = Some(PleNvfp4Row {
                record: self.pages[page].record(slot),
                scale2,
            });
        }
        output
            .into_iter()
            .map(|row| row.context("PLE read left an empty row"))
            .collect()
    }

    fn cache_page_ref(&mut self, key: (usize, usize), source: AlignedPageRef) {
        if self.cache_capacity_pages == 0 {
            return;
        }
        self.cache_clock = self.cache_clock.wrapping_add(1);
        let generation = self.cache_clock;
        let mut bytes = Box::new([0u8; EXPECTED_PAGE_BYTES]);
        unsafe {
            std::ptr::copy_nonoverlapping(source.ptr, bytes.as_mut_ptr(), EXPECTED_PAGE_BYTES);
        }
        self.cache.insert(key, CachedPage { bytes, generation });
        self.cache_order.push_back((key, generation));
        while self.cache.len() > self.cache_capacity_pages {
            let Some((old_key, old_generation)) = self.cache_order.pop_front() else {
                break;
            };
            if self
                .cache
                .get(&old_key)
                .is_some_and(|page| page.generation == old_generation)
            {
                self.cache.remove(&old_key);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct AlignedPageRef {
    ptr: *const u8,
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    let fd = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "PLE O_DIRECT offload requires Linux",
    ))
}

#[cfg(unix)]
fn raw_fd(file: &File) -> RawFd {
    use std::os::fd::AsRawFd;
    file.as_raw_fd()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_packing_has_no_cross_page_rows() {
        assert_eq!(EXPECTED_RECORDS_PER_PAGE * EXPECTED_RECORD_BYTES, 8190);
        for row in 0..10_000 {
            let slot = row % EXPECTED_RECORDS_PER_PAGE;
            assert!((slot + 1) * EXPECTED_RECORD_BYTES <= EXPECTED_PAGE_BYTES);
        }
    }
}
