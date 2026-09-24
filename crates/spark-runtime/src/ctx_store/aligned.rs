// SPDX-License-Identifier: AGPL-3.0-only

//! Page-aligned, zero-initialised host buffer for `O_DIRECT` I/O.
//!
//! `O_DIRECT` needs the buffer address, the file offset and the transfer
//! length to be multiples of the logical block size. Every buffer is
//! allocated with a capacity rounded up to [`ALIGN`] so a section can be
//! written or read in one call without a bounce copy; the tail padding is
//! zero and is never interpreted.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;

use anyhow::{Result, bail};

/// Alignment and padding granule of every buffer and file section.
pub const ALIGN: usize = 4096;

/// Round `n` up to the next multiple of [`ALIGN`].
pub fn round_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// A zeroed host buffer whose address and capacity are [`ALIGN`]-aligned.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

// SAFETY: the buffer exclusively owns its allocation; no interior aliasing.
unsafe impl Send for AlignedBuf {}
// SAFETY: shared access only hands out `&[u8]`.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Allocate `len` logical bytes (capacity padded to [`ALIGN`]), zeroed.
    pub fn zeroed(len: usize) -> Result<Self> {
        let cap = round_up(len).max(ALIGN);
        let layout = Layout::from_size_align(cap, ALIGN)?;
        // SAFETY: layout has non-zero size.
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            bail!("ctx_store: failed to allocate {cap} aligned bytes");
        };
        Ok(Self { ptr, len, cap })
    }

    /// Copy `bytes` into a new aligned buffer.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let mut buf = Self::zeroed(bytes.len())?;
        buf.as_mut_slice().copy_from_slice(bytes);
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity including the zero padding (always a multiple of [`ALIGN`]).
    pub fn padded_len(&self) -> usize {
        self.cap
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `len <= cap` bytes are allocated and initialised (zeroed).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The whole padded allocation, for aligned whole-block I/O.
    pub fn padded_slice(&self) -> &[u8] {
        // SAFETY: `cap` bytes are allocated and initialised (zeroed).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    pub fn padded_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with exactly this layout.
        unsafe {
            dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align_unchecked(self.cap, ALIGN),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_padded_and_aligned() {
        let buf = AlignedBuf::zeroed(5000).unwrap();
        assert_eq!(buf.len(), 5000);
        assert_eq!(buf.padded_len(), 8192);
        assert_eq!(buf.as_slice().as_ptr() as usize % ALIGN, 0);
        assert!(buf.padded_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn empty_buffer_still_has_one_granule() {
        let buf = AlignedBuf::zeroed(0).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.padded_len(), ALIGN);
    }

    #[test]
    fn from_slice_round_trips() {
        let buf = AlignedBuf::from_slice(&[1, 2, 3]).unwrap();
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        assert_eq!(&buf.padded_slice()[3..8], &[0; 5]);
    }
}
