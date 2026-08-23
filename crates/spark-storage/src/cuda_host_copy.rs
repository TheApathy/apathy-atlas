// SPDX-License-Identifier: AGPL-3.0-only

//! Typed CUDA host-copy boundaries for the storage-only driver shim.

use anyhow::{Result, bail};
use std::ffi::c_void;

use super::stream_sync;

unsafe extern "C" {
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    fn cuMemFreeHost(p: *mut c_void) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
}

pub struct PinnedBuffer {
    ptr: std::ptr::NonNull<c_void>,
    bytes: usize,
}

// SAFETY: the allocation has a stable address, safe mutation requires an
// exclusive borrow, and safe shared access exposes only immutable descriptors.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            bail!("pinned allocation length must be nonzero");
        }
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { cuMemAllocHost_v2(&mut ptr, bytes) };
        if status != 0 {
            bail!("cuMemAllocHost_v2({bytes}) failed: {status}");
        }
        let ptr = std::ptr::NonNull::new(ptr)
            .ok_or_else(|| anyhow::anyhow!("cuMemAllocHost_v2 returned a null allocation"))?;
        Ok(Self { ptr, bytes })
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Borrow a prefix as a non-forgeable page-locked source descriptor.
    pub fn pinned_slice(&self, bytes: usize) -> Result<PinnedHostSlice<'_>> {
        if bytes > self.bytes {
            bail!(
                "pinned slice length {bytes} exceeds allocation length {}",
                self.bytes
            );
        }
        Ok(PinnedHostSlice {
            ptr: self.ptr.as_ptr().cast_const(),
            bytes,
            _owner: std::marker::PhantomData,
        })
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFreeHost(self.ptr.as_ptr());
        }
    }
}

/// Opaque borrow of a CUDA page-locked allocation.
pub struct PinnedHostSlice<'a> {
    ptr: *const c_void,
    bytes: usize,
    _owner: std::marker::PhantomData<&'a PinnedBuffer>,
}

/// Submit a genuinely asynchronous copy from page-locked host storage.
///
/// # Safety
/// The source owner, destination device allocation, stream and CUDA context
/// must remain alive, and the source must not be mutated, until a completion
/// barrier ordered after this submission succeeds. Callers must conservatively
/// retain that fallback even when submission reports an error.
pub unsafe fn copy_h_to_d_pinned_async(
    dst: u64,
    src: PinnedHostSlice<'_>,
    stream: u64,
) -> Result<()> {
    let status = unsafe { cuMemcpyHtoDAsync_v2(dst, src.ptr, src.bytes, stream) };
    if status != 0 {
        bail!("cuMemcpyHtoDAsync_v2 failed: {status}");
    }
    Ok(())
}

/// Copy from a page-locked owner and return only after the source is reusable.
pub fn copy_h_to_d_pinned(dst: u64, src: PinnedHostSlice<'_>, stream: u64) -> Result<()> {
    stream_sync(stream)?;
    let status = unsafe { cuMemcpyHtoD_v2(dst, src.ptr, src.bytes) };
    if status != 0 {
        bail!("cuMemcpyHtoD_v2 failed: {status}");
    }
    Ok(())
}

mod h2d_element {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for half::bf16 {}
}

/// Initialized plain-data element that can be viewed as host source bytes.
pub trait H2dElement: h2d_element::Sealed {}

impl H2dElement for u8 {}
impl H2dElement for i32 {}
impl H2dElement for u32 {}
impl H2dElement for u64 {}
impl H2dElement for f32 {}
impl H2dElement for half::bf16 {}

/// One member of an ordered pageable-host upload group.
pub struct HostToDeviceCopy<'a> {
    dst: u64,
    src: &'a [u8],
}

impl<'a> HostToDeviceCopy<'a> {
    pub fn new<T: H2dElement>(dst: u64, src: &'a [T]) -> Self {
        // SAFETY: every admitted element has no padding, all source bytes are
        // initialized, and the byte slice cannot outlive `src`.
        let src = unsafe {
            std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src))
        };
        Self { dst, src }
    }
}

/// Copy an ordered group from ordinary pageable host storage.
///
/// A nonempty group drains the producer stream exactly once, then performs
/// synchronous copies in order. A failing member suppresses all later copies.
pub fn copy_h_to_d_group(copies: &[HostToDeviceCopy<'_>], stream: u64) -> Result<()> {
    if copies.is_empty() {
        return Ok(());
    }
    stream_sync(stream)?;
    for (index, copy) in copies.iter().enumerate() {
        let status = unsafe {
            cuMemcpyHtoD_v2(copy.dst, copy.src.as_ptr().cast::<c_void>(), copy.src.len())
        };
        if status != 0 {
            bail!("cuMemcpyHtoD_v2 group member {index} failed: {status}");
        }
    }
    Ok(())
}

/// Copy one ordinary pageable host slice to device memory.
pub fn copy_h_to_d<T: H2dElement>(dst: u64, src: &[T], stream: u64) -> Result<()> {
    copy_h_to_d_group(&[HostToDeviceCopy::new(dst, src)], stream)
}

mod d2h_element {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for f32 {}
    impl Sealed for half::bf16 {}
}

/// Host element whose complete bit-pattern space is valid for CUDA readback.
pub trait D2hElement: d2h_element::Sealed {}

impl D2hElement for u8 {}
impl D2hElement for f32 {}
impl D2hElement for half::bf16 {}

/// Copy device bytes into ordinary pageable host storage.
///
/// The producer stream is drained before the streamless synchronous copy, so
/// callers do not need page-locked memory or a trailing stream sync.
pub fn copy_d_to_h<T: D2hElement>(dst: &mut [T], src: u64, stream: u64) -> Result<()> {
    stream_sync(stream)?;
    let bytes = std::mem::size_of_val(dst);
    let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr().cast::<c_void>(), src, bytes) };
    if status != 0 {
        bail!("cuMemcpyDtoH_v2 failed: {status}");
    }
    Ok(())
}
