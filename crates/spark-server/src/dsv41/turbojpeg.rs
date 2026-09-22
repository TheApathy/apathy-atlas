// SPDX-License-Identifier: AGPL-3.0-only

//! JPEG decode through libjpeg-turbo, loaded at runtime.
//!
//! Pillow decodes JPEG with libjpeg-turbo (3.0.3 bundled in Pillow 10.4). The
//! `image` crate's zune-jpeg disagrees with it on ~28% of RGB values, by up to
//! 12 levels, on our fixtures, so a V4.1 image would reach the model as
//! different pixels than under the Python engine. The system libjpeg-turbo
//! (Ubuntu `libturbojpeg`, 2.1.5) decodes those fixtures BIT-IDENTICALLY to
//! Pillow, measured through ctypes and again by `vision::tests`.
//!
//! The library is dlopen'd (`libturbojpeg.so.0`): no build-time dependency.
//! If it is absent, the caller falls back to zune-jpeg with a warning.
//! libjpeg-turbo is BSD-3-Clause / IJG / zlib, which is AGPL-compatible.

use std::ffi::{c_int, c_uchar, c_ulong, c_void};
use std::sync::OnceLock;

use anyhow::{Result, bail};
use image::RgbImage;
use libloading::Library;

type InitFn = unsafe extern "C" fn() -> *mut c_void;
type HeaderFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type DecompressFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;

const TJPF_RGB: c_int = 0;

struct TurboJpeg {
    _lib: Library,
    init: InitFn,
    header: HeaderFn,
    decompress: DecompressFn,
    destroy: DestroyFn,
}

fn load() -> Option<TurboJpeg> {
    let path = std::env::var("ATLAS_TURBOJPEG").unwrap_or_else(|_| "libturbojpeg.so.0".into());
    // SAFETY: loading a C library by name; the symbols below are the stable
    // TurboJPEG 2.x API (tjInitDecompress, tjDecompressHeader3,
    // tjDecompress2, tjDestroy), present in 2.0+ and 3.x.
    unsafe {
        let lib = Library::new(&path).ok()?;
        let init = *lib.get::<InitFn>(b"tjInitDecompress\0").ok()?;
        let header = *lib.get::<HeaderFn>(b"tjDecompressHeader3\0").ok()?;
        let decompress = *lib.get::<DecompressFn>(b"tjDecompress2\0").ok()?;
        let destroy = *lib.get::<DestroyFn>(b"tjDestroy\0").ok()?;
        Some(TurboJpeg {
            _lib: lib,
            init,
            header,
            decompress,
            destroy,
        })
    }
}

fn library() -> Option<&'static TurboJpeg> {
    static LIB: OnceLock<Option<TurboJpeg>> = OnceLock::new();
    LIB.get_or_init(|| {
        let lib = load();
        if lib.is_none() {
            tracing::warn!(
                "libturbojpeg.so.0 not found: DeepSeek-V4.1 JPEG input falls back to zune-jpeg, whose \
                 pixels differ from the Python engine's (install libturbojpeg to match)"
            );
        }
        lib
    })
    .as_ref()
}

pub fn available() -> bool {
    library().is_some()
}

/// Decode a JPEG to RGB with libjpeg-turbo's defaults (ISLOW IDCT, fancy
/// upsampling), as Pillow does. `Ok(None)` when the library is unavailable.
pub fn decode_rgb(raw: &[u8]) -> Result<Option<RgbImage>> {
    let Some(tj) = library() else { return Ok(None) };
    // SAFETY: the handle is created and destroyed here; the output buffer is
    // sized from the header, width*height*3 for TJPF_RGB with pitch 0.
    unsafe {
        let h = (tj.init)();
        if h.is_null() {
            bail!("tjInitDecompress failed");
        }
        let (mut w, mut hgt, mut sub, mut cs) = (0, 0, 0, 0);
        let rc = (tj.header)(
            h,
            raw.as_ptr(),
            raw.len() as c_ulong,
            &mut w,
            &mut hgt,
            &mut sub,
            &mut cs,
        );
        if rc != 0 || w <= 0 || hgt <= 0 {
            (tj.destroy)(h);
            bail!("invalid PNG/JPEG image");
        }
        let mut buf = vec![0u8; w as usize * hgt as usize * 3];
        let rc = (tj.decompress)(
            h,
            raw.as_ptr(),
            raw.len() as c_ulong,
            buf.as_mut_ptr(),
            w,
            0,
            hgt,
            TJPF_RGB,
            0,
        );
        (tj.destroy)(h);
        if rc != 0 {
            bail!("invalid PNG/JPEG image");
        }
        Ok(RgbImage::from_raw(w as u32, hgt as u32, buf))
    }
}
