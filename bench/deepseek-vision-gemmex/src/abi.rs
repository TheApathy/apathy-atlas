// SPDX-License-Identifier: AGPL-3.0-only
//! CUDA 13 cublas_api.h C ABI; no CUDA/library initialization at import time.
use crate::protocol::{Command, GemmIo};
use anyhow::{Result, ensure};
use deepseek_vision_p1::{
    cuda_abi::{Handle, sym},
    driver::Driver,
};
use libloading::Library;
use std::{ffi::c_void, path::Path, ptr};

type Status = i32;
struct Api {
    create: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    version: unsafe extern "C" fn(Handle, *mut i32) -> Status,
    stream: unsafe extern "C" fn(Handle, Handle) -> Status,
    workspace: unsafe extern "C" fn(Handle, *mut c_void, usize) -> Status,
    pointer_mode: unsafe extern "C" fn(Handle, i32) -> Status,
    math_mode: unsafe extern "C" fn(Handle, i32) -> Status,
    get_math_mode: unsafe extern "C" fn(Handle, *mut i32) -> Status,
    gemm: unsafe extern "C" fn(
        Handle,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const c_void,
        *const c_void,
        i32,
        i32,
        *const c_void,
        i32,
        i32,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
    ) -> Status,
}
impl Api {
    unsafe fn load(lib: &Library) -> Result<Self> {
        macro_rules! s {
            ($n:literal) => {
                unsafe { sym(lib, concat!($n, "\0").as_bytes()) }?
            };
        }
        Ok(Self {
            create: s!("cublasCreate_v2"),
            destroy: s!("cublasDestroy_v2"),
            version: s!("cublasGetVersion_v2"),
            stream: s!("cublasSetStream_v2"),
            workspace: s!("cublasSetWorkspace_v2"),
            pointer_mode: s!("cublasSetPointerMode_v2"),
            math_mode: s!("cublasSetMathMode"),
            get_math_mode: s!("cublasGetMathMode"),
            gemm: s!("cublasGemmEx"),
        })
    }
}
fn check(code: i32, name: &str) -> Result<()> {
    ensure!(code == 0, "cuBLAS {name} failed: {code}");
    Ok(())
}
pub struct Blas<'a> {
    api: Api,
    handle: Handle,
    driver: &'a Driver,
    pub version: i32,
    _library: Library,
    _lt_library: Library,
}
impl<'a> Blas<'a> {
    pub fn open(driver: &'a Driver, path: &Path, lt_path: &Path) -> Result<Self> {
        ensure!(
            std::mem::size_of::<usize>() == 8 && cfg!(target_endian = "little"),
            "64-bit little-endian ABI required"
        );
        driver.sync()?;
        // Explicitly retain the admitted Lt dependency for the handle lifetime.
        let lt_library = unsafe { Library::new(lt_path) }?;
        let library = unsafe { Library::new(path) }?;
        let api = unsafe { Api::load(&library) }?;
        let mut blas = Self {
            api,
            handle: ptr::null_mut(),
            driver,
            version: 0,
            _library: library,
            _lt_library: lt_library,
        };
        unsafe {
            check((blas.api.create)(&mut blas.handle), "create")?;
            ensure!(!blas.handle.is_null(), "null cuBLAS handle");
            check(
                (blas.api.version)(blas.handle, &mut blas.version),
                "version",
            )?;
        }
        Ok(blas)
    }
    pub fn close(&mut self) -> Result<()> {
        if self.handle.is_null() {
            return Ok(());
        }
        let mut errors = Vec::new();
        if let Err(e) = self.driver.sync() {
            errors.push(format!("drain: {e:#}"));
        }
        if let Err(e) = self.execute(Command::SetMathMode(0)) {
            errors.push(format!("restore: {e:#}"));
        }
        let status = unsafe { (self.api.destroy)(self.handle) };
        self.handle = ptr::null_mut();
        if let Err(e) = check(status, "destroy") {
            errors.push(e.to_string());
        }
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }
}
impl GemmIo for Blas<'_> {
    fn execute(&mut self, command: Command) -> Result<()> {
        ensure!(!self.handle.is_null(), "cuBLAS owner is closed");
        unsafe {
            match command {
                Command::SetStream(s) => {
                    ensure!(
                        s != 0 && s == self.driver.stream as u64,
                        "stream must belong to driver"
                    );
                    check((self.api.stream)(self.handle, s as Handle), "set stream")
                }
                Command::SetWorkspace(s) => check(
                    (self.api.workspace)(self.handle, s.ptr as *mut c_void, s.bytes),
                    "set workspace",
                ),
                Command::SetHostPointerMode => {
                    check((self.api.pointer_mode)(self.handle, 0), "host pointer mode")
                }
                Command::SetMathMode(mode) => {
                    ensure!(matches!(mode, 0 | 16), "unreviewed math mode");
                    check((self.api.math_mode)(self.handle, mode), "set math mode")?;
                    let mut actual = -1;
                    check(
                        (self.api.get_math_mode)(self.handle, &mut actual),
                        "get math mode",
                    )?;
                    ensure!(actual == mode, "math mode was not applied");
                    Ok(())
                }
                Command::Synchronize => self.driver.sync(),
                Command::Gemm(c) => check(
                    (self.api.gemm)(
                        self.handle,
                        c.transa,
                        c.transb,
                        c.m,
                        c.n,
                        c.k,
                        (&c.alpha as *const f32).cast(),
                        c.a as *const c_void,
                        c.a_type,
                        c.lda,
                        c.b as *const c_void,
                        c.b_type,
                        c.ldb,
                        (&c.beta as *const f32).cast(),
                        c.c as *mut c_void,
                        c.c_type,
                        c.ldc,
                        c.compute_type,
                        c.algorithm,
                    ),
                    "GemmEx",
                ),
            }
        }
    }
}
impl Drop for Blas<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            eprintln!("cuBLAS emergency cleanup: {e:#}");
        }
    }
}
