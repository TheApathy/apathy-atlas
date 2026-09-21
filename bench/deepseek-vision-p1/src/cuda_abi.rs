// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::Result;
use libloading::Library;
use std::ffi::{c_char, c_void};
pub type Handle = *mut c_void;
pub unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
    Ok(*unsafe { lib.get::<T>(name) }?)
}
pub struct Api {
    pub init: unsafe extern "C" fn(u32) -> i32,
    pub device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub device_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32,
    pub device_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
    pub version: unsafe extern "C" fn(*mut i32) -> i32,
    pub context_create: unsafe extern "C" fn(*mut Handle, u32, i32) -> i32,
    pub context_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub context_sync: unsafe extern "C" fn() -> i32,
    pub stream_create: unsafe extern "C" fn(*mut Handle, u32) -> i32,
    pub stream_sync: unsafe extern "C" fn(Handle) -> i32,
    pub stream_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    pub free: unsafe extern "C" fn(u64) -> i32,
    pub h2d: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    pub d2h: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    pub memset: unsafe extern "C" fn(u64, u8, usize) -> i32,
    pub module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> i32,
    pub module_unload: unsafe extern "C" fn(Handle) -> i32,
    pub function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> i32,
    pub launch: unsafe extern "C" fn(
        Handle,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
}
impl Api {
    pub unsafe fn load(lib: &Library) -> Result<Self> {
        macro_rules! s {
            ($n:literal) => {
                unsafe { sym(lib, concat!($n, "\0").as_bytes()) }?
            };
        }
        Ok(Self {
            init: s!("cuInit"),
            device_get: s!("cuDeviceGet"),
            device_name: s!("cuDeviceGetName"),
            device_attribute: s!("cuDeviceGetAttribute"),
            version: s!("cuDriverGetVersion"),
            context_create: s!("cuCtxCreate_v2"),
            context_destroy: s!("cuCtxDestroy_v2"),
            context_sync: s!("cuCtxSynchronize"),
            stream_create: s!("cuStreamCreate"),
            stream_sync: s!("cuStreamSynchronize"),
            stream_destroy: s!("cuStreamDestroy_v2"),
            alloc: s!("cuMemAlloc_v2"),
            free: s!("cuMemFree_v2"),
            h2d: s!("cuMemcpyHtoD_v2"),
            d2h: s!("cuMemcpyDtoH_v2"),
            memset: s!("cuMemsetD8_v2"),
            module_load: s!("cuModuleLoadData"),
            module_unload: s!("cuModuleUnload"),
            function: s!("cuModuleGetFunction"),
            launch: s!("cuLaunchKernel"),
        })
    }
}
