// SPDX-License-Identifier: AGPL-3.0-only
//! ABI matches pinned spark-runtime/src/cublaslt.rs. No global handles/workspace.
use crate::cuda_abi::{Handle, sym};
use anyhow::Result;
use libloading::Library;
use std::ffi::c_void;
#[repr(C)]
pub struct Heuristic {
    pub algo: [u8; 64],
    pub workspace: usize,
    pub state: i32,
    pub waves: f32,
    pub reserved: [i32; 4],
}
pub struct Api {
    pub create: unsafe extern "C" fn(*mut Handle) -> i32,
    pub destroy: unsafe extern "C" fn(Handle) -> i32,
    pub version: unsafe extern "C" fn() -> usize,
    pub desc_create: unsafe extern "C" fn(*mut Handle, i32, i32) -> i32,
    pub desc_set: unsafe extern "C" fn(Handle, u32, *const c_void, usize) -> i32,
    pub desc_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub layout_create: unsafe extern "C" fn(*mut Handle, i32, u64, u64, i64) -> i32,
    pub layout_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub pref_create: unsafe extern "C" fn(*mut Handle) -> i32,
    pub pref_set: unsafe extern "C" fn(Handle, u32, *const c_void, usize) -> i32,
    pub pref_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub heuristic: unsafe extern "C" fn(
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        i32,
        *mut Heuristic,
        *mut i32,
    ) -> i32,
    pub matmul: unsafe extern "C" fn(
        Handle,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *const c_void,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *mut c_void,
        Handle,
        *const c_void,
        *mut c_void,
        usize,
        Handle,
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
            create: s!("cublasLtCreate"),
            destroy: s!("cublasLtDestroy"),
            version: s!("cublasLtGetVersion"),
            desc_create: s!("cublasLtMatmulDescCreate"),
            desc_set: s!("cublasLtMatmulDescSetAttribute"),
            desc_destroy: s!("cublasLtMatmulDescDestroy"),
            layout_create: s!("cublasLtMatrixLayoutCreate"),
            layout_destroy: s!("cublasLtMatrixLayoutDestroy"),
            pref_create: s!("cublasLtMatmulPreferenceCreate"),
            pref_set: s!("cublasLtMatmulPreferenceSetAttribute"),
            pref_destroy: s!("cublasLtMatmulPreferenceDestroy"),
            heuristic: s!("cublasLtMatmulAlgoGetHeuristic"),
            matmul: s!("cublasLtMatmul"),
        })
    }
}
#[test]
fn heuristic_matches_installed_64_bit_abi() {
    assert_eq!(std::mem::size_of::<Heuristic>(), 96);
    assert_eq!(std::mem::offset_of!(Heuristic, workspace), 64);
    assert_eq!(std::mem::offset_of!(Heuristic, state), 72);
}
