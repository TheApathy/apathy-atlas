// SPDX-License-Identifier: AGPL-3.0-only
//! Owned first-heuristic Lt adapter. No algorithm mutation or timed autotuning.
use crate::lt_plan::{BoundLt, LtCall, LtIo, SelectedAlgorithm};
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{
    cuda_abi::{Handle, sym},
    driver::Driver,
    io,
    lt_abi::{Api, Heuristic},
};
use libloading::Library;
use serde_json::{Value, json};
use std::{
    ffi::c_void,
    mem::{align_of, size_of},
    path::Path,
    ptr,
};

type ConfigGet = unsafe extern "C" fn(*const c_void, u32, *mut c_void, usize, *mut usize) -> i32;
fn check(status: i32, name: &str) -> Result<()> {
    ensure!(status == 0, "cuBLASLt {name}: {status}");
    Ok(())
}
pub struct Lt<'a> {
    api: Api,
    config_get: ConfigGet,
    driver: &'a Driver,
    handle: Handle,
    desc: Handle,
    pref: Handle,
    layouts: Vec<Handle>,
    call: Option<LtCall>,
    requested_mask: Option<u32>,
    heuristic: Heuristic,
    selected: bool,
    version: usize,
    _library: Library,
}
impl<'a> Lt<'a> {
    pub fn new(driver: &'a Driver, path: &Path) -> Result<Self> {
        ensure!(
            size_of::<usize>() == 8 && cfg!(target_endian = "little"),
            "64-bit LE ABI required"
        );
        ensure!(
            size_of::<Heuristic>() == 96 && align_of::<Heuristic>() >= 8,
            "Lt heuristic ABI drift"
        );
        let library = unsafe { Library::new(path) }?;
        let api = unsafe { Api::load(&library) }?;
        let config_get = unsafe { sym(&library, b"cublasLtMatmulAlgoConfigGetAttribute\0") }?;
        Ok(Self {
            api,
            config_get,
            driver,
            handle: ptr::null_mut(),
            desc: ptr::null_mut(),
            pref: ptr::null_mut(),
            layouts: vec![],
            call: None,
            requested_mask: None,
            selected: false,
            version: 0,
            heuristic: Heuristic {
                algo: [0; 64],
                workspace: 0,
                state: -1,
                waves: 0.0,
                reserved: [0; 4],
            },
            _library: library,
        })
    }
    fn attribute(&self, attr: u32) -> Result<u32> {
        let mut value = 0u32;
        let mut size_written = 0usize;
        check(
            unsafe {
                (self.config_get)(
                    self.heuristic.algo.as_ptr().cast(),
                    attr,
                    (&mut value as *mut u32).cast(),
                    size_of::<u32>(),
                    &mut size_written,
                )
            },
            "algorithm attribute",
        )?;
        ensure!(
            size_written == size_of::<u32>(),
            "Lt algorithm attribute size drift"
        );
        Ok(value)
    }
    pub fn receipt(&self, a: &SelectedAlgorithm) -> Result<Value> {
        let c = self.call.context("Lt call receipt missing")?;
        Ok(
            json!({"function":"cublasLtMatmul","requested_reduction_mask":self.requested_mask,
            "baseline_preference":"unset means pinned CUDA13 default mask7; not a Torch-backend assertion",
            "algorithm_id":a.algorithm_id,"tile_id":a.tile_id,"split_k":a.split_k,
            "reduction_scheme":a.reduction_scheme,"workspace_used":a.workspace_bytes,
            "workspace_available":c.workspace.bytes,"waves":a.waves,"cublaslt_version":self.version,
            "algorithm_bytes":self.heuristic.algo.to_vec(),"algorithm_sha256":io::hash(&self.heuristic.algo)?,
            "heuristic_count_requested":1,"autotune":false,"transpose":c.transpose,"m_n_k":c.m_n_k,
            "weight_layout":c.weight_layout,"input_layout":c.input_layout,"output_layout":c.output_layout,
            "A_B_C_D_dtype":14,"compute_type":c.compute_type,"scale_type":c.scale_type,
            "alpha":c.alpha,"beta":c.beta,"stream":"owned nondefault","tf32_operand_path":false}),
        )
    }
}
impl LtIo for Lt<'_> {
    fn configure(&mut self, bound: &BoundLt) -> Result<()> {
        ensure!(self.call.is_none(), "Lt owner cannot be reconfigured");
        let c = bound.call();
        ensure!(
            c.stream == self.driver.stream as u64,
            "Lt stream does not belong to CUDA owner"
        );
        self.call = Some(c);
        self.requested_mask = bound.mode().preference_mask();
        unsafe {
            check((self.api.create)(&mut self.handle), "create")?;
            ensure!(!self.handle.is_null(), "null Lt handle");
            self.version = (self.api.version)();
            ensure!(self.version > 0, "missing Lt version");
            check(
                (self.api.desc_create)(&mut self.desc, c.compute_type, c.scale_type),
                "descriptor create",
            )?;
            ensure!(!self.desc.is_null(), "null Lt descriptor");
            for (attr, value) in [(3u32, c.transpose[0]), (4, c.transpose[1])] {
                check(
                    (self.api.desc_set)(
                        self.desc,
                        attr,
                        (&value as *const i32).cast(),
                        size_of::<i32>(),
                    ),
                    "transpose",
                )?;
            }
            for (dtype, layout) in [
                (c.a_type, c.weight_layout),
                (c.b_type, c.input_layout),
                (c.c_type, c.output_layout),
            ] {
                let mut handle = ptr::null_mut();
                let status = (self.api.layout_create)(
                    &mut handle,
                    dtype,
                    layout[0],
                    layout[1],
                    layout[2] as i64,
                );
                // Preserve any returned handle even if the API reports failure.
                if !handle.is_null() {
                    self.layouts.push(handle);
                }
                check(status, "layout create")?;
                ensure!(!handle.is_null(), "null Lt layout");
            }
            check((self.api.pref_create)(&mut self.pref), "preference create")?;
            ensure!(!self.pref.is_null(), "null Lt preference");
            check(
                (self.api.pref_set)(
                    self.pref,
                    1,
                    (&c.workspace.bytes as *const usize).cast(),
                    size_of::<usize>(),
                ),
                "workspace preference",
            )?;
            if let Some(mask) = self.requested_mask {
                check(
                    (self.api.pref_set)(
                        self.pref,
                        3,
                        (&mask as *const u32).cast(),
                        size_of::<u32>(),
                    ),
                    "reduction preference",
                )?;
            }
        }
        Ok(())
    }
    fn select_first(&mut self) -> Result<SelectedAlgorithm> {
        let c = self.call.context("Lt not configured")?;
        ensure!(
            self.layouts.len() == 3 && !self.pref.is_null(),
            "Lt configuration incomplete"
        );
        let (a, b, y) = (self.layouts[0], self.layouts[1], self.layouts[2]);
        let mut returned = 0;
        check(
            unsafe {
                (self.api.heuristic)(
                    self.handle,
                    self.desc,
                    a,
                    b,
                    y,
                    y,
                    self.pref,
                    c.heuristic_count,
                    &mut self.heuristic,
                    &mut returned,
                )
            },
            "first heuristic",
        )?;
        ensure!(
            returned == 1 && self.heuristic.state == 0,
            "no successful first heuristic"
        );
        let selected = SelectedAlgorithm {
            returned,
            state: self.heuristic.state,
            algorithm_id: self.attribute(0)? as i32,
            tile_id: self.attribute(1)?,
            split_k: self.attribute(2)? as i32,
            reduction_scheme: self.attribute(3)?,
            workspace_bytes: self.heuristic.workspace,
            waves: self.heuristic.waves,
        };
        self.selected = true;
        Ok(selected)
    }
    fn matmul(&mut self) -> Result<()> {
        let c = self.call.context("Lt not configured")?;
        ensure!(
            self.selected && self.layouts.len() == 3,
            "Lt heuristic unavailable"
        );
        let (a, b, y) = (self.layouts[0], self.layouts[1], self.layouts[2]);
        check(
            unsafe {
                (self.api.matmul)(
                    self.handle,
                    self.desc,
                    (&c.alpha as *const f32).cast(),
                    c.a as *const c_void,
                    a,
                    c.b as *const c_void,
                    b,
                    (&c.beta as *const f32).cast(),
                    c.c as *const c_void,
                    y,
                    c.d as *mut c_void,
                    y,
                    self.heuristic.algo.as_ptr().cast(),
                    c.workspace.ptr as *mut c_void,
                    c.workspace.bytes,
                    c.stream as Handle,
                )
            },
            "matmul",
        )
    }
    fn synchronize(&mut self) -> Result<()> {
        self.driver.sync()
    }
    fn close(&mut self) -> Result<()> {
        let mut errors = Vec::new();
        macro_rules! record {
            ($call:expr,$name:literal) => {
                if let Err(e) = check(unsafe { $call }, $name) {
                    errors.push(e.to_string());
                }
            };
        }
        if !self.pref.is_null() {
            record!((self.api.pref_destroy)(self.pref), "preference destroy");
            self.pref = ptr::null_mut();
        }
        for layout in self.layouts.drain(..).rev() {
            record!((self.api.layout_destroy)(layout), "layout destroy");
        }
        if !self.desc.is_null() {
            record!((self.api.desc_destroy)(self.desc), "descriptor destroy");
            self.desc = ptr::null_mut();
        }
        if !self.handle.is_null() {
            record!((self.api.destroy)(self.handle), "handle destroy");
            self.handle = ptr::null_mut();
        }
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }
}
impl Drop for Lt<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null()
            || !self.desc.is_null()
            || !self.pref.is_null()
            || !self.layouts.is_empty()
        {
            let drain = self.synchronize();
            let close = self.close();
            if drain.is_err() || close.is_err() {
                eprintln!("Lt emergency completion={drain:?}; cleanup={close:?}");
            }
        }
    }
}
