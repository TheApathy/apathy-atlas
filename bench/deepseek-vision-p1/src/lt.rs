// SPDX-License-Identifier: AGPL-3.0-only
use crate::{
    cuda_abi::Handle,
    driver::{Buffer, Driver},
    io,
    lt_abi::{Api, Heuristic},
};
use anyhow::{Result, ensure};
use libloading::Library;
use serde_json::{Value, json};
use std::{ffi::c_void, path::Path, ptr};
const WORKSPACE: usize = 64 * 1024 * 1024;
struct Lt {
    api: Api,
    handle: Handle,
    desc: Handle,
    layouts: Vec<Handle>,
    pref: Handle,
    _library: Library,
}
fn check(code: i32, name: &str) -> Result<()> {
    ensure!(code == 0, "cuBLASLt {name}: {code}");
    Ok(())
}
impl Lt {
    fn new(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }?;
        let api = unsafe { Api::load(&library) }?;
        let mut lt = Self {
            api,
            handle: ptr::null_mut(),
            desc: ptr::null_mut(),
            layouts: vec![],
            pref: ptr::null_mut(),
            _library: library,
        };
        unsafe {
            check((lt.api.create)(&mut lt.handle), "create")?;
        }
        Ok(lt)
    }
    fn run(
        &mut self,
        d: &Driver,
        a: Buffer,
        w: Buffer,
        out: Buffer,
        workspace: Buffer,
    ) -> Result<Value> {
        ensure!(
            a.bytes == 20 * 2816 * 2
                && w.bytes == 1024 * 2816 * 2
                && out.bytes == 20 * 1024 * 2
                && workspace.bytes == WORKSPACE,
            "Lt fixed dimensions"
        );
        ensure!(std::mem::size_of::<Heuristic>() == 96, "Lt ABI host width");
        unsafe {
            // No TF32 input or FP8 fast-accumulation path: BF16 A/B, FP32 compute.
            check((self.api.desc_create)(&mut self.desc, 68, 0), "desc")?;
            for (attr, value) in [(3u32, 1i32), (4, 0)] {
                check(
                    (self.api.desc_set)(self.desc, attr, (&value as *const i32).cast(), 4),
                    "transpose",
                )?;
            }
            for (rows, cols, ld) in [
                (2816u64, 1024u64, 2816i64),
                (2816, 20, 2816),
                (1024, 20, 1024),
            ] {
                let mut layout = ptr::null_mut();
                check(
                    (self.api.layout_create)(&mut layout, 14, rows, cols, ld),
                    "layout",
                )?;
                self.layouts.push(layout);
            }
            check((self.api.pref_create)(&mut self.pref), "preference")?;
            check(
                (self.api.pref_set)(self.pref, 1, (&WORKSPACE as *const usize).cast(), 8),
                "workspace preference",
            )?;
            let mut heuristic = Heuristic {
                algo: [0; 64],
                workspace: 0,
                state: -1,
                waves: 0.0,
                reserved: [0; 4],
            };
            let mut returned = 0;
            let (la, lb, lc) = (self.layouts[0], self.layouts[1], self.layouts[2]);
            check(
                (self.api.heuristic)(
                    self.handle,
                    self.desc,
                    la,
                    lb,
                    lc,
                    lc,
                    self.pref,
                    1,
                    &mut heuristic,
                    &mut returned,
                ),
                "heuristic",
            )?;
            ensure!(
                returned == 1
                    && heuristic.state == 0
                    && heuristic.workspace <= WORKSPACE
                    && heuristic.waves.is_finite(),
                "no valid bounded first heuristic"
            );
            let (alpha, beta) = (1.0f32, 0.0f32);
            check(
                (self.api.matmul)(
                    self.handle,
                    self.desc,
                    (&alpha as *const f32).cast(),
                    w.ptr as *const c_void,
                    la,
                    a.ptr as *const c_void,
                    lb,
                    (&beta as *const f32).cast(),
                    out.ptr as *const c_void,
                    lc,
                    out.ptr as *mut c_void,
                    lc,
                    heuristic.algo.as_ptr().cast(),
                    workspace.ptr as *mut c_void,
                    WORKSPACE,
                    d.stream,
                ),
                "matmul",
            )?;
            d.sync()?;
            Ok(
                json!({"cublaslt_version":(self.api.version)(),"compute_type":"CUBLAS_COMPUTE_32F",
                "A_B_C_dtype":"BF16","tf32_operand_path":false,"alpha":1,"beta":0,
                "math_contract":"library first heuristic; not an assertion of PyTorch reduced-precision mode",
                "transpose_A":"T","transpose_B":"N","weight_layout":[2816,1024,2816],
                "activation_layout":[2816,20,2816],"output_layout":[1024,20,1024],
                "heuristic_count_requested":1,"autotune":false,"algorithm_bytes":heuristic.algo.to_vec(),
                "algorithm_sha256":io::hash(&heuristic.algo)?,"heuristic_workspace":heuristic.workspace,
                "workspace_bytes":WORKSPACE,"waves":heuristic.waves}),
            )
        }
    }
    fn close(&mut self) -> Result<()> {
        let mut errors = Vec::new();
        macro_rules! record {
            ($v:expr,$name:literal) => {
                if let Err(e) = check(unsafe { $v }, $name) {
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
            record!((self.api.desc_destroy)(self.desc), "desc destroy");
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
impl Drop for Lt {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            if let Err(e) = self.close() {
                eprintln!("Lt emergency cleanup: {e}");
            }
        }
    }
}
pub fn fc2(d: &mut Driver, path: &Path, a: Buffer, w: Buffer, out: Buffer) -> Result<Value> {
    let workspace = d.allocate(WORKSPACE, 0)?;
    let mut lt = Lt::new(path)?;
    let result = lt.run(d, a, w, out, workspace);
    // Always drain before destroying descriptors/handle, including launch error.
    let drain = d.sync();
    let close = lt.close();
    match (result, drain, close) {
        (Ok(v), Ok(()), Ok(())) => Ok(v),
        (a, b, c) => anyhow::bail!("Lt run/cleanup failed: result={a:?}, drain={b:?}, close={c:?}"),
    }
}
