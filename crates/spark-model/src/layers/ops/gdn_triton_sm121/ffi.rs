// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};
use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;
pub(super) const FUNC_MAX_THREADS: i32 = 0;
pub(super) const FUNC_STATIC_SHARED: i32 = 1;
pub(super) const FUNC_LOCAL_BYTES: i32 = 3;
pub(super) const FUNC_REGISTERS: i32 = 4;
pub(super) const FUNC_MAX_DYNAMIC_SHARED: i32 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Module(NonNull<c_void>);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct Function(NonNull<c_void>);

pub(super) struct ContextLease {
    context: NonNull<c_void>,
    device: i32,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuCtxGetCurrent(context: *mut *mut c_void) -> i32;
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: i32) -> i32;
    fn cuDevicePrimaryCtxRelease_v2(device: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, length: i32, device: i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn cuModuleUnload(module: *mut c_void) -> i32;
    fn cuModuleGetFunction(
        function: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: i32, function: *mut c_void) -> i32;
    fn cuFuncSetAttribute(function: *mut c_void, attribute: i32, value: i32) -> i32;
    fn cuLaunchKernel(
        function: *mut c_void,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_bytes: u32,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

fn check(status: i32, operation: &str) -> Result<()> {
    if status != 0 {
        bail!("{operation} failed with CUDA status {status}");
    }
    Ok(())
}

unsafe fn attribute(device: i32, kind: i32) -> Result<i32> {
    let mut value = 0;
    check(
        unsafe { cuDeviceGetAttribute(&mut value, kind, device) },
        "cuDeviceGetAttribute",
    )?;
    Ok(value)
}

impl ContextLease {
    pub(super) unsafe fn retain_current_gb10() -> Result<Self> {
        let mut current = std::ptr::null_mut();
        check(unsafe { cuCtxGetCurrent(&mut current) }, "cuCtxGetCurrent")?;
        let current =
            NonNull::new(current).ok_or_else(|| anyhow::anyhow!("no current CUDA context"))?;
        let mut device = -1;
        check(unsafe { cuCtxGetDevice(&mut device) }, "cuCtxGetDevice")?;
        let mut retained = std::ptr::null_mut();
        check(
            unsafe { cuDevicePrimaryCtxRetain(&mut retained, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        if retained != current.as_ptr() {
            let _ = unsafe { cuDevicePrimaryCtxRelease_v2(device) };
            bail!("current CUDA context is not the retained primary context");
        }
        let lease = Self {
            context: current,
            device,
            _not_send_or_sync: PhantomData,
        };
        let mut name = [0i8; 256];
        check(
            unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, device) },
            "cuDeviceGetName",
        )?;
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_str()?;
        let identity = (
            unsafe { attribute(device, ATTR_COMPUTE_CAPABILITY_MAJOR) }?,
            unsafe { attribute(device, ATTR_COMPUTE_CAPABILITY_MINOR) }?,
            unsafe { attribute(device, ATTR_MULTIPROCESSOR_COUNT) }?,
        );
        if !name.contains("GB10") || identity != (12, 1, 48) {
            bail!("frozen Triton requires exact GB10 SM12.1/48SM");
        }
        Ok(lease)
    }

    pub(super) unsafe fn ensure_current(&self) -> Result<()> {
        let mut current = std::ptr::null_mut();
        check(unsafe { cuCtxGetCurrent(&mut current) }, "cuCtxGetCurrent")?;
        ensure!(current == self.context.as_ptr(), "CUDA context changed");
        Ok(())
    }
}

impl Drop for ContextLease {
    fn drop(&mut self) {
        let _ = unsafe { cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}

pub(super) unsafe fn load_data(bytes: &[u8]) -> Result<Module> {
    ensure!(!bytes.is_empty(), "empty cubin image");
    let mut module = std::ptr::null_mut();
    check(
        unsafe { cuModuleLoadData(&mut module, bytes.as_ptr().cast()) },
        "cuModuleLoadData",
    )?;
    NonNull::new(module)
        .map(Module)
        .ok_or_else(|| anyhow::anyhow!("cuModuleLoadData returned null"))
}

pub(super) unsafe fn unload(module: Module) {
    let _ = unsafe { cuModuleUnload(module.0.as_ptr()) };
}

pub(super) unsafe fn function(module: Module, name: &str) -> Result<Function> {
    let name = CString::new(name)?;
    let mut function = std::ptr::null_mut();
    check(
        unsafe { cuModuleGetFunction(&mut function, module.0.as_ptr(), name.as_ptr()) },
        "cuModuleGetFunction",
    )?;
    NonNull::new(function)
        .map(Function)
        .ok_or_else(|| anyhow::anyhow!("cuModuleGetFunction returned null"))
}

pub(super) unsafe fn func_attribute(function: Function, kind: i32) -> Result<i32> {
    let mut value = 0;
    check(
        unsafe { cuFuncGetAttribute(&mut value, kind, function.0.as_ptr()) },
        "cuFuncGetAttribute",
    )?;
    Ok(value)
}

pub(super) unsafe fn set_dynamic_shared(function: Function, bytes: u32) -> Result<()> {
    let bytes = i32::try_from(bytes)?;
    check(
        unsafe { cuFuncSetAttribute(function.0.as_ptr(), FUNC_MAX_DYNAMIC_SHARED, bytes) },
        "cuFuncSetAttribute",
    )
}

pub(super) unsafe fn launch(
    function: Function,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    stream: u64,
    params: &mut [*mut c_void],
) -> Result<()> {
    ensure!(stream != 0, "default CUDA stream rejected");
    let stream = usize::try_from(stream)? as *mut c_void;
    check(
        unsafe {
            cuLaunchKernel(
                function.0.as_ptr(),
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "cuLaunchKernel",
    )
}
