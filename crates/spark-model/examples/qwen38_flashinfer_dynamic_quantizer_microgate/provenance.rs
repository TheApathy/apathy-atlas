// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::{c_char, c_void};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::KernelHandle;

use super::contract::{
    ATTR_COMPUTE_CAPABILITY_MAJOR, ATTR_COMPUTE_CAPABILITY_MINOR, ATTR_MULTIPROCESSOR_COUNT,
    FUNC_ATTR_LOCAL_SIZE_BYTES, FUNC_ATTR_MAX_THREADS_PER_BLOCK, FUNC_ATTR_NUM_REGS,
    FUNC_ATTR_SHARED_SIZE_BYTES, FunctionResources,
};

unsafe extern "C" {
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, length: i32, device: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: u32, function: *mut c_void) -> i32;
}

pub(super) fn exact_bundle() -> Result<Vec<(&'static str, &'static str)>> {
    ensure!(
        std::env::var("ATLAS_TARGET_MODEL").as_deref() == Ok("qwen3.8-27b"),
        "requires ATLAS_TARGET_MODEL=qwen3.8-27b"
    );
    ensure!(
        std::env::var("ATLAS_TARGET_QUANT").as_deref() == Ok("nvfp4"),
        "requires ATLAS_TARGET_QUANT=nvfp4"
    );
    let mut matches: Vec<_> = atlas_kernels::available_targets()
        .into_iter()
        .filter(|set| {
            set.target.arch == "sm_121"
                && set.target.model == "qwen3.8-27b"
                && set.target.quant == "nvfp4"
        })
        .collect();
    ensure!(
        matches.len() == 1,
        "expected exactly one SM121 Qwen3.8 NVFP4 bundle"
    );
    Ok(matches.pop().context("exact bundle disappeared")?.modules)
}

pub(super) fn require_gb10() -> Result<()> {
    let mut device = -1;
    ensure!(
        unsafe { cuCtxGetDevice(&mut device) } == 0,
        "cuCtxGetDevice failed"
    );
    let attribute = |kind| -> Result<i32> {
        let mut value = -1;
        ensure!(
            unsafe { cuDeviceGetAttribute(&mut value, kind, device) } == 0,
            "cuDeviceGetAttribute({kind}) failed"
        );
        Ok(value)
    };
    let (major, minor, sms) = (
        attribute(ATTR_COMPUTE_CAPABILITY_MAJOR)?,
        attribute(ATTR_COMPUTE_CAPABILITY_MINOR)?,
        attribute(ATTR_MULTIPROCESSOR_COUNT)?,
    );
    let mut raw_name = [0 as c_char; 256];
    ensure!(
        unsafe { cuDeviceGetName(raw_name.as_mut_ptr(), raw_name.len() as i32, device) } == 0,
        "cuDeviceGetName failed"
    );
    let name = unsafe { std::ffi::CStr::from_ptr(raw_name.as_ptr()) }.to_string_lossy();
    ensure!(
        major == 12 && minor == 1 && sms == 48,
        "requires GB10 SM121/48SM, got {name} sm_{major}{minor} sms={sms}"
    );
    println!("DEVICE name={name} sm={major}.{minor} sms={sms} exact=PASS");
    Ok(())
}

pub(super) fn function_resources(kernel: KernelHandle) -> Result<FunctionResources> {
    let function = kernel.0 as usize as *mut c_void;
    let attribute = |kind, label| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuFuncGetAttribute(&mut value, kind, function) };
        ensure!(
            status == 0 && value >= 0,
            "cuFuncGetAttribute({label}) failed: {status}"
        );
        Ok(value)
    };
    Ok(FunctionResources {
        max_threads: attribute(FUNC_ATTR_MAX_THREADS_PER_BLOCK, "MAX_THREADS")?,
        shared_bytes: attribute(FUNC_ATTR_SHARED_SIZE_BYTES, "SHARED")?,
        local_bytes: attribute(FUNC_ATTR_LOCAL_SIZE_BYTES, "LOCAL")?,
        registers: attribute(FUNC_ATTR_NUM_REGS, "REGISTERS")?,
    })
}
