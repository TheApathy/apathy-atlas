// SPDX-License-Identifier: AGPL-3.0-only

//! Exact identity for CUDA transform-cache producers.

use super::{cuCtxGetDevice, cuDeviceGetAttribute, cuDriverGetVersion};
use anyhow::{Result, bail};

pub(super) fn identity(registry_identity: &str) -> Result<String> {
    const COMPUTE_CAPABILITY_MAJOR: u32 = 75;
    const COMPUTE_CAPABILITY_MINOR: u32 = 76;
    let mut device = 0i32;
    let mut driver = 0i32;
    let mut major = 0i32;
    let mut minor = 0i32;
    let check = |status: i32, label: &str| -> Result<()> {
        if status != 0 {
            bail!("{label} failed while identifying transform-cache runtime: {status}");
        }
        Ok(())
    };
    check(unsafe { cuCtxGetDevice(&mut device) }, "cuCtxGetDevice")?;
    check(
        unsafe { cuDriverGetVersion(&mut driver) },
        "cuDriverGetVersion",
    )?;
    check(
        unsafe { cuDeviceGetAttribute(&mut major, COMPUTE_CAPABILITY_MAJOR, device) },
        "cuDeviceGetAttribute(compute-major)",
    )?;
    check(
        unsafe { cuDeviceGetAttribute(&mut minor, COMPUTE_CAPABILITY_MINOR, device) },
        "cuDeviceGetAttribute(compute-minor)",
    )?;
    Ok(format!(
        "{registry_identity};cuda_device={device};cuda_driver={driver};\
         compute_capability={major}.{minor}",
    ))
}
