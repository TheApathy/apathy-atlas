// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

pub(super) fn timed_sample(mut launch: impl FnMut() -> Result<()>) -> Result<f32> {
    const CHAINS: usize = 10;
    let (mut start, mut end) = (0u64, 0u64);
    unsafe {
        if cuEventCreate(&mut start, 0) != 0 || cuEventCreate(&mut end, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        if cuEventRecord(start, 0) != 0 {
            bail!("cuEventRecord(start) failed");
        }
    }
    for _ in 0..CHAINS {
        launch()?;
    }
    let mut elapsed = 0.0f32;
    unsafe {
        if cuEventRecord(end, 0) != 0 || cuEventSynchronize(end) != 0 {
            bail!("cuEventRecord/synchronize(end) failed");
        }
        if cuEventElapsedTime(&mut elapsed, start, end) != 0 {
            bail!("cuEventElapsedTime failed");
        }
        cuEventDestroy_v2(start);
        cuEventDestroy_v2(end);
    }
    Ok(elapsed * 1000.0 / CHAINS as f32)
}

pub(super) fn percentile(samples: &mut [f32], numerator: usize, denominator: usize) -> f32 {
    samples.sort_by(f32::total_cmp);
    samples[(samples.len() - 1) * numerator / denominator]
}
