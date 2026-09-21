// SPDX-License-Identifier: AGPL-3.0-only

//! GPU promotion gate for the exact DeepSeek K64/K2 M64xN128/N256 prefill rungs.
//!
//! Compares the generic persistent N64 kernel, the fixed-shape N64 kernel,
//! and the fixed-shape N128/N256 kernels byte for byte at both production
//! matrix shapes. It also proves that each wider entry fails closed on a wrong
//! block size and reports same-boot CUDA-event timings.

use anyhow::{Result, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const COUNTS: [usize; 5] = [1, 63, 64, 65, 129];
const SHAPES: [(usize, usize); 2] = [(2048usize, 4096usize), (4096, 2048)];

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn u16(&mut self) -> u16 {
        (self.next() >> 40) as u16
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    input: DevicePtr,
    trellis: DevicePtr,
    offsets: DevicePtr,
    experts: u32,
    n: u32,
    k: u32,
    n_tile: u32,
    threads: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([experts * (n / n_tile), 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(input)
        .arg_ptr(trellis)
        .arg_ptr(output)
        .arg_ptr(offsets)
        .arg_ptr(DevicePtr(0))
        .arg_u32(experts)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(2)
        .arg_u32(1)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn time_launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    input: DevicePtr,
    trellis: DevicePtr,
    offsets: DevicePtr,
    experts: u32,
    n: u32,
    k: u32,
    n_tile: u32,
    threads: u32,
) -> Result<f32> {
    const ITERS: usize = 5;
    for _ in 0..2 {
        launch(
            gpu, kernel, output, input, trellis, offsets, experts, n, k, n_tile, threads,
        )?;
    }
    gpu.synchronize(0)?;
    let (mut start, mut end) = (0u64, 0u64);
    unsafe {
        if cuEventCreate(&mut start, 0) != 0 || cuEventCreate(&mut end, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        if cuEventRecord(start, 0) != 0 {
            bail!("cuEventRecord(start) failed");
        }
    }
    for _ in 0..ITERS {
        launch(
            gpu, kernel, output, input, trellis, offsets, experts, n, k, n_tile, threads,
        )?;
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
    Ok(elapsed / ITERS as f32)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let generic = gpu.kernel("exl3_grouped_prefill_k64_k2", "exl3_grouped_prefill_k64_k2")?;
    let fixed = [
        gpu.kernel(
            "exl3_grouped_prefill_k64_k2_gu",
            "exl3_grouped_prefill_k64_k2_gu",
        )?,
        gpu.kernel(
            "exl3_grouped_prefill_k64_k2_down",
            "exl3_grouped_prefill_k64_k2_down",
        )?,
    ];
    let wide = [
        gpu.kernel(
            "exl3_grouped_prefill_k64_n128_k2_gu",
            "exl3_grouped_prefill_k64_n128_k2_gu",
        )?,
        gpu.kernel(
            "exl3_grouped_prefill_k64_n128_k2_down",
            "exl3_grouped_prefill_k64_n128_k2_down",
        )?,
    ];
    let widest = [
        gpu.kernel(
            "exl3_grouped_prefill_k64_n256_k2_gu",
            "exl3_grouped_prefill_k64_n256_k2_gu",
        )?,
        gpu.kernel(
            "exl3_grouped_prefill_k64_n256_k2_down",
            "exl3_grouped_prefill_k64_n256_k2_down",
        )?,
    ];
    let rows: usize = COUNTS.iter().sum();
    let mut offsets = vec![0i32];
    for count in COUNTS {
        offsets.push(offsets.last().copied().unwrap() + count as i32);
    }
    let offset_bytes: Vec<u8> = offsets.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_offsets = upload(gpu, &offset_bytes)?;
    let mut pass = true;

    for (shape, &(n, k)) in SHAPES.iter().enumerate() {
        let mut rng = Rng(0x4e31_3238_2026_0827 ^ shape as u64);
        let input: Vec<u16> = (0..rows * k)
            .map(|_| bf16::from_f32((rng.unit() - 0.5) * 0.5).to_bits())
            .collect();
        let input_bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_input = upload(gpu, &input_bytes)?;
        let words = (k / 16) * (n / 16) * 32;
        let trellis_host: Vec<Vec<u16>> = (0..COUNTS.len())
            .map(|_| (0..words).map(|_| rng.u16()).collect())
            .collect();
        let mut d_experts = Vec::with_capacity(COUNTS.len());
        for expert in &trellis_host {
            let bytes: Vec<u8> = expert.iter().flat_map(|x| x.to_le_bytes()).collect();
            d_experts.push(upload(gpu, &bytes)?);
        }
        let table: Vec<u8> = d_experts
            .iter()
            .flat_map(|ptr| ptr.0.to_le_bytes())
            .collect();
        let d_table = upload(gpu, &table)?;
        let output_bytes = rows * n * 2;
        let d_generic = gpu.alloc(output_bytes)?;
        let d_fixed = gpu.alloc(output_bytes)?;
        let d_wide = gpu.alloc(output_bytes)?;
        let d_widest = gpu.alloc(output_bytes)?;

        for (ptr, poison) in [
            (d_generic, 0xa5),
            (d_fixed, 0x5a),
            (d_wide, 0xc3),
            (d_widest, 0x3c),
        ] {
            gpu.memset(ptr, poison, output_bytes)?;
        }
        launch(
            gpu, generic, d_generic, d_input, d_table, d_offsets, 5, n as u32, k as u32, 64, 128,
        )?;
        launch(
            gpu,
            fixed[shape],
            d_fixed,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            64,
            128,
        )?;
        launch(
            gpu,
            wide[shape],
            d_wide,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            128,
            256,
        )?;
        launch(
            gpu,
            widest[shape],
            d_widest,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            256,
            512,
        )?;
        gpu.synchronize(0)?;
        let mut baseline = vec![0u8; output_bytes];
        let mut fixed_out = vec![0u8; output_bytes];
        let mut wide_out = vec![0u8; output_bytes];
        let mut widest_out = vec![0u8; output_bytes];
        gpu.copy_d2h(d_generic, &mut baseline)?;
        gpu.copy_d2h(d_fixed, &mut fixed_out)?;
        gpu.copy_d2h(d_wide, &mut wide_out)?;
        gpu.copy_d2h(d_widest, &mut widest_out)?;
        let parity = baseline == fixed_out && baseline == wide_out && baseline == widest_out;
        let n128_eq_n256 = wide_out == widest_out;
        let nontrivial = baseline.iter().any(|&byte| byte != 0xa5);

        gpu.memset(d_wide, 0xd7, output_bytes)?;
        launch(
            gpu,
            wide[shape],
            d_wide,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            128,
            128,
        )?;
        gpu.synchronize(0)?;
        gpu.copy_d2h(d_wide, &mut wide_out)?;
        let wrong_block_unchanged = wide_out.iter().all(|&byte| byte == 0xd7);
        gpu.memset(d_widest, 0x7d, output_bytes)?;
        launch(
            gpu,
            widest[shape],
            d_widest,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            256,
            256,
        )?;
        gpu.synchronize(0)?;
        gpu.copy_d2h(d_widest, &mut widest_out)?;
        let wrong_n256_block_unchanged = widest_out.iter().all(|&byte| byte == 0x7d);
        pass &= parity && nontrivial && wrong_block_unchanged && wrong_n256_block_unchanged;

        let n64_ms = time_launch(
            gpu,
            fixed[shape],
            d_fixed,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            64,
            128,
        )?;
        let n128_ms = time_launch(
            gpu,
            wide[shape],
            d_wide,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            128,
            256,
        )?;
        let n256_ms = time_launch(
            gpu,
            widest[shape],
            d_widest,
            d_input,
            d_table,
            d_offsets,
            5,
            n as u32,
            k as u32,
            256,
            512,
        )?;
        eprintln!(
            "N={n} K={k}: n64 == wide {parity}, n128 == n256 {n128_eq_n256}, \
             nontrivial={nontrivial}, wrong-block={wrong_block_unchanged}, \
             wrong-N256-block={wrong_n256_block_unchanged}, N64={n64_ms:.3} ms \
             N128={n128_ms:.3} ms N256={n256_ms:.3} ms"
        );

        for ptr in d_experts
            .into_iter()
            .chain([d_input, d_table, d_generic, d_fixed, d_wide, d_widest])
        {
            let _ = gpu.free(ptr);
        }
    }
    let _ = gpu.free(d_offsets);
    if !pass {
        bail!("EXL3 N128/N256 prefill promotion gate failed");
    }
    Ok(())
}
