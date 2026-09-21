// SPDX-License-Identifier: AGPL-3.0-only

//! GPU promotion gate for the exact DeepSeek-V4 dual gate/up H128 pre-rotation.
//!
//! Two legacy launches are the byte oracle. Short tails and production N=2410
//! exercise distinct expert sign tables, gathered token rows, output canaries,
//! exact launch guards, and ABBA CUDA-event timing after parity succeeds.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "exl3_fused_blend_microtest/timing.rs"]
mod timing;
use timing::{percentile, timed_sample};

const H: usize = 4096;
const TOPK: usize = 6;
const EXPERTS: usize = 256;
const TOKEN_CASES: [usize; 5] = [1, 17, 65, 256, 2410];

struct Guarded {
    base: DevicePtr,
    data: DevicePtr,
    len: usize,
}

impl Guarded {
    fn alloc(gpu: &dyn GpuBackend, len: usize, poison: u8) -> Result<Self> {
        let base = gpu.alloc(len + 8192)?;
        gpu.memset(base, 0xcd, len + 8192)?;
        let data = base.offset(4096);
        gpu.memset(data, poison, len)?;
        Ok(Self { base, data, len })
    }

    fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<Self> {
        let buffer = Self::alloc(gpu, bytes.len(), 0)?;
        gpu.copy_h2d(bytes, buffer.data)?;
        Ok(buffer)
    }

    fn canary_ok(&self, gpu: &dyn GpuBackend) -> Result<bool> {
        let mut before = vec![0u8; 4096];
        let mut after = vec![0u8; 4096];
        gpu.copy_d2h(self.base, &mut before)?;
        gpu.copy_d2h(self.data.offset(self.len), &mut after)?;
        Ok(before.iter().chain(&after).all(|&byte| byte == 0xcd))
    }
}

struct Kernels {
    single: KernelHandle,
    dual: KernelHandle,
}

fn bf16_input(tokens: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens * H * 2);
    for i in 0..tokens * H {
        let value = half::bf16::from_f32(((i * 17 % 257) as f32 - 128.0) / 61.0);
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn sign_data(salt: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(EXPERTS * H * 2);
    for expert in 0..EXPERTS {
        for column in 0..H {
            let bits = if (expert * 131 + column * 17 + salt).count_ones() & 1 == 0 {
                0x3c00u16
            } else {
                0xbc00u16
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
    bytes
}

fn table_bytes(data: DevicePtr) -> Vec<u8> {
    (0..EXPERTS)
        .flat_map(|expert| (data.0 + (expert * H * 2) as u64).to_le_bytes())
        .collect()
}

fn metadata(tokens: usize) -> (Vec<u8>, Vec<u8>) {
    let mut assignments = Vec::with_capacity(tokens * TOPK);
    for token in 0..tokens {
        for slot in 0..TOPK {
            let expert = (token * 17 + slot * 37) % EXPERTS;
            assignments.push((expert, token, slot));
        }
    }
    assignments.sort_unstable_by_key(|&(expert, token, slot)| (expert, token, slot));
    let experts = assignments
        .iter()
        .flat_map(|&(expert, _, _)| (expert as i32).to_le_bytes())
        .collect();
    let token_ids = assignments
        .iter()
        .flat_map(|&(_, token, _)| (token as i32).to_le_bytes())
        .collect();
    (experts, token_ids)
}

#[allow(clippy::too_many_arguments)]
fn launch_single(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    token_ids: DevicePtr,
    expert_ids: DevicePtr,
    signs: DevicePtr,
    output: DevicePtr,
    rows: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 4, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(token_ids)
        .arg_ptr(expert_ids)
        .arg_ptr(signs)
        .arg_ptr(output)
        .arg_u32(H as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_dual_geometry(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    token_ids: DevicePtr,
    expert_ids: DevicePtr,
    gate_signs: DevicePtr,
    up_signs: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    k: u32,
    rows: u32,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
        .arg_ptr(input)
        .arg_ptr(token_ids)
        .arg_ptr(expert_ids)
        .arg_ptr(gate_signs)
        .arg_ptr(up_signs)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_u32(k)
        .arg_u32(rows)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    token_ids: DevicePtr,
    expert_ids: DevicePtr,
    gate_signs: DevicePtr,
    up_signs: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    rows: u32,
) -> Result<()> {
    launch_dual_geometry(
        gpu,
        kernel,
        input,
        token_ids,
        expert_ids,
        gate_signs,
        up_signs,
        gate_out,
        up_out,
        H as u32,
        rows,
        [rows, 4, 1],
        [256, 1, 1],
    )
}

#[allow(clippy::too_many_arguments)]
fn run_chain(
    gpu: &dyn GpuBackend,
    kernels: &Kernels,
    dual: bool,
    input: DevicePtr,
    token_ids: DevicePtr,
    expert_ids: DevicePtr,
    gate_signs: DevicePtr,
    up_signs: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    rows: u32,
) -> Result<()> {
    if dual {
        launch_dual(
            gpu,
            kernels.dual,
            input,
            token_ids,
            expert_ids,
            gate_signs,
            up_signs,
            gate_out,
            up_out,
            rows,
        )
    } else {
        launch_single(
            gpu,
            kernels.single,
            input,
            token_ids,
            expert_ids,
            gate_signs,
            gate_out,
            rows,
        )?;
        launch_single(
            gpu,
            kernels.single,
            input,
            token_ids,
            expert_ids,
            up_signs,
            up_out,
            rows,
        )
    }
}

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut output = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn timing_report(
    gpu: &dyn GpuBackend,
    kernels: &Kernels,
    input: DevicePtr,
    token_ids: DevicePtr,
    expert_ids: DevicePtr,
    gate_signs: DevicePtr,
    up_signs: DevicePtr,
    legacy_gate: DevicePtr,
    legacy_up: DevicePtr,
    dual_gate: DevicePtr,
    dual_up: DevicePtr,
) -> Result<()> {
    let rows = (2410 * TOPK) as u32;
    let run = |dual: bool| {
        run_chain(
            gpu,
            kernels,
            dual,
            input,
            token_ids,
            expert_ids,
            gate_signs,
            up_signs,
            if dual { dual_gate } else { legacy_gate },
            if dual { dual_up } else { legacy_up },
            rows,
        )
    };
    for _ in 0..20 {
        run(false)?;
        run(true)?;
    }
    gpu.synchronize(0)?;
    let (mut legacy_us, mut dual_us) = (Vec::with_capacity(31), Vec::with_capacity(31));
    for dual in [false, true, true, false].into_iter().cycle() {
        if legacy_us.len() == 31 && dual_us.len() == 31 {
            break;
        }
        if (dual && dual_us.len() == 31) || (!dual && legacy_us.len() == 31) {
            continue;
        }
        let sample = timed_sample(|| run(dual))?;
        if dual {
            dual_us.push(sample);
        } else {
            legacy_us.push(sample);
        }
    }
    for (name, samples) in [
        ("legacy pre+pre", legacy_us.as_mut_slice()),
        ("dual pre", dual_us.as_mut_slice()),
    ] {
        let p10 = percentile(samples, 1, 10);
        let median = percentile(samples, 1, 2);
        let p90 = percentile(samples, 9, 10);
        eprintln!("{name}: p10={p10:.3} us median={median:.3} us p90={p90:.3} us");
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let kernels = Kernels {
        single: gpu.kernel("exl3_gemv", "exl3_h128_pre_rows")?,
        dual: gpu.kernel("exl3_gemv_k2", "exl3_h128_pre_dual_rows_h4096")?,
    };

    for tokens in TOKEN_CASES {
        let rows = tokens * TOPK;
        let bytes = rows * H * 2;
        let input = Guarded::upload(gpu, &bf16_input(tokens))?;
        let (expert_bytes, token_bytes) = metadata(tokens);
        let expert_ids = Guarded::upload(gpu, &expert_bytes)?;
        let token_ids = Guarded::upload(gpu, &token_bytes)?;
        let gate_sign_data = Guarded::upload(gpu, &sign_data(0x51))?;
        let up_sign_data = Guarded::upload(gpu, &sign_data(0xa7))?;
        let gate_signs = Guarded::upload(gpu, &table_bytes(gate_sign_data.data))?;
        let up_signs = Guarded::upload(gpu, &table_bytes(up_sign_data.data))?;
        let legacy_gate = Guarded::alloc(gpu, bytes, 0xa1)?;
        let legacy_up = Guarded::alloc(gpu, bytes, 0xb2)?;
        let dual_gate = Guarded::alloc(gpu, bytes, 0xc3)?;
        let dual_up = Guarded::alloc(gpu, bytes, 0xd4)?;

        run_chain(
            gpu,
            &kernels,
            false,
            input.data,
            token_ids.data,
            expert_ids.data,
            gate_signs.data,
            up_signs.data,
            legacy_gate.data,
            legacy_up.data,
            rows as u32,
        )?;
        run_chain(
            gpu,
            &kernels,
            true,
            input.data,
            token_ids.data,
            expert_ids.data,
            gate_signs.data,
            up_signs.data,
            dual_gate.data,
            dual_up.data,
            rows as u32,
        )?;
        gpu.synchronize(0)?;
        let expected_gate = download(gpu, legacy_gate.data, bytes)?;
        let expected_up = download(gpu, legacy_up.data, bytes)?;
        if expected_gate != download(gpu, dual_gate.data, bytes)?
            || expected_up != download(gpu, dual_up.data, bytes)?
            || expected_gate == expected_up
        {
            bail!("dual pre byte parity failed at N={tokens}");
        }

        if tokens == 17 {
            let identity_rows = 7u32;
            let identity_bytes = identity_rows as usize * H * 2;
            gpu.memset(legacy_gate.data, 0xa1, bytes)?;
            gpu.memset(legacy_up.data, 0xb2, bytes)?;
            gpu.memset(dual_gate.data, 0xc3, bytes)?;
            gpu.memset(dual_up.data, 0xd4, bytes)?;
            run_chain(
                gpu,
                &kernels,
                false,
                input.data,
                DevicePtr(0),
                expert_ids.data,
                gate_signs.data,
                up_signs.data,
                legacy_gate.data,
                legacy_up.data,
                identity_rows,
            )?;
            run_chain(
                gpu,
                &kernels,
                true,
                input.data,
                DevicePtr(0),
                expert_ids.data,
                gate_signs.data,
                up_signs.data,
                dual_gate.data,
                dual_up.data,
                identity_rows,
            )?;
            gpu.synchronize(0)?;
            if download(gpu, legacy_gate.data, identity_bytes)?
                != download(gpu, dual_gate.data, identity_bytes)?
                || download(gpu, legacy_up.data, identity_bytes)?
                    != download(gpu, dual_up.data, identity_bytes)?
            {
                bail!("dual pre identity-gather byte parity failed");
            }
        }

        if tokens == 2410 {
            let invalid = [
                (4095, rows as u32, [rows as u32, 4, 1], [256, 1, 1]),
                (H as u32, rows as u32 + 1, [rows as u32, 4, 1], [256, 1, 1]),
                (H as u32, rows as u32, [rows as u32, 3, 1], [256, 1, 1]),
                (H as u32, rows as u32, [rows as u32, 4, 2], [256, 1, 1]),
                (H as u32, rows as u32, [rows as u32, 4, 1], [128, 1, 1]),
                (H as u32, rows as u32, [rows as u32, 4, 1], [256, 2, 1]),
                (H as u32, rows as u32, [rows as u32, 4, 1], [256, 1, 2]),
            ];
            for (k, arg_rows, grid, block) in invalid {
                gpu.memset(dual_gate.data, 0xe5, bytes)?;
                gpu.memset(dual_up.data, 0xe5, bytes)?;
                launch_dual_geometry(
                    gpu,
                    kernels.dual,
                    input.data,
                    token_ids.data,
                    expert_ids.data,
                    gate_signs.data,
                    up_signs.data,
                    dual_gate.data,
                    dual_up.data,
                    k,
                    arg_rows,
                    grid,
                    block,
                )?;
                gpu.synchronize(0)?;
                if download(gpu, dual_gate.data, bytes)?
                    .iter()
                    .any(|&v| v != 0xe5)
                    || download(gpu, dual_up.data, bytes)?
                        .iter()
                        .any(|&v| v != 0xe5)
                {
                    bail!("dual pre did not fail closed: k={k} grid={grid:?} block={block:?}");
                }
            }
            timing_report(
                gpu,
                &kernels,
                input.data,
                token_ids.data,
                expert_ids.data,
                gate_signs.data,
                up_signs.data,
                legacy_gate.data,
                legacy_up.data,
                dual_gate.data,
                dual_up.data,
            )?;
        }

        for output in [&legacy_gate, &legacy_up, &dual_gate, &dual_up] {
            if !output.canary_ok(gpu)? {
                bail!("dual pre output canary changed at N={tokens}");
            }
        }
        eprintln!("N={tokens}: gate/up byte parity passed");
        for buffer in [
            input,
            expert_ids,
            token_ids,
            gate_sign_data,
            up_sign_data,
            gate_signs,
            up_signs,
            legacy_gate,
            legacy_up,
            dual_gate,
            dual_up,
        ] {
            gpu.free(buffer.base)?;
        }
    }
    Ok(())
}
