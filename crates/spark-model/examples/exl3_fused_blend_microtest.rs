// SPDX-License-Identifier: AGPL-3.0-only

//! GPU promotion gate for the exact DeepSeek-V4 H4096 EXL3 routed/shared tail.
//!
//! The legacy two-launch chain is the byte oracle. Production N=2410 plus short
//! tails exercise the one-CTA-per-token fused entry, NULL-gate behavior,
//! output canaries, and fail-closed geometry. Timing runs only after parity.

use anyhow::{Result, bail};
use half::bf16;
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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

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
    post: KernelHandle,
    blend: KernelHandle,
    fused: KernelHandle,
}

fn bf16_data(count: usize, rng: &mut Rng, scale: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 2);
    for _ in 0..count {
        let bits = bf16::from_f32((rng.unit() - 0.5) * scale).to_bits();
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn metadata(tokens: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let rows = tokens * TOPK;
    let mut assignments = Vec::with_capacity(rows);
    for token in 0..tokens {
        for k in 0..TOPK {
            let expert = (token * 17 + k * 37) % EXPERTS;
            assignments.push((expert, token * TOPK + k));
        }
    }
    assignments.sort_unstable_by_key(|&(expert, slot)| (expert, slot));

    let mut token_to_perm = vec![0i32; rows];
    let mut sorted_experts = Vec::with_capacity(rows);
    for (row, &(expert, slot)) in assignments.iter().enumerate() {
        token_to_perm[slot] = row as i32;
        sorted_experts.push(expert as i32);
    }
    let route_weights = [0.375f32, 0.3, 0.2625, 0.225, 0.1875, 0.15];
    let weights = (0..tokens).flat_map(|_| route_weights).collect::<Vec<_>>();
    (token_to_perm, sorted_experts, weights)
}

#[allow(clippy::too_many_arguments)]
fn launch_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw: DevicePtr,
    out: DevicePtr,
    token_to_perm: DevicePtr,
    weights: DevicePtr,
    sorted_experts: DevicePtr,
    svh_table: DevicePtr,
    tokens: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens, 4, 1])
        .block([256, 1, 1])
        .arg_ptr(raw)
        .arg_ptr(out)
        .arg_ptr(token_to_perm)
        .arg_ptr(weights)
        .arg_ptr(sorted_experts)
        .arg_ptr(svh_table)
        .arg_u32(H as u32)
        .arg_u32(tokens)
        .arg_u32(TOPK as u32)
        .launch(0)
}

fn launch_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    out: DevicePtr,
    shared: DevicePtr,
    normed: DevicePtr,
    gate: DevicePtr,
    tokens: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(out)
        .arg_ptr(shared)
        .arg_ptr(normed)
        .arg_ptr(gate)
        .arg_u32(H as u32)
        .arg_u32(tokens)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_fused_geometry(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw: DevicePtr,
    out: DevicePtr,
    token_to_perm: DevicePtr,
    weights: DevicePtr,
    sorted_experts: DevicePtr,
    svh_table: DevicePtr,
    shared: DevicePtr,
    normed: DevicePtr,
    gate: DevicePtr,
    h: u32,
    tokens: u32,
    topk: u32,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
        .arg_ptr(raw)
        .arg_ptr(out)
        .arg_ptr(token_to_perm)
        .arg_ptr(weights)
        .arg_ptr(sorted_experts)
        .arg_ptr(svh_table)
        .arg_ptr(shared)
        .arg_ptr(normed)
        .arg_ptr(gate)
        .arg_u32(h)
        .arg_u32(tokens)
        .arg_u32(topk)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_fused(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw: DevicePtr,
    out: DevicePtr,
    token_to_perm: DevicePtr,
    weights: DevicePtr,
    sorted_experts: DevicePtr,
    svh_table: DevicePtr,
    shared: DevicePtr,
    normed: DevicePtr,
    gate: DevicePtr,
    tokens: u32,
) -> Result<()> {
    launch_fused_geometry(
        gpu,
        kernel,
        raw,
        out,
        token_to_perm,
        weights,
        sorted_experts,
        svh_table,
        shared,
        normed,
        gate,
        H as u32,
        tokens,
        TOPK as u32,
        [tokens, 1, 1],
        [256, 1, 1],
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_chain(
    gpu: &dyn GpuBackend,
    kernels: &Kernels,
    fused: bool,
    raw: DevicePtr,
    out: DevicePtr,
    token_to_perm: DevicePtr,
    weights: DevicePtr,
    sorted_experts: DevicePtr,
    svh_table: DevicePtr,
    shared: DevicePtr,
    normed: DevicePtr,
    gate: DevicePtr,
    tokens: u32,
) -> Result<()> {
    if fused {
        launch_fused(
            gpu,
            kernels.fused,
            raw,
            out,
            token_to_perm,
            weights,
            sorted_experts,
            svh_table,
            shared,
            normed,
            gate,
            tokens,
        )
    } else {
        launch_post(
            gpu,
            kernels.post,
            raw,
            out,
            token_to_perm,
            weights,
            sorted_experts,
            svh_table,
            tokens,
        )?;
        launch_blend(gpu, kernels.blend, out, shared, normed, gate, tokens)
    }
}

#[allow(clippy::too_many_arguments)]
fn timing_report(
    gpu: &dyn GpuBackend,
    kernels: &Kernels,
    raw: DevicePtr,
    legacy: DevicePtr,
    fused_out: DevicePtr,
    token_to_perm: DevicePtr,
    weights: DevicePtr,
    sorted_experts: DevicePtr,
    svh_table: DevicePtr,
    shared: DevicePtr,
    normed: DevicePtr,
    gate: DevicePtr,
) -> Result<()> {
    let tokens = 2410u32;
    let run = |is_fused: bool| {
        launch_chain(
            gpu,
            kernels,
            is_fused,
            raw,
            if is_fused { fused_out } else { legacy },
            token_to_perm,
            weights,
            sorted_experts,
            svh_table,
            shared,
            normed,
            gate,
            tokens,
        )
    };
    for _ in 0..20 {
        run(false)?;
        run(true)?;
    }
    gpu.synchronize(0)?;

    let (mut legacy_us, mut fused_us) = (Vec::with_capacity(31), Vec::with_capacity(31));
    let order = [false, true, true, false];
    while legacy_us.len() < 31 || fused_us.len() < 31 {
        for &is_fused in &order {
            if (is_fused && fused_us.len() == 31) || (!is_fused && legacy_us.len() == 31) {
                continue;
            }
            let sample = timed_sample(|| run(is_fused))?;
            if is_fused {
                fused_us.push(sample);
            } else {
                legacy_us.push(sample);
            }
        }
    }
    let report = |name: &str, values: &mut [f32]| {
        let p10 = percentile(values, 1, 10);
        let median = percentile(values, 1, 2);
        let p90 = percentile(values, 9, 10);
        eprintln!("{name}: p10={p10:.3} us median={median:.3} us p90={p90:.3} us");
    };
    report("legacy post+blend", &mut legacy_us);
    report("fused post+blend", &mut fused_us);
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let kernels = Kernels {
        post: gpu.kernel("exl3_gemv_k2", "exl3_h128_post_unpermute_rows_h4096")?,
        blend: gpu.kernel("moe", "moe_batched_blend")?,
        fused: gpu.kernel("exl3_gemv_k2", "exl3_h128_post_unpermute_blend_h4096")?,
    };

    for tokens in TOKEN_CASES {
        let rows = tokens * TOPK;
        let mut rng = Rng(0x5441_494c_2026_0827 ^ tokens as u64);
        let raw = Guarded::upload(gpu, &bf16_data(rows * H, &mut rng, 1.5))?;
        let shared = Guarded::upload(gpu, &bf16_data(tokens * H, &mut rng, 0.75))?;
        let normed = Guarded::upload(gpu, &bf16_data(tokens * H, &mut rng, 0.0625))?;
        let gate = Guarded::upload(gpu, &bf16_data(H, &mut rng, 0.0625))?;
        let (token_to_perm, sorted_experts, weights) = metadata(tokens);
        let t2p = Guarded::upload(gpu, &i32_bytes(&token_to_perm))?;
        let sorted = Guarded::upload(gpu, &i32_bytes(&sorted_experts))?;
        let weights = Guarded::upload(gpu, &f32_bytes(&weights))?;
        let mut svh_bytes = Vec::with_capacity(EXPERTS * H * 2);
        for _ in 0..EXPERTS * H {
            let bits = if rng.next() & 1 == 0 {
                0x3c00u16
            } else {
                0xbc00u16
            };
            svh_bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let svh = Guarded::upload(gpu, &svh_bytes)?;
        let table_bytes = (0..EXPERTS)
            .flat_map(|expert| (svh.data.0 + (expert * H * 2) as u64).to_le_bytes())
            .collect::<Vec<_>>();
        let svh_table = Guarded::upload(gpu, &table_bytes)?;
        let bytes = tokens * H * 2;
        let legacy = Guarded::alloc(gpu, bytes, 0xa5)?;
        let fused_out = Guarded::alloc(gpu, bytes, 0x5a)?;

        for gate_ptr in [gate.data, DevicePtr(0)] {
            gpu.memset(legacy.data, 0xa5, bytes)?;
            gpu.memset(fused_out.data, 0x5a, bytes)?;
            launch_chain(
                gpu,
                &kernels,
                false,
                raw.data,
                legacy.data,
                t2p.data,
                weights.data,
                sorted.data,
                svh_table.data,
                shared.data,
                normed.data,
                gate_ptr,
                tokens as u32,
            )?;
            launch_chain(
                gpu,
                &kernels,
                true,
                raw.data,
                fused_out.data,
                t2p.data,
                weights.data,
                sorted.data,
                svh_table.data,
                shared.data,
                normed.data,
                gate_ptr,
                tokens as u32,
            )?;
            gpu.synchronize(0)?;
            let mut expected = vec![0u8; bytes];
            let mut actual = vec![0u8; bytes];
            gpu.copy_d2h(legacy.data, &mut expected)?;
            gpu.copy_d2h(fused_out.data, &mut actual)?;
            if expected != actual || expected.iter().all(|&byte| byte == 0xa5) {
                bail!(
                    "tail parity failed at N={tokens}, null_gate={}",
                    gate_ptr.is_null()
                );
            }
        }

        if tokens == 2410 {
            let invalid = [
                (4095, TOPK as u32, [tokens as u32, 1, 1], [256, 1, 1]),
                (H as u32, 5, [tokens as u32, 1, 1], [256, 1, 1]),
                (H as u32, TOPK as u32, [tokens as u32, 2, 1], [256, 1, 1]),
                (H as u32, TOPK as u32, [tokens as u32, 1, 1], [128, 1, 1]),
            ];
            for (h, topk, grid, block) in invalid {
                gpu.memset(fused_out.data, 0xd7, bytes)?;
                launch_fused_geometry(
                    gpu,
                    kernels.fused,
                    raw.data,
                    fused_out.data,
                    t2p.data,
                    weights.data,
                    sorted.data,
                    svh_table.data,
                    shared.data,
                    normed.data,
                    gate.data,
                    h,
                    tokens as u32,
                    topk,
                    grid,
                    block,
                )?;
                gpu.synchronize(0)?;
                let mut actual = vec![0u8; bytes];
                gpu.copy_d2h(fused_out.data, &mut actual)?;
                if actual.iter().any(|&byte| byte != 0xd7) {
                    bail!(
                        "fused tail did not fail closed: h={h} topk={topk} grid={grid:?} block={block:?}"
                    );
                }
            }
            timing_report(
                gpu,
                &kernels,
                raw.data,
                legacy.data,
                fused_out.data,
                t2p.data,
                weights.data,
                sorted.data,
                svh_table.data,
                shared.data,
                normed.data,
                gate.data,
            )?;
        }

        if !legacy.canary_ok(gpu)? || !fused_out.canary_ok(gpu)? {
            bail!("output canary changed at N={tokens}");
        }
        eprintln!("N={tokens}: byte parity passed (gated and NULL-gate)");
        for buffer in [
            raw, shared, normed, gate, t2p, sorted, weights, svh, svh_table, legacy, fused_out,
        ] {
            gpu.free(buffer.base)?;
        }
    }
    Ok(())
}
