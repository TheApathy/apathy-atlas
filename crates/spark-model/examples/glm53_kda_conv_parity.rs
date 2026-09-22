// SPDX-License-Identifier: AGPL-3.0-only

//! Physical parity gate for Q=8 KDA convolution versus eight ordered Q=1 stages.

use anyhow::{Context, Result, ensure};
use half::bf16;
use spark_model::layers::ops::{
    GgmlIqBuffer, Glm53KdaConvBuffers, Glm53KdaConvKernel, Glm53KdaConvPlan,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const CHANNELS: usize = 8_192;
const QUERIES: usize = 8;
const STREAM_BYTES: usize = QUERIES * CHANNELS * 2;
const ONE_STREAM_BYTES: usize = CHANNELS * 2;
const STATE_BYTES: usize = 3 * CHANNELS * 4 * 4;

struct Allocations<'a> {
    gpu: &'a dyn GpuBackend,
    ptrs: Vec<DevicePtr>,
}

impl<'a> Allocations<'a> {
    fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self { gpu, ptrs: vec![] }
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<GgmlIqBuffer> {
        let ptr = self.gpu.alloc(bytes.len())?;
        self.ptrs.push(ptr);
        self.gpu.copy_h2d(bytes, ptr)?;
        Ok(GgmlIqBuffer {
            ptr,
            bytes: bytes.len(),
        })
    }

    fn zeroed(&mut self, bytes: usize) -> Result<GgmlIqBuffer> {
        let ptr = self.gpu.alloc(bytes)?;
        self.ptrs.push(ptr);
        self.gpu.memset(ptr, 0, bytes)?;
        Ok(GgmlIqBuffer { ptr, bytes })
    }
}

impl Drop for Allocations<'_> {
    fn drop(&mut self) {
        for ptr in self.ptrs.drain(..).rev() {
            let _ = self.gpu.free(ptr);
        }
    }
}

fn bf16_pattern(elements: usize, salt: usize) -> Vec<u8> {
    (0..elements)
        .flat_map(|i| {
            let value = (((i * 13 + salt) % 101) as f32 - 50.0) / 64.0;
            bf16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect()
}

fn f32_pattern(elements: usize, salt: usize) -> Vec<u8> {
    (0..elements)
        .flat_map(|i| {
            let value = (((i * 17 + salt) % 127) as f32 - 63.0) / 256.0;
            value.to_bits().to_le_bytes()
        })
        .collect()
}

fn run(gpu: &dyn GpuBackend, batched: bool) -> Result<Vec<u8>> {
    let kernel = Glm53KdaConvKernel::load(gpu)?;
    let mut a = Allocations::new(gpu);
    let q_input = a.upload(&bf16_pattern(QUERIES * CHANNELS, 1))?;
    let k_input = a.upload(&bf16_pattern(QUERIES * CHANNELS, 2))?;
    let v_input = a.upload(&bf16_pattern(QUERIES * CHANNELS, 3))?;
    let q_weight = a.upload(&f32_pattern(CHANNELS * 4, 4))?;
    let k_weight = a.upload(&f32_pattern(CHANNELS * 4, 5))?;
    let v_weight = a.upload(&f32_pattern(CHANNELS * 4, 6))?;
    let initial_state = f32_pattern(3 * CHANNELS * 4, 7);
    let persistent = a.upload(&initial_state)?;
    let staged = a.zeroed(STATE_BYTES)?;
    let q_output = a.zeroed(STREAM_BYTES)?;
    let k_output = a.zeroed(STREAM_BYTES)?;
    let v_output = a.zeroed(STREAM_BYTES)?;
    let published_ends = a.zeroed(4)?;
    let published_nonces = a.zeroed(8)?;
    let logical_lengths = a.zeroed(4)?;
    let stream = gpu.create_stream()?;

    let buffers = |row: usize, queries: usize| Glm53KdaConvBuffers {
        q_input_bf16: slice(q_input, row * ONE_STREAM_BYTES, queries * ONE_STREAM_BYTES),
        k_input_bf16: slice(k_input, row * ONE_STREAM_BYTES, queries * ONE_STREAM_BYTES),
        v_input_bf16: slice(v_input, row * ONE_STREAM_BYTES, queries * ONE_STREAM_BYTES),
        q_weight_f32: q_weight,
        k_weight_f32: k_weight,
        v_weight_f32: v_weight,
        persistent_state_f32: persistent,
        staged_state_f32: staged,
        q_output_bf16: slice(q_output, row * ONE_STREAM_BYTES, queries * ONE_STREAM_BYTES),
        k_output_bf16: slice(k_output, row * ONE_STREAM_BYTES, queries * ONE_STREAM_BYTES),
        v_output_bf16: slice(v_output, row * ONE_STREAM_BYTES, queries * ONE_STREAM_BYTES),
        published_ends_u32: published_ends,
        published_nonces_u64: published_nonces,
        logical_lengths_u32: logical_lengths,
    };

    if batched {
        let plan = Glm53KdaConvPlan::new(1, 8, 1024, 0, 8, 64, 128, 4, 99)?;
        kernel.launch_stage(gpu, plan, buffers(0, QUERIES), stream)?;
        gpu.copy_d2d_async(staged.ptr, persistent.ptr, STATE_BYTES, stream)?;
    } else {
        for row in 0..QUERIES {
            let plan = Glm53KdaConvPlan::new(
                1,
                1,
                1024,
                row as u32,
                row as u32 + 1,
                64,
                128,
                4,
                row as u64 + 1,
            )?;
            kernel.launch_stage(gpu, plan, buffers(row, 1), stream)?;
            gpu.copy_d2d_async(staged.ptr, persistent.ptr, STATE_BYTES, stream)?;
        }
    }
    gpu.synchronize(stream)?;

    let mut result = Vec::with_capacity(3 * STREAM_BYTES + 2 * STATE_BYTES);
    for buffer in [q_output, k_output, v_output, persistent, staged] {
        let mut bytes = vec![0u8; buffer.bytes];
        gpu.copy_d2h(buffer.ptr, &mut bytes)?;
        result.extend(bytes);
    }
    Ok(result)
}

fn slice(buffer: GgmlIqBuffer, offset: usize, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: buffer.ptr.offset(offset),
        bytes,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let serial = run(&backend, false).context("ordered Q1 convolution")?;
    let batched = run(&backend, true).context("Q8 convolution plus final carry copy")?;
    ensure!(
        batched == serial,
        "Q8 KDA convolution differs from eight ordered Q1 stages"
    );
    println!(
        "RESULT: PASS rows=8 compared_bytes={} fnv1a64={:016x}",
        batched.len(),
        fnv1a64(&batched)
    );
    Ok(())
}
