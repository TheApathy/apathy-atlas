// SPDX-License-Identifier: AGPL-3.0-only

//! Physical exactness and timing gate for GLM-5.3 KDA register-resident variants.

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use half::bf16;
use spark_model::layers::ops::{
    GgmlIqBuffer, Glm53KdaPrefillBuffers, Glm53KdaPrefillKernel, Glm53KdaPrefillPlan,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const HEADS: usize = 64;
const DIM: usize = 128;
const MAX_ROWS: usize = 1_875;
const ROWS: [usize; 4] = [128, 512, 1_024, MAX_ROWS];

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

fn bf16_pattern(elements: usize, salt: usize, denominator: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 2);
    for index in 0..elements {
        let value = (((index * 13 + salt) % 101) as f32 - 50.0) / denominator;
        bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    bytes
}

fn decay_pattern(elements: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 4);
    for index in 0..elements {
        let value = -0.000_5 * (1 + index % 7) as f32;
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn state_pattern(elements: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 4);
    for index in 0..elements {
        let value = (((index * 17 + 5) % 127) as f32 - 63.0) / 4_096.0;
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn prefix(buffer: GgmlIqBuffer, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: buffer.ptr,
        bytes,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn ensure_finite_bf16(bytes: &[u8]) -> Result<()> {
    ensure!(bytes.len().is_multiple_of(2), "odd BF16 byte count");
    for raw in bytes.chunks_exact(2) {
        let value = bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
        ensure!(value.is_finite(), "non-finite BF16 output");
    }
    Ok(())
}

fn ensure_finite_f32(bytes: &[u8]) -> Result<()> {
    ensure!(bytes.len().is_multiple_of(4), "unaligned F32 byte count");
    for raw in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        ensure!(value.is_finite(), "non-finite F32 state");
    }
    Ok(())
}

fn main() -> Result<()> {
    let selected = match std::env::var("ATLAS_GLM53_KDA_RR_COLUMNS").as_deref() {
        Err(_) | Ok("1") => 8,
        Ok("0") => 1,
        Ok(value) => anyhow::bail!(
            "ATLAS_GLM53_KDA_RR_COLUMNS must be absent or exactly 0 or 1; got {value:?}"
        ),
    };
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernel = Glm53KdaPrefillKernel::load(&gpu)?;
    let stream = gpu.create_stream()?;
    let max_vectors = MAX_ROWS * HEADS * DIM;
    let token_groups = MAX_ROWS * HEADS;
    let state_elements = HEADS * DIM * DIM;
    let initial_state = state_pattern(state_elements);
    let mut allocations = Allocations::new(&gpu);
    let state = allocations.upload(&initial_state)?;
    let query = allocations.upload(&bf16_pattern(max_vectors, 1, 64.0))?;
    let key = allocations.upload(&bf16_pattern(max_vectors, 2, 64.0))?;
    let value = allocations.upload(&bf16_pattern(max_vectors, 3, 128.0))?;
    let decay = allocations.upload(&decay_pattern(max_vectors))?;
    let beta = allocations.upload(&bf16_pattern(token_groups, 54, 512.0))?;
    let output = allocations.zeroed(max_vectors * 2)?;

    for rows in ROWS {
        let plan = Glm53KdaPrefillPlan::new(1, rows as u32, 64, 128, 128)?;
        let buffers = Glm53KdaPrefillBuffers {
            state_f32: state,
            query_bf16: prefix(query, plan.vector_bytes),
            key_bf16: prefix(key, plan.vector_bytes),
            value_bf16: prefix(value, plan.vector_bytes),
            log_decay_f32: prefix(decay, plan.decay_bytes),
            beta_bf16: prefix(beta, plan.beta_bytes),
            output_bf16: prefix(output, plan.vector_bytes),
        };

        gpu.copy_h2d(&initial_state, state.ptr)?;
        gpu.memset(output.ptr, 0, plan.vector_bytes)?;
        kernel.launch_register_resident(&gpu, plan, buffers, stream)?;
        gpu.synchronize(stream)?;
        let mut output_bytes = vec![0u8; plan.vector_bytes];
        let mut state_bytes = vec![0u8; plan.state_bytes];
        gpu.copy_d2h(output.ptr, &mut output_bytes)?;
        gpu.copy_d2h(state.ptr, &mut state_bytes)?;
        ensure_finite_bf16(&output_bytes).with_context(|| format!("rows={rows} output"))?;
        ensure_finite_f32(&state_bytes).with_context(|| format!("rows={rows} state"))?;
        let output_hash = fnv1a64(&output_bytes);
        let state_hash = fnv1a64(&state_bytes);

        for _ in 0..2 {
            gpu.copy_h2d(&initial_state, state.ptr)?;
            kernel.launch_register_resident(&gpu, plan, buffers, stream)?;
            gpu.synchronize(stream)?;
        }
        let mut samples = Vec::with_capacity(7);
        for _ in 0..7 {
            gpu.copy_h2d(&initial_state, state.ptr)?;
            gpu.synchronize(stream)?;
            let started = Instant::now();
            kernel.launch_register_resident(&gpu, plan, buffers, stream)?;
            gpu.synchronize(stream)?;
            samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "KDA_RR: PASS columns={selected} rows={rows} median_ms={:.6} output_fnv={output_hash:016x} state_fnv={state_hash:016x}",
            samples[samples.len() / 2]
        );
    }
    Ok(())
}
