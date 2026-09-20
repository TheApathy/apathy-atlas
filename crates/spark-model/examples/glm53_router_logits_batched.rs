// SPDX-License-Identifier: AGPL-3.0-only

//! Physical exactness and timing gate for GLM-5.3 router token-reuse variants.

use std::time::Instant;

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops::{
    GgmlIqBuffer, Glm53RouterBuffers, Glm53RouterKernels, Glm53RouterPlan,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const HIDDEN: usize = 4_096;
const EXPERTS: usize = 288;
const TOP_K: usize = 8;
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

fn bf16_pattern(elements: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 2);
    for index in 0..elements {
        let value = (((index * 13 + 7) % 101) as f32 - 50.0) / 128.0;
        bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    bytes
}

fn f32_pattern(elements: usize, salt: usize, denominator: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 4);
    for index in 0..elements {
        let value = (((index * 17 + salt) % 127) as f32 - 63.0) / denominator;
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

fn download(gpu: &dyn GpuBackend, buffer: GgmlIqBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; buffer.bytes];
    gpu.copy_d2h(buffer.ptr, &mut bytes)?;
    Ok(bytes)
}

fn main() -> Result<()> {
    let selected =
        std::env::var("ATLAS_GLM53_ROUTER_LOGITS_TOKENS").unwrap_or_else(|_| "auto".to_owned());
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernels = Glm53RouterKernels::load(&gpu)?;
    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(&gpu);
    let input = allocations.upload(&bf16_pattern(MAX_ROWS * HIDDEN))?;
    let router = allocations.upload(&f32_pattern(EXPERTS * HIDDEN, 11, 512.0))?;
    let bias = allocations.upload(&f32_pattern(EXPERTS, 19, 4_096.0))?;
    let logits = allocations.zeroed(MAX_ROWS * EXPERTS * 4)?;
    let indices = allocations.zeroed(MAX_ROWS * TOP_K * 4)?;
    let weights = allocations.zeroed(MAX_ROWS * TOP_K * 4)?;
    let probs = allocations.zeroed(MAX_ROWS * EXPERTS * 4)?;
    let biased = allocations.zeroed(MAX_ROWS * EXPERTS * 4)?;

    for rows in ROWS {
        let plan = Glm53RouterPlan::new(rows as u32, 4096, 288, 8)?;
        let buffers = Glm53RouterBuffers {
            input_bf16: prefix(input, plan.input_bytes),
            router_f32: router,
            bias_f32: bias,
            logits_f32: prefix(logits, plan.logits_bytes),
            indices_u32: prefix(indices, plan.indices_bytes),
            weights_f32: prefix(weights, plan.weights_bytes),
            probs_f32: prefix(probs, plan.scores_bytes),
            biased_f32: prefix(biased, plan.scores_bytes),
        };
        kernels.launch(&gpu, plan, buffers, stream)?;
        gpu.synchronize(stream)?;
        let logits_bytes = download(&gpu, buffers.logits_f32)?;
        let indices_bytes = download(&gpu, buffers.indices_u32)?;
        let weights_bytes = download(&gpu, buffers.weights_f32)?;
        let probs_bytes = download(&gpu, buffers.probs_f32)?;
        let biased_bytes = download(&gpu, buffers.biased_f32)?;
        for raw in logits_bytes.chunks_exact(4) {
            ensure!(
                f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]).is_finite(),
                "non-finite router logit at rows={rows}"
            );
        }
        let route_hash = [indices_bytes, weights_bytes, probs_bytes, biased_bytes]
            .into_iter()
            .fold(0xcbf29ce484222325u64, |hash, bytes| {
                bytes.iter().fold(hash, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
                })
            });

        for _ in 0..2 {
            kernels.launch(&gpu, plan, buffers, stream)?;
            gpu.synchronize(stream)?;
        }
        let mut samples = Vec::with_capacity(7);
        for _ in 0..7 {
            let started = Instant::now();
            kernels.launch(&gpu, plan, buffers, stream)?;
            gpu.synchronize(stream)?;
            samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "ROUTER: PASS tokens_per_block={selected} rows={rows} median_ms={:.6} logits_fnv={:016x} route_fnv={route_hash:016x}",
            samples[samples.len() / 2],
            fnv1a64(&logits_bytes),
        );
    }
    Ok(())
}
