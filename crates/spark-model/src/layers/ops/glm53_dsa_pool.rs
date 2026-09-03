// SPDX-License-Identifier: AGPL-3.0-only

//! Exact online kpool4 compression for the GLM-5.3 DSA indexer.
//! Outputs are staging: partial acceptance must publish only a recomputed or
//! copied accepted prefix through the transactional cache metadata.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const INDEX_DIM: u32 = 128;
const KPOOL: u32 = 4;
/// Derived from the single definition in spark-runtime, never restated.
const TAIL_CAPACITY: u32 = spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY;
const MAX_TOKENS: u32 = 65_520;
const MAX_POSITIONS: u32 = 1_048_576;
const THREADS: u32 = INDEX_DIM;
const MAX_GRID_X: u64 = 2_147_483_647;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaPoolPlan {
    pub batch: u32,
    pub tokens: u32,
    pub start_position: u32,
    pub end_position: u32,
    pub initial_tail: u32,
    pub complete_pools: u32,
    pub final_tail: u32,
    pub blocks: u32,
    pub input_vector_bytes: usize,
    pub input_validity_bytes: usize,
    pub tail_vector_bytes: usize,
    pub tail_validity_bytes: usize,
    pub ape_bytes: usize,
    pub pool_vector_bytes: usize,
    pub pool_validity_bytes: usize,
}

impl Glm53DsaPoolPlan {
    pub fn new(
        batch: u32,
        tokens: u32,
        start_position: u32,
        index_dim: u32,
        kpool: u32,
        max_positions: u32,
    ) -> Result<Self> {
        if batch == 0 || tokens == 0 || tokens > MAX_TOKENS {
            bail!("GLM DSA pool compression requires batch>0 and T in 1..=65,520");
        }
        if index_dim != INDEX_DIM || kpool != KPOOL || max_positions != MAX_POSITIONS {
            bail!("GLM DSA pool compression requires exact dim128/kpool4/context1,048,576");
        }
        let end_position = start_position
            .checked_add(tokens)
            .context("GLM DSA pool logical-position overflow")?;
        if start_position >= MAX_POSITIONS || end_position > MAX_POSITIONS {
            bail!("GLM DSA pool compression exceeds absolute position 1,048,575");
        }
        let initial_tail = start_position % KPOOL;
        let combined = initial_tail
            .checked_add(tokens)
            .context("GLM DSA pool combined-row overflow")?;
        let complete_pools = combined / KPOOL;
        let final_tail = combined % KPOOL;
        let rows = u64::from(batch)
            .checked_mul(u64::from(tokens))
            .context("GLM DSA pool input-row overflow")?;
        let pool_rows = u64::from(batch)
            .checked_mul(u64::from(complete_pools))
            .context("GLM DSA pool output-row overflow")?;
        let tail_rows = u64::from(batch)
            .checked_mul(u64::from(TAIL_CAPACITY))
            .context("GLM DSA pool tail-row overflow")?;
        let blocks = u64::from(batch)
            .checked_mul(u64::from(complete_pools) + 1)
            .context("GLM DSA pool CUDA-grid overflow")?;
        if blocks > MAX_GRID_X {
            bail!("GLM DSA pool CUDA grid.x exceeds the supported limit");
        }
        Ok(Self {
            batch,
            tokens,
            start_position,
            end_position,
            initial_tail,
            complete_pools,
            final_tail,
            blocks: u32::try_from(blocks)?,
            input_vector_bytes: vector_bytes(rows)?,
            input_validity_bytes: usize::try_from(rows)?,
            tail_vector_bytes: vector_bytes(tail_rows)?,
            tail_validity_bytes: usize::try_from(tail_rows)?,
            ape_bytes: usize::try_from(u64::from(KPOOL) * u64::from(INDEX_DIM) * 4)?,
            pool_vector_bytes: vector_bytes(pool_rows)?,
            pool_validity_bytes: usize::try_from(pool_rows)?,
        })
    }

    pub fn validate(self) -> Result<()> {
        let rebuilt = Self::new(
            self.batch,
            self.tokens,
            self.start_position,
            INDEX_DIM,
            KPOOL,
            MAX_POSITIONS,
        )?;
        if rebuilt != self {
            bail!("forged GLM DSA pool plan");
        }
        Ok(())
    }
}

fn vector_bytes(rows: u64) -> Result<usize> {
    usize::try_from(
        rows.checked_mul(u64::from(INDEX_DIM))
            .and_then(|elements| elements.checked_mul(2))
            .context("GLM DSA pool vector-byte overflow")?,
    )
    .context("GLM DSA pool vector extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaPoolBuffers {
    pub input_keys_bf16: GgmlIqBuffer,
    pub input_gates_bf16: GgmlIqBuffer,
    pub input_validity_u8: GgmlIqBuffer,
    pub prior_tail_keys_bf16: GgmlIqBuffer,
    pub prior_tail_gates_bf16: GgmlIqBuffer,
    pub prior_tail_validity_u8: GgmlIqBuffer,
    pub ape_f32: GgmlIqBuffer,
    pub output_pool_keys_bf16: GgmlIqBuffer,
    pub output_pool_validity_u8: GgmlIqBuffer,
    pub output_tail_keys_bf16: GgmlIqBuffer,
    pub output_tail_gates_bf16: GgmlIqBuffer,
    pub output_tail_validity_u8: GgmlIqBuffer,
}

pub struct Glm53DsaPoolKernel {
    pool_k4: KernelHandle,
}

impl Glm53DsaPoolKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            pool_k4: gpu.kernel("glm53_dsa_pool", "atlas_glm53_dsa_pool_k4")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaPoolPlan,
        buffers: Glm53DsaPoolBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.pool_k4)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.input_keys_bf16.ptr)
            .arg_ptr(buffers.input_gates_bf16.ptr)
            .arg_ptr(buffers.input_validity_u8.ptr)
            .arg_ptr(buffers.prior_tail_keys_bf16.ptr)
            .arg_ptr(buffers.prior_tail_gates_bf16.ptr)
            .arg_ptr(buffers.prior_tail_validity_u8.ptr)
            .arg_ptr(buffers.ape_f32.ptr)
            .arg_ptr(buffers.output_pool_keys_bf16.ptr)
            .arg_ptr(buffers.output_pool_validity_u8.ptr)
            .arg_ptr(buffers.output_tail_keys_bf16.ptr)
            .arg_ptr(buffers.output_tail_gates_bf16.ptr)
            .arg_ptr(buffers.output_tail_validity_u8.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(plan.start_position)
            .arg_u32(plan.initial_tail)
            .arg_u32(plan.complete_pools)
            .arg_u32(plan.final_tail)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53DsaPoolPlan, buffers: Glm53DsaPoolBuffers) -> Result<()> {
    let named = [
        (
            "input keys",
            buffers.input_keys_bf16,
            plan.input_vector_bytes,
        ),
        (
            "input gates",
            buffers.input_gates_bf16,
            plan.input_vector_bytes,
        ),
        (
            "input validity",
            buffers.input_validity_u8,
            plan.input_validity_bytes,
        ),
        (
            "prior tail keys",
            buffers.prior_tail_keys_bf16,
            plan.tail_vector_bytes,
        ),
        (
            "prior tail gates",
            buffers.prior_tail_gates_bf16,
            plan.tail_vector_bytes,
        ),
        (
            "prior tail validity",
            buffers.prior_tail_validity_u8,
            plan.tail_validity_bytes,
        ),
        ("APE", buffers.ape_f32, plan.ape_bytes),
        (
            "output pool keys",
            buffers.output_pool_keys_bf16,
            plan.pool_vector_bytes,
        ),
        (
            "output pool validity",
            buffers.output_pool_validity_u8,
            plan.pool_validity_bytes,
        ),
        (
            "output tail keys",
            buffers.output_tail_keys_bf16,
            plan.tail_vector_bytes,
        ),
        (
            "output tail gates",
            buffers.output_tail_gates_bf16,
            plan.tail_vector_bytes,
        ),
        (
            "output tail validity",
            buffers.output_tail_validity_u8,
            plan.tail_validity_bytes,
        ),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.bytes != expected
            || (expected == 0 && buffer.ptr != DevicePtr::NULL)
            || (expected != 0 && buffer.ptr == DevicePtr::NULL)
        {
            bail!("GLM DSA pool {name} buffer is null or has the wrong extent");
        }
        if expected != 0 {
            let end = buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA pool {name} address overflow"))?;
            ranges.push((buffer.ptr.0, end));
        }
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DSA pool device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn rounded(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    fn accumulate_products(keys: [f32; 4], probabilities: [f32; 4]) -> f32 {
        let mut output = 0.0f32;
        for slot in 0..4 {
            let product = rounded(rounded(probabilities[slot]) * rounded(keys[slot]));
            output += product;
        }
        rounded(output)
    }

    fn reference(keys: [f32; 4], gates: [f32; 4], valid: [bool; 4], ape: [f32; 4]) -> (f32, bool) {
        let mut logits = [f32::NEG_INFINITY; 4];
        let mut maximum = f32::NEG_INFINITY;
        for slot in 0..4 {
            if valid[slot] {
                logits[slot] = rounded(gates[slot]) + ape[slot];
                maximum = maximum.max(logits[slot]);
            }
        }
        let mut weights = [0.0f32; 4];
        let mut denominator = 0.0f32;
        for slot in 0..4 {
            if valid[slot] {
                weights[slot] = (logits[slot] - maximum).exp();
                denominator += weights[slot];
            }
        }
        let mut probabilities = [0.0f32; 4];
        for slot in 0..4 {
            probabilities[slot] = if denominator == 0.0 {
                0.0
            } else {
                weights[slot] / denominator
            };
        }
        (
            accumulate_products(keys, probabilities),
            valid.into_iter().all(|entry| entry),
        )
    }

    #[test]
    fn plan_pins_tail_pool_boundaries_and_context_limit() {
        let crossing = Glm53DsaPoolPlan::new(2, 1, 3, 128, 4, 1_048_576).unwrap();
        assert_eq!(
            (
                crossing.initial_tail,
                crossing.complete_pools,
                crossing.final_tail
            ),
            (3, 1, 0)
        );
        assert_eq!(crossing.blocks, 4);
        assert_eq!(crossing.input_vector_bytes, 2 * 1 * 128 * 2);
        assert_eq!(crossing.pool_vector_bytes, 2 * 1 * 128 * 2);
        assert_eq!(crossing.tail_vector_bytes, 2 * 3 * 128 * 2);
        assert_eq!(crossing.ape_bytes, 4 * 128 * 4);
        assert!(Glm53DsaPoolPlan::new(1, 65_520, 983_056, 128, 4, 1_048_576).is_ok());
        assert!(Glm53DsaPoolPlan::new(1, 65_521, 0, 128, 4, 1_048_576).is_err());
        assert!(Glm53DsaPoolPlan::new(1, 1, 1_048_576, 128, 4, 1_048_576).is_err());
        assert!(Glm53DsaPoolPlan::new(1, 1, 0, 64, 4, 1_048_576).is_err());
        assert!(Glm53DsaPoolPlan::new(u32::MAX, 1, 0, 128, 4, 1_048_576).is_err());
    }

    #[test]
    fn reference_pins_invalid_nan_zero_and_bf16_chronology() {
        let empty = reference([1.0, 2.0, 4.0, 8.0], [0.0; 4], [false; 4], [0.0; 4]);
        assert_eq!(empty.0.to_bits(), 0.0f32.to_bits());
        assert!(!empty.1);
        let partial = reference(
            [1.0, 2.0, 4.0, 8.0],
            [0.0; 4],
            [true, false, true, false],
            [0.0; 4],
        );
        assert_eq!(partial.0.to_bits(), rounded(2.5).to_bits());
        assert!(!partial.1);
        let full = reference([1.0, 2.0, 4.0, 8.0], [0.0; 4], [true; 4], [0.0; 4]);
        assert!(full.1);
        assert_eq!(full.0.to_bits(), rounded(3.75).to_bits());

        let keys = [41.0, -15.8125, -46.75, -15.6875];
        let probabilities = [0.3711, 0.1807, 0.1748, 0.2734];
        let opmath = accumulate_products(keys, probabilities);
        let mut sequential_bf16 = 0.0f32;
        for slot in 0..4 {
            let product = rounded(rounded(probabilities[slot]) * rounded(keys[slot]));
            sequential_bf16 = rounded(sequential_bf16 + product);
        }
        assert_eq!(bf16::from_f32(opmath).to_bits() as i16, -16_880);
        assert_eq!(bf16::from_f32(sequential_bf16).to_bits() as i16, -16_864);
        assert_ne!(opmath.to_bits(), sequential_bf16.to_bits());
    }

    #[test]
    fn forged_or_aliased_buffers_fail_before_launch() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53DsaPoolKernel::load(&gpu).unwrap();
        let plan = Glm53DsaPoolPlan::new(1, 1, 0, 128, 4, 1_048_576).unwrap();
        let mut address = 0x10_0000u64;
        let mut next = |bytes: usize| {
            let buffer = GgmlIqBuffer {
                ptr: DevicePtr(address),
                bytes,
            };
            address += u64::try_from(bytes).unwrap() + 0x1000;
            buffer
        };
        let valid = Glm53DsaPoolBuffers {
            input_keys_bf16: next(plan.input_vector_bytes),
            input_gates_bf16: next(plan.input_vector_bytes),
            input_validity_u8: next(plan.input_validity_bytes),
            prior_tail_keys_bf16: next(plan.tail_vector_bytes),
            prior_tail_gates_bf16: next(plan.tail_vector_bytes),
            prior_tail_validity_u8: next(plan.tail_validity_bytes),
            ape_f32: next(plan.ape_bytes),
            output_pool_keys_bf16: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
            output_pool_validity_u8: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                bytes: 0,
            },
            output_tail_keys_bf16: next(plan.tail_vector_bytes),
            output_tail_gates_bf16: next(plan.tail_vector_bytes),
            output_tail_validity_u8: next(plan.tail_validity_bytes),
        };
        let mut forged = plan;
        forged.blocks += 1;
        assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53DsaPoolBuffers {
                        output_tail_keys_bf16: valid.prior_tail_keys_bf16,
                        ..valid
                    },
                    0
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
