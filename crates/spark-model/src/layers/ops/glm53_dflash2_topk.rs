// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const VOCAB_SIZE: u32 = 154_880;
const TOP_K: u32 = 16;
const MAX_BLOCK_TOKENS: u32 = 8;
const MAX_GRID_X: u32 = i32::MAX as u32;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2TopkPlan {
    pub batch: u32,
    pub tokens: u32,
    pub rows: u32,
    pub logits_bytes: usize,
    pub candidate_bytes: usize,
    pub unary_bytes: usize,
    pub status_bytes: usize,
}

impl Glm53Dflash2TopkPlan {
    pub fn new(batch: u32, tokens: u32, vocab_size: u32, top_k: u32) -> Result<Self> {
        if batch == 0 || tokens == 0 || tokens > MAX_BLOCK_TOKENS {
            bail!("GLM DFlash2 top-k requires batch>0 and 1..=8 tokens");
        }
        if vocab_size != VOCAB_SIZE || top_k != TOP_K {
            bail!("GLM DFlash2 top-k requires exact V154880/top16 geometry");
        }
        let rows = batch
            .checked_mul(tokens)
            .context("GLM DFlash2 top-k row overflow")?;
        if rows > MAX_GRID_X {
            bail!("GLM DFlash2 top-k CUDA grid overflow");
        }
        let bytes = |columns: u32, width: usize, label: &str| -> Result<usize> {
            usize::try_from(rows)?
                .checked_mul(usize::try_from(columns)?)
                .and_then(|count| count.checked_mul(width))
                .with_context(|| format!("GLM DFlash2 top-k {label} byte overflow"))
        };
        Ok(Self {
            batch,
            tokens,
            rows,
            logits_bytes: bytes(vocab_size, 2, "logits")?,
            candidate_bytes: bytes(top_k, 4, "candidate")?,
            unary_bytes: bytes(top_k, 4, "unary")?,
            status_bytes: bytes(1, 4, "status")?,
        })
    }

    fn validate(self) -> Result<()> {
        let expected = Self::new(self.batch, self.tokens, VOCAB_SIZE, TOP_K)?;
        if self != expected {
            bail!("GLM DFlash2 top-k plan was modified after admission");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2TopkBuffers {
    pub logits_bf16: GgmlIqBuffer,
    pub candidates_u32: GgmlIqBuffer,
    /// Exact selected BF16 values widened to F32.
    pub unary_f32: GgmlIqBuffer,
    /// Per-row zero on success and one when the input contains non-finite data.
    pub status_u32: GgmlIqBuffer,
}

pub struct Glm53Dflash2TopkKernel {
    topk: KernelHandle,
}

impl Glm53Dflash2TopkKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            topk: gpu.kernel("glm53_dflash2_topk", "atlas_glm53_dflash2_topk_bf16")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Dflash2TopkPlan,
        buffers: Glm53Dflash2TopkBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.topk)
            .grid([plan.rows, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.logits_bf16.ptr)
            .arg_ptr(buffers.candidates_u32.ptr)
            .arg_ptr(buffers.unary_f32.ptr)
            .arg_ptr(buffers.status_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(VOCAB_SIZE)
            .arg_u32(TOP_K)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53Dflash2TopkPlan, buffers: Glm53Dflash2TopkBuffers) -> Result<()> {
    let named = [
        ("logits", buffers.logits_bf16, plan.logits_bytes),
        ("candidates", buffers.candidates_u32, plan.candidate_bytes),
        ("unary", buffers.unary_f32, plan.unary_bytes),
        ("status", buffers.status_u32, plan.status_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != expected {
            bail!("GLM DFlash2 top-k {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM DFlash2 top-k {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DFlash2 top-k device buffers overlap");
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

    fn reference_topk(logits: &[f32], top_k: usize) -> Option<Vec<(u32, f32)>> {
        let values = logits.iter().copied().map(rounded).collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut remaining = values.into_iter().enumerate().collect::<Vec<_>>();
        remaining.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        Some(
            remaining
                .into_iter()
                .take(top_k)
                .map(|(token, value)| (token as u32, value))
                .collect(),
        )
    }

    #[test]
    fn exact_geometry_and_extents_are_pinned() {
        let plan = Glm53Dflash2TopkPlan::new(2, 8, 154_880, 16).unwrap();
        assert_eq!(plan.rows, 16);
        assert_eq!(plan.logits_bytes, 2 * 8 * 154_880 * 2);
        assert_eq!(plan.candidate_bytes, 2 * 8 * 16 * 4);
        assert_eq!(plan.unary_bytes, 2 * 8 * 16 * 4);
        assert_eq!(plan.status_bytes, 2 * 8 * 4);
        assert!(Glm53Dflash2TopkPlan::new(0, 8, 154_880, 16).is_err());
        assert!(Glm53Dflash2TopkPlan::new(1, 0, 154_880, 16).is_err());
        assert!(Glm53Dflash2TopkPlan::new(1, 9, 154_880, 16).is_err());
        assert!(Glm53Dflash2TopkPlan::new(1, 8, 154_879, 16).is_err());
        assert!(Glm53Dflash2TopkPlan::new(1, 8, 154_880, 15).is_err());
    }

    #[test]
    fn cpu_oracle_pins_unique_ids_lower_token_ties_and_finite_rows() {
        let logits = (0..24)
            .map(|token| rounded((token % 4) as f32 - 2.0))
            .collect::<Vec<_>>();
        let selected = reference_topk(&logits, 16).unwrap();
        assert_eq!(selected.len(), 16);
        assert_eq!(selected[0], (3, 1.0));
        assert_eq!(selected[1], (7, 1.0));
        assert_eq!(selected[2], (11, 1.0));
        let mut ids = selected.iter().map(|entry| entry.0).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 16);
        assert!(selected.iter().all(|entry| rounded(entry.1) == entry.1));

        let mut invalid = logits;
        invalid[23] = f32::NAN;
        assert!(reference_topk(&invalid, 16).is_none());
    }

    #[test]
    fn aliases_and_forged_plans_fail_before_launch() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53Dflash2TopkKernel::load(&gpu).unwrap();
        let plan = Glm53Dflash2TopkPlan::new(1, 8, 154_880, 16).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53Dflash2TopkBuffers {
            logits_bf16: at(0x10_0000, plan.logits_bytes),
            candidates_u32: at(0x40_0000, plan.candidate_bytes),
            unary_f32: at(0x50_0000, plan.unary_bytes),
            status_u32: at(0x60_0000, plan.status_bytes),
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53Dflash2TopkBuffers {
                        unary_f32: valid.candidates_u32,
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        let mut forged = plan;
        forged.rows += 1;
        assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
