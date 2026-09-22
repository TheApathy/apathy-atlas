// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const SELECTOR_RANK: u32 = 256;
const SELECTOR_TOP_K: u32 = 16;
const PHYSICAL_VOCAB: u32 = 154_880;
const LOGICAL_VOCAB: u32 = PHYSICAL_VOCAB;
const MASK_TOKEN_ID: u32 = 154_856;
const MAX_BLOCK_TOKENS: u32 = 8;
const MAX_GRID_X: u32 = i32::MAX as u32;
const THREADS: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2SelectorPlan {
    pub batch: u32,
    pub tokens: u32,
    pub blocks: u32,
    pub unary_bytes: usize,
    pub candidate_bytes: usize,
    pub hidden_bytes: usize,
    pub codebook_bytes: usize,
    pub anchor_bytes: usize,
    pub producer_status_bytes: usize,
    pub path_bytes: usize,
    pub status_bytes: usize,
}

impl Glm53Dflash2SelectorPlan {
    pub fn new(
        batch: u32,
        tokens: u32,
        selector_rank: u32,
        selector_top_k: u32,
        physical_vocab: u32,
        logical_vocab: u32,
        mask_token_id: u32,
    ) -> Result<Self> {
        if batch == 0 || batch > MAX_GRID_X || tokens == 0 || tokens > MAX_BLOCK_TOKENS {
            bail!("GLM DFlash2 selector requires batch>0 and 1..=8 tokens");
        }
        if selector_rank != SELECTOR_RANK || selector_top_k != SELECTOR_TOP_K {
            bail!("GLM DFlash2 selector requires exact rank256/top16 geometry");
        }
        if physical_vocab != PHYSICAL_VOCAB
            || logical_vocab != LOGICAL_VOCAB
            || mask_token_id != MASK_TOKEN_ID
            || mask_token_id >= logical_vocab
        {
            bail!("GLM DFlash2 selector vocabulary or mask-token contract drift");
        }
        let rows = usize::try_from(batch)?
            .checked_mul(usize::try_from(tokens)?)
            .context("GLM DFlash2 selector row overflow")?;
        let row_bytes = |columns: u32, width: usize, label: &str| -> Result<usize> {
            rows.checked_mul(usize::try_from(columns)?)
                .and_then(|count| count.checked_mul(width))
                .with_context(|| format!("GLM DFlash2 selector {label} byte overflow"))
        };
        let batch_bytes = |columns: u32, width: usize, label: &str| -> Result<usize> {
            usize::try_from(batch)?
                .checked_mul(usize::try_from(columns)?)
                .and_then(|count| count.checked_mul(width))
                .with_context(|| format!("GLM DFlash2 selector {label} byte overflow"))
        };
        let codebook_bytes = usize::try_from(physical_vocab)?
            .checked_mul(usize::try_from(selector_rank)?)
            .and_then(|count| count.checked_mul(2))
            .context("GLM DFlash2 selector codebook byte overflow")?;
        Ok(Self {
            batch,
            tokens,
            blocks: batch,
            unary_bytes: row_bytes(selector_top_k, 4, "unary")?,
            candidate_bytes: row_bytes(selector_top_k, 4, "candidate")?,
            hidden_bytes: row_bytes(selector_rank, 2, "hidden")?,
            codebook_bytes,
            anchor_bytes: batch_bytes(1, 4, "anchor")?,
            producer_status_bytes: row_bytes(1, 4, "producer status")?,
            path_bytes: row_bytes(1, 4, "path")?,
            status_bytes: batch_bytes(1, 4, "status")?,
        })
    }

    fn validate(self) -> Result<()> {
        let expected = Self::new(
            self.batch,
            self.tokens,
            SELECTOR_RANK,
            SELECTOR_TOP_K,
            PHYSICAL_VOCAB,
            LOGICAL_VOCAB,
            MASK_TOKEN_ID,
        )?;
        if self != expected {
            bail!("GLM DFlash2 selector plan was modified after admission");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2SelectorBuffers {
    /// Exact BF16 logits widened to F32 by the top-k producer.
    pub unary_f32: GgmlIqBuffer,
    pub candidates_u32: GgmlIqBuffer,
    pub hidden_bf16: GgmlIqBuffer,
    pub predecessor_bf16: GgmlIqBuffer,
    pub successor_bf16: GgmlIqBuffer,
    pub anchors_u32: GgmlIqBuffer,
    /// Per-row status from the top-k producer; every value must be zero.
    pub producer_status_u32: GgmlIqBuffer,
    pub path_u32: GgmlIqBuffer,
    /// Per-batch zero on success and one on rejected device data.
    pub status_u32: GgmlIqBuffer,
}

pub struct Glm53Dflash2SelectorKernel {
    greedy_walk: KernelHandle,
}

impl Glm53Dflash2SelectorKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            greedy_walk: gpu.kernel(
                "glm53_dflash2_selector",
                "atlas_glm53_dflash2_selector_walk",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Dflash2SelectorPlan,
        buffers: Glm53Dflash2SelectorBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.greedy_walk)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.unary_f32.ptr)
            .arg_ptr(buffers.candidates_u32.ptr)
            .arg_ptr(buffers.hidden_bf16.ptr)
            .arg_ptr(buffers.predecessor_bf16.ptr)
            .arg_ptr(buffers.successor_bf16.ptr)
            .arg_ptr(buffers.anchors_u32.ptr)
            .arg_ptr(buffers.producer_status_u32.ptr)
            .arg_ptr(buffers.path_u32.ptr)
            .arg_ptr(buffers.status_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(SELECTOR_RANK)
            .arg_u32(SELECTOR_TOP_K)
            .arg_u32(PHYSICAL_VOCAB)
            .arg_u32(LOGICAL_VOCAB)
            .arg_u32(MASK_TOKEN_ID)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53Dflash2SelectorPlan,
    buffers: Glm53Dflash2SelectorBuffers,
) -> Result<()> {
    let named = [
        ("unary", buffers.unary_f32, plan.unary_bytes),
        ("candidates", buffers.candidates_u32, plan.candidate_bytes),
        ("hidden", buffers.hidden_bf16, plan.hidden_bytes),
        (
            "predecessor codebook",
            buffers.predecessor_bf16,
            plan.codebook_bytes,
        ),
        (
            "successor codebook",
            buffers.successor_bf16,
            plan.codebook_bytes,
        ),
        ("anchors", buffers.anchors_u32, plan.anchor_bytes),
        (
            "producer status",
            buffers.producer_status_u32,
            plan.producer_status_bytes,
        ),
        ("path", buffers.path_u32, plan.path_bytes),
        ("status", buffers.status_u32, plan.status_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != expected {
            bail!("GLM DFlash2 selector {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM DFlash2 selector {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DFlash2 selector device buffers overlap");
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

    fn scheduled_score(unary: f32, pred: &[f32], hidden: &[f32], succ: &[f32]) -> f32 {
        let mut pair = 0.0f32;
        for ((&pred, &hidden), &succ) in pred.iter().zip(hidden).zip(succ) {
            let gate = rounded(pred * hidden);
            pair = gate.mul_add(succ, pair);
        }
        rounded(rounded(unary) + rounded(pair))
    }

    fn walk(
        unary: &[f32],
        candidates: &[u32],
        hidden: &[f32],
        pred: &[f32],
        succ: &[f32],
        tokens: usize,
        top_k: usize,
        rank: usize,
        producer_status: &[u32],
        mut predecessor: u32,
    ) -> Option<Vec<u32>> {
        if producer_status.len() != tokens || producer_status.iter().any(|&status| status != 0) {
            return None;
        }
        let mut path = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let mut best_slot = 0;
            let mut best = f32::NEG_INFINITY;
            for slot in 0..top_k {
                let candidate = candidates[token * top_k + slot] as usize;
                let score = scheduled_score(
                    unary[token * top_k + slot],
                    &pred[predecessor as usize * rank..][..rank],
                    &hidden[token * rank..][..rank],
                    &succ[candidate * rank..][..rank],
                );
                if score > best {
                    best = score;
                    best_slot = slot;
                }
            }
            predecessor = candidates[token * top_k + best_slot];
            path.push(predecessor);
        }
        Some(path)
    }

    #[test]
    fn exact_geometry_extents_and_bf16_schedule_are_pinned() {
        let plan = Glm53Dflash2SelectorPlan::new(2, 8, 256, 16, 154_880, 154_880, 154_856).unwrap();
        assert_eq!(plan.blocks, 2);
        assert_eq!(plan.unary_bytes, 2 * 8 * 16 * 4);
        assert_eq!(plan.candidate_bytes, 2 * 8 * 16 * 4);
        assert_eq!(plan.hidden_bytes, 2 * 8 * 256 * 2);
        assert_eq!(plan.codebook_bytes, 154_880 * 256 * 2);
        assert_eq!(plan.anchor_bytes, 2 * 4);
        assert_eq!(plan.producer_status_bytes, 2 * 8 * 4);
        assert_eq!(plan.path_bytes, 2 * 8 * 4);
        assert_eq!(plan.status_bytes, 2 * 4);
        assert!(Glm53Dflash2SelectorPlan::new(0, 8, 256, 16, 154_880, 154_880, 154_856).is_err());
        assert!(Glm53Dflash2SelectorPlan::new(1, 9, 256, 16, 154_880, 154_880, 154_856).is_err());
        assert!(Glm53Dflash2SelectorPlan::new(1, 8, 255, 16, 154_880, 154_880, 154_856).is_err());
        assert!(Glm53Dflash2SelectorPlan::new(1, 8, 256, 16, 154_880, 154_880, 154_855).is_err());

        let mut found_rounding_difference = false;
        for seed in 1..4096usize {
            let sample = |factor: usize| rounded(((seed * factor) % 257) as f32 / 31.0 - 4.0);
            let pred = [sample(17), sample(29), sample(43)];
            let hidden = [sample(61), sample(73), sample(89)];
            let succ = [sample(101), sample(113), sample(127)];
            let exact = scheduled_score(sample(137), &pred, &hidden, &succ);
            let fused = rounded(
                sample(137)
                    + pred
                        .iter()
                        .zip(hidden)
                        .zip(succ)
                        .map(|((&p, h), s)| p * h * s)
                        .sum::<f32>(),
            );
            if exact.to_bits() != fused.to_bits() {
                found_rounding_difference = true;
                break;
            }
        }
        assert!(found_rounding_difference);
    }

    #[test]
    fn predecessor_is_the_prior_selected_token_and_ties_use_first_slot() {
        let unary = [0.5, 0.0, 0.0, 0.0];
        let candidates = [1, 2, 3, 4];
        let hidden = [1.0, 1.0, 1.0, 1.0];
        let pred = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let succ = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            walk(
                &unary,
                &candidates,
                &hidden,
                &pred,
                &succ,
                2,
                2,
                2,
                &[0, 0],
                0,
            )
            .unwrap(),
            [1, 3]
        );

        let tied = walk(
            &[0.0, 0.0],
            &[7, 6],
            &[0.0],
            &[0.0; 8],
            &[0.0; 8],
            1,
            2,
            1,
            &[0],
            0,
        )
        .unwrap();
        assert_eq!(tied, [7]);

        // Stale candidate/unary rows can be structurally valid. A failed
        // producer status must still prevent a selector path from existing.
        assert!(
            walk(
                &unary,
                &candidates,
                &hidden,
                &pred,
                &succ,
                2,
                2,
                2,
                &[0, 1],
                0,
            )
            .is_none()
        );
    }

    #[test]
    fn invalid_buffers_fail_before_launch() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53Dflash2SelectorKernel::load(&gpu).unwrap();
        let plan = Glm53Dflash2SelectorPlan::new(2, 8, 256, 16, 154_880, 154_880, 154_856).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53Dflash2SelectorBuffers {
            unary_f32: at(0x10_0000, plan.unary_bytes),
            candidates_u32: at(0x20_0000, plan.candidate_bytes),
            hidden_bf16: at(0x30_0000, plan.hidden_bytes),
            predecessor_bf16: at(0x1000_0000, plan.codebook_bytes),
            successor_bf16: at(0x2000_0000, plan.codebook_bytes),
            anchors_u32: at(0x3000_0000, plan.anchor_bytes),
            producer_status_u32: at(0x3001_0000, plan.producer_status_bytes),
            path_u32: at(0x3100_0000, plan.path_bytes),
            status_u32: at(0x3200_0000, plan.status_bytes),
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53Dflash2SelectorBuffers {
                        path_u32: valid.producer_status_u32,
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        let mut forged = plan;
        forged.tokens = 7;
        assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
