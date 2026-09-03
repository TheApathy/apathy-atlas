// SPDX-License-Identifier: AGPL-3.0-only

//! Effectful but unregistered B1/T1 GLM-5.3 bootstrap.
//!
//! One call synchronously copies one admitted token ID through the backend's
//! default-stream-complete `copy_h2d`, then enqueues exactly two kernels: Q5_K
//! gather followed by mHC expansion on the same nonzero caller stream. It owns
//! no allocation, loads no kernel, touches no persistent sequence state, and
//! does not claim that a layer, logits, or a full token forward has completed.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{
    GgmlIqBuffer, GgmlQ5EmbeddingBuffers, GgmlQ5EmbeddingKernel, GgmlQ5EmbeddingPlan,
    Glm53HyperKernels, Glm53HyperPlan,
};

const TOKENS: u32 = 1;
const VOCAB: u32 = 154_880;
const HIDDEN: u32 = 4_096;
const HC: u32 = 4;
const SINKHORN_ITERS: u32 = 20;
const KERNEL_LAUNCHES: u32 = 2;
/// Number of device buffers the bootstrap validates and fingerprints. The
/// `named` table in `validate_buffers` is annotated with this length, so a
/// buffer added there without widening the fingerprint fails to compile.
const BUFFER_COUNT: usize = 4;
static NEXT_OWNER_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53B1T1BootstrapPlan {
    pub token_id: u32,
    pub embedding: GgmlQ5EmbeddingPlan,
    pub hyper: Glm53HyperPlan,
    pub kernel_launches: u32,
}

impl Glm53B1T1BootstrapPlan {
    pub fn new(token_id: u32) -> Result<Self> {
        if token_id >= VOCAB {
            bail!("GLM B1/T1 bootstrap token ID exceeds vocab154880");
        }
        Ok(Self {
            token_id,
            embedding: GgmlQ5EmbeddingPlan::new(TOKENS, VOCAB, HIDDEN)?,
            hyper: Glm53HyperPlan::new(TOKENS, HIDDEN, HC, SINKHORN_ITERS)?,
            kernel_launches: KERNEL_LAUNCHES,
        })
    }

    pub fn validate(self) -> Result<()> {
        if Self::new(self.token_id)? != self {
            bail!("forged GLM B1/T1 bootstrap plan");
        }
        if self.embedding.destination_bytes != self.hyper.hidden_bytes {
            bail!("GLM bootstrap embedding/hyper hidden extent drift");
        }
        Ok(())
    }
}

/// Trusted crate-internal view boundary. The later executor must construct
/// these views from one live allocation owner; this bootstrap only validates
/// their exact extents and disjoint ranges and never assumes ownership.
#[derive(Clone, Copy)]
pub(crate) struct Glm53B1T1BootstrapBuffers {
    /// Already-admitted exact Q5_K `[154880,4096]` source bytes.
    pub(crate) source_q5_k: GgmlIqBuffer,
    /// Caller-owned transient `u32[1]`; the admitted scalar is copied here.
    pub(crate) token_ids_u32: GgmlIqBuffer,
    /// Caller-owned transient BF16 `[1,4096]` embedding destination.
    pub(crate) hidden_bf16: GgmlIqBuffer,
    /// Caller-owned transient BF16 `[1,4,4096]` mHC destination.
    pub(crate) streams_bf16: GgmlIqBuffer,
}

impl fmt::Debug for Glm53B1T1BootstrapBuffers {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53B1T1BootstrapBuffers")
            .field("source_q5_k_bytes", &self.source_q5_k.bytes)
            .field("token_ids_u32_bytes", &self.token_ids_u32.bytes)
            .field("hidden_bf16_bytes", &self.hidden_bf16.bytes)
            .field("streams_bf16_bytes", &self.streams_bf16.bytes)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53B1T1BootstrapStage {
    /// Both kernels were enqueued in order on the caller's stream.
    MhcExpandEnqueued,
}

/// This receipt is only an ordered enqueue receipt. It is not a stream-status,
/// persistent-state, layer, logits, runnable-model, or full-token receipt.
#[must_use = "the bootstrap enqueue receipt must be consumed by the later executor"]
#[derive(PartialEq, Eq)]
pub(crate) struct Glm53B1T1BootstrapReceipt {
    owner_generation: u64,
    transaction_nonce: u64,
    token_id: u32,
    stream: u64,
    buffer_ranges: [(u64, u64); BUFFER_COUNT],
    stage: Glm53B1T1BootstrapStage,
    kernel_launches: u32,
}

impl Glm53B1T1BootstrapReceipt {
    pub(crate) const fn owner_generation(&self) -> u64 {
        self.owner_generation
    }

    pub(crate) const fn transaction_nonce(&self) -> u64 {
        self.transaction_nonce
    }

    pub(crate) const fn token_id(&self) -> u32 {
        self.token_id
    }

    pub(crate) const fn stream(&self) -> u64 {
        self.stream
    }

    pub(crate) const fn stage(&self) -> Glm53B1T1BootstrapStage {
        self.stage
    }

    pub(crate) const fn kernel_launches(&self) -> u32 {
        self.kernel_launches
    }

    pub(crate) const fn buffer_ranges(&self) -> [(u64, u64); BUFFER_COUNT] {
        self.buffer_ranges
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ActiveBootstrap {
    transaction_nonce: u64,
    token_id: u32,
    stream: u64,
    buffer_ranges: [(u64, u64); BUFFER_COUNT],
}

/// Borrowed, preloaded kernel wrappers. Their allocation and module ownership
/// remain with the later model executor; this seam cannot load or free them.
pub(crate) struct Glm53B1T1BootstrapKernels<'a> {
    embedding: &'a GgmlQ5EmbeddingKernel,
    hyper: &'a Glm53HyperKernels,
    owner_generation: u64,
    next_nonce: u64,
    active: Option<ActiveBootstrap>,
}

impl<'a> Glm53B1T1BootstrapKernels<'a> {
    pub(crate) fn new(
        embedding: &'a GgmlQ5EmbeddingKernel,
        hyper: &'a Glm53HyperKernels,
    ) -> Result<Self> {
        let owner_generation = NEXT_OWNER_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != 0).then(|| current.checked_add(1)).flatten()
            })
            .map_err(|_| anyhow::anyhow!("GLM B1/T1 bootstrap owner allocator exhausted"))?;
        Ok(Self {
            embedding,
            hyper,
            owner_generation,
            next_nonce: 1,
            active: None,
        })
    }

    /// On any error the caller must discard all three transient staging
    /// buffers. No persistent state or logical position has been published.
    pub(crate) fn execute(
        &mut self,
        gpu: &dyn GpuBackend,
        plan: Glm53B1T1BootstrapPlan,
        buffers: Glm53B1T1BootstrapBuffers,
        stream: u64,
    ) -> Result<Glm53B1T1BootstrapReceipt> {
        if self.active.is_some() {
            bail!("GLM B1/T1 bootstrap receipt has not been consumed");
        }
        plan.validate()?;
        if stream == 0 {
            bail!("GLM B1/T1 bootstrap requires a nonzero caller stream");
        }
        let buffer_ranges = validate_buffers(plan, buffers)?;
        if gpu.stream_is_capturing(stream) {
            bail!("GLM B1/T1 bootstrap is unavailable during graph capture");
        }
        let transaction_nonce = self.next_nonce;
        self.next_nonce = transaction_nonce
            .checked_add(1)
            .context("GLM B1/T1 bootstrap transaction nonce exhausted")?;
        // This API completes its default-stream copy before returning, so the
        // temporary host bytes are gone before either caller-stream launch.
        gpu.copy_h2d(&plan.token_id.to_le_bytes(), buffers.token_ids_u32.ptr)?;
        self.embedding.launch(
            gpu,
            plan.embedding,
            GgmlQ5EmbeddingBuffers {
                source_q5_k: buffers.source_q5_k,
                token_ids_u32: buffers.token_ids_u32,
                destination_bf16: buffers.hidden_bf16,
            },
            stream,
        )?;
        self.hyper.expand(
            gpu,
            plan.hyper,
            buffers.hidden_bf16,
            buffers.streams_bf16,
            stream,
        )?;
        self.active = Some(ActiveBootstrap {
            transaction_nonce,
            token_id: plan.token_id,
            stream,
            buffer_ranges,
        });
        Ok(Glm53B1T1BootstrapReceipt {
            owner_generation: self.owner_generation,
            transaction_nonce,
            token_id: plan.token_id,
            stream,
            buffer_ranges,
            stage: Glm53B1T1BootstrapStage::MhcExpandEnqueued,
            kernel_launches: KERNEL_LAUNCHES,
        })
    }

    /// Retires exactly the current linear receipt. A stale, replayed,
    /// cross-owner, or field-corrupted receipt leaves the active receipt live.
    pub(crate) fn consume_receipt(&mut self, receipt: Glm53B1T1BootstrapReceipt) -> Result<()> {
        let expected = self
            .active
            .context("GLM B1/T1 bootstrap has no active receipt")?;
        if receipt.owner_generation == 0
            || receipt.transaction_nonce == 0
            || receipt.owner_generation != self.owner_generation
            || receipt.transaction_nonce != expected.transaction_nonce
            || receipt.token_id != expected.token_id
            || receipt.stream != expected.stream
            || receipt.buffer_ranges != expected.buffer_ranges
            || receipt.stage != Glm53B1T1BootstrapStage::MhcExpandEnqueued
            || receipt.kernel_launches != KERNEL_LAUNCHES
        {
            bail!("stale, cross-owner, or forged GLM B1/T1 bootstrap receipt");
        }
        self.active = None;
        Ok(())
    }
}

fn validate_buffers(
    plan: Glm53B1T1BootstrapPlan,
    buffers: Glm53B1T1BootstrapBuffers,
) -> Result<[(u64, u64); BUFFER_COUNT]> {
    let named: [_; BUFFER_COUNT] = [
        (
            "Q5_K source",
            buffers.source_q5_k,
            plan.embedding.source_bytes,
            4u64,
        ),
        (
            "token staging",
            buffers.token_ids_u32,
            plan.embedding.token_ids_bytes,
            4,
        ),
        (
            "hidden",
            buffers.hidden_bf16,
            plan.embedding.destination_bytes,
            2,
        ),
        (
            "mHC streams",
            buffers.streams_bf16,
            plan.hyper.streams_bytes,
            2,
        ),
    ];
    let mut ranges = [(0u64, 0u64); BUFFER_COUNT];
    for (slot, (name, buffer, expected, alignment)) in named.iter().copied().enumerate() {
        if buffer.ptr == DevicePtr::NULL
            || buffer.bytes != expected
            || buffer.ptr.0 % alignment != 0
        {
            bail!("GLM B1/T1 bootstrap {name} pointer, alignment, or extent mismatch");
        }
        ranges[slot] = (
            buffer.ptr.0,
            buffer
                .ptr
                .0
                .checked_add(u64::try_from(buffer.bytes)?)
                .with_context(|| format!("GLM B1/T1 bootstrap {name} address overflow"))?,
        );
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM B1/T1 bootstrap device buffers overlap");
            }
        }
    }
    Ok(ranges)
}

#[cfg(test)]
#[path = "b1t1_bootstrap_tests.rs"]
mod tests;
