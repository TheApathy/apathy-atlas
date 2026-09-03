// SPDX-License-Identifier: AGPL-3.0-only

//! CPU ownership and storage planner for GLM-5.3 DeepSeek Sparse Attention.
//!
//! Separate from [`super::PagedKvCache`]: one absorbed MLA latent plus one
//! compressed index-pool stream, never dense expanded K/V tensors.

use anyhow::{Context, Result, bail};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

pub const GLM53_DSA_MAX_POSITIONS: u32 = 1_048_576;
pub const GLM53_DSA_LAYERS: usize = 11;
pub const GLM53_DSA_LATENT_RANK: usize = 512;
pub const GLM53_DSA_INDEX_DIM: usize = 128;
pub const GLM53_DSA_KPOOL: u32 = 4;
pub const GLM53_DSA_RUNTIME_KERNEL_IMPLEMENTED: bool = false;
const MAX_POSITIONS: usize = GLM53_DSA_MAX_POSITIONS as usize;
const POOL_CAPACITY: usize = MAX_POSITIONS / GLM53_DSA_KPOOL as usize;
/// Slots the raw index tail must hold. THE definition -- every other site
/// derives from this one rather than restating `KPOOL - 1`.
///
/// It is KPOOL-1, and the derivation is exact. A pool completes once its last
/// position has arrived and the walk publishes that pool's row into the cache
/// BEFORE score/topk, so the count of pools visible to the scorer at position
/// q is `(q + 1) / KPOOL` and pooled coverage ends at KPOOL*pools - 1. The
/// tail carries the remainder, `(q + 1) - KPOOL*pools = (q + 1) mod KPOOL`,
/// whose maximum is KPOOL-1. The current position itself never needs a tail
/// slot: it enters selection through the query metadata and the causal mask.
///
/// A capacity of KPOOL looks necessary under the OLD pool count `q / KPOOL`,
/// which excluded the pool completing on this very step and left its group
/// with nowhere to live. Raising the capacity would have papered over that and
/// diverged from the reference: GLM5Next carries three tail slots, pinned here
/// as `OUTPUT_WIDTH = INDEX_TOPK + KPOOL - 1 = 2051` and in the reference model
/// at `glm53_dsa_topk_tests.rs`, which covers q = 7 as pools {0,1} plus a
/// three-slot tail plus the current position. The pool count was the wrong
/// half, not this constant.
pub const GLM53_DSA_TAIL_CAPACITY: u32 = GLM53_DSA_KPOOL - 1;

const TAIL_CAPACITY: usize = GLM53_DSA_TAIL_CAPACITY as usize;
const REGION_ALIGNMENT: usize = 256;
static NEXT_DEVICE_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// Opaque identity for one CPU cache and its future backing device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Glm53DsaDeviceIdentity(NonZeroU64);

fn identity_step(current: u64) -> Result<(Glm53DsaDeviceIdentity, u64)> {
    let identity = NonZeroU64::new(current).context("GLM DSA device identity cannot be zero")?;
    let next = current
        .checked_add(1)
        .context("GLM DSA device identity exhausted")?;
    Ok((Glm53DsaDeviceIdentity(identity), next))
}

fn issue_device_identity() -> Result<Glm53DsaDeviceIdentity> {
    let current = NEXT_DEVICE_IDENTITY
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            identity_step(current).ok().map(|(_, next)| next)
        })
        .map_err(|_| anyhow::anyhow!("GLM DSA device identity exhausted"))?;
    Ok(identity_step(current)?.0)
}

/// Storage width admitted by the planner. Runtime DSA kernels are not landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaStorage {
    Bf16,
    Fp8,
}
impl Glm53DsaStorage {
    const fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }
}

/// One non-overlapping region in a single backing allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaRegion {
    pub offset: usize,
    pub bytes: usize,
}
/// Exact all-layer capacity layout. Validity is one byte per pool/tail slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53DsaLayout {
    pub batch: usize,
    pub storage: Glm53DsaStorage,
    pub latent: Glm53DsaRegion,
    pub pool_keys: Glm53DsaRegion,
    pub pool_validity: Glm53DsaRegion,
    pub tail_keys: Glm53DsaRegion,
    pub tail_gates: Glm53DsaRegion,
    pub tail_validity: Glm53DsaRegion,
    pub total_bytes: usize,
}

impl Glm53DsaLayout {
    pub fn new(batch: usize, storage: Glm53DsaStorage) -> Result<Self> {
        if batch == 0 {
            bail!("GLM DSA cache requires a positive batch capacity");
        }
        let bytes = storage.element_bytes();
        let latent_bytes = product(&[
            batch,
            GLM53_DSA_LAYERS,
            MAX_POSITIONS,
            GLM53_DSA_LATENT_RANK,
            bytes,
        ])?;
        let pool_key_bytes = product(&[
            batch,
            GLM53_DSA_LAYERS,
            POOL_CAPACITY,
            GLM53_DSA_INDEX_DIM,
            bytes,
        ])?;
        let pool_validity_bytes = product(&[batch, GLM53_DSA_LAYERS, POOL_CAPACITY])?;
        let tail_bytes = product(&[
            batch,
            GLM53_DSA_LAYERS,
            TAIL_CAPACITY,
            GLM53_DSA_INDEX_DIM,
            bytes,
        ])?;
        let tail_validity_bytes = product(&[batch, GLM53_DSA_LAYERS, TAIL_CAPACITY])?;

        let mut cursor = 0;
        let latent = place_region(&mut cursor, latent_bytes)?;
        let pool_keys = place_region(&mut cursor, pool_key_bytes)?;
        let pool_validity = place_region(&mut cursor, pool_validity_bytes)?;
        let tail_keys = place_region(&mut cursor, tail_bytes)?;
        let tail_gates = place_region(&mut cursor, tail_bytes)?;
        let tail_validity = place_region(&mut cursor, tail_validity_bytes)?;
        let total_bytes = align_up(cursor)?;
        Ok(Self {
            batch,
            storage,
            latent,
            pool_keys,
            pool_validity,
            tail_keys,
            tail_gates,
            tail_validity,
            total_bytes,
        })
    }
}

fn product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .context("GLM DSA layout byte overflow")
    })
}

fn align_up(value: usize) -> Result<usize> {
    value
        .checked_add(REGION_ALIGNMENT - 1)
        .map(|sum| sum & !(REGION_ALIGNMENT - 1))
        .context("GLM DSA layout alignment overflow")
}

fn place_region(cursor: &mut usize, bytes: usize) -> Result<Glm53DsaRegion> {
    let offset = align_up(*cursor)?;
    *cursor = offset
        .checked_add(bytes)
        .context("GLM DSA layout region overflow")?;
    Ok(Glm53DsaRegion { offset, bytes })
}

/// Generation-bound handle; a freed handle cannot name its replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaSequenceHandle {
    device_identity: Glm53DsaDeviceIdentity,
    slot: usize,
    generation: u64,
}
impl Glm53DsaSequenceHandle {
    pub const fn device_identity(self) -> Glm53DsaDeviceIdentity {
        self.device_identity
    }
    pub const fn slot(self) -> usize {
        self.slot
    }
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Constant-size logical cache state; no per-position CPU table is rebuilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaSequenceView {
    pub logical_len: u32,
    pub complete_pools: u32,
    pub valid_complete_pools: u32,
    pub tail_len: u8,
    pub tail_valid_mask: u8,
}

impl Glm53DsaSequenceView {
    fn for_length(logical_len: u32) -> Self {
        let complete_pools = logical_len / GLM53_DSA_KPOOL;
        let tail_len = (logical_len % GLM53_DSA_KPOOL) as u8;
        let tail_valid_mask = if tail_len == 0 {
            0
        } else {
            (1u8 << tail_len) - 1
        };
        Self {
            logical_len,
            complete_pools,
            valid_complete_pools: complete_pools,
            tail_len,
            tail_valid_mask,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaAppendPlan {
    pub handle: Glm53DsaSequenceHandle,
    pub nonce: u64,
    pub start_position: u32,
    pub token_count: u32,
    /// Exclusive logical end; the last planned absolute position is `end_position - 1`.
    pub end_position: u32,
    pub first_pool_to_write: u32,
    pub complete_pools_to_write: u32,
    pub initial_tail_len: u8,
    pub final_tail_len: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaPrefixSnapshot {
    owner: Glm53DsaSequenceHandle,
    epoch: u64,
    pub view: Glm53DsaSequenceView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SequenceState {
    logical_len: u32,
    epoch: u64,
    active: Option<Glm53DsaAppendPlan>,
    poisoned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Slot {
    generation: u64,
    state: Option<SequenceState>,
}

/// CPU ownership state for cache slots. Device allocation/copies are external.
pub struct Glm53DsaCache {
    device_identity: Glm53DsaDeviceIdentity,
    layout: Glm53DsaLayout,
    slots: Vec<Slot>,
    next_nonce: u64,
}
impl Glm53DsaCache {
    pub fn new(batch: usize, storage: Glm53DsaStorage) -> Result<Self> {
        let layout = Glm53DsaLayout::new(batch, storage)?;
        let device_identity = issue_device_identity()?;
        let empty = Slot {
            generation: 1,
            state: None,
        };
        Ok(Self {
            device_identity,
            layout,
            slots: vec![empty; batch],
            next_nonce: 0,
        })
    }

    pub fn layout(&self) -> &Glm53DsaLayout {
        &self.layout
    }

    pub const fn device_identity(&self) -> Glm53DsaDeviceIdentity {
        self.device_identity
    }

    pub fn claim_sequence(&mut self) -> Result<Glm53DsaSequenceHandle> {
        let (slot, entry) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.state.is_none())
            .context("GLM DSA cache has no free sequence slot")?;
        entry.state = Some(SequenceState {
            logical_len: 0,
            epoch: 0,
            active: None,
            poisoned: false,
        });
        Ok(Glm53DsaSequenceHandle {
            device_identity: self.device_identity,
            slot,
            generation: entry.generation,
        })
    }

    pub fn sequence_view(&self, handle: Glm53DsaSequenceHandle) -> Result<Glm53DsaSequenceView> {
        let logical_len = self.state(handle)?.logical_len;
        Ok(Glm53DsaSequenceView::for_length(logical_len))
    }

    /// Permanently quarantine after an uncertain effect, retaining diagnostic state.
    pub fn poison_sequence(&mut self, handle: Glm53DsaSequenceHandle) -> Result<()> {
        self.owned_state_mut(handle)?.poisoned = true;
        Ok(())
    }
    pub fn sequence_is_poisoned(&self, handle: Glm53DsaSequenceHandle) -> Result<bool> {
        Ok(self.owned_state(handle)?.poisoned)
    }

    pub fn begin_append(
        &mut self,
        handle: Glm53DsaSequenceHandle,
        token_count: u32,
    ) -> Result<Glm53DsaAppendPlan> {
        if token_count == 0 {
            bail!("GLM DSA append requires at least one token");
        }
        let state = *self.state(handle)?;
        if state.active.is_some() {
            bail!("GLM DSA sequence already has an active transaction");
        }
        let end_position = state
            .logical_len
            .checked_add(token_count)
            .context("GLM DSA append position overflow")?;
        if end_position > GLM53_DSA_MAX_POSITIONS {
            bail!("GLM DSA append exceeds position 1,048,575");
        }
        self.next_nonce = self
            .next_nonce
            .checked_add(1)
            .context("GLM DSA transaction nonce overflow")?;
        let before = Glm53DsaSequenceView::for_length(state.logical_len);
        let after = Glm53DsaSequenceView::for_length(end_position);
        let plan = Glm53DsaAppendPlan {
            handle,
            nonce: self.next_nonce,
            start_position: state.logical_len,
            token_count,
            end_position,
            first_pool_to_write: before.complete_pools,
            complete_pools_to_write: after.complete_pools - before.complete_pools,
            initial_tail_len: before.tail_len,
            final_tail_len: after.tail_len,
        };
        self.state_mut(handle)?.active = Some(plan);
        Ok(plan)
    }

    /// Device rows must remain staged until this decision. A partial commit
    /// exposes only its accepted prefix; later staged rows stay hidden until overwrite.
    pub fn commit_append(
        &mut self,
        plan: Glm53DsaAppendPlan,
        accepted_tokens: u32,
    ) -> Result<Glm53DsaSequenceView> {
        if accepted_tokens > plan.token_count {
            bail!("GLM DSA commit exceeds the planned token count");
        }
        let state = self.state_mut(plan.handle)?;
        if state.active != Some(plan) {
            bail!("stale or foreign GLM DSA append transaction");
        }
        let logical_len = plan
            .start_position
            .checked_add(accepted_tokens)
            .context("GLM DSA commit position overflow")?;
        let epoch = state
            .epoch
            .checked_add(1)
            .context("GLM DSA sequence epoch overflow")?;
        state.logical_len = logical_len;
        state.epoch = epoch;
        state.active = None;
        Ok(Glm53DsaSequenceView::for_length(logical_len))
    }

    pub fn rollback_append(&mut self, plan: Glm53DsaAppendPlan) -> Result<()> {
        let state = self.state_mut(plan.handle)?;
        if state.active != Some(plan) {
            bail!("stale or foreign GLM DSA append transaction");
        }
        state.active = None;
        Ok(())
    }

    pub fn snapshot_prefix(
        &self,
        handle: Glm53DsaSequenceHandle,
        logical_len: u32,
    ) -> Result<Glm53DsaPrefixSnapshot> {
        let state = self.state(handle)?;
        if state.active.is_some() || logical_len > state.logical_len {
            bail!("GLM DSA prefix is not a committed owned range");
        }
        if logical_len != state.logical_len && logical_len % GLM53_DSA_KPOOL != 0 {
            bail!("a truncated GLM DSA prefix must end on a complete-pool boundary");
        }
        Ok(Glm53DsaPrefixSnapshot {
            owner: handle,
            epoch: state.epoch,
            view: Glm53DsaSequenceView::for_length(logical_len),
        })
    }

    /// Owner-local only; cross-slot reuse requires an explicit device copy.
    pub fn restore_prefix(
        &mut self,
        handle: Glm53DsaSequenceHandle,
        prefix: Glm53DsaPrefixSnapshot,
    ) -> Result<Glm53DsaSequenceView> {
        if prefix.owner != handle {
            bail!("GLM DSA prefix alias across sequence owners is forbidden");
        }
        let state = self.state_mut(handle)?;
        if state.active.is_some() || state.epoch != prefix.epoch {
            bail!("stale GLM DSA prefix snapshot");
        }
        state.epoch = state
            .epoch
            .checked_add(1)
            .context("GLM DSA sequence epoch overflow")?;
        state.logical_len = prefix.view.logical_len;
        Ok(prefix.view)
    }

    pub fn free_sequence(&mut self, handle: Glm53DsaSequenceHandle) -> Result<()> {
        let entry = self.entry_mut(handle)?;
        let state = entry
            .state
            .context("GLM DSA sequence slot is already free")?;
        if state.poisoned {
            bail!("cannot free a poisoned GLM DSA sequence");
        }
        if state.active.is_some() {
            bail!("cannot free a GLM DSA sequence with an active transaction");
        }
        let generation = entry
            .generation
            .checked_add(1)
            .context("GLM DSA sequence generation overflow")?;
        entry.state = None;
        entry.generation = generation;
        Ok(())
    }

    fn state(&self, handle: Glm53DsaSequenceHandle) -> Result<&SequenceState> {
        let state = self.owned_state(handle)?;
        if state.poisoned {
            bail!("GLM DSA sequence is poisoned");
        }
        Ok(state)
    }

    fn state_mut(&mut self, handle: Glm53DsaSequenceHandle) -> Result<&mut SequenceState> {
        let state = self.owned_state_mut(handle)?;
        if state.poisoned {
            bail!("GLM DSA sequence is poisoned");
        }
        Ok(state)
    }

    fn owned_state(&self, handle: Glm53DsaSequenceHandle) -> Result<&SequenceState> {
        self.entry(handle)?
            .state
            .as_ref()
            .context("GLM DSA sequence slot is free")
    }

    fn owned_state_mut(&mut self, handle: Glm53DsaSequenceHandle) -> Result<&mut SequenceState> {
        self.entry_mut(handle)?
            .state
            .as_mut()
            .context("GLM DSA sequence slot is free")
    }

    fn entry(&self, handle: Glm53DsaSequenceHandle) -> Result<&Slot> {
        if handle.device_identity != self.device_identity {
            bail!("foreign GLM DSA device identity");
        }
        let entry = self
            .slots
            .get(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }

    fn entry_mut(&mut self, handle: Glm53DsaSequenceHandle) -> Result<&mut Slot> {
        if handle.device_identity != self.device_identity {
            bail!("foreign GLM DSA device identity");
        }
        let entry = self
            .slots
            .get_mut(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }
}
#[cfg(test)]
#[path = "glm53_dsa_tests.rs"]
mod tests;
