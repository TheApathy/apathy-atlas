// SPDX-License-Identifier: AGPL-3.0-only

//! Logits/argmax boundary only: real policy and ordinary transitions run above
//! this fixture. Unique BF16 maxima avoid imposing a new tie-breaking contract.

use super::fixture::{MODEL, row};
use crate::scheduler::{Model, SequenceState};
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) const BASE: DevicePtr = DevicePtr(0x1000);
pub(super) const VOCAB: usize = 5;

pub(super) struct Boundary {
    pub bytes: Vec<u8>,
    picks: Vec<u32>,
    argmax_calls: AtomicUsize,
    copies: Mutex<Vec<(usize, usize)>>,
    pub fail_scalar: bool,
}

impl Boundary {
    pub fn new(rows: &[[f32; VOCAB]]) -> Self {
        let bytes: Vec<u8> = rows.iter().flat_map(|values| row(values)).collect();
        let picks = bytes
            .chunks_exact(VOCAB * 2)
            .map(|raw| {
                let values: Vec<f32> = raw
                    .chunks_exact(2)
                    .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                    .collect();
                assert!(values.iter().all(|v| v.is_finite()));
                let (index, maximum) = values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                assert_eq!(values.iter().filter(|v| **v == *maximum).count(), 1);
                index as u32
            })
            .collect();
        Self {
            bytes,
            picks,
            argmax_calls: AtomicUsize::new(0),
            copies: Mutex::new(Vec::new()),
            fail_scalar: false,
        }
    }

    pub fn argmax_calls(&self) -> usize {
        self.argmax_calls.load(Ordering::Relaxed)
    }
    pub fn full_reads(&self) -> usize {
        self.copies
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, len)| *len >= VOCAB * 2)
            .count()
    }
    pub fn scalar_offsets(&self) -> Vec<usize> {
        self.copies
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(offset, len)| (*len == 2).then_some(*offset))
            .collect()
    }

    fn checked_row(&self, ptr: DevicePtr) -> Result<usize> {
        let offset = ptr
            .0
            .checked_sub(BASE.0)
            .ok_or_else(|| anyhow::anyhow!("logits pointer precedes fixture"))?
            as usize;
        ensure!(offset.is_multiple_of(VOCAB * 2), "unaligned logits row");
        let row = offset / (VOCAB * 2);
        ensure!(row < self.picks.len(), "logits row outside fixture");
        Ok(row)
    }
}

macro_rules! forbidden_gpu_methods {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        $(fn $name(&self, $($arg: $ty),*) -> $ret { MODEL.$name($($arg),*) })*
    };
}

impl Model for Boundary {
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        BASE
    }
    fn copy_logits_to_host(&self, ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let offset = ptr
            .0
            .checked_sub(BASE.0)
            .ok_or_else(|| anyhow::anyhow!("copy precedes fixture"))? as usize;
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| anyhow::anyhow!("copy extent overflow"))?;
        let source = self
            .bytes
            .get(offset..end)
            .ok_or_else(|| anyhow::anyhow!("copy outside fixture"))?;
        self.copies.lock().unwrap().push((offset, dst.len()));
        ensure!(
            !(self.fail_scalar && dst.len() == 2),
            "injected positivity-copy failure"
        );
        dst.copy_from_slice(source);
        Ok(())
    }
    fn argmax_on_device(&self, ptr: DevicePtr, _: u64) -> Result<u32> {
        let row = self.checked_row(ptr)?;
        self.argmax_calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.picks[row])
    }
    fn argmax_batch(&self, ptr: DevicePtr, rows: usize, _: u64) -> Result<Vec<u32>> {
        let start = self.checked_row(ptr)?;
        let end = start
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("argmax extent overflow"))?;
        let picks = self
            .picks
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("argmax outside fixture"))?;
        self.argmax_calls.fetch_add(1, Ordering::Relaxed);
        Ok(picks.to_vec())
    }
    forbidden_gpu_methods! {
        fn prefill(tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn prefill_chunk(tokens: &[u32], seq: &mut SequenceState, start: usize, len: usize, last: bool, stream: u64) -> Result<DevicePtr>;
        fn decode(token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn decode_batch(tokens: &[u32], seqs: &mut [&mut SequenceState], stream: u64) -> Result<DevicePtr>;
        fn bind_gpu_to_thread() -> Result<()>;
        fn alloc_sequence() -> Result<SequenceState>;
        fn hidden_after_norm() -> DevicePtr;
        fn decode_verify(tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<Vec<u32>>;
        fn checkpoint_ssm_states(seq: &mut SequenceState) -> Result<()>;
        fn rollback_ssm_states(seq: &mut SequenceState, accepted: usize) -> Result<()>;
        fn generate_speculative(tokens: &[u32], params: &spark_runtime::sampler::SamplingParams, drafts: usize) -> Result<spark_model::engine::GenerateResult>;
        fn has_proposer() -> bool;
        fn has_self_speculative() -> bool;
        fn decode_draft(token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn cache_sequence(seq: &SequenceState) -> ();
        fn free_sequence(seq: &mut SequenceState) -> Result<()>;
        fn compact_sequence(seq: &mut SequenceState, slot: usize) -> Result<()>;
        fn detach_slot_for_reuse(seq: &mut SequenceState) -> ();
        fn decode_verify_graphed(tokens: &[u32; 2], seq: &mut SequenceState, stream: u64) -> Result<[u32; 2]>;
        fn decode_verify_graphed_k3(tokens: &[u32; 3], seq: &mut SequenceState, stream: u64) -> Result<[u32; 3]>;
        fn decode_verify_graphed_k4(tokens: &[u32; 4], seq: &mut SequenceState, stream: u64) -> Result<[u32; 4]>;
        fn save_hidden_for_mtp(index: usize, stream: u64) -> Result<()>;
        fn run_mtp_propose(token: u32, position: usize, seq: &mut SequenceState, stream: u64) -> Result<Option<u32>>;
        fn run_mtp_propose_multi(token: u32, position: usize, drafts: usize, seq: &mut SequenceState, stream: u64, mask: Option<&[i32]>) -> Result<Vec<u32>>;
        fn trim_proposer_state(seq: &mut SequenceState, accepted: usize, stream: u64) -> Result<()>;
    }
}
