// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use spark_model::traits::SequenceState;
use spark_runtime::gpu::DevicePtr;

use super::super::*;

pub(super) fn seq_state() -> SequenceState {
    SequenceState {
        tokens: Vec::new(),
        block_table: Vec::new(),
        seq_len: 9,
        layer_states: Vec::new(),
        proposer_state: None,
        proposer_state_alt: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        chunked_prefill_meta: None,
        cached_prefix_tokens: 0,
        prompt_len: 0,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}

pub(super) struct TestModel {
    proposal: Mutex<Option<Result<Vec<u32>>>>,
    tree: Mutex<Option<DDTreePayload>>,
    pub(super) proposals: AtomicUsize,
    pub(super) takes: AtomicUsize,
}

impl TestModel {
    pub(super) fn new(proposal: Result<Vec<u32>>, tree: Option<DDTreePayload>) -> Self {
        Self {
            proposal: Mutex::new(Some(proposal)),
            tree: Mutex::new(tree),
            proposals: AtomicUsize::new(0),
            takes: AtomicUsize::new(0),
        }
    }
}

pub(super) struct Frame {
    pub(super) state: SequenceState,
    pub(super) drafts: Vec<u32>,
    pub(super) tree: Option<DDTreePayload>,
}

impl SchedulerProposalFrame for Frame {
    type State = SequenceState;
    type Tree = DDTreePayload;

    fn position(&self) -> usize {
        self.state.seq_len
    }
    fn parts(
        &mut self,
    ) -> (
        &mut SequenceState,
        &mut Vec<u32>,
        &mut Option<DDTreePayload>,
    ) {
        (&mut self.state, &mut self.drafts, &mut self.tree)
    }
}

pub(super) fn tree(token: u32) -> DDTreePayload {
    DDTreePayload {
        tree_token_ids: vec![token],
        parent_indices: vec![-1],
    }
}

impl Model for TestModel {
    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        _: usize,
        _: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn vocab_size(&self) -> usize {
        0
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(seq_state())
    }
    fn copy_logits_to_host(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        unreachable!()
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        unreachable!()
    }
    fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        unreachable!()
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        unreachable!()
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!()
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn cache_sequence(&self, _: &SequenceState) {}
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, slot: usize) -> Result<()> {
        seq.slot_idx = slot;
        Ok(())
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        unreachable!()
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        unreachable!()
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        unreachable!()
    }
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        unreachable!()
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.proposals.fetch_add(1, Ordering::SeqCst);
        self.proposal.lock().unwrap().take().expect("one proposal")
    }
    fn take_pending_tree_payload(&self, _: &mut SequenceState) -> Option<DDTreePayload> {
        self.takes.fetch_add(1, Ordering::SeqCst);
        self.tree.lock().unwrap().take()
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
}
