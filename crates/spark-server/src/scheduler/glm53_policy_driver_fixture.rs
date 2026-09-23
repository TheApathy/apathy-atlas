// SPDX-License-Identifier: AGPL-3.0-only

//! Model transport/ownership boundary only. Acceptance, policy, publication and
//! ordinary transitions use the production implementations, never fake picks.

use super::boundary::{BASE, Boundary, VOCAB};
use super::fixture::MODEL;
use crate::scheduler::{Model, SequenceState, StreamEvent};
use anyhow::{Result, bail, ensure};
use spark_model::model::glm53::verify_policy_binding::{
    BoundVerifyRequest, VerifyBindingOwner, VerifyFrame,
};
use spark_model::model::glm53::verify_policy_transaction::{
    LogitsIo, VerifyCommitIo, VerifyOutcome, VerifyPolicy, VerifyRequest,
    run_verify_policy_transaction,
};
use spark_runtime::gpu::DevicePtr;
use std::collections::VecDeque;
use std::sync::Mutex;

pub(super) const STREAM: u64 = 37;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Fault {
    None,
    Bind,
    Copy,
    ShortCopy,
    Nan,
    Abort,
    Commit,
    WrongEnd,
    WrongPrefix,
    Propose,
    OrdinaryDecode,
    OrdinaryCopy,
}

pub(super) struct DriverModel {
    pub wide: Boundary,
    pub ordinary: Boundary,
    pub fault: Fault,
    pub policy_required: bool,
    owner: Mutex<VerifyBindingOwner>,
    frame: Mutex<VerifyFrame>,
    prefix: Mutex<Vec<u32>>,
    proposals: Mutex<VecDeque<Vec<u32>>>,
    calls: Mutex<Vec<&'static str>>,
    inputs: Mutex<Vec<Vec<u32>>>,
    receiver: Mutex<Option<tokio::sync::mpsc::Receiver<StreamEvent>>>,
}

impl DriverModel {
    pub fn new(rows: &[[f32; VOCAB]], drafts: Vec<u32>) -> Self {
        Self {
            wide: Boundary::new(rows),
            ordinary: Boundary::new(&[[0.0, 10.0, 9.0, -1.0, -2.0]]),
            fault: Fault::None,
            policy_required: true,
            owner: Mutex::new(VerifyBindingOwner::new()),
            frame: Mutex::new(VerifyFrame {
                generation: 1,
                nonce: 0,
                position: 2,
                capacity: 64,
                vocab: VOCAB,
                stream: STREAM,
            }),
            prefix: Mutex::new(vec![0, 0]),
            proposals: Mutex::new(VecDeque::from([drafts])),
            calls: Mutex::new(Vec::new()),
            inputs: Mutex::new(Vec::new()),
            receiver: Mutex::new(None),
        }
    }

    pub fn stream(&self) -> crate::scheduler::ResponseSink {
        let (sender, receiver) = tokio::sync::mpsc::channel(32);
        *self.receiver.lock().unwrap() = Some(receiver);
        crate::scheduler::ResponseSink::Streaming(sender)
    }

    pub fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().clone()
    }
    pub fn inputs(&self) -> Vec<Vec<u32>> {
        self.inputs.lock().unwrap().clone()
    }
    pub fn position(&self) -> usize {
        self.frame.lock().unwrap().position
    }
    pub fn prefix(&self) -> Vec<u32> {
        self.prefix.lock().unwrap().clone()
    }
    pub fn count(&self, call: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|&&item| item == call)
            .count()
    }
    fn record(&self, call: &'static str) {
        self.calls.lock().unwrap().push(call);
    }

    pub fn emitted(&self) -> Vec<u32> {
        let mut receiver = self.receiver.lock().unwrap();
        let Some(receiver) = receiver.as_mut() else {
            return Vec::new();
        };
        let mut tokens = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            match event {
                StreamEvent::Token(token) | StreamEvent::TokenWithLogprobs(token, _) => {
                    tokens.push(token)
                }
                _ => panic!("driver cannot emit terminal/error frames owned by retirement"),
            }
        }
        tokens
    }

    fn require_quiet_stream(&self) {
        if let Some(receiver) = self.receiver.lock().unwrap().as_mut() {
            assert!(
                receiver.try_recv().is_err(),
                "event escaped before commit/restore"
            );
        }
    }

    fn validate_host(&self, seq: &SequenceState) -> Result<()> {
        ensure!(seq.slot_idx == 0, "unexpected slot");
        ensure!(
            seq.tokens == self.prefix(),
            "fixture host/model prefix divergence"
        );
        ensure!(
            seq.seq_len == self.position() && seq.kv_valid_tokens == seq.seq_len,
            "fixture host/model extent divergence"
        );
        Ok(())
    }
}

struct WideSource<'a>(&'a DriverModel);
impl LogitsIo for WideSource<'_> {
    fn copy_logits(&mut self, destination: &mut [u8]) -> Result<usize> {
        self.0.record("copy");
        self.0.require_quiet_stream();
        if matches!(self.0.fault, Fault::Copy | Fault::Abort) {
            bail!("injected full-row copy failure");
        }
        ensure!(
            destination.len() == self.0.wide.bytes.len(),
            "not the full distinct-row extent"
        );
        destination.copy_from_slice(&self.0.wide.bytes);
        if self.0.fault == Fault::Nan {
            destination[..2].copy_from_slice(&0x7fc1u16.to_le_bytes());
        }
        Ok(destination.len()
            - if self.0.fault == Fault::ShortCopy {
                2
            } else {
                0
            })
    }
}

struct Target<'a>(&'a DriverModel);
impl VerifyCommitIo for Target<'_> {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        self.0.require_quiet_stream();
        self.0.record("commit");
        if self.0.fault == Fault::Commit {
            bail!("injected irreversible commit failure");
        }
        ensure!(self.0.position() == request.start(), "commit start drift");
        self.0
            .prefix
            .lock()
            .unwrap()
            .extend_from_slice(&request.inputs()[..rows]);
        self.0.frame.lock().unwrap().position = request.start() + rows;
        Ok(self.0.position() + usize::from(self.0.fault == Fault::WrongEnd))
    }
    fn abort_staged(&mut self) -> Result<()> {
        self.0.require_quiet_stream();
        self.0.record("abort");
        ensure!(self.0.fault != Fault::Abort, "injected DSA restore failure");
        Ok(())
    }
    fn poison(&mut self) {
        self.0.record("poison");
    }
}

macro_rules! forbidden_gpu_methods {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        $(fn $name(&self, $($arg: $ty),*) -> $ret { MODEL.$name($($arg),*) })*
    };
}

impl Model for DriverModel {
    fn requires_verify_policy(&self) -> bool {
        self.policy_required
    }
    fn default_stream(&self) -> u64 {
        STREAM
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn sync_secondary(&self) -> Result<()> {
        self.record("sync");
        Ok(())
    }
    fn poison_verify_policy(&self, stream: u64) {
        assert_eq!(stream, STREAM);
        self.record("poison");
    }
    fn bind_verify_policy_request(
        &self,
        inputs: &[u32],
        seq: &SequenceState,
        stream: u64,
    ) -> Result<BoundVerifyRequest> {
        self.record("bind");
        ensure!(self.fault != Fault::Bind, "injected bind failure");
        ensure!(stream == STREAM, "driver discarded model stream");
        self.validate_host(seq)?;
        let mut frame = self.frame.lock().unwrap();
        frame.nonce += 1;
        let mut prefix = seq.tokens.clone();
        if self.fault == Fault::WrongPrefix {
            prefix[0] = 4;
        }
        self.owner.lock().unwrap().bind(*frame, &prefix, inputs)
    }
    fn decode_verify_with_policy(
        &self,
        bound: BoundVerifyRequest,
        policy: &mut dyn VerifyPolicy,
    ) -> Result<VerifyOutcome> {
        let current = *self.frame.lock().unwrap();
        let request = self.owner.lock().unwrap().consume(bound, current)?;
        self.frame.lock().unwrap().nonce += 1;
        self.record("verify");
        self.inputs.lock().unwrap().push(request.inputs().to_vec());
        run_verify_policy_transaction(&request, &mut WideSource(self), policy, &mut Target(self))
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        position: usize,
        drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.record("propose");
        ensure!(self.fault != Fault::Propose, "injected proposal failure");
        ensure!(
            stream == STREAM && position == self.position(),
            "proposal position/stream mismatch"
        );
        self.validate_host(seq)?;
        let mut result = self
            .proposals
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default();
        result.truncate(drafts);
        Ok(result)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            tokens.len() == 1 && seqs.len() == 1,
            "GLM ordinary replay must be C1"
        );
        self.decode(tokens[0], seqs[0], stream)
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.record("decode");
        ensure!(
            self.fault != Fault::OrdinaryDecode,
            "injected ordinary decode failure"
        );
        ensure!(stream == STREAM, "ordinary driver discarded model stream");
        self.validate_host(seq)?;
        self.prefix.lock().unwrap().push(token);
        self.frame.lock().unwrap().position += 1;
        seq.tokens.push(token);
        seq.seq_len += 1;
        seq.kv_valid_tokens = seq.seq_len;
        Ok(BASE)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        BASE
    }
    fn argmax_batch(&self, ptr: DevicePtr, rows: usize, stream: u64) -> Result<Vec<u32>> {
        ensure!(stream == STREAM || stream == 0, "unexpected sampler stream");
        self.ordinary.argmax_batch(ptr, rows, stream)
    }
    fn argmax_on_device(&self, ptr: DevicePtr, stream: u64) -> Result<u32> {
        self.ordinary.argmax_on_device(ptr, stream)
    }
    fn copy_logits_to_host(&self, ptr: DevicePtr, destination: &mut [u8]) -> Result<()> {
        self.record("ordinary_copy");
        ensure!(
            self.fault != Fault::OrdinaryCopy,
            "injected ordinary copy failure"
        );
        self.ordinary.copy_logits_to_host(ptr, destination)
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        panic!("legacy verify entered")
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        panic!("legacy verify entered")
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        panic!("legacy verify entered")
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        panic!("legacy verify entered")
    }
    forbidden_gpu_methods! {
        fn prefill(tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn prefill_chunk(tokens: &[u32], seq: &mut SequenceState, start: usize, len: usize, last: bool, stream: u64) -> Result<DevicePtr>;
        fn bind_gpu_to_thread() -> Result<()>;
        fn alloc_sequence() -> Result<SequenceState>;
        fn hidden_after_norm() -> DevicePtr;
        fn checkpoint_ssm_states(seq: &mut SequenceState) -> Result<()>;
        fn rollback_ssm_states(seq: &mut SequenceState, accepted: usize) -> Result<()>;
        fn generate_speculative(tokens: &[u32], params: &spark_runtime::sampler::SamplingParams, drafts: usize) -> Result<spark_model::engine::GenerateResult>;
        fn decode_draft(token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;
        fn cache_sequence(seq: &SequenceState) -> ();
        fn free_sequence(seq: &mut SequenceState) -> Result<()>;
        fn compact_sequence(seq: &mut SequenceState, slot: usize) -> Result<()>;
        fn detach_slot_for_reuse(seq: &mut SequenceState) -> ();
        fn save_hidden_for_mtp(index: usize, stream: u64) -> Result<()>;
        fn run_mtp_propose(token: u32, position: usize, seq: &mut SequenceState, stream: u64) -> Result<Option<u32>>;
        fn trim_proposer_state(seq: &mut SequenceState, accepted: usize, stream: u64) -> Result<()>;
    }
}
