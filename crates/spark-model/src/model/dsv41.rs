// SPDX-License-Identifier: AGPL-3.0-only

//! Served DeepSeek-V4.1: a standalone [`Model`] over [`V41Forward`], like the NLLB precedent
//! (`factory/build.rs` dispatches it before the decoder-only `TransformerLayer` stack).
//!
//! Why standalone: V4.1's residual is a 4-stream mHC state with a pre_mix carried between
//! blocks, the engram needs host token ids, and prefill is CED + SWA bounded replay (layers
//! 21..39 run once over the prompt tail). None of that fits the generic per-layer
//! `hidden/residual` loop without contorting it; here the validated forward runs as is.
//!
//! **One live sequence at a time.** The attention lane's compressed-KV caches are sized at
//! load from `max_seq` and are model-wide, so a second concurrent sequence would overwrite
//! the first's. `alloc_sequence` refuses a second live sequence rather than corrupting it.

use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::sync::Mutex;

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::deepseek_v41_engram::EngramHashState;
use crate::traits::{Model, SequenceState};
use crate::weight_loader::deepseek_v41::forward::{PassHook, PrefillMode, V41Forward, V41Seq};
use crate::weight_loader::deepseek_v41::fwd::{Tap, V41Dims, V41RoutedMoe};
use crate::weight_loader::deepseek_v41::attn_block::AttnCore;
use crate::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops};

/// The lanes' halves of the forward: the attention core (compress / index / sparse
/// attention) with its per-pass hook, and the routed MoE.
pub struct V41Lanes {
    pub hook: Box<dyn PassHook + Send + Sync>,
    pub core: Box<dyn AttnCore + Send + Sync>,
    pub moe: Box<dyn V41RoutedMoe + Send + Sync>,
}

/// Build the lanes' halves for serving.
///
/// NOT WIRED YET, deliberately a hard stop: the attention core must come from dsv41-attention2's
/// `Dsv41Attention` (compress / sparse_index_select / sparse_attention) and the routed MoE
/// from dsv41-engine2's library path. A stand-in for either would serve fluent wrong tokens.
pub fn build_lanes(
    _store: &WeightStore,
    _config: &ModelConfig,
    _gpu: &dyn GpuBackend,
    _max_seq: usize,
) -> Result<V41Lanes> {
    bail!(
        "DeepSeek-V4.1 serving: the attention core (OWNER dsv41-attention2) and the routed MoE \
         (OWNER dsv41-engine2) are not wired into model::dsv41::build_lanes yet. Everything else \
         — mHC, norms, shared expert, engram, attention projections, SWA-replay orchestration, \
         head — is in V41Forward. This is a deliberate hard stop, not a fallback."
    )
}

pub struct Dsv41Model {
    gpu: Box<dyn GpuBackend>,
    kernels: Dsv41Kernels,
    fwd: V41Forward,
    lanes: V41Lanes,
    model_dir: std::path::PathBuf,
    vocab: usize,
    mode: PrefillMode,
    /// bf16 `[vocab]` logits of the last prefill / decode.
    logits: DevicePtr,
    seqs: Mutex<HashMap<usize, V41Seq>>,
    next_slot: Mutex<usize>,
    tap: Tap,
}

// SAFETY-adjacent: every field is either immutable after construction, behind a Mutex, or a
// device pointer; the lanes' trait objects are Send + Sync by bound.
impl Dsv41Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &ModelConfig,
        store: &WeightStore,
        gpu: Box<dyn GpuBackend>,
        model_dir: &std::path::Path,
        lanes: V41Lanes,
        max_seq: usize,
        max_chunk: usize,
    ) -> Result<Self> {
        gpu.bind_to_thread()?;
        let dims = V41Dims::from_config(config)?;
        let kernels = Dsv41Kernels::load(gpu.as_ref())?;
        let stream = gpu.default_stream();
        let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream };
        let threads = std::env::var("ATLAS_DSV41_ENGRAM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let fwd = V41Forward::load(store, &ops, dims, config.vocab_size, config.num_hidden_layers, max_chunk, max_seq, model_dir, threads)?;
        let mode = match std::env::var("ATLAS_DSV41_PREFILL").ok().as_deref() {
            None | Some("replay") => PrefillMode::Replay,
            Some("full") => {
                tracing::warn!("DeepSeek-V4.1: ATLAS_DSV41_PREFILL=full is a DEBUG mode (no SWA replay)");
                PrefillMode::Full
            }
            Some(o) => bail!("ATLAS_DSV41_PREFILL={o}: use replay (default) or full"),
        };
        // MM_TILE rows: the head GEMM runs as one 16-row tile; row 0 is the result.
        let logits = gpu.alloc(crate::weight_loader::deepseek_v41::ops::MM_TILE * config.vocab_size * 2)?;
        let tap = match std::env::var("ATLAS_DSV41_TAP_DIR") {
            Ok(d) => Tap::to_dir(d.into(), Vec::new())?,
            Err(_) => Tap::off(),
        };
        tracing::info!("DeepSeek-V4.1 served model ready: max_seq {max_seq}, chunk {max_chunk}, prefill {mode:?}");
        Ok(Self {
            gpu,
            kernels,
            fwd,
            lanes,
            model_dir: model_dir.to_path_buf(),
            vocab: config.vocab_size,
            mode,
            logits,
            seqs: Mutex::new(HashMap::new()),
            next_slot: Mutex::new(0),
            tap,
        })
    }

    fn ops(&self) -> Ops<'_> {
        Ops { gpu: self.gpu.as_ref(), k: &self.kernels, stream: self.gpu.default_stream() }
    }

    fn with_seq<R>(&self, slot: usize, f: impl FnOnce(&mut V41Seq) -> Result<R>) -> Result<R> {
        let mut map = self.seqs.lock().expect("dsv41 seqs poisoned");
        let seq = map.get_mut(&slot).context("dsv41: no state for slot (alloc_sequence not called?)")?;
        f(seq)
    }

    fn argmax_host(&self, logits: DevicePtr) -> Result<u32> {
        let mut host = vec![0u8; self.vocab * 2];
        self.gpu.synchronize(self.gpu.default_stream())?;
        self.gpu.copy_d2h(logits, &mut host)?;
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, c) in host.chunks_exact(2).enumerate() {
            let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        Ok(best.0)
    }

    fn prefill_all(&self, tokens: &[u32], seq: &mut SequenceState, start: usize, last: bool) -> Result<DevicePtr> {
        let ops = self.ops();
        let (l, m) = (&self.lanes, self.mode);
        self.with_seq(seq.slot_idx, |s| {
            ensure!(s.len == start, "dsv41 prefill: sequence holds {} positions, chunk starts at {start}", s.len);
            for chunk in tokens.chunks(self.fwd.max_chunk) {
                self.fwd.prefill_chunk(&ops, s, chunk, m, l.hook.as_ref(), l.core.as_ref(), l.moe.as_ref(), &self.tap)?;
            }
            if last {
                self.fwd.finish_prefill(&ops, s, m, l.hook.as_ref(), l.core.as_ref(), l.moe.as_ref(), &self.tap, self.logits)?;
            }
            Ok(())
        })?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = start + tokens.len();
        if last {
            seq.prompt_len = seq.seq_len;
        }
        Ok(self.logits)
    }
}

impl Model for Dsv41Model {
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        self.prefill_all(tokens, seq, 0, true)
    }

    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        _stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(chunk_start + chunk_len <= tokens.len(), "dsv41 prefill_chunk: range past the prompt");
        self.prefill_all(&tokens[chunk_start..chunk_start + chunk_len], seq, chunk_start, is_last_chunk)
    }

    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        let ops = self.ops();
        let l = &self.lanes;
        self.with_seq(seq.slot_idx, |s| {
            self.fwd.decode(&ops, s, token, l.hook.as_ref(), l.core.as_ref(), l.moe.as_ref(), &self.tap, self.logits)
        })?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.logits)
    }

    fn decode_batch(&self, tokens: &[u32], seqs: &mut [&mut SequenceState], stream: u64) -> Result<DevicePtr> {
        ensure!(tokens.len() == 1 && seqs.len() == 1, "dsv41: batch size is 1 (got {})", seqs.len());
        self.decode(tokens[0], seqs[0], stream)
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    fn alloc_sequence(&self) -> Result<SequenceState> {
        let mut map = self.seqs.lock().expect("dsv41 seqs poisoned");
        ensure!(
            map.is_empty(),
            "dsv41: one live sequence at a time (the attention lane's compressed-KV caches are \
             model-wide); {} still allocated",
            map.len()
        );
        let hash = EngramHashState::for_checkpoint(&self.model_dir)?;
        let slot = {
            let mut n = self.next_slot.lock().expect("slot poisoned");
            *n += 1;
            *n
        };
        map.insert(slot, V41Seq::new(self.gpu.as_ref(), &self.fwd.dims, hash)?);
        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,
            acquired_adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states: Vec::new(),
            proposer_state: None,
            slot_idx: slot,
            ssm_slot: None,
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: Vec::new(),
            block_fork: None,
            block_fork_scratch: Vec::new(),
            tree_payload: None,
            tree_branch_scratch: Vec::new(),
            tree_scratch_pool: Vec::new(),
            tree_scratch_persistent: false,
        })
    }

    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        if seq.slot_idx == usize::MAX {
            return Ok(());
        }
        if let Some(s) = self.seqs.lock().expect("dsv41 seqs poisoned").remove(&seq.slot_idx) {
            s.free(self.gpu.as_ref())?;
        }
        Ok(())
    }

    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        let mut map = self.seqs.lock().expect("dsv41 seqs poisoned");
        if let Some(s) = map.remove(&seq.slot_idx) {
            map.insert(new_slot, s);
        }
        seq.slot_idx = new_slot;
        Ok(())
    }

    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        seq.slot_idx = usize::MAX;
    }

    fn cache_sequence(&self, _seq: &SequenceState) {}

    fn num_free_blocks(&self) -> usize {
        1 << 20
    }

    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits
    }

    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        self.argmax_host(logits_ptr)
    }

    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        (0..n).map(|i| self.argmax_host(logits_ptr.offset(i * self.vocab * 2))).collect()
    }

    // ── not applicable yet: no speculation (MTP unported), no SSM ──

    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr(0)
    }

    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        bail!("dsv41: speculative verify not supported yet (MTP is unported)")
    }

    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }

    fn generate_speculative(
        &self,
        _tokens: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        bail!("dsv41: speculative decoding not supported yet")
    }

    fn has_proposer(&self) -> bool {
        false
    }

    fn has_self_speculative(&self) -> bool {
        false
    }

    fn decode_draft(&self, _token: u32, _seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        bail!("dsv41: self-speculative draft not supported")
    }

    fn decode_verify_graphed(&self, _t: &[u32; 2], _s: &mut SequenceState, _st: u64) -> Result<[u32; 2]> {
        bail!("dsv41: graphed verify not supported")
    }

    fn decode_verify_graphed_k3(&self, _t: &[u32; 3], _s: &mut SequenceState, _st: u64) -> Result<[u32; 3]> {
        bail!("dsv41: graphed verify not supported")
    }

    fn decode_verify_graphed_k4(&self, _t: &[u32; 4], _s: &mut SequenceState, _st: u64) -> Result<[u32; 4]> {
        bail!("dsv41: graphed verify not supported")
    }

    fn save_hidden_for_mtp(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn run_mtp_propose(&self, _token: u32, _position: usize, _seq: &mut SequenceState, _stream: u64) -> Result<Option<u32>> {
        Ok(None)
    }

    fn run_mtp_propose_multi(
        &self,
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _seq: &mut SequenceState,
        _stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn trim_proposer_state(&self, _seq: &mut SequenceState, _num_accepted: usize, _stream: u64) -> Result<()> {
        Ok(())
    }
}
