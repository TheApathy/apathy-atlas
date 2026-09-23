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
use crate::weight_loader::deepseek_v41::device_allocs::{DeviceAllocs, SharedGpu};
use crate::weight_loader::deepseek_v41::dspark::{Dspark, T_VERIFY};
use crate::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops};

/// `swiglu_limit` and `routed_scaling_factor` of the checkpoint's text_config, shared by the
/// main routed MoE and the DSpark drafter's MoE.
const MOE_SWIGLU_LIMIT: f32 = 10.0;
const MOE_ROUTE_SCALE: f32 = 1.5;

/// ATLAS_LOG_PINNED_ALGO=1: print the stream and every ATLAS_*/DSV41_* variable, so a serve
/// process and a driver process can be diffed for the environment they actually ran with.
pub fn log_run_identity(who: &str, stream: u64) {
    if std::env::var("ATLAS_LOG_PINNED_ALGO").as_deref() != Ok("1") {
        return;
    }
    eprintln!("RUN_IDENTITY {who} stream={stream:#x}");
    let mut vars: Vec<(String, String)> =
        std::env::vars().filter(|(k, _)| k.starts_with("ATLAS_") || k.starts_with("DSV41_")).collect();
    vars.sort();
    for (k, v) in vars {
        eprintln!("RUN_ENV {who} {k}={v}");
    }
}

/// The lanes' halves of the forward: the attention core (compress / index / sparse
/// attention) with its per-pass hook, and the routed MoE.
pub struct V41Lanes {
    pub hook: Box<dyn PassHook + Send + Sync>,
    pub core: Box<dyn AttnCore + Send + Sync>,
    pub moe: Box<dyn V41RoutedMoe + Send + Sync>,
}

/// The lanes' halves: attention2's sparse core (one object = core + hook, state model-wide)
/// and engine2's routed MoE over the FULL keep=124 arena.
#[allow(clippy::too_many_arguments)]
fn build_lanes(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    kernels: &Dsv41Kernels,
    shared: &crate::weight_loader::deepseek_v41::device_allocs::SharedGpu,
    fwd: &V41Forward,
    model_dir: &std::path::Path,
    max_seq: usize,
    max_chunk: usize,
) -> Result<V41Lanes> {
    use crate::layers::deepseek_v41_attn::core::Dsv41SparseCore;
    use crate::weight_loader::deepseek_v41::cb3_arena::{Cb3ExpertArena, resolve_packed_keep};
    use crate::weight_loader::deepseek_v41::moe_forward::{Cb3RoutedMoe, RouterF32};
    let core = std::sync::Arc::new(Dsv41SparseCore::load(shared, store, config, max_seq, max_chunk, fwd.freqs_c)?);
    let pack_dir = model_dir.join("k154-cb3");
    let manifest = std::fs::read_to_string(pack_dir.join("manifest.json"))
        .with_context(|| format!("DeepSeek-V4.1 expert pack manifest at {}", pack_dir.display()))?;
    let pack = atlas_core::config::ExpertPack::parse(&manifest, resolve_packed_keep()?)?;
    // The arena and the MoE own an Arc to the backend and FREE their memory on drop.
    let arena = std::sync::Arc::new(Cb3ExpertArena::load(&pack_dir, &pack, shared)?);
    tracing::info!("DeepSeek-V4.1: {:.2} GB of CB3 experts resident (keep={})", arena.resident_bytes() as f64 / 1e9, arena.packed_keep());
    let stream = gpu.default_stream();
    let routers = (0..config.num_hidden_layers)
        .map(|l| Ok((l, RouterF32::load(store, l, config.hidden_size, gpu, kernels, stream)?)))
        .collect::<Result<Vec<_>>>()?;
    let moe = Cb3RoutedMoe::new(shared.clone(), *kernels, config, arena, routers, MOE_SWIGLU_LIMIT, MOE_ROUTE_SCALE, max_chunk.max(128))?;
    Ok(V41Lanes { hook: Box::new(core.clone()), core: Box::new(core), moe: Box::new(moe) })
}

pub struct Dsv41Model {
    /// Owned backend handle. Every owner of device memory below holds its OWN clone, so the
    /// backend outlives all of them whatever the field drop order.
    gpu: SharedGpu,
    kernels: Dsv41Kernels,
    fwd: V41Forward,
    lanes: V41Lanes,
    model_dir: std::path::PathBuf,
    vocab: usize,
    mode: PrefillMode,
    /// The model's own allocations (the logits), freed on drop.
    own: DeviceAllocs,
    /// bf16 `[vocab]` logits of the last prefill / decode.
    logits: DevicePtr,
    seqs: Mutex<HashMap<usize, V41Seq>>,
    next_slot: Mutex<usize>,
    tap: Tap,
    /// The DSpark greedy speculator (drafter loaded): `decode_multi` runs its steps. Behind a
    /// Mutex because its drafter MoE keeps interior scratch state.
    dspark: Option<Mutex<Dspark>>,
    /// Per-sequence DSpark counters (steps, accepted drafts), logged when the sequence is freed.
    spec_stats: Mutex<(usize, usize)>,
    /// The input token and drafts of an uncommitted `spec_verify`.
    spec_pending: Mutex<Option<(u32, Vec<u32>)>>,
    /// Whether the last spec_verify drew SAMPLED drafts (its q rows are valid).
    spec_sampled: Mutex<bool>,
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
        max_seq: usize,
        max_chunk: usize,
    ) -> Result<Self> {
        gpu.bind_to_thread()?;
        // Nothing is leaked: every owner of device memory (the forward's scratch/tables, each
        // sequence's rings, the logits, the MoE, the arena) holds an Arc to the backend and frees
        // on drop, so dropping the model returns its device memory (TUI model swap).
        let shared: SharedGpu = std::sync::Arc::from(gpu);
        let gpu: &dyn GpuBackend = shared.as_ref();
        let dims = V41Dims::from_config(config)?;
        let kernels_owned = Dsv41Kernels::load(gpu)?;
        let kernels = &kernels_owned;
        let stream = gpu.default_stream();
        let ops = Ops { gpu, k: kernels, stream };
        log_run_identity("serve", stream);
        let threads = std::env::var("ATLAS_DSV41_ENGRAM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let mut fwd = V41Forward::load(store, &ops, dims, config.vocab_size, config.num_hidden_layers, max_chunk, max_seq, model_dir, threads)?;
        if crate::weight_loader::deepseek_v41::dspark_enabled() {
            // DSpark drafter (ATLAS_DSV41_DSPARK=1): its tensors are only in the store then.
            let bytes = crate::weight_loader::deepseek_v41::mtp::mtp_store_bytes(store);
            fwd.dspark = Some(crate::weight_loader::deepseek_v41::mtp::DsparkWeights::load(store, &dims, config.vocab_size)?);
            fwd.enable_dspark_seed(gpu)?;
            tracing::info!("DeepSeek-V4.1 DSpark drafter loaded: {:.2} GB of mtp.* in the store (+ seed buffer)", bytes as f64 / 1e9);
        }
        fwd.own_allocations(shared.clone());
        fwd.vision = load_vision(store, model_dir, dims.hidden, &shared)?;
        let lanes = build_lanes(store, config, gpu, kernels, &shared, &fwd, model_dir, max_seq, max_chunk)?;
        let mode = match std::env::var("ATLAS_DSV41_PREFILL").ok().as_deref() {
            None | Some("replay") => PrefillMode::Replay,
            Some("full") => {
                tracing::warn!("DeepSeek-V4.1: ATLAS_DSV41_PREFILL=full is a DEBUG mode (no SWA replay)");
                PrefillMode::Full
            }
            Some(o) => bail!("ATLAS_DSV41_PREFILL={o}: use replay (default) or full"),
        };
        // MM_TILE rows: the head GEMM runs as one 16-row tile; row 0 is the result.
        let mut own = DeviceAllocs::owned(shared.clone());
        let logits = own.alloc(gpu, crate::weight_loader::deepseek_v41::ops::MM_TILE * config.vocab_size * 2)?;
        let tap = match std::env::var("ATLAS_DSV41_TAP_DIR") {
            Ok(d) => Tap::to_dir(d.into(), Vec::new())?,
            Err(_) => Tap::off(),
        };
        let dspark = if fwd.dspark.is_some() {
            Some(Mutex::new(Dspark::new(&shared, &fwd, MOE_SWIGLU_LIMIT, MOE_ROUTE_SCALE)?))
        } else {
            None
        };
        tracing::info!(
            "DeepSeek-V4.1 served model ready: max_seq {max_seq}, chunk {max_chunk}, prefill {mode:?}, DSpark speculation {}",
            if dspark.is_some() { "ON (greedy steps)" } else { "off" }
        );
        Ok(Self {
            gpu: shared.clone(),
            kernels: kernels_owned,
            own,
            fwd,
            lanes,
            model_dir: model_dir.to_path_buf(),
            vocab: config.vocab_size,
            mode,
            logits,
            seqs: Mutex::new(HashMap::new()),
            next_slot: Mutex::new(0),
            tap,
            dspark,
            spec_stats: Mutex::new((0, 0)),
            spec_pending: Mutex::new(None),
            spec_sampled: Mutex::new(false),
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
                // The drafter ring from the replay rows (the last tail_rows prompt positions).
                if let Some(ds) = &self.dspark {
                    let rows = s.tail_rows;
                    ds.lock().expect("dspark poisoned").seed(&ops, &self.fwd, rows, s.len - rows)?;
                }
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

/// The V4.1 vision tower, when the checkpoint has one (`vision.*` tensors and a
/// `vision_config`). None for a text-only checkpoint.
fn load_vision(
    store: &WeightStore,
    model_dir: &std::path::Path,
    hidden: usize,
    shared: &SharedGpu,
) -> Result<Option<crate::weight_loader::deepseek_v41::image_splice::V41ImageSplice>> {
    let gpu = shared.as_ref();
    if store.get("vision.patch_embed.proj.weight").is_err() {
        return Ok(None);
    }
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let v = &raw["vision_config"];
    let u = |k: &str| v[k].as_u64().map(|x| x as usize).with_context(|| format!("vision_config.{k}"));
    let enc = crate::layers::deepseek_vision::DeepSeekVisionEncoder::load_v41(
        store,
        u("hidden_size")?,
        u("intermediate_size")?,
        u("num_attention_heads")?,
        u("num_hidden_layers")?,
        u("patch_size")?,
        u("downsample_ratio")?,
        v["rope_theta"].as_f64().context("vision_config.rope_theta")?,
        u("max_image_tokens")?,
        hidden,
        gpu,
        Some(shared.clone()),
    )?;
    // Memory: the tower's weights (~0.97 GB) are already in the store; this is the
    // encoder scratch arena on top (sized for max_image_tokens, allocated here once).
    tracing::info!(
        "DeepSeek-V4.1 vision tower loaded (image input enabled): encoder scratch {:.2} GB \
         + ~0.97 GB of vision/aligner weights in the store",
        enc.scratch_bytes()? as f64 / 1e9
    );
    Ok(Some(crate::weight_loader::deepseek_v41::image_splice::V41ImageSplice::new(enc, hidden, Some(shared.clone()))))
}

impl Drop for Dsv41Model {
    /// Sequences still allocated when the model goes (a TUI swap mid-request) are freed here;
    /// everything else frees through its own `DeviceAllocs`.
    fn drop(&mut self) {
        if let Ok(mut map) = self.seqs.lock() {
            for (_, s) in map.drain() {
                if let Err(e) = s.free(self.gpu.as_ref()) {
                    tracing::warn!("Dsv41Model drop: freeing a live sequence failed: {e}");
                }
            }
        }
    }
}

impl Model for Dsv41Model {
    fn prepare_vision_embed(&self, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        match &self.fwd.vision {
            Some(v) => v.encode(self.gpu.as_ref(), images),
            None if images.is_empty() => Ok(()),
            None => bail!("image input, but this DeepSeek-V4.1 model has no vision tower loaded"),
        }
    }

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
            self.fwd.decode(&ops, s, token, l.hook.as_ref(), l.core.as_ref(), l.moe.as_ref(), &self.tap, self.logits)?;
            // A plain step (non-greedy request, or near max_seq) still feeds the drafter ring,
            // so a later speculative step drafts from current context.
            if let Some(ds) = &self.dspark {
                ds.lock().expect("dspark poisoned").seed(&ops, &self.fwd, 1, s.len - 1)?;
            }
            Ok(())
        })?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.logits)
    }

    fn has_internal_spec(&self) -> bool {
        self.dspark.is_some()
    }

    fn spec_verify(&self, token: u32, seq: &mut SequenceState, sampling: Option<&crate::traits::SpecSampling>, _stream: u64) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        let ds = self.dspark.as_ref().context("dsv41 spec_verify: DSpark is not loaded")?;
        let ops = self.ops();
        let l = &self.lanes;
        let r = self.with_seq(seq.slot_idx, |s| {
            ensure!(s.len == seq.seq_len, "dsv41 spec_verify: model holds {} positions, sequence {}", s.len, seq.seq_len);
            if s.len + T_VERIFY > self.fwd.max_seq {
                // The verify pass writes T_VERIFY positions.
                return Ok(None);
            }
            let ds = ds.lock().expect("dspark poisoned");
            // Sampled drafts (T > 0): q_i = softmax(markov_logits_i / T) at the caller's uniforms.
            let draft_sampling = match sampling {
                Some(sp) => {
                    ensure!(sp.draft_uniforms.len() == crate::weight_loader::deepseek_v41::dspark::B, "dsv41 spec_verify: {} draft uniforms for {} drafts", sp.draft_uniforms.len(), crate::weight_loader::deepseek_v41::dspark::B);
                    let mut u = [0f32; crate::weight_loader::deepseek_v41::dspark::B];
                    u.copy_from_slice(&sp.draft_uniforms);
                    Some((sp.temperature, u))
                }
                None => None,
            };
            *self.spec_sampled.lock().expect("dsv41 spec sampled poisoned") = draft_sampling.as_ref().is_some_and(|(t, _)| *t > 0.0);
            ds.set_draft_sampling(draft_sampling);
            let (drafts, _a, am) = ds.propose_verify(&ops, &self.fwd, s, token, l.hook.as_ref(), l.core.as_ref(), l.moe.as_ref(), &self.tap, self.logits)?;
            Ok(Some((drafts, am)))
        })?;
        if let Some((drafts, _)) = &r {
            *self.spec_pending.lock().expect("dsv41 spec pending poisoned") = Some((token, drafts.clone()));
        }
        Ok(r)
    }

    fn spec_draft_probs(&self) -> Option<DevicePtr> {
        let sampled = *self.spec_sampled.lock().expect("dsv41 spec sampled poisoned");
        match (&self.dspark, sampled) {
            (Some(ds), true) => Some(ds.lock().expect("dspark poisoned").draft_probs()),
            _ => None,
        }
    }

    fn spec_commit(&self, seq: &mut SequenceState, accepted: usize, _stream: u64) -> Result<()> {
        let ds = self.dspark.as_ref().context("dsv41 spec_commit: DSpark is not loaded")?;
        let (token, drafts) = self.spec_pending.lock().expect("dsv41 spec pending poisoned").take().context("dsv41 spec_commit without spec_verify")?;
        ensure!(accepted <= drafts.len(), "dsv41 spec_commit: {accepted} > {} drafts", drafts.len());
        let ops = self.ops();
        let pos = seq.seq_len;
        self.with_seq(seq.slot_idx, |s| {
            ds.lock().expect("dspark poisoned").commit(&ops, &self.fwd, s, pos, accepted, self.lanes.hook.as_ref())
        })?;
        {
            let mut st = self.spec_stats.lock().expect("dsv41 spec stats poisoned");
            st.0 += 1;
            st.1 += accepted;
        }
        seq.tokens.push(token);
        seq.tokens.extend_from_slice(&drafts[..accepted]);
        seq.seq_len += accepted + 1;
        Ok(())
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
        let (steps, acc) = std::mem::take(&mut *self.spec_stats.lock().expect("dsv41 spec stats poisoned"));
        if steps > 0 {
            tracing::info!("DSpark: {steps} speculative steps, {acc} drafts accepted ({:.2} tokens/step)", (steps + acc) as f64 / steps as f64);
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

    // Refused, not no-ops: a generic speculation caller must never silently skip the restore of
    // the rings, compressor pending state and engram hash. DSpark rolls back through
    // `V41Forward::rollback` directly.
    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        anyhow::bail!("dsv41: speculation state rollback goes through V41Forward::rollback (DSpark), not the generic SSM hooks")
    }

    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _n: usize) -> Result<()> {
        anyhow::bail!("dsv41: speculation state rollback goes through V41Forward::rollback (DSpark), not the generic SSM hooks")
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
