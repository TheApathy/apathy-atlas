// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 model forward: embedding -> 40 blocks -> head, in the reference's
//! prefill shape, **CED + Decoder SWA Bounded Replay** (lead ruling; tech report §2.2/§3.2.2,
//! `engine/model.py::forward(encoder_only=True)` + `decoder_replay`):
//!
//! ```text
//! prefill, per chunk:  layers 0..=20 over the chunk      (the ENCODER: everything that
//!                                                         writes global KV)
//! prefill, once:       layers 21..=39 over the LAST min(128, P) prompt tokens, window
//!                      truncated to that segment (win_lo = P - n), reusing L20's
//!                      top-k / candidate rows for those tokens            (the REPLAY)
//! decode:              all 40 layers, one token
//! ```
//!
//! The replay is exact for the prompt's last row and is what the production engine serves;
//! a full-prompt decoder is a debug mode only (`PrefillMode::Full`).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::attn_block::{self, AttnCore, AttnScratch, HEAD_DIM, RING, V41AttnWeights, WINDOW, compress_ratio};
use super::fwd::{BlockControl, PassScratch, Tap, V41AttentionBlock, V41BlockWeights, V41Dims, V41RoutedMoe, block};
use super::ops::{Ops, RopeSpec, RopeTable, bf16_tensor, bytemuck_u32, prof};
use crate::layers::deepseek_v41_engram::{
    EngramGather, EngramHashState, N_HEAD_COLS, engram_dead_heads, engram_dead_heads_with_carry,
    update_dead_carry,
};

/// Set to dump `engram_in` (the block's `h` going INTO the engram projection),
/// `engram_rows_masked` (the post-mask, cast-to-bf16 rows the wkv linear
/// actually reads) alongside the existing `engram_out` tap, and to log the
/// masked-head count per chunk from the EXECUTING path (host-side, from the
/// same buffer that gets uploaded to the device -- not a separate recompute).
/// Off by default: the extra taps cost a D2H copy nobody wants paying for on
/// every run.
pub const ENGRAM_DEBUG_ENV: &str = "ATLAS_DSV41_ENGRAM_DEBUG";

/// Legacy policy only: prompts at least this long prefill in `max_chunk` pieces; shorter ones in
/// [`SHORT_CHUNK`].
pub const LONG_PROMPT: usize = 6144;
/// Legacy policy only: the chunk for prompts under [`LONG_PROMPT`] tokens.
pub const SHORT_CHUNK: usize = 2048;
/// Balanced chunks are rounded up to a multiple of this.
pub const CHUNK_ALIGN: usize = 128;
/// A/B controls, read per prefill: `legacy` = the old [`LONG_PROMPT`]/[`SHORT_CHUNK`] rule, a
/// number N = fixed chunks of min(N, max_chunk). Unset or `balanced` = [`balanced_chunk_len`].
pub const CHUNK_POLICY_ENV: &str = "ATLAS_DSV41_CHUNK_POLICY";

/// The fewest chunks of at most `max_chunk` that cover `n` tokens, sized equally (rounded up to
/// [`CHUNK_ALIGN`]) so there is no tiny tail re-paying the per-chunk costs: 8192 at 3968 is
/// 2816+2816+2560, not 3968+3968+256; 4096 is 2048+2048, not 3968+128.
pub fn balanced_chunk_len(n: usize, max_chunk: usize) -> usize {
    let pieces = n.div_ceil(max_chunk).max(1);
    n.div_ceil(pieces).next_multiple_of(CHUNK_ALIGN).min(max_chunk)
}

fn engram_debug() -> bool {
    std::env::var(ENGRAM_DEBUG_ENV).is_ok()
}

/// Attribution tool (team-lead's 3-arm plan): override the PREFILL dead-head mask on the
/// EXECUTING path, so a comparison against the oracle needs no magnitude reasoning --
/// whichever arm lands closest answers directly. `ported` (default, unset) is the real path
/// and is a no-op. Never applies to decode: decode's own correctness (block-local, no carry)
/// is a separate, already-tested question, and overriding it here would conflate the two.
pub const ENGRAM_DEAD_ARM_ENV: &str = "ATLAS_DSV41_ENGRAM_DEAD_ARM";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeadArm {
    Ported,
    Off,
    Shifted,
}

fn dead_arm() -> Result<DeadArm> {
    match std::env::var(ENGRAM_DEAD_ARM_ENV) {
        Err(_) => Ok(DeadArm::Ported),
        Ok(v) => match v.as_str() {
            "ported" => Ok(DeadArm::Ported),
            "off" => Ok(DeadArm::Off),
            "shifted" => Ok(DeadArm::Shifted),
            other => anyhow::bail!("{ENGRAM_DEAD_ARM_ENV}: ported|off|shifted, got {other}"),
        },
    }
}

/// Apply `arm` to an already-computed [T, N_HEAD_COLS] mask. `Ported` is a no-op (returns
/// `dead` unchanged); the other two exist only for this attribution tool.
fn apply_dead_arm(arm: DeadArm, dead: Vec<bool>, t: usize) -> Vec<bool> {
    match arm {
        DeadArm::Ported => dead,
        DeadArm::Off => vec![false; dead.len()],
        DeadArm::Shifted => {
            // Same construction as dead_heads.rs's own negative control: OR the mask with
            // itself shifted forward one position.
            let mut wrong = dead.clone();
            for p in (1..t).rev() {
                for c in 0..N_HEAD_COLS {
                    wrong[p * N_HEAD_COLS + c] = dead[p * N_HEAD_COLS + c] || dead[(p - 1) * N_HEAD_COLS + c];
                }
            }
            wrong
        }
    }
}

/// `candidate_source_layer`: the last encoder layer.
pub const ENCODER_LAST: usize = 20;

/// `ATLAS_DSV41_SHARED_RESIDENT=0` turns OFF the resident bf16 shared-expert weights (default
/// ON: +[`super::fwd::shared_resident_bytes`] = 70.8 MB per layer, 2.83 GB for all 40, as far
/// as free memory allows above a 16 GB floor, made after the arena; see [`V41Forward::make_resident`]).
pub const SHARED_RESIDENT_ENV: &str = "ATLAS_DSV41_SHARED_RESIDENT";

/// Free device memory that must remain after the resident copies. They are made AFTER the CB3
/// arena, the attention core and (with DSpark) the drafter are resident
/// ([`V41Forward::make_resident`]), so this is the same 16 GB floor the arena keeps for KV,
/// activations and page cache — nothing loaded later is budgeted against a guess.
const SHARED_RESIDENT_MIN_FREE: u64 = 16_000_000_000;

/// The opt-in attention copies keep MORE free memory: measured with DSpark on, prefill and decode
/// transients take ~2 GB below the free memory seen at residency time (a 16 GB check gave a 14 GB
/// low-water), so 20 GB here keeps the real low-water >= 16 GB (lead, 2026-09-23).
const ATTN_RESIDENT_MIN_FREE: u64 = 20_000_000_000;

/// `ATLAS_DSV41_ATTN_RESIDENT` (default ON; `0` = off): resident bf16 copies of the attention's
/// biggest FP8 weights, wq_b (84 MB), wo_b (84 MB) and wo_a (67 MB) per layer, encoder layers
/// first, as far as free memory allows above a 20 GB floor (the shared expert keeps 16 GB). The
/// per-layer fallback makes it ELASTIC: 40/40 layers (9.40 GB) plain, 11-21 with DSpark; the
/// measured low-water stayed >= 16 GB with DSpark and at 32K.
/// `=swap_control` makes them resident with each even/odd layer pair's wq_b SWAPPED (real
/// matrices, the wrong layer's) - a byte gate must FAIL under it.
pub const ATTN_RESIDENT_ENV: &str = "ATLAS_DSV41_ATTN_RESIDENT";

/// See [`ATTN_RESIDENT_ENV`]. Returns the allocations to own.
fn make_attn_resident(ops: &Ops, attn: &mut [V41AttnWeights]) -> Result<Vec<DevicePtr>> {
    // Default ON (lead, 2026-09-23: the 20 GB floor held under DSpark and at 32K); "0" = off.
    let mode = std::env::var(ATTN_RESIDENT_ENV).unwrap_or_else(|_| "1".into());
    if mode == "0" {
        tracing::info!("DeepSeek-V4.1: resident attention weights OFF ({ATTN_RESIDENT_ENV}=0)");
        return Ok(Vec::new());
    }
    ensure!(mode == "1" || mode == "swap_control", "{ATTN_RESIDENT_ENV}={mode}: use 1 (default), 0 or swap_control");
    let per_layer = |a: &V41AttnWeights| (a.wq_b.bf16_bytes() + a.wo_b.bf16_bytes() + a.wo_a.bf16_bytes()) as u64;
    let mut owned = Vec::new();
    let mut resident = 0usize;
    for a in attn.iter_mut() {
        let free = ops.gpu.free_memory().context("attention residency: querying free memory")? as u64;
        if free < per_layer(a) + ATTN_RESIDENT_MIN_FREE {
            break;
        }
        for w in [&mut a.wq_b, &mut a.wo_b, &mut a.wo_a] {
            let p = ops.gpu.alloc(w.bf16_bytes())?;
            ops.dequant(w, p)?;
            w.bf16 = Some(p);
            owned.push(p);
        }
        resident += 1;
    }
    if mode == "swap_control" {
        for pair in attn[..resident].chunks_exact_mut(2) {
            let (a, b) = pair.split_at_mut(1);
            std::mem::swap(&mut a[0].wq_b.bf16, &mut b[0].wq_b.bf16);
        }
    }
    ops.gpu.synchronize(ops.stream)?;
    let bytes: u64 = attn[..resident].iter().map(per_layer).sum();
    eprintln!(
        "DeepSeek-V4.1: resident attention on {resident} of {} layers ({:.2} GB, floor {:.0} GB)",
        attn.len(),
        bytes as f64 / 1e9,
        ATTN_RESIDENT_MIN_FREE as f64 / 1e9
    );
    tracing::info!(
        "DeepSeek-V4.1: resident bf16 attention wq_b/wo_b/wo_a on {resident} of {} layers ({:.2} GB, floor {:.0} GB{})",
        attn.len(),
        bytes as f64 / 1e9,
        ATTN_RESIDENT_MIN_FREE as f64 / 1e9,
        if mode == "swap_control" { ", SWAP CONTROL" } else { "" },
    );
    Ok(owned)
}

/// Resident bf16 shared-expert weights, layer by layer: the encoder layers `0..=ENCODER_LAST`
/// first (every prefill chunk), then the replay layers (once per prompt, T <= 128). Each takes
/// the dequant path otherwise. Byte-identical to the per-pass dequant; see
/// `SharedExpert::make_resident`. A layer is made resident only while free memory stays above
/// [`SHARED_RESIDENT_MIN_FREE`]; past that, the remaining layers FALL BACK to the dequant path
/// (logged), never a refusal. Returns the allocations to own.
fn make_shared_resident(
    ops: &Ops,
    dims: &V41Dims,
    blocks: &mut [V41BlockWeights],
) -> Result<Vec<DevicePtr>> {
    if std::env::var(SHARED_RESIDENT_ENV).as_deref() == Ok("0") {
        tracing::info!("DeepSeek-V4.1: resident shared-expert weights OFF ({SHARED_RESIDENT_ENV}=0)");
        return Ok(Vec::new());
    }
    let per_layer = super::fwd::shared_resident_bytes(dims) as u64;
    let mut owned = Vec::with_capacity(2 * blocks.len());
    let mut resident = 0usize;
    for b in blocks.iter_mut() {
        let free = ops.gpu.free_memory().context("shared-expert residency: querying free memory")? as u64;
        if free < per_layer + SHARED_RESIDENT_MIN_FREE {
            break;
        }
        owned.extend(b.shared.make_resident(ops)?);
        resident += 1;
    }
    ops.gpu.synchronize(ops.stream)?;
    eprintln!(
        "DeepSeek-V4.1: resident shared expert on {resident} of {} layers (floor {:.0} GB)",
        blocks.len(),
        SHARED_RESIDENT_MIN_FREE as f64 / 1e9
    );
    tracing::info!(
        "DeepSeek-V4.1: resident bf16 shared-expert weights on {resident} of {} layers ({:.2} GB; the rest use the \
         dequant path; {:.0} GB floor kept)",
        blocks.len(),
        (resident as u64 * per_layer) as f64 / 1e9,
        SHARED_RESIDENT_MIN_FREE as f64 / 1e9,
    );
    Ok(owned)
}
pub const N_LAYERS: usize = 40;

/// Which pass a core call belongs to. The attention lane's state machine keys on it: the
/// replay must install L20's tail selection before layer 21 and must not recompute it on
/// the inheriting layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// Layers 0..=20 over one prompt chunk.
    EncoderChunk,
    /// Layers 21..=39 over the last min(128, P) prompt rows.
    Replay,
    /// One generated token through all 40 layers.
    Decode,
    /// DSpark verify: the accepted token plus the drafts (T <= 8) through all 40 layers. Behaves as
    /// Decode everywhere (decode-size kernels, no dead-head carry); followed by a rollback.
    Verify,
    /// Debug: all 40 layers over a whole prompt chunk (no replay).
    FullChunk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    /// THE prefill path.
    Replay,
    /// Debug only: every layer over every prompt token.
    Full,
}

/// Per-pass hook for the attention lane, called once before the first layer of a pass.
pub trait PassHook {
    fn begin_pass(&self, kind: PassKind, start: usize, t: usize) -> Result<()>;
    /// DSpark: discard every position >= `n` after a Verify pass (see BRIEF.md, "DSPARK ROLLBACK
    /// CONTRACT"). The default REFUSES: a core that carries state across positions and does not
    /// implement this would otherwise silently keep the rejected drafts' state.
    fn rollback(&self, _ops: &Ops, n: usize) -> Result<()> {
        anyhow::bail!("this attention core cannot roll back (to {n} positions)")
    }
    /// CUDA-graph decode: switch the core to shape-static Decode/Verify passes reading their
    /// start from the device i32 `dstart`. Refuses by default (a core that is not shape-static
    /// would replay a stale shape silently).
    fn enable_graph(&self, _dstart: DevicePtr) -> Result<()> {
        anyhow::bail!("this attention core is not shape-static; decode graphs need one")
    }
    /// CUDA-graph decode: the host bookkeeping of a pass that was REPLAYED (its launches came
    /// from a graph, `run` was not called).
    fn replay_step(&self, kind: PassKind, start: usize, _t: usize) -> Result<()> {
        anyhow::bail!("this attention core cannot account for a replayed {kind:?} pass at {start}")
    }
}

/// Per-sequence state this module owns. (ckv / ik / pending belong to the attention lane.)
pub struct V41Seq {
    /// Unique per allocation (process-wide): CUDA graphs bake this sequence's buffers in and
    /// are keyed by it, so a new sequence never replays a graph captured on a freed one.
    pub id: u64,
    /// One window ring per layer, `[RING, 512]` bf16, slot = pos % RING.
    pub rings: Vec<DevicePtr>,
    pub hash: EngramHashState,
    /// Positions absorbed so far.
    pub len: usize,
    /// The last min(128, len) rows of the ENCODER output stream and its pre_mix, carried
    /// across chunks for the replay (`Model._rep_keep`).
    pub tail_h: DevicePtr,
    pub tail_pre: DevicePtr,
    pub tail_rows: usize,
    /// Token ids of those tail rows (the replay's MoE routes by them).
    pub tail_ids: Vec<u32>,
    /// The trailing up-to-`MAX_LOOKBACK` raw ids from the most recently processed
    /// chunk/token, carried so `engram_dead_heads_with_carry` sees look-back
    /// across a chunk boundary the way `engine/v41_engine.py:755` does (it hashes
    /// the WHOLE image-expanded prompt once, then slices per chunk) rather than
    /// `engine/model.py:747`'s local fallback, which only applies to decode/text
    /// (dsv41-parity, 2026-09-22).
    pub dead_carry: Vec<u32>,
    /// Prefetched engram hashes, keyed by chunk `start_pos`. Populated by
    /// [`V41Forward::plan_engram_prefetch`] before the chunk loop runs; `pass()` consumes
    /// (removes) an entry here instead of calling `hash.forward()` again when one exists, since
    /// the hash state is append-only and can only be advanced once per position.
    pub engram_hash_cache: std::collections::HashMap<usize, Vec<i64>>,
    /// Prefetched, dequantized (host, PRE-mask) engram rows, keyed by `(start_pos, layer)`.
    /// `run_layers` checks here first and uploads straight from it, skipping the synchronous
    /// NVMe gather, whenever an entry exists (either already received off
    /// `engram_prefetch_rx`, or inserted directly by a caller that isn't using the channel).
    pub engram_row_cache: std::collections::HashMap<(usize, usize), Vec<f32>>,
    /// Set for the duration of a prefetching `prefill()` call: the channel a background thread
    /// is sending `((start_pos, layer), rows)` down, in job order (chunk-major, then this
    /// model's engram layers in order). `run_layers`'s engram branch blocks on this -- not on a
    /// per-chunk wait in `prefill`'s loop -- so the wait happens exactly when that SPECIFIC
    /// layer's rows are needed, overlapping with every layer before it in the SAME chunk (the
    /// embedding step plus layers 0..L-1), not only with earlier chunks. A single-chunk prompt
    /// (chunk >= prompt length) has no chunk-level overlap to exploit at all; per-layer overlap
    /// is what makes prefetch pay off there (dsv41-lead, 2026-09-22).
    pub engram_prefetch_rx: Option<std::sync::mpsc::Receiver<((usize, usize), Result<Vec<f32>>)>>,
}

impl V41Seq {
    pub fn new(gpu: &dyn GpuBackend, dims: &V41Dims, hash: EngramHashState) -> Result<Self> {
        let rings = (0..N_LAYERS)
            .map(|_| {
                let r = gpu.alloc(RING * HEAD_DIM * 2)?;
                gpu.memset(r, 0, RING * HEAD_DIM * 2)?;
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            rings,
            hash,
            len: 0,
            tail_h: gpu.alloc(WINDOW * dims.hc * dims.hidden * 2)?,
            tail_pre: gpu.alloc(WINDOW * dims.hc * 4)?,
            tail_rows: 0,
            tail_ids: Vec::new(),
            dead_carry: Vec::new(),
            engram_hash_cache: std::collections::HashMap::new(),
            engram_row_cache: std::collections::HashMap::new(),
            engram_prefetch_rx: None,
        })
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for r in self.rings {
            gpu.free(r)?;
        }
        gpu.free(self.tail_h)?;
        gpu.free(self.tail_pre)
    }
}

/// Everything resident for the forward, independent of any one sequence.
pub struct V41Forward {
    pub dims: V41Dims,
    pub vocab: usize,
    pub blocks: Vec<V41BlockWeights>,
    pub attn: Vec<V41AttnWeights>,
    pub embed: DevicePtr,
    pub norm: DevicePtr,
    pub head: DevicePtr,
    pub freqs_c: RopeTable,
    pub freqs_w: RopeTable,
    pub scratch: PassScratch,
    pub attn_scratch: AttnScratch,
    pub engram: Vec<(usize, EngramGather)>,
    /// V4.1 image input (the tower + prompt-embedding splice); None = text-only.
    pub vision: Option<super::image_splice::V41ImageSplice>,
    pub ids_dev: DevicePtr,
    pub max_chunk: usize,
    pub max_seq: usize,
    /// Set only around the replay's run_layers when TRIM_LAST is on: the last prompt token's id
    /// (the MoE's pass tokens for the one-row post-attention path of the last layer).
    trim_last_id: std::sync::Mutex<Option<u32>>,
    /// Every device allocation this forward made (scratch, RoPE tables, ids, the engram
    /// q*k weight products), freed on drop once [`Self::own_allocations`] hands it a backend.
    /// Until then (the driver) it only records them.
    pub allocs: super::device_allocs::DeviceAllocs,
    /// DSpark seed (`Model.forward`'s `main_hiddens`): bf16 `[rows, 3 * hidden]`, filled with the
    /// hc-mean of the INPUT stream of L37, L38, L39 on every pass that runs them (replay,
    /// decode, verify). `None` unless the drafter is loaded (`enable_dspark_seed`).
    pub dspark_seed: Option<DevicePtr>,
    /// Resident bf16 shared-expert weights (see [`SHARED_RESIDENT_ENV`]).
    pub shared_resident: Vec<DevicePtr>,
    /// The DSpark drafter's weights (store-resident), when loaded. Consumed by dsv41-decode's
    /// draft/verify path.
    pub dspark: Option<super::mtp::DsparkWeights>,
    /// Whole-step CUDA graphs for Decode/Verify passes (`enable_graphs`); None = eager.
    pub graph: Option<GraphState>,
}

/// CUDA-graph state: the device start scalar, the engram rows gathered BEFORE the graph (the
/// NVMe gather is host work and cannot be captured), and one instantiated graph per
/// (pass kind, rows, sequence) -- the rings baked into a graph belong to one sequence, so the
/// first pass of a new sequence destroys every graph of the previous one.
pub struct GraphState {
    pub dstart: DevicePtr,
    /// (engram layer, rows buffer `[MAX_T, 24, 256]` f32).
    pub pre_rows: Vec<(usize, DevicePtr)>,
    graphs: std::sync::Mutex<std::collections::HashMap<(u8, usize, u64), spark_runtime::gpu::GraphHandle>>,
    /// Graphs captured so far (a replay never adds one).
    pub captures: std::sync::atomic::AtomicUsize,
    /// The `V41Seq::id` the cached graphs were captured on.
    seq_id: std::sync::Mutex<Option<u64>>,
    /// Destroys every instantiated graph on drop (TUI model swap), when the forward owns its
    /// allocations (served model); the driver leaves this None.
    owner: Option<super::device_allocs::SharedGpu>,
}

impl Drop for GraphState {
    fn drop(&mut self) {
        let Some(gpu) = &self.owner else { return };
        let _ = gpu.bind_to_thread();
        for (_, h) in self.graphs.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default() {
            if let Err(e) = gpu.destroy_graph(h) {
                tracing::warn!("GraphState drop: destroying a decode graph failed: {e}");
            }
        }
    }
}

/// Rows a graphed pass may carry (Decode = 1, Verify = 1 + drafts).
pub const GRAPH_MAX_T: usize = 8;

pub fn rope_specs() -> (RopeSpec, RopeSpec) {
    let w = RopeSpec { dim: 64, original_seq_len: 0, base: 10000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let c = RopeSpec { original_seq_len: 65536, base: 160000.0, ..w };
    (c, w)
}

impl V41Forward {
    /// `n_layers < 40` is for the driver's partial runs; serving always passes 40.
    pub fn load(
        store: &WeightStore,
        ops: &Ops,
        dims: V41Dims,
        vocab: usize,
        n_layers: usize,
        max_chunk: usize,
        max_seq: usize,
        model_dir: &std::path::Path,
        engram_threads: usize,
    ) -> Result<Self> {
        let blocks = (0..n_layers).map(|l| V41BlockWeights::load(store, l, &dims, ops)).collect::<Result<Vec<_>>>()?;

        let attn = (0..n_layers).map(|l| V41AttnWeights::load(store, l, dims.hidden)).collect::<Result<Vec<_>>>()?;

        let largest = blocks
            .iter()
            .flat_map(|b| [b.shared.w1, b.shared.w2, b.shared.w3].into_iter().chain(b.engram.as_ref().map(|e| e.wkv)))
            .map(|w| w.n * w.k)
            .chain(attn.iter().map(|a| a.largest_weight()))
            .max()
            .unwrap_or(0);
        let (c, w) = rope_specs();
        // FP8 dense GEMMs at M > 16 are issued at ONE M for every chunk (see ops::fp8_fixed_m);
        // every activation buffer below holds tiled_rows(max(max_chunk, 128)) rows.
        super::ops::set_fp8_fixed_m(super::ops::tiled_rows(max_chunk.max(WINDOW)));
        let engram = blocks
            .iter()
            .filter(|b| b.engram.is_some())
            .map(|b| Ok((b.layer, EngramGather::open(model_dir, b.layer, engram_threads)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dims,
            vocab,
            embed: bf16_tensor(store, "embed.weight", &[vocab, dims.hidden])?,
            norm: bf16_tensor(store, "norm.weight", &[dims.hidden])?,
            head: bf16_tensor(store, "head.weight", &[vocab, dims.hidden])?,
            freqs_c: c.upload(ops.gpu, max_seq + 8)?,
            freqs_w: w.upload(ops.gpu, max_seq + 8)?,
            scratch: PassScratch::new(ops.gpu, &dims, max_chunk.max(WINDOW), largest)?,
            attn_scratch: AttnScratch::new(ops.gpu, max_chunk.max(WINDOW))?,
            ids_dev: ops.gpu.alloc(super::ops::tiled_rows(max_chunk.max(WINDOW)) * 4)?,
            blocks,
            attn,
            engram,
            vision: None,
            max_chunk,
            max_seq,
            trim_last_id: std::sync::Mutex::new(None),
            allocs: super::device_allocs::DeviceAllocs::unowned(),
            dspark_seed: None,
            dspark: None,
            graph: None,
            shared_resident: Vec::new(),
        })
    }

    /// Allocate the DSpark seed buffer (call before [`Self::own_allocations`] so it is owned too).
    /// Turn on whole-step CUDA graphs for Decode/Verify passes. `hook` is switched to its
    /// shape-static mode reading the same device start.
    pub fn enable_graphs(&mut self, gpu: &dyn GpuBackend, hook: &dyn PassHook) -> Result<()> {
        let dstart = gpu.alloc(256)?;
        self.allocs.adopt(dstart);
        let mut pre_rows = Vec::new();
        for (layer, _) in &self.engram {
            let p = gpu.alloc(GRAPH_MAX_T * N_HEAD_COLS * 256 * 4)?;
            self.allocs.adopt(p);
            pre_rows.push((*layer, p));
        }
        hook.enable_graph(dstart)?;
        self.graph = Some(GraphState {
            dstart,
            pre_rows,
            graphs: std::sync::Mutex::new(std::collections::HashMap::new()),
            captures: std::sync::atomic::AtomicUsize::new(0),
            seq_id: std::sync::Mutex::new(None),
            owner: self.allocs.owner(),
        });
        Ok(())
    }

    pub fn enable_dspark_seed(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let rows = super::ops::tiled_rows(self.max_chunk.max(WINDOW));
        let cols = super::mtp::DSPARK_TARGET_LAYERS.len() * self.dims.hidden;
        let p = gpu.alloc(rows * cols * 2)?;
        self.allocs.adopt(p);
        self.dspark_seed = Some(p);
        Ok(())
    }

    /// Make the resident bf16 weight copies ([`SHARED_RESIDENT_ENV`], default on; and the opt-in
    /// [`ATTN_RESIDENT_ENV`]) — call once, AFTER the CB3 arena, the attention core and any
    /// drafter are loaded, so each layer's copy is checked against the real remaining memory.
    /// Byte-identical to the per-pass dequant; the allocations join [`Self::allocs`].
    pub fn make_resident(&mut self, ops: &Ops) -> Result<()> {
        let mut new = make_shared_resident(ops, &self.dims, &mut self.blocks)?;
        new.extend(make_attn_resident(ops, &mut self.attn)?);
        for p in &new {
            self.allocs.adopt(*p);
        }
        self.shared_resident.extend(new);
        Ok(())
    }

    /// Take ownership of every allocation made in [`Self::load`] so dropping the forward frees
    /// it (serving). The driver skips this and lets process exit clean up.
    pub fn own_allocations(&mut self, gpu: super::device_allocs::SharedGpu) {
        let mut a = super::device_allocs::DeviceAllocs::owned(gpu);
        for p in self.scratch.allocations().iter().chain(self.attn_scratch.allocations()) {
            a.adopt(*p);
        }
        for t in [self.freqs_c, self.freqs_w] {
            a.adopt(t.cos);
            a.adopt(t.sin);
        }
        a.adopt(self.ids_dev);
        if let Some(p) = self.dspark_seed {
            a.adopt(p);
        }
        for b in &self.blocks {
            if let Some(e) = &b.engram {
                a.adopt(e.weight);
            }
        }
        // Resident copies made before this point move to the new owner; later ones are adopted
        // by make_resident itself.
        for p in &self.shared_resident {
            a.adopt(*p);
        }
        self.allocs = a;
    }

    fn rope(&self, layer: usize) -> &RopeTable {
        if compress_ratio(layer) != 0 { &self.freqs_c } else { &self.freqs_w }
    }

    /// Layers `layers` over `t` rows at absolute `start`, in place on `scratch.h/pre_mix`.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        layers: std::ops::Range<usize>,
        t: usize,
        start: usize,
        win_lo: usize,
        hashes: Option<&[i64]>,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        self.run_layers_pre(ops, seq, layers, t, start, win_lo, hashes, None, core, moe, tap)
    }

    /// [`Self::run_layers`] with the engram rows optionally PRE-GATHERED per layer (graph mode:
    /// the NVMe gather ran before the capture/replay).
    #[allow(clippy::too_many_arguments)]
    fn run_layers_pre(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        layers: std::ops::Range<usize>,
        t: usize,
        start: usize,
        win_lo: usize,
        hashes: Option<&[i64]>,
        pre: Option<&[(usize, DevicePtr)]>,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        let s = &self.scratch;
        for l in layers {
            let w = &self.blocks[l];
            if let (Some(e), Some(pre)) = (&w.engram, pre) {
                let rows = pre.iter().find(|(pl, _)| *pl == l).context("no pre-gathered engram rows")?.1;
                e.forward(ops, s.h, rows, s.engram_dead, t, s, &self.dims)?;
            } else if let Some(e) = &w.engram {
                let all = hashes.context("engram layer reached without hashes")?;
                let li = if l == 1 { 0 } else { 1 };
                let rows: Vec<i64> = (0..t).flat_map(|tok| all[(tok * 2 + li) * 24..(tok * 2 + li + 1) * 24].iter().copied()).collect();
                let g = &self.engram.iter().find(|(el, _)| *el == l).context("no engram gather")?.1;
                // A prefetch thread may already have this chunk's rows staged in host memory
                // (see `V41Forward::prefill`/`engram_prefetch_rx`); if so, skip the synchronous
                // NVMe gather and just upload. Falls back to the live gather otherwise (decode,
                // or prefill without prefetch) -- correctness never depends on prefetch having
                // run. Blocking on the CHANNEL here, not on a cache check alone, is what gives
                // per-LAYER overlap: this wait only happens when THIS layer's rows are actually
                // needed, so it overlaps with every layer before it in the same chunk (the
                // embed step and layers 0..L-1), not only with earlier chunks -- the thing that
                // makes prefetch pay off even on a single-chunk prompt.
                if !seq.engram_row_cache.contains_key(&(start, l)) {
                    if let Some(rx) = &seq.engram_prefetch_rx {
                        loop {
                            let (key, host) =
                                rx.recv().context("engram prefetch thread ended before sending this layer's rows")?;
                            let found = key == (start, l);
                            seq.engram_row_cache.insert(key, host?);
                            if found {
                                break;
                            }
                        }
                    }
                }
                match seq.engram_row_cache.remove(&(start, l)) {
                    Some(host) => {
                        prof(ops, "engram.upload_prefetched", || {
                            EngramGather::upload_rows(&host, s.engram_rows, ops.gpu, ops.stream, l)
                        })?;
                    }
                    None => {
                        prof(ops, "engram.gather", || g.gather_rows_gpu(&rows, t, s.engram_rows, ops.gpu, ops.stream))?;
                    }
                }
                let debug = engram_debug();
                if debug {
                    tap.bf16(ops, "engram_in", l, s.h, &[t, self.dims.hc, self.dims.hidden])?;
                    tap.f32(ops, "engram_rows_premask", l, s.engram_rows, &[t, N_HEAD_COLS, 256])?;
                }
                // `pass` uploaded this chunk's dead-head mask into `s.engram_dead`.
                prof(ops, "engram.proj", || e.forward(ops, s.h, s.engram_rows, s.engram_dead, t, s, &self.dims))?;
                if debug {
                    // The mask+cast the wkv linear actually reads, AFTER dsv41_engram_rows_bf16
                    // runs -- if this doesn't differ from engram_rows_premask on a chunk the
                    // executing mask log says has masked cells, the mask isn't reaching the GEMM.
                    tap.bf16(ops, "engram_rows_masked", l, s.engram_rows_bf16, &[t, N_HEAD_COLS, 256])?;
                }
                tap.bf16(ops, "engram_out", l, s.h, &[t, self.dims.hc, self.dims.hidden])?;
            }
            if let (Some(seed), Some(col)) =
                (self.dspark_seed, super::mtp::DSPARK_TARGET_LAYERS.iter().position(|&x| x == l))
            {
                let d = self.dims.hidden;
                ops.hc_mean_bf16(s.h, seed, t, d, super::mtp::DSPARK_TARGET_LAYERS.len() * d, col * d)?;
            }
            let adapter = AttnAdapter { fwd: self, ring: seq.rings[l], win_lo, core, tap };
            // TRIM_LAST (replay only): the model's last layer needs its post-attention path for the
            // last row only -- nothing reads the other rows' layer-39 output (the DSpark seed takes
            // the INPUT streams of L37-39, captured above).
            let trim = *self.trim_last_id.lock().expect("trim lock");
            let post_from = match trim {
                Some(id) if l + 1 == self.blocks.len() && t > 1 => {
                    moe.begin_pass(&[id])?;
                    t - 1
                }
                _ => 0,
            };
            block(ops, w, &self.dims, s, t, start, &adapter, moe, tap, BlockControl::None, post_from)?;
        }
        Ok(())
    }

    /// Embed `ids` into the stream, hash them for the engram, run `layers`.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        ids: &[u32],
        start: usize,
        layers: std::ops::Range<usize>,
        kind: PassKind,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        let t = ids.len();
        ensure!(seq.len == start, "sequence holds {} positions, pass starts at {start}", seq.len);
        ensure!(start + t <= self.max_seq, "position {} exceeds max_seq {}", start + t, self.max_seq);
        // `prefetch_engram` may have already computed this chunk's hashes (and advanced
        // `seq.hash`'s append-only state) up front; reuse them instead of calling forward()
        // again, which would either panic (position mismatch) or double-count.
        let hashes = match seq.engram_hash_cache.remove(&start) {
            Some(h) => h,
            None => seq.hash.forward(ids, start, None)?,
        };
        // PREFILL (`engine/v41_engine.py:755`) hashes the WHOLE image-expanded prompt once and
        // slices per chunk -- carrying the trailing MAX_LOOKBACK raw ids across chunks is the
        // exact equivalent (see deepseek_v41_engram::dead_heads's module doc). DECODE does NOT
        // carry: `m.forward(block, pos, prefill=False)` (v41_engine.py:912, 1037) passes no
        // dead_heads, so `model.py:746-747`'s fallback computes it fresh from that call's own
        // (single-token) ids with no history. Corrected 2026-09-22 by dsv41-parity: an earlier
        // version of this carried on decode too, which is wrong at exactly the position right
        // after an image-ending prompt's first generated token. Replay never reaches an engram
        // layer, so it never exercises this branch either way.
        let mut dead_bools = match kind {
            PassKind::Decode | PassKind::Verify => engram_dead_heads(ids),
            _ => engram_dead_heads_with_carry(&seq.dead_carry, ids),
        };
        if !matches!(kind, PassKind::Decode | PassKind::Verify) {
            // Attribution tool only, real path is a no-op (`DeadArm::Ported`) -- see
            // `apply_dead_arm`'s doc. Applied AFTER the carry, so `--dead-arm shifted`'s shift
            // is relative to what the executing path actually computed, not a re-derivation.
            dead_bools = apply_dead_arm(dead_arm()?, dead_bools, t);
        }
        let dead: Vec<u8> = dead_bools.into_iter().map(u8::from).collect();
        if !matches!(kind, PassKind::Decode | PassKind::Verify) {
            update_dead_carry(&mut seq.dead_carry, ids);
        }
        if engram_debug() {
            // Straight off the host buffer this chunk is about to upload -- proves the mask
            // reaches the point of upload, not merely that it was computed somewhere upstream.
            let masked_cells = dead.iter().filter(|&&d| d != 0).count();
            let masked_positions = dead.chunks(N_HEAD_COLS).filter(|row| row.iter().any(|&d| d != 0)).count();
            eprintln!(
                "ENGRAM_DEBUG pass start={start} t={t}: masked_cells={masked_cells}/{} masked_positions={masked_positions}/{t}",
                dead.len()
            );
        }
        ops.gpu.copy_h2d_async(&dead, self.scratch.engram_dead, ops.stream)?;
        ops.gpu.copy_h2d_async(bytemuck_u32(ids), self.ids_dev, ops.stream)?;
        ops.embed(self.embed, self.ids_dev, self.scratch.x, t, self.dims.hidden)?;
        // Image spans (prefill only; decode ids are never image ids). A no-op, with no
        // GPU work, unless this request encoded images.
        if !matches!(kind, PassKind::Decode | PassKind::Verify)
            && let Some(v) = &self.vision
        {
            v.splice(ops.gpu, ops.stream, self.embed, ids, start, self.scratch.x)?;
        }
        ops.hc_expand(self.scratch.x, self.scratch.h, self.scratch.pre_mix, t, self.dims.hidden)?;
        // Decode-size kernels key on the PASS KIND (never on t): see ops::set_decode_pass.
        super::ops::set_decode_pass(matches!(kind, PassKind::Decode | PassKind::Verify));
        super::ops::set_replay_pass(matches!(kind, PassKind::Replay));
        hook.begin_pass(kind, start, t)?;
        moe.begin_pass(ids)?;
        self.run_layers(ops, seq, layers, t, start, 0, Some(&hashes), core, moe, tap)?;
        seq.len = start + t;
        Ok(())
    }

    /// Keep the last `min(128, …)` encoder output rows across chunks (`_rep_keep`).
    fn keep_tail(&self, ops: &Ops, seq: &mut V41Seq, ids: &[u32]) -> Result<()> {
        let t = ids.len();
        seq.tail_ids.extend_from_slice(ids);
        let drop = seq.tail_ids.len().saturating_sub(WINDOW);
        seq.tail_ids.drain(..drop);
        let (hc, d) = (self.dims.hc, self.dims.hidden);
        let (hrow, prow) = (hc * d * 2, hc * 4);
        let take = t.min(WINDOW);
        let keep_old = (WINDOW - take).min(seq.tail_rows);
        if keep_old > 0 && keep_old < seq.tail_rows {
            // shift the newest `keep_old` old rows to the front (non-overlapping when
            // keep_old <= tail_rows - keep_old; otherwise go through the x scratch)
            let from = seq.tail_rows - keep_old;
            let tmp_h = self.scratch.engram_kv;
            ops.gpu.copy_d2d_async(seq.tail_h.offset(from * hrow), tmp_h, keep_old * hrow, ops.stream)?;
            ops.gpu.copy_d2d_async(tmp_h, seq.tail_h, keep_old * hrow, ops.stream)?;
            let tmp_p = self.scratch.engram_rows;
            ops.gpu.copy_d2d_async(seq.tail_pre.offset(from * prow), tmp_p, keep_old * prow, ops.stream)?;
            ops.gpu.copy_d2d_async(tmp_p, seq.tail_pre, keep_old * prow, ops.stream)?;
        }
        let src = t - take;
        ops.gpu.copy_d2d_async(self.scratch.h.offset(src * hrow), seq.tail_h.offset(keep_old * hrow), take * hrow, ops.stream)?;
        ops.gpu.copy_d2d_async(self.scratch.pre_mix.offset(src * prow), seq.tail_pre.offset(keep_old * prow), take * prow, ops.stream)?;
        seq.tail_rows = keep_old + take;
        Ok(())
    }

    /// One prompt chunk at `seq.len`. Replay mode runs only the encoder (layers 0..=20) and
    /// keeps the stream's tail; Full mode runs every layer. Call [`Self::finish_prefill`]
    /// after the LAST chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        chunk: &[u32],
        mode: PrefillMode,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        ensure!(!chunk.is_empty() && chunk.len() <= self.max_chunk, "chunk of {} (max {})", chunk.len(), self.max_chunk);
        let start = seq.len;
        if start == 0 {
            seq.tail_rows = 0;
            seq.tail_ids.clear();
            // Robustness (dsv41-parity): V41Seq is rebuilt per request today
            // (Dsv41Model::alloc_sequence), so this is currently unreachable with a nonempty
            // carry -- but if a sequence slot is ever reused, a stale dead_carry from the
            // PREVIOUS request would otherwise silently leak into this request's first chunk.
            seq.dead_carry.clear();
        }
        let n = self.blocks.len();
        match mode {
            PrefillMode::Replay => {
                let enc = 0..(ENCODER_LAST + 1).min(n);
                self.pass(ops, seq, chunk, start, enc, PassKind::EncoderChunk, hook, core, moe, tap)?;
                self.keep_tail(ops, seq, chunk)
            }
            PrefillMode::Full => {
                self.pass(ops, seq, chunk, start, 0..n, PassKind::FullChunk, hook, core, moe, tap)?;
                seq.tail_rows = chunk.len(); // rows of the last chunk live in scratch.h
                Ok(())
            }
        }
    }

    /// After the last prompt chunk: the decoder replay (Replay mode), then bf16 logits
    /// `[vocab]` of the prompt's last row.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_prefill(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        mode: PrefillMode,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        let n = self.blocks.len();
        let t = seq.tail_rows;
        ensure!(t > 0, "finish_prefill before any prompt chunk");
        if mode == PrefillMode::Replay && n > ENCODER_LAST + 1 {
            // decoder_replay: layers 21..39 over the tail, window truncated to it.
            let start = seq.len - t;
            let (hrow, prow) = (self.dims.hc * self.dims.hidden * 2, self.dims.hc * 4);
            ops.gpu.copy_d2d_async(seq.tail_h, self.scratch.h, t * hrow, ops.stream)?;
            ops.gpu.copy_d2d_async(seq.tail_pre, self.scratch.pre_mix, t * prow, ops.stream)?;
            super::ops::set_decode_pass(false);
            super::ops::set_replay_pass(true);
            hook.begin_pass(PassKind::Replay, start, t)?;
            ensure!(seq.tail_ids.len() == t, "replay tail holds {} ids for {t} rows", seq.tail_ids.len());
            moe.begin_pass(&seq.tail_ids)?;
            if trim_last_enabled() {
                *self.trim_last_id.lock().expect("trim lock") = seq.tail_ids.last().copied();
            }
            let r = self.run_layers(ops, seq, (ENCODER_LAST + 1)..n, t, start, start, None, core, moe, tap);
            *self.trim_last_id.lock().expect("trim lock") = None;
            super::ops::set_replay_pass(false);
            r?;
        }
        prof(ops, "head", || super::fwd::final_logits_last_row(ops, &self.dims, &self.scratch, t, self.norm, self.head, self.vocab, logits))
    }

    /// The whole prompt: chunks of `max_chunk`, then [`Self::finish_prefill`].
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        ids: &[u32],
        mode: PrefillMode,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        ensure!(!ids.is_empty(), "empty prompt");
        let chunk_len = self.prefill_chunk_len(ids.len())?;
        let prefetch = !self.engram.is_empty() && std::env::var("ATLAS_DSV41_ENGRAM_PREFETCH").as_deref() != Ok("0");
        if !prefetch {
            for chunk in ids.chunks(chunk_len) {
                self.prefill_chunk(ops, seq, chunk, mode, hook, core, moe, tap)?;
            }
            return self.finish_prefill(ops, seq, mode, hook, core, moe, tap, logits);
        }

        // Real overlap requires the chunk loop (this thread's GPU work) to run INSIDE the same
        // scope as the background gather thread, not after it -- joining before this thread
        // reaches the loop (the original version of this code did exactly that bug: it moved
        // the ~73 ms/chunk NVMe cost earlier without ever overlapping it, still fully serial;
        // caught before it was ever GPU-measured). `receiver.recv()` below blocks only on THIS
        // chunk's own (start, layer) result, so a chunk whose gather finished early costs
        // nothing extra, and a chunk that hasn't finished yet blocks no more than the
        // synchronous path already would have.
        let (hashes_by_start, jobs) = self.plan_engram_prefetch(seq, ids)?;
        for (start, hashes) in hashes_by_start {
            seq.engram_hash_cache.insert(start, hashes);
        }
        // Staging memory bound (team-lead's ask): each row is ENGRAM_HEAD_DIM=256 f32 = 1024 B,
        // 24 rows/token/layer -- 24 KiB/token/layer, 48 KiB/token across both engram layers.
        // Bounded by max_seq (8192 in this driver's V41Forward::load call): 8192 * 48 KiB ~=
        // 384 MiB worst case, well under this box's ~25 GB floor. Logged, not capped -- capping
        // would reintroduce exactly the "this chunk's gather sits on its own critical path"
        // cost for whichever chunk falls outside the cap, for no memory-pressure benefit here.
        let staging_bytes: usize = jobs.iter().map(|(_, _, ids)| (ids.len() / 24) * 256 * 4).sum();
        tracing::debug!("engram prefetch: {} jobs, {:.1} MiB staged", jobs.len(), staging_bytes as f64 / (1024.0 * 1024.0));
        let (tx, rx) = std::sync::mpsc::channel::<((usize, usize), Result<Vec<f32>>)>();
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(move || {
                for (start, l, row_ids) in jobs {
                    let g = &self.engram.iter().find(|(el, _)| *el == l).expect("layer just read from self.engram").1;
                    let t = row_ids.len() / 24;
                    // A closed receiver (main thread returned early on an earlier error, or
                    // dropped the receiver at the end of prefill) just stops the sends -- not an
                    // error on this side.
                    if tx.send(((start, l), g.gather_rows_host(&row_ids, t))).is_err() {
                        return;
                    }
                }
            });
            // NOT waited on here: `run_layers`'s engram branch blocks on `seq.engram_prefetch_rx`
            // itself, exactly when it reaches a given (start, layer), so the wait overlaps with
            // every layer before it in the SAME chunk -- not just with earlier chunks. A single-
            // chunk prompt has no chunk-level overlap to exploit; this is what makes prefetch
            // pay off there at all.
            seq.engram_prefetch_rx = Some(rx);
            for chunk in ids.chunks(chunk_len) {
                self.prefill_chunk(ops, seq, chunk, mode, hook, core, moe, tap)?;
            }
            seq.engram_prefetch_rx = None; // drop -> closes our end; the sender thread (already
            // finished, since the scope can't exit until it's joined) is unaffected either way
            Ok(())
        })?;
        self.finish_prefill(ops, seq, mode, hook, core, moe, tap, logits)
    }

    /// DSpark verify: `ids` (the accepted token + the drafts, <= 8) at positions `seq.len..`, as ONE
    /// Verify pass through every layer; bf16 logits of EVERY row, `[t, vocab]` (row i predicts
    /// position seq.len + i + 1). `logits` must hold `tiled_rows(t)` rows. Follow with
    /// [`Self::rollback`] to the accepted length.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        ids: &[u32],
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        ensure!(!ids.is_empty() && ids.len() <= super::ops::GEMV_MAX_M, "verify block of {} ids", ids.len());
        if self.graph.is_some() {
            return self.step_graphed(ops, seq, ids, PassKind::Verify, hook, core, moe, logits);
        }
        let start = seq.len;
        self.pass(ops, seq, ids, start, 0..self.blocks.len(), PassKind::Verify, hook, core, moe, tap)?;
        prof(ops, "head", || super::fwd::final_logits_rows(ops, &self.dims, &self.scratch, ids.len(), self.norm, self.head, self.vocab, logits))
    }

    /// Discard every position >= `n` (after a Verify pass): the attention core's carried state
    /// (`hook`), the engram n-gram history, and the sequence length. The window rings and the
    /// compressed caches are append-only and need nothing (BRIEF.md rollback contract).
    pub fn rollback(&self, ops: &Ops, seq: &mut V41Seq, n: usize, hook: &dyn PassHook) -> Result<()> {
        ensure!(n <= seq.len, "rollback to {n} positions but the sequence holds {}", seq.len);
        if n == seq.len {
            return Ok(());
        }
        hook.rollback(ops, n)?;
        seq.hash.rollback(n)?;
        seq.len = n;
        Ok(())
    }

    /// The chunk a whole-prompt prefill of `n` tokens uses: [`balanced_chunk_len`] (window10,
    /// 8192-token prompt: chunk 3968 1791 vs chunk 2048 1712 tok/s; on 4096, 3968+128 lost to
    /// 2048+2048 -- the tail re-pays the per-chunk costs). [`CHUNK_POLICY_ENV`]`=legacy` restores
    /// the fixed rule. Output does not depend on it (chunk invariance).
    pub fn prefill_chunk_len(&self, n: usize) -> Result<usize> {
        Ok(match std::env::var(CHUNK_POLICY_ENV).as_deref() {
            Err(_) | Ok("balanced") => balanced_chunk_len(n, self.max_chunk),
            Ok("legacy") => if n >= LONG_PROMPT { self.max_chunk } else { self.max_chunk.min(SHORT_CHUNK) },
            Ok(v) => match v.parse::<usize>() {
                Ok(c) if c > 0 => c.min(self.max_chunk),
                _ => anyhow::bail!("{CHUNK_POLICY_ENV}={v}: expected balanced, legacy or a chunk size"),
            },
        })
    }

    /// Compute every chunk's engram hashes up front (cheap, CPU-only, must be sequential --
    /// the hash cache is append-only) and the row-gather job list, WITHOUT touching NVMe.
    /// Split out of `prefill` so the hash-forward calls (which need `&mut seq`) happen before
    /// `prefill`'s `std::thread::scope` borrows `seq` for the chunk loop.
    fn plan_engram_prefetch(&self, seq: &mut V41Seq, ids: &[u32]) -> Result<(Vec<(usize, Vec<i64>)>, Vec<(usize, usize, Vec<i64>)>)> {
        let mut start = seq.len;
        let mut hashes_by_start = Vec::new();
        let mut jobs: Vec<(usize, usize, Vec<i64>)> = Vec::new();
        for chunk in ids.chunks(self.prefill_chunk_len(ids.len())?) {
            let t = chunk.len();
            let hashes = seq.hash.forward(chunk, start, None)?;
            for &(l, _) in &self.engram {
                let li = if l == 1 { 0 } else { 1 };
                let rows: Vec<i64> =
                    (0..t).flat_map(|tok| hashes[(tok * 2 + li) * 24..(tok * 2 + li + 1) * 24].iter().copied()).collect();
                jobs.push((start, l, rows));
            }
            hashes_by_start.push((start, hashes));
            start += t;
        }
        Ok((hashes_by_start, jobs))
    }

    /// One decode step at position `seq.len`; bf16 logits `[vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        token: u32,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        if self.graph.is_some() {
            return self.step_graphed(ops, seq, &[token], PassKind::Decode, hook, core, moe, logits);
        }
        let start = seq.len;
        self.pass(ops, seq, &[token], start, 0..self.blocks.len(), PassKind::Decode, hook, core, moe, tap)?;
        prof(ops, "head", || super::fwd::final_logits_last_row(ops, &self.dims, &self.scratch, 1, self.norm, self.head, self.vocab, logits))
    }

    /// One Decode (t = 1) or Verify (t <= 8) pass as a CUDA GRAPH. Host work that cannot be
    /// captured runs first: the engram hashing and both layers' NVMe row gathers, the dead-head
    /// mask and ids uploads, the MoE's id upload, and the device start. The first step of each
    /// (kind, t, sequence) captures the graph (the core runs and keeps its host bookkeeping);
    /// every later one replays it (the core accounts via `replay_step`). Logits land in `logits`
    /// exactly as the eager paths leave them (last row for Decode, all rows for Verify).
    #[allow(clippy::too_many_arguments)]
    pub fn step_graphed(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        ids: &[u32],
        kind: PassKind,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        logits: DevicePtr,
    ) -> Result<()> {
        let g = self.graph.as_ref().context("step_graphed without enable_graphs")?;
        ensure!(matches!(kind, PassKind::Decode | PassKind::Verify), "only Decode/Verify passes are graphed, not {kind:?}");
        let t = ids.len();
        ensure!(t >= 1 && t <= GRAPH_MAX_T, "graphed pass of {t} rows");
        let start = seq.len;
        ensure!(start + t <= self.max_seq, "position {} exceeds max_seq {}", start + t, self.max_seq);
        let (gpu, stream) = (ops.gpu, ops.stream);
        // ---- host work, outside the graph
        let hashes = seq.hash.forward(ids, start, None)?;
        let dead: Vec<u8> = engram_dead_heads(ids).into_iter().map(u8::from).collect();
        gpu.copy_h2d_async(&dead, self.scratch.engram_dead, stream)?;
        gpu.copy_h2d_async(bytemuck_u32(ids), self.ids_dev, stream)?;
        // NEGATIVE CONTROL ONLY: ATLAS_DSV41_CONTROL_FREEZE_GRAPH_START=1 never advances the
        // device start after the first pass, as a graph that baked in its position would.
        let freeze = std::env::var("ATLAS_DSV41_CONTROL_FREEZE_GRAPH_START").as_deref() == Ok("1");
        if !(freeze && !g.graphs.lock().expect("graph cache poisoned").is_empty()) {
            gpu.memset_u32_async(g.dstart, start as u32, 1, stream)?;
        }
        super::ops::set_decode_pass(true);
        moe.begin_pass(ids)?;
        // Segments. SEGMENTED (default): one graph per span between engram
        // layers, each engram layer's NVMe gather issued right before ITS segment, so the host
        // reads while the GPU runs the previous segment (eager's overlap). Otherwise one graph
        // for the whole pass with both gathers in front of it.
        let n = self.blocks.len();
        // Default ON (measured full model: -3.21 ms/step vs eager; one whole-pass graph only -0.8,
        // its NVMe gathers serialised). `ATLAS_DSV41_GRAPH_SEGMENTED=0` = one graph (A/B only).
        let segmented = std::env::var("ATLAS_DSV41_GRAPH_SEGMENTED").as_deref() != Ok("0");
        let mut bounds: Vec<usize> = vec![0];
        if segmented {
            bounds.extend(g.pre_rows.iter().map(|(l, _)| *l).filter(|&l| l > 0 && l < n));
            bounds.sort_unstable();
            bounds.dedup();
        }
        bounds.push(n);
        let gather = |layer: usize, buf: DevicePtr| -> Result<()> {
            let li = if layer == 1 { 0 } else { 1 };
            let rows: Vec<i64> = (0..t).flat_map(|tok| hashes[(tok * 2 + li) * 24..(tok * 2 + li + 1) * 24].iter().copied()).collect();
            let gth = &self.engram.iter().find(|(el, _)| *el == layer).context("no engram gather")?.1;
            gth.gather_rows_gpu(&rows, t, buf, gpu, stream)
        };
        if !segmented {
            for (layer, buf) in &g.pre_rows {
                gather(*layer, *buf)?;
            }
        }
        // A new sequence: its rings/tails are different buffers (or the SAME addresses reused by
        // the allocator for different roles), so no graph of the previous one may be replayed.
        // NEGATIVE CONTROL ONLY: ATLAS_DSV41_CONTROL_GRAPH_KEEP_STALE=1 keeps and replays them.
        let keep_stale = std::env::var("ATLAS_DSV41_CONTROL_GRAPH_KEEP_STALE").as_deref() == Ok("1");
        {
            let mut cur = g.seq_id.lock().expect("graph seq poisoned");
            if *cur != Some(seq.id) {
                if !keep_stale {
                    let old = std::mem::take(&mut *g.graphs.lock().expect("graph cache poisoned"));
                    if !old.is_empty() {
                        gpu.synchronize(stream)?;
                        for (_, h) in old {
                            gpu.destroy_graph(h)?;
                        }
                    }
                }
                *cur = Some(seq.id);
            }
        }
        let base_key = (kind as u8, t, if keep_stale { 0 } else { seq.id });
        let replay = g.graphs.lock().expect("graph cache poisoned").contains_key(&(base_key.0, base_key.1, base_key.2 ^ ((bounds.len() as u64) << 56)));
        super::ops::set_graph_start(Some(g.dstart));
        let r = (|| -> Result<()> {
            if replay {
                hook.replay_step(kind, start, t)?;
            } else {
                hook.begin_pass(kind, start, t)?;
            }
            for (si, w) in bounds.windows(2).enumerate() {
                let (lo, hi) = (w[0], w[1]);
                if segmented {
                    if let Some((_, buf)) = g.pre_rows.iter().find(|(l, _)| *l == lo) {
                        gather(lo, *buf)?;
                    }
                }
                // Keys: segment 0 of a (kind, t, sequence) carries the segment COUNT in the top
                // byte, so a segmented and an unsegmented capture never alias.
                let key = (base_key.0, base_key.1, base_key.2 ^ (((bounds.len() as u64) << 56) | ((si as u64) << 48)));
                let h = if replay {
                    g.graphs.lock().expect("graph cache poisoned").get(&key).copied().context("graph segment missing")?
                } else {
                    gpu.begin_capture(stream)?;
                    let body = self.graph_segment(ops, seq, t, start, kind, lo..hi, &g.pre_rows, core, moe, logits);
                    let graph = gpu.end_capture(stream);
                    body?;
                    let h = graph?;
                    g.graphs.lock().expect("graph cache poisoned").insert(key, h);
                    g.captures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    h
                };
                gpu.launch_graph(h, stream)?;
            }
            Ok(())
        })();
        super::ops::set_graph_start(None);
        r?;
        seq.len = start + t;
        Ok(())
    }

    /// Layers `layers` of a graphed Decode/Verify pass, in eager order: the embed before layer 0
    /// and the head after the last layer.
    #[allow(clippy::too_many_arguments)]
    fn graph_segment(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        t: usize,
        start: usize,
        kind: PassKind,
        layers: std::ops::Range<usize>,
        pre: &[(usize, DevicePtr)],
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        logits: DevicePtr,
    ) -> Result<()> {
        let s = &self.scratch;
        let last = layers.end == self.blocks.len();
        if layers.start == 0 {
            ops.embed(self.embed, self.ids_dev, s.x, t, self.dims.hidden)?;
            ops.hc_expand(s.x, s.h, s.pre_mix, t, self.dims.hidden)?;
        }
        self.run_layers_pre(ops, seq, layers, t, start, 0, None, Some(pre), core, moe, &Tap::off())?;
        if !last {
            return Ok(());
        }
        match kind {
            PassKind::Decode => super::fwd::final_logits_last_row(ops, &self.dims, s, t, self.norm, self.head, self.vocab, logits),
            _ => super::fwd::final_logits_rows(ops, &self.dims, s, t, self.norm, self.head, self.vocab, logits),
        }
    }
}

/// `V41AttentionBlock` for one layer's call: the projections here, the core from the lane.
struct AttnAdapter<'a> {
    fwd: &'a V41Forward,
    ring: DevicePtr,
    win_lo: usize,
    core: &'a dyn AttnCore,
    tap: &'a Tap,
}

impl V41AttentionBlock for AttnAdapter<'_> {
    fn forward(&self, ops: &Ops, layer: usize, x: DevicePtr, out: DevicePtr, t: usize, start: usize) -> Result<()> {
        attn_block::attention(
            ops,
            &self.fwd.attn[layer],
            &self.fwd.attn_scratch,
            self.fwd.scratch.wscratch,
            self.fwd.rope(layer),
            self.ring,
            x,
            out,
            t,
            start,
            self.win_lo,
            self.fwd.dims.norm_eps,
            self.core,
            self.tap,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::balanced_chunk_len;

    #[test]
    fn balanced_chunk_len_splits_evenly_with_no_tiny_tail() {
        assert_eq!(balanced_chunk_len(8192, 3968), 2816);
        assert_eq!(balanced_chunk_len(4096, 3968), 2048);
        assert_eq!(balanced_chunk_len(6144, 3968), 3072);
        assert_eq!(balanced_chunk_len(16384, 3968), 3328);
        assert_eq!(balanced_chunk_len(1024, 3968), 1024);
        assert_eq!(balanced_chunk_len(8192, 2048), 2048);
        assert_eq!(balanced_chunk_len(1, 1024), 128);
    }

    #[test]
    fn balanced_chunk_len_uses_the_fewest_chunks_within_the_cap() {
        for max in [1000, 1024, 2048, 3968] {
            for n in 1..20_000 {
                let c = balanced_chunk_len(n, max);
                assert!(c >= 1 && c <= max, "n {n} max {max}: chunk {c}");
                assert_eq!(n.div_ceil(c), n.div_ceil(max), "n {n} max {max}: chunk {c}");
            }
        }
    }
}
