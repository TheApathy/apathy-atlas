// SPDX-License-Identifier: AGPL-3.0-only

//! Post-transform weight artifact cache (fast engine recovery, Phase 1).
//!
//! Restarting the server re-runs every per-layer weight transform even though
//! the raw checkpoint is unchanged and already warm in the page cache. The
//! expensive ones are host round-trips: [`crate::weight_map::QuantizedWeight::transpose_for_gemm`]
//! does a D2H of the packed weight, a scalar byte-wise transpose on the CPU,
//! and an H2D of the result — ~9 times per layer on qwen3.8-27b.
//!
//! This module stores the *output* buffers of those transforms in a
//! content-keyed directory. On a later start the buffers are mmap'd and
//! pushed straight H2D, skipping the transform entirely.
//!
//! Everything here is off unless `ATLAS_WEIGHT_CACHE=1`. With the flag unset
//! [`get`] returns `None` and every call site falls through to the original
//! code path, so the default build is byte-for-byte unchanged.
//!
//! Env:
//!   - `ATLAS_WEIGHT_CACHE=1`        — enable (default off)
//!   - `ATLAS_WEIGHT_CACHE_DIR=<p>`  — cache root (default `~/.cache/atlas-weight-cache`)
//!   - `ATLAS_WEIGHT_CACHE_VERIFY=1` — on every hit, ALSO run the real
//!     transform and byte-compare a sample; mismatches log an error and the
//!     freshly computed buffer is served instead of the cached one.
//!
//! On-disk layout, one directory per fingerprint:
//!   `<root>/<fingerprint>/blob.bin`   — all parts concatenated, 64B aligned
//!   `<root>/<fingerprint>/index.json` — slot → [(offset, len)]
//!
//! One blob plus one index rather than a file per layer: the read side wants a
//! single mmap and one sequential fault-in pass instead of ~600 opens, and the
//! index doubles as the commit marker. It is written last via a temp file and
//! `rename`, so a blob torn by a crash or a failed load has no index and is
//! simply ignored on the next start.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

mod evict;

/// Bump on ANY change to the on-disk layout, to the set of fingerprint
/// inputs, or to the meaning of a slot key. Existing caches then fail the
/// key check and are rewritten rather than silently reused.
pub const CACHE_FORMAT_VERSION: u32 = 2;

/// Parts are padded to this boundary inside the blob so each mmap slice
/// handed to `copy_h2d` starts aligned.
const PART_ALIGN: u64 = 64;

/// Runtime env vars that change what a transform produces. Any of these
/// flipping must produce a different cache key — otherwise a run with
/// `ATLAS_FFN_M16_TRANSPOSED=1` would serve buffers built without it.
/// Adding a transform-affecting env var means adding it here.
const TRANSFORM_ENV_KEYS: &[&str] = &[
    "ATLAS_ATTN_QKV_SPLITK",
    "ATLAS_ATTN_SLIDING_WINDOW",
    "ATLAS_FAKE_NVFP4",
    "ATLAS_FFN_GATEUP_SPLITK",
    "ATLAS_FFN_M16_TRANSPOSED",
    "ATLAS_FFN_PREDEQUANT_FP8",
    "ATLAS_FFN_W3_LAYERS",
    "ATLAS_FFN_W3_SIDECAR",
    "ATLAS_FORCE_NVFP4_MOE",
    "ATLAS_MODEL_PATH",
    "ATLAS_SSM_GDN_LAZY",
    "ATLAS_TARGET_MODEL",
    "ATLAS_TARGET_QUANT",
    "TQ_PLUS_WEIGHT_ROTATION",
];

/// Tensors sampled for the content fingerprint, and bytes read from the
/// start of each. Two checkpoints with identical names/shapes/dtypes (an
/// abliterated re-quant vs the official one, say) differ only in content, so
/// shape metadata alone is not a safe key. Sampling is a heuristic, not a
/// full content hash: it is sized to separate checkpoints that differ
/// broadly, which is what swapping a model variant does.
const CONTENT_SAMPLE_TENSORS: usize = 24;
const CONTENT_SAMPLE_BYTES: usize = 2048;

/// Bytes compared per part under `ATLAS_WEIGHT_CACHE_VERIFY=1`.
const VERIFY_WINDOW: usize = 4096;

// ─────────────────────────── fingerprint ───────────────────────────

/// Non-cryptographic 128-bit fingerprint accumulator (two FNV-1a-64 lanes
/// with different offset bases). Used only to decide whether a cache
/// directory belongs to this exact model + build + env, never for security.
/// Every write is length-prefixed so `"ab" + "c"` and `"a" + "bc"` differ.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    lane_a: u64,
    lane_b: u64,
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::new()
    }
}

impl Fingerprint {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    const BASIS_A: u64 = 0xcbf2_9ce4_8422_2325;
    const BASIS_B: u64 = 0x9e37_79b9_7f4a_7c15;

    pub fn new() -> Self {
        Self {
            lane_a: Self::BASIS_A,
            lane_b: Self::BASIS_B,
        }
    }

    pub fn write(&mut self, bytes: &[u8]) -> &mut Self {
        for b in (bytes.len() as u64).to_le_bytes() {
            self.mix(b);
        }
        for &b in bytes {
            self.mix(b);
        }
        self
    }

    pub fn write_str(&mut self, s: &str) -> &mut Self {
        self.write(s.as_bytes())
    }

    pub fn write_usize(&mut self, v: usize) -> &mut Self {
        self.write(&(v as u64).to_le_bytes())
    }

    pub fn write_u32(&mut self, v: u32) -> &mut Self {
        self.write(&v.to_le_bytes())
    }

    fn mix(&mut self, b: u8) {
        self.lane_a = (self.lane_a ^ b as u64).wrapping_mul(Self::PRIME);
        self.lane_b = (self.lane_b ^ b as u64).wrapping_mul(Self::PRIME);
    }

    /// Stable lowercase hex, 32 chars. This is the cache directory name.
    pub fn finish_hex(&self) -> String {
        format!("{:016x}{:016x}", self.lane_a, self.lane_b)
    }
}

/// Hash the fields of [`atlas_core::config::ModelConfig`] that any weight
/// transform depends on. Listed explicitly rather than derived from `Serialize`
/// (the struct has no `Serialize` impl) — adding a config field that changes a
/// transform means adding it here AND bumping [`CACHE_FORMAT_VERSION`].
pub fn hash_model_config(fp: &mut Fingerprint, config: &atlas_core::config::ModelConfig) {
    fp.write_str("model_config.v1");
    fp.write_usize(config.hidden_size);
    fp.write_usize(config.num_hidden_layers);
    fp.write_usize(config.intermediate_size);
    fp.write_usize(config.vocab_size);
    fp.write_usize(config.num_attention_heads);
    fp.write_usize(config.num_key_value_heads);
    fp.write_usize(config.head_dim);
    fp.write_usize(config.linear_num_key_heads);
    fp.write_usize(config.linear_key_head_dim);
    fp.write_usize(config.linear_num_value_heads);
    fp.write_usize(config.linear_value_head_dim);
    fp.write_usize(config.num_experts);
    fp.write_usize(config.moe_intermediate_size);
    fp.write_usize(config.tp_rank);
    fp.write_usize(config.tp_world_size);
    fp.write_str(&config.weight_prefix);
    fp.write(&[config.attn_gated as u8]);
    fp.write_usize(config.layer_types.len());
    for lt in &config.layer_types {
        fp.write_str(&format!("{lt:?}"));
    }
}

/// Hash the build identity and every transform-affecting env var.
pub fn hash_build_and_env(fp: &mut Fingerprint) {
    fp.write_str("build.v1");
    fp.write_u32(CACHE_FORMAT_VERSION);
    fp.write_str(env!("CARGO_PKG_VERSION"));
    // Kernel target selected at compile time. A binary built for a different
    // ATLAS_TARGET_MODEL can produce different transform outputs, so the
    // build-time value belongs in the key alongside the runtime one.
    fp.write_str(option_env!("ATLAS_TARGET_MODEL").unwrap_or(""));
    fp.write_str(option_env!("ATLAS_TARGET_QUANT").unwrap_or(""));
    for key in TRANSFORM_ENV_KEYS {
        fp.write_str(key);
        fp.write_str(&std::env::var(key).unwrap_or_default());
    }
}

/// Hash tensor names, dtypes and shapes, plus a bounded content sample so
/// two checkpoints with the same geometry but different weights cannot
/// share a cache directory.
fn hash_weight_store(fp: &mut Fingerprint, store: &WeightStore, gpu: &dyn GpuBackend) {
    fp.write_str("store.v1");
    let mut names: Vec<&str> = store.names().collect();
    names.sort_unstable();
    fp.write_usize(names.len());
    for name in &names {
        let Ok(t) = store.get(name) else { continue };
        fp.write_str(name);
        fp.write_str(&format!("{:?}", t.dtype));
        fp.write_usize(t.shape.len());
        for d in &t.shape {
            fp.write_usize(*d);
        }
    }

    // Content sample: leading bytes of evenly spread tensors.
    if names.is_empty() {
        return;
    }
    let stride = names.len().div_ceil(CONTENT_SAMPLE_TENSORS).max(1);
    let mut buf = vec![0u8; CONTENT_SAMPLE_BYTES];
    for name in names.iter().step_by(stride) {
        let Ok(t) = store.get(name) else { continue };
        let size = t.byte_size();
        if size == 0 || t.ptr == DevicePtr::NULL {
            continue;
        }
        let take = CONTENT_SAMPLE_BYTES.min(size);
        if gpu.copy_d2h(t.ptr, &mut buf[..take]).is_ok() {
            fp.write_str(name);
            fp.write_usize(size);
            fp.write(&buf[..take]);
        }
    }
}

/// Full cache key for this process: model content + config + build + env.
pub fn compute_fingerprint(
    store: &WeightStore,
    config: &atlas_core::config::ModelConfig,
    gpu: &dyn GpuBackend,
    variant_tag: &str,
) -> String {
    let mut fp = Fingerprint::new();
    hash_build_and_env(&mut fp);
    fp.write_str("variant");
    fp.write_str(variant_tag);
    hash_model_config(&mut fp, config);
    hash_weight_store(&mut fp, store, gpu);
    fp.finish_hex()
}

// ─────────────────────────── slot geometry ───────────────────────────

/// Byte lengths of the two buffers `QuantizedWeight::transpose_for_gemm`
/// allocates for an [N, K] weight: (packed, scale).
///
/// This MIRRORS the arithmetic in that method — `n_pad` rounds N up to 64,
/// packed is `[K/2, n_pad]`, scale is `[K/16, n_pad]`. `transpose_for_gemm_cached`
/// cross-checks the result against the lengths recorded in the cache index and
/// recomputes on any disagreement, so a drift between the two surfaces as a
/// slow load, never as wrong numerics. Changing the transposed layout means
/// changing both and bumping [`CACHE_FORMAT_VERSION`].
pub fn transposed_lens(n: usize, k: usize) -> (usize, usize) {
    const GROUP_SIZE: usize = 16;
    let n_pad = n.div_ceil(64) * 64;
    ((k / 2) * n_pad, (k / GROUP_SIZE) * n_pad)
}

// ─────────────────────────── index ───────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Part {
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexEntry {
    pub slot: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheIndex {
    pub format_version: u32,
    pub fingerprint: String,
    pub blob_len: u64,
    pub entries: Vec<IndexEntry>,
}

impl CacheIndex {
    /// Reject an index that was written by a different layout or for a
    /// different model. Callers treat `false` as "no cache".
    pub fn matches(&self, fingerprint: &str) -> bool {
        self.format_version == CACHE_FORMAT_VERSION && self.fingerprint == fingerprint
    }

    pub fn to_map(&self) -> HashMap<String, Vec<Part>> {
        self.entries
            .iter()
            .map(|e| (e.slot.clone(), e.parts.clone()))
            .collect()
    }
}

// ─────────────────────────── cache ───────────────────────────

struct Reader {
    mmap: memmap2::Mmap,
    slots: HashMap<String, Vec<Part>>,
}

struct Writer {
    file: BufWriter<File>,
    offset: u64,
    entries: Vec<IndexEntry>,
}

#[derive(Default)]
struct Stats {
    hits: usize,
    misses: usize,
    verified: usize,
    verify_failures: usize,
}

pub struct TransformCache {
    /// Cache root holding every fingerprint directory. Retained so the
    /// eviction pass can see sibling variants.
    root: PathBuf,
    dir: PathBuf,
    fingerprint: String,
    reader: Option<Reader>,
    writer: Mutex<Option<Writer>>,
    verify: bool,
    stats: Mutex<Stats>,
}

impl TransformCache {
    pub fn verify_enabled(&self) -> bool {
        self.verify
    }

    /// Fetch a slot's parts and push them H2D into fresh allocations.
    ///
    /// `expected` is the byte length the caller derived from the model
    /// geometry. Any disagreement with the index is treated as a miss rather
    /// than trusted — that is the backstop against a transform whose output
    /// layout drifted without [`CACHE_FORMAT_VERSION`] being bumped.
    pub fn load_parts(
        &self,
        slot: &str,
        gpu: &dyn GpuBackend,
        expected: &[usize],
    ) -> Option<Vec<DevicePtr>> {
        let reader = self.reader.as_ref()?;
        let parts = reader.slots.get(slot)?;
        if parts.len() != expected.len() {
            tracing::warn!(
                "weight cache: slot {slot} has {} parts, expected {}; recomputing",
                parts.len(),
                expected.len()
            );
            return None;
        }
        let mut out = Vec::with_capacity(parts.len());
        for (part, &want) in parts.iter().zip(expected) {
            if part.len as usize != want {
                tracing::warn!(
                    "weight cache: slot {slot} part len {} != expected {want}; recomputing",
                    part.len
                );
                self.free_all(gpu, &out);
                return None;
            }
            let start = part.offset as usize;
            let end = start + want;
            let Some(src) = reader.mmap.get(start..end) else {
                tracing::warn!("weight cache: slot {slot} runs past blob end; recomputing");
                self.free_all(gpu, &out);
                return None;
            };
            match gpu
                .alloc(want)
                .and_then(|p| gpu.copy_h2d(src, p).map(|()| p))
            {
                Ok(p) => out.push(p),
                Err(e) => {
                    tracing::warn!("weight cache: H2D for slot {slot} failed ({e:#}); recomputing");
                    self.free_all(gpu, &out);
                    return None;
                }
            }
        }
        self.stats.lock().hits += 1;
        Some(out)
    }

    /// Append freshly transformed buffers to the blob. Errors are logged and
    /// swallowed: a cache we failed to write is a slow next start, never a
    /// wrong answer, so it must not fail the load.
    pub fn store_parts(&self, slot: &str, gpu: &dyn GpuBackend, bufs: &[(DevicePtr, usize)]) {
        let mut guard = self.writer.lock();
        let Some(writer) = guard.as_mut() else {
            return;
        };
        self.stats.lock().misses += 1;
        let mut parts = Vec::with_capacity(bufs.len());
        for &(ptr, len) in bufs {
            let pad = (PART_ALIGN - writer.offset % PART_ALIGN) % PART_ALIGN;
            if pad > 0 {
                if let Err(e) = writer.file.write_all(&vec![0u8; pad as usize]) {
                    tracing::warn!("weight cache: pad write failed ({e}); disabling writer");
                    *guard = None;
                    return;
                }
                writer.offset += pad;
            }
            let mut host = vec![0u8; len];
            if let Err(e) = gpu.copy_d2h(ptr, &mut host) {
                tracing::warn!(
                    "weight cache: D2H for slot {slot} failed ({e:#}); disabling writer"
                );
                *guard = None;
                return;
            }
            if let Err(e) = writer.file.write_all(&host) {
                tracing::warn!("weight cache: blob write failed ({e}); disabling writer");
                *guard = None;
                return;
            }
            parts.push(Part {
                offset: writer.offset,
                len: len as u64,
            });
            writer.offset += len as u64;
        }
        writer.entries.push(IndexEntry {
            slot: slot.to_string(),
            parts,
        });
    }

    /// Byte-compare a sample of each part: head window, tail window, and a
    /// deterministic middle window. Returns `false` on any mismatch.
    pub fn verify_pair(
        &self,
        slot: &str,
        gpu: &dyn GpuBackend,
        cached: &[(DevicePtr, usize)],
        fresh: &[(DevicePtr, usize)],
    ) -> bool {
        let mut ok = true;
        for (idx, (&(c_ptr, c_len), &(f_ptr, f_len))) in cached.iter().zip(fresh).enumerate() {
            if c_len != f_len {
                tracing::error!("weight cache VERIFY: {slot}[{idx}] length {c_len} != {f_len}");
                ok = false;
                continue;
            }
            let mut a = vec![0u8; c_len];
            let mut b = vec![0u8; f_len];
            if gpu.copy_d2h(c_ptr, &mut a).is_err() || gpu.copy_d2h(f_ptr, &mut b).is_err() {
                tracing::error!("weight cache VERIFY: {slot}[{idx}] D2H failed");
                ok = false;
                continue;
            }
            for (label, range) in sample_windows(c_len) {
                if a[range.clone()] != b[range.clone()] {
                    tracing::error!(
                        "weight cache VERIFY FAIL: {slot}[{idx}] {label} window {range:?} differs"
                    );
                    ok = false;
                }
            }
        }
        let mut stats = self.stats.lock();
        stats.verified += 1;
        if !ok {
            stats.verify_failures += 1;
        } else {
            tracing::debug!("weight cache VERIFY PASS: {slot}");
        }
        ok
    }

    fn free_all(&self, gpu: &dyn GpuBackend, ptrs: &[DevicePtr]) {
        for &p in ptrs {
            let _ = gpu.free(p);
        }
    }

    /// Stamp this cache as used, then prune siblings back under budget unless
    /// `ATLAS_WEIGHT_CACHE_EVICT=0` opts out.
    ///
    /// The `last_used` stamp is written unconditionally, including under the
    /// opt-out. It costs one small file write and it means recency data is
    /// already accurate whenever pruning is switched back on, instead of every
    /// directory looking equally cold on the first pass afterwards.
    ///
    /// Called only once the active directory is known-good (a hit, or a build
    /// whose index has just been published), so a variant we are about to
    /// serve from can never be chosen as a victim. Best-effort throughout:
    /// a cache we could not prune is a disk-space problem, never a
    /// correctness one, and must not fail a load that already succeeded.
    fn touch_and_evict(&self) {
        let now = evict::unix_now();
        evict::touch_last_used(&self.dir, now);
        if let Err(e) = evict::run(
            &self.root,
            &self.fingerprint,
            evict::budget_bytes(),
            evict::keep_limit(),
            now,
            evict::eviction_enabled(),
        ) {
            tracing::warn!("weight cache: eviction pass failed ({e:#})");
        }
    }

    /// Flush the blob and publish the index. Called once after all layers
    /// are built; skipping it (because the load failed) leaves the blob
    /// without an index, which the next start ignores.
    pub fn finish(&self) {
        let stats = self.stats.lock();
        if self.verify {
            tracing::info!(
                "weight cache VERIFY summary: {} slots checked, {} failures",
                stats.verified,
                stats.verify_failures,
            );
        }
        let mut guard = self.writer.lock();
        let Some(mut writer) = guard.take() else {
            tracing::info!(
                "weight cache: {} hits, {} misses (read-only, {})",
                stats.hits,
                stats.misses,
                self.dir.display(),
            );
            return;
        };
        let entries = std::mem::take(&mut writer.entries);
        let blob_len = writer.offset;
        if let Err(e) = writer
            .file
            .flush()
            .and_then(|()| writer.file.get_ref().sync_all())
        {
            tracing::warn!("weight cache: blob sync failed ({e}); not publishing index");
            return;
        }
        let index = CacheIndex {
            format_version: CACHE_FORMAT_VERSION,
            fingerprint: self.fingerprint.clone(),
            blob_len,
            entries,
        };
        if let Err(e) = write_index(&self.dir, &index) {
            tracing::warn!("weight cache: index write failed ({e:#}); cache not published");
            return;
        }
        tracing::info!(
            "weight cache WRITTEN: {} slots, {:.2} GiB at {}",
            index.entries.len(),
            blob_len as f64 / (1024.0 * 1024.0 * 1024.0),
            self.dir.display(),
        );
        // Only now is this directory a committed cache, so only now is it
        // safe to count it against the budget and prune the older variants.
        self.touch_and_evict();
    }
}

/// Head / middle / tail windows used by verify mode.
fn sample_windows(len: usize) -> Vec<(&'static str, std::ops::Range<usize>)> {
    if len == 0 {
        return Vec::new();
    }
    let w = VERIFY_WINDOW.min(len);
    let mid_start = (len / 2).saturating_sub(w / 2).min(len - w);
    vec![
        ("head", 0..w),
        ("mid", mid_start..mid_start + w),
        ("tail", len - w..len),
    ]
}

fn write_index(dir: &std::path::Path, index: &CacheIndex) -> Result<()> {
    let tmp = dir.join("index.json.tmp");
    let final_path = dir.join("index.json");
    let bytes = serde_json::to_vec(index).context("serialize cache index")?;
    fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, &final_path).with_context(|| format!("publish {}", final_path.display()))?;
    Ok(())
}

// ─────────────────────────── process-wide handle ───────────────────────────

static CACHE: OnceLock<Option<TransformCache>> = OnceLock::new();

fn env_flag(key: &str) -> bool {
    std::env::var(key).ok().as_deref() == Some("1")
}

fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("ATLAS_WEIGHT_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("atlas-weight-cache")
}

/// Open (or create) the cache for this model. Idempotent — the first call
/// wins, later calls are no-ops. Returns without doing anything when
/// `ATLAS_WEIGHT_CACHE` is not `1`.
pub fn init(
    store: &WeightStore,
    config: &atlas_core::config::ModelConfig,
    gpu: &dyn GpuBackend,
    variant_tag: &str,
) {
    if CACHE.get().is_some() {
        return;
    }
    if !env_flag("ATLAS_WEIGHT_CACHE") {
        let _ = CACHE.set(None);
        return;
    }
    let verify = env_flag("ATLAS_WEIGHT_CACHE_VERIFY");
    let fingerprint = compute_fingerprint(store, config, gpu, variant_tag);
    let cache = match open(cache_root(), fingerprint, verify) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!("weight cache: disabled — {e:#}");
            None
        }
    };
    let _ = CACHE.set(cache);
    // On a hit the cache is usable the moment `open` returns, so mark it used
    // and prune siblings now. On a miss the active directory is not yet
    // committed — `finish` runs the pass once the index is published, so an
    // interrupted build never counts as a live cache worth keeping.
    if let Some(cache) = get()
        && cache.reader.is_some()
    {
        cache.touch_and_evict();
    }
}

fn open(root: PathBuf, fingerprint: String, verify: bool) -> Result<TransformCache> {
    let dir = root.join(&fingerprint);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let blob_path = dir.join("blob.bin");
    let index_path = dir.join("index.json");

    // Read path: an index is only present when a previous run completed.
    if let Ok(bytes) = fs::read(&index_path)
        && let Ok(index) = serde_json::from_slice::<CacheIndex>(&bytes)
        && index.matches(&fingerprint)
        && let Ok(file) = File::open(&blob_path)
    {
        // SAFETY: the blob is written once and published by rename; nothing
        // in this process mutates it while the mmap is alive.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap {}", blob_path.display()))?;
        if mmap.len() as u64 >= index.blob_len {
            tracing::info!(
                "weight cache HIT: {} slots, {:.2} GiB at {}{}",
                index.entries.len(),
                index.blob_len as f64 / (1024.0 * 1024.0 * 1024.0),
                dir.display(),
                if verify { " (VERIFY on)" } else { "" },
            );
            return Ok(TransformCache {
                root,
                dir,
                fingerprint,
                reader: Some(Reader {
                    slots: index.to_map(),
                    mmap,
                }),
                writer: Mutex::new(None),
                verify,
                stats: Mutex::new(Stats::default()),
            });
        }
        tracing::warn!("weight cache: blob shorter than index claims; rebuilding");
    }

    // Write path: truncate any stale blob and drop any stale index so a
    // crash mid-write cannot leave an index pointing at new bytes.
    let _ = fs::remove_file(&index_path);
    let file =
        File::create(&blob_path).with_context(|| format!("create {}", blob_path.display()))?;
    tracing::info!("weight cache MISS: building at {}", dir.display());
    Ok(TransformCache {
        root,
        dir,
        fingerprint,
        reader: None,
        writer: Mutex::new(Some(Writer {
            file: BufWriter::with_capacity(8 * 1024 * 1024, file),
            offset: 0,
            entries: Vec::new(),
        })),
        verify,
        stats: Mutex::new(Stats::default()),
    })
}

/// The process-wide cache, or `None` when the feature is off.
pub fn get() -> Option<&'static TransformCache> {
    CACHE.get().and_then(|c| c.as_ref())
}

/// Publish the cache. Call only after every layer has loaded successfully.
pub fn finish() {
    if let Some(cache) = get() {
        cache.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_and_length_prefixed() {
        let mut a = Fingerprint::new();
        a.write_str("ab").write_str("c");
        let mut b = Fingerprint::new();
        b.write_str("a").write_str("bc");
        assert_ne!(
            a.finish_hex(),
            b.finish_hex(),
            "length prefixing must stop concatenation collisions"
        );

        let mut c = Fingerprint::new();
        c.write_str("ab").write_str("c");
        assert_eq!(a.finish_hex(), c.finish_hex(), "same input, same digest");
        assert_eq!(a.finish_hex().len(), 32);
        assert!(a.finish_hex().chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn fingerprint_distinguishes_scalar_kinds() {
        let mut a = Fingerprint::new();
        a.write_usize(1);
        let mut b = Fingerprint::new();
        b.write_u32(1);
        assert_ne!(
            a.finish_hex(),
            b.finish_hex(),
            "u32 and usize widths differ"
        );
    }

    #[test]
    fn build_env_change_changes_fingerprint() {
        // SAFETY: single-threaded test; no other thread reads the env here.
        unsafe { std::env::set_var("ATLAS_FFN_M16_TRANSPOSED", "0") };
        let mut a = Fingerprint::new();
        hash_build_and_env(&mut a);
        unsafe { std::env::set_var("ATLAS_FFN_M16_TRANSPOSED", "1") };
        let mut b = Fingerprint::new();
        hash_build_and_env(&mut b);
        unsafe { std::env::remove_var("ATLAS_FFN_M16_TRANSPOSED") };
        assert_ne!(
            a.finish_hex(),
            b.finish_hex(),
            "a transform-affecting env var must change the key"
        );
    }

    fn sample_index() -> CacheIndex {
        CacheIndex {
            format_version: CACHE_FORMAT_VERSION,
            fingerprint: "deadbeefdeadbeefdeadbeefdeadbeef".to_string(),
            blob_len: 320,
            entries: vec![
                IndexEntry {
                    slot: "L0.self_attn.q_proj.t".to_string(),
                    parts: vec![
                        Part {
                            offset: 0,
                            len: 128,
                        },
                        Part {
                            offset: 128,
                            len: 64,
                        },
                    ],
                },
                IndexEntry {
                    slot: "L0.self_attn.k_proj.t".to_string(),
                    parts: vec![Part {
                        offset: 192,
                        len: 128,
                    }],
                },
            ],
        }
    }

    #[test]
    fn index_round_trips_through_json() {
        let index = sample_index();
        let bytes = serde_json::to_vec(&index).unwrap();
        let back: CacheIndex = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(index, back);
    }

    #[test]
    fn index_rejects_wrong_key_and_version() {
        let index = sample_index();
        assert!(index.matches(&index.fingerprint));
        assert!(!index.matches("0000000000000000000000000000abcd"));

        let stale = CacheIndex {
            format_version: CACHE_FORMAT_VERSION + 1,
            ..index.clone()
        };
        assert!(!stale.matches(&index.fingerprint));
    }

    #[test]
    fn index_map_preserves_part_order() {
        let map = sample_index().to_map();
        let q = map.get("L0.self_attn.q_proj.t").unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!(
            q[0],
            Part {
                offset: 0,
                len: 128
            }
        );
        assert_eq!(
            q[1],
            Part {
                offset: 128,
                len: 64
            }
        );
        assert!(map.contains_key("L0.self_attn.k_proj.t"));
        assert!(!map.contains_key("L1.self_attn.q_proj.t"));
    }

    #[test]
    fn verify_windows_cover_head_mid_tail_without_overflow() {
        for len in [1usize, 100, VERIFY_WINDOW, VERIFY_WINDOW * 3 + 7] {
            for (label, range) in sample_windows(len) {
                assert!(
                    range.end <= len,
                    "{label} window {range:?} escapes len {len}"
                );
                assert!(range.start < range.end, "{label} window is empty");
            }
        }
        assert!(sample_windows(0).is_empty());
    }

    #[test]
    fn transposed_lens_matches_transpose_for_gemm_layout() {
        // N already 64-aligned: n_pad == n, so packed is n*k/2 and scale n*k/16.
        assert_eq!(transposed_lens(128, 5120), (128 * 2560, 128 * 320));
        // qwen3.8-27b attention O proj: N = hidden = 5120, K = heads*head_dim.
        assert_eq!(transposed_lens(5120, 4096), (5120 * 2048, 5120 * 256));
        // Odd N (248077-vocab lm_head class) pads up to the next multiple of 64.
        let (w, s) = transposed_lens(100, 64);
        assert_eq!(w, 32 * 128, "n_pad must be 128, not 100");
        assert_eq!(s, 4 * 128);
    }

    #[test]
    fn transposed_lens_padding_never_shrinks_the_buffer() {
        for n in [1usize, 63, 64, 65, 4096, 5120, 248077] {
            let (w, s) = transposed_lens(n, 1024);
            assert!(w >= n * 512, "packed buffer must cover all {n} rows");
            assert!(s >= n * 64, "scale buffer must cover all {n} rows");
        }
    }

    #[test]
    fn part_alignment_padding_is_computed_correctly() {
        // Mirrors the arithmetic in store_parts.
        for offset in [0u64, 1, 63, 64, 65, 4096, 4097] {
            let pad = (PART_ALIGN - offset % PART_ALIGN) % PART_ALIGN;
            assert_eq!((offset + pad) % PART_ALIGN, 0);
            assert!(pad < PART_ALIGN);
        }
    }
}
