// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler side of on-disk context checkpoints (`ATLAS_CTX_CACHE=1`).
//!
//! - After the last chunk of a prompt prefill, the model's full state is
//!   captured to host memory and written to NVMe on a background thread.
//! - When a new prompt starts with a checkpointed prefix of `C` tokens, the
//!   state is restored into the fresh sequence and prefill starts at `C`.
//!
//! Exactness: a restored run equals a full prefill whose chunk boundaries
//! include `C` (`ATLAS_CTX_CACHE_MODE=reference` produces exactly that
//! reference run from the same checkpoint index, for the byte-exact gate).
//! Like any extra chunk boundary it is not bitwise identical to a prefill
//! that never splits at `C`.
//!
//! Knobs: `ATLAS_CTX_CACHE_DIR` (default `~/.cache/atlas-ctx`),
//! `ATLAS_CTX_CACHE_GB` (disk budget, default 16), `ATLAS_CTX_MIN_TOKENS`
//! (smallest prompt worth checkpointing, default 2048),
//! `ATLAS_CTX_MAX_GB` (largest single checkpoint, default 8).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use parking_lot::RwLock;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::ctx_store::{CtxStore, KeyBuilder, hex, prefix_hash};
use spark_runtime::gpu::DevicePtr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Restore checkpoints and write new ones.
    Restore,
    /// Gate reference: never restore or write; split the prefill at the
    /// checkpoint depth a restore would have used.
    Reference,
}

struct CtxCache {
    store: CtxStore,
    mode: Mode,
    min_tokens: usize,
    max_bytes: u64,
}

static CTX: RwLock<Option<CtxCache>> = RwLock::new(None);

fn env_num(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Inputs that identify the numerics of this server, beyond the model's
/// own geometry and the `ATLAS_*` environment.
pub(crate) struct ServeIdentity<'a> {
    pub model_dir: &'a std::path::Path,
    pub draft_model: Option<&'a str>,
    pub kernel_target: &'a str,
    pub kv_cache_dtype: &'a str,
    pub block_size: usize,
    pub max_prefill_tokens: usize,
}

/// (Re)initialise for the model just loaded. Called on every model load so
/// a swap never reuses another model's index.
pub(crate) fn init(model: &dyn Model, id: &ServeIdentity<'_>) {
    *CTX.write() = None;
    if std::env::var("ATLAS_CTX_CACHE").as_deref() != Ok("1") {
        return;
    }
    match build(model, id) {
        Ok(Some(c)) => {
            tracing::info!(
                "ctx-cache: enabled ({:?}) dir={} key={} entries={} ({:.2} GiB, budget {:.0} GiB)",
                c.mode,
                c.store.dir().display(),
                &hex(c.store.key())[..16],
                c.store.entries().len(),
                c.store.total_bytes() as f64 / (1u64 << 30) as f64,
                env_num("ATLAS_CTX_CACHE_GB", 16.0),
            );
            *CTX.write() = Some(c);
        }
        Ok(None) => {}
        Err(e) => tracing::warn!("ctx-cache: disabled: {e:#}"),
    }
}

fn build(model: &dyn Model, id: &ServeIdentity<'_>) -> Result<Option<CtxCache>> {
    let Some(geometry) = model.ctx_geometry() else {
        return Ok(None);
    };
    if model.is_ep() {
        tracing::info!("ctx-cache: unsupported under expert parallelism");
        return Ok(None);
    }
    let t0 = Instant::now();
    let mut k = KeyBuilder::new();
    k.add("engine", &spark_runtime::ctx_store::engine_digest()?)
        .add("model", &spark_runtime::ctx_store::dir_fingerprint(id.model_dir)?)
        .add("geometry", geometry.as_bytes())
        .add("kernel_target", id.kernel_target.as_bytes())
        .add("kv_dtype", id.kv_cache_dtype.as_bytes())
        .add("block_size", &id.block_size.to_le_bytes())
        .add("max_prefill_tokens", &id.max_prefill_tokens.to_le_bytes());
    if let Some(d) = id.draft_model {
        let p = std::path::Path::new(d);
        let fp = if p.is_dir() {
            spark_runtime::ctx_store::dir_fingerprint(p)?.to_vec()
        } else {
            d.as_bytes().to_vec()
        };
        k.add("draft_model", &fp);
    }
    for (name, value) in spark_runtime::ctx_store::recipe_env() {
        k.add(&name, value.as_bytes());
    }
    let key = k.finish();
    let root = std::env::var("ATLAS_CTX_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".cache/atlas-ctx")
        });
    let budget = (env_num("ATLAS_CTX_CACHE_GB", 16.0) * (1u64 << 30) as f64) as u64;
    let store = CtxStore::open(&root, key, budget)?;
    let mode = match std::env::var("ATLAS_CTX_CACHE_MODE").as_deref() {
        Ok("reference") => Mode::Reference,
        _ => Mode::Restore,
    };
    tracing::info!(
        "ctx-cache: identity computed in {:.0} ms ({geometry})",
        t0.elapsed().as_secs_f64() * 1e3
    );
    Ok(Some(CtxCache {
        store,
        mode,
        min_tokens: env_num("ATLAS_CTX_MIN_TOKENS", 2048.0) as usize,
        max_bytes: (env_num("ATLAS_CTX_MAX_GB", 8.0) * (1u64 << 30) as f64) as u64,
    }))
}

/// Where the first prefill chunk starts and where it must stop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefillStart {
    /// First token to prefill (the restored depth, or 0).
    pub start: usize,
    /// Reference mode: a chunk boundary is forced here (0 = none).
    pub split_at: usize,
}

/// First-chunk length honouring a forced split.
pub(crate) fn first_chunk_len(start: PrefillStart, total: usize, max_prefill: usize) -> usize {
    let len = (total - start.start).min(max_prefill);
    clamp_to_split(start.start, len, start.split_at)
}

/// Shorten a chunk at `offset` so it ends exactly at `split_at` when it
/// would otherwise cross it. Applied after any alignment rounding: the
/// split point is the end of an earlier prompt and must be hit exactly.
pub(crate) fn clamp_to_split(offset: usize, len: usize, split_at: usize) -> usize {
    if split_at > offset && offset + len > split_at {
        split_at - offset
    } else {
        len
    }
}

/// Look up `prompt`; restore into `seq` on a hit. Never fails the request:
/// any problem logs, leaves (or re-creates) a fresh `seq`, and prefills
/// from 0.
pub(crate) fn try_restore(
    model: &dyn Model,
    prompt: &[u32],
    seq: &mut SequenceState,
    stream: u64,
) -> PrefillStart {
    let guard = CTX.read();
    let Some(cc) = guard.as_ref() else {
        return PrefillStart::default();
    };
    let t0 = Instant::now();
    let Some(entry) = cc.store.lookup(prompt) else {
        return PrefillStart::default();
    };
    if cc.mode == Mode::Reference {
        tracing::info!(
            "ctx-cache: REFERENCE mode: checkpoint at {} of {} tokens not restored; \
             forcing a chunk boundary there",
            entry.c,
            prompt.len()
        );
        return PrefillStart { start: 0, split_at: entry.c };
    }
    let snap = match cc.store.load(&entry, prompt) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                "ctx-cache: checkpoint {} REJECTED and deleted ({e:#}); full prefill",
                entry.path.display()
            );
            return PrefillStart::default();
        }
    };
    let read_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    match model.ctx_restore(&snap, seq, stream) {
        Ok(()) => {
            let c = entry.c;
            seq.prompt_len = prompt.len();
            tracing::info!(
                "ctx-cache: RESTORED {c} of {} tokens ({:.1} MiB): read+verify {read_ms:.0} ms, \
                 upload {:.0} ms; prefilling {} suffix tokens",
                prompt.len(),
                entry.bytes as f64 / (1 << 20) as f64,
                t1.elapsed().as_secs_f64() * 1e3,
                prompt.len() - c,
            );
            PrefillStart { start: c, split_at: 0 }
        }
        Err(e) => {
            tracing::warn!("ctx-cache: restore failed ({e:#}); full prefill");
            let session = seq.session_hash;
            if let Err(fe) = model.free_sequence(seq) {
                tracing::error!("ctx-cache: free after failed restore: {fe:#}");
            }
            match model.alloc_sequence() {
                Ok(fresh) => {
                    *seq = fresh;
                    seq.session_hash = session;
                }
                Err(ae) => tracing::error!("ctx-cache: re-alloc after failed restore: {ae:#}"),
            }
            PrefillStart::default()
        }
    }
}

/// Capture the post-prefill state of `seq` and queue it for writing.
/// Call after the LAST prefill chunk (and its SSM normalisation), before
/// the first decode step touches the state.
pub(crate) fn after_prefill(model: &dyn Model, seq: &SequenceState, stream: u64) {
    let guard = CTX.read();
    let Some(cc) = guard.as_ref() else {
        return;
    };
    if cc.mode != Mode::Restore || seq.seq_len < cc.min_tokens {
        return;
    }
    let hash = prefix_hash(&seq.tokens);
    if cc
        .store
        .entries()
        .iter()
        .any(|e| e.c == seq.seq_len && e.hash == hash)
    {
        return;
    }
    if cc.store.is_busy() {
        tracing::info!(
            "ctx-cache: writer busy; checkpoint of {} tokens skipped",
            seq.seq_len
        );
        return;
    }
    let t0 = Instant::now();
    match model.ctx_capture(seq, stream) {
        Ok(Some(snap)) => {
            let bytes = snap.state_bytes() as u64;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if bytes > cc.max_bytes {
                tracing::warn!(
                    "ctx-cache: checkpoint of {} tokens is {:.2} GiB > ATLAS_CTX_MAX_GB; dropped",
                    seq.seq_len,
                    bytes as f64 / (1u64 << 30) as f64
                );
                return;
            }
            tracing::info!(
                "ctx-cache: captured {} tokens ({:.1} MiB) in {ms:.0} ms; writing in background",
                seq.seq_len,
                bytes as f64 / (1 << 20) as f64
            );
            cc.store.submit(snap);
        }
        Ok(None) => {}
        Err(e) => tracing::warn!("ctx-cache: capture failed: {e:#}"),
    }
}

/// Wait for an in-flight checkpoint write (gate harness, shutdown).
pub(crate) fn wait_idle(timeout: Duration) -> bool {
    CTX.read()
        .as_ref()
        .is_none_or(|c| c.store.wait_idle(timeout))
}

static DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Gate hook: `ATLAS_CTX_GATE_DUMP=<dir>` writes the last-token prefill
/// logits bytes to `<dir>/<n>-<prompt_len>.bin`, in request order.
pub(crate) fn dump_logits(model: &dyn Model, logits: DevicePtr, prompt_len: usize) {
    let Ok(dir) = std::env::var("ATLAS_CTX_GATE_DUMP") else {
        return;
    };
    let width = if model.logits_ptr_is_fp32(logits) { 4 } else { 2 };
    let mut buf = vec![0u8; model.vocab_size() * width];
    let n = DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = PathBuf::from(dir).join(format!("{n:03}-{prompt_len}.bin"));
    let r = model
        .copy_logits_to_host(logits, &mut buf)
        .and_then(|_| std::fs::write(&path, &buf).map_err(Into::into));
    match r {
        Ok(()) => tracing::info!("ctx-cache: gate logits -> {}", path.display()),
        Err(e) => tracing::warn!("ctx-cache: gate logits dump failed: {e:#}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_clamps_only_a_crossing_chunk() {
        assert_eq!(clamp_to_split(0, 8192, 5000), 5000);
        assert_eq!(clamp_to_split(0, 4000, 5000), 4000);
        assert_eq!(clamp_to_split(4096, 8192, 5001), 905);
        assert_eq!(clamp_to_split(5001, 8192, 5001), 8192);
        assert_eq!(clamp_to_split(0, 8192, 0), 8192);
    }

    #[test]
    fn first_chunk_starts_at_the_restored_depth() {
        let restored = PrefillStart { start: 5001, split_at: 0 };
        assert_eq!(first_chunk_len(restored, 6000, 8192), 999);
        assert_eq!(first_chunk_len(restored, 20000, 8192), 8192);
        let reference = PrefillStart { start: 0, split_at: 5001 };
        assert_eq!(first_chunk_len(reference, 6000, 8192), 5001);
        assert_eq!(first_chunk_len(PrefillStart::default(), 6000, 8192), 6000);
    }
}
