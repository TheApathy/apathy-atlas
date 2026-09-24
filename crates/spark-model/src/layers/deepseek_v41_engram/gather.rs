// SPDX-License-Identifier: AGPL-3.0-only

//! The `Dsv41Engram` seam implementation: row ids in, dequantized rows on the device.
//!
//! This is the narrow half of the lane. The engine hands us `[num_tokens, 24]` host-side
//! row ids (produced by [`super::hash::EngramHashState`]) and a device buffer; we pull the
//! rows off NVMe, dequantize, and upload.
//!
//! ## Two contracts that are easy to get wrong
//!
//! * **Dead-head masking stays OUT of here.** The reference applies `masked_fill` AFTER the
//!   fetch, so this returns PRE-mask rows and the mask travels separately. Folding the mask
//!   in here would change the values on the first multimodal prompt and nowhere else, which
//!   is the worst possible place for a divergence to first appear.
//! * **Decode reads are NOT prefetchable.** The n-gram at position p includes the token just
//!   sampled, so the row ids for step N are unknowable until step N-1 has sampled. The
//!   ~1.2 ms sits on the critical path and cannot be hidden behind the previous step.
//!
//! ## Cost
//!
//! Measured on this box, cold, `/proc/diskstats`-verified at 4684 device bytes per requested
//! row (one 4 KiB page per 256 B row): 104 ms per 2048-token prefill chunk (a 19,637 tok/s
//! ceiling) and 1.21 ms per decode token (827 tok/s). Against a ~1190 tok/s prefill engine
//! and a ~30 ms decode step, engram is ~4% and nothing should be contorted to avoid it.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use spark_storage::engram_tier::{
    DEFAULT_ENGRAM_THREADS, ENGRAM_HEAD_DIM, ENGRAM_ROW_BYTES, EngramShard, EngramTier,
    dequant_row,
};

use crate::layer::ForwardContext;
use crate::weight_loader::deepseek_v41::seams::{
    Dsv41Engram, Dsv41EngramLoader, ENGRAM_LAYERS, ENGRAM_ROWS_PER_TOKEN, ENGRAM_ROW_DIM,
    register_engram_loader,
};

/// One engram layer's table, behind a batched NVMe row gather.
pub struct EngramGather {
    tier: EngramTier,
    layer: usize,
}

impl EngramGather {
    /// Open the shard holding `layer`'s table. The 95 GB tensor is never read into the
    /// `WeightStore` — it is row-gathered on demand — so this takes the model directory
    /// and goes to the file itself.
    pub fn open(model_dir: &Path, layer: usize, threads: usize) -> Result<Self> {
        let map = read_weight_map(model_dir)?;
        let key = format!("layers.{layer}.engram.embed.weight");
        let file = map
            .get(&key)
            .with_context(|| format!("model.safetensors.index.json has no entry for {key}"))?;
        let shard = EngramShard::open(&model_dir.join(file), layer as u32)?;
        let tier = EngramTier::new(vec![shard], threads)?;
        Ok(Self { tier, layer })
    }
}

impl Dsv41Engram for EngramGather {
    fn gather_rows(
        &self,
        row_ids: &[i64],
        num_tokens: usize,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.gather_rows_gpu(row_ids, num_tokens, out, ctx.gpu, stream)
    }
}

impl EngramGather {
    /// [`Dsv41Engram::gather_rows`] without a `ForwardContext` — it only ever needed the
    /// GPU handle — so drivers and tests outside the layer loop can call it.
    pub fn gather_rows_gpu(
        &self,
        row_ids: &[i64],
        num_tokens: usize,
        out: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let host = self.gather_rows_host(row_ids, num_tokens)?;
        Self::upload_rows(&host, out, gpu, stream, self.layer)
    }

    /// The NVMe-bound half of [`Self::gather_rows_gpu`], with no device involved at all: dedup,
    /// pread, dequantize, return the host buffer. Split out so a caller can run this on a
    /// background thread (the disk read and dequant are the ~73 ms/chunk cost; the upload is
    /// microseconds by comparison) and only pay [`Self::upload_rows`] on the critical path —
    /// see `V41Forward::prefetch_engram`.
    pub fn gather_rows_host(&self, row_ids: &[i64], num_tokens: usize) -> Result<Vec<f32>> {
        let want = num_tokens * ENGRAM_ROWS_PER_TOKEN;
        if row_ids.len() != want {
            bail!(
                "engram layer {}: got {} row ids, expected {num_tokens} x {ENGRAM_ROWS_PER_TOKEN} = {want}",
                self.layer,
                row_ids.len()
            );
        }
        if ENGRAM_ROW_DIM != ENGRAM_HEAD_DIM {
            bail!("seam row dim {ENGRAM_ROW_DIM} != table head dim {ENGRAM_HEAD_DIM}");
        }
        if row_ids.iter().any(|&r| r < 0) {
            bail!("engram layer {}: negative row id", self.layer);
        }

        // Dedup before touching the disk: a duplicate costs a full 4 KiB page fault, and
        // real prompts repeat ~23% of their rows (2-grams most, 4-grams least).
        let ids: Vec<u64> = row_ids.iter().map(|&r| r as u64).collect();
        let (raw, inverse) = self.tier.gather_dedup(0, &ids)?;

        let mut host = vec![0f32; want * ENGRAM_HEAD_DIM];
        for (k, &iv) in inverse.iter().enumerate() {
            let src = &raw[iv as usize * ENGRAM_ROW_BYTES..(iv as usize + 1) * ENGRAM_ROW_BYTES];
            let dst = &mut host[k * ENGRAM_HEAD_DIM..(k + 1) * ENGRAM_HEAD_DIM];
            dequant_row(src, dst)?;
        }
        Ok(host)
    }

    /// Upload an already-dequantized host row buffer (from [`Self::gather_rows_host`], live or
    /// prefetched) to the device. `layer` is only for the error message.
    pub fn upload_rows(host: &[f32], out: DevicePtr, gpu: &dyn GpuBackend, stream: u64, layer: usize) -> Result<()> {
        // SAFETY-adjacent note: `host` is f32 and `out` is documented as f32, so the byte
        // count is the contract. A dtype change on either side must change both.
        let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host)) };
        gpu.copy_h2d_async(bytes, out, stream)
            .with_context(|| format!("engram layer {layer}: H2D of {} bytes", bytes.len()))
    }
}

/// Reads `model.safetensors.index.json` -> `{tensor name: shard file}`.
fn read_weight_map(model_dir: &Path) -> Result<HashMap<String, String>> {
    let p = model_dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", p.display()))?;
    let wm = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .with_context(|| format!("{}: no weight_map object", p.display()))?;
    Ok(wm
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect())
}

/// Builds an [`EngramGather`] for layers 1 and 14, and `None` for the other 38.
pub struct EngramLoader {
    threads: usize,
}

impl Dsv41EngramLoader for EngramLoader {
    fn load_engram(
        &self,
        layer: usize,
        model_dir: &Path,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<Box<dyn Dsv41Engram>>> {
        if !ENGRAM_LAYERS.contains(&layer) {
            return Ok(None);
        }
        Ok(Some(Box::new(EngramGather::open(model_dir, layer, self.threads)?)))
    }
}

static LOADER: OnceLock<EngramLoader> = OnceLock::new();

/// Install this lane's engram loader. Call before `load_layers`.
///
/// The model directory arrives per-call on [`Dsv41EngramLoader::load_engram`] rather than
/// being captured here: the tables are ~95 GB each, are never in the `WeightStore`, and are
/// opened as files. The seam names that source explicitly so the signature cannot drift away
/// from where the bytes actually come from.
///
/// Note the initialisation and the registration are deliberately two statements. Calling
/// `register_engram_loader` from inside `get_or_init` would take the registry's write lock
/// while holding the `OnceLock`'s initialisation lock, which is the shape that deadlocks.
pub fn register(threads: Option<usize>) {
    let loader =
        LOADER.get_or_init(|| EngramLoader { threads: threads.unwrap_or(DEFAULT_ENGRAM_THREADS) });
    register_engram_loader(loader);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_map_parses_and_reports_a_missing_file() {
        let d = std::env::temp_dir().join(format!("engram-wm-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("model.safetensors.index.json"),
            r#"{"weight_map":{"layers.1.engram.embed.weight":"shard-a.safetensors"}}"#,
        )
        .unwrap();
        let m = read_weight_map(&d).unwrap();
        assert_eq!(m.get("layers.1.engram.embed.weight").map(String::as_str), Some("shard-a.safetensors"));

        // The shard named by the map does not exist, so opening must fail rather than
        // yield an engram that returns zeros.
        assert!(EngramGather::open(&d, 1, 4).is_err());
        // ...and a layer the map does not mention fails for a DIFFERENT reason, naming it.
        let e = match EngramGather::open(&d, 14, 4) {
            Ok(_) => panic!("layer 14 is absent from the map and must not open"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("layers.14.engram.embed.weight"), "unhelpful error: {e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn missing_index_json_is_an_error_not_an_empty_map() {
        let d = std::env::temp_dir().join(format!("engram-noidx-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert!(read_weight_map(&d).is_err(), "a missing index must not read as an empty map");
        std::fs::remove_dir_all(&d).ok();
    }

    /// The loader must return `None` off the engram layers — that is the correct answer for
    /// 38 of 40 layers, and the one place an empty success is not a silent gap.
    /// `config` is only a carrier here — `load_engram` reads nothing from it, so any
    /// valid ModelConfig exercises the same path.
    #[test]
    fn non_engram_layers_return_none_without_touching_the_disk() {
        let l = EngramLoader { threads: 4 };
        let dir = Path::new("/nonexistent");
        let cfg = ModelConfig::qwen3_next_80b_nvfp4();
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        for layer in [0usize, 2, 13, 15, 39] {
            let got = l.load_engram(layer, dir, &cfg, &gpu).unwrap();
            assert!(got.is_none(), "layer {layer} should have no engram");
        }
        // And the engram layers DO try (and fail, since the path is bogus) — otherwise the
        // check above would pass for a loader that returns None for everything.
        for layer in ENGRAM_LAYERS {
            assert!(
                l.load_engram(layer, dir, &cfg, &gpu).is_err(),
                "layer {layer} must attempt to open its table"
            );
        }
    }
}
