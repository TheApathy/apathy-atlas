// SPDX-License-Identifier: AGPL-3.0-only

//! W3 FFN sidecar loader — mixed-precision byte-reduction lane.
//!
//! Reads the offline-repacked 3-bit FFN weights produced by
//! `local/tools/repack_w3.py` and uploads them (plus host-transposed GEMM
//! copies) for the layers named in `ATLAS_FFN_W3_LAYERS`.
//!
//! Gating (ALL required — otherwise the default W4 path is untouched):
//!   * `ATLAS_FFN_W3_LAYERS="3,7,12"` (or `a-b` ranges) names the layers
//!   * `ATLAS_FFN_W3_SIDECAR=/path/to/w3_ffn_sidecar.safetensors` points
//!     at the sidecar file (explicit — the loader has no model-dir handle)
//!   * the sidecar actually contains all 9 tensors for the layer
//!
//! Sidecar tensor names, per layer prefix `lp` (e.g.
//! `model.language_model.layers.3`) and proj in {gate,up,down}_proj:
//!   `{lp}.mlp.{proj}.w3_weight`         U8  [N, 3K/8]
//!   `{lp}.mlp.{proj}.w3_weight_scale`   U8  [N, K/16] (FP8-E4M3 bytes)
//!   `{lp}.mlp.{proj}.w3_weight_scale_2` F32 [1]
//!
//! Failure policy: fail-open with a warning — a missing/corrupt sidecar or
//! shape mismatch logs and returns `None`, leaving the layer on W4.

use std::collections::BTreeSet;
use std::io::Read;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{W3_GROUP_SIZE, parse_w3_layer_set, w3_row_bytes, w3_transpose_host};

use super::QuantizedWeight;

/// Layer set parsed from `ATLAS_FFN_W3_LAYERS` (empty when unset).
pub fn w3_layer_set() -> &'static BTreeSet<usize> {
    static SET: OnceLock<BTreeSet<usize>> = OnceLock::new();
    SET.get_or_init(|| {
        std::env::var("ATLAS_FFN_W3_LAYERS")
            .map(|s| parse_w3_layer_set(&s))
            .unwrap_or_default()
    })
}

/// Sidecar path from `ATLAS_FFN_W3_SIDECAR` (None when unset/empty).
pub fn w3_sidecar_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("ATLAS_FFN_W3_SIDECAR")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}

/// One layer's uploaded W3 FFN weights: GEMV (`[N, 3K/8]`) + transposed
/// GEMM (`[3K/8, N_pad64]`) copies of gate / up / down.
pub struct W3FfnLayerWeights {
    pub gate: QuantizedWeight,
    pub up: QuantizedWeight,
    pub down: QuantizedWeight,
    pub gate_t: QuantizedWeight,
    pub up_t: QuantizedWeight,
    pub down_t: QuantizedWeight,
}

// ── Minimal host-side safetensors reader ────────────────────────────────────
// (spark-runtime's WeightStore uploads straight to GPU with the model's
// dtype whitelist; the sidecar is a separate host file with U8 payloads, so
// a ~60-line reader keeps this self-contained without new dependencies.)

struct SidecarFile {
    header: serde_json::Map<String, serde_json::Value>,
    data: Vec<u8>,
}

impl SidecarFile {
    fn open(path: &str) -> Result<Self> {
        let mut f = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)?;
        let hlen = u64::from_le_bytes(len8) as usize;
        if hlen > 128 << 20 {
            bail!("sidecar header too large: {hlen}");
        }
        let mut hbuf = vec![0u8; hlen];
        f.read_exact(&mut hbuf)?;
        let header: serde_json::Value =
            serde_json::from_slice(&hbuf).context("sidecar header JSON")?;
        let serde_json::Value::Object(header) = header else {
            bail!("sidecar header is not a JSON object");
        };
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;
        Ok(Self { header, data })
    }

    fn contains(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    /// Raw bytes + shape for a tensor, validating the expected dtype tag.
    fn tensor(&self, name: &str, want_dtype: &str) -> Result<(&[u8], Vec<usize>)> {
        let meta = self
            .header
            .get(name)
            .with_context(|| format!("sidecar missing tensor {name}"))?;
        let dtype = meta["dtype"].as_str().unwrap_or("");
        if dtype != want_dtype {
            bail!("{name}: dtype {dtype}, expected {want_dtype}");
        }
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .with_context(|| format!("{name}: bad shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = meta["data_offsets"]
            .as_array()
            .with_context(|| format!("{name}: bad data_offsets"))?;
        let (a, b) = (
            offs[0].as_u64().unwrap_or(0) as usize,
            offs[1].as_u64().unwrap_or(0) as usize,
        );
        if b < a || b > self.data.len() {
            bail!("{name}: data_offsets out of range");
        }
        Ok((&self.data[a..b], shape))
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// Load + upload one projection: returns (gemv_weight, gemm_t_weight).
fn load_proj(
    sidecar: &SidecarFile,
    gpu: &dyn GpuBackend,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<(QuantizedWeight, QuantizedWeight)> {
    let (packed, pshape) = sidecar.tensor(&format!("{prefix}.w3_weight"), "U8")?;
    let (scales, sshape) = sidecar.tensor(&format!("{prefix}.w3_weight_scale"), "U8")?;
    let (s2_raw, _) = sidecar.tensor(&format!("{prefix}.w3_weight_scale_2"), "F32")?;

    let row_bytes = w3_row_bytes(k);
    let num_groups = k / W3_GROUP_SIZE;
    if pshape != [n, row_bytes] {
        bail!("{prefix}.w3_weight: shape {pshape:?}, expected [{n}, {row_bytes}]");
    }
    if sshape != [n, num_groups] {
        bail!("{prefix}.w3_weight_scale: shape {sshape:?}, expected [{n}, {num_groups}]");
    }
    if s2_raw.len() != 4 {
        bail!(
            "{prefix}.w3_weight_scale_2: expected 4 bytes, got {}",
            s2_raw.len()
        );
    }
    let scale2 = f32::from_le_bytes([s2_raw[0], s2_raw[1], s2_raw[2], s2_raw[3]]);
    if !scale2.is_finite() || scale2 <= 0.0 {
        bail!("{prefix}: bad scale2 {scale2}");
    }

    let gemv = QuantizedWeight {
        weight: upload(gpu, packed)?,
        weight_scale: upload(gpu, scales)?,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    };

    // Transposed GEMM copies (same 64-pad rule as transpose_for_gemm).
    let (packed_t, _) = w3_transpose_host(packed, n, row_bytes);
    let (scales_t, _) = w3_transpose_host(scales, n, num_groups);
    let gemm_t = QuantizedWeight {
        weight: upload(gpu, &packed_t)?,
        weight_scale: upload(gpu, &scales_t)?,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    };
    Ok((gemv, gemm_t))
}

/// Load layer `i`'s W3 FFN weights when gated on and present in the
/// sidecar. `lp` is the layer prefix (e.g. `model.language_model.layers.3`),
/// `hidden`/`inter` the FFN dims. Fail-open: any error logs a warning and
/// returns `Ok(None)` so the layer stays on the default W4 path.
pub fn maybe_load_w3_ffn(
    i: usize,
    lp: &str,
    gpu: &dyn GpuBackend,
    hidden: usize,
    inter: usize,
) -> Result<Option<W3FfnLayerWeights>> {
    if !w3_layer_set().contains(&i) {
        return Ok(None);
    }
    let Some(path) = w3_sidecar_path() else {
        tracing::warn!(
            "ATLAS_FFN_W3_LAYERS includes layer {i} but ATLAS_FFN_W3_SIDECAR is unset — \
             staying on W4"
        );
        return Ok(None);
    };

    static FILE: OnceLock<Option<SidecarFile>> = OnceLock::new();
    let sidecar = FILE.get_or_init(|| match SidecarFile::open(path) {
        Ok(f) => Some(f),
        Err(e) => {
            tracing::warn!("W3 sidecar {path} unreadable ({e:#}) — all W3 layers stay on W4");
            None
        }
    });
    let Some(sidecar) = sidecar else {
        return Ok(None);
    };

    let gate_prefix = format!("{lp}.mlp.gate_proj");
    if !sidecar.contains(&format!("{gate_prefix}.w3_weight")) {
        tracing::warn!("W3 sidecar has no tensors for layer {i} ({lp}) — staying on W4");
        return Ok(None);
    }

    let load = || -> Result<W3FfnLayerWeights> {
        let (gate, gate_t) = load_proj(sidecar, gpu, &gate_prefix, inter, hidden)?;
        let (up, up_t) = load_proj(sidecar, gpu, &format!("{lp}.mlp.up_proj"), inter, hidden)?;
        let (down, down_t) =
            load_proj(sidecar, gpu, &format!("{lp}.mlp.down_proj"), hidden, inter)?;
        Ok(W3FfnLayerWeights {
            gate,
            up,
            down,
            gate_t,
            up_t,
            down_t,
        })
    };
    match load() {
        Ok(w) => {
            tracing::info!(
                "layer {i}: FFN routed to W3 (3-bit) weights from sidecar — packed bytes \
                 -25% vs NVFP4 on gate/up/down"
            );
            Ok(Some(w))
        }
        Err(e) => {
            tracing::warn!("layer {i}: W3 sidecar load failed ({e:#}) — staying on W4");
            Ok(None)
        }
    }
}
