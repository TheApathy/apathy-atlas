// SPDX-License-Identifier: AGPL-3.0-only

//! Loader for W3 Lloyd-Max (3-bit) routed-expert layer files (`ATLAS_MOE_W3=1`).
//!
//! Reads the per-layer `layer_{L:03}.w3x` files produced offline by the
//! `w3-requant` tool (spark-storage, `crates/spark-storage/src/w3.rs` is the
//! format SSOT — the header constants below are a deliberate small duplicate;
//! bump both on ANY change). One GPU allocation per layer holds the whole
//! payload; per-expert `QuantizedWeight`s are interior pointers into it, so
//! the standard `build_ptr_table` machinery works unchanged:
//!
//!   * `weight`       → `[N, K*3/8]` Turbo3-packed 3-bit codebook indices
//!   * `weight_scale` → `[N, K/16]` FP8-E4M3 group scales (UNCHANGED NVFP4)
//!   * `weight_scale_2` → per-projection f32 scale2 (UNCHANGED NVFP4)
//!
//! plus the layer's 8-entry codebook uploaded as a `[8]` f32 device buffer
//! (passed to every `_w3` kernel as its last argument).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{ExpertWeight, QuantizedWeight};

// ── Format constants (keep in sync with spark_storage::w3) ────────────────
const W3_MAGIC: u32 = 0x4D4C_3357; // "W3LM"
const W3_VERSION: u32 = 1;
const W3_HEADER_BYTES: usize = 64;
const W3_PAYLOAD_ALIGN: usize = 4096;
const GROUP_SIZE: usize = 16;

/// `ATLAS_MOE_W3=1` master switch.
pub fn w3_enabled() -> bool {
    matches!(
        std::env::var("ATLAS_MOE_W3").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Cache directory: `ATLAS_MOE_W3_DIR`, default `./w3cache`.
pub fn w3cache_dir() -> PathBuf {
    std::env::var("ATLAS_MOE_W3_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./w3cache"))
}

pub fn w3_layer_path(dir: &Path, layer: usize) -> PathBuf {
    dir.join(format!("layer_{layer:03}.w3x"))
}

/// One MoE layer's W3 experts, device-resident.
pub struct W3LoadedLayer {
    /// Per-expert W3 weights (interior pointers into `payload_dev`).
    pub experts: Vec<ExpertWeight>,
    /// `[8]` f32 Lloyd-Max codebook on device.
    pub lut_dev: DevicePtr,
    /// Host copy of the codebook (diagnostics).
    pub lut: [f32; 8],
    /// Total device bytes uploaded (payload + lut).
    pub device_bytes: usize,
}

struct W3Header {
    layer: u32,
    num_experts: u32,
    hidden: u32,
    inter: u32,
    lut: [f32; 8],
}

fn parse_header(buf: &[u8]) -> Option<W3Header> {
    if buf.len() < W3_HEADER_BYTES {
        return None;
    }
    let r = |off: usize| u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    if r(0) != W3_MAGIC || r(4) != W3_VERSION || r(24) != GROUP_SIZE as u32 {
        return None;
    }
    let mut lut = [0.0f32; 8];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = f32::from_le_bytes([
            buf[32 + i * 4],
            buf[33 + i * 4],
            buf[34 + i * 4],
            buf[35 + i * 4],
        ]);
    }
    Some(W3Header {
        layer: r(8),
        num_experts: r(12),
        hidden: r(16),
        inter: r(20),
        lut,
    })
}

/// Load one layer's W3 experts onto the GPU. Validates the header against
/// the model dims; any mismatch is an error (caller falls back to NVFP4).
pub fn load_w3_layer(
    dir: &Path,
    layer: usize,
    num_experts: usize,
    hidden: usize,
    inter: usize,
    gpu: &dyn GpuBackend,
) -> Result<W3LoadedLayer> {
    let path = w3_layer_path(dir, layer);
    let buf = std::fs::read(&path).with_context(|| format!("read {path:?}"))?;
    let h = parse_header(&buf).with_context(|| format!("{path:?}: bad W3 header"))?;
    ensure!(
        h.layer as usize == layer
            && h.num_experts as usize == num_experts
            && h.hidden as usize == hidden
            && h.inter as usize == inter,
        "{path:?}: header (layer={} experts={} hidden={} inter={}) does not match \
         model (layer={layer} experts={num_experts} hidden={hidden} inter={inter})",
        h.layer,
        h.num_experts,
        h.hidden,
        h.inter,
    );
    ensure!(
        hidden % 32 == 0 && inter % 32 == 0,
        "{path:?}: W3 kernels need K % 32 == 0 (hidden={hidden}, inter={inter})"
    );

    // Geometry (mirror of spark_storage::w3::W3LayerGeom).
    let gu_packed3 = inter * hidden * 3 / 8;
    let gu_scale = inter * hidden / GROUP_SIZE;
    let down_packed3 = hidden * inter * 3 / 8;
    let down_scale = hidden * inter / GROUP_SIZE;
    let expert_stride = 2 * (gu_packed3 + gu_scale) + down_packed3 + down_scale;
    let scale2_off = W3_HEADER_BYTES;
    let scale2_bytes = num_experts * 3 * 4;
    let payload_off =
        (W3_HEADER_BYTES + scale2_bytes).div_ceil(W3_PAYLOAD_ALIGN) * W3_PAYLOAD_ALIGN;
    let expect_len = payload_off + num_experts * expert_stride;
    ensure!(
        buf.len() == expect_len,
        "{path:?}: file is {} bytes, expected {expect_len}",
        buf.len()
    );

    // scale2 table.
    let mut scale2 = vec![0f32; num_experts * 3];
    for (i, v) in scale2.iter_mut().enumerate() {
        let o = scale2_off + i * 4;
        *v = f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        ensure!(
            v.is_finite(),
            "{path:?}: non-finite scale2 at expert {} proj {}",
            i / 3,
            i % 3
        );
    }

    // One allocation for the whole payload; per-expert interior pointers.
    let payload = &buf[payload_off..];
    let base = gpu.alloc(payload.len()).with_context(|| {
        format!(
            "alloc {} MiB W3 payload for layer {layer}",
            payload.len() >> 20
        )
    })?;
    gpu.copy_h2d(payload, base)?;

    let lut_bytes: Vec<u8> = h.lut.iter().flat_map(|v| v.to_le_bytes()).collect();
    let lut_dev = gpu.alloc(lut_bytes.len())?;
    gpu.copy_h2d(&lut_bytes, lut_dev)?;

    // Sub-buffer offsets within one expert record (gate_p, gate_s, up_p,
    // up_s, down_p, down_s) — contiguous, no padding in format v1.
    let offs = [
        0,
        gu_packed3,
        gu_packed3 + gu_scale,
        2 * gu_packed3 + gu_scale,
        2 * (gu_packed3 + gu_scale),
        2 * (gu_packed3 + gu_scale) + down_packed3,
    ];
    let qw = |e: usize, p: usize| -> QuantizedWeight {
        QuantizedWeight {
            weight: base.offset(e * expert_stride + offs[p * 2]),
            weight_scale: base.offset(e * expert_stride + offs[p * 2 + 1]),
            weight_scale_2: scale2[e * 3 + p],
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        }
    };
    let experts = (0..num_experts)
        .map(|e| ExpertWeight {
            gate_proj: qw(e, 0),
            up_proj: qw(e, 1),
            down_proj: qw(e, 2),
        })
        .collect();

    Ok(W3LoadedLayer {
        experts,
        lut_dev,
        lut: h.lut,
        device_bytes: payload.len() + lut_bytes.len(),
    })
}
