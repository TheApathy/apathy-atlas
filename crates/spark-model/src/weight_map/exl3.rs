// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis-coded (2.0/3.0 bpw) expert weight representation + loaders.
//!
//! Ground truth is the reference tp1 checkpoint headers
//! (`/home/flocka/sparkinfer-ref/data/tp1`, `quant_method: "exl3"`): each
//! routed-expert projection ships four tensors under
//! `layers.{L}.ffn.experts.{E}.{w1|w2|w3}.rank0.*`:
//!
//! | tensor    | dtype | shape             | meaning                          |
//! |-----------|-------|-------------------|----------------------------------|
//! | `trellis` | I16   | `[K/16, N/16, 48]`| 96 B per 16x16 tile = 3.000 bpw  |
//! | `suh`     | F16   | `[K]`             | input-side sign vector           |
//! | `svh`     | F16   | `[N]`             | output-side sign vector          |
//! | `mcg`     | I32   | scalar            | 3INST cb=1 selector (0xCBAC1FED) |
//!
//! Only the ROUTED experts are EXL3; attention, shared expert, gate, embed
//! and head stay FP8/BF16 (see `docs/EXPERT-3BPW-PLAN.md` §1). The trellis
//! tiles are consumed by `exl3_gemv_m1` exactly in their on-disk `[K/16,
//! N/16, 48]` layout — NO transpose pass (unlike the MXFP4 unified-layout
//! path in `factory/m2_setup.rs`).
//!
//! Decode-format detail lives in `docs/kernels/exl3-gemv.md`.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// The 3INST cb=1 multiplicative-congruential constant. Compile-time in the
/// kernels; the checkpoint's per-matrix `mcg` scalar must match.
pub const EXL3_MCG_MULT: u32 = 0xCBAC_1FED;

/// One EXL3 trellis-coded weight matrix (logical `W: [N, K]`, i.e. the GEMV
/// computes `y[N] = W · x[K]` through the Hadamard/sign pipeline).
#[derive(Debug, Clone, Copy)]
pub struct Exl3Weight {
    /// I16 `[K/16, N/16, 16*bits]` trellis payload, device-resident as on disk.
    pub trellis: DevicePtr,
    /// F16 `[K]` input-side sign vector (native F16 — see loader exemption).
    pub suh: DevicePtr,
    /// F16 `[N]` output-side sign vector.
    pub svh: DevicePtr,
    /// Output rows.
    pub n: u32,
    /// Input columns.
    pub k: u32,
    /// Trellis bitrate. K2 uses 32 I16 words/tile; K3 uses 48.
    pub bits: u32,
}

/// Per-expert EXL3 triplet in Atlas naming: gate = checkpoint `w1`,
/// up = `w3`, down = `w2` (matches `assemble_moe`'s NVFP4 mapping).
#[derive(Debug, Clone, Copy)]
pub struct Exl3ExpertWeight {
    pub gate_proj: Exl3Weight,
    pub up_proj: Exl3Weight,
    pub down_proj: Exl3Weight,
}

/// True when `layer_prefix` ships EXL3 routed experts (probes expert 0 w1).
/// Keyed off the actual tensor, not `quantization_config`, so a mixed tree
/// (e.g. our own future 144-expert quant) detects per layer.
pub fn store_has_exl3_experts(store: &WeightStore, layer_prefix: &str) -> bool {
    let hf_prefix = if layer_prefix.starts_with("layers.") {
        format!("model.{layer_prefix}")
    } else {
        layer_prefix.to_string()
    };
    store.contains(&format!("{layer_prefix}.ffn.experts.0.w1.rank0.trellis"))
        || store.contains(&format!("{hf_prefix}.mlp.experts.0.gate_proj.trellis"))
}

fn exl3_tensor_key(store: &WeightStore, prefix: &str, name: &str) -> String {
    let ranked = format!("{prefix}.rank0.{name}");
    if store.contains(&ranked) {
        ranked
    } else {
        format!("{prefix}.{name}")
    }
}

/// Load one EXL3 projection (`prefix` = `…ffn.experts.{E}.{w1|w2|w3}`).
///
/// Validates dtypes/shapes against the format contract and the `mcg`
/// codebook selector against the compile-time constant, then returns the
/// device pointers the store already landed (zero-copy — the tiles are
/// consumed as-is by `exl3_gemv_m1`).
pub fn exl3_weight(store: &WeightStore, prefix: &str, gpu: &dyn GpuBackend) -> Result<Exl3Weight> {
    let trellis_key = exl3_tensor_key(store, prefix, "trellis");
    let trellis = store
        .get(&trellis_key)
        .with_context(|| format!("{prefix}: missing EXL3 trellis tensor"))?;
    ensure!(
        trellis.dtype == WeightDtype::Int16,
        "{trellis_key}: dtype {:?} != I16",
        trellis.dtype
    );
    ensure!(
        trellis.shape.len() == 3 && matches!(trellis.shape[2], 32 | 48),
        "{trellis_key}: shape {:?} is not K2/K3 [K/16, N/16, 32|48]",
        trellis.shape
    );
    let bits = (trellis.shape[2] / 16) as u32;
    let k = (trellis.shape[0] * 16) as u32;
    let n = (trellis.shape[1] * 16) as u32;
    ensure!(
        n % 128 == 0 && k % 128 == 0,
        "{prefix}: EXL3 GEMV requires N,K % 128 == 0 (got N={n}, K={k})"
    );

    let suh_key = exl3_tensor_key(store, prefix, "suh");
    let suh = store
        .get(&suh_key)
        .with_context(|| format!("{prefix}: missing EXL3 suh"))?;
    ensure!(
        suh.dtype == WeightDtype::F16 && suh.num_elements() == k as usize,
        "{prefix}.rank0.suh: expected F16 [{k}], got {:?} {:?} (the loader must \
         keep .suh/.svh native F16 — see load_fns::exl3_keep_f16)",
        suh.dtype,
        suh.shape
    );
    let svh_key = exl3_tensor_key(store, prefix, "svh");
    let svh = store
        .get(&svh_key)
        .with_context(|| format!("{prefix}: missing EXL3 svh"))?;
    ensure!(
        svh.dtype == WeightDtype::F16 && svh.num_elements() == n as usize,
        "{prefix}.rank0.svh: expected F16 [{n}], got {:?} {:?}",
        svh.dtype,
        svh.shape
    );

    // mcg: scalar I32 codebook selector. The kernel hardcodes 0xCBAC1FED
    // (3INST cb=1); any other value means a codebook we do not decode.
    let mcg_key = exl3_tensor_key(store, prefix, "mcg");
    let mcg = store
        .get(&mcg_key)
        .with_context(|| format!("{prefix}: missing EXL3 mcg"))?;
    ensure!(
        mcg.dtype == WeightDtype::Int32,
        "{prefix}.rank0.mcg: dtype {:?} != I32",
        mcg.dtype
    );
    let mut buf = [0u8; 4];
    gpu.copy_d2h(mcg.ptr, &mut buf)?;
    let mcg_val = u32::from_le_bytes(buf);
    ensure!(
        mcg_val == EXL3_MCG_MULT,
        "{prefix}.rank0.mcg = {mcg_val:#010x} != {EXL3_MCG_MULT:#010x}: unknown \
         EXL3 codebook — the exl3_gemv kernels decode 3INST cb=1 only"
    );

    Ok(Exl3Weight {
        trellis: trellis.ptr,
        suh: suh.ptr,
        svh: svh.ptr,
        n,
        k,
        bits,
    })
}

/// Load one expert's EXL3 triplet (`ep` = `…ffn.experts.{E}`), mapping the
/// checkpoint's w1/w3/w2 to gate/up/down and cross-checking that the three
/// matrices agree on hidden/intermediate dims.
pub fn exl3_expert(
    store: &WeightStore,
    ep: &str,
    gpu: &dyn GpuBackend,
) -> Result<Exl3ExpertWeight> {
    let gate_proj = exl3_weight(store, &format!("{ep}.w1"), gpu)?;
    let up_proj = exl3_weight(store, &format!("{ep}.w3"), gpu)?;
    let down_proj = exl3_weight(store, &format!("{ep}.w2"), gpu)?;
    ensure!(
        gate_proj.n == up_proj.n
            && gate_proj.k == up_proj.k
            && down_proj.k == gate_proj.n
            && down_proj.n == gate_proj.k
            && gate_proj.bits == up_proj.bits
            && gate_proj.bits == down_proj.bits,
        "{ep}: inconsistent EXL3 expert dims — w1 [{}x{}], w3 [{}x{}], w2 [{}x{}]",
        gate_proj.n,
        gate_proj.k,
        up_proj.n,
        up_proj.k,
        down_proj.n,
        down_proj.k
    );
    Ok(Exl3ExpertWeight {
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// Load an expert using Hugging Face projection names.
pub fn exl3_hf_expert(
    store: &WeightStore,
    ep: &str,
    gpu: &dyn GpuBackend,
) -> Result<Exl3ExpertWeight> {
    let gate_proj = exl3_weight(store, &format!("{ep}.gate_proj"), gpu)?;
    let up_proj = exl3_weight(store, &format!("{ep}.up_proj"), gpu)?;
    let down_proj = exl3_weight(store, &format!("{ep}.down_proj"), gpu)?;
    ensure!(
        gate_proj.n == up_proj.n
            && gate_proj.k == up_proj.k
            && down_proj.k == gate_proj.n
            && down_proj.n == gate_proj.k
            && gate_proj.bits == up_proj.bits
            && gate_proj.bits == down_proj.bits,
        "{ep}: inconsistent EXL3 expert dimensions"
    );
    Ok(Exl3ExpertWeight {
        gate_proj,
        up_proj,
        down_proj,
    })
}
