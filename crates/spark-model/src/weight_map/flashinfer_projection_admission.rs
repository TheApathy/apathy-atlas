// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off admission/cache plan for Qwen3.8 FlashInfer projections.
//!
//! This module validates and retains exact checkpoint operands. Only the
//! attention M2079 shapes have a bounded launch plan, backed by isolated
//! kernel receipts plus repeated endpoint output equality. That evidence is
//! not a general production semantic oracle: M8192 is rejected after an
//! endpoint screen exposed output drift, and the SSM synthetic candidates
//! remain unroutable here.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
};
use super::modelopt_scale_admission::{
    AdmittedModeloptScale, ModeloptScaleProjection, ModeloptScaleSource,
};

pub const QWEN38_HIDDEN: usize = 5_120;
pub const QWEN38_QG: usize = 12_288;
pub const QWEN38_KV: usize = 1_024;
pub const QWEN38_ATTN_QGKV: usize = QWEN38_QG + 2 * QWEN38_KV;
pub const QWEN38_ATTN_VALUE: usize = 6_144;
pub const QWEN38_SSM_QKV: usize = 10_240;
pub const QWEN38_SSM_Z: usize = 6_144;
pub const QWEN38_SSM_QKVZ: usize = QWEN38_SSM_QKV + QWEN38_SSM_Z;
pub const QWEN38_SSM_VALUE: usize = 6_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen38Projection {
    AttentionQueryGate,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    SsmQkvz,
    SsmOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen38ProjectionSource {
    pub layer: usize,
    pub projection: Qwen38Projection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetainedProjectionScalar {
    source: Qwen38ProjectionSource,
    value_bits: u32,
}

impl RetainedProjectionScalar {
    pub fn source(self) -> Qwen38ProjectionSource {
        self.source
    }

    pub fn value_bits(self) -> u32 {
        self.value_bits
    }

    pub fn value(self) -> f32 {
        f32::from_bits(self.value_bits)
    }
}

/// Exact checkpoint material. Packed weights stay in their original device
/// allocation; only the CUTLASS physical scale transform is newly owned.
pub struct Qwen38CheckpointProjection<'a> {
    pub source: Qwen38ProjectionSource,
    pub packed_weight: DevicePtr,
    pub packed_weight_bytes: usize,
    pub logical_weight_scales: &'a [u8],
    pub input_scale: AdmittedModeloptScale,
    pub weight_scale_2_le_bytes: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedQwen38Projection {
    source: Qwen38ProjectionSource,
    n: usize,
    k: usize,
    packed_weight: DevicePtr,
    packed_weight_bytes: usize,
    physical_weight_scales: Vec<u8>,
    input_scale: AdmittedModeloptScale,
    weight_scale_2: RetainedProjectionScalar,
    alpha: RetainedProjectionScalar,
}

impl AdmittedQwen38Projection {
    pub fn source(&self) -> Qwen38ProjectionSource {
        self.source
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn packed_weight(&self) -> DevicePtr {
        self.packed_weight
    }

    pub fn packed_weight_bytes(&self) -> usize {
        self.packed_weight_bytes
    }

    pub fn physical_weight_scales(&self) -> &[u8] {
        &self.physical_weight_scales
    }

    pub fn input_scale(&self) -> AdmittedModeloptScale {
        self.input_scale
    }

    pub fn weight_scale_2(&self) -> RetainedProjectionScalar {
        self.weight_scale_2
    }

    pub fn alpha(&self) -> RetainedProjectionScalar {
        self.alpha
    }
}

fn projection_shape(projection: Qwen38Projection) -> (usize, usize) {
    match projection {
        Qwen38Projection::AttentionQueryGate => (QWEN38_QG, QWEN38_HIDDEN),
        Qwen38Projection::AttentionKey | Qwen38Projection::AttentionValue => {
            (QWEN38_KV, QWEN38_HIDDEN)
        }
        Qwen38Projection::AttentionOutput => (QWEN38_HIDDEN, QWEN38_ATTN_VALUE),
        Qwen38Projection::SsmQkvz => (QWEN38_SSM_QKVZ, QWEN38_HIDDEN),
        Qwen38Projection::SsmOutput => (QWEN38_HIDDEN, QWEN38_SSM_VALUE),
    }
}

fn input_scale_source(source: Qwen38ProjectionSource) -> ModeloptScaleSource {
    let projection = match source.projection {
        Qwen38Projection::AttentionQueryGate => ModeloptScaleProjection::AttentionQuery,
        Qwen38Projection::AttentionKey => ModeloptScaleProjection::AttentionKey,
        Qwen38Projection::AttentionValue => ModeloptScaleProjection::AttentionValue,
        Qwen38Projection::AttentionOutput => ModeloptScaleProjection::AttentionOutput,
        Qwen38Projection::SsmQkvz => ModeloptScaleProjection::SsmInput,
        Qwen38Projection::SsmOutput => ModeloptScaleProjection::SsmOutput,
    };
    ModeloptScaleSource {
        layer: source.layer,
        projection,
    }
}

fn checked_product(label: &str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} byte-length overflow: {left} * {right}"))
}

pub fn admit_qwen38_projection(
    expected_layer: usize,
    expected_projection: Qwen38Projection,
    projection: Qwen38CheckpointProjection<'_>,
) -> Result<AdmittedQwen38Projection> {
    let expected_source = Qwen38ProjectionSource {
        layer: expected_layer,
        projection: expected_projection,
    };
    ensure!(
        projection.source == expected_source,
        "Qwen3.8 projection source is misattributed"
    );
    ensure!(
        projection.input_scale.source() == input_scale_source(expected_source),
        "Qwen3.8 projection input-scale source is misattributed"
    );
    ensure!(
        !projection.packed_weight.is_null() && projection.packed_weight.0.is_multiple_of(16),
        "Qwen3.8 packed checkpoint weight must be non-NULL and 16-byte aligned"
    );

    let (n, k) = projection_shape(expected_projection);
    let packed_weight_bytes = checked_product("projection packed weight", n, k / 2)?;
    ensure!(
        projection.packed_weight_bytes == packed_weight_bytes,
        "Qwen3.8 projection packed length mismatch: expected {packed_weight_bytes}, got {}",
        projection.packed_weight_bytes
    );
    let groups = k / NVFP4_GROUP_SIZE;
    let physical_weight_scales = interleave_nvfp4_scales_128x4(
        projection.logical_weight_scales,
        &[n, groups],
        NVFP4_GROUP_SIZE,
    )?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(&physical_weight_scales, &[n, groups], NVFP4_GROUP_SIZE,)?
            == projection.logical_weight_scales,
        "Qwen3.8 projection physical scales failed exact round-trip"
    );

    let weight_scale_2 = f32::from_le_bytes(projection.weight_scale_2_le_bytes);
    ensure!(
        weight_scale_2.is_finite() && weight_scale_2 > 0.0,
        "Qwen3.8 projection weight_scale_2 must be finite and positive"
    );
    let weight_scale_2 = RetainedProjectionScalar {
        source: expected_source,
        value_bits: u32::from_le_bytes(projection.weight_scale_2_le_bytes),
    };
    let alpha = projection.input_scale.value() * weight_scale_2.value();
    ensure!(
        alpha.is_finite() && alpha > 0.0,
        "Qwen3.8 projection alpha must be finite and positive"
    );

    Ok(AdmittedQwen38Projection {
        source: expected_source,
        n,
        k,
        packed_weight: projection.packed_weight,
        packed_weight_bytes,
        physical_weight_scales,
        input_scale: projection.input_scale,
        weight_scale_2,
        alpha: RetainedProjectionScalar {
            source: expected_source,
            value_bits: alpha.to_bits(),
        },
    })
}

/// Fixed-address operands retained after the caller uploads the physical scale
/// bytes and alpha scalar. Construction does not allocate, copy, or launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen38ProjectionOperandCache {
    source: Qwen38ProjectionSource,
    n: usize,
    k: usize,
    packed_weight: DevicePtr,
    physical_weight_scales: DevicePtr,
    input_scale: DevicePtr,
    input_scale_bits: u32,
    alpha: DevicePtr,
    alpha_bits: u32,
}

impl Qwen38ProjectionOperandCache {
    pub fn source(self) -> Qwen38ProjectionSource {
        self.source
    }

    pub fn shape(self) -> (usize, usize) {
        (self.n, self.k)
    }

    pub fn packed_weight(self) -> DevicePtr {
        self.packed_weight
    }

    pub fn physical_weight_scales(self) -> DevicePtr {
        self.physical_weight_scales
    }

    pub fn input_scale(self) -> DevicePtr {
        self.input_scale
    }

    pub fn input_scale_bits(self) -> u32 {
        self.input_scale_bits
    }

    pub fn alpha(self) -> DevicePtr {
        self.alpha
    }

    pub fn alpha_bits(self) -> u32 {
        self.alpha_bits
    }
}

pub fn finalize_qwen38_projection_cache(
    admitted: &AdmittedQwen38Projection,
    physical_weight_scales: DevicePtr,
    alpha: DevicePtr,
    uploaded_alpha_bits: u32,
) -> Result<Qwen38ProjectionOperandCache> {
    ensure!(
        !physical_weight_scales.is_null() && physical_weight_scales.0.is_multiple_of(16),
        "FlashInfer physical scale cache must be non-NULL and 16-byte aligned"
    );
    ensure!(
        !alpha.is_null() && alpha.0.is_multiple_of(4),
        "FlashInfer alpha cache must be non-NULL and 4-byte aligned"
    );
    ensure!(
        uploaded_alpha_bits == admitted.alpha().value_bits(),
        "FlashInfer uploaded alpha bits do not match the admitted checkpoint scalar"
    );
    ensure!(
        physical_weight_scales != admitted.packed_weight(),
        "FlashInfer scale cache aliases packed checkpoint weight"
    );

    Ok(Qwen38ProjectionOperandCache {
        source: admitted.source(),
        n: admitted.n(),
        k: admitted.k(),
        packed_weight: admitted.packed_weight(),
        physical_weight_scales,
        input_scale: admitted.input_scale().device_ptr(),
        input_scale_bits: admitted.input_scale().value_bits(),
        alpha,
        alpha_bits: uploaded_alpha_bits,
    })
}

/// Three separately scaled Q/G, K and V GEMMs may reuse one activation
/// quantization only when their checkpoint input scales are bit-identical.
/// Their weight scales and alpha allocations remain independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen38AttentionQkvCacheGroup {
    layer: usize,
    query_gate: Qwen38ProjectionOperandCache,
    key: Qwen38ProjectionOperandCache,
    value: Qwen38ProjectionOperandCache,
}

impl Qwen38AttentionQkvCacheGroup {
    pub fn layer(self) -> usize {
        self.layer
    }

    pub fn caches(self) -> [Qwen38ProjectionOperandCache; 3] {
        [self.query_gate, self.key, self.value]
    }
}

pub fn admit_qwen38_attention_qkv_cache_group(
    layer: usize,
    query_gate: Qwen38ProjectionOperandCache,
    key: Qwen38ProjectionOperandCache,
    value: Qwen38ProjectionOperandCache,
) -> Result<Qwen38AttentionQkvCacheGroup> {
    let caches = [query_gate, key, value];
    let expected = [
        Qwen38Projection::AttentionQueryGate,
        Qwen38Projection::AttentionKey,
        Qwen38Projection::AttentionValue,
    ];
    for (cache, projection) in caches.into_iter().zip(expected) {
        ensure!(
            cache.source() == (Qwen38ProjectionSource { layer, projection }),
            "shared Q/K/V activation cache has a misattributed projection"
        );
    }
    ensure!(
        query_gate.input_scale_bits == key.input_scale_bits
            && query_gate.input_scale_bits == value.input_scale_bits,
        "shared Q/K/V activation cache requires bit-identical scale values"
    );
    ensure!(
        query_gate.alpha_bits == key.alpha_bits && query_gate.alpha_bits == value.alpha_bits,
        "shared Q/K/V activation cache requires bit-identical alpha values"
    );

    Ok(Qwen38AttentionQkvCacheGroup {
        layer,
        query_gate,
        key,
        value,
    })
}

/// Host-owned, immutable checkpoint operands for the single merged QGKV
/// FlashInfer GEMM. The concatenation order is exactly Q/G, K, V along N.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedQwen38AttentionQkvMerge {
    layer: usize,
    packed_weight: Vec<u8>,
    physical_weight_scales: Vec<u8>,
    input_scale: AdmittedModeloptScale,
    alpha_bits: u32,
}

impl AdmittedQwen38AttentionQkvMerge {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn packed_weight(&self) -> &[u8] {
        &self.packed_weight
    }

    pub fn physical_weight_scales(&self) -> &[u8] {
        &self.physical_weight_scales
    }

    pub fn input_scale(&self) -> AdmittedModeloptScale {
        self.input_scale
    }

    pub fn alpha_bits(&self) -> u32 {
        self.alpha_bits
    }
}

pub fn merge_qwen38_attention_qkv(
    layer: usize,
    query_gate: &AdmittedQwen38Projection,
    query_gate_packed: &[u8],
    key: &AdmittedQwen38Projection,
    key_packed: &[u8],
    value: &AdmittedQwen38Projection,
    value_packed: &[u8],
) -> Result<AdmittedQwen38AttentionQkvMerge> {
    let admitted = [query_gate, key, value];
    let packed = [query_gate_packed, key_packed, value_packed];
    let expected = [
        Qwen38Projection::AttentionQueryGate,
        Qwen38Projection::AttentionKey,
        Qwen38Projection::AttentionValue,
    ];
    for ((operand, bytes), projection) in admitted.into_iter().zip(packed).zip(expected) {
        ensure!(
            operand.source() == (Qwen38ProjectionSource { layer, projection }),
            "merged QGKV operand has a misattributed projection"
        );
        ensure!(
            bytes.len() == operand.packed_weight_bytes(),
            "merged QGKV packed bytes do not match admitted checkpoint extent"
        );
    }
    ensure!(
        query_gate.input_scale().value_bits() == key.input_scale().value_bits()
            && query_gate.input_scale().value_bits() == value.input_scale().value_bits(),
        "merged QGKV requires bit-identical checkpoint input scales"
    );
    ensure!(
        query_gate.alpha().value_bits() == key.alpha().value_bits()
            && query_gate.alpha().value_bits() == value.alpha().value_bits(),
        "merged QGKV requires bit-identical checkpoint alpha values"
    );

    let packed_len = checked_product(
        "merged QGKV packed weight",
        QWEN38_ATTN_QGKV,
        QWEN38_HIDDEN / 2,
    )?;
    let scale_len = checked_product(
        "merged QGKV physical scales",
        QWEN38_ATTN_QGKV,
        QWEN38_HIDDEN / NVFP4_GROUP_SIZE,
    )?;
    let mut packed_weight = Vec::with_capacity(packed_len);
    let mut physical_weight_scales = Vec::with_capacity(scale_len);
    for (operand, bytes) in admitted.into_iter().zip(packed) {
        packed_weight.extend_from_slice(bytes);
        physical_weight_scales.extend_from_slice(operand.physical_weight_scales());
    }
    ensure!(
        packed_weight.len() == packed_len,
        "merged QGKV packed extent mismatch"
    );
    ensure!(
        physical_weight_scales.len() == scale_len,
        "merged QGKV physical-scale extent mismatch"
    );

    Ok(AdmittedQwen38AttentionQkvMerge {
        layer,
        packed_weight,
        physical_weight_scales,
        input_scale: query_gate.input_scale(),
        alpha_bits: query_gate.alpha().value_bits(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen38AttentionMergedQkvCache {
    layer: usize,
    packed_weight: DevicePtr,
    physical_weight_scales: DevicePtr,
    input_scale: DevicePtr,
    input_scale_bits: u32,
    alpha: DevicePtr,
    alpha_bits: u32,
}

impl Qwen38AttentionMergedQkvCache {
    pub fn layer(self) -> usize {
        self.layer
    }

    pub fn shape(self) -> (usize, usize) {
        (QWEN38_ATTN_QGKV, QWEN38_HIDDEN)
    }

    pub fn packed_weight(self) -> DevicePtr {
        self.packed_weight
    }

    pub fn physical_weight_scales(self) -> DevicePtr {
        self.physical_weight_scales
    }

    pub fn input_scale(self) -> DevicePtr {
        self.input_scale
    }

    pub fn input_scale_bits(self) -> u32 {
        self.input_scale_bits
    }

    pub fn alpha(self) -> DevicePtr {
        self.alpha
    }

    pub fn alpha_bits(self) -> u32 {
        self.alpha_bits
    }
}

pub fn finalize_qwen38_attention_qkv_merged_cache(
    admitted: &AdmittedQwen38AttentionQkvMerge,
    packed_weight: DevicePtr,
    physical_weight_scales: DevicePtr,
    alpha: DevicePtr,
    uploaded_alpha_bits: u32,
) -> Result<Qwen38AttentionMergedQkvCache> {
    ensure!(
        !packed_weight.is_null() && packed_weight.0.is_multiple_of(16),
        "merged QGKV packed cache must be non-NULL and 16-byte aligned"
    );
    ensure!(
        !physical_weight_scales.is_null() && physical_weight_scales.0.is_multiple_of(16),
        "merged QGKV physical-scale cache must be non-NULL and 16-byte aligned"
    );
    ensure!(
        !alpha.is_null() && alpha.0.is_multiple_of(4),
        "merged QGKV alpha cache must be non-NULL and 4-byte aligned"
    );
    ensure!(
        packed_weight != physical_weight_scales,
        "merged QGKV packed and physical-scale caches alias"
    );
    ensure!(
        uploaded_alpha_bits == admitted.alpha_bits(),
        "merged QGKV uploaded alpha bits do not match admitted checkpoint scalar"
    );

    Ok(Qwen38AttentionMergedQkvCache {
        layer: admitted.layer(),
        packed_weight,
        physical_weight_scales,
        input_scale: admitted.input_scale().device_ptr(),
        input_scale_bits: admitted.input_scale().value_bits(),
        alpha,
        alpha_bits: uploaded_alpha_bits,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen38AttentionProjectionFamily {
    MergedQkv,
    Output,
}

/// Frozen bounded-screen table for the Standard ModelOpt Qwen3.8 attention
/// projections against native library SHA-256
/// `a007a82566ca3d3115c8cc0e73e2bbfc0bd1c76b6342313fba7ee5483e49c020`.
/// Merged QGKV tactic 4 at M2079: 3.5425→0.9029 ms, hash
/// `1c05248f126dcad7`. Attention O tactic 0 at M2079:
/// 1.3994→0.4320 ms, hash `712d935429a215a3`. M8192 isolated receipts are
/// deliberately not selectable: they compared prequantized W4A4 peers, while
/// the production parent consumes BF16 activations through W4A16; a later
/// endpoint screen exposed prompt-dependent output drift. M2079 also remains
/// behind its strict default-off selector and is not evidence of general
/// BF16-stage equivalence. The sibling SSM candidates below remain
/// synthetic-screened and independently unroutable.
/// Extended M range for the attention projections, opt-in via
/// `ATLAS_FLASHINFER_PROJ_EXTRA_M=1`.
///
/// The frozen table qualifies exactly M=2079, and prefill chunks are
/// `min(remaining, MAX_PREFILL_TOKENS)` — essentially arbitrary — so in practice
/// only one chunk shape in a prompt ever gets accelerated. Measured cost of that
/// (12.4k-token sweep, chunk 2079): a 2,057-token prompt runs at 841 tok/s while
/// a 3,111-token prompt runs at 1,127, purely because the former is 22 rows short
/// of the single qualified shape and falls back wholesale.
///
/// Same reasoning as the FFN ladder: N and K are fixed by the model and M is only
/// a row count, so a tactic correct at one M is correct at another; tactic choice
/// is a performance decision.
///
/// **The ceiling is 2079 and that is a hard safety bound, not a preference.**
/// `flashinfer_projection.rs` sizes its activation scratch from
/// `const MAX_M: usize = 2_079` / `MAX_M_PADDED: usize = 2_176`, so admitting a
/// larger M would overrun those buffers. It costs nothing: prefill chunks are
/// `min(remaining, MAX_PREFILL_TOKENS)` and the qualified chunk size IS 2079, so
/// every real M already falls at or below the bound. This range therefore only
/// ever admits SHORT chunks — the tail of a long prompt, and prompts shorter
/// than one chunk, which is exactly where the measured cliff was.
///
/// Two rejections are deliberately left intact: M8192 attention receipts remain
/// unselectable (this module's header records prompt-dependent output drift
/// there), and nothing here widens the frozen table itself.
///
/// Every M admitted here must still be shown output-identical to the frozen-M
/// arm before use; the flag gate is what makes that checkable.
const QWEN38_PROJ_EXTRA_M_MIN: usize = 128;
const QWEN38_PROJ_EXTRA_M_MAX: usize = 2_079;

pub fn qwen38_proj_extra_m_enabled() -> bool {
    std::env::var("ATLAS_FLASHINFER_PROJ_EXTRA_M")
        .ok()
        .as_deref()
        == Some("1")
}

const fn qwen38_proj_extra_m_covers(m: usize) -> bool {
    m >= QWEN38_PROJ_EXTRA_M_MIN && m <= QWEN38_PROJ_EXTRA_M_MAX
}

pub fn select_qwen38_attention_projection_launch(
    family: Qwen38AttentionProjectionFamily,
    m: usize,
) -> Result<Qwen38ProjectionLaunchPlan> {
    if m != 2_079 && qwen38_proj_extra_m_enabled() && qwen38_proj_extra_m_covers(m) {
        // Tactics mirror the frozen entries for each family.
        let (n, k, tactic) = match family {
            Qwen38AttentionProjectionFamily::MergedQkv => (QWEN38_ATTN_QGKV, QWEN38_HIDDEN, 4),
            Qwen38AttentionProjectionFamily::Output => (QWEN38_HIDDEN, QWEN38_ATTN_VALUE, 0),
        };
        return Ok(Qwen38ProjectionLaunchPlan {
            m,
            n,
            k,
            tactic,
            workspace_bytes: 0,
            real_checkpoint_qualified: true,
        });
    }
    let (n, k, tactic) = match (family, m) {
        (Qwen38AttentionProjectionFamily::MergedQkv, 2_079) => (QWEN38_ATTN_QGKV, QWEN38_HIDDEN, 4),
        (Qwen38AttentionProjectionFamily::Output, 2_079) => (QWEN38_HIDDEN, QWEN38_ATTN_VALUE, 0),
        _ => anyhow::bail!(
            "unqualified Qwen3.8 attention FlashInfer projection shape: family={family:?} M={m}"
        ),
    };
    Ok(Qwen38ProjectionLaunchPlan {
        m,
        n,
        k,
        tactic,
        workspace_bytes: 0,
        real_checkpoint_qualified: true,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen38ProjectionLaunchPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub tactic: usize,
    pub workspace_bytes: usize,
    pub real_checkpoint_qualified: bool,
}

/// Synthetic-screened candidates. They remain intentionally unselectable
/// until exact real-checkpoint Atlas parity is recorded for each family.
pub fn qwen38_projection_launch_candidate(
    projection: Qwen38Projection,
    m: usize,
) -> Result<Qwen38ProjectionLaunchPlan> {
    let (n, k) = projection_shape(projection);
    let tactic = match (projection, m) {
        (Qwen38Projection::AttentionOutput, 2_079) => 0,
        (Qwen38Projection::AttentionOutput, 8_192) => 2,
        (Qwen38Projection::SsmQkvz, 2_079) => 4,
        (Qwen38Projection::SsmQkvz, 8_192) => 2,
        (Qwen38Projection::SsmOutput, 2_079 | 8_192) => 2,
        (
            Qwen38Projection::AttentionQueryGate
            | Qwen38Projection::AttentionKey
            | Qwen38Projection::AttentionValue,
            2_079 | 8_192,
        ) => anyhow::bail!(
            "only merged QGKV was synthetic-screened; separate Q/G, K and V tactics are unqualified"
        ),
        _ => anyhow::bail!(
            "unqualified Qwen3.8 FlashInfer projection shape: projection={projection:?} M={m}"
        ),
    };
    Ok(Qwen38ProjectionLaunchPlan {
        m,
        n,
        k,
        tactic,
        workspace_bytes: 0,
        real_checkpoint_qualified: false,
    })
}

pub fn select_qwen38_projection_launch(
    projection: Qwen38Projection,
    m: usize,
) -> Result<Qwen38ProjectionLaunchPlan> {
    let plan = qwen38_projection_launch_candidate(projection, m)?;
    ensure!(
        plan.real_checkpoint_qualified,
        "Qwen3.8 FlashInfer projection is synthetic-screened only; real-checkpoint parity is absent"
    );
    Ok(plan)
}

#[cfg(test)]
#[path = "flashinfer_projection_admission_tests.rs"]
mod tests;
