// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(SiLU(gate) * up)
//! 2 fused kernel launches per decode token (dual GEMV + SiLU-fused down GEMV).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::sync::{Mutex, MutexGuard};

use crate::layer::ForwardContext;
use crate::layers::ffn_dual_tuned_enabled;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

#[cfg(all(feature = "cuda", target_os = "linux"))]
const QUALIFIED_FLASHINFER_SM121_SHA256: [u8; 32] = [
    0xa0, 0x07, 0xa8, 0x25, 0x66, 0xca, 0x3d, 0x31, 0x15, 0xc8, 0xcc, 0x0e, 0x73, 0xe2, 0xbb, 0xfc,
    0x0b, 0xd1, 0xc7, 0x6b, 0x63, 0x42, 0x31, 0x3f, 0xba, 0x7e, 0xe5, 0x48, 0x3e, 0x49, 0xc0, 0x20,
];

/// Scratch buffers for the inline BF16 → NVFP4 activation prequant
/// (W4A4 `nvfp4_nvfp4_gemm` fast path). The exact FlashInfer route
/// preallocates its maximum admitted shape during model construction; other
/// routes allocate on first prefill and resize in-place if M or K grows.
///
/// Sizes (per row M, per col K):
///   - `a_packed`: M × K/2 bytes (E2M1 nibbles)
///   - `a_scale`:  ceil(M/128)×128 × K/16 bytes (FP8 E4M3 scales; the
///     ordinary Atlas layout uses the logical prefix while the optional
///     CUTLASS route uses the full padded 128x4 layout)
///   - `a_max`:    4 bytes (FP32 per-tensor absmax scratch)
///
/// One arena per `DenseFfnLayer` (= per transformer layer). Reused across
/// gate / up / down GEMMs within a layer: gate/up share K=H, down has
/// K=Inter, so the arena is sized for `max(H, Inter)` × `max_M`.
struct E2m1Scratch {
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    a_max: DevicePtr,
    /// Current row capacity (M) the buffers can hold.
    cap_m: usize,
    /// Current column capacity (K) the buffers can hold (full K, not K/2).
    cap_k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct E2m1ScratchLayout {
    cap_m: usize,
    cap_k: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    max_bytes: usize,
}

fn e2m1_scratch_layout(m: usize, k: usize) -> Result<E2m1ScratchLayout> {
    ensure!(m > 0, "W4A4 activation scratch requires M > 0");
    ensure!(
        k >= 16 && k.is_multiple_of(16),
        "W4A4 activation scratch requires K >= 16 and K % 16 == 0; got {k}"
    );
    let cap_m = m.max(128);
    let cap_k = k;
    let elements = cap_m
        .checked_mul(cap_k)
        .context("W4A4 activation scratch size overflow")?;
    let packed_bytes = elements / 2;
    let scale_rows = cap_m
        .div_ceil(128)
        .checked_mul(128)
        .context("W4A4 activation scratch row-padding overflow")?;
    let scale_bytes = scale_rows
        .checked_mul(cap_k)
        .context("W4A4 padded activation-scale scratch size overflow")?
        / 16;
    Ok(E2m1ScratchLayout {
        cap_m,
        cap_k,
        packed_bytes,
        scale_bytes,
        max_bytes: std::mem::size_of::<f32>(),
    })
}

#[derive(Debug, Clone, Copy)]
struct E2m1CheckpointScales {
    /// SGLang's merged gate/up linear takes the maximum of the checkpoint's
    /// logical-partition scales before quantizing their shared input.
    gate_up: f32,
    down: f32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferFfnProjection {
    weight: DevicePtr,
    weight_scales_128x4: DevicePtr,
    weight_scales_hash: u64,
    alpha_f32: DevicePtr,
    input_scale_bits: u32,
    alpha_bits: u32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferMergedGateUp {
    layer: usize,
    weight: DevicePtr,
    weight_scales_128x4: DevicePtr,
    weight_hash: u64,
    weight_scales_hash: u64,
    alpha_f32: DevicePtr,
    input_scale_bits: u32,
    alpha_bits: u32,
    source_gate_weight: DevicePtr,
    source_up_weight: DevicePtr,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferFfnPrefill {
    layer: usize,
    library: ops::flashinfer_sm121::FlashInferSm121,
    merged_gate_up: FlashinferMergedGateUp,
    down: FlashinferFfnProjection,
    quantize_atlas_128x4_k: KernelHandle,
    quantize_merged_silu_atlas_128x4_k: KernelHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlashinferFfnPrefillRoute {
    Disabled,
    Ineligible,
    Missing,
    Complete,
}

/// `m_qualified` is supplied by the caller rather than matched inline so the
/// extended ladder (`ATLAS_FLASHINFER_FFN_EXTRA_M=1`) shares one source of truth
/// with `qwen38_ffn_launch_candidate`; the frozen pair stays the default.
const fn flashinfer_ffn_prefill_route(
    requested: bool,
    exact_qwen38_dense: bool,
    m_qualified: bool,
    prepared: bool,
) -> FlashinferFfnPrefillRoute {
    if !requested {
        FlashinferFfnPrefillRoute::Disabled
    } else if !exact_qwen38_dense || !m_qualified {
        FlashinferFfnPrefillRoute::Ineligible
    } else if !prepared {
        FlashinferFfnPrefillRoute::Missing
    } else {
        FlashinferFfnPrefillRoute::Complete
    }
}

const fn e2m1_kmajor_projection_shape(m: u32, n: u32, k: u32) -> bool {
    m >= 128 && n.is_multiple_of(128) && k.is_multiple_of(64)
}

const E2M1_KMAJOR_M256_MIN_M: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum E2m1KmajorKernel {
    M128,
    M256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum E2m1Scope {
    Full,
    DownOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum E2m1PrefillRoute {
    Disabled,
    Ineligible,
    Complete(E2m1Scope),
    Missing(E2m1Scope),
}

const fn e2m1_prefill_route(
    full_requested: bool,
    down_requested: bool,
    shape_eligible: bool,
    full_ready: bool,
    down_ready: bool,
) -> E2m1PrefillRoute {
    let scope = if full_requested {
        E2m1Scope::Full
    } else if down_requested {
        E2m1Scope::DownOnly
    } else {
        return E2m1PrefillRoute::Disabled;
    };
    if !shape_eligible {
        return E2m1PrefillRoute::Ineligible;
    }
    match scope {
        E2m1Scope::Full if full_ready => E2m1PrefillRoute::Complete(scope),
        E2m1Scope::DownOnly if down_ready => E2m1PrefillRoute::Complete(scope),
        E2m1Scope::Full | E2m1Scope::DownOnly => E2m1PrefillRoute::Missing(scope),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum E2m1SelectedKernel {
    RowMajorM64,
    KmajorM128,
    KmajorM256,
}

const fn e2m1_selected_kernel(kmajor: bool, m256_requested: bool, m: u32) -> E2m1SelectedKernel {
    if !kmajor {
        E2m1SelectedKernel::RowMajorM64
    } else if matches!(
        e2m1_kmajor_kernel(m, m256_requested),
        E2m1KmajorKernel::M256
    ) {
        E2m1SelectedKernel::KmajorM256
    } else {
        E2m1SelectedKernel::KmajorM128
    }
}

const fn e2m1_kmajor_kernel(m: u32, m256_requested: bool) -> E2m1KmajorKernel {
    if m256_requested && m >= E2M1_KMAJOR_M256_MIN_M {
        E2m1KmajorKernel::M256
    } else {
        E2m1KmajorKernel::M128
    }
}

fn validated_e2m1_checkpoint_scales(gate: f32, up: f32, down: f32) -> Result<E2m1CheckpointScales> {
    for (name, value) in [("gate", gate), ("up", up), ("down", down)] {
        ensure!(
            value.is_finite() && value > 0.0,
            "ATLAS_E2M1_STATIC_SCALE requires a finite positive {name} input_scale; got {value}"
        );
    }
    Ok(E2m1CheckpointScales {
        gate_up: gate.max(up),
        down,
    })
}

fn read_e2m1_input_scale(gpu: &dyn GpuBackend, ptr: DevicePtr, projection: &str) -> Result<f32> {
    ensure!(
        ptr != DevicePtr::NULL,
        "ATLAS_E2M1_STATIC_SCALE requires {projection}.input_scale in the checkpoint"
    );
    let mut bytes = [0u8; 4];
    gpu.copy_d2h(ptr, &mut bytes)
        .with_context(|| format!("read {projection}.input_scale for ATLAS_E2M1_STATIC_SCALE"))?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_scale_fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn ensure_flashinfer_ffn_disjoint(
    left: DevicePtr,
    left_len: usize,
    right: DevicePtr,
    right_len: usize,
) -> Result<()> {
    let left_end = left
        .0
        .checked_add(u64::try_from(left_len).context("left FFN pointer extent exceeds u64")?)
        .context("left FFN pointer range overflow")?;
    let right_end = right
        .0
        .checked_add(u64::try_from(right_len).context("right FFN pointer extent exceeds u64")?)
        .context("right FFN pointer range overflow")?;
    ensure!(
        left_end <= right.0 || right_end <= left.0,
        "FlashInfer FFN buffers overlap"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_ffn_extent(rows: usize, cols: usize, element_bytes: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .context("FlashInfer FFN buffer extent overflow")
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_ffn_activation_scale_extent(rows: usize, cols: usize) -> Result<usize> {
    use crate::weight_map::cutlass_scale_layout::{CUTLASS_SCALE_ROW_TILE, NVFP4_GROUP_SIZE};

    ensure!(
        rows > 0,
        "FlashInfer FFN activation-scale extent requires rows > 0"
    );
    ensure!(
        cols >= NVFP4_GROUP_SIZE && cols.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN activation-scale extent requires cols divisible by {NVFP4_GROUP_SIZE}"
    );
    let padded_rows = rows
        .checked_add(CUTLASS_SCALE_ROW_TILE - 1)
        .context("FlashInfer FFN activation-scale row padding overflow")?
        / CUTLASS_SCALE_ROW_TILE;
    let padded_rows = padded_rows
        .checked_mul(CUTLASS_SCALE_ROW_TILE)
        .context("FlashInfer FFN padded activation-scale rows overflow")?;
    flashinfer_ffn_extent(padded_rows, cols / NVFP4_GROUP_SIZE, 1)
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn read_flashinfer_ffn_checkpoint_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<(
    Vec<u8>,
    Vec<u8>,
    crate::weight_map::modelopt_scale_admission::AdmittedModeloptScale,
)> {
    use crate::weight_map::cutlass_scale_layout::NVFP4_GROUP_SIZE;
    use crate::weight_map::modelopt_scale_admission::{
        ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
    };
    use spark_runtime::weights::WeightDtype;

    ensure!(
        !weight.weight.is_null() && !weight.weight_scale.is_null() && !weight.input_scale.is_null(),
        "FlashInfer FFN {projection:?} requires original non-NULL ModelOpt operands"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN {projection:?} has invalid N/K geometry"
    );
    let packed_len = n
        .checked_mul(k / 2)
        .context("FlashInfer FFN packed-weight length overflow")?;
    let scale_len = n
        .checked_mul(k / NVFP4_GROUP_SIZE)
        .context("FlashInfer FFN weight-scale length overflow")?;
    let mut packed = vec![0_u8; packed_len];
    let mut logical_scales = vec![0_u8; scale_len];
    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.weight, &mut packed)
        .with_context(|| format!("read ModelOpt FFN {projection:?} packed weight"))?;
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .with_context(|| format!("read ModelOpt FFN {projection:?} block scales"))?;
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .with_context(|| format!("read ModelOpt FFN {projection:?} input scale"))?;
    let modelopt_projection = match projection {
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Gate => {
            ModeloptScaleProjection::FfnGate
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Up => {
            ModeloptScaleProjection::FfnUp
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Down => {
            ModeloptScaleProjection::FfnDown
        }
    };
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    Ok((packed, logical_scales, input_scale))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_merged_gate_up(
    gpu: &dyn GpuBackend,
    layer: usize,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Result<FlashinferMergedGateUp> {
    use crate::weight_map::flashinfer_ffn_admission::{
        QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnCheckpointProjection, Qwen38FfnProjection,
        Qwen38FfnSource, admit_qwen38_merged_gate_up,
    };

    let (gate_packed, gate_scales, gate_input_scale) = read_flashinfer_ffn_checkpoint_projection(
        gpu,
        layer,
        Qwen38FfnProjection::Gate,
        gate,
        QWEN38_INTERMEDIATE,
        QWEN38_HIDDEN,
    )?;
    let (up_packed, up_scales, up_input_scale) = read_flashinfer_ffn_checkpoint_projection(
        gpu,
        layer,
        Qwen38FfnProjection::Up,
        up,
        QWEN38_INTERMEDIATE,
        QWEN38_HIDDEN,
    )?;
    let admitted = admit_qwen38_merged_gate_up(
        layer,
        Qwen38FfnCheckpointProjection {
            source: Qwen38FfnSource {
                layer,
                projection: Qwen38FfnProjection::Gate,
            },
            packed_weight: &gate_packed,
            logical_weight_scales: &gate_scales,
            input_scale: gate_input_scale,
            weight_scale_2_le_bytes: gate.weight_scale_2.to_le_bytes(),
        },
        Qwen38FfnCheckpointProjection {
            source: Qwen38FfnSource {
                layer,
                projection: Qwen38FfnProjection::Up,
            },
            packed_weight: &up_packed,
            logical_weight_scales: &up_scales,
            input_scale: up_input_scale,
            weight_scale_2_le_bytes: up.weight_scale_2.to_le_bytes(),
        },
    )?;

    let merged_weight = gpu.alloc(admitted.packed_weight().len())?;
    gpu.copy_h2d(admitted.packed_weight(), merged_weight)
        .context("upload immutable merged FFN gate/up packed weight")?;
    let merged_scales = gpu.alloc(admitted.physical_weight_scales().len())?;
    gpu.copy_h2d(admitted.physical_weight_scales(), merged_scales)
        .context("upload immutable merged FFN gate/up physical scales")?;
    let alpha_f32 = gpu.alloc(std::mem::size_of::<f32>())?;
    gpu.copy_h2d(&admitted.shared_alpha_bits().to_le_bytes(), alpha_f32)
        .context("upload immutable merged FFN gate/up alpha")?;

    Ok(FlashinferMergedGateUp {
        layer,
        weight: merged_weight,
        weight_scales_128x4: merged_scales,
        weight_hash: flashinfer_scale_fingerprint(admitted.packed_weight()),
        weight_scales_hash: flashinfer_scale_fingerprint(admitted.physical_weight_scales()),
        alpha_f32,
        input_scale_bits: admitted.input_scales()[0].value_bits(),
        alpha_bits: admitted.shared_alpha_bits(),
        source_gate_weight: gate.weight,
        source_up_weight: up.weight,
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_ffn_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<FlashinferFfnProjection> {
    use crate::weight_map::cutlass_scale_layout::{
        NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
    };
    use crate::weight_map::modelopt_scale_admission::{
        ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
    };
    use spark_runtime::weights::WeightDtype;

    ensure!(
        !weight.weight.is_null(),
        "FlashInfer FFN packed weight is null"
    );
    ensure!(
        !weight.weight_scale.is_null(),
        "FlashInfer FFN block-scale pointer is null"
    );
    ensure!(
        !weight.input_scale.is_null(),
        "FlashInfer FFN input-scale pointer is null"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN projection has invalid N/K geometry"
    );

    let groups = k / NVFP4_GROUP_SIZE;
    let scale_len = n
        .checked_mul(groups)
        .context("FlashInfer FFN weight-scale length overflow")?;
    let mut logical_scales = vec![0_u8; scale_len];
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .context("read ModelOpt FFN block scales for CUTLASS interleave")?;
    let physical_scales =
        interleave_nvfp4_scales_128x4(&logical_scales, &[n, groups], NVFP4_GROUP_SIZE)?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(&physical_scales, &[n, groups], NVFP4_GROUP_SIZE,)?
            == logical_scales,
        "FlashInfer FFN physical block scales failed exact round trip"
    );

    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .context("read ModelOpt FFN input scale")?;
    let modelopt_projection = match projection {
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Gate => {
            ModeloptScaleProjection::FfnGate
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Up => {
            ModeloptScaleProjection::FfnUp
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Down => {
            ModeloptScaleProjection::FfnDown
        }
    };
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    ensure!(
        weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
        "FlashInfer FFN weight_scale_2 must be finite and positive"
    );
    let alpha = input_scale.value() * weight.weight_scale_2;
    ensure!(
        alpha.is_finite() && alpha > 0.0,
        "FlashInfer FFN alpha must be finite and positive"
    );

    let weight_scales_128x4 = gpu.alloc(physical_scales.len())?;
    gpu.copy_h2d(&physical_scales, weight_scales_128x4)
        .context("upload FlashInfer FFN physical block scales")?;
    let alpha_f32 = gpu.alloc(std::mem::size_of::<f32>())?;
    gpu.copy_h2d(&alpha.to_le_bytes(), alpha_f32)
        .context("upload FlashInfer FFN alpha")?;

    Ok(FlashinferFfnProjection {
        weight: weight.weight,
        weight_scales_128x4,
        weight_scales_hash: flashinfer_scale_fingerprint(&physical_scales),
        alpha_f32,
        input_scale_bits: input_scale.value_bits(),
        alpha_bits: alpha.to_bits(),
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn log_flashinfer_ffn_success(
    layer: usize,
    operation: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnOperation,
    m: usize,
) {
    use crate::weight_map::flashinfer_ffn_admission::{
        Qwen38FfnOperation, select_qwen38_ffn_launch,
    };

    static RECEIPTS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let (family_offset, family) = match operation {
        Qwen38FfnOperation::MergedGateUp => (0, "ffn_merged_gate_up"),
        Qwen38FfnOperation::Down => (2, "ffn_down"),
        Qwen38FfnOperation::Gate | Qwen38FfnOperation::Up => {
            unreachable!("production FFN route does not launch separate gate/up")
        }
    };
    let bit = 1_u8 << (family_offset + usize::from(m == 8_192));
    if RECEIPTS.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit != 0 {
        return;
    }
    let plan = select_qwen38_ffn_launch(operation, m)
        .expect("a successful FFN route already selected an exact frozen plan");
    tracing::info!(
        layer,
        m,
        tactic = plan.tactic,
        family,
        "routed Qwen3.8 dense FFN prefill through FlashInfer SM121"
    );
    eprintln!(
        "ATLAS_PREFILL_FFN_FLASHINFER ENGAGED family={family} layer={layer} M={m} tactic={}",
        plan.tactic
    );
}

fn load_e2m1_checkpoint_scales(
    weights: &DenseFfnWeights,
    gpu: &dyn GpuBackend,
) -> Result<Option<E2m1CheckpointScales>> {
    let flashinfer_requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
    let requested = flashinfer_requested
        || std::env::var("ATLAS_E2M1_STATIC_SCALE").ok().as_deref() == Some("1");
    if !requested {
        return Ok(None);
    }
    let full_w4a4 = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
    let down_w4a4 = std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
    ensure!(
        full_w4a4 || down_w4a4 || flashinfer_requested,
        "ATLAS_E2M1_STATIC_SCALE=1 requires an active W4A4 or FlashInfer FFN route"
    );

    let scales = validated_e2m1_checkpoint_scales(
        read_e2m1_input_scale(gpu, weights.gate_proj.input_scale, "gate_proj")?,
        read_e2m1_input_scale(gpu, weights.up_proj.input_scale, "up_proj")?,
        read_e2m1_input_scale(gpu, weights.down_proj.input_scale, "down_proj")?,
    )?;
    tracing::info!(
        gate_up_input_scale = scales.gate_up,
        down_input_scale = scales.down,
        "enabled checkpoint-static W4A4 activation scales"
    );
    Ok(Some(scales))
}

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct E2m1RouteProof {
    scope: E2m1Scope,
    kernel: E2m1SelectedKernel,
    checkpoint_static: bool,
    fused_silu_input: bool,
    activation: FfnActivation,
    m: u32,
    h: u32,
    inter: u32,
}

fn log_e2m1_route_once(proof: E2m1RouteProof) {
    static SEEN: std::sync::OnceLock<Mutex<std::collections::HashSet<E2m1RouteProof>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !seen.insert(proof) {
        return;
    }
    let scope = match proof.scope {
        E2m1Scope::Full => "full",
        E2m1Scope::DownOnly => "down-only",
    };
    let gate_up = match proof.scope {
        E2m1Scope::Full => "e2m1",
        E2m1Scope::DownOnly => "w4a16-m128",
    };
    let kernel = match proof.kernel {
        E2m1SelectedKernel::RowMajorM64 => "row-major-m64",
        E2m1SelectedKernel::KmajorM128 => "kmajor-m128",
        E2m1SelectedKernel::KmajorM256 => "kmajor-m256",
    };
    let activation_scale = if proof.checkpoint_static {
        "checkpoint-static"
    } else {
        "dynamic-absmax"
    };
    let down_input = if proof.fused_silu_input {
        "fused-silu-nvfp4"
    } else {
        "separate-bf16-silu"
    };
    let activation = match proof.activation {
        FfnActivation::SiLU => "silu",
        FfnActivation::GeLU => "gelu",
    };
    tracing::info!(
        "ENGAGED ATLAS_E2M1_PREFILL: scope={scope} gate_up={gate_up} \
         e2m1_kernel={kernel} activation_scale={activation_scale} \
         down_input={down_input} activation={activation} M={} H={} intermediate={}",
        proof.m,
        proof.h,
        proof.inter,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillFfnFastRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

const fn prefill_ffn_fast_route(
    requested: bool,
    shape_eligible: bool,
    weights_ready: bool,
    m16_kernel_ready: bool,
    m128_kernel_ready: bool,
) -> PrefillFfnFastRoute {
    if !requested {
        return PrefillFfnFastRoute::Disabled;
    }
    if !shape_eligible {
        return PrefillFfnFastRoute::Ineligible;
    }
    if weights_ready && m16_kernel_ready && m128_kernel_ready {
        PrefillFfnFastRoute::Complete
    } else {
        PrefillFfnFastRoute::Missing
    }
}

fn log_prefill_ffn_fast_route_once(m: u32, h: u32, inter: u32) {
    static FAST: std::sync::Once = std::sync::Once::new();
    FAST.call_once(|| {
        tracing::info!(
            "ENGAGED ATLAS_PREFILL_FFN_FAST: approximate transposed M128 route complete M={m} H={h} intermediate={inter}"
        );
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillFusedEpilogueRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

fn prefill_fused_epilogue_route(
    requested: bool,
    parent_pipe_ready: bool,
    has_kernel: bool,
    activation: FfnActivation,
    m: u32,
    h: u32,
    inter: u32,
) -> PrefillFusedEpilogueRoute {
    if !requested {
        return PrefillFusedEpilogueRoute::Disabled;
    }
    if activation != FfnActivation::SiLU
        || m <= 32
        || !h.is_multiple_of(64)
        || !inter.is_multiple_of(64)
    {
        return PrefillFusedEpilogueRoute::Ineligible;
    }
    if parent_pipe_ready && has_kernel {
        PrefillFusedEpilogueRoute::Complete
    } else {
        PrefillFusedEpilogueRoute::Missing
    }
}

fn log_prefill_pipe_route_once() {
    static PIPE: std::sync::Once = std::sync::Once::new();
    PIPE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_PIPE: ordinary exact pipe route");
    });
}

fn log_prefill_up_fused_route_once() {
    static UP_FUSED: std::sync::Once = std::sync::Once::new();
    UP_FUSED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_FUSED_EPILOGUE: exact up-only fusion");
    });
}

fn log_prefill_dual_fused_route_once() {
    static DUAL_FUSED: std::sync::Once = std::sync::Once::new();
    DUAL_FUSED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_DUAL_FUSED: exact gate/up fusion");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactFfnDispatch {
    Existing,
    Batched,
    PerRowK1,
}

const fn exact_ffn_dispatch(
    rows: u32,
    exact_w4_silu_eligible: bool,
    selected_tier_present: bool,
) -> ExactFfnDispatch {
    if !exact_w4_silu_eligible || rows <= 1 || rows > 32 {
        ExactFfnDispatch::Existing
    } else if selected_tier_present {
        ExactFfnDispatch::Batched
    } else {
        ExactFfnDispatch::PerRowK1
    }
}

const fn exact_ffn_auto_kgamma_applicable(rows: u32, exact_w4_silu_eligible: bool) -> bool {
    matches!(
        exact_ffn_dispatch(rows, exact_w4_silu_eligible, false),
        ExactFfnDispatch::PerRowK1
    )
}

const fn w3_kgamma_applicable(rows: u32, activation_silu: bool, prepared: bool) -> bool {
    rows > 1 && rows <= 32 && activation_silu && prepared
}

#[derive(Debug, Clone, Copy)]
struct ExactFfnMaterializedAvailability {
    m8_dual_silu: bool,
    m8_f32_down: bool,
    m17_dual_silu: bool,
    m17_f32_down: bool,
    m17_fused_dual_silu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactFfnMaterializedRoute {
    Inline,
    Split,
    FusedM17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactFfnPhysicalRoute {
    Native,
    SplitM8x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactFfnRowOffsets {
    input: usize,
    gate_up: usize,
    preactivation: usize,
    output: usize,
}

fn exact_ffn_row_offsets(
    first_row: u32,
    hidden: u32,
    intermediate: u32,
) -> Option<ExactFfnRowOffsets> {
    let row = first_row as usize;
    let hidden = hidden as usize;
    let intermediate = intermediate as usize;
    Some(ExactFfnRowOffsets {
        input: row.checked_mul(hidden)?.checked_mul(2)?,
        gate_up: row.checked_mul(intermediate)?.checked_mul(2)?,
        preactivation: row.checked_mul(intermediate)?.checked_mul(4)?,
        output: row.checked_mul(hidden)?.checked_mul(2)?,
    })
}

fn exact_ffn_physical_route(
    rows: u32,
    intermediate: u32,
    split_m8_enabled: bool,
    m8_exact_complete: bool,
    m8_materialized_complete: bool,
    scratch_bytes: usize,
) -> ExactFfnPhysicalRoute {
    let required_bytes = (rows as usize)
        .checked_mul(intermediate as usize)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
    if rows == 16
        && split_m8_enabled
        && m8_exact_complete
        && m8_materialized_complete
        && required_bytes.is_some_and(|required| required <= scratch_bytes)
    {
        ExactFfnPhysicalRoute::SplitM8x2
    } else {
        ExactFfnPhysicalRoute::Native
    }
}

fn exact_ffn_split_m8_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_EXACT_FFN_SPLIT_M8").ok().as_deref() == Some("1"))
}

fn exact_ffn_lowreg_gate_up_m16_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_EXACT_FFN_LOWREG_GATE_UP_M16")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// `ATLAS_FFN_TC=1` REFREEZE switch: bypass the bit-exact scalar-GEMV FFN
/// path (`forward_kgamma_exact` / `forward_kgamma_k1_rows`) and route
/// `forward_kgamma` through the tensor-core transposed-weight GEMMs
/// (m32_n64 / m128). The MMA reduction order, BF16 weight rounding and
/// dequant re-association differ from the serial FMA oracle, so enabling
/// this changes token output: the reference completion hash MUST be
/// re-established ("refreeze") after turning it on. Default off.
fn exact_ffn_tc_override() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_FFN_TC").ok().as_deref() == Some("1"))
}

fn exact_ffn_materialized_route(
    rows: u32,
    intermediate: u32,
    handles: ExactFfnMaterializedAvailability,
    scratch_bytes: usize,
) -> ExactFfnMaterializedRoute {
    let (dual_silu_handle_present, f32_down_handle_present) = match rows {
        5..=8 => (handles.m8_dual_silu, handles.m8_f32_down),
        9..=17 => (handles.m17_dual_silu, handles.m17_f32_down),
        _ => return ExactFfnMaterializedRoute::Inline,
    };
    let required_bytes = (rows as usize)
        .checked_mul(intermediate as usize)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
    if required_bytes.is_none_or(|required| required > scratch_bytes) || !f32_down_handle_present {
        return ExactFfnMaterializedRoute::Inline;
    }
    if matches!(rows, 9..=17) && handles.m17_fused_dual_silu {
        ExactFfnMaterializedRoute::FusedM17
    } else if dual_silu_handle_present {
        ExactFfnMaterializedRoute::Split
    } else {
        ExactFfnMaterializedRoute::Inline
    }
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    /// Exact K1-arithmetic-order W4 FFN kernels for dynamic M=2..=32.
    /// Missing selected-tier handles fail closed to independent K1 rows.
    exact_ffn_kernels: ops::W4a16ExactFfnKernels,
    /// Optional full-M16 gate/up projections. The diagnostic route is atomic;
    /// any missing handle retains the current split-M8 implementation.
    exact_ffn_lowreg_m16: ops::W4a16ExactFfnLowregM16Kernels,
    /// Single-warp-per-output M=1 decode GEMVs — lossless, 8 outputs per
    /// 256-thread block instead of 4, no cross-warp smem round-trip. Loaded
    /// via `try_kernel` so a kernel cache built before these symbols existed
    /// still links; `KernelHandle(0)` on a miss falls back to the 64-thread
    /// base kernels, which are bit-identical.
    w4a16_gemv_sw: KernelHandle,
    w4a16_gemv_dual_sw: KernelHandle,
    w4a16_gemv_silu_input_sw: KernelHandle,
    /// `ATLAS_NO_GEMV_SW != "1"`, cached at construction.
    gemv_sw: bool,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    /// Tuned dual-batch3 variant: fuses gate+up into the SAME CTA so the
    /// 3-token activation vector is loaded once per CTA. Gated behind the
    /// `ATLAS_FFN_DUAL_TUNED=1` env var; falls back to the baseline kernel
    /// when off. Loaded via `try_kernel` so older built kernel caches that
    /// pre-date this symbol still link.
    w4a16_gemv_dual_batch3_tuned: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    w4a16_gemm: KernelHandle,
    /// cp.async double-buffered byte-exact shadow of `w4a16_gemm` for the
    /// prefill FFN path. Same dequant arithmetic + MMA order; only the load
    /// pipeline differs. Loaded via `try_kernel`; handle 0 disables the
    /// `ATLAS_PREFILL_FFN_PIPE` route silently.
    w4a16_gemm_pipe: KernelHandle,
    /// Exact up-projection + SiLU(gate) epilogue variant used only by the
    /// opt-in large-M prefill pipe route. It preserves the BF16 up round trip
    /// while avoiding the full up activation buffer and standalone SiLU
    /// launch. Missing symbol keeps the existing three-stage route.
    w4a16_gemm_pipe_silu_mul: KernelHandle,
    /// Exact large-M gate+up+SiLU fusion. Both projection accumulators retain
    /// their BF16 materialization boundary, but A is loaded once and neither
    /// intermediate activation is written to global memory.
    w4a16_gemm_pipe_dual: KernelHandle,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    /// Transposed (`nvfp4_t` layout) FFN projections — populated only when
    /// `ATLAS_FFN_M16_TRANSPOSED=1` and the loader successfully built the
    /// transposed copies via `QuantizedWeight::transpose_for_gemm`. When
    /// `Some`, `forward_kgamma` routes gate/up/down through the
    /// `w4a16_gemm_n128_m16` (M_TILE=16) kernel which has near-zero MMA
    /// accumulator waste at M=γ+1 (typically 17). Falls back to the
    /// non-transposed `w4a16_gemm` (M_TILE=64) when `None` OR when the
    /// `w4a16_gemm_t_m16` kernel symbol is missing.
    gate_proj_t: Option<QuantizedWeight>,
    up_proj_t: Option<QuantizedWeight>,
    down_proj_t: Option<QuantizedWeight>,
    /// `w4a16_gemm_t_m16` — small-M (M_TILE=16) transposed-weight GEMM.
    /// Loaded via `try_kernel`; handle is 0 when the symbol is missing
    /// (older PTX caches), in which case `forward_kgamma` always uses the
    /// non-transposed fallback regardless of the transposed weights.
    w4a16_gemm_t_m16: KernelHandle,
    /// `w4a16_gemm_t_m16_n64` — small-M (M_TILE=16), small-N (N_TILE=64)
    /// transposed-weight GEMM, tuned for the K=3 MTP verify path on dense
    /// Qwen3.6-27B (M=3 padded to MMA-16). At intermediate=17408 the N=128
    /// parent only fields ~136 CTAs/projection (1.2 CTAs/SM on GB10) so
    /// the SMs are starved; N=64 doubles the grid to ~272 CTAs/projection
    /// at half the per-CTA work. Loaded via `try_kernel`; handle 0 falls
    /// back to the GEMV path silently.
    w4a16_gemm_t_m16_n64: KernelHandle,
    /// `w4a16_gemm_t_m128` — large-M (M_TILE=128, N_TILE=128)
    /// transposed-weight GEMM. Loaded via `try_kernel`; handle is 0 when
    /// the symbol is missing. Used by `forward_prefill` when
    /// `ATLAS_PREFILL_FFN_FAST=1` AND transposed weights are installed
    /// AND M >= 128. Mirrors the attention `w4a16_gemm_t_m128_k`
    /// dispatch in `qwen3_attention/prefill_weights.rs`. Designed for
    /// large-M prefill: kernel comment claims ~2x speedup over
    /// `w4a16_gemm_t` at ISL>128. For Qwen3.6-27B prefill at M=3575,
    /// N=17408 (gate/up dual): grid=(136, 28)=3808 CTAs vs the default
    /// (272, 56)=15232 CTAs — 4x fewer CTAs but 4x more work per CTA,
    /// and ~2x less weight DRAM traffic.
    w4a16_gemm_t_m128: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` — DFlash K=17 verify specialization:
    /// single B read (one 32-row M-tile) × 272 CTAs (N_TILE=64).
    /// Loaded via `try_kernel`; 0 falls back to m128/m16.
    w4a16_gemm_t_m32_n64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu` — FUSED gate_proj + up_proj +
    /// SiLU·mul for the K=γ verify path. Loads the shared [M,K] input
    /// tile once, streams BOTH transposed weights (gate + up), and writes
    /// only the fused silu(gate)*up [M,N] activation in one launch (vs the
    /// baseline's two m32_n64 GEMMs + standalone `moe_silu_mul`). Gated by
    /// `ATLAS_FFN_FUSED_GATEUP`. Loaded via `try_kernel`; handle 0 keeps
    /// the split gate/up path.
    w4a16_gemm_t_m32_n64_gateup_silu: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe` — DEQUANT-IN-REGISTERS fork of
    /// the fused gate+up+SiLU kernel (`ATLAS_DEQUANT_PIPE=1`). Byte-exact
    /// with `w4a16_gemm_t_m32_n64_gateup_silu` (same shape, accumulation
    /// order, and BF16 round-trips) but the NVFP4→FP8 dequant runs in
    /// registers immediately before each MMA instead of via a `smem_B_fp8`
    /// staging array — dropping the 2nd per-K-step `__syncthreads`, shrinking
    /// SMEM ~27% (10.9 KB vs 15.0 KB → higher occupancy), and using
    /// `cp.async.wait_group<1>` so the next tile's load overlaps the current
    /// dequant+MMA. Loaded via `try_kernel`; handle 0 keeps the staged fused
    /// kernel as silent fallback.
    w4a16_gemm_t_m32_n64_gateup_silu_pipe: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64` — K_STEP=64 fork of the
    /// `_pipe` register-dequant kernel (`ATLAS_GATEUP_K64=1`). Doubles the
    /// K-tile from 32 to 64 elements: 80 K-loop iterations vs 160, halving
    /// sync count and loop overhead. Each step issues two m16n8k32 MMAs per
    /// accumulator (K[0..31] then K[32..63]) and 2× the cp.async volume
    /// (6 KB vs 3 KB per stage), so the background load overlaps more of the
    /// inline dequant+compute work. Takes priority over ATLAS_DEQUANT_PIPE.
    /// Requires K divisible by 64 (hidden_size 5120 qualifies). Handle 0
    /// silently falls back through _pipe → staged fused kernel.
    w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_splitk` — split-K variant of the above for
    /// the DFlash K=17 verify `down_proj` ([M=17,N=5120,K=16384]). The
    /// single-slice kernel fields only 80 CTAs (N=5120/64) and is
    /// occupancy-starved on the long K=16384 loop (~91 GB/s vs gate/up's
    /// ~163). Split-K multiplies CTAs by k_splits into an FP32 workspace,
    /// then `reduce_splitk_f32_to_bf16` sums + downcasts. Gated by
    /// `ATLAS_FFN_DOWN_SPLITK` (default 4; 0/1 disables). Loaded via
    /// `try_kernel` — handle 0 keeps the single-slice m32_n64 path.
    w4a16_gemm_t_m32_n64_splitk: KernelHandle,
    /// `reduce_splitk_f32_to_bf16` — companion reduce kernel for the
    /// split-K down_proj. Sums the k_splits FP32 partial bands → BF16.
    reduce_splitk_k: KernelHandle,
    /// Lazily-allocated FP32 split-K workspace [k_splits, M, N].
    /// `Mutex` because `forward_kgamma` takes `&self`.
    splitk_workspace: Mutex<Option<DevicePtr>>,
    /// `w4a16_gemm_t_m128_v2` — 8-warp (blockDim 256) shadow of
    /// `w4a16_gemm_t_m128`. Same 2-stage cp.async pipeline + same SMEM
    /// footprint (~29.8KB → 3 CTAs/SM), but parallelizes chunk 0 and
    /// chunk 1 MMA computation across warps {0-3} and {4-7} instead of
    /// serializing both chunks on 4 warps. Yields 2× more warps/SM (768
    /// vs 384) → more MMA pipeline slots in flight. Originally a
    /// MiniMax-only kernel (kernels/gb10/minimax-m2-229b/nvfp4/
    /// w4a16_gemm_v2.cu) — copied verbatim into the qwen3.6-27b target
    /// so the FFN prefill GEMM can route through it. Gated by
    /// `ATLAS_FFN_M128_V2=1`. Loaded via `try_kernel`; handle 0 keeps
    /// the v1 path as silent fallback.
    w4a16_gemm_t_m128_v2: KernelHandle,
    /// `fp8_gemm_t_m128` — large-M FP8×FP8 GEMM kernel for pre-dequanted
    /// FFN weights. Loaded via `try_kernel`. When set together with
    /// `gate_fp8`/`up_fp8`/`down_fp8` (installed by
    /// `predequant_for_prefill`), the FFN prefill bypasses NVFP4 dequant
    /// entirely — saving 1 __syncthreads + the 16-iteration dequant
    /// phase per K-step inside `w4a16_gemm_t_m128`. Mirrors the
    /// attention `predequant_for_prefill` + `fp8_gemm_n128_m128`
    /// pattern (qwen3_attention/prefill_weights.rs:161).
    fp8_gemm_t_m128_k: KernelHandle,
    /// Pre-dequanted FP8 [N, K] gate weight. `Some` only when
    /// `ATLAS_FFN_PREDEQUANT_FP8=1` is set at startup and the
    /// `predequant_nvfp4_to_fp8` + `fp8_gemm_t_m128` kernels are present.
    /// Memory cost: N×K bytes per projection (e.g. 17408×5120 = 89 MB
    /// for gate/up, 5120×17408 = 89 MB for down → ~17 GB per Qwen3.6-27B
    /// run, ~270 MB per layer × 64 layers). Roughly DOUBLES the FFN
    /// weight footprint vs NVFP4-only — gate by intent.
    gate_fp8: Option<DevicePtr>,
    up_fp8: Option<DevicePtr>,
    down_fp8: Option<DevicePtr>,
    /// W4A4 NVFP4×NVFP4 native tensor-core GEMM
    /// (`nvfp4_cutlass::nvfp4_nvfp4_gemm_t_m64`). Loaded via `try_kernel`
    /// — handle 0 silently disables the `ATLAS_E2M1_GEMM` fast path.
    nvfp4_gemm_k: KernelHandle,
    /// K-major M=128 W4A4 shadow consuming `transpose_for_gemm` buffers.
    /// Handle 0 makes an eligible explicit `ATLAS_E2M1_KMAJOR=1` request
    /// fail closed; the ordinary row-major W4A4 route is otherwise unchanged.
    nvfp4_gemm_kmajor_m128_k: KernelHandle,
    /// Sixteen-warp M=256 K-major long-prefill shadow. An eligible explicit
    /// `ATLAS_E2M1_KMAJOR_M256=1` request fails closed when this is absent;
    /// M128 remains the K-major fallback below the M256 saturation threshold.
    nvfp4_gemm_kmajor_m256_k: KernelHandle,
    /// `quantize_nvfp4::nvfp4_global_absmax` — per-tensor absmax scan
    /// used to derive the activation `scale2` for the W4A4 path.
    nvfp4_absmax_k: KernelHandle,
    /// `quantize_nvfp4::quantize_bf16_to_nvfp4` — per-row E2M1 quantizer
    /// used to convert BF16 activations to the W4A4 GEMM input layout.
    nvfp4_quantize_k: KernelHandle,
    /// Fused SiLU(gate)*up -> NVFP4 down-input quantizer. Handle 0 makes an
    /// eligible explicit `ATLAS_E2M1_SILU_QUANT=1` request fail closed.
    nvfp4_silu_quantize_k: KernelHandle,
    /// Scratch arena for the W4A4 fast path. Holds packed activation nibbles,
    /// per-group FP8 scales, and absmax scratch. FlashInfer preparation
    /// allocates its maximum admitted shape at construction; other routes
    /// allocate lazily and resize if M or K grows. `Mutex` because
    /// `forward_prefill` takes `&self`.
    e2m1_scratch: Mutex<Option<E2m1Scratch>>,
    /// Serializes every full launch group that uses `e2m1_scratch` and binds
    /// it to one CUDA stream. The kernels are asynchronous, so allocation-only
    /// locking is insufficient: a later caller on another stream could
    /// overwrite or free operands still in flight. Reuse on the same stream is
    /// safe because CUDA preserves enqueue order.
    e2m1_use_stream: Mutex<Option<u64>>,
    /// Optional ModelOpt calibration scales read once during layer
    /// construction. `ATLAS_E2M1_STATIC_SCALE=1` uses these with the existing
    /// on-device quantizer, eliminating runtime absmax scans and host syncs.
    /// This is a different activation-quantization contract from dynamic
    /// absmax and remains default-off pending output-quality qualification.
    e2m1_checkpoint_scales: Option<E2m1CheckpointScales>,
    /// Exact Atlas-quantization / FlashInfer-GEMM prefill route. Constructed
    /// only by the Qwen3.8 loader after every original checkpoint operand,
    /// scalar, physical 128x4 scale copy, native symbol and zero-workspace
    /// tactic has passed admission. Default-unrouted on every other build.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    flashinfer_ffn_prefill: Option<FlashinferFfnPrefill>,
    /// `ffn_sparsity_measure` — TEAL-style activation-sparsity observer for
    /// the sparsity-drafted self-speculation feasibility gate
    /// (`ATLAS_MEASURE_FFN_SPARSITY=1`). Loaded via `try_kernel`; handle 0
    /// disables the measurement silently (older kernel caches).
    ffn_sparsity_measure_k: KernelHandle,
    /// Lazily-allocated device counter buffers for the sparsity measurement.
    /// `Mutex` because `forward` takes `&self`. Allocated on first measured
    /// `forward` call (gpu.alloc is illegal during graph capture, but the
    /// self-spec draft + measurement run EAGER — see the gate docs). None
    /// until the first measured forward.
    sparsity_meas: Mutex<Option<SparsityMeas>>,
    /// W3 (3-bit weight) FFN projections — mixed-precision byte-reduction
    /// lane. `Some` only when `ATLAS_FFN_W3_LAYERS` names this layer AND
    /// the repacked sidecar (`ATLAS_FFN_W3_SIDECAR`, built by
    /// `local/tools/repack_w3.py`) contained its tensors; installed by the
    /// loader via `set_w3_weights`. GEMV layout `[N, 3K/8]` — used by the
    /// single-token `forward` SiLU path (dual gate/up + fused SiLU down).
    /// Cuts packed FFN weight bytes 25% vs NVFP4 on a weight-bandwidth-
    /// bound decode. NOT md5-gated (weights differ from W4 by
    /// construction) — quality is protected by the ABBA eval gate; the
    /// default path (gate off / no sidecar) stays byte-identical.
    w3_weights: Option<DenseFfnWeights>,
    /// Transposed W3 copies (`[3K/8, N_pad64]`) for the K=γ verify GEMM
    /// path (`w3a16_gemm_t_m32_n64`). Built host-side by the sidecar
    /// loader. `forward_kgamma` routes gate/up/down through the W3 GEMM
    /// when set (superseding the W4 m32/fused/split-K variants on this
    /// layer). Other paths (prefill, K=2/3 batched GEMV) intentionally
    /// stay on the retained W4 weights — they are not weight-bandwidth-
    /// bound the same way and keep their higher-precision copies.
    w3_weights_t: Option<DenseFfnWeights>,
    /// `w3a16_gemv_dual` — W3 gate+up dual GEMV. `try_kernel`; handle 0
    /// (older PTX caches) disables the W3 GEMV path silently.
    w3a16_gemv_dual_k: KernelHandle,
    /// `w3a16_gemv_silu_input` — W3 fused SiLU-input down GEMV.
    w3a16_gemv_silu_input_k: KernelHandle,
    /// `w3a16_gemm_t_m32_n64` — W3 clone of the m32_n64 verify GEMM.
    w3a16_gemm_t_m32_n64_k: KernelHandle,
    /// `ffn_build_keep_chunks` — on-device keep-chunk selector for the SPARSE
    /// self-spec DRAFT path (`ATLAS_SELF_SPEC_SPARSE=1`). Handle 0 disables.
    ffn_build_keep_chunks_k: KernelHandle,
    /// `w4a16_gemv_sparse_cols` — column-sparse GEMV for the SPARSE draft
    /// path. Handle 0 disables the sparse draft (falls back to dense GEMV).
    w4a16_gemv_sparse_cols_k: KernelHandle,
    /// Lazily-allocated per-layer keep_idx / keep_len device scratch for the
    /// sparse draft path. `Mutex` because `forward_draft_sparse` takes `&self`.
    sparse_draft_scratch: Mutex<Option<SparseDraftScratch>>,
}

/// Per-layer device counter buffers for the FFN activation-sparsity
/// measurement. Two sites (gate/up input + down input), each with a
/// `NUM_THRESH`-entry u32 histogram (below-threshold counts, atomically
/// accumulated) and a 2-entry u32 `count` ([0]=rows seen, [1]=elements seen).
struct SparsityMeas {
    /// Histogram for site 0 (gate/up input, K=hidden). `NUM_THRESH` u32s.
    hist_gateup: DevicePtr,
    /// [rows, elements] u32 counter for site 0.
    count_gateup: DevicePtr,
    /// Histogram for site 1 (down input, K=intermediate). `NUM_THRESH` u32s.
    hist_down: DevicePtr,
    /// [rows, elements] u32 counter for site 1.
    count_down: DevicePtr,
    /// Dedicated BF16 scratch [1, intermediate] into which the observer
    /// recomputes `silu(gate)*up` for the DOWN-input site. This is a SEPARATE
    /// buffer from the token-stream's `gate_out`/`up_out` — the fused down
    /// GEMV (`w4a16_gemv_silu_input`) applies SiLU internally and never
    /// materialises a standalone silu'd vector, so the observer computes its
    /// own copy here WITHOUT touching the buffers the fused kernel reads.
    /// Keeps the measurement a pure reader (token stream byte-identical).
    meas_silu: DevicePtr,
    /// Number of measured `forward` calls since process start on this layer.
    /// Drives the periodic D2H dump cadence.
    steps: u64,
}

/// Per-layer device scratch for the SPARSE self-spec draft: `keep_idx`
/// (surviving k8-chunk indices, capacity `K_max/8`) + `keep_len` (1 u32).
struct SparseDraftScratch {
    /// Surviving k8-chunk index list, sized for the largest K this layer
    /// might sparsify (down input K=intermediate). `keep_idx.len == K/8`.
    keep_idx: DevicePtr,
    /// Single u32: number of surviving chunks written by
    /// `ffn_build_keep_chunks`.
    keep_len: DevicePtr,
    /// Capacity in k8 chunks (= K_max/8) the `keep_idx` buffer can hold.
    cap_chunks: usize,
}

impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        let e2m1_checkpoint_scales = load_e2m1_checkpoint_scales(&weights, gpu)?;

        Ok(Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            exact_ffn_kernels: ops::W4a16ExactFfnKernels::new(
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m4"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m8"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m17"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m32"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m4"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m8"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m17"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m32"),
            )
            .with_materialized_m8(
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_silu_f32_exact_m8"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_f32_input_exact_m8"),
            )
            .with_materialized_m17(
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_dual_silu_f32_exact_m17",
                ),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_f32_input_exact_m17"),
            )
            .with_fused_materialized_m17(super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_dual_exact_materialize_f32_m17",
            ))
            .with_rt2(
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_dual_exact_materialize_f32_rt2_m17",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_f32_input_exact_rt2_m8",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_f32_input_exact_rt2_m17",
                ),
            ),
            exact_ffn_lowreg_m16: ops::W4a16ExactFfnLowregM16Kernels::new(
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gemv_gate_exact_m16_lowreg",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gemv_up_exact_m16_lowreg",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gate_up_materialize_f32_m16",
                ),
            ),
            w4a16_gemv_sw: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_dual_sw: super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
            w4a16_gemv_silu_input_sw: super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            gemv_sw: ops::gemv_sw_enabled(),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_dual_batch3_tuned: super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_dual_batch3_tuned",
            ),
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_pipe: super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            w4a16_gemm_pipe_silu_mul: super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_silu_mul"),
            w4a16_gemm_pipe_dual: super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_dual"),
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
            // Optional small-M (M_TILE=16) transposed-weight GEMM.
            // Missing on older kernel caches — handle 0 disables routing
            // through this kernel regardless of the transposed weights.
            w4a16_gemm_t_m16: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            // Optional small-M (M_TILE=16), small-N (N_TILE=64) variant for
            // K=3 MTP verify on dense Qwen3.6-27B. Missing on non-qwen3.6-27b
            // kernel caches — handle 0 falls back to GEMV silently.
            w4a16_gemm_t_m16_n64: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16_n64"),
            // Optional large-M (M_TILE=128, N_TILE=128) transposed-weight
            // GEMM for prefill. Handle 0 disables the
            // `ATLAS_PREFILL_FFN_FAST` fast path silently.
            w4a16_gemm_t_m128: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m32_n64: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            // Optional fused gate+up+silu kernel. Handle 0 keeps the split
            // gate/up path (ATLAS_FFN_FUSED_GATEUP).
            w4a16_gemm_t_m32_n64_gateup_silu: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu",
            ),
            // Optional dequant-in-registers fork of the fused kernel. Handle
            // 0 keeps the staged fused kernel (ATLAS_DEQUANT_PIPE).
            w4a16_gemm_t_m32_n64_gateup_silu_pipe: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe",
            ),
            // Optional K_STEP=64 register-dequant fork (ATLAS_GATEUP_K64).
            // Handle 0 falls back through _pipe → staged fused kernel.
            w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64",
            ),
            // Optional split-K down_proj variant + reduce. Handle 0 keeps
            // the single-slice m32_n64 path (ATLAS_FFN_DOWN_SPLITK).
            w4a16_gemm_t_m32_n64_splitk: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_splitk",
            ),
            reduce_splitk_k: super::try_kernel(gpu, "w4a16", "reduce_splitk_f32_to_bf16"),
            splitk_workspace: Mutex::new(None),
            // Optional 8-warp shadow of the M=128 kernel. Handle 0
            // disables `ATLAS_FFN_M128_V2` silently.
            w4a16_gemm_t_m128_v2: super::try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2"),
            // Optional FP8×FP8 M=128 GEMM for the predequant fast path.
            // Handle 0 disables `ATLAS_FFN_PREDEQUANT_FP8` silently.
            fp8_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128"),
            gate_fp8: None,
            up_fp8: None,
            down_fp8: None,
            // Optional W4A4 (NVFP4×NVFP4) native tensor-core GEMM. Loaded
            // via `try_kernel` — handle 0 disables the `ATLAS_E2M1_GEMM`
            // fast path silently. The companion absmax + quantize kernels
            // are already loaded by every NVFP4 weight loader; we re-fetch
            // them here so the dispatch site can launch without plumbing
            // them through the forward context.
            nvfp4_gemm_k: super::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_t_m64"),
            nvfp4_gemm_kmajor_m128_k: super::try_kernel(
                gpu,
                "nvfp4_cutlass",
                "nvfp4_nvfp4_gemm_kmajor_m128",
            ),
            nvfp4_gemm_kmajor_m256_k: super::try_kernel(
                gpu,
                "nvfp4_cutlass",
                "nvfp4_nvfp4_gemm_kmajor_m256",
            ),
            nvfp4_absmax_k: super::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
            nvfp4_quantize_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            nvfp4_silu_quantize_k: super::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_silu_mul_bf16_to_nvfp4",
            ),
            e2m1_scratch: Mutex::new(None),
            e2m1_use_stream: Mutex::new(None),
            e2m1_checkpoint_scales,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            flashinfer_ffn_prefill: None,
            // Sparsity-drafted self-speculation kernels (default-off features).
            // The .cu files live in kernels/gb10/common/ and register under
            // their file-stem module names. try_kernel → handle 0 disables the
            // feature silently on caches built before these kernels existed.
            // W3 (3-bit) FFN lane — weights installed later by the loader
            // (set_w3_weights) iff ATLAS_FFN_W3_LAYERS + sidecar match.
            // Kernels live in kernels/gb10/common/w3a16_gemv.cu /
            // w3a16_gemm.cu; try_kernel → handle 0 on caches built before
            // they existed (W3 then stays fully disabled).
            w3_weights: None,
            w3_weights_t: None,
            w3a16_gemv_dual_k: super::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_dual"),
            w3a16_gemv_silu_input_k: super::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_silu_input"),
            w3a16_gemm_t_m32_n64_k: super::try_kernel(gpu, "w3a16_gemm", "w3a16_gemm_t_m32_n64"),
            ffn_sparsity_measure_k: super::try_kernel(
                gpu,
                "ffn_sparsity_measure",
                "ffn_sparsity_measure",
            ),
            sparsity_meas: Mutex::new(None),
            ffn_build_keep_chunks_k: super::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "ffn_build_keep_chunks",
            ),
            w4a16_gemv_sparse_cols_k: super::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "w4a16_gemv_sparse_cols",
            ),
            sparse_draft_scratch: Mutex::new(None),
        })
    }

    /// Whether the W4A4 (NVFP4×NVFP4) native tensor-core FFN prefill path
    /// is wired up. Returns true when all three required kernel symbols
    /// are present in the loaded module. Used by `forward_prefill` to
    /// choose between the W4A4 fast path and the existing fp8/v2/m128
    /// fallbacks.
    pub fn has_e2m1_ffn(&self) -> bool {
        self.nvfp4_gemm_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
            && (self.e2m1_checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0)
    }

    /// Whether the K-major W4A4 implementation has every runtime component
    /// other than the per-layer transformed weights (installed after `new`).
    fn has_e2m1_kmajor_runtime(&self) -> bool {
        self.nvfp4_gemm_kmajor_m128_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
            && (self.e2m1_checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0)
    }

    fn has_e2m1_kmajor_m256_runtime(&self) -> bool {
        self.nvfp4_gemm_kmajor_m256_k.0 != 0 && self.has_e2m1_kmajor_runtime()
    }

    /// Ensure the W4A4 activation scratch arena has capacity for `m` rows
    /// and `k` columns. Reallocates in-place if either dimension grew.
    ///
    /// Returns `(a_packed, a_scale, a_max)`.
    fn ensure_e2m1_scratch(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let layout = e2m1_scratch_layout(m, k)?;
        let mut slot = self.e2m1_scratch.lock().unwrap();
        let needs_realloc = match slot.as_ref() {
            Some(s) => s.cap_m < layout.cap_m || s.cap_k < layout.cap_k,
            None => true,
        };
        if needs_realloc {
            // Free previous (if any) so we don't leak when M or K grows.
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.a_packed);
                let _ = gpu.free(prev.a_scale);
                let _ = gpu.free(prev.a_max);
            }
            let a_packed = gpu.alloc(layout.packed_bytes)?;
            let a_scale = gpu.alloc(layout.scale_bytes)?;
            let a_max = gpu.alloc(layout.max_bytes)?;
            *slot = Some(E2m1Scratch {
                a_packed,
                a_scale,
                a_max,
                cap_m: layout.cap_m,
                cap_k: layout.cap_k,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.a_packed, s.a_scale, s.a_max))
    }

    fn lock_e2m1_use(&self, stream: u64) -> Result<MutexGuard<'_, Option<u64>>> {
        let mut admitted_stream = self
            .e2m1_use_stream
            .lock()
            .map_err(|_| anyhow::anyhow!("W4A4 activation scratch stream-binding lock poisoned"))?;
        match *admitted_stream {
            Some(existing) => ensure!(
                existing == stream,
                "W4A4 activation scratch is bound to CUDA stream {existing:#x}; rejected concurrent/reordered use on {stream:#x}"
            ),
            None => *admitted_stream = Some(stream),
        }
        Ok(admitted_stream)
    }

    /// Quantize one BF16 activation matrix for the W4A4 path.
    ///
    /// The returned packed values and scales remain owned by this layer's
    /// scratch arena. Callers may issue multiple same-stream GEMMs from them
    /// before preparing a different input. Gate and up share the same BF16
    /// input, so preparing once avoids a duplicate absmax scan, host readback,
    /// stream synchronization, and quantization launch without changing any
    /// quantized byte or GEMM arithmetic.
    fn prepare_e2m1_input(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, f32)> {
        let (a_packed, a_scale, a_max) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;

        let a_scale2 = if let Some(scale2) = checkpoint_scale2 {
            // ModelOpt calibrated this scalar offline and stored it as
            // `.input_scale`. It was read once during layer construction, so
            // the unchanged on-device quantizer can launch immediately.
            scale2
        } else {
            // Dynamic fallback: scan BF16 [M,K], drain the stream, and derive
            // a per-request scale. This remains the default W4A4 contract.
            ctx.gpu.memset_async(a_max, 0, 4, stream)?;
            ops::nvfp4_global_absmax(ctx.gpu, self.nvfp4_absmax_k, input, a_max, m * k, stream)?;
            ctx.gpu.synchronize(stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h(a_max, &mut bytes)?;
            let global_max = f32::from_le_bytes(bytes);
            if global_max > 0.0 {
                global_max / (6.0 * 448.0)
            } else {
                1.0
            }
        };

        // Phase 3: per-row E2M1 quantization of the activation. Writes
        // `a_packed` [M, K/2] + `a_scale` [M, K/16] in place.
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            input,
            a_packed,
            a_scale,
            a_scale2,
            m,
            k,
            stream,
        )?;

        Ok((a_packed, a_scale, a_scale2))
    }

    /// Materialize the SwiGLU down input directly in NVFP4 scratch.
    ///
    /// The CUDA kernel rounds each SiLU(gate)*up value through BF16 before
    /// deriving group scales and nibbles, matching the standalone activation
    /// kernel's numerical boundary without writing the temporary BF16 matrix.
    fn prepare_e2m1_silu_input(
        &self,
        ctx: &ForwardContext,
        gate: DevicePtr,
        up: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: f32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, f32)> {
        let (a_packed, a_scale, _) = self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;
        ops::quantize_silu_mul_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_silu_quantize_k,
            gate,
            up,
            a_packed,
            a_scale,
            checkpoint_scale2,
            m,
            k,
            stream,
        )?;
        Ok((a_packed, a_scale, checkpoint_scale2))
    }

    /// Dispatch one native W4A4 GEMM from an already prepared activation.
    ///
    /// `weight` is either the standard `[N, K/2]` row-major NVFP4 weight or,
    /// when `kmajor_m128` is true, the existing `[K/2,N]` / `[K/16,N]`
    /// transformed pair. Both kernels retain the same native block-scaled
    /// OMMA and BF16 epilogue order.
    #[allow(clippy::too_many_arguments)]
    fn forward_e2m1_prepared(
        &self,
        ctx: &ForwardContext,
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        a_scale2: f32,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        kmajor_m128: bool,
        stream: u64,
    ) -> Result<()> {
        let scale2_ab = a_scale2 * weight.weight_scale_2;
        if kmajor_m128 {
            if e2m1_kmajor_kernel(m, crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled())
                == E2m1KmajorKernel::M256
            {
                ops::nvfp4_nvfp4_gemm_kmajor_m256(
                    ctx.gpu,
                    self.nvfp4_gemm_kmajor_m256_k,
                    a_packed,
                    a_scale,
                    weight.weight,
                    weight.weight_scale,
                    scale2_ab,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )?;
            } else {
                ops::nvfp4_nvfp4_gemm_kmajor_m128(
                    ctx.gpu,
                    self.nvfp4_gemm_kmajor_m128_k,
                    a_packed,
                    a_scale,
                    weight.weight,
                    weight.weight_scale,
                    scale2_ab,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )?;
            }
        } else {
            ops::nvfp4_nvfp4_gemm(
                ctx.gpu,
                self.nvfp4_gemm_k,
                a_packed,
                a_scale,
                weight.weight,
                weight.weight_scale,
                scale2_ab,
                output,
                m,
                n,
                k,
                stream,
            )?;
        }
        Ok(())
    }

    /// Run the W4A4 fast path for one FFN projection: prequant BF16 input
    /// to NVFP4, then dispatch `nvfp4_nvfp4_gemm`.
    #[allow(clippy::too_many_arguments)]
    fn forward_e2m1_proj(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        kmajor_m128: bool,
        stream: u64,
    ) -> Result<()> {
        let (a_packed, a_scale, a_scale2) =
            self.prepare_e2m1_input(ctx, input, m, k, checkpoint_scale2, stream)?;
        self.forward_e2m1_prepared(
            ctx,
            a_packed,
            a_scale,
            a_scale2,
            weight,
            output,
            m,
            n,
            k,
            kmajor_m128,
            stream,
        )
    }

    /// Install transposed (`nvfp4_t` layout) FFN projection weights for
    /// the `forward_kgamma` M_TILE=16 fast path. Called by the loader
    /// after `DenseFfnLayer::new` when `ATLAS_FFN_M16_TRANSPOSED=1`.
    /// Takes ownership of `gate_proj_t` / `up_proj_t` / `down_proj_t`
    /// (additional allocations alongside the standard `weights.*_proj`;
    /// the originals are kept for decode-side GEMV paths that target the
    /// HuggingFace `[N, K/2]` layout).
    pub fn set_transposed_weights(
        &mut self,
        gate_proj_t: QuantizedWeight,
        up_proj_t: QuantizedWeight,
        down_proj_t: QuantizedWeight,
    ) {
        self.gate_proj_t = Some(gate_proj_t);
        self.up_proj_t = Some(up_proj_t);
        self.down_proj_t = Some(down_proj_t);
    }

    /// Build the exact, default-off FlashInfer FFN prefill operands for one
    /// Qwen3.8 layer. Gate and up are copied once into an immutable merged
    /// gate-then-up operand; down retains the original checkpoint allocation.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn prepare_flashinfer_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        layer: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, Qwen38FfnProjection,
            qwen38_ffn_launch_candidate, qwen38_merged_gate_up_retained_bytes,
        };
        use ops::flashinfer_sm121::{FlashInferSm121, FlashInferSm121Shape};

        ensure!(
            crate::layers::prefill_ffn_flashinfer_enabled()?,
            "prepare_flashinfer_prefill called while its route is disabled"
        );
        ensure!(
            self.flashinfer_ffn_prefill.is_none(),
            "FlashInfer FFN operands are already prepared"
        );
        ensure!(
            hidden == QWEN38_HIDDEN && intermediate == QWEN38_INTERMEDIATE,
            "FlashInfer FFN route requires Qwen3.8 H={QWEN38_HIDDEN} I={QWEN38_INTERMEDIATE}; got H={hidden} I={intermediate}"
        );
        ensure!(
            self.activation == FfnActivation::SiLU,
            "FlashInfer FFN route requires a SiLU-gated dense FFN"
        );
        ensure!(
            self.e2m1_checkpoint_scales.is_some(),
            "FlashInfer FFN route requires admitted checkpoint-static activation scales"
        );

        let library_path = std::env::var_os("ATLAS_FLASHINFER_SM121_LIB").ok_or_else(|| {
            anyhow::anyhow!("ATLAS_PREFILL_FFN_FLASHINFER=1 requires ATLAS_FLASHINFER_SM121_LIB")
        })?;
        let library = FlashInferSm121::open_with_sha256(
            std::path::Path::new(&library_path),
            QUALIFIED_FLASHINFER_SM121_SHA256,
        )?;
        let quantize_atlas_128x4_k = gpu.kernel(
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        )?;
        let quantize_merged_silu_atlas_128x4_k = gpu.kernel(
            "quantize_nvfp4",
            "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
        )?;
        ensure!(
            quantize_atlas_128x4_k.0 != 0 && quantize_merged_silu_atlas_128x4_k.0 != 0,
            "FlashInfer FFN preparation resolved a NULL required kernel"
        );

        // Freeze the exact zero-workspace tactics before retaining any route.
        // Querying at construction catches a mismatched native library rather
        // than discovering it after some layer projections have run.
        let stream = gpu.default_stream();
        for (operation, m) in [
            (Qwen38FfnOperation::MergedGateUp, 2_079),
            (Qwen38FfnOperation::Down, 2_079),
            (Qwen38FfnOperation::MergedGateUp, 8_192),
            (Qwen38FfnOperation::Down, 8_192),
        ] {
            let plan = qwen38_ffn_launch_candidate(operation, m)?;
            ensure!(
                plan.performance_qualified && plan.workspace_bytes == 0,
                "FlashInfer FFN launch is not fully qualified: {operation:?} M={m}"
            );
            let shape = FlashInferSm121Shape::new(
                u8::try_from(plan.tactic).context("FlashInfer tactic exceeds u8")?,
                plan.m,
                plan.n,
                plan.k,
                1,
            )?;
            let prepared = library.prepare_borrowed_zero_workspace(gpu, shape, stream)?;
            ensure!(
                prepared.shape() == shape,
                "FlashInfer FFN preparation changed its frozen shape"
            );
        }

        // This exact allocation is already charged by the server's model-load
        // preflight. Materialize it now so the first measured M=2079 request
        // cannot pay 64 lazy allocations (or discover an OOM after serving has
        // started). The runtime route reuses this fixed-address maximum shape.
        let scratch_layout = e2m1_scratch_layout(8_192, intermediate)?;
        let (scratch_packed, scratch_scales, scratch_max) =
            self.ensure_e2m1_scratch(gpu, scratch_layout.cap_m, scratch_layout.cap_k)?;
        for pointer in [scratch_packed, scratch_scales, scratch_max] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN construction scratch is NULL or misaligned"
            );
        }
        {
            let retained = self.e2m1_scratch.lock().unwrap();
            let retained = retained
                .as_ref()
                .context("FlashInfer FFN construction scratch was not retained")?;
            ensure!(
                retained.cap_m == scratch_layout.cap_m && retained.cap_k == scratch_layout.cap_k,
                "FlashInfer FFN construction scratch retained the wrong capacity"
            );
        }

        let merged_gate_up = build_flashinfer_merged_gate_up(
            gpu,
            layer,
            &self.weights.gate_proj,
            &self.weights.up_proj,
        )?;
        let down = build_flashinfer_ffn_projection(
            gpu,
            layer,
            Qwen38FfnProjection::Down,
            &self.weights.down_proj,
            hidden,
            intermediate,
        )?;

        let (library_device, library_inode) = library.file_identity();
        tracing::info!(
            layer,
            library = %library.path().display(),
            library_sha256 = %library.sha256_hex(),
            library_device,
            library_inode,
            merged_gate_up_weight = format_args!("{:#x}", merged_gate_up.weight.0),
            down_weight = format_args!("{:#x}", down.weight.0),
            merged_gate_up_weight_hash = format_args!("{:016x}", merged_gate_up.weight_hash),
            merged_gate_up_scale_hash = format_args!("{:016x}", merged_gate_up.weight_scales_hash),
            down_scale_hash = format_args!("{:016x}", down.weight_scales_hash),
            merged_gate_up_alpha_bits = format_args!("{:08x}", merged_gate_up.alpha_bits),
            down_alpha_bits = format_args!("{:08x}", down.alpha_bits),
            merged_gate_up_retained_bytes = qwen38_merged_gate_up_retained_bytes(),
            scratch_cap_m = scratch_layout.cap_m,
            scratch_cap_k = scratch_layout.cap_k,
            scratch_packed_bytes = scratch_layout.packed_bytes,
            scratch_scale_bytes = scratch_layout.scale_bytes,
            "admitted FlashInfer SM121 FFN prefill operands"
        );
        self.flashinfer_ffn_prefill = Some(FlashinferFfnPrefill {
            layer,
            library,
            merged_gate_up,
            down,
            quantize_atlas_128x4_k,
            quantize_merged_silu_atlas_128x4_k,
        });
        Ok(())
    }

    /// Try the fully-admitted FlashInfer FFN route for an exact qualified
    /// chunk size. Other rows (notably a final short prefill tail) retain the
    /// existing Atlas path unchanged.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn try_forward_flashinfer_prefill(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, select_qwen38_ffn_launch,
        };
        use ops::flashinfer_sm121::{FlashInferSm121Buffers, FlashInferSm121Shape};

        let requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
        let exact_qwen38_dense = ctx.config.model_type == "qwen3_5"
            && ctx.config.num_experts == 0
            && h as usize == QWEN38_HIDDEN
            && inter as usize == QWEN38_INTERMEDIATE
            && ctx.config.tp_world_size.max(1) == 1;
        let m_qualified = matches!(m, 2_079 | 8_192)
            || crate::weight_map::flashinfer_ffn_admission::qwen38_ffn_launch_candidate(
                Qwen38FfnOperation::MergedGateUp,
                m as usize,
            )
            .is_ok();
        match flashinfer_ffn_prefill_route(
            requested,
            exact_qwen38_dense,
            m_qualified,
            self.flashinfer_ffn_prefill.is_some(),
        ) {
            FlashinferFfnPrefillRoute::Disabled | FlashinferFfnPrefillRoute::Ineligible => {
                return Ok(false);
            }
            FlashinferFfnPrefillRoute::Missing => anyhow::bail!(
                "ATLAS_PREFILL_FFN_FLASHINFER=1 requires prepared merged FFN operands for exact M={m}"
            ),
            FlashinferFfnPrefillRoute::Complete => {}
        }
        let route = self.flashinfer_ffn_prefill.as_ref().unwrap();
        ensure!(
            route.merged_gate_up.layer == route.layer
                && route.merged_gate_up.source_gate_weight == self.weights.gate_proj.weight
                && route.merged_gate_up.source_up_weight == self.weights.up_proj.weight
                && route.down.weight == self.weights.down_proj.weight,
            "FlashInfer FFN checkpoint operand identity changed after preparation"
        );

        let mut merged_plan = select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, m as usize)?;
        let mut down_plan = select_qwen38_ffn_launch(Qwen38FfnOperation::Down, m as usize)?;
        if let Some((gu, dn)) = crate::layers::flashinfer_ffn_tactic_override() {
            static SEEN: std::sync::Once = std::sync::Once::new();
            SEEN.call_once(|| tracing::warn!("ATLAS_FLASHINFER_FFN_TACTIC override: merged_gate_up={gu} down={dn} (diagnostic, not exact-qualified)"));
            merged_plan.tactic = gu;
            down_plan.tactic = dn;
        }
        ensure!(
            merged_plan.workspace_bytes == 0 && down_plan.workspace_bytes == 0,
            "FlashInfer FFN runtime selected a nonzero-workspace plan"
        );

        let rows = m as usize;
        let input_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let projection_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 2)?;
        let merged_bytes = flashinfer_ffn_extent(rows, 2 * QWEN38_INTERMEDIATE, 2)?;
        let output_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let activation_fp4_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 1)? / 2;
        let activation_scale_bytes =
            flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)?;
        ensure!(
            ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes,
            "merged FlashInfer FFN output arena is undersized"
        );
        ensure!(
            ctx.buffers.sizes().expert_gate_out >= projection_bytes
                && ctx.buffers.sizes().expert_up_out >= projection_bytes
                && ctx.buffers.sizes().moe_output >= output_bytes,
            "FlashInfer FFN destination arena is undersized"
        );

        let merged_out = ctx.buffers.ffn_gate_up_bf16();
        let output = ctx.buffers.moe_output();
        for pointer in [input, merged_out, gate_out, up_out, output] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN input/output pointer is NULL or misaligned"
            );
        }
        ensure_flashinfer_ffn_disjoint(input, input_bytes, merged_out, merged_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, gate_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, gate_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(gate_out, projection_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(gate_out, projection_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(up_out, projection_bytes, output, output_bytes)?;

        let merged_shape = FlashInferSm121Shape::new(
            u8::try_from(merged_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            merged_plan.m,
            merged_plan.n,
            merged_plan.k,
            1,
        )?;
        let down_shape = FlashInferSm121Shape::new(
            u8::try_from(down_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            down_plan.m,
            down_plan.n,
            down_plan.k,
            1,
        )?;
        let mut merged_launch =
            route
                .library
                .prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)?;
        let mut down_launch = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)?;

        // Allocate once at the larger down-projection K so no asynchronous
        // activation operand can be freed while it is still in flight. All
        // route shapes, symbols, destinations and native plans are already
        // preflighted before the first quantization/device output effect.
        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (activation_fp4, activation_scales, _) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
        for pointer in [activation_fp4, activation_scales] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN activation scratch is NULL or misaligned"
            );
        }
        ensure_flashinfer_ffn_disjoint(
            activation_fp4,
            activation_fp4_bytes,
            activation_scales,
            activation_scale_bytes,
        )?;
        for (pointer, bytes) in [
            (input, input_bytes),
            (merged_out, merged_bytes),
            (gate_out, projection_bytes),
            (up_out, projection_bytes),
            (output, output_bytes),
        ] {
            ensure_flashinfer_ffn_disjoint(activation_fp4, activation_fp4_bytes, pointer, bytes)?;
            ensure_flashinfer_ffn_disjoint(
                activation_scales,
                activation_scale_bytes,
                pointer,
                bytes,
            )?;
        }

        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_atlas_128x4_k,
            input,
            activation_fp4,
            activation_scales,
            f32::from_bits(route.merged_gate_up.input_scale_bits),
            m,
            h,
            stream,
        )?;
        merged_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: merged_out,
                activation_fp4,
                weight_fp4: route.merged_gate_up.weight,
                activation_scales,
                weight_scales: route.merged_gate_up.weight_scales_128x4,
                global_scale_f32: route.merged_gate_up.alpha_f32,
            },
            stream,
        )?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_merged_silu_atlas_128x4_k,
            merged_out,
            activation_fp4,
            activation_scales,
            f32::from_bits(route.down.input_scale_bits),
            m,
            inter,
            stream,
        )?;

        down_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: output,
                activation_fp4,
                weight_fp4: route.down.weight,
                activation_scales,
                weight_scales: route.down.weight_scales_128x4,
                global_scale_f32: route.down.alpha_f32,
            },
            stream,
        )?;

        log_flashinfer_ffn_success(route.layer, Qwen38FfnOperation::MergedGateUp, merged_plan.m);
        log_flashinfer_ffn_success(route.layer, Qwen38FfnOperation::Down, down_plan.m);
        Ok(true)
    }

    /// Install W3 (3-bit) FFN weights for this layer — GEMV-layout copies
    /// (used by the single-token `forward` SiLU path) and transposed
    /// GEMM-layout copies (used by `forward_kgamma`). Called by the loader
    /// when `ATLAS_FFN_W3_LAYERS` names this layer and the sidecar tensors
    /// loaded cleanly (see `weight_map::w3_sidecar`). The original W4
    /// weights are RETAINED for the paths W3 does not cover (prefill,
    /// K=2/3 batched GEMV, GELU models, sparse draft).
    pub fn set_w3_weights(&mut self, gemv: DenseFfnWeights, gemm_t: DenseFfnWeights) {
        self.w3_weights = Some(gemv);
        self.w3_weights_t = Some(gemm_t);
    }

    /// Whether the W3 single-token GEMV path is fully wired: weights
    /// installed + both kernel symbols present + SiLU activation (the W3
    /// GEMV set has no GELU-fused down kernel; GELU models stay on W4).
    fn has_w3_gemv(&self) -> bool {
        self.w3_weights.is_some()
            && self.w3a16_gemv_dual_k.0 != 0
            && self.w3a16_gemv_silu_input_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether the W3 K=γ verify GEMM path is fully wired.
    fn has_w3_gemm(&self) -> bool {
        self.w3_weights_t.is_some()
            && self.w3a16_gemm_t_m32_n64_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether ANY W3 routing is active on this layer (loader log helper).
    pub fn has_w3(&self) -> bool {
        self.has_w3_gemv() || self.has_w3_gemm()
    }

    /// Eagerly allocate the FP32 split-K workspace for the down_proj
    /// (`[k_splits, 32, n]` where n = hidden, M padded to the M_TILE=32
    /// of the split-K kernel). Called at load time (pre-graph-capture)
    /// because `gpu.alloc` is illegal during CUDA graph capture. No-op
    /// when split-K is disabled or the kernel symbols are missing.
    pub fn alloc_splitk_workspace(&self, gpu: &dyn GpuBackend, n: u32) -> Result<()> {
        // `n` is the largest split-K output dim this layer might use. The
        // down_proj path needs N=hidden; the gate/up path (when
        // `ATLAS_FFN_GATEUP_SPLITK` is on) needs N=intermediate. Callers pass
        // `max(hidden, intermediate)` so ONE FP32 workspace serves both.
        // gate and up run back-to-back on the same stream and each fully
        // reduces into its own output before the next partial phase, so they
        // can safely share this scratch.
        let down_splits = crate::layers::ffn_down_splitk();
        let gateup_splits = crate::layers::ffn_gateup_splitk();
        if (down_splits == 0 && gateup_splits == 0)
            || self.w4a16_gemm_t_m32_n64_splitk.0 == 0
            || self.reduce_splitk_k.0 == 0
        {
            return Ok(());
        }
        let mut slot = self.splitk_workspace.lock().unwrap();
        if slot.is_none() {
            // 8 = max split clamp; 32 = M_TILE of the split-K kernel.
            let bytes = 8usize * 32 * n as usize * 4;
            *slot = Some(gpu.alloc(bytes)?);
        }
        Ok(())
    }

    /// Whether the M_TILE=16 transposed-weight path is wired up.
    /// Used by `forward_kgamma` for dispatch and by the loader log.
    pub fn has_transposed_ffn(&self) -> bool {
        self.gate_proj_t.is_some()
            && self.up_proj_t.is_some()
            && self.down_proj_t.is_some()
            && self.w4a16_gemm_t_m16.0 != 0
    }

    /// Whether the predequanted-FP8 FFN prefill path is wired up.
    /// Returns true when all three projections have FP8 buffers AND the
    /// `fp8_gemm_t_m128` kernel symbol is present. Used by
    /// `forward_prefill` to choose between the NVFP4 + dequant path and
    /// the predequant fast path.
    pub fn has_fp8_ffn(&self) -> bool {
        self.gate_fp8.is_some()
            && self.up_fp8.is_some()
            && self.down_fp8.is_some()
            && self.fp8_gemm_t_m128_k.0 != 0
    }

    /// Pre-dequant NVFP4 FFN weights to FP8 [N, K] for the predequant
    /// fast prefill path. Mirrors
    /// `Qwen3AttentionLayer::predequant_for_prefill` — uses the
    /// NON-transposed weights in `self.weights.gate_proj` etc. (the
    /// `predequant_nvfp4_to_fp8` kernel reads the original [N, K/2]
    /// NVFP4 layout). Allocates 3 × N×K bytes of GPU memory per layer.
    ///
    /// Called by the loader after `DenseFfnLayer::new` when the
    /// `ATLAS_FFN_PREDEQUANT_FP8` env var is set. Silently no-ops
    /// when the kernel symbol is missing.
    ///
    /// `inter` = intermediate_size (gate/up output dim; down input dim).
    /// `hidden` = hidden_size (gate/up input dim; down output dim).
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        hidden: usize,
        inter: usize,
        stream: u64,
    ) -> Result<()> {
        if self.fp8_gemm_t_m128_k.0 == 0 {
            return Ok(()); // FP8 kernel not available — silently skip
        }
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        // gate_proj: [inter, hidden] NVFP4 → [inter, hidden] FP8
        self.gate_fp8 = Some(self.weights.gate_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // up_proj: same shape as gate
        self.up_fp8 = Some(self.weights.up_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // down_proj: [hidden, inter]
        self.down_fp8 = Some(self.weights.down_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            hidden,
            inter,
            stream,
        )?);
        Ok(())
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Whether the FFN activation-sparsity MEASUREMENT harness is wired up:
    /// the env gate is on AND the measure kernel symbol is present.
    fn sparsity_measure_active(&self) -> bool {
        crate::layers::measure_ffn_sparsity_enabled() && self.ffn_sparsity_measure_k.0 != 0
    }

    /// Ensure the per-layer sparsity-measurement counter buffers exist and are
    /// zeroed on first allocation. Returns the four device pointers. Allocated
    /// lazily on the first measured `forward` (never during graph capture —
    /// the measured path runs eager).
    fn ensure_sparsity_meas(
        &self,
        gpu: &dyn GpuBackend,
        inter: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let mut slot = self.sparsity_meas.lock().unwrap();
        if slot.is_none() {
            let hist_gateup = gpu.alloc(n_thresh * 4)?;
            let count_gateup = gpu.alloc(2 * 4)?;
            let hist_down = gpu.alloc(n_thresh * 4)?;
            let count_down = gpu.alloc(2 * 4)?;
            let meas_silu = gpu.alloc(inter * 2)?; // BF16 [1, intermediate]
            // Zero the accumulators up front (kernel uses atomicAdd).
            gpu.memset(hist_gateup, 0, n_thresh * 4)?;
            gpu.memset(count_gateup, 0, 2 * 4)?;
            gpu.memset(hist_down, 0, n_thresh * 4)?;
            gpu.memset(count_down, 0, 2 * 4)?;
            *slot = Some(SparsityMeas {
                hist_gateup,
                count_gateup,
                hist_down,
                count_down,
                meas_silu,
                steps: 0,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((
            s.hist_gateup,
            s.count_gateup,
            s.hist_down,
            s.count_down,
            s.meas_silu,
        ))
    }

    /// Observer: launch `ffn_sparsity_measure` on `input` at the given site.
    /// PURE READER — never mutates `input` or any token-stream buffer; writes
    /// only into the dedicated `hist`/`count` accumulators. Called from
    /// `forward` at the two FFN sites when the measurement gate is on.
    fn measure_sparsity_site(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        hist: DevicePtr,
        count: DevicePtr,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::ffn_sparsity_measure(
            ctx.gpu,
            self.ffn_sparsity_measure_k,
            input,
            hist,
            count,
            k,
            stream,
        )
    }

    /// Periodic D2H dump of the accumulated per-site sparsity histograms,
    /// averaged over all rows measured since process start. Emits a
    /// `tracing::info` line every `measure_ffn_sparsity_dump_every` measured
    /// forwards. Bumps the per-layer step counter each call. No-op when the
    /// dump cadence has not been reached.
    ///
    /// The reported fraction for threshold t at a site is
    /// `hist[t] / elements_seen` — the mean below-threshold activation
    /// fraction, i.e. the UPPER BOUND on the column-skip weight-byte savings
    /// for that projection at that threshold. The go/no-go number is the
    /// down-input (K=intermediate) fraction at the 1% threshold.
    fn maybe_dump_sparsity(&self, ctx: &ForwardContext, layer_tag: &str) -> Result<()> {
        let every = crate::layers::measure_ffn_sparsity_dump_every();
        let (hist_gateup, count_gateup, hist_down, count_down, steps) = {
            let mut slot = self.sparsity_meas.lock().unwrap();
            let Some(s) = slot.as_mut() else {
                return Ok(());
            };
            s.steps += 1;
            if !s.steps.is_multiple_of(every) {
                return Ok(());
            }
            (
                s.hist_gateup,
                s.count_gateup,
                s.hist_down,
                s.count_down,
                s.steps,
            )
        };

        // Sync so the accumulators reflect all launched measurements, then
        // read the histograms + counts back to the host.
        ctx.gpu.synchronize(ctx.gpu.default_stream())?;
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let read_hist = |hist: DevicePtr| -> Result<Vec<u32>> {
            let mut bytes = vec![0u8; n_thresh * 4];
            ctx.gpu.copy_d2h(hist, &mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let read_count = |count: DevicePtr| -> Result<(u64, u64)> {
            let mut bytes = vec![0u8; 2 * 4];
            ctx.gpu.copy_d2h(count, &mut bytes)?;
            let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
            let elems = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
            Ok((rows, elems))
        };

        let fmt_site = |hist: &[u32], elems: u64| -> String {
            if elems == 0 {
                return "n/a".to_string();
            }
            ops::SPARSITY_TAU
                .iter()
                .zip(hist.iter())
                .map(|(tau, &cnt)| {
                    let frac = cnt as f64 / elems as f64;
                    format!("{:.1}%tau={:.1}%", tau * 100.0, frac * 100.0)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let hg = read_hist(hist_gateup)?;
        let hd = read_hist(hist_down)?;
        let (rows_g, elems_g) = read_count(count_gateup)?;
        let (rows_d, elems_d) = read_count(count_down)?;

        tracing::info!(
            "FFN_SPARSITY[{layer_tag}] steps={steps} \
             gateup_in(K=hidden rows={rows_g}): {} | \
             down_in(K=inter rows={rows_d}): {}",
            fmt_site(&hg, elems_g),
            fmt_site(&hd, elems_d),
        );
        Ok(())
    }

    /// Whether the column-sparse self-spec DRAFT FFN path is wired up: both
    /// kernel symbols present. The env gate (`ATLAS_SELF_SPEC_SPARSE`) is
    /// checked by the caller (`step_self_spec`) so this only reports capability.
    pub fn has_sparse_draft(&self) -> bool {
        self.ffn_build_keep_chunks_k.0 != 0 && self.w4a16_gemv_sparse_cols_k.0 != 0
    }

    /// Ensure the per-layer sparse-draft scratch (`keep_idx` + `keep_len`)
    /// exists with capacity for `k` (the largest K this layer will sparsify —
    /// the down input K=intermediate). Allocated lazily on the first sparse
    /// draft forward (eager path, no graph capture).
    fn ensure_sparse_draft_scratch(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let need_chunks = k / 8;
        let mut slot = self.sparse_draft_scratch.lock().unwrap();
        let realloc = match slot.as_ref() {
            Some(s) => s.cap_chunks < need_chunks,
            None => true,
        };
        if realloc {
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.keep_idx);
                let _ = gpu.free(prev.keep_len);
            }
            let keep_idx = gpu.alloc(need_chunks * 4)?; // u32 per chunk
            let keep_len = gpu.alloc(4)?; // single u32
            *slot = Some(SparseDraftScratch {
                keep_idx,
                keep_len,
                cap_chunks: need_chunks,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.keep_idx, s.keep_len))
    }

    /// SPARSE self-spec DRAFT single-token FFN forward.
    ///
    /// Same gate/up GEMV shape as `forward` (gate/up input is the dense
    /// residual stream — low activation sparsity, per the TEAL analysis, so
    /// it stays a dense dual GEMV), then swaps the down_proj GEMV for the
    /// column-sparse path: `ffn_build_keep_chunks` thresholds the silu(gate)*up
    /// activation into a surviving-chunk list, then `w4a16_gemv_sparse_cols`
    /// reads only those weight columns. APPROXIMATE by design — the dense
    /// verify is the lossless oracle, so this only proposes.
    ///
    /// `thresh_frac` is the keep threshold as a fraction of per-row max-abs
    /// (e.g. 0.01 for 1%). Falls back to the exact `forward` dense path when
    /// the sparse kernels are missing (`has_sparse_draft` false), so callers
    /// can always invoke it safely.
    ///
    /// EAGER only (the self-spec draft never captures a CUDA graph): the
    /// `keep_len` scalar is read back D2H before the sparse GEMV launch.
    pub fn forward_draft_sparse(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        thresh_frac: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Capability + BF16-weight guard: the sparse kernels operate on the
        // NVFP4 `QuantizedWeight` layout only. Fall back to the exact dense
        // forward when sparse kernels are missing or BF16 weights are active.
        if !self.has_sparse_draft() || self.bf16_weights.is_some() {
            return self.forward(input, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // gate/up stay DENSE (residual-stream input, low sparsity).
        ops::w4a16_decode_gemv_dual(
            ctx.gpu,
            self.w4a16_gemv_dual,
            self.w4a16_gemv_dual_sw,
            self.gemv_sw,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;

        // silu(gate)*up → gate_out (the down-proj input we sparsify).
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            inter,
            stream,
        )?;

        // Threshold the down-input into a surviving k8-chunk list.
        let (keep_idx, keep_len) = self.ensure_sparse_draft_scratch(ctx.gpu, inter as usize)?;
        ops::ffn_build_keep_chunks(
            ctx.gpu,
            self.ffn_build_keep_chunks_k,
            gate_out,
            thresh_frac,
            keep_idx,
            keep_len,
            inter,
            stream,
        )?;

        // Read back keep_len (scalar-by-value kernel arg). Sync is acceptable
        // on the eager draft path; it also bounds the sparse GEMV's loop.
        ctx.gpu.synchronize(stream)?;
        let mut kl_bytes = [0u8; 4];
        ctx.gpu.copy_d2h(keep_len, &mut kl_bytes)?;
        let keep_len_val = u32::from_le_bytes(kl_bytes);

        // Column-sparse down_proj GEMV over the surviving chunks only.
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_sparse_cols(
            ctx.gpu,
            self.w4a16_gemv_sparse_cols_k,
            gate_out,
            &self.weights.down_proj,
            keep_idx,
            keep_len_val,
            output,
            h,
            inter,
            stream,
        )?;
        Ok(output)
    }

    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_m1", {
            ops::w4a16_decode_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                self.w4a16_gemv_dual_sw,
                self.gemv_sw,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── FFN activation-sparsity MEASUREMENT (observer, default-off) ──
        // Runs ONLY when ATLAS_MEASURE_FFN_SPARSITY=1 and the kernel symbol is
        // present. Pure reader: measures `input` (gate/up in, K=hidden) and a
        // freshly-recomputed `silu(gate)*up` copy (down in, K=inter) into
        // dedicated counter buffers. It never touches `input`, `gate_out`,
        // `up_out`, or `output`, so the token stream stays byte-identical
        // whether the gate is on or off (counting-md5 constitution preserved).
        //
        // Skipped under CUDA graph capture: the observer lazily `gpu.alloc`s its
        // counter buffers on first use, which is illegal mid-capture. The
        // measurement run is intended for eager decode (the operator sets the
        // gate for a measurement window); skipping graphed steps only omits a
        // subset of rows from the average and never perturbs the token stream.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, meas_silu) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            // Site 0: gate/up input (residual stream, K=hidden).
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            // Site 1: down input = silu(gate)*up (K=intermediate). Recompute
            // into the observer's OWN scratch so gate_out/up_out (read by the
            // fused down GEMV below) are untouched.
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                meas_silu,
                inter,
                stream,
            )?;
            self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn")?;
        }

        let output = ctx.buffers.moe_output();
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                crate::kprof!(ctx.gpu, stream, "ffn_down_silu_m1", {
                    ops::w4a16_decode_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        self.w4a16_gemv_silu_input_sw,
                        self.gemv_sw,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_m1", {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
                crate::kprof!(ctx.gpu, stream, "ffn_down_m1", {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        self.w4a16_gemv_sw,
                        self.gemv_sw,
                        gate_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
        }

        Ok(output)
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        // Tensor-core M=3 fast path (ATLAS_TC_NVFP4_K3=1):
        //
        // Routes through `w4a16_gemm_t_m16_n64` (small-M, N_TILE=64) for
        // gate / up / down. Dispatches 3 GEMM launches (gate, up, down)
        // instead of the GEMV path's 3 launches (dual gate+up, silu, down)
        // but each GEMM runs on tensor cores via m16n8k32 e4m3 MMA.
        //
        // The first attempt (routing to `forward_kgamma(n=3)` which used
        // `w4a16_gemm_t_m16` with N_TILE=128) measured −23% mean tok/s
        // on AEON-27B because the 136-CTA grid starved GB10's 110 SMs.
        // The N_TILE=64 variant fields 272 CTAs/projection (~2.5 CTAs/SM)
        // at half the per-CTA work — designed to keep the tensor-core
        // pipeline fed at M=3.
        //
        // Bounds: requires transposed weights + the n64 kernel symbol
        // loaded; falls through to the GEMV path otherwise.
        if crate::layers::tc_nvfp4_k3_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m16_n64.0 != 0
        {
            let h = ctx.config.hidden_size as u32;
            let inter = ctx.config.intermediate_size as u32;
            let gate_out_buf = ctx.buffers.expert_gate_out();
            let up_out_buf = ctx.buffers.expert_up_out();
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    gt,
                    gate_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    ut,
                    up_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out_buf,
                    up_out_buf,
                    gate_out_buf,
                    3 * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    gate_out_buf,
                    dt,
                    output,
                    3,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            return Ok(());
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens. Tuned variant (gated by
        // `ATLAS_FFN_DUAL_TUNED=1`) fuses both projections into the SAME CTA
        // so the 3-token activation vector is loaded once per CTA instead of
        // twice. Falls back to the baseline kernel when the env var is unset
        // OR the tuned kernel symbol was not present in the loaded cache.
        let use_tuned = ffn_dual_tuned_enabled() && self.w4a16_gemv_dual_batch3_tuned.0 != 0;
        let dual_kernel = if use_tuned {
            self.w4a16_gemv_dual_batch3_tuned
        } else {
            self.w4a16_gemv_dual_batch3
        };
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
            if use_tuned {
                ops::w4a16_gemv_dual_batch3_tuned(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv_dual_batch3(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                3 * inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    fn measure_exact_kgamma_inputs(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        if !self.sparsity_measure_active() || ctx.graph_capture {
            return Ok(());
        }

        let (hist_g, count_g, hist_d, count_d, meas_silu) =
            self.ensure_sparsity_meas(ctx.gpu, intermediate as usize)?;
        self.measure_sparsity_site(ctx, input, hist_g, count_g, hidden, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            meas_silu,
            intermediate,
            stream,
        )?;
        self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, intermediate, stream)?;
        self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma_exact")
    }

    fn forward_kgamma_exact_split_m8_down(
        &self,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const GROUP_ROWS: u32 = 8;
        let preactivation = ctx.buffers.ssm_conv_out_f32();
        let output = ctx.buffers.moe_output();

        for first_row in [0, GROUP_ROWS] {
            let offsets = exact_ffn_row_offsets(first_row, hidden, intermediate)
                .ok_or_else(|| anyhow::anyhow!("exact FFN M8 down row-offset overflow"))?;
            ops::w4a16_gemv_f32_input_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                preactivation.offset(offsets.preactivation),
                &self.weights.down_proj,
                output.offset(offsets.output),
                GROUP_ROWS,
                hidden,
                intermediate,
                stream,
            )?;
        }
        Ok(())
    }

    fn forward_kgamma_exact_lowreg_gate_up_m16(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const ROWS: u32 = 16;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let preactivation = ctx.buffers.ssm_conv_out_f32();

        ops::w4a16_gemv_gate_exact_m16_lowreg(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            input,
            &self.weights.gate_proj,
            gate_out,
            ROWS,
            intermediate,
            hidden,
            stream,
        )?;
        ops::w4a16_gemv_up_exact_m16_lowreg(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            input,
            &self.weights.up_proj,
            up_out,
            ROWS,
            intermediate,
            hidden,
            stream,
        )?;
        self.measure_exact_kgamma_inputs(
            input,
            gate_out,
            up_out,
            ctx,
            hidden,
            intermediate,
            stream,
        )?;
        ops::w4a16_gate_up_materialize_f32_m16(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            gate_out,
            up_out,
            preactivation,
            ROWS,
            intermediate,
            stream,
        )?;
        self.forward_kgamma_exact_split_m8_down(ctx, hidden, intermediate, stream)
    }

    fn forward_kgamma_exact_split_m8(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const GROUP_ROWS: u32 = 8;
        if ops::exact_ffn_lowreg_m16_route(
            2 * GROUP_ROWS,
            exact_ffn_lowreg_gate_up_m16_enabled(),
            self.exact_ffn_lowreg_m16,
        ) == Some(ops::ExactFfnLowregM16Route::Lowreg)
        {
            return self.forward_kgamma_exact_lowreg_gate_up_m16(
                input,
                ctx,
                hidden,
                intermediate,
                stream,
            );
        }
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let preactivation = ctx.buffers.ssm_conv_out_f32();
        let output = ctx.buffers.moe_output();

        for first_row in [0, GROUP_ROWS] {
            let offsets = exact_ffn_row_offsets(first_row, hidden, intermediate)
                .ok_or_else(|| anyhow::anyhow!("exact FFN M8 row-offset overflow"))?;
            let input_group = input.offset(offsets.input);
            let gate_group = gate_out.offset(offsets.gate_up);
            let up_group = up_out.offset(offsets.gate_up);
            let preactivation_group = preactivation.offset(offsets.preactivation);
            let output_group = output.offset(offsets.output);

            ops::w4a16_gemv_dual_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                input_group,
                &self.weights.gate_proj,
                gate_group,
                &self.weights.up_proj,
                up_group,
                GROUP_ROWS,
                intermediate,
                hidden,
                stream,
            )?;
            self.measure_exact_kgamma_inputs(
                input_group,
                gate_group,
                up_group,
                ctx,
                hidden,
                intermediate,
                stream,
            )?;
            ops::w4a16_gemv_dual_silu_f32_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                gate_group,
                up_group,
                preactivation_group,
                GROUP_ROWS,
                intermediate,
                stream,
            )?;
            ops::w4a16_gemv_f32_input_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                preactivation_group,
                &self.weights.down_proj,
                output_group,
                GROUP_ROWS,
                hidden,
                intermediate,
                stream,
            )?;
        }
        Ok(())
    }

    fn forward_kgamma_exact(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size as u32;
        let intermediate = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        if exact_ffn_physical_route(
            rows,
            intermediate,
            exact_ffn_split_m8_enabled(),
            self.exact_ffn_kernels
                .tier_is_complete(ops::ExactFfnTier::M8),
            self.exact_ffn_kernels.materialized_m8_is_complete(),
            ctx.buffers.sizes().ssm_conv_out_f32,
        ) == ExactFfnPhysicalRoute::SplitM8x2
        {
            return self.forward_kgamma_exact_split_m8(input, ctx, hidden, intermediate, stream);
        }

        // Attention/SSM has completed before FFN on this stream, so its FP32
        // recurrence workspace is dead until the next layer. Reuse it only
        // when the arena's authoritative byte capacity covers every row.
        let route = exact_ffn_materialized_route(
            rows,
            intermediate,
            ExactFfnMaterializedAvailability {
                m8_dual_silu: self.exact_ffn_kernels.dual_silu_f32_m8_is_present(),
                m8_f32_down: self.exact_ffn_kernels.f32_input_m8_is_present(),
                m17_dual_silu: self.exact_ffn_kernels.dual_silu_f32_m17_is_present(),
                m17_f32_down: self.exact_ffn_kernels.f32_input_m17_is_present(),
                // The fused stage does not expose gate/up tensors required by
                // sparsity diagnostics; retain the split exact route then.
                m17_fused_dual_silu: self.exact_ffn_kernels.fused_materialized_m17_is_present()
                    && (!self.sparsity_measure_active() || ctx.graph_capture),
            },
            ctx.buffers.sizes().ssm_conv_out_f32,
        );

        let preactivation_f32 = ctx.buffers.ssm_conv_out_f32();
        match route {
            ExactFfnMaterializedRoute::FusedM17 => {
                ops::w4a16_gemv_dual_materialize_f32_exact_m17(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    preactivation_f32,
                    rows,
                    intermediate,
                    hidden,
                    stream,
                )?;
            }
            ExactFfnMaterializedRoute::Split | ExactFfnMaterializedRoute::Inline => {
                ops::w4a16_gemv_dual_exact(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    rows,
                    intermediate,
                    hidden,
                    stream,
                )?;
                self.measure_exact_kgamma_inputs(
                    input,
                    gate_out,
                    up_out,
                    ctx,
                    hidden,
                    intermediate,
                    stream,
                )?;
            }
        }

        match route {
            ExactFfnMaterializedRoute::Split => ops::w4a16_gemv_dual_silu_f32_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                gate_out,
                up_out,
                preactivation_f32,
                rows,
                intermediate,
                stream,
            )?,
            ExactFfnMaterializedRoute::Inline => {
                return ops::w4a16_gemv_silu_input_exact(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    gate_out,
                    up_out,
                    &self.weights.down_proj,
                    ctx.buffers.moe_output(),
                    rows,
                    hidden,
                    intermediate,
                    stream,
                );
            }
            ExactFfnMaterializedRoute::FusedM17 => {}
        }
        ops::w4a16_gemv_f32_input_exact(
            ctx.gpu,
            self.exact_ffn_kernels,
            preactivation_f32,
            &self.weights.down_proj,
            ctx.buffers.moe_output(),
            rows,
            hidden,
            intermediate,
            stream,
        )
    }

    fn forward_kgamma_k1_rows(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size as u32;
        let intermediate = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let output = ctx.buffers.moe_output();

        for row in 0..rows as usize {
            let input_row = input.offset(row * hidden as usize * 2);
            let gate_row = gate_out.offset(row * intermediate as usize * 2);
            let up_row = up_out.offset(row * intermediate as usize * 2);
            let output_row = output.offset(row * hidden as usize * 2);
            ops::w4a16_decode_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                self.w4a16_gemv_dual_sw,
                self.gemv_sw,
                input_row,
                &self.weights.gate_proj,
                gate_row,
                &self.weights.up_proj,
                up_row,
                intermediate,
                hidden,
                stream,
            )?;
            ops::w4a16_decode_gemv_silu_input(
                ctx.gpu,
                self.w4a16_gemv_silu_input,
                self.w4a16_gemv_silu_input_sw,
                self.gemv_sw,
                gate_row,
                up_row,
                &self.weights.down_proj,
                output_row,
                hidden,
                intermediate,
                stream,
            )?;
        }

        self.measure_exact_kgamma_inputs(input, gate_out, up_out, ctx, hidden, intermediate, stream)
    }

    /// K=γ verify batch (DFlash γ ≥ 16, typical n=17 with γ=16).
    ///
    /// Replaces the per-token loop that calls `forward()` n times (n=γ+1).
    /// Each `forward()` call runs 2 M=1 GEMVs that re-read 134 MB of NVFP4
    /// FFN weights from LPDDR5X; per-step this costs `64 layers × 17 tokens
    /// × 134 MB = 145 GB` of redundant weight bandwidth.
    ///
    /// This path issues 3 `w4a16_gemm` calls with M=n. The standard
    /// (non-transposed) NVFP4 GEMM has M_TILE=64; M=17 fits inside a single
    /// CTA-row with some accumulator waste but loads the weight tile once
    /// per layer instead of n times. Expected: ~64 × 134 MB = 8.6 GB per
    /// step, an ~18× reduction in FFN-loop bandwidth.
    ///
    /// Sequence: gate_proj (GEMM M=n) → up_proj (GEMM M=n) → silu_mul
    /// (n × intermediate) → down_proj (GEMM M=n).
    /// W4+SiLU rows 2..=32 instead use exact dynamic-M dual and fused-down
    /// GEMVs that preserve K1 accumulation order. If either selected-tier
    /// symbol is missing, every row runs through the ordinary K1 kernels;
    /// BF16, W3, GELU, and rows above 32 retain their existing routes.
    ///
    /// Reuses `ctx.buffers.expert_gate_out` / `expert_up_out` /
    /// `moe_output`, which are sized for `max_batch_tokens × intermediate`
    /// (always ≥ n, see `buffers/sizes.rs`).
    ///
    /// Output is written to `ctx.buffers.moe_output()`; callers downstream
    /// already consume from there (see `ms_phase_ffn` for the n==3 branch
    /// which is the contract this matches at higher n).
    ///
    /// When `ATLAS_FFN_M16_TRANSPOSED=1` and the loader installed transposed
    /// (`nvfp4_t`) FFN weights via `set_transposed_weights`, this path
    /// dispatches gate/up/down through `w4a16_gemm_n128_m16` (M_TILE=16,
    /// near-zero MMA accumulator waste at M ≤ 32). Otherwise the
    /// non-transposed `w4a16_gemm` (M_TILE=64) is used as the fallback —
    /// layout-compatible with the standard HuggingFace `[N, K/2]` weights
    /// but discards ~73% of accumulator writes at M=17. The transposed
    /// fast path is the K=γ verify analogue of the existing SSM `qkvz` /
    /// `out_proj` and DFlash drafter routings.
    /// BF16-weight fallback is not supported on this path (Gemma-4 dense
    /// has no MTP / DFlash, so n is always 1 there).
    pub fn forward_kgamma(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(
            n > 1,
            "forward_kgamma is for batched verify; use forward() at n=1"
        );
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        let exact_w4_silu_eligible = self.exact_kgamma_applicable(n);
        let exact_tier_present = matches!(
            self.exact_ffn_kernels.route_for_rows(n),
            Some(ops::ExactFfnRoute::Exact(_))
        );
        match exact_ffn_dispatch(n, exact_w4_silu_eligible, exact_tier_present) {
            ExactFfnDispatch::Batched => {
                return self.forward_kgamma_exact(input, ctx, n, stream);
            }
            ExactFfnDispatch::PerRowK1 => {
                return self.forward_kgamma_k1_rows(input, ctx, n, stream);
            }
            ExactFfnDispatch::Existing => {}
        }

        if w3_kgamma_applicable(
            n,
            self.activation == FfnActivation::SiLU,
            self.has_w3_gemm(),
        ) {
            let w3 = self
                .w3_weights_t
                .as_ref()
                .context("W3 K-gamma route selected without transposed weights")?;
            crate::kprof!(ctx.gpu, stream, "ffn_gate_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    input,
                    &w3.gate_proj,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_up_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    input,
                    &w3.up_proj,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_w3_kgamma", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    n * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    gate_out,
                    &w3.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "ENGAGED W3 FFN K-gamma: approximate ghost/quality-gated target route"
                );
            });
            return Ok(());
        }

        // Route through M_TILE=16 + transposed weights when:
        //   - ATLAS_FFN_M16_TRANSPOSED=1 OR ATLAS_TC_NVFP4_M16=1
        //   - the loader installed transposed copies of all 3 FFN projections
        //   - the `w4a16_gemm_t_m16` kernel symbol is present (try_kernel)
        //   - n ≤ 32 (the kernel's intended small-M window)
        // The combined gate matches the SSM `qkvz` / drafter pattern.
        // WIDE window (SASS audit 2026-07-08): at c>=2 batched verify,
        // M = 17c exceeds 32 and this whole transposed family used to
        // disengage — silent fallback to legacy `w4a16_gemm` (no cp.async,
        // scalar LDG.U8, ~4x sector overfetch, issue-capped ~47% of DRAM
        // BW): the measured cause of the concurrency ceiling. Route
        // 32 < n <= 256 through `w4a16_gemm_t_m128` (y-tiled; at M=136 two
        // weight reads still beat legacy's 3 sweeps x overfetch). Requires
        // the m128 kernel — the m16 kernel's small-M window is NOT widened.
        let wide_m128 = n > 32
            && n <= 256
            && crate::layers::ffn_kgamma_wide_enabled()
            && crate::layers::ffn_kgamma_m128_enabled()
            && self.w4a16_gemm_t_m128.0 != 0
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn();
        let m16_path = (n <= 32
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn())
            || wide_m128;
        // m128 upgrade of the m16 path: ONE M-tile at n ≤ 128 → single
        // weight read (m16 re-reads B per 16-row tile: 2× traffic at
        // n=17 on a memory-bound GEMM). See ffn_kgamma_m128_enabled.
        //
        // FAIL-CLOSED GUARD (2026-08-19): ATLAS_FFN_KGAMMA_M128=0 selects
        // the M_TILE=16-only route (`w4a16_gemm_t_m16`, 2 M-tile rows at
        // n=17). That route is NON-DETERMINISTIC in this binary — three
        // identical 1,500-token MinHeap requests returned three different
        // completions (513/1500/1500 tokens, three distinct SHAs). The
        // m128/m32 route is deterministic (verified: 3/3 identical,
        // 41.7 tok/s). Fail closed: when the transposed m16 family would
        // otherwise be the only route, force the deterministic m128/m32
        // path regardless of the flag and warn once.
        let m128_forced =
            m16_path && !crate::layers::ffn_kgamma_m128_enabled() && self.w4a16_gemm_t_m128.0 != 0;
        if m128_forced {
            static FORCED: std::sync::Once = std::sync::Once::new();
            FORCED.call_once(|| {
                tracing::warn!(
                    "ATLAS_FFN_KGAMMA_M128=0 requests the M_TILE=16 verify FFN route, \
                     which is NON-DETERMINISTIC in this binary — forcing the \
                     deterministic m128/m32 route instead (fail-closed)."
                );
            });
        }
        let m128_path =
            (m16_path && crate::layers::ffn_kgamma_m128_enabled() && self.w4a16_gemm_t_m128.0 != 0)
                || m128_forced;
        if crate::layers::ffn_kgamma_m128_enabled() && self.w4a16_gemm_t_m128.0 == 0 {
            static WARNED: std::sync::Once = std::sync::Once::new();
            crate::layers::warn_kernel_fallback(
                &WARNED,
                "ATLAS_FFN_KGAMMA_M128=1",
                "w4a16_gemm_t_m128",
                "the K=γ verify FFN stays on the M_TILE=64 path, and every knob \
                 downstream of m128_path (ATLAS_FFN_KGAMMA_M32, \
                 ATLAS_FFN_GATEUP_SPLITK, ATLAS_FFN_FUSED_GATEUP) is disabled with it",
            );
        }
        // m32_n64: single B read AND full SM occupancy — strictly better
        // than m128 at n ≤ 32 when the kernel symbol is present. The n<=32
        // bound keeps the WIDE window off m32/fused/split-K variants (their
        // M_TILE=32 shapes don't cover wide M).
        // ATLAS_FFN_KGAMMA_M32=0 opts out for bisection.
        let m32_path = m128_path
            && n <= 32
            && self.w4a16_gemm_t_m32_n64.0 != 0
            && std::env::var("ATLAS_FFN_KGAMMA_M32").ok().as_deref() != Some("0");

        // gate_proj GEMM: [n, H] → [n, inter]
        // Split-K [M=n, N=inter, K=h] when ATLAS_FFN_GATEUP_SPLITK is set —
        // slices K across gridDim.z into the shared FP32 workspace (lossless,
        // token-exact). Falls through to the single-slice m32_n64 path below.
        let gateup_splitk = crate::layers::ffn_gateup_splitk();
        let gateup_ws = if gateup_splitk > 0 {
            *self.splitk_workspace.lock().unwrap()
        } else {
            None
        };
        let gateup_splitk_ok = m32_path
            && gateup_splitk > 0
            && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
            && self.reduce_splitk_k.0 != 0
            && gateup_ws.is_some();

        // FUSED gate+up+silu (ATLAS_FFN_FUSED_GATEUP=1): one launch reads
        // the shared [n,H] input once, streams both transposed weights, and
        // writes silu(gate)*up into `gate_out` (the same buffer moe_silu_mul
        // targets) — replacing the gate GEMM + up GEMM + silu_mul below.
        // Supersedes gateup split-K. Requires the m32 transposed path + the
        // fused kernel symbol; byte-exact (BF16 activation round-trip matched
        // in-kernel). Falls through to the split path otherwise.
        let fused_gateup = m32_path
            && !gateup_splitk_ok
            && crate::layers::ffn_fused_gateup_enabled()
            && self.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0;
        if crate::layers::ffn_fused_gateup_enabled() && self.w4a16_gemm_t_m32_n64_gateup_silu.0 == 0
        {
            static WARNED: std::sync::Once = std::sync::Once::new();
            crate::layers::warn_kernel_fallback(
                &WARNED,
                "ATLAS_FFN_FUSED_GATEUP=1",
                "w4a16_gemm_t_m32_n64_gateup_silu",
                "gate/up stay on two separate GEMMs plus a silu_mul launch",
            );
        }
        if fused_gateup {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            // Kernel selection priority (highest first):
            //   ATLAS_GATEUP_K64=1  → _pipe_k64 (K_STEP=64, reg-dequant)
            //   ATLAS_DEQUANT_PIPE=1 → _pipe    (K_STEP=32, reg-dequant)
            //   default              → staged fused (smem_B_fp8 staging)
            let fused_kernel = if crate::layers::gateup_k64_enabled()
                && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64.0 != 0
                && h.is_multiple_of(64)
            // K must be divisible by 64
            {
                self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64
            } else if crate::layers::dequant_pipe_enabled()
                && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe.0 != 0
            {
                self.w4a16_gemm_t_m32_n64_gateup_silu_pipe
            } else {
                self.w4a16_gemm_t_m32_n64_gateup_silu
            };
            crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                ops::w4a16_gemm_n64_m32_gateup_silu(
                    ctx.gpu,
                    fused_kernel,
                    input,
                    gt,
                    ut,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else {
            crate::kprof!(ctx.gpu, stream, "ffn_gate_kgamma", {
                if gateup_splitk_ok {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        gt,
                        gate_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m128_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m16_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        input,
                        &self.weights.gate_proj,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            // up_proj GEMM: [n, H] → [n, inter]. Same split-K treatment as gate;
            // reuses the shared workspace (gate's reduce already consumed it).
            crate::kprof!(ctx.gpu, stream, "ffn_up_kgamma", {
                if gateup_splitk_ok {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        ut,
                        up_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m128_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m16_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        input,
                        &self.weights.up_proj,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            // activation(gate) * up for n tokens
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_kgamma", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    n * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } // end !fused_gateup

        // FFN activation-sparsity MEASUREMENT (ATLAS_MEASURE_FFN_SPARSITY).
        // This is the K=γ verify FFN — the path DFlash decode ACTUALLY uses
        // (forward() M=1 is not hit per decode step). Row 0 of the [n, *]
        // buffers is a representative decode activation. `input` is the
        // gate/up input (K=hidden); `gate_out` now holds silu(gate)*up, the
        // down_proj input (K=intermediate) — no recompute needed here (unlike
        // forward(), which consumes gate_out in its fused path). PURE READER:
        // measures row 0 into dedicated accumulators, never mutates the token
        // stream. Skipped under graph capture (lazy alloc illegal mid-capture)
        // → run with ATLAS_DFLASH_DEBUG_NO_GRAPH=1 to observe eager decode.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, _meas) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            self.measure_sparsity_site(ctx, gate_out, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma")?;
        }

        // down_proj GEMM: [n, inter] → [n, H]
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
            // Split-K down_proj: [M=n, N=h, K=inter]. The single-slice
            // m32_n64 kernel is occupancy-starved here (N=h=5120 → 80 CTAs
            // vs gate/up's 256 at N=inter=16384) and grinds a long K-loop.
            // Split-K multiplies CTAs by `splits` into an FP32 workspace,
            // then reduces → BF16. Gated by ATLAS_FFN_DOWN_SPLITK; falls
            // through to the single-slice path when disabled or unallocated.
            let splitk = crate::layers::ffn_down_splitk();
            let ws = if splitk > 0 {
                *self.splitk_workspace.lock().unwrap()
            } else {
                None
            };
            if let (true, Some(ws)) = (
                m32_path
                    && splitk > 0
                    && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
                    && self.reduce_splitk_k.0 != 0,
                ws,
            ) {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n64_m32_splitk(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk,
                    self.reduce_splitk_k,
                    gate_out,
                    dt,
                    output,
                    ws,
                    n,
                    h,
                    inter,
                    h, // ldb == N for tightly-packed T-weight
                    splitk,
                    stream,
                )?;
            } else if m32_path {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64,
                    gate_out,
                    dt,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            } else if m128_path {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128,
                    gate_out,
                    dt,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            } else if m16_path {
                let dt = self.down_proj_t.as_ref().unwrap();
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16,
                    gate_out,
                    dt,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    /// Whether `forward_kgamma` is the correctness-owned route for this FFN.
    ///
    /// Exact W4+SiLU rows must not depend on the legacy tensor-core opt-in:
    /// the exact dispatcher either selects the dynamic-M kernel tier or
    /// fails closed to ordinary K1 rows when that tier is unavailable.
    pub fn exact_kgamma_applicable(&self, rows: u32) -> bool {
        !exact_ffn_tc_override()
            && exact_ffn_auto_kgamma_applicable(
                rows,
                self.bf16_weights.is_none()
                    && !self.has_w3_gemm()
                    && self.activation == FfnActivation::SiLU,
            )
    }

    /// N-token prefill: GEMM for all projections.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 prefill dispatch: dense_gemm_bf16 for all three projections.
        // (Gemma-4 dense path — bypasses NVFP4 entirely; not affected by
        // ATLAS_PREFILL_FFN_FAST.)
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.try_forward_flashinfer_prefill(input, m, h, inter, gate_out, up_out, ctx, stream)? {
            return Ok(());
        }

        // Large-M W4A4 native NVFP4×NVFP4 fast path: route through the
        // CUTLASS-style `nvfp4_nvfp4_gemm_t_m64` kernel (native E2M1
        // tensor-core MMA) when:
        //   - ATLAS_E2M1_GEMM=1 was set at startup
        //   - all three required kernel symbols loaded (nvfp4_cutlass +
        //     quantize_nvfp4)
        //   - M >= 128 (kernel's intended window — matches the other
        //     prefill fast paths)
        // Each GEMM prequantizes BF16 activations to NVFP4 inline
        // (absmax + per-group E2M1) before dispatching the native MMA.
        // Theoretical 2× MFU lift over `w4a16_gemm_t_m128` (BF16×NVFP4)
        // since we eliminate the inner-loop dequant AND halve activation
        // DRAM traffic (0.5 B/elt vs 2 B/elt BF16). Takes precedence
        // over the fp8/v2/m128 paths when active. Disabled or geometrically
        // ineligible requests retain those fallbacks; eligible incomplete
        // requests fail before projection.
        let e2m1_full_requested = crate::layers::prefill_ffn_e2m1_enabled();
        let e2m1_down_requested = crate::layers::prefill_ffn_e2m1_down_only_enabled();
        let e2m1_kmajor_requested = crate::layers::prefill_ffn_e2m1_kmajor_enabled();
        let e2m1_kmajor_m256_requested = crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled();
        let e2m1_silu_quant_requested = crate::layers::prefill_ffn_e2m1_silu_quant_enabled();
        if m >= 128 && e2m1_silu_quant_requested {
            ensure!(
                e2m1_full_requested || e2m1_down_requested,
                "ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_GEMM=1 or ATLAS_E2M1_GEMM_DOWN_ONLY=1"
            );
            ensure!(
                self.e2m1_checkpoint_scales.is_some(),
                "ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_STATIC_SCALE=1 and valid checkpoint input scales"
            );
            ensure!(
                self.activation == FfnActivation::SiLU,
                "ATLAS_E2M1_SILU_QUANT=1 is valid only for SiLU-gated FFNs"
            );
            ensure!(
                self.nvfp4_silu_quantize_k.0 != 0,
                "ATLAS_E2M1_SILU_QUANT=1 requires quantize_silu_mul_bf16_to_nvfp4 in the quantize_nvfp4 module"
            );
        }
        if m >= 128 && e2m1_kmajor_requested {
            ensure!(
                e2m1_full_requested || e2m1_down_requested,
                "ATLAS_E2M1_KMAJOR=1 requires ATLAS_E2M1_GEMM=1 or ATLAS_E2M1_GEMM_DOWN_ONLY=1"
            );
            ensure!(
                self.has_e2m1_kmajor_runtime(),
                "ATLAS_E2M1_KMAJOR=1 requires nvfp4_nvfp4_gemm_kmajor_m128 plus the activation quantizer"
            );
            ensure!(
                self.has_transposed_ffn(),
                "ATLAS_E2M1_KMAJOR=1 requires transformed gate/up/down weights (ATLAS_FFN_M16_TRANSPOSED=1)"
            );
            ensure!(
                e2m1_kmajor_projection_shape(m, inter, h)
                    && e2m1_kmajor_projection_shape(m, h, inter),
                "ATLAS_E2M1_KMAJOR=1 requires M>=128, N divisible by 128, and K divisible by 64 for every FFN projection; got M={m} hidden={h} intermediate={inter}"
            );
        }
        if e2m1_kmajor_m256_requested {
            ensure!(
                e2m1_kmajor_requested,
                "ATLAS_E2M1_KMAJOR_M256=1 requires ATLAS_E2M1_KMAJOR=1"
            );
            if m >= E2M1_KMAJOR_M256_MIN_M {
                ensure!(
                    self.has_e2m1_kmajor_m256_runtime(),
                    "ATLAS_E2M1_KMAJOR_M256=1 requires nvfp4_nvfp4_gemm_kmajor_m256 plus the M128 K-major runtime"
                );
            }
        }
        let kmajor_geometry_ready =
            e2m1_kmajor_projection_shape(m, inter, h) && e2m1_kmajor_projection_shape(m, h, inter);
        let selected_kernel =
            e2m1_selected_kernel(e2m1_kmajor_requested, e2m1_kmajor_m256_requested, m);
        let e2m1_runtime_ready = if e2m1_kmajor_requested {
            self.has_e2m1_kmajor_runtime()
                && self.has_transposed_ffn()
                && kmajor_geometry_ready
                && (selected_kernel != E2m1SelectedKernel::KmajorM256
                    || self.has_e2m1_kmajor_m256_runtime())
        } else {
            self.has_e2m1_ffn()
        };
        let silu_quant_ready = !e2m1_silu_quant_requested
            || (self.e2m1_checkpoint_scales.is_some()
                && self.activation == FfnActivation::SiLU
                && self.nvfp4_silu_quantize_k.0 != 0);

        // Per-shape E2M1 dispatch: gate/up stay on `w4a16_gemm_t_m128`,
        // down_proj routes through native E2M1 hardware MMA. Per the
        // shape table at the `prefill_ffn_e2m1_down_only_enabled` doc
        // site: gate (K=5120,N=17408) + up (same) are faster on the
        // BF16×NVFP4 w4a16 m128 path, while down (K=17408,N=5120) is
        // 1.31× faster via E2M1 MMA — net ~30% down savings with no
        // gate/up regression. Mutually exclusive with the all-three
        // `e2m1_fast_path`.
        let e2m1_route = e2m1_prefill_route(
            e2m1_full_requested,
            e2m1_down_requested,
            m >= 128,
            e2m1_runtime_ready && silu_quant_ready,
            e2m1_runtime_ready
                && silu_quant_ready
                && self.has_transposed_ffn()
                && self.w4a16_gemm_t_m128.0 != 0,
        );
        let (e2m1_fast_path, e2m1_down_only_path) = match e2m1_route {
            E2m1PrefillRoute::Complete(E2m1Scope::Full) => (true, false),
            E2m1PrefillRoute::Complete(E2m1Scope::DownOnly) => (false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full) => {
                bail!(
                    "ATLAS_E2M1_GEMM=1 requires the complete selected W4A4 runtime before gate projection"
                );
            }
            E2m1PrefillRoute::Missing(E2m1Scope::DownOnly) => {
                bail!(
                    "ATLAS_E2M1_GEMM_DOWN_ONLY=1 requires the complete selected W4A4 runtime plus transformed W4A16 M128 gate/up before gate projection"
                );
            }
            E2m1PrefillRoute::Disabled | E2m1PrefillRoute::Ineligible => (false, false),
        };
        if let E2m1PrefillRoute::Complete(scope) = e2m1_route {
            log_e2m1_route_once(E2m1RouteProof {
                scope,
                kernel: selected_kernel,
                checkpoint_static: self.e2m1_checkpoint_scales.is_some(),
                fused_silu_input: e2m1_silu_quant_requested,
                activation: self.activation,
                m,
                h,
                inter,
            });
        }

        // Large-M FP8 predequant fast path: route through the
        // `fp8_gemm_t_m128` kernel (BF16 A × pre-dequanted FP8 B) when:
        //   - ATLAS_FFN_PREDEQUANT_FP8=1 was set at startup
        //   - `predequant_for_prefill` ran successfully (gate/up/down
        //     FP8 buffers installed)
        //   - the `fp8_gemm_t_m128` kernel symbol is present
        //   - M >= 128
        // Saves the entire DEQUANT phase + one __syncthreads per
        // K-step inside w4a16_gemm_t_m128. Empirically a REGRESSION
        // on Qwen3.6-27B (5.68s → 8.11s TTFT at ISL=3603) because the
        // 2× B DRAM traffic (FP8 1 byte vs NVFP4 0.5 byte/elt) at
        // K=5120 + N=17408 exceeds the dequant savings — the FFN GEMM
        // is memory-bound, not compute-bound. Falls back to the w4a16
        // m128 path when not active. Mirrors the attention
        // `predequant_for_prefill` + `fp8_gemm_n128_m128` pattern.
        let fp8_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && m >= 128
            && crate::layers::prefill_ffn_fp8_enabled()
            && self.has_fp8_ffn();

        // Large-M v2 fast path: route through the 8-warp shadow
        // `w4a16_gemm_t_m128_v2` kernel when:
        //   - ATLAS_FFN_M128_V2=1
        //   - transposed (`nvfp4_t`) FFN weights installed
        //   - the v2 kernel symbol is present
        //   - M >= 128
        // Same SMEM footprint + 2-stage pipeline as v1 but doubles the
        // active warp count (256 threads/CTA, 4 warps per chunk run
        // chunk 0 and chunk 1 MMAs in parallel instead of serial). Net
        // result on compute-bound GEMMs: ~10-20% kernel-time win. On
        // memory-bound GEMMs: roughly neutral. Falls back to v1 when
        // not active. Original kernel:
        // kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu —
        // copied verbatim into qwen3.6-27b/.
        let v2_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && !fp8_fast_path
            && m >= 128
            && crate::layers::prefill_ffn_m128_v2_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m128_v2.0 != 0;

        // Large-M fast path: route through the transposed-weight
        // M_TILE=128 kernel (`w4a16_gemm_t_m128`) when:
        //   - ATLAS_PREFILL_FFN_FAST=1
        //   - transposed (`nvfp4_t`) FFN weights installed at load
        //     (requires ATLAS_FFN_M16_TRANSPOSED=1)
        //   - the kernel symbol is present in the loaded module
        //   - M >= 128 (kernel's intended window — small-M would waste
        //     128-row CTAs the same way the M_TILE=64 path wastes 64-row
        //     CTAs at M < 64)
        // Mirrors the attention `w4a16_gemm_m128_dispatch` pattern in
        // `qwen3_attention/prefill_weights.rs:14`. Disabled or ineligible
        // requests retain the lower-priority route. An eligible explicit
        // request fails closed when any required weight/kernel is absent so
        // a stale bundle cannot be mislabeled as a FAST benchmark.
        let fast_shape_eligible =
            !e2m1_fast_path && !e2m1_down_only_path && !fp8_fast_path && !v2_fast_path && m >= 128;
        let fast_route = prefill_ffn_fast_route(
            crate::layers::prefill_ffn_fast_enabled(),
            fast_shape_eligible,
            self.gate_proj_t.is_some() && self.up_proj_t.is_some() && self.down_proj_t.is_some(),
            self.w4a16_gemm_t_m16.0 != 0,
            self.w4a16_gemm_t_m128.0 != 0,
        );
        let fast_path = match fast_route {
            PrefillFfnFastRoute::Complete => true,
            PrefillFfnFastRoute::Missing if !crate::layers::prefill_ffn_fast_explicit() => {
                static DEFAULT_MISSING: std::sync::Once = std::sync::Once::new();
                DEFAULT_MISSING.call_once(|| {
                    tracing::warn!(
                        "default-on ATLAS_PREFILL_FFN_FAST skipped: transposed FFN weights or \
                         w4a16_gemm_t_m16/m128 are absent for this checkpoint; using the \
                         M_TILE=64 route (set ATLAS_PREFILL_FFN_FAST=1 to fail closed instead)"
                    );
                });
                false
            }
            PrefillFfnFastRoute::Missing => {
                bail!(
                    "ATLAS_PREFILL_FFN_FAST=1 requires transformed gate/up/down weights plus w4a16_gemm_t_m16 and w4a16_gemm_t_m128 before gate projection"
                );
            }
            PrefillFfnFastRoute::Disabled | PrefillFfnFastRoute::Ineligible => false,
        };

        // One-shot info log on first prefill so the bench harness can
        // verify which path is firing. The atomic ensures we log once
        // across all 64 layers per process. Uses both tracing::info! and
        // eprintln! — the latter bypasses tracing's BufWriter when the
        // server is launched under `nohup ... > file 2>&1` (stderr is
        // line-buffered to a regular file, so a single `\n`-terminated
        // write hits disk immediately, even before process teardown).
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let has_t = self.has_transposed_ffn();
            let has_fp8 = self.has_fp8_ffn();
            let has_e2m1 = self.has_e2m1_ffn();
            let has_e2m1_kmajor = self.has_e2m1_kmajor_runtime();
            let has_e2m1_kmajor_m256 = self.has_e2m1_kmajor_m256_runtime();
            let m128_ok = self.w4a16_gemm_t_m128.0 != 0;
            let m128_v2_ok = self.w4a16_gemm_t_m128_v2.0 != 0;
            let fp8_m128_ok = self.fp8_gemm_t_m128_k.0 != 0;
            let e2m1_ok = self.nvfp4_gemm_k.0 != 0;
            let gate_on = std::env::var("ATLAS_PREFILL_FFN_FAST").ok().as_deref() == Some("1");
            let fp8_gate_on =
                std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1");
            let v2_gate_on = std::env::var("ATLAS_FFN_M128_V2").ok().as_deref() == Some("1");
            let e2m1_gate_on = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
            let e2m1_down_only_gate_on =
                std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
            let e2m1_static_scale_on = self.e2m1_checkpoint_scales.is_some();
            tracing::info!(
                m,
                inter,
                h,
                e2m1_fast_path,
                e2m1_down_only_path,
                fp8_fast_path,
                v2_fast_path,
                fast_path,
                has_e2m1_ffn = has_e2m1,
                has_fp8_ffn = has_fp8,
                has_transposed = has_t,
                e2m1_kernel = e2m1_ok,
                fp8_m128_kernel = fp8_m128_ok,
                m128_v2_kernel = m128_v2_ok,
                m128_kernel = m128_ok,
                gate = gate_on,
                fp8_gate = fp8_gate_on,
                v2_gate = v2_gate_on,
                e2m1_gate = e2m1_gate_on,
                e2m1_down_only_gate = e2m1_down_only_gate_on,
                e2m1_static_scale = e2m1_static_scale_on,
                e2m1_kmajor_requested,
                e2m1_kmajor_m256_requested,
                e2m1_silu_quant_requested,
                has_e2m1_kmajor,
                has_e2m1_kmajor_m256,
                "dense_ffn forward_prefill dispatch (one-shot)"
            );
            eprintln!(
                "[atlas-prefill-ffn] dispatch: M={m} inter={inter} h={h} \
                 e2m1_fast_path={e2m1_fast_path} \
                 e2m1_down_only_path={e2m1_down_only_path} \
                 fp8_fast_path={fp8_fast_path} v2_fast_path={v2_fast_path} \
                 fast_path={fast_path} has_e2m1={has_e2m1} has_fp8={has_fp8} \
                 has_transposed={has_t} e2m1_kernel={e2m1_ok} \
                 fp8_m128_kernel={fp8_m128_ok} \
                 m128_v2_kernel={m128_v2_ok} m128_kernel={m128_ok} \
                 e2m1_gate=ATLAS_E2M1_GEMM={e2m1_gate_on} \
                 e2m1_down_only_gate=ATLAS_E2M1_GEMM_DOWN_ONLY={e2m1_down_only_gate_on} \
                 e2m1_static_scale=ATLAS_E2M1_STATIC_SCALE={e2m1_static_scale_on} \
                 e2m1_kmajor=ATLAS_E2M1_KMAJOR={e2m1_kmajor_requested} \
                 e2m1_kmajor_m256=ATLAS_E2M1_KMAJOR_M256={e2m1_kmajor_m256_requested} \
                 e2m1_silu_quant=ATLAS_E2M1_SILU_QUANT={e2m1_silu_quant_requested} \
                 e2m1_kmajor_kernel={has_e2m1_kmajor} \
                 e2m1_kmajor_m256_kernel={has_e2m1_kmajor_m256} \
                 gate=ATLAS_PREFILL_FFN_FAST={gate_on} \
                 fp8_gate=ATLAS_FFN_PREDEQUANT_FP8={fp8_gate_on} \
                 v2_gate=ATLAS_FFN_M128_V2={v2_gate_on}"
            );
        }

        if e2m1_fast_path {
            let _scratch_use = self.lock_e2m1_use(stream)?;
            // Native W4A4 NVFP4×NVFP4 dispatch: prequant activations to
            // NVFP4 in-place, then issue the CUTLASS-style E2M1×E2M1 MMA
            // GEMM. Uses the standard `[N, K/2]` HuggingFace weight
            // layout (NOT the `nvfp4_t` transposed layout) — matches the
            // kernel's coalesced gmem read pattern.
            //
            // gate_proj and up_proj share the exact same [M, H] BF16 input.
            // Prepare it once: the old path repeated an identical global
            // absmax, D2H synchronization/readback, and quantization before
            // each projection even though both consumed identical bytes.
            let gate_up_scale = self.e2m1_checkpoint_scales.map(|scales| scales.gate_up);
            let gate_weight = if e2m1_kmajor_requested {
                self.gate_proj_t.as_ref().unwrap()
            } else {
                &self.weights.gate_proj
            };
            let up_weight = if e2m1_kmajor_requested {
                self.up_proj_t.as_ref().unwrap()
            } else {
                &self.weights.up_proj
            };
            let down_weight = if e2m1_kmajor_requested {
                self.down_proj_t.as_ref().unwrap()
            } else {
                &self.weights.down_proj
            };
            let (a_packed, a_scale, a_scale2) =
                self.prepare_e2m1_input(ctx, input, m, h, gate_up_scale, stream)?;
            self.forward_e2m1_prepared(
                ctx,
                a_packed,
                a_scale,
                a_scale2,
                gate_weight,
                gate_out,
                m,
                inter,
                h,
                e2m1_kmajor_requested,
                stream,
            )?;
            self.forward_e2m1_prepared(
                ctx,
                a_packed,
                a_scale,
                a_scale2,
                up_weight,
                up_out,
                m,
                inter,
                h,
                e2m1_kmajor_requested,
                stream,
            )?;
            // down_proj: [M, inter] BF16 → [M, H] BF16
            let output = ctx.buffers.moe_output();
            if e2m1_silu_quant_requested {
                let down_scale = self.e2m1_checkpoint_scales.unwrap().down;
                let (a_packed, a_scale, a_scale2) = self
                    .prepare_e2m1_silu_input(ctx, gate_out, up_out, m, inter, down_scale, stream)?;
                self.forward_e2m1_prepared(
                    ctx,
                    a_packed,
                    a_scale,
                    a_scale2,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    e2m1_kmajor_requested,
                    stream,
                )?;
            } else {
                // SiLU/GELU(gate) * up for all M tokens
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                self.forward_e2m1_proj(
                    ctx,
                    gate_out,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    self.e2m1_checkpoint_scales.map(|scales| scales.down),
                    e2m1_kmajor_requested,
                    stream,
                )?;
            }
            return Ok(());
        }

        if e2m1_down_only_path {
            let _scratch_use = self.lock_e2m1_use(stream)?;
            // gate/up stay on the w4a16 m128 fast path; only down_proj
            // routes through E2M1 hardware MMA. Matches the shape table
            // documented at `prefill_ffn_e2m1_down_only_enabled` — net
            // ~30% down_proj savings, no gate/up regression vs the
            // standard `fast_path`.
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            let down_weight = if e2m1_kmajor_requested {
                self.down_proj_t.as_ref().unwrap()
            } else {
                &self.weights.down_proj
            };
            if e2m1_silu_quant_requested {
                let down_scale = self.e2m1_checkpoint_scales.unwrap().down;
                let (a_packed, a_scale, a_scale2) = self
                    .prepare_e2m1_silu_input(ctx, gate_out, up_out, m, inter, down_scale, stream)?;
                self.forward_e2m1_prepared(
                    ctx,
                    a_packed,
                    a_scale,
                    a_scale2,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    e2m1_kmajor_requested,
                    stream,
                )?;
            } else {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                self.forward_e2m1_proj(
                    ctx,
                    gate_out,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    self.e2m1_checkpoint_scales.map(|scales| scales.down),
                    e2m1_kmajor_requested,
                    stream,
                )?;
            }
            return Ok(());
        }

        if v2_fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();
            // gate_proj GEMM via 8-warp v2 kernel
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fp8_fast_path {
            // Pre-dequanted FP8 weights — single sync per K-step in
            // the kernel, no DEQUANT phase. Same A×B GEMM math, just
            // bypassing the inner-loop NVFP4 → FP8 dequant.
            let gfp8 = self.gate_fp8.unwrap();
            let ufp8 = self.up_fp8.unwrap();
            let dfp8 = self.down_fp8.unwrap();

            // gate_proj GEMM: [M, H] BF16 × [inter, H] FP8 → [M, inter] BF16
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                gfp8,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                ufp8,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                gate_out,
                dfp8,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            // gate_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            // up_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            // activation(gate) * up for all M tokens (SiLU or GELU)
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            // down_proj GEMM: [M, inter] → [M, H]
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            log_prefill_ffn_fast_route_once(m, h, inter);
            return Ok(());
        }

        // Byte-exact cp.async pipelined path (ATLAS_PREFILL_FFN_PIPE=1):
        // `w4a16_gemm_pipe` is a load-pipelined byte-exact shadow of the
        // baseline M_TILE=64 kernel below (same B dequant arithmetic, same
        // m16n8k16 MMA sequence/accumulation order) — only the weight-load
        // pipeline differs. Small-M prefills pay a latency-bound fixed cost
        // per layer on the baseline kernel (~8ms/layer at M=18); the pipe
        // kernel overlaps the stream and should land near the ~190 GB/s the
        // verify-path kernels achieve. Default off; an eligible explicit
        // request with handle 0 fails before launching the gate projection.
        // `w4a16_gemm_pipe` is byte-exact ONLY when the K reduction is a
        // multiple of its 64-row stage; a partial final stage would load past
        // the row's packed-weight boundary and silently corrupt output. The
        // FFN shapes (h=5120, inter=17408) satisfy this, but guard explicitly
        // so any future topology change falls back to the exact baseline.
        let pipe_requested = crate::layers::prefill_ffn_pipe_enabled();
        let pipe_shape_eligible = h.is_multiple_of(64) && inter.is_multiple_of(64);
        if pipe_requested && pipe_shape_eligible && self.w4a16_gemm_pipe.0 == 0 {
            bail!(
                "ATLAS_PREFILL_FFN_PIPE requested for an eligible shape but w4a16_gemm_pipe is missing"
            );
        }
        let pipe_ready = pipe_requested && pipe_shape_eligible && self.w4a16_gemm_pipe.0 != 0;
        let dual_route = prefill_fused_epilogue_route(
            crate::layers::prefill_ffn_dual_fused_enabled()?,
            pipe_ready,
            self.w4a16_gemm_pipe_dual.0 != 0,
            self.activation,
            m,
            h,
            inter,
        );
        if dual_route == PrefillFusedEpilogueRoute::Missing {
            bail!(
                "ATLAS_PREFILL_FFN_DUAL_FUSED requested for an eligible shape but its parent pipe or dual kernel is missing"
            );
        }
        let dual_fused = dual_route == PrefillFusedEpilogueRoute::Complete;
        let fused_route = if dual_fused {
            // Dual is authoritative when both explicit candidate flags are set.
            PrefillFusedEpilogueRoute::Disabled
        } else {
            prefill_fused_epilogue_route(
                crate::layers::prefill_ffn_fused_epilogue_enabled(),
                pipe_ready,
                self.w4a16_gemm_pipe_silu_mul.0 != 0,
                self.activation,
                m,
                h,
                inter,
            )
        };
        if fused_route == PrefillFusedEpilogueRoute::Missing {
            bail!(
                "ATLAS_PREFILL_FFN_FUSED_EPILOGUE requested for an eligible shape but its parent pipe or fused kernel is missing"
            );
        }
        let fused_epilogue = fused_route == PrefillFusedEpilogueRoute::Complete;

        if pipe_ready {
            if !dual_fused && fused_epilogue {
                log_prefill_up_fused_route_once();
            } else if !dual_fused {
                log_prefill_pipe_route_once();
            }
            if dual_fused {
                ops::w4a16_gemm_pipe_dual(
                    ctx.gpu,
                    self.w4a16_gemm_pipe_dual,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    gate_out,
                    DevicePtr::NULL,
                    true,
                    m,
                    inter,
                    h,
                    stream,
                )?;
                log_prefill_dual_fused_route_once();
            } else {
                ops::w4a16_gemm_pipe(
                    ctx.gpu,
                    self.w4a16_gemm_pipe,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
            }
            if !dual_fused && fused_epilogue {
                ops::w4a16_gemm_pipe_silu_mul(
                    ctx.gpu,
                    self.w4a16_gemm_pipe_silu_mul,
                    input,
                    &self.weights.up_proj,
                    gate_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
            } else if !dual_fused {
                ops::w4a16_gemm_pipe(
                    ctx.gpu,
                    self.w4a16_gemm_pipe,
                    input,
                    &self.weights.up_proj,
                    up_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
            }
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_pipe(
                ctx.gpu,
                self.w4a16_gemm_pipe,
                gate_out,
                &self.weights.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Baseline M_TILE=64 path (unchanged).
        // gate_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;

        // up_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;

        // activation(gate) * up for all M tokens (SiLU or GELU)
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;

        // down_proj GEMM: [M, inter] → [M, H]
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// Batched forward (per-token loop). Used by forward_batched in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}

#[cfg(test)]
#[path = "dense_ffn/exact_route_tests.rs"]
mod exact_route_tests;

#[cfg(test)]
#[path = "dense_ffn/w3_route_tests.rs"]
mod w3_route_tests;

#[cfg(test)]
mod flashinfer_merged_ffn_tests {
    use super::{
        FlashinferFfnPrefillRoute as Route, e2m1_scratch_layout, flashinfer_ffn_prefill_route,
    };
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    use super::{
        ensure_flashinfer_ffn_disjoint, flashinfer_ffn_activation_scale_extent,
        flashinfer_ffn_extent,
    };
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    use spark_runtime::gpu::{DevicePtr, KernelHandle, mock::MockGpuBackend};

    #[test]
    fn explicit_route_is_exact_and_missing_is_fail_closed() {
        // `m_qualified` is now supplied by the caller (see the fn doc): the
        // frozen pair 2079/8192 qualifies, 2080 does not.
        let q = |m: u32| matches!(m, 2_079 | 8_192);
        assert_eq!(
            flashinfer_ffn_prefill_route(false, true, q(2_079), false),
            Route::Disabled
        );
        assert_eq!(
            flashinfer_ffn_prefill_route(true, false, q(2_079), true),
            Route::Ineligible
        );
        assert_eq!(
            flashinfer_ffn_prefill_route(true, true, q(2_080), false),
            Route::Ineligible
        );
        for m in [2_079u32, 8_192] {
            assert_eq!(
                flashinfer_ffn_prefill_route(true, true, q(m), false),
                Route::Missing
            );
            assert_eq!(
                flashinfer_ffn_prefill_route(true, true, q(m), true),
                Route::Complete
            );
        }
    }

    /// The extended ladder must stay OFF by default: with the env unset, an M
    /// that is only in `QWEN38_FFN_EXTRA_M` must still be rejected, so default
    /// behaviour is bit-for-bit the frozen table.
    #[test]
    fn extended_m_ladder_is_off_unless_requested() {
        use crate::weight_map::flashinfer_ffn_admission::{
            Qwen38FfnOperation, qwen38_ffn_extra_m_enabled, qwen38_ffn_launch_candidate,
        };
        if qwen38_ffn_extra_m_enabled() {
            return; // caller opted in; the negative assertion does not apply
        }
        for m in [1_024usize, 2_048, 4_096, 6_144, 7_168, 8_187] {
            assert!(
                qwen38_ffn_launch_candidate(Qwen38FfnOperation::MergedGateUp, m).is_err(),
                "M={m} must be unqualified while ATLAS_FLASHINFER_FFN_EXTRA_M is unset"
            );
        }
    }

    #[test]
    fn construction_scratch_layout_matches_the_charged_maximum() {
        let layout = e2m1_scratch_layout(8_192, 17_408).unwrap();
        assert_eq!(layout.cap_m, 8_192);
        assert_eq!(layout.cap_k, 17_408);
        assert_eq!(layout.packed_bytes, 71_303_168);
        assert_eq!(layout.scale_bytes, 8_912_896);
        assert_eq!(layout.max_bytes, 4);
        assert_eq!(
            layout.packed_bytes + layout.scale_bytes + layout.max_bytes,
            80_216_068
        );
        assert!(e2m1_scratch_layout(usize::MAX, 17_408).is_err());
        assert!(e2m1_scratch_layout(8_192, usize::MAX - 15).is_err());
    }

    #[test]
    fn production_order_preflights_then_merges_directly_quantizes_and_completes_down() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("fn try_forward_flashinfer_prefill(")
            .expect("FlashInfer FFN runtime route");
        let end = source[start..]
            .find("/// Install W3")
            .map(|offset| start + offset)
            .expect("FlashInfer FFN runtime boundary");
        let body = &source[start..end];

        let missing = body.find("FlashinferFfnPrefillRoute::Missing").unwrap();
        let down_preflight = body.find("let mut down_launch").unwrap();
        let scratch_guard = body.find("self.lock_e2m1_use(stream)").unwrap();
        let physical_scale_extent = body
            .find("flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)")
            .unwrap();
        let quantize = body
            .find("ops::quantize_bf16_to_nvfp4_atlas_128x4(")
            .unwrap();
        let merged = body.find("merged_launch.launch_eager(").unwrap();
        let direct_silu_quantize = body
            .find("ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .unwrap();
        let down = body.find("down_launch.launch_eager(").unwrap();
        let receipt = body.find("log_flashinfer_ffn_success(").unwrap();
        assert!(
            missing < physical_scale_extent
                && physical_scale_extent < down_preflight
                && down_preflight < scratch_guard
                && scratch_guard < quantize
        );
        assert!(quantize < merged && merged < direct_silu_quantize && direct_silu_quantize < down);
        assert!(down < receipt);
        assert!(!body.contains("ops::silu_mul("));
        assert!(!body.contains("KernelLaunch::new(ctx.gpu, route.split_gate_up_k)"));
        assert!(!body.contains("ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4("));
        assert!(body.contains("f32::from_bits(route.down.input_scale_bits)"));
        assert!(body.contains("ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes"));
        assert!(body.contains("ctx.buffers.ffn_gate_up_bf16()"));
        assert!(!body.contains("Qwen38FfnOperation::Gate, m as usize"));
        assert!(!body.contains("Qwen38FfnOperation::Up, m as usize"));
        assert!(!body.contains("route.stream"));
        assert!(body.contains("prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)"));
        assert!(body.contains("prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)"));
    }

    #[test]
    fn construction_stream_is_preflight_only_and_runtime_owns_launch_stream() {
        let source = include_str!("dense_ffn.rs");
        let struct_start = source.find("struct FlashinferFfnPrefill {").unwrap();
        let struct_end = source[struct_start..]
            .find("\n}")
            .map(|offset| struct_start + offset)
            .unwrap();
        assert!(!source[struct_start..struct_end].contains("stream: u64"));

        let prepare_start = source.find("pub fn prepare_flashinfer_prefill(").unwrap();
        let runtime_start = source.find("fn try_forward_flashinfer_prefill(").unwrap();
        let prepare = &source[prepare_start..runtime_start];
        assert!(prepare.contains("let stream = gpu.default_stream();"));
        assert!(!prepare.contains("stream,\n        });"));

        let runtime_end = source[runtime_start..]
            .find("/// Install W3")
            .map(|offset| runtime_start + offset)
            .unwrap();
        let runtime = &source[runtime_start..runtime_end];
        assert!(!runtime.contains("stream == route.stream"));
        assert!(runtime.contains("self.lock_e2m1_use(stream)"));

        let lock_start = source.find("fn lock_e2m1_use(").unwrap();
        let lock_end = source[lock_start..]
            .find("/// Quantize one BF16 activation matrix")
            .map(|offset| lock_start + offset)
            .unwrap();
        let lock = &source[lock_start..lock_end];
        assert!(lock.contains(".e2m1_use_stream"));
        assert!(lock.contains(".lock()"));
        assert!(lock.contains(".map_err("));
        assert!(!lock.contains(".unwrap()"));
    }

    #[test]
    fn split_kernel_is_exact_row_major_gate_then_up() {
        let cuda = include_str!("../../../../kernels/gb10/common/flashinfer_projection_split.cu");
        let start = cuda
            .find("void flashinfer_projection_split_ffn_gate_up(")
            .expect("merged FFN split kernel");
        let body = &cuda[start..];
        assert!(body.contains("constexpr uint32_t INTERMEDIATE = 17408"));
        assert!(body.contains("constexpr uint32_t TOTAL = 2 * INTERMEDIATE"));
        assert!(body.contains("merged + row * TOTAL"));
        assert!(body.contains("gate_dst[vector] = src[vector]"));
        assert!(body.contains("up_dst[vector] = src[VECTORS + vector]"));
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn physical_activation_scale_extent_is_exact_for_qualified_rows() {
        assert_eq!(
            flashinfer_ffn_activation_scale_extent(2_079, 17_408).unwrap(),
            2_367_488
        );
        assert_eq!(
            flashinfer_ffn_activation_scale_extent(8_192, 17_408).unwrap(),
            8_912_896
        );
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn physical_activation_scale_extent_rejects_padding_overflow() {
        assert!(flashinfer_ffn_activation_scale_extent(usize::MAX, 17_408).is_err());
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn physical_tail_overlap_is_rejected_before_launch() {
        let rows = 2_079;
        let cols = 17_408;
        let logical_bytes = flashinfer_ffn_extent(rows, cols / 16, 1).unwrap();
        let physical_bytes = flashinfer_ffn_activation_scale_extent(rows, cols).unwrap();
        assert_eq!(physical_bytes - logical_bytes, 105_536);

        let scales = DevicePtr(0x1_0000);
        let destination = DevicePtr(scales.0 + u64::try_from(logical_bytes).unwrap());
        assert!(
            ensure_flashinfer_ffn_disjoint(scales, logical_bytes, destination, 16).is_ok(),
            "the hostile destination begins exactly after the logical prefix"
        );
        assert!(
            ensure_flashinfer_ffn_disjoint(scales, physical_bytes, destination, 16).is_err(),
            "the padded physical tail must reject the same destination before launch"
        );
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn fused_physical_wrapper_accepts_exact_rows_and_rejects_hostile_inputs_before_launch() {
        let gpu = MockGpuBackend::new();
        let gate = DevicePtr(0x0010_0000);
        let up = DevicePtr(0x1000_0000);
        let packed = DevicePtr(0x2000_0000);
        let scales = DevicePtr(0x3000_0000);
        for rows in [2_079, 8_192] {
            super::ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                KernelHandle(0x55),
                gate,
                up,
                packed,
                scales,
                0.25,
                rows,
                17_408,
                0x77,
            )
            .unwrap();
        }
        assert_eq!(gpu.launch_count(), 2);

        for (kernel, gate, up, packed, scales, scale2, rows, cols) in [
            (
                KernelHandle(0),
                gate,
                up,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                DevicePtr::NULL,
                up,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                DevicePtr::NULL,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                up,
                DevicePtr::NULL,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                up,
                packed,
                DevicePtr::NULL,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                up,
                packed,
                scales,
                f32::NAN,
                2_079,
                17_408,
            ),
            (KernelHandle(1), gate, up, packed, scales, 0.25, 0, 17_408),
            (KernelHandle(1), gate, up, packed, scales, 0.25, 2_079, 16),
            (
                KernelHandle(1),
                gate,
                up,
                packed,
                scales,
                0.25,
                u32::MAX,
                17_408,
            ),
        ] {
            assert!(
                super::ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    &gpu, kernel, gate, up, packed, scales, scale2, rows, cols, 0x77,
                )
                .is_err()
            );
        }
        assert_eq!(gpu.launch_count(), 2);
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn direct_merged_wrapper_is_abi_separate_and_fails_before_launch() {
        let gpu = MockGpuBackend::new();
        let merged = DevicePtr(0x0010_0000);
        let packed = DevicePtr(0x2000_0000);
        let scales = DevicePtr(0x3000_0000);
        for rows in [2_079, 8_192] {
            super::ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                KernelHandle(0x66),
                merged,
                packed,
                scales,
                0.25,
                rows,
                17_408,
                0x77,
            )
            .unwrap();
        }
        assert_eq!(gpu.launch_count(), 2);

        for (kernel, merged, packed, scales, scale2, rows, cols) in [
            (KernelHandle(0), merged, packed, scales, 0.25, 2_079, 17_408),
            (
                KernelHandle(1),
                DevicePtr::NULL,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                merged,
                DevicePtr::NULL,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                merged,
                packed,
                DevicePtr::NULL,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                merged,
                packed,
                scales,
                f32::NAN,
                2_079,
                17_408,
            ),
            (KernelHandle(1), merged, packed, scales, 0.25, 0, 17_408),
            (KernelHandle(1), merged, packed, scales, 0.25, 2_079, 16),
            (
                KernelHandle(1),
                merged,
                packed,
                scales,
                0.25,
                u32::MAX,
                17_408,
            ),
        ] {
            assert!(
                super::ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    &gpu, kernel, merged, packed, scales, scale2, rows, cols, 0x77,
                )
                .is_err()
            );
        }
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn fused_physical_symbol_layout_abi_handle_and_receipt_are_source_bound() {
        let cuda = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
        let logical = cuda
            .find("void quantize_silu_mul_bf16_to_nvfp4(")
            .expect("existing logical W4A4 fused symbol");
        let physical = cuda
            .find("void quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .expect("ABI-separated physical fused symbol");
        assert!(logical < physical);
        let physical_body = &cuda[physical..];
        assert!(physical_body.contains("((N + 127u) / 128u) * 128u"));
        assert!(physical_body.contains("if (row >= N)"));
        assert!(physical_body.contains("silu_scale_offset_128x4(row, group, num_groups)] = 0"));
        assert!(physical_body.contains("silu_mul_bf16_round(gate, up"));
        assert!(physical_body.contains("float_to_fp8_e4m3(fp8_float)"));
        assert!(physical_body.contains("quantize_e2m1(v0)"));

        let direct = cuda
            .find("void quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .expect("ABI-separated direct merged physical symbol");
        assert!(physical < direct);
        let direct_body = &cuda[direct..];
        assert!(direct_body.contains("static_cast<unsigned long long>(row) * (2ull * K)"));
        assert!(direct_body.contains("const __nv_bfloat16* up = gate + K"));
        assert!(direct_body.contains("float rounded[GROUP_SIZE]"));
        assert_eq!(
            direct_body.matches("silu_mul_bf16_round(gate, up").count(),
            1
        );
        assert!(direct_body.contains("rounded[i] = value"));
        assert!(direct_body.contains("quantize_e2m1(rounded[i] * inv_eff)"));
        assert!(direct_body.contains("silu_scale_offset_128x4(row, group, num_groups)] = 0"));

        let wrapper_source = include_str!("ops/gemm_dense.rs");
        let wrapper_start = wrapper_source
            .find("pub fn quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .expect("direct merged physical Rust wrapper");
        let wrapper_end = wrapper_source[wrapper_start..]
            .find("/// Native NVFP4")
            .map(|offset| wrapper_start + offset)
            .expect("physical fused Rust wrapper boundary");
        let wrapper = &wrapper_source[wrapper_start..wrapper_end];
        let merged = wrapper.find(".arg_ptr(merged_gate_up)").unwrap();
        let packed = wrapper.find(".arg_ptr(packed_out)").unwrap();
        let scales = wrapper.find(".arg_ptr(scale_out)").unwrap();
        let scale2 = wrapper.find(".arg_f32(scale2)").unwrap();
        let rows = wrapper.find(".arg_u32(rows)").unwrap();
        let cols = wrapper.find(".arg_u32(cols)").unwrap();
        assert!(merged < packed && packed < scales && scales < scale2);
        assert!(scale2 < rows && rows < cols);
        assert!(wrapper.contains("cols >= 64 && cols.is_multiple_of(64)"));
        assert!(wrapper.contains(".checked_add(127)"));
        assert!(wrapper.contains(".grid([padded_rows.min(96), 1, 1])"));

        let source = include_str!("dense_ffn.rs");
        let prepare_start = source.find("pub fn prepare_flashinfer_prefill(").unwrap();
        let prepare_end = source[prepare_start..]
            .find("fn try_forward_flashinfer_prefill(")
            .map(|offset| prepare_start + offset)
            .unwrap();
        let prepare = &source[prepare_start..prepare_end];
        let symbol = prepare
            .find("\"quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4\"")
            .unwrap();
        let nonzero = prepare
            .find("quantize_merged_silu_atlas_128x4_k.0 != 0")
            .unwrap();
        let plan_freeze = prepare.find("for (operation, m) in [").unwrap();
        let scratch_preallocation = prepare
            .find("e2m1_scratch_layout(8_192, intermediate)")
            .unwrap();
        let scratch_capacity_check = prepare
            .find("retained.cap_m == scratch_layout.cap_m")
            .unwrap();
        let retained_copy = prepare.find("build_flashinfer_merged_gate_up(").unwrap();
        let route_retention = prepare.find("self.flashinfer_ffn_prefill = Some(").unwrap();
        assert!(
            symbol < nonzero
                && nonzero < plan_freeze
                && plan_freeze < scratch_preallocation
                && scratch_preallocation < scratch_capacity_check
                && scratch_capacity_check < retained_copy
                && retained_copy < route_retention
        );
        assert!(prepare.contains("\"quantize_nvfp4\""));
        assert!(prepare.contains("quantize_merged_silu_atlas_128x4_k,"));
        assert!(!prepare.contains("\"flashinfer_projection_split\""));
    }
}

#[cfg(test)]
mod prefill_ffn_fast_route_tests {
    use super::{PrefillFfnFastRoute as Route, prefill_ffn_fast_route};

    #[test]
    fn distinguishes_disabled_ineligible_complete_and_every_missing_dependency() {
        assert_eq!(
            prefill_ffn_fast_route(false, true, true, true, true),
            Route::Disabled
        );
        assert_eq!(
            prefill_ffn_fast_route(true, false, false, false, false),
            Route::Ineligible
        );
        assert_eq!(
            prefill_ffn_fast_route(true, true, true, true, true),
            Route::Complete
        );
        for readiness in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            assert_eq!(
                prefill_ffn_fast_route(true, true, readiness.0, readiness.1, readiness.2),
                Route::Missing
            );
        }
    }

    #[test]
    fn missing_fails_before_projection_and_marker_follows_completed_down_projection() {
        let source = include_str!("dense_ffn.rs");
        let select = source.find("let fast_route =").expect("FAST selector");
        let branch = source[select..]
            .find("if fast_path {")
            .map(|offset| select + offset)
            .expect("FAST projection branch");
        let selection = &source[select..branch];
        assert!(selection.contains("PrefillFfnFastRoute::Missing"));
        assert!(selection.contains("before gate projection"));

        let end = source[branch..]
            .find("// Byte-exact cp.async pipelined path")
            .map(|offset| branch + offset)
            .expect("FAST projection boundary");
        let body = &source[branch..end];
        let down = body
            .rfind("ops::w4a16_gemm_n128_m128(")
            .expect("FAST down projection");
        let marker = body
            .find("log_prefill_ffn_fast_route_once(m, h, inter);")
            .expect("FAST completion marker");
        let returned = body.rfind("return Ok(());").expect("FAST return");
        assert!(down < marker && marker < returned);
        assert!(source.contains(
            "ENGAGED ATLAS_PREFILL_FFN_FAST: approximate transposed M128 route complete"
        ));
    }
}

#[cfg(test)]
mod prefill_fused_epilogue_tests {
    use super::{FfnActivation, PrefillFusedEpilogueRoute, prefill_fused_epilogue_route};

    #[test]
    fn route_distinguishes_disabled_ineligible_complete_and_missing() {
        assert_eq!(
            prefill_fused_epilogue_route(false, true, true, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Disabled
        );
        for (activation, m, h, inter) in [
            (FfnActivation::GeLU, 8192, 5120, 17408),
            (FfnActivation::SiLU, 32, 5120, 17408),
            (FfnActivation::SiLU, 8192, 5119, 17408),
            (FfnActivation::SiLU, 8192, 5120, 17407),
        ] {
            assert_eq!(
                prefill_fused_epilogue_route(true, true, true, activation, m, h, inter),
                PrefillFusedEpilogueRoute::Ineligible
            );
        }
        assert_eq!(
            prefill_fused_epilogue_route(true, true, true, FfnActivation::SiLU, 33, 5120, 17408,),
            PrefillFusedEpilogueRoute::Complete
        );
        assert_eq!(
            prefill_fused_epilogue_route(true, false, true, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Missing
        );
        assert_eq!(
            prefill_fused_epilogue_route(true, true, false, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Missing
        );
    }

    #[test]
    fn source_fails_before_projection_and_logs_each_engaged_route() {
        let source = include_str!("dense_ffn.rs");
        let dispatch_start = source.find("let pipe_requested =").unwrap();
        let dispatch_end = source[dispatch_start..]
            .find("// Baseline M_TILE=64 path")
            .map(|offset| dispatch_start + offset)
            .unwrap();
        let dispatch = &source[dispatch_start..dispatch_end];
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_PIPE requested for an eligible shape but w4a16_gemm_pipe is missing"
        ));
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_DUAL_FUSED requested for an eligible shape but its parent pipe or dual kernel is missing"
        ));
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_FUSED_EPILOGUE requested for an eligible shape but its parent pipe or fused kernel is missing"
        ));
        assert!(
            dispatch.contains("Dual is authoritative when both explicit candidate flags are set")
        );
        let log_start = source.find("fn log_prefill_pipe_route_once").unwrap();
        let log_end = source[log_start..]
            .find("enum ExactFfnDispatch")
            .map(|offset| log_start + offset)
            .unwrap();
        let loggers = &source[log_start..log_end];
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_PIPE"));
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_FUSED_EPILOGUE"));
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_DUAL_FUSED"));

        let dual_launch = dispatch.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let dual_marker = dispatch
            .find("log_prefill_dual_fused_route_once();")
            .unwrap();
        assert!(dual_launch < dual_marker);
    }

    #[test]
    fn legacy_boolean_silent_fallback_is_absent() {
        let source = include_str!("dense_ffn.rs");
        let legacy_name = ["fn use_prefill", "_fused_epilogue("].concat();
        assert!(!source.contains(&legacy_name));
    }

    #[test]
    fn cuda_source_retains_bf16_round_trip_and_in_place_abi() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
        ));
        let start = source
            .find("extern \"C\" __global__ void w4a16_gemm_pipe_silu_mul")
            .expect("fused prefill epilogue kernel is missing");
        let body = &source[start..];
        assert!(body.contains("__nv_bfloat16 up_bf16 = __float2bfloat16(value)"));
        assert!(body.contains("float u = __bfloat162float(up_bf16)"));
        assert!(body.contains("float g = __bfloat162float(gate_in_out[out_idx])"));
        assert!(body.contains("gate_in_out[out_idx] = __float2bfloat16(g * sigmoid_g * u)"));
    }

    #[test]
    fn dual_cuda_source_retains_both_bf16_round_trips_and_independent_weights() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
        ));
        let start = source
            .find("extern \"C\" __global__ void w4a16_gemm_pipe_dual")
            .expect("dual fused prefill kernel is missing");
        let body = &source[start..];
        assert!(body.contains("const unsigned char* __restrict__ gate_packed"));
        assert!(body.contains("const unsigned char* __restrict__ up_packed"));
        assert!(body.contains("__nv_bfloat16* C1"));
        assert!(body.contains("if (fuse_silu != 0)"));
        assert!(body.contains("__bfloat162float(__float2bfloat16(gate_value))"));
        assert!(body.contains("__bfloat162float(__float2bfloat16(up_value))"));
        assert!(body.contains("__float2bfloat16(g * sigmoid_g * u)"));
        assert!(body.contains("C0[out_idx] = __float2bfloat16(g)"));
        assert!(body.contains("C1[out_idx] = __float2bfloat16(u)"));
    }

    #[test]
    fn dual_rust_abi_matches_cuda_and_fused_call_uses_null_second_output() {
        let ops = include_str!("ops/gemm_dense.rs");
        let start = ops
            .find("pub fn w4a16_gemm_pipe_dual(")
            .expect("dual Rust launcher is missing");
        let body = &ops[start..];
        let ordered = [
            ".arg_ptr(input)",
            ".arg_ptr(gate.weight)",
            ".arg_ptr(gate.weight_scale)",
            ".arg_f32(gate.weight_scale_2)",
            ".arg_ptr(up.weight)",
            ".arg_ptr(up.weight_scale)",
            ".arg_f32(up.weight_scale_2)",
            ".arg_ptr(output_first)",
            ".arg_ptr(output_second)",
            ".arg_u32(u32::from(fuse_silu))",
            ".arg_u32(m)",
            ".arg_u32(n)",
            ".arg_u32(k)",
        ];
        let mut cursor = 0;
        for argument in ordered {
            let relative = body[cursor..]
                .find(argument)
                .unwrap_or_else(|| panic!("missing or misordered dual argument {argument}"));
            cursor += relative + argument.len();
        }

        let host = include_str!("dense_ffn.rs");
        let dispatch_start = host.find("let pipe_requested =").unwrap();
        let dispatch = &host[dispatch_start..];
        let call = dispatch.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let call_body = &dispatch[call..];
        let null_output = call_body.find("DevicePtr::NULL").unwrap();
        let fused = call_body.find("true,").unwrap();
        let launch_end = call_body.find(")?;").unwrap();
        assert!(null_output < fused && fused < launch_end);
    }
}

#[cfg(test)]
mod e2m1_prefill_reuse_tests {
    use super::{
        E2M1_KMAJOR_M256_MIN_M, E2m1KmajorKernel, E2m1PrefillRoute, E2m1Scope, E2m1SelectedKernel,
        e2m1_kmajor_kernel, e2m1_kmajor_projection_shape, e2m1_prefill_route, e2m1_selected_kernel,
        validated_e2m1_checkpoint_scales,
    };

    #[test]
    fn base_w4a4_selector_is_full_authoritative_and_fail_closed() {
        assert_eq!(
            e2m1_prefill_route(false, false, true, false, false),
            E2m1PrefillRoute::Disabled
        );
        assert_eq!(
            e2m1_prefill_route(true, false, false, false, false),
            E2m1PrefillRoute::Ineligible
        );
        assert_eq!(
            e2m1_prefill_route(true, false, true, true, false),
            E2m1PrefillRoute::Complete(E2m1Scope::Full)
        );
        assert_eq!(
            e2m1_prefill_route(false, true, true, false, true),
            E2m1PrefillRoute::Complete(E2m1Scope::DownOnly)
        );
        assert_eq!(
            e2m1_prefill_route(true, false, true, false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full)
        );
        assert_eq!(
            e2m1_prefill_route(false, true, true, true, false),
            E2m1PrefillRoute::Missing(E2m1Scope::DownOnly)
        );
        assert_eq!(
            e2m1_prefill_route(true, true, true, false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full),
            "full must never downgrade to a ready down-only route"
        );
    }

    #[test]
    fn selected_w4a4_kernel_names_the_actual_m2048_boundary() {
        assert_eq!(
            e2m1_selected_kernel(false, true, 8192),
            E2m1SelectedKernel::RowMajorM64
        );
        assert_eq!(
            e2m1_selected_kernel(true, true, 2047),
            E2m1SelectedKernel::KmajorM128
        );
        assert_eq!(
            e2m1_selected_kernel(true, true, 2048),
            E2m1SelectedKernel::KmajorM256
        );
        assert_eq!(
            e2m1_selected_kernel(true, false, 8192),
            E2m1SelectedKernel::KmajorM128
        );
    }

    #[test]
    fn base_w4a4_missing_route_precedes_projection_and_marker_is_actual_only() {
        let source = include_str!("dense_ffn.rs");
        let select = source.find("let e2m1_route =").expect("W4A4 selector");
        let projection = source[select..]
            .find("if e2m1_fast_path {")
            .map(|offset| select + offset)
            .expect("W4A4 projection");
        let selection = &source[select..projection];
        assert!(selection.contains("E2m1PrefillRoute::Missing(E2m1Scope::Full)"));
        assert!(selection.contains("E2m1PrefillRoute::Missing(E2m1Scope::DownOnly)"));
        assert!(selection.contains("before gate projection"));

        let log_start = source.find("fn log_e2m1_route_once").expect("W4A4 logger");
        // Inspect this logger, not unrelated helpers inserted after it.
        // Inner blocks are indented; this is its top-level closing brace.
        let log_end = source[log_start..]
            .find("\n}\n")
            .map(|offset| log_start + offset + 3)
            .expect("W4A4 logger boundary");
        let logger = &source[log_start..log_end];
        assert!(logger.contains("ENGAGED ATLAS_E2M1_PREFILL"));
        for field in [
            "scope={scope}",
            "gate_up={gate_up}",
            "e2m1_kernel={kernel}",
            "activation_scale={activation_scale}",
            "down_input={down_input}",
            "activation={activation}",
            "M={}",
            "H={}",
            "intermediate={}",
        ] {
            assert!(logger.contains(field), "missing actual field {field}");
        }
        assert!(!logger.contains("requested"));
        assert!(!logger.contains("has_e2m1"));
    }

    #[test]
    fn kmajor_m128_accepts_only_native_ffn_tile_geometry() {
        assert!(e2m1_kmajor_projection_shape(8192, 17408, 5120));
        assert!(e2m1_kmajor_projection_shape(8192, 5120, 17408));
        assert!(!e2m1_kmajor_projection_shape(127, 17408, 5120));
        assert!(!e2m1_kmajor_projection_shape(128, 17409, 5120));
        assert!(!e2m1_kmajor_projection_shape(128, 17408, 5119));
    }

    #[test]
    fn kmajor_kernel_stages_transformed_weights_for_eight_row_warps() {
        let source =
            include_str!("../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
        let start = source
            .find("void nvfp4_nvfp4_gemm_kmajor_m128(")
            .expect("K-major W4A4 kernel is missing");
        let end = source[start..]
            .find("// K-major M256 shadow.")
            .map(|offset| start + offset)
            .expect("K-major kernel boundary is missing");
        let body = &source[start..end];

        assert!(source.contains("__launch_bounds__(256)\nvoid nvfp4_nvfp4_gemm_kmajor_m128"));
        assert!(body.contains("const unsigned char* __restrict__ B_packed_t"));
        assert!(body.contains("const unsigned char* __restrict__ B_scale_t"));
        assert!(body.contains("B_packed_t[(unsigned long long)gk_byte * N + gn]"));
        assert!(body.contains("B_scale_t[(unsigned long long)gg * N + gn]"));
        assert!(body.contains("smem_Bp_km[2][N_TILE_LG][K_STEP / 2]"));
        assert!(body.contains("smem_Bs_km[2][N_TILE_LG][K_STEP / GROUP_SIZE]"));
        assert!(body.contains("warp_m_offset = warp_id * 16"));
        assert!(body.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale"));
        assert!(body.contains("acc[nt][0] * scale2_ab"));
        assert_eq!(body.matches("__syncthreads();").count(), 2);
    }

    #[test]
    fn kmajor_m256_routes_only_at_the_saturated_long_prefill_threshold() {
        assert_eq!(E2M1_KMAJOR_M256_MIN_M, 2048);
        for m in [0, 127, 128, 256, 1024, 2047] {
            assert_eq!(e2m1_kmajor_kernel(m, true), E2m1KmajorKernel::M128);
        }
        for m in [2048, 2049, 8192, u32::MAX] {
            assert_eq!(e2m1_kmajor_kernel(m, true), E2m1KmajorKernel::M256);
            assert_eq!(e2m1_kmajor_kernel(m, false), E2m1KmajorKernel::M128);
        }
    }

    #[test]
    fn kmajor_m256_reuses_one_weight_tile_across_sixteen_row_warps() {
        let source =
            include_str!("../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
        let start = source
            .find("void nvfp4_nvfp4_gemm_kmajor_m256(")
            .expect("K-major M256 W4A4 kernel is missing");
        let end = source[start..]
            .find("// FIX PATH 3 LANDED")
            .map(|offset| start + offset)
            .expect("K-major M256 kernel boundary is missing");
        let body = &source[start..end];

        assert!(source.contains("__launch_bounds__(512, 1)\nvoid nvfp4_nvfp4_gemm_kmajor_m256"));
        assert!(body.contains("cta_m = blockIdx.y * 256u"));
        assert!(body.contains("smem_Ap_km256[2][256][K_STEP / 2]"));
        assert!(body.contains("smem_Bp_km256[2][N_TILE_LG][K_STEP / 2]"));
        assert!(body.contains("smem_Bs_km256[2][N_TILE_LG][K_STEP / GROUP_SIZE]"));
        assert!(body.contains("if (threadIdx.x < 256)"));
        assert!(body.contains("warp_m_offset = warp_id * 16"));
        assert!(body.contains("B_packed_t[(unsigned long long)gk_byte * N + gn]"));
        assert!(body.contains("B_scale_t[(unsigned long long)gg * N + gn]"));
        assert!(body.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale"));
        assert!(body.contains("acc[nt][0] * scale2_ab"));
        assert_eq!(body.matches("__syncthreads();").count(), 2);
    }

    #[test]
    fn kmajor_dispatch_fails_closed_and_uses_only_transformed_weights() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("let e2m1_kmajor_requested")
            .expect("K-major request gate is missing");
        let end = source[start..]
            .find("if v2_fast_path {")
            .map(|offset| start + offset)
            .expect("K-major dispatch boundary is missing");
        let body = &source[start..end];

        assert!(body.contains("self.has_e2m1_kmajor_runtime()"));
        assert!(body.contains("self.has_e2m1_kmajor_m256_runtime()"));
        assert!(body.contains("self.has_transposed_ffn()"));
        assert!(body.contains("e2m1_kmajor_projection_shape(m, inter, h)"));
        assert!(body.contains("self.gate_proj_t.as_ref().unwrap()"));
        assert!(body.contains("self.up_proj_t.as_ref().unwrap()"));
        assert!(body.contains("self.down_proj_t.as_ref().unwrap()"));
        assert!(source.contains("ops::nvfp4_nvfp4_gemm_kmajor_m128("));
        assert!(source.contains("ops::nvfp4_nvfp4_gemm_kmajor_m256("));

        let wrapper = include_str!("ops/gemm_dense.rs");
        assert!(wrapper.contains("pub fn nvfp4_nvfp4_gemm_kmajor_m256("));
        assert!(wrapper.contains(".grid([n / 128, div_ceil(m, 256), 1])"));
        assert!(wrapper.contains(".block([512, 1, 1])"));
    }

    #[test]
    fn static_scale_matches_merged_gate_up_contract_and_rejects_bad_metadata() {
        let scales = validated_e2m1_checkpoint_scales(0.001, 0.00125, 0.002).unwrap();
        assert_eq!(scales.gate_up.to_bits(), 0.00125_f32.to_bits());
        assert_eq!(scales.down.to_bits(), 0.002_f32.to_bits());

        for bad in [0.0, -0.1, f32::INFINITY, f32::NAN] {
            assert!(validated_e2m1_checkpoint_scales(bad, 0.001, 0.002).is_err());
            assert!(validated_e2m1_checkpoint_scales(0.001, bad, 0.002).is_err());
            assert!(validated_e2m1_checkpoint_scales(0.001, 0.002, bad).is_err());
        }
    }

    #[test]
    fn static_scale_bypasses_runtime_absmax_and_sync_before_quantization() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("fn prepare_e2m1_input(")
            .expect("W4A4 activation preparation is missing");
        let end = source[start..]
            .find("fn forward_e2m1_prepared(")
            .map(|offset| start + offset)
            .expect("prepared W4A4 GEMM boundary is missing");
        let body = &source[start..end];

        let fixed = body
            .find("if let Some(scale2) = checkpoint_scale2")
            .expect("checkpoint-static branch is missing");
        let dynamic = body
            .find("ctx.gpu.synchronize(stream)?")
            .expect("dynamic absmax fallback is missing");
        let quantize = body
            .find("ops::quantize_bf16_to_nvfp4(")
            .expect("shared on-device quantizer is missing");
        assert!(fixed < dynamic && dynamic < quantize);
    }

    #[test]
    fn full_w4a4_gate_and_up_share_one_prepared_activation() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("if e2m1_fast_path {")
            .expect("full W4A4 dispatch is missing");
        let end = source[start..]
            .find("if e2m1_down_only_path {")
            .map(|offset| start + offset)
            .expect("down-only dispatch boundary is missing");
        let branch = &source[start..end];

        assert_eq!(branch.matches("self.prepare_e2m1_input(").count(), 1);
        assert_eq!(branch.matches("self.forward_e2m1_prepared(").count(), 3);
        assert_eq!(branch.matches("self.forward_e2m1_proj(").count(), 1);
        assert_eq!(branch.matches(".prepare_e2m1_silu_input(").count(), 1);
        assert!(branch.contains("&self.weights.gate_proj"));
        assert!(branch.contains("&self.weights.up_proj"));
        assert!(branch.contains("&self.weights.down_proj"));
    }

    #[test]
    fn fused_silu_quant_preserves_bf16_boundary_and_fails_closed() {
        let cuda = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
        let start = cuda
            .find("void quantize_silu_mul_bf16_to_nvfp4(")
            .expect("fused SwiGLU quantizer is missing");
        let body = &cuda[start..];
        assert!(cuda.contains("return __bfloat162float(__float2bfloat16(g * sigmoid_g * u));"));
        assert!(body.contains("float_to_fp8_e4m3(fp8_float)"));
        assert!(body.contains("quantize_e2m1(v0)"));
        assert!(body.contains("quantize_e2m1(v1)"));

        let rust = include_str!("dense_ffn.rs");
        assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_STATIC_SCALE=1"));
        assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 is valid only for SiLU-gated FFNs"));
        assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 requires quantize_silu_mul_bf16_to_nvfp4"));
    }

    #[test]
    fn kmajor_transform_and_tile_scatter_reconstruct_parent_layout() {
        const N: usize = 256;
        const K: usize = 192;
        let packed_k = K / 2;
        let scale_k = K / 16;

        let packed_parent: Vec<u32> = (0..N * packed_k).map(|index| index as u32).collect();
        let scale_parent: Vec<u32> = (0..N * scale_k)
            .map(|index| 1_000_000 + index as u32)
            .collect();
        let mut packed_t = vec![0_u32; packed_parent.len()];
        let mut scale_t = vec![0_u32; scale_parent.len()];
        for column in 0..N {
            for k_byte in 0..packed_k {
                packed_t[k_byte * N + column] = packed_parent[column * packed_k + k_byte];
            }
            for group in 0..scale_k {
                scale_t[group * N + column] = scale_parent[column * scale_k + group];
            }
        }

        for n_base in [0, 128] {
            for k_base in [0, 64, 128] {
                for local_n in 0..128 {
                    let column = n_base + local_n;
                    for local_byte in 0..32 {
                        let k_byte = k_base / 2 + local_byte;
                        let staged = packed_t[k_byte * N + column];
                        assert_eq!(staged, packed_parent[column * packed_k + k_byte]);
                    }
                    for local_group in 0..4 {
                        let group = k_base / 16 + local_group;
                        let staged = scale_t[group * N + column];
                        assert_eq!(staged, scale_parent[column * scale_k + group]);
                    }
                }
            }
        }
    }
}
