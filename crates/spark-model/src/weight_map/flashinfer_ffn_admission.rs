// SPDX-License-Identifier: AGPL-3.0-only

//! Host-only admission plan for a future Qwen3.8 FlashInfer FFN route.
//!
//! The native ABI owns one global FP32 scale per GEMM. Therefore gate and up
//! may share one merged GEMM only when they use bit-identical activation
//! scales and their independently computed `input_scale * weight_scale_2`
//! values are bit-identical. This module never rescales E4M3 block scales.

use anyhow::{Context, Result, ensure};

use super::cutlass_scale_layout::{
    CUTLASS_SCALE_ROW_TILE, NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4,
    interleave_nvfp4_scales_128x4,
};
use super::modelopt_scale_admission::{
    AdmittedModeloptScale, ModeloptScaleProjection, ModeloptScaleSource,
};

pub const QWEN38_HIDDEN: usize = 5_120;
pub const QWEN38_INTERMEDIATE: usize = 17_408;

/// Device bytes retained by one merged gate/up operand: exact packed E2M1
/// bytes, exact physical E4M3 scales, and one shared FP32 alpha.
pub const fn qwen38_merged_gate_up_retained_bytes() -> usize {
    2 * QWEN38_INTERMEDIATE * QWEN38_HIDDEN / 2
        + 2 * QWEN38_INTERMEDIATE * QWEN38_HIDDEN / NVFP4_GROUP_SIZE
        + std::mem::size_of::<f32>()
}

/// Additional construction bytes relative to the prior separate gate/up
/// FlashInfer route, whose two physical-scale copies and two alphas are
/// replaced by the merged physical scales and one shared alpha.
pub const fn qwen38_merged_gate_up_additional_bytes() -> usize {
    2 * QWEN38_INTERMEDIATE * QWEN38_HIDDEN / 2 - std::mem::size_of::<f32>()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen38FfnProjection {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen38FfnSource {
    pub layer: usize,
    pub projection: Qwen38FfnProjection,
}

/// Exact retained scalar receipt; no float equality is used for identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetainedFfnScalar {
    source: Qwen38FfnSource,
    value_bits: u32,
}

impl RetainedFfnScalar {
    pub fn source(self) -> Qwen38FfnSource {
        self.source
    }

    pub fn value_bits(self) -> u32 {
        self.value_bits
    }

    pub fn value(self) -> f32 {
        f32::from_bits(self.value_bits)
    }
}

/// Borrowed checkpoint material for one ordinary dense FFN projection.
pub struct Qwen38FfnCheckpointProjection<'a> {
    pub source: Qwen38FfnSource,
    /// ModelOpt packed E2M1 `[N,K/2]`, in logical output-row order.
    pub packed_weight: &'a [u8],
    /// ModelOpt E4M3 `[I,H/16]` bytes in logical output-row order.
    pub logical_weight_scales: &'a [u8],
    pub input_scale: AdmittedModeloptScale,
    pub weight_scale_2_le_bytes: [u8; 4],
}

/// Primary production representation: checkpoint-packed bytes are borrowed,
/// while the exact physical scale transform is owned once per projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedQwen38FfnProjection<'a> {
    source: Qwen38FfnSource,
    n: usize,
    k: usize,
    packed_weight: &'a [u8],
    physical_weight_scales: Vec<u8>,
    input_scale: AdmittedModeloptScale,
    weight_scale_2: RetainedFfnScalar,
    alpha: RetainedFfnScalar,
}

impl<'a> AdmittedQwen38FfnProjection<'a> {
    pub fn source(&self) -> Qwen38FfnSource {
        self.source
    }

    pub fn packed_weight(&self) -> &'a [u8] {
        self.packed_weight
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn physical_weight_scales(&self) -> &[u8] {
        &self.physical_weight_scales
    }

    pub fn input_scale(&self) -> AdmittedModeloptScale {
        self.input_scale
    }

    pub fn weight_scale_2(&self) -> RetainedFfnScalar {
        self.weight_scale_2
    }

    pub fn alpha(&self) -> RetainedFfnScalar {
        self.alpha
    }
}

/// Owned, byte-exact gate-then-up material and its complete scalar receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedMergedGateUp {
    layer: usize,
    packed_weight: Vec<u8>,
    physical_weight_scales: Vec<u8>,
    input_scales: [AdmittedModeloptScale; 2],
    weight_scale_2: [RetainedFfnScalar; 2],
    alpha: [RetainedFfnScalar; 2],
}

impl AdmittedMergedGateUp {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn packed_weight(&self) -> &[u8] {
        &self.packed_weight
    }

    pub fn physical_weight_scales(&self) -> &[u8] {
        &self.physical_weight_scales
    }

    pub fn input_scales(&self) -> [AdmittedModeloptScale; 2] {
        self.input_scales
    }

    pub fn weight_scale_2(&self) -> [RetainedFfnScalar; 2] {
        self.weight_scale_2
    }

    pub fn alpha(&self) -> [RetainedFfnScalar; 2] {
        self.alpha
    }

    pub fn shared_alpha_bits(&self) -> u32 {
        self.alpha[0].value_bits
    }
}

fn checked_product(label: &str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} byte-length overflow: {left} * {right}"))
}

fn append_exact(dst: &mut Vec<u8>, first: &[u8], second: &[u8]) -> Result<()> {
    let len = first
        .len()
        .checked_add(second.len())
        .context("merged byte-length overflow")?;
    dst.try_reserve_exact(len)
        .context("unable to allocate merged checkpoint bytes")?;
    dst.extend_from_slice(first);
    dst.extend_from_slice(second);
    Ok(())
}

fn expected_input_source(source: Qwen38FfnSource) -> ModeloptScaleSource {
    let projection = match source.projection {
        Qwen38FfnProjection::Gate => ModeloptScaleProjection::FfnGate,
        Qwen38FfnProjection::Up => ModeloptScaleProjection::FfnUp,
        Qwen38FfnProjection::Down => ModeloptScaleProjection::FfnDown,
    };
    ModeloptScaleSource {
        layer: source.layer,
        projection,
    }
}

fn projection_shape(projection: Qwen38FfnProjection) -> (usize, usize) {
    match projection {
        Qwen38FfnProjection::Gate | Qwen38FfnProjection::Up => (QWEN38_INTERMEDIATE, QWEN38_HIDDEN),
        Qwen38FfnProjection::Down => (QWEN38_HIDDEN, QWEN38_INTERMEDIATE),
    }
}

fn admit_scalar(source: Qwen38FfnSource, bytes: [u8; 4]) -> Result<RetainedFfnScalar> {
    let value = f32::from_le_bytes(bytes);
    ensure!(
        value.is_finite() && value > 0.0,
        "FFN checkpoint scalar must be finite and positive"
    );
    Ok(RetainedFfnScalar {
        source,
        value_bits: u32::from_le_bytes(bytes),
    })
}

fn alpha_receipt(
    source: Qwen38FfnSource,
    input_scale: AdmittedModeloptScale,
    weight_scale_2: RetainedFfnScalar,
) -> Result<RetainedFfnScalar> {
    let value = input_scale.value() * weight_scale_2.value();
    ensure!(
        value.is_finite() && value > 0.0,
        "FFN input_scale * weight_scale_2 must be finite and positive"
    );
    Ok(RetainedFfnScalar {
        source,
        value_bits: value.to_bits(),
    })
}

pub fn admit_qwen38_ffn_projection<'a>(
    expected_layer: usize,
    expected_projection: Qwen38FfnProjection,
    projection: Qwen38FfnCheckpointProjection<'a>,
) -> Result<AdmittedQwen38FfnProjection<'a>> {
    let expected_source = Qwen38FfnSource {
        layer: expected_layer,
        projection: expected_projection,
    };
    ensure!(
        projection.source == expected_source,
        "FFN checkpoint projection source is misattributed"
    );
    ensure!(
        projection.input_scale.source() == expected_input_source(expected_source),
        "FFN input-scale source does not match its checkpoint projection"
    );

    let (n, k) = projection_shape(expected_projection);
    let packed_len = checked_product("projection packed weight", n, k / 2)?;
    ensure!(
        projection.packed_weight.len() == packed_len,
        "Qwen3.8 FFN packed projection requires {packed_len} bytes, got {}",
        projection.packed_weight.len()
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
        "FFN physical scale layout failed exact round-trip receipt"
    );

    let weight_scale_2 = admit_scalar(expected_source, projection.weight_scale_2_le_bytes)?;
    let alpha = alpha_receipt(expected_source, projection.input_scale, weight_scale_2)?;
    Ok(AdmittedQwen38FfnProjection {
        source: expected_source,
        n,
        k,
        packed_weight: projection.packed_weight,
        physical_weight_scales,
        input_scale: projection.input_scale,
        weight_scale_2,
        alpha,
    })
}

/// Merge gate then up without changing packed nibbles or E4M3 scale bytes.
pub fn admit_qwen38_merged_gate_up(
    layer: usize,
    gate: Qwen38FfnCheckpointProjection<'_>,
    up: Qwen38FfnCheckpointProjection<'_>,
) -> Result<AdmittedMergedGateUp> {
    let gate = admit_qwen38_ffn_projection(layer, Qwen38FfnProjection::Gate, gate)?;
    let up = admit_qwen38_ffn_projection(layer, Qwen38FfnProjection::Up, up)?;

    ensure!(
        gate.input_scale().value_bits() == up.input_scale().value_bits(),
        "merged gate/up requires bit-identical input scales"
    );
    ensure!(
        gate.alpha().value_bits == up.alpha().value_bits,
        "merged gate/up requires bit-identical global alpha"
    );

    let mut packed_weight = Vec::new();
    append_exact(&mut packed_weight, gate.packed_weight(), up.packed_weight())?;

    let gate_logical_scales = deinterleave_nvfp4_scales_128x4(
        gate.physical_weight_scales(),
        &[QWEN38_INTERMEDIATE, QWEN38_HIDDEN / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )?;
    let up_logical_scales = deinterleave_nvfp4_scales_128x4(
        up.physical_weight_scales(),
        &[QWEN38_INTERMEDIATE, QWEN38_HIDDEN / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )?;
    let mut merged_logical_scales = Vec::new();
    append_exact(
        &mut merged_logical_scales,
        &gate_logical_scales,
        &up_logical_scales,
    )?;
    let groups = QWEN38_HIDDEN / NVFP4_GROUP_SIZE;
    let physical_weight_scales = interleave_nvfp4_scales_128x4(
        &merged_logical_scales,
        &[2 * QWEN38_INTERMEDIATE, groups],
        NVFP4_GROUP_SIZE,
    )?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(
            &physical_weight_scales,
            &[2 * QWEN38_INTERMEDIATE, groups],
            NVFP4_GROUP_SIZE,
        )? == merged_logical_scales,
        "merged gate/up physical scales failed exact round-trip receipt"
    );

    Ok(AdmittedMergedGateUp {
        layer,
        packed_weight,
        physical_weight_scales,
        input_scales: [gate.input_scale(), up.input_scale()],
        weight_scale_2: [gate.weight_scale_2(), up.weight_scale_2()],
        alpha: [gate.alpha(), up.alpha()],
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen38FfnOperation {
    Gate,
    Up,
    MergedGateUp,
    Down,
}

/// Exact native-ABI shape and tactic frozen by the existing raw GB10 gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen38FfnLaunchPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub tactic: usize,
    pub workspace_bytes: usize,
    /// False means shape/layout admission is known but production must reject.
    pub performance_qualified: bool,
}

/// Extended M ladder, opt-in via `ATLAS_FLASHINFER_FFN_EXTRA_M=1`.
///
/// The frozen table below qualifies exactly M=2079 and M=8192, so every other
/// prompt-chunk length silently falls back to the ordinary route. That is the
/// dominant real-world cost of this feature: with `MAX_PREFILL_TOKENS=8192` a
/// 16,379-token prompt splits into 8192 + 8187, and the 8,187-row tail — five
/// rows short of qualifying — drags measured prefill to 453 tok/s, *below* the
/// 521 tok/s of a 12.4k prompt. vLLM dispatches FlashInfer at arbitrary M; this
/// ladder narrows that gap.
///
/// N and K are fixed by the model, and M is only a row count: in a GEMM each
/// output row depends solely on its own input row, so a tactic that is correct
/// at one M is correct at another. Tactic choice is a performance decision.
/// **That argument still has to be paid for empirically** — every M added here
/// must be shown byte-identical to the ordinary route before use, which is what
/// the flag gate is for. Nothing here changes default behaviour.
/// Any prefill chunk in `[QWEN38_FFN_EXTRA_M_MIN, QWEN38_FFN_EXTRA_M_MAX]`.
///
/// This started as a fixed list (1024/2048/4096/6144/7168/8187) which fixed the
/// specific 8,187-row tail but left the underlying defect: prompt length still
/// decided whether the tail was fast, so measured prefill swung between 453 and
/// 762 tok/s on the same build. Chunks are `min(remaining, MAX_PREFILL_TOKENS)`,
/// i.e. essentially arbitrary, so a list can never cover them.
///
/// Upper bound is the largest frozen entry (8192); chunking never exceeds
/// `MAX_PREFILL_TOKENS`, and the preflight activation-scratch charge is sized
/// for that maximum. Lower bound is one CUTLASS tile — below 128 rows the tile
/// is mostly padding and the ordinary route is the better choice anyway, so
/// there is nothing to win and a tail-shape risk to take.
const QWEN38_FFN_EXTRA_M_MIN: usize = 128;
const QWEN38_FFN_EXTRA_M_MAX: usize = 8_192;

pub fn qwen38_ffn_extra_m_enabled() -> bool {
    std::env::var("ATLAS_FLASHINFER_FFN_EXTRA_M").ok().as_deref() == Some("1")
}

/// True when `m` is inside the opt-in extended range.
const fn qwen38_ffn_extra_m_covers(m: usize) -> bool {
    m >= QWEN38_FFN_EXTRA_M_MIN && m <= QWEN38_FFN_EXTRA_M_MAX
}

/// Tactic mirrors the frozen table's split: the large-M entries use tactic 2 for
/// the gate/up families and tactic 4 for down; small M uses tactic 4 throughout.
const fn qwen38_ffn_extra_tactic(operation: Qwen38FfnOperation, m: usize) -> usize {
    match operation {
        Qwen38FfnOperation::Down => 4,
        _ if m >= 4_096 => 2,
        _ => 4,
    }
}

/// Return the frozen candidate table, including deliberately blocked entries.
pub fn qwen38_ffn_launch_candidate(
    operation: Qwen38FfnOperation,
    m: usize,
) -> Result<Qwen38FfnLaunchPlan> {
    if qwen38_ffn_extra_m_enabled()
        && qwen38_ffn_extra_m_covers(m)
        && !matches!(m, 2_079 | 8_192)
    {
        let tactic = qwen38_ffn_extra_tactic(operation, m);
        let (n, k) = match operation {
            Qwen38FfnOperation::Gate | Qwen38FfnOperation::Up => {
                (QWEN38_INTERMEDIATE, QWEN38_HIDDEN)
            }
            Qwen38FfnOperation::MergedGateUp => (2 * QWEN38_INTERMEDIATE, QWEN38_HIDDEN),
            Qwen38FfnOperation::Down => (QWEN38_HIDDEN, QWEN38_INTERMEDIATE),
        };
        return Ok(Qwen38FfnLaunchPlan {
            m,
            n,
            k,
            tactic,
            workspace_bytes: 0,
            performance_qualified: true,
        });
    }
    let (n, k, tactic, performance_qualified) = match (operation, m) {
        (Qwen38FfnOperation::Gate | Qwen38FfnOperation::Up, 2_079) => {
            (QWEN38_INTERMEDIATE, QWEN38_HIDDEN, 4, true)
        }
        (Qwen38FfnOperation::Gate | Qwen38FfnOperation::Up, 8_192) => {
            (QWEN38_INTERMEDIATE, QWEN38_HIDDEN, 2, true)
        }
        (Qwen38FfnOperation::MergedGateUp, 2_079) => {
            (2 * QWEN38_INTERMEDIATE, QWEN38_HIDDEN, 4, true)
        }
        (Qwen38FfnOperation::MergedGateUp, 8_192) => {
            (2 * QWEN38_INTERMEDIATE, QWEN38_HIDDEN, 2, true)
        }
        // Real-checkpoint tactic-4 gate: exact BF16 parity, workspace=0,
        // parent 4.0265 ms vs candidate 1.4616 ms at M2079.
        (Qwen38FfnOperation::Down, 2_079) => (QWEN38_HIDDEN, QWEN38_INTERMEDIATE, 4, true),
        (Qwen38FfnOperation::Down, 8_192) => (QWEN38_HIDDEN, QWEN38_INTERMEDIATE, 4, true),
        _ => {
            anyhow::bail!("unqualified Qwen3.8 FlashInfer FFN shape: operation={operation:?} M={m}")
        }
    };
    Ok(Qwen38FfnLaunchPlan {
        m,
        n,
        k,
        tactic,
        workspace_bytes: 0,
        performance_qualified,
    })
}

/// Select only a candidate with real-checkpoint timing for this exact tactic.
pub fn select_qwen38_ffn_launch(
    operation: Qwen38FfnOperation,
    m: usize,
) -> Result<Qwen38FfnLaunchPlan> {
    let plan = qwen38_ffn_launch_candidate(operation, m)?;
    ensure!(
        plan.performance_qualified,
        "Qwen3.8 FlashInfer FFN tactic is selected but performance-unqualified"
    );
    Ok(plan)
}

/// Owned zero-padded logical and 128x4 physical activation-scale allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddedActivationScales {
    pub logical_rows: usize,
    pub padded_rows: usize,
    pub groups: usize,
    logical_padded: Vec<u8>,
    physical: Vec<u8>,
}

impl PaddedActivationScales {
    pub fn logical_padded(&self) -> &[u8] {
        &self.logical_padded
    }

    pub fn physical(&self) -> &[u8] {
        &self.physical
    }
}

/// Allocate explicit zero padding and convert logical activation scales.
pub fn allocate_qwen38_activation_scales(
    launch: Qwen38FfnLaunchPlan,
    logical: &[u8],
) -> Result<PaddedActivationScales> {
    ensure!(
        launch
            == select_qwen38_ffn_launch(
                if launch.n == QWEN38_INTERMEDIATE && launch.k == QWEN38_HIDDEN {
                    Qwen38FfnOperation::Gate
                } else if launch.n == 2 * QWEN38_INTERMEDIATE && launch.k == QWEN38_HIDDEN {
                    Qwen38FfnOperation::MergedGateUp
                } else if launch.n == QWEN38_HIDDEN && launch.k == QWEN38_INTERMEDIATE {
                    Qwen38FfnOperation::Down
                } else {
                    anyhow::bail!(
                        "activation-scale allocation requires an admitted FFN launch plan"
                    )
                },
                launch.m,
            )?,
        "activation-scale allocation requires an exact admitted FFN launch plan"
    );
    ensure!(
        launch.k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FFN K must be divisible by the NVFP4 group size"
    );
    let groups = launch.k / NVFP4_GROUP_SIZE;
    let expected = checked_product("logical activation scales", launch.m, groups)?;
    ensure!(
        logical.len() == expected,
        "logical activation scales require {expected} bytes, got {}",
        logical.len()
    );
    let padded_rows = launch
        .m
        .checked_add(CUTLASS_SCALE_ROW_TILE - 1)
        .context("activation row padding overflow")?
        / CUTLASS_SCALE_ROW_TILE
        * CUTLASS_SCALE_ROW_TILE;
    let padded_len = checked_product("padded activation scales", padded_rows, groups)?;
    let mut logical_padded = Vec::new();
    logical_padded
        .try_reserve_exact(padded_len)
        .context("unable to allocate padded activation scales")?;
    logical_padded.resize(padded_len, 0);
    logical_padded[..logical.len()].copy_from_slice(logical);
    let physical =
        interleave_nvfp4_scales_128x4(&logical_padded, &[padded_rows, groups], NVFP4_GROUP_SIZE)?;
    Ok(PaddedActivationScales {
        logical_rows: launch.m,
        padded_rows,
        groups,
        logical_padded,
        physical,
    })
}

#[cfg(test)]
#[path = "flashinfer_ffn_admission_tests.rs"]
mod tests;
