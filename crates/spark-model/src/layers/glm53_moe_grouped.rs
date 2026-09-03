// SPDX-License-Identifier: AGPL-3.0-only

//! Grouped GLM-5.3 MoE expert matmul: one launch over (rows, top_k).
//!
//! The serial seam in `glm53_moe_serial` drains the pipeline once per MoE layer
//! to read the eight route ids to the HOST, then issues one launch per slot at
//! a host-computed `base + id*expert_bytes`. That drain measured 91.5% of
//! decode wall time. The grouped kernel reads the route ids from DEVICE memory
//! and does the multiply-add itself, so nothing has to synchronize to learn
//! them, and it groups over (row, slot) rather than slot alone so prefill and
//! speculative verify get the same win rather than only decode.
//!
//! THE SERIAL SEAM IS NOT DELETED AND MUST NOT BE. It stays as the permanent
//! reference implementation behind `ATLAS_GLM53_MOE`, which defaults to it.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::gguf::GgmlType;

use super::ops::{GgmlIqBuffer, GgmlIqMmqPlan};

const MMQ_TILE_COLUMNS: u32 = 128;
const MMQ_TILE_ROWS: u32 = 128;
const QUANTIZE_THREADS: u32 = 128;
/// The kernel ABI passes every extent as a signed 32-bit int.
const ABI_MAX: u64 = i32::MAX as u64;

/// Which MoE path the walk takes. `serial-reference` is the default and is a
/// permanent reference implementation, not a stepping stone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53MoePath {
    SerialReference,
    Grouped,
}

impl Glm53MoePath {
    pub const ENV: &'static str = "ATLAS_GLM53_MOE";

    /// Read the gate. Unset is the reference path; an unrecognised value is an
    /// ERROR rather than a silent fall back to the default, because a typo
    /// that quietly selects the slow correct path would be scored as a
    /// grouped-path measurement.
    pub fn from_env() -> Result<Self> {
        match std::env::var(Self::ENV) {
            Err(_) => Ok(Self::SerialReference),
            Ok(value) => Self::parse(&value),
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "serial-reference" => Ok(Self::SerialReference),
            "grouped" => Ok(Self::Grouped),
            other => bail!(
                "{} must be `grouped` or `serial-reference`, got `{other}`",
                Self::ENV
            ),
        }
    }
}

/// A packed expert bank, re-checked for uniform stride AT THE LAUNCH SITE.
///
/// The grouped kernel's whole premise is `base + id*expert_bytes`, so the
/// uniformity is not a property to inherit from the loader by assumption. It
/// is re-derived here from the type and geometry and re-checked against the
/// bank's own byte extent, so a future layout change (a per-expert alignment
/// pad, a mixed-type bank, a bank that grew a header) fails loudly at the
/// launch rather than reading a neighbouring expert's weights and producing
/// fluent, wrong output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53GroupedBank {
    pub base: DevicePtr,
    pub kind: GgmlType,
    pub inner: u32,
    pub columns: u32,
    pub experts: u32,
    pub expert_bytes: usize,
    pub total_bytes: usize,
}

impl Glm53GroupedBank {
    pub fn new(
        bank: GgmlIqBuffer,
        kind: GgmlType,
        inner: u32,
        columns: u32,
        experts: u32,
        expert_bytes: usize,
    ) -> Result<Self> {
        let (base, total_bytes) = (bank.ptr, bank.bytes);
        ensure!(base != DevicePtr::NULL, "GLM grouped MoE bank is null");
        ensure!(
            experts > 0 && inner > 0 && columns > 0,
            "GLM grouped MoE bank geometry must be non-zero, got \
             experts={experts} inner={inner} columns={columns}"
        );
        // THE UNIFORM-STRIDE GUARD. `expert_bytes` must be exactly what one
        // expert's MMQ plan says, and the bank must be exactly that many
        // times the expert count -- no header, no padding, no slack.
        let derived = GgmlIqMmqPlan::new(kind, 1, columns, inner)?.weight_bytes;
        ensure!(
            expert_bytes == derived,
            "GLM grouped MoE stride is not uniform: bank reports {expert_bytes} \
             bytes per expert, the [{inner}, {columns}] {kind:?} plan needs \
             {derived}. The grouped kernel addresses experts as \
             base + id*expert_bytes and cannot express a non-uniform bank."
        );
        let packed = expert_bytes
            .checked_mul(usize::try_from(experts)?)
            .context("GLM grouped MoE bank byte count overflow")?;
        ensure!(
            packed == total_bytes,
            "GLM grouped MoE bank is {total_bytes} bytes, {experts} experts at a \
             uniform {expert_bytes} need exactly {packed}. A remainder means the \
             bank is not densely packed and the stride is a fiction."
        );
        base.0
            .checked_add(u64::try_from(total_bytes)?)
            .context("GLM grouped MoE bank end address overflow")?;
        // The kernel takes `expert_bytes` as an `int`.
        ensure!(
            u64::try_from(expert_bytes)? <= ABI_MAX,
            "GLM grouped MoE expert stride {expert_bytes} exceeds the kernel ABI"
        );
        Ok(Self {
            base,
            kind,
            inner,
            columns,
            experts,
            expert_bytes,
            total_bytes,
        })
    }
}

/// One grouped launch: `rows * top_k` independent (row, slot) tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53GroupedMoePlan {
    pub bank: Glm53GroupedBank,
    pub rows: u32,
    pub top_k: u32,
    pub pairs: u32,
    /// Every pair runs the SAME single-expert MMQ shape; only the base pointer
    /// differs. Sharing one plan is what makes the grouping legal.
    pub tile: GgmlIqMmqPlan,
    pub activations: Glm53GroupedActivations,
    /// Ints of activation between consecutive pairs; zero when shared.
    pub activation_pair_stride: u32,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub route_bytes: usize,
}

/// Whether every (row, slot) pair reads the SAME activation or its own.
///
/// Gate and up are `Shared`: all top_k experts consume the identical token
/// hidden state. Down is `PerPair`: each slot consumes its own swiglu output.
/// Down must be expressible or the host still needs the route ids to address
/// the expert bank, the 3.91 ms drain survives, and the grouped path buys
/// nothing at all -- two of three matmuls is not two thirds of the win.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53GroupedActivations {
    Shared,
    PerPair,
}

impl Glm53GroupedMoePlan {
    pub fn new(bank: Glm53GroupedBank, rows: u32, top_k: u32) -> Result<Self> {
        Self::with_activations(bank, rows, top_k, Glm53GroupedActivations::Shared)
    }

    pub fn with_activations(
        bank: Glm53GroupedBank,
        rows: u32,
        top_k: u32,
        activations: Glm53GroupedActivations,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && top_k > 0,
            "GLM grouped MoE needs rows>0 and top_k>0, got rows={rows} top_k={top_k}"
        );
        ensure!(
            top_k <= bank.experts,
            "GLM grouped MoE top_k {top_k} exceeds the bank's {} experts",
            bank.experts
        );
        let pairs = rows
            .checked_mul(top_k)
            .context("GLM grouped MoE (row, slot) pair count overflow")?;
        // gridDim.z carries the pair, so it bounds the batch the same way the
        // tile grid bounds rows and columns.
        ensure!(
            u64::from(pairs) <= ABI_MAX,
            "GLM grouped MoE {pairs} pairs exceed the grid ABI"
        );
        let tile = GgmlIqMmqPlan::new(bank.kind, rows, bank.columns, bank.inner)?;
        let output_bytes = tile
            .output_bytes
            .checked_mul(usize::try_from(top_k)?)
            .context("GLM grouped MoE output byte count overflow")?;
        let route_bytes = usize::try_from(pairs)?
            .checked_mul(4)
            .context("GLM grouped MoE route byte count overflow")?;
        // The quantized activation block is `activation_bytes` of i32 words.
        // A per-pair layout holds one such block per pair, back to back.
        let words = u32::try_from(tile.activation_bytes / 4)
            .context("GLM grouped MoE activation stride exceeds the kernel ABI")?;
        let (activation_pair_stride, activation_bytes) = match activations {
            Glm53GroupedActivations::Shared => (0, tile.activation_bytes),
            Glm53GroupedActivations::PerPair => (
                words,
                tile.activation_bytes
                    .checked_mul(usize::try_from(pairs)?)
                    .context("GLM grouped MoE per-pair activation overflow")?,
            ),
        };
        // THE LAYOUT-AGREEMENT CHECK. The quantize that fills a per-pair buffer
        // runs at `rows = pairs`; the tile that reads it runs at `rows` per
        // pair. Two plans over one buffer, and a disagreement between them is
        // SILENT -- it yields plausible wrong numbers rather than an error,
        // which is the class of bug this project has paid for four times.
        //
        // `GgmlIqMmqPlan` lays activations out as `rows` contiguous blocks of
        // `inner_padded / Q8_BLOCK_VALUES`, exactly linear in rows with no
        // cross-row padding, so the two descriptions CAN agree. This asserts
        // that they do, by deriving the quantize plan from the same geometry
        // rather than trusting the arithmetic above.
        if activations == Glm53GroupedActivations::PerPair {
            let quantize = GgmlIqMmqPlan::new(bank.kind, pairs, bank.columns, bank.inner)?;
            ensure!(
                quantize.activation_bytes == activation_bytes
                    && quantize.inner_padded == tile.inner_padded,
                "GLM grouped MoE per-pair layout disagreement: the quantize at \
                 rows={pairs} produces {} bytes with K padded to {}, the tile \
                 addresses {activation_bytes} bytes with K padded to {}. The two \
                 must describe the same buffer.",
                quantize.activation_bytes,
                quantize.inner_padded,
                tile.inner_padded
            );
        }
        ensure!(
            tile.activation_bytes.is_multiple_of(4),
            "GLM grouped MoE activation block is {} bytes, not a whole number of \
             i32 words; a per-pair stride cannot address it",
            tile.activation_bytes
        );
        Ok(Self {
            bank,
            rows,
            top_k,
            pairs,
            tile,
            activations,
            activation_pair_stride,
            activation_bytes,
            output_bytes,
            route_bytes,
        })
    }
}

impl Glm53GroupedMoePlan {
    /// The plan the caller must use for the quantize that fills a per-pair
    /// activation buffer. Derived here so the caller cannot pick a different
    /// row count than the one `with_activations` verified.
    pub fn quantize_plan(self) -> Result<GgmlIqMmqPlan> {
        match self.activations {
            Glm53GroupedActivations::Shared => Ok(self.tile),
            Glm53GroupedActivations::PerPair => {
                GgmlIqMmqPlan::new(self.bank.kind, self.pairs, self.bank.columns, self.bank.inner)
            }
        }
    }
}

pub struct Glm53GroupedMoeKernels {
    kind: GgmlType,
    grouped_nc: KernelHandle,
    grouped_wc: KernelHandle,
    quantize: KernelHandle,
}

impl Glm53GroupedMoeKernels {
    pub fn load(gpu: &dyn GpuBackend, kind: GgmlType) -> Result<Self> {
        let tag = grouped_tag(kind)?;
        Ok(Self {
            kind,
            grouped_nc: gpu.kernel("ggml_iq_mmq", &format!("atlas_{tag}_moe_grouped_nc"))?,
            grouped_wc: gpu.kernel("ggml_iq_mmq", &format!("atlas_{tag}_moe_grouped_wc"))?,
            quantize: gpu.kernel("ggml_iq_mmq", quantize_name(kind)?)?,
        })
    }

    /// Quantize the activations ONCE, then run every (row, slot) tile.
    ///
    /// `route_ids_u32` stays on the device; the whole point is that the host
    /// never reads it. The kernel fails closed on an id outside
    /// `[0, experts)`, so a corrupt route leaves that slot's output slice
    /// untouched rather than reading arbitrary weights.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53GroupedMoePlan,
        input_bf16: DevicePtr,
        route_ids_u32: GgmlIqBuffer,
        activations: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.kind == plan.bank.kind,
            "GLM grouped MoE kernel is {:?}, plan is {:?}",
            self.kind,
            plan.bank.kind
        );
        ensure!(
            route_ids_u32.bytes == plan.route_bytes
                && activations.bytes == plan.activation_bytes
                && output.bytes == plan.output_bytes,
            "GLM grouped MoE buffer extents mismatch: routes {}/{} activations \
             {}/{} output {}/{}",
            route_ids_u32.bytes,
            plan.route_bytes,
            activations.bytes,
            plan.activation_bytes,
            output.bytes,
            plan.output_bytes
        );
        let tile = plan.tile;
        ensure!(
            plan.activations == Glm53GroupedActivations::Shared,
            "GLM grouped MoE launch() quantizes ONE activation block; a per-pair \
             plan must quantize its own and call launch_tiles"
        );
        // Same quantize pass as the serial path, and only ONE of it: all
        // `top_k` experts consume the identical activation row.
        KernelLaunch::new(gpu, self.quantize)
            .grid([
                tile.rows,
                div_ceil(tile.inner_padded, 4 * QUANTIZE_THREADS),
                1,
            ])
            .block([QUANTIZE_THREADS, 1, 1])
            .arg_ptr(input_bf16)
            .arg_ptr(activations.ptr)
            .arg_u64(u64::from(tile.inner))
            .arg_u64(u64::from(tile.inner))
            .arg_u64(u64::from(tile.inner_padded))
            .arg_u32(tile.rows)
            .launch(stream)?;
        self.launch_tiles(gpu, plan, route_ids_u32, activations, output, stream)
    }

    /// The tile half only, for a caller that has already quantized.
    ///
    /// The down projection needs TOP_K activation blocks quantized from TOP_K
    /// swiglu rows, which is one quantize launch at `rows = pairs` -- a
    /// different row count from the down tile's own `rows = 1` per pair. One
    /// plan cannot describe both, so the caller runs the quantize with its own
    /// plan and calls this.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiles(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53GroupedMoePlan,
        route_ids_u32: GgmlIqBuffer,
        activations: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        let tile = plan.tile;
        ensure!(
            self.kind == plan.bank.kind
                && route_ids_u32.bytes == plan.route_bytes
                && activations.bytes == plan.activation_bytes
                && output.bytes == plan.output_bytes,
            "GLM grouped MoE tile launch: kind or buffer extents mismatch"
        );
        let grouped = if tile.columns.is_multiple_of(MMQ_TILE_COLUMNS) {
            self.grouped_nc
        } else {
            self.grouped_wc
        };
        KernelLaunch::new(gpu, grouped)
            .grid([
                div_ceil(tile.columns, MMQ_TILE_COLUMNS),
                div_ceil(tile.rows, MMQ_TILE_ROWS),
                plan.pairs,
            ])
            .block([32, 8, 1])
            .shared_mem(tile.shared_mem)
            .arg_ptr(plan.bank.base)
            .arg_u32(u32::try_from(plan.bank.expert_bytes)?)
            .arg_ptr(route_ids_u32.ptr)
            .arg_u32(plan.top_k)
            .arg_u32(plan.bank.experts)
            .arg_ptr(activations.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(tile.columns)
            .arg_u32(tile.rows)
            .arg_u32(tile.inner)
            .arg_u32(tile.weight_row_stride())
            .arg_u32(tile.rows)
            .arg_u32(tile.columns)
            .arg_u32(plan.activation_pair_stride)
            .launch(stream)
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

/// Only the types the grouped kernel is actually DEFINED for.
///
/// This list is the eleven `ATLAS_DEFINE_GROUPED_MOE_MMQ` invocations in
/// `ggml_iq_mmq.cu`, verified against the 22 `_moe_grouped_[nw]c` entries in
/// the compiled t3 PTX rather than assumed from the serial path's twelve. The
/// serial path additionally handles Q3_K, which has no grouped definition;
/// asking for it must fail rather than resolve a symbol never emitted.
fn grouped_tag(kind: GgmlType) -> Result<&'static str> {
    Ok(match kind {
        GgmlType::Q2_K => "q2_k",
        GgmlType::Q4_K => "q4_k",
        GgmlType::Q5_K => "q5_k",
        GgmlType::Q6_K => "q6_k",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::IQ2_XXS => "iq2_xxs",
        GgmlType::IQ2_XS => "iq2_xs",
        GgmlType::IQ2_S => "iq2_s",
        GgmlType::IQ3_XXS => "iq3_xxs",
        GgmlType::IQ3_S => "iq3_s",
        GgmlType::IQ4_XS => "iq4_xs",
        other => bail!(
            "GLM grouped MoE has no kernel for {other:?}; the serial reference \
             path does. Run this bank on {}=serial-reference.",
            Glm53MoePath::ENV
        ),
    })
}

/// The quantize kernel is per-TYPE, not per-block-size. Taken from the single
/// mapping in `ggml_iq_mmq` rather than restated here: a second copy that
/// drifts would quantize with the wrong scale layout and produce plausible
/// numbers.
fn quantize_name(kind: GgmlType) -> Result<&'static str> {
    Ok(super::ops::ggml_iq_mmq::kernel_names(kind)?.2)
}

#[cfg(test)]
#[path = "glm53_moe_grouped_tests.rs"]
mod tests;

impl Glm53GroupedBank {
    /// Build from a loaded expert bank, re-running the uniform-stride guard.
    pub fn from_bank(bank: &crate::weight_loader::Glm53GgufMatrixBank) -> Result<Self> {
        let (buffer, kind, inner, columns, experts, expert_bytes) = bank.grouped_parts();
        Self::new(buffer, kind, inner, columns, experts, expert_bytes)
    }
}
