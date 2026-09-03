// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::gguf::GgmlType;

const Q8_BLOCK_VALUES: usize = 128;
const Q8_BLOCK_BYTES: usize = 144;
const QUANTIZE_THREADS: u32 = 128;
/// K advanced by one iteration of the MMQ tile loop (`MMQ_ITER_K` in mmq.cuh).
///
/// The tile loop is not sliced: one iteration always consumes this many K, so a
/// matrix whose K is not a multiple of it makes the last iteration read one
/// extra activation block past the row. Upstream llama.cpp pads the quantized
/// activation buffer to this granularity for exactly that reason; the padding
/// quantizes to d == 0, so the overrun contributes exactly zero.
const MMQ_ITER_K: usize = 256;

/// DELIBERATE-REGRESSION GATE. Flip to `false` to reproduce the pre-fix K=128
/// behaviour, where the activation buffer covers only the real K and the tile's
/// second half reads whatever the previous matmul left in the shared Q8 scratch.
///
/// This exists to answer one question we have never actually tested: whether the
/// stale-block read is load-bearing for GLM-5.3 coherence. It affects ONLY
/// matmuls whose K is not a multiple of `MMQ_ITER_K`, which across the whole GLM
/// walk is `ssm_f_b` and `ssm_g_b` at K=128 and nothing else.
///
/// Leave this `true` outside the A/B. A binary built with `false` self-identifies
/// via the distinct `.context` string on the unpadded branch below, so a
/// regression build can never be mistaken for a normal one in `strings`.
pub(crate) const MMQ_ACTIVATION_PADDING: bool = true;

/// Build-identity marker, force-emitted so `strings` can POSITIVELY name which
/// side of the gate a binary was built on.
///
/// An absence-of-string check is not good enough: with padding off, LLVM proves
/// `u32::try_from` on a value that round-tripped from a `u32` cannot fail and
/// deletes the error path along with its message, so BOTH `.context` strings
/// vanish and a regression binary looks identical to one where the marker was
/// simply renamed. Today cost us most of a day to binaries we could not
/// attribute; identity must be assertable, not inferable from a missing match.
#[used]
static ATLAS_MMQ_BUILD_MARKER: &str = if MMQ_ACTIVATION_PADDING {
    "ATLAS_MMQ_PADDING=on"
} else {
    "ATLAS_MMQ_PADDING=off-DELIBERATE-REGRESSION"
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlIqMmqPlan {
    pub kind: GgmlType,
    pub rows: u32,
    pub columns: u32,
    pub inner: u32,
    /// `inner` rounded up to [`MMQ_ITER_K`]: the K extent the activation buffer
    /// covers and the quantize pass zero-fills.
    pub inner_padded: u32,
    pub weight_bytes: usize,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub shared_mem: u32,
}

impl GgmlIqMmqPlan {
    pub fn new(kind: GgmlType, rows: u32, columns: u32, inner: u32) -> Result<Self> {
        if rows == 0 || columns == 0 || inner == 0 {
            bail!("GGML IQ MMQ requires nonzero M/N/K");
        }
        let (block_values, block_bytes, shared_mem) = layout(kind)?;
        let inner_usize = usize::try_from(inner)?;
        let inner_alignment = block_values.max(Q8_BLOCK_VALUES);
        if !inner_usize.is_multiple_of(inner_alignment) {
            bail!("GGML IQ MMQ K must be divisible by {inner_alignment} for the selected type");
        }
        let weight_blocks = usize::try_from(columns)?
            .checked_mul(inner_usize / block_values)
            .context("GGML IQ weight block count overflow")?;
        let weight_bytes = weight_blocks
            .checked_mul(block_bytes)
            .context("GGML IQ weight byte count overflow")?;
        let inner_padded = if MMQ_ACTIVATION_PADDING {
            inner_usize.next_multiple_of(MMQ_ITER_K)
        } else {
            inner_usize
        };
        let activation_bytes = usize::try_from(rows)?
            .checked_mul(inner_padded / Q8_BLOCK_VALUES)
            .and_then(|blocks| blocks.checked_mul(Q8_BLOCK_BYTES))
            .context("GGML IQ activation byte count overflow")?;
        let output_bytes = usize::try_from(rows)?
            .checked_mul(usize::try_from(columns)?)
            .and_then(|elements| elements.checked_mul(2))
            .context("GGML IQ output byte count overflow")?;
        Ok(Self {
            kind,
            rows,
            columns,
            inner,
            inner_padded: u32::try_from(inner_padded)
                .context("GGML IQ padded K exceeds kernel ABI")?,
            weight_bytes,
            activation_bytes,
            output_bytes,
            shared_mem,
        })
    }

    pub(crate) fn weight_row_stride(self) -> u32 {
        self.inner / layout(self.kind).expect("validated plan type").0 as u32
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GgmlIqBuffer {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

pub struct GgmlIqMmqKernels {
    kind: GgmlType,
    mmq_nc: KernelHandle,
    mmq_wc: KernelHandle,
    quantize: KernelHandle,
}

impl GgmlIqMmqKernels {
    pub fn load(gpu: &dyn GpuBackend, kind: GgmlType) -> Result<Self> {
        let (nc, wc, quantize) = kernel_names(kind)?;
        Ok(Self {
            kind,
            mmq_nc: gpu.kernel("ggml_iq_mmq", nc)?,
            mmq_wc: gpu.kernel("ggml_iq_mmq", wc)?,
            quantize: gpu.kernel("ggml_iq_mmq", quantize)?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: GgmlIqMmqPlan,
        input_bf16: DevicePtr,
        weights: GgmlIqBuffer,
        activations: GgmlIqBuffer,
        output: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        if self.kind != plan.kind
            || weights.bytes != plan.weight_bytes
            || activations.bytes != plan.activation_bytes
            || output.bytes != plan.output_bytes
        {
            bail!("GGML IQ MMQ kernel type or device buffer extent mismatch");
        }
        // The quantize pass covers the padded K: `ne00` (real width) bounds the
        // loads and `ne0` (padded width) the blocks written, so the tail block
        // is zero-filled rather than left as whatever the arena last held.
        let quant_grid_y = div_ceil(plan.inner_padded, 4 * QUANTIZE_THREADS);
        KernelLaunch::new(gpu, self.quantize)
            .grid([plan.rows, quant_grid_y, 1])
            .block([QUANTIZE_THREADS, 1, 1])
            .arg_ptr(input_bf16)
            .arg_ptr(activations.ptr)
            .arg_u64(plan.inner as u64)
            .arg_u64(plan.inner as u64)
            .arg_u64(plan.inner_padded as u64)
            .arg_u32(plan.rows)
            .launch(stream)?;
        let mmq = if plan.columns.is_multiple_of(128) {
            self.mmq_nc
        } else {
            self.mmq_wc
        };
        KernelLaunch::new(gpu, mmq)
            .grid([div_ceil(plan.columns, 128), div_ceil(plan.rows, 128), 1])
            .block([32, 8, 1])
            .shared_mem(plan.shared_mem)
            .arg_ptr(weights.ptr)
            .arg_ptr(activations.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(plan.columns)
            .arg_u32(plan.rows)
            .arg_u32(plan.inner)
            .arg_u32(plan.weight_row_stride())
            .arg_u32(plan.rows)
            .arg_u32(plan.columns)
            .launch(stream)
    }
}

fn layout(kind: GgmlType) -> Result<(usize, usize, u32)> {
    Ok(match kind {
        GgmlType::Q2_K => (256, 84, 70_144),
        GgmlType::Q3_K => (256, 110, 61_952),
        GgmlType::Q4_K => (256, 144, 57_856),
        GgmlType::Q5_K => (256, 176, 57_856),
        GgmlType::Q6_K => (256, 210, 57_856),
        GgmlType::Q8_0 => (32, 34, 57_856),
        // IQ2_XXS shares the Q8_0 MMA/DP4A tile shape with IQ3_XXS/IQ3_S/IQ4_XS,
        // so it takes their scratch size rather than the wider IQ2_XS/IQ2_S one.
        GgmlType::IQ2_XXS => (256, 66, 57_856),
        GgmlType::IQ2_XS => (256, 74, 61_952),
        GgmlType::IQ2_S => (256, 82, 61_952),
        GgmlType::IQ3_XXS => (256, 98, 57_856),
        GgmlType::IQ3_S => (256, 110, 57_856),
        GgmlType::IQ4_XS => (256, 136, 57_856),
        _ => bail!("GGML type has no pinned IQ3 MMQ specialization"),
    })
}

pub(crate) fn kernel_names(kind: GgmlType) -> Result<(&'static str, &'static str, &'static str)> {
    let d4 = "atlas_q8_1_quantize_d4_bf16";
    let ds4 = "atlas_q8_1_quantize_ds4_bf16";
    Ok(match kind {
        GgmlType::Q2_K => (
            "atlas_q2_k_mmq128_nc",
            "atlas_q2_k_mmq128_wc",
            "atlas_q8_1_quantize_d2s6_bf16",
        ),
        GgmlType::Q3_K => ("atlas_q3_k_mmq128_nc", "atlas_q3_k_mmq128_wc", d4),
        GgmlType::Q4_K => ("atlas_q4_k_mmq128_nc", "atlas_q4_k_mmq128_wc", ds4),
        GgmlType::Q5_K => ("atlas_q5_k_mmq128_nc", "atlas_q5_k_mmq128_wc", ds4),
        GgmlType::Q6_K => ("atlas_q6_k_mmq128_nc", "atlas_q6_k_mmq128_wc", d4),
        GgmlType::Q8_0 => ("atlas_q8_0_mmq128_nc", "atlas_q8_0_mmq128_wc", d4),
        GgmlType::IQ2_XXS => ("atlas_iq2_xxs_mmq128_nc", "atlas_iq2_xxs_mmq128_wc", d4),
        GgmlType::IQ2_XS => ("atlas_iq2_xs_mmq128_nc", "atlas_iq2_xs_mmq128_wc", d4),
        GgmlType::IQ2_S => ("atlas_iq2_s_mmq128_nc", "atlas_iq2_s_mmq128_wc", d4),
        GgmlType::IQ3_XXS => ("atlas_iq3_xxs_mmq128_nc", "atlas_iq3_xxs_mmq128_wc", d4),
        GgmlType::IQ3_S => ("atlas_iq3_s_mmq128_nc", "atlas_iq3_s_mmq128_wc", d4),
        GgmlType::IQ4_XS => ("atlas_iq4_xs_mmq128_nc", "atlas_iq4_xs_mmq128_wc", d4),
        _ => bail!("GGML type has no pinned IQ3 MMQ specialization"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn exact_recipe_layouts_and_overflow_fail_closed() {
        for (kind, bytes_per_256, shared) in [
            (GgmlType::Q2_K, 84, 70_144),
            (GgmlType::Q3_K, 110, 61_952),
            (GgmlType::Q4_K, 144, 57_856),
            (GgmlType::Q5_K, 176, 57_856),
            (GgmlType::Q6_K, 210, 57_856),
            (GgmlType::Q8_0, 272, 57_856),
            (GgmlType::IQ2_XXS, 66, 57_856),
            (GgmlType::IQ2_XS, 74, 61_952),
            (GgmlType::IQ2_S, 82, 61_952),
            (GgmlType::IQ3_XXS, 98, 57_856),
            (GgmlType::IQ3_S, 110, 57_856),
            (GgmlType::IQ4_XS, 136, 57_856),
        ] {
            let plan = GgmlIqMmqPlan::new(kind, 2, 3, 256).unwrap();
            assert_eq!(plan.weight_bytes, 3 * bytes_per_256);
            assert_eq!(plan.activation_bytes, 2 * 2 * 144);
            assert_eq!(plan.output_bytes, 12);
            assert_eq!(plan.shared_mem, shared);
        }
        assert!(GgmlIqMmqPlan::new(GgmlType::F32, 1, 1, 256).is_err());
        // K=128 is a *half* MMQ tile iteration, so the activation buffer must
        // still cover a full MMQ_ITER_K=256 of zero-padded K: the tile loop
        // reads both halves unconditionally. GLM-5.3 ssm_f_b / ssm_g_b are
        // exactly this shape.
        let q8_k128 = GgmlIqMmqPlan::new(GgmlType::Q8_0, 1, 3, 128).unwrap();
        assert_eq!(q8_k128.weight_bytes, 3 * 4 * 34);
        // Pinned against the gate rather than deleted: with padding on this is a
        // full 256-K buffer, and in the deliberate-regression build it collapses
        // to the real 128 K, which is exactly the pre-fix overrun.
        if MMQ_ACTIVATION_PADDING {
            assert_eq!(q8_k128.inner_padded, 256);
            assert_eq!(q8_k128.activation_bytes, 2 * 144);
        } else {
            assert_eq!(q8_k128.inner_padded, 128);
            assert_eq!(q8_k128.activation_bytes, 144);
        }
        for kind in [
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
        ] {
            assert!(GgmlIqMmqPlan::new(kind, 1, 1, 128).is_err());
        }
        assert!(GgmlIqMmqPlan::new(GgmlType::IQ3_S, 1, 1, 128).is_err());
        assert!(GgmlIqMmqPlan::new(GgmlType::IQ3_S, u32::MAX, u32::MAX, 256).is_err());
    }

    #[test]
    fn q2_recipe_types_bind_exact_vendored_symbols() {
        for (kind, expected) in [
            (
                GgmlType::Q4_K,
                (
                    "atlas_q4_k_mmq128_nc",
                    "atlas_q4_k_mmq128_wc",
                    "atlas_q8_1_quantize_ds4_bf16",
                ),
            ),
            (
                GgmlType::Q5_K,
                (
                    "atlas_q5_k_mmq128_nc",
                    "atlas_q5_k_mmq128_wc",
                    "atlas_q8_1_quantize_ds4_bf16",
                ),
            ),
            (
                GgmlType::IQ2_XS,
                (
                    "atlas_iq2_xs_mmq128_nc",
                    "atlas_iq2_xs_mmq128_wc",
                    "atlas_q8_1_quantize_d4_bf16",
                ),
            ),
            (
                GgmlType::IQ3_XXS,
                (
                    "atlas_iq3_xxs_mmq128_nc",
                    "atlas_iq3_xxs_mmq128_wc",
                    "atlas_q8_1_quantize_d4_bf16",
                ),
            ),
            (
                GgmlType::IQ2_XXS,
                (
                    "atlas_iq2_xxs_mmq128_nc",
                    "atlas_iq2_xxs_mmq128_wc",
                    "atlas_q8_1_quantize_d4_bf16",
                ),
            ),
        ] {
            assert_eq!(kernel_names(kind).unwrap(), expected);
        }
    }

    #[test]
    fn q2_recipe_kind_and_extents_reject_before_launch() {
        let gpu = MockGpuBackend::new();
        let q4 = GgmlIqMmqPlan::new(GgmlType::Q4_K, 1, 128, 4096).unwrap();
        let q5 = GgmlIqMmqPlan::new(GgmlType::Q5_K, 1, 128, 4096).unwrap();
        let kernels = GgmlIqMmqKernels::load(&gpu, GgmlType::Q4_K).unwrap();
        let view = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        assert!(
            kernels
                .launch(
                    &gpu,
                    q5,
                    DevicePtr(1),
                    view(2, q5.weight_bytes),
                    view(3, q5.activation_bytes),
                    view(4, q5.output_bytes),
                    0,
                )
                .is_err()
        );
        assert!(
            kernels
                .launch(
                    &gpu,
                    q4,
                    DevicePtr(1),
                    view(2, q4.weight_bytes),
                    view(3, q4.activation_bytes + 1),
                    view(4, q4.output_bytes),
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernels
            .launch(
                &gpu,
                q4,
                DevicePtr(1),
                view(2, q4.weight_bytes),
                view(3, q4.activation_bytes),
                view(4, q4.output_bytes),
                0,
            )
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn launch_requires_exact_extents_before_two_kernel_nodes() {
        let gpu = MockGpuBackend::new();
        let plan = GgmlIqMmqPlan::new(GgmlType::IQ3_S, 129, 129, 4096).unwrap();
        let kernels = GgmlIqMmqKernels::load(&gpu, plan.kind).unwrap();
        let view = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        assert!(
            kernels
                .launch(
                    &gpu,
                    plan,
                    DevicePtr(1),
                    view(2, plan.weight_bytes - 1),
                    view(3, plan.activation_bytes),
                    view(4, plan.output_bytes),
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernels
            .launch(
                &gpu,
                plan,
                DevicePtr(1),
                view(2, plan.weight_bytes),
                view(3, plan.activation_bytes),
                view(4, plan.output_bytes),
                0,
            )
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }
}
