// SPDX-License-Identifier: AGPL-3.0-only

//! Default-unrouted device-resident NVFP4 activation scaling.
//!
//! This op is intentionally limited to the two target Qwen3.8 SSM prefill
//! projection widths and two qualified row counts. It only enqueues work on
//! the caller's stream:
//!
//! 1. clear the device absmax and status scalars;
//! 2. run the existing Atlas BF16 absmax reduction;
//! 3. derive `scale2_a` on device while emitting Atlas-byte-identical packed
//!    E2M1 activations and CUTLASS/FlashInfer 128x4 physical E4M3 scales;
//! 4. publish `scale2_a * weight_scale_2` to a separate device scalar.
//!
//! There is no host readback or stream synchronization. The last kernel traps
//! before publishing an alpha if the quantizer reported non-finite input, so a
//! following same-stream FlashInfer C-ABI launch cannot consume stale output.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const QWEN38_SSM_QKVZ_K: u32 = 5_120;
pub const QWEN38_SSM_OUTPUT_K: u32 = 6_144;
pub const QWEN38_DYNAMIC_ROWS: [u32; 2] = [2_079, 8_192];

/// Frozen launch geometry and exact caller-owned buffer sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4DynamicScalePlan {
    pub rows: u32,
    pub cols: u32,
    pub padded_rows: u32,
    pub total_elements: u32,
    pub input_bytes: usize,
    pub packed_bytes: usize,
    pub physical_scale_bytes: usize,
}

impl Nvfp4DynamicScalePlan {
    pub fn qwen38_ssm_projection(rows: u32, cols: u32) -> Result<Self> {
        ensure!(
            QWEN38_DYNAMIC_ROWS.contains(&rows),
            "device dynamic NVFP4 is bounded to Qwen3.8 SSM rows 2079 or 8192; got {rows}"
        );
        ensure!(
            matches!(cols, QWEN38_SSM_QKVZ_K | QWEN38_SSM_OUTPUT_K),
            "device dynamic NVFP4 is bounded to Qwen3.8 SSM K={QWEN38_SSM_QKVZ_K} or K={QWEN38_SSM_OUTPUT_K}; got {cols}"
        );
        ensure!(
            cols.is_multiple_of(64),
            "device dynamic NVFP4 requires K divisible by 64"
        );

        let padded_rows = rows
            .checked_add(127)
            .ok_or_else(|| anyhow::anyhow!("dynamic NVFP4 padded-row overflow"))?
            / 128
            * 128;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("dynamic NVFP4 element-count overflow"))?;
        let total_usize = usize::try_from(total)?;
        let physical_scale_elements = padded_rows
            .checked_mul(cols / 16)
            .ok_or_else(|| anyhow::anyhow!("dynamic NVFP4 physical-scale overflow"))?;

        Ok(Self {
            rows,
            cols,
            padded_rows,
            total_elements: total,
            input_bytes: total_usize
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("dynamic NVFP4 input-byte overflow"))?,
            packed_bytes: total_usize / 2,
            physical_scale_bytes: usize::try_from(physical_scale_elements)?,
        })
    }

    /// Compatibility constructor for callers that require the QKVZ width.
    pub fn qwen38_ssm_qkvz(rows: u32, cols: u32) -> Result<Self> {
        ensure!(
            cols == QWEN38_SSM_QKVZ_K,
            "Qwen3.8 SSM QKVZ dynamic NVFP4 requires K={QWEN38_SSM_QKVZ_K}; got {cols}"
        );
        Self::qwen38_ssm_projection(rows, cols)
    }
}

/// The three ABI-separated kernels used by the device-only chain.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4DynamicScaleKernels {
    pub absmax: KernelHandle,
    pub quantize_from_absmax: KernelHandle,
    pub combined_alpha: KernelHandle,
}

impl Nvfp4DynamicScaleKernels {
    /// Resolve symbols without enabling a production route.
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernels = Self {
            absmax: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
            quantize_from_absmax: gpu.kernel(
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
            )?,
            combined_alpha: gpu.kernel(
                "quantize_bf16_to_nvfp4_cutlass",
                "nvfp4_dynamic_combined_alpha",
            )?,
        };
        kernels.validate()?;
        Ok(kernels)
    }

    fn validate(self) -> Result<()> {
        ensure!(self.absmax.0 != 0, "dynamic NVFP4 absmax kernel is missing");
        ensure!(
            self.quantize_from_absmax.0 != 0,
            "dynamic NVFP4 quantizer kernel is missing"
        );
        ensure!(
            self.combined_alpha.0 != 0,
            "dynamic NVFP4 combined-alpha kernel is missing"
        );
        Ok(())
    }
}

/// Caller-owned device operands. All outputs remain live through the following
/// same-stream FlashInfer projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4DynamicScaleBuffers {
    pub input_bf16: DevicePtr,
    pub packed_e2m1: DevicePtr,
    pub scales_e4m3_128x4: DevicePtr,
    pub global_max_f32: DevicePtr,
    pub scale2_a_f32: DevicePtr,
    pub status_u32: DevicePtr,
    pub combined_alpha_f32: DevicePtr,
}

/// Stream-bound result of the asynchronous preparation chain.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4DynamicScaleOutput {
    activation_fp4: DevicePtr,
    activation_scales: DevicePtr,
    global_scale_f32: DevicePtr,
    scale2_a_f32: DevicePtr,
    status_u32: DevicePtr,
    stream: u64,
}

/// The three pointers consumed by the following FlashInfer projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4DynamicFlashInferInputs {
    pub activation_fp4: DevicePtr,
    pub activation_scales: DevicePtr,
    pub global_scale_f32: DevicePtr,
}

impl Nvfp4DynamicScaleOutput {
    /// Expose C-ABI inputs only for the exact stream on which they were
    /// produced. This prevents a caller from racing the asynchronous scalar
    /// publication on another stream.
    pub fn for_flashinfer(self, stream: u64) -> Result<Nvfp4DynamicFlashInferInputs> {
        ensure!(
            stream == self.stream,
            "dynamic NVFP4 output is bound to stream {}, not {stream}",
            self.stream
        );
        Ok(Nvfp4DynamicFlashInferInputs {
            activation_fp4: self.activation_fp4,
            activation_scales: self.activation_scales,
            global_scale_f32: self.global_scale_f32,
        })
    }

    pub fn scale2_a_f32(self) -> DevicePtr {
        self.scale2_a_f32
    }

    pub fn status_u32(self) -> DevicePtr {
        self.status_u32
    }
}

fn checked_span(
    name: &str,
    pointer: DevicePtr,
    bytes: usize,
    alignment: u64,
) -> Result<(u64, u64)> {
    ensure!(!pointer.is_null(), "dynamic NVFP4 {name} pointer is null");
    ensure!(
        pointer.0.is_multiple_of(alignment),
        "dynamic NVFP4 {name} pointer is not {alignment}-byte aligned"
    );
    let end = pointer
        .0
        .checked_add(u64::try_from(bytes)?)
        .ok_or_else(|| anyhow::anyhow!("dynamic NVFP4 {name} address range overflow"))?;
    Ok((pointer.0, end))
}

fn spans_overlap(lhs: (u64, u64), rhs: (u64, u64)) -> bool {
    lhs.0 < rhs.1 && rhs.0 < lhs.1
}

fn validate_buffers(plan: Nvfp4DynamicScalePlan, buffers: Nvfp4DynamicScaleBuffers) -> Result<()> {
    let spans = [
        (
            "input_bf16",
            checked_span("input_bf16", buffers.input_bf16, plan.input_bytes, 16)?,
        ),
        (
            "packed_e2m1",
            checked_span("packed_e2m1", buffers.packed_e2m1, plan.packed_bytes, 16)?,
        ),
        (
            "scales_e4m3_128x4",
            checked_span(
                "scales_e4m3_128x4",
                buffers.scales_e4m3_128x4,
                plan.physical_scale_bytes,
                16,
            )?,
        ),
        (
            "global_max_f32",
            checked_span("global_max_f32", buffers.global_max_f32, 4, 4)?,
        ),
        (
            "scale2_a_f32",
            checked_span("scale2_a_f32", buffers.scale2_a_f32, 4, 4)?,
        ),
        (
            "status_u32",
            checked_span("status_u32", buffers.status_u32, 4, 4)?,
        ),
        (
            "combined_alpha_f32",
            checked_span("combined_alpha_f32", buffers.combined_alpha_f32, 4, 4)?,
        ),
    ];
    for left in 0..spans.len() {
        for right in left + 1..spans.len() {
            ensure!(
                !spans_overlap(spans[left].1, spans[right].1),
                "dynamic NVFP4 buffers overlap: {} and {}",
                spans[left].0,
                spans[right].0
            );
        }
    }
    Ok(())
}

/// Enqueue the complete device-only activation preparation chain.
///
/// The returned `global_scale_f32` is `scale2_a * weight_scale_2`, not raw
/// `scale2_a`, and is therefore directly suitable for
/// `FlashInferSm121Buffers::global_scale_f32`. The
/// caller must retain every buffer until the following projection completes on
/// `stream`.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_qwen38_ssm_projection_dynamic_scale(
    gpu: &dyn GpuBackend,
    kernels: Nvfp4DynamicScaleKernels,
    buffers: Nvfp4DynamicScaleBuffers,
    rows: u32,
    cols: u32,
    weight_scale_2: f32,
    stream: u64,
) -> Result<Nvfp4DynamicScaleOutput> {
    let plan =
        validate_qwen38_ssm_projection_dynamic_scale(kernels, buffers, rows, cols, weight_scale_2)?;

    // All operations are ordered on the caller's stream. Do not replace these
    // with synchronous memset/copy or add a host readback between phases.
    gpu.memset_async(buffers.global_max_f32, 0, 4, stream)?;
    gpu.memset_async(buffers.status_u32, 0, 4, stream)?;

    let absmax_grid = (plan.total_elements / 256).clamp(1, 1_024);
    KernelLaunch::new(gpu, kernels.absmax)
        .grid([absmax_grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(buffers.input_bf16)
        .arg_ptr(buffers.global_max_f32)
        .arg_u32(plan.total_elements)
        .launch(stream)?;

    KernelLaunch::new(gpu, kernels.quantize_from_absmax)
        .grid([plan.padded_rows.min(96), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(buffers.input_bf16)
        .arg_ptr(buffers.packed_e2m1)
        .arg_ptr(buffers.scales_e4m3_128x4)
        .arg_ptr(buffers.global_max_f32)
        .arg_ptr(buffers.scale2_a_f32)
        .arg_ptr(buffers.status_u32)
        .arg_u32(plan.rows)
        .arg_u32(plan.cols)
        .launch(stream)?;

    KernelLaunch::new(gpu, kernels.combined_alpha)
        .grid([1, 1, 1])
        .block([1, 1, 1])
        .arg_ptr(buffers.scale2_a_f32)
        .arg_f32(weight_scale_2)
        .arg_ptr(buffers.status_u32)
        .arg_ptr(buffers.combined_alpha_f32)
        .launch(stream)?;

    Ok(Nvfp4DynamicScaleOutput {
        activation_fp4: buffers.packed_e2m1,
        activation_scales: buffers.scales_e4m3_128x4,
        global_scale_f32: buffers.combined_alpha_f32,
        scale2_a_f32: buffers.scale2_a_f32,
        status_u32: buffers.status_u32,
        stream,
    })
}

/// Validate the complete device-only chain without launching or mutating any
/// buffer. Production routes use this before their first parent or candidate
/// operation so an explicitly requested but incomplete route fails closed.
pub fn validate_qwen38_ssm_projection_dynamic_scale(
    kernels: Nvfp4DynamicScaleKernels,
    buffers: Nvfp4DynamicScaleBuffers,
    rows: u32,
    cols: u32,
    weight_scale_2: f32,
) -> Result<Nvfp4DynamicScalePlan> {
    kernels.validate()?;
    let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(rows, cols)?;
    ensure!(
        weight_scale_2.is_finite() && weight_scale_2 > 0.0,
        "dynamic NVFP4 requires finite positive weight_scale_2"
    );
    validate_buffers(plan, buffers)?;
    Ok(plan)
}

/// Compatibility entry point retaining the original QKVZ-only admission.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_qwen38_ssm_qkvz_dynamic_scale(
    gpu: &dyn GpuBackend,
    kernels: Nvfp4DynamicScaleKernels,
    buffers: Nvfp4DynamicScaleBuffers,
    rows: u32,
    cols: u32,
    weight_scale_2: f32,
    stream: u64,
) -> Result<Nvfp4DynamicScaleOutput> {
    let _ = Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(rows, cols)?;
    enqueue_qwen38_ssm_projection_dynamic_scale(
        gpu,
        kernels,
        buffers,
        rows,
        cols,
        weight_scale_2,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn kernels() -> Nvfp4DynamicScaleKernels {
        Nvfp4DynamicScaleKernels {
            absmax: KernelHandle(1),
            quantize_from_absmax: KernelHandle(2),
            combined_alpha: KernelHandle(3),
        }
    }

    fn buffers() -> Nvfp4DynamicScaleBuffers {
        Nvfp4DynamicScaleBuffers {
            input_bf16: DevicePtr(0x0010_0000),
            packed_e2m1: DevicePtr(0x0400_0000),
            scales_e4m3_128x4: DevicePtr(0x0600_0000),
            global_max_f32: DevicePtr(0x0700_0000),
            scale2_a_f32: DevicePtr(0x0700_0010),
            status_u32: DevicePtr(0x0700_0020),
            combined_alpha_f32: DevicePtr(0x0700_0030),
        }
    }

    #[test]
    fn exact_qwen38_layout_sizes() {
        let short = Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(2_079, 5_120).unwrap();
        assert_eq!(short.padded_rows, 2_176);
        assert_eq!(short.total_elements, 10_644_480);
        assert_eq!(short.input_bytes, 21_288_960);
        assert_eq!(short.packed_bytes, 5_322_240);
        assert_eq!(short.physical_scale_bytes, 696_320);

        let long = Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(8_192, 5_120).unwrap();
        assert_eq!(long.padded_rows, 8_192);
        assert_eq!(long.total_elements, 41_943_040);
        assert_eq!(long.input_bytes, 83_886_080);
        assert_eq!(long.packed_bytes, 20_971_520);
        assert_eq!(long.physical_scale_bytes, 2_621_440);

        let output_short = Nvfp4DynamicScalePlan::qwen38_ssm_projection(2_079, 6_144).unwrap();
        assert_eq!(output_short.padded_rows, 2_176);
        assert_eq!(output_short.total_elements, 12_773_376);
        assert_eq!(output_short.input_bytes, 25_546_752);
        assert_eq!(output_short.packed_bytes, 6_386_688);
        assert_eq!(output_short.physical_scale_bytes, 835_584);

        let output_long = Nvfp4DynamicScalePlan::qwen38_ssm_projection(8_192, 6_144).unwrap();
        assert_eq!(output_long.padded_rows, 8_192);
        assert_eq!(output_long.total_elements, 50_331_648);
        assert_eq!(output_long.input_bytes, 100_663_296);
        assert_eq!(output_long.packed_bytes, 25_165_824);
        assert_eq!(output_long.physical_scale_bytes, 3_145_728);
    }

    #[test]
    fn wrong_shape_is_rejected() {
        for (rows, cols) in [(0, 5_120), (2_048, 5_120), (2_079, 4_096), (8_193, 6_144)] {
            assert!(Nvfp4DynamicScalePlan::qwen38_ssm_projection(rows, cols).is_err());
        }
        assert!(Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(2_079, 6_144).is_err());
    }

    #[test]
    fn missing_kernel_invalid_scalar_and_alias_fail_before_effect() {
        let gpu = MockGpuBackend::new();
        let mut missing = kernels();
        missing.combined_alpha = KernelHandle(0);
        assert!(
            enqueue_qwen38_ssm_qkvz_dynamic_scale(&gpu, missing, buffers(), 2_079, 5_120, 0.75, 7,)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        assert!(
            enqueue_qwen38_ssm_qkvz_dynamic_scale(
                &gpu,
                kernels(),
                buffers(),
                2_079,
                5_120,
                f32::NAN,
                7,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        let mut aliased = buffers();
        aliased.packed_e2m1 = aliased.input_bf16;
        assert!(
            enqueue_qwen38_ssm_qkvz_dynamic_scale(&gpu, kernels(), aliased, 2_079, 5_120, 0.75, 7,)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn enqueues_exact_three_kernel_chain_without_host_sync() {
        let gpu = MockGpuBackend::new();
        // The mock memset implementation requires its destination allocations.
        let max = gpu.alloc(4).unwrap();
        let scale2 = gpu.alloc(4).unwrap();
        let status = gpu.alloc(4).unwrap();
        let alpha = gpu.alloc(4).unwrap();
        let mut owned = buffers();
        owned.global_max_f32 = max;
        owned.scale2_a_f32 = scale2;
        owned.status_u32 = status;
        owned.combined_alpha_f32 = alpha;
        let output =
            enqueue_qwen38_ssm_qkvz_dynamic_scale(&gpu, kernels(), owned, 2_079, 5_120, 0.75, 91)
                .unwrap();
        assert_eq!(gpu.launch_count(), 3);
        assert!(output.for_flashinfer(92).is_err());
        let flashinfer = output.for_flashinfer(91).unwrap();
        assert_eq!(flashinfer.activation_fp4, owned.packed_e2m1);
        assert_eq!(flashinfer.activation_scales, owned.scales_e4m3_128x4);
        assert_eq!(flashinfer.global_scale_f32, alpha);
        assert_eq!(output.scale2_a_f32(), scale2);
        assert_eq!(output.status_u32(), status);
    }

    #[test]
    fn source_contract_is_device_only_and_precise() {
        let rust = include_str!("nvfp4_dynamic_scale.rs");
        let forbidden_copy = ["copy", "_d2h("].concat();
        let forbidden_sync = [".synchro", "nize("].concat();
        assert!(!rust.contains(&forbidden_copy));
        assert!(!rust.contains(&forbidden_sync));
        let memset = rust.find("memset_async(buffers.global_max_f32").unwrap();
        let absmax = rust.find("KernelLaunch::new(gpu, kernels.absmax)").unwrap();
        let quantize = rust
            .find("KernelLaunch::new(gpu, kernels.quantize_from_absmax)")
            .unwrap();
        let alpha = rust
            .find("KernelLaunch::new(gpu, kernels.combined_alpha)")
            .unwrap();
        assert!(memset < absmax && absmax < quantize && quantize < alpha);

        let cuda = include_str!(
            "../../../../../kernels/gb10/qwen3.8-27b/nvfp4/quantize_bf16_to_nvfp4_cutlass.cu"
        );
        assert!(cuda.contains("quantize_bf16_to_nvfp4_atlas_128x4_from_absmax"));
        assert!(cuda.contains("nvfp4_dynamic_combined_alpha"));
        assert!(cuda.contains("__fdiv_rn(maximum, 2688.0f)"));
        assert!(cuda.contains("mul_rn(activation_scale, weight_scale_2)"));
        assert!(cuda.contains("scale_offset_128x4(row, group, groups)"));
        assert!(cuda.contains("asm volatile(\"trap;\")"));
    }
}
