// SPDX-License-Identifier: AGPL-3.0-only

//! Unified target projection dispatch for compressed EXL3 and native BF16.

use std::cell::Cell;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Glm53Exl3Buffer, glm53_exact_verify_active, native_rows_selected};
use crate::weight_loader::{
    Glm53Exl3Bf16LinearBuffers, Glm53Exl3Linear, Glm53Exl3NativeDtype, Glm53Exl3NativeTensor,
};

pub const GLM53_EXL3_MAX_INPUT_F16_BYTES: usize = 16_384 * 2;
pub const GLM53_EXL3_MAX_OUTPUT_F16_BYTES: usize = 154_880 * 2;
/// Maximum explicit-default-off layer-major prompt chunk. The exact verifier
/// continues to admit at most eight rows; only the dedicated prompt scope uses
/// the larger scratch prefixes.
pub const GLM53_EXL3_MAX_WIDE_ROWS: usize = 2_048;
pub const GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES: usize =
    GLM53_EXL3_MAX_WIDE_ROWS * GLM53_EXL3_MAX_INPUT_F16_BYTES;
pub const GLM53_EXL3_MAX_WIDE_OUTPUT_F16_BYTES: usize =
    GLM53_EXL3_MAX_WIDE_ROWS * GLM53_EXL3_MAX_OUTPUT_F16_BYTES;
pub const GLM53_EXL3_LOCK_BYTES: usize = 1024 * 1024 * size_of::<i32>();

thread_local! {
    static EXACT_WIDE_PREFILL_DEPTH: Cell<u32> = const { Cell::new(0) };
    static LAYER_MAJOR_PREFILL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct ExactWidePrefillGuard;

impl Drop for ExactWidePrefillGuard {
    fn drop(&mut self) {
        EXACT_WIDE_PREFILL_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

struct LayerMajorPrefillGuard;

impl Drop for LayerMajorPrefillGuard {
    fn drop(&mut self) {
        LAYER_MAJOR_PREFILL_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

pub(crate) fn with_glm53_exact_wide_prefill<T>(operation: impl FnOnce() -> T) -> T {
    EXACT_WIDE_PREFILL_DEPTH.with(|depth| {
        depth.set(
            depth
                .get()
                .checked_add(1)
                .expect("GLM exact-wide-prefill scope overflow"),
        )
    });
    let _guard = ExactWidePrefillGuard;
    operation()
}

pub(crate) fn glm53_exact_wide_prefill_active() -> bool {
    EXACT_WIDE_PREFILL_DEPTH.with(|depth| depth.get() != 0)
}

pub(crate) fn with_glm53_layer_major_prefill<T>(operation: impl FnOnce() -> T) -> T {
    LAYER_MAJOR_PREFILL_DEPTH.with(|depth| {
        depth.set(
            depth
                .get()
                .checked_add(1)
                .expect("GLM layer-major-prefill scope overflow"),
        )
    });
    let _guard = LayerMajorPrefillGuard;
    operation()
}

pub(crate) fn glm53_layer_major_prefill_active() -> bool {
    LAYER_MAJOR_PREFILL_DEPTH.with(|depth| depth.get() != 0)
}

#[derive(Clone, Copy)]
pub enum Glm53Exl3Projection<'a> {
    Compressed(&'a Glm53Exl3Linear),
    NativeBf16(&'a Glm53Exl3NativeTensor),
    /// Native tensor is stored `[K,N]`; compute activation times weight without
    /// transposing it. Used by absorbed DSA K only.
    NativeBf16ActWeight(&'a Glm53Exl3NativeTensor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm53Exl3ProjectionPlan {
    pub rows: u32,
    pub input: u32,
    pub output: u32,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub compressed: bool,
}

#[derive(Clone, Copy)]
pub struct Glm53Exl3ProjectionScratch {
    pub input_f16: Glm53Exl3Buffer,
    pub output_f16: Glm53Exl3Buffer,
    pub locks_i32: Glm53Exl3Buffer,
    pub input_hadamard_f16: Glm53Exl3Buffer,
}

#[derive(Clone, Copy)]
pub struct Glm53Exl3ProjectionBuffers {
    pub input_bf16: Glm53Exl3Buffer,
    pub output_bf16: Glm53Exl3Buffer,
    pub scratch: Glm53Exl3ProjectionScratch,
}

impl Glm53Exl3Projection<'_> {
    pub fn plan(self, rows: u32) -> Result<Glm53Exl3ProjectionPlan> {
        ensure!(rows != 0, "GLM EXL3 projection rows must be nonzero");
        let (input, output, compressed) = match self {
            Self::Compressed(linear) => (linear.size_k(), linear.size_n(), true),
            Self::NativeBf16(tensor) => {
                ensure!(
                    tensor.dtype() == Glm53Exl3NativeDtype::Bf16 && tensor.shape().len() == 2,
                    "GLM EXL3 native projection must be rank-2 BF16"
                );
                let output = u32::try_from(tensor.shape()[0])
                    .context("GLM EXL3 native projection N exceeds u32")?;
                let input = u32::try_from(tensor.shape()[1])
                    .context("GLM EXL3 native projection K exceeds u32")?;
                (input, output, false)
            }
            Self::NativeBf16ActWeight(tensor) => {
                ensure!(
                    tensor.dtype() == Glm53Exl3NativeDtype::Bf16 && tensor.shape().len() == 2,
                    "GLM EXL3 native act-weight projection must be rank-2 BF16"
                );
                let input = u32::try_from(tensor.shape()[0])
                    .context("GLM EXL3 native act-weight K exceeds u32")?;
                let output = u32::try_from(tensor.shape()[1])
                    .context("GLM EXL3 native act-weight N exceeds u32")?;
                (input, output, false)
            }
        };
        let bytes = |width: u32| -> Result<usize> {
            usize::try_from(rows)?
                .checked_mul(usize::try_from(width)?)
                .and_then(|elements| elements.checked_mul(2))
                .context("GLM EXL3 projection byte count overflow")
        };
        Ok(Glm53Exl3ProjectionPlan {
            rows,
            input,
            output,
            input_bytes: bytes(input)?,
            output_bytes: bytes(output)?,
            compressed,
        })
    }

    pub fn launch(
        self,
        gpu: &dyn GpuBackend,
        plan: Glm53Exl3ProjectionPlan,
        buffers: Glm53Exl3ProjectionBuffers,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.plan(plan.rows)? == plan,
            "GLM EXL3 projection plan drift"
        );
        validate_activation("input", buffers.input_bf16, plan.input_bytes)?;
        validate_activation("output", buffers.output_bf16, plan.output_bytes)?;
        // Verification needs the tokenwise arithmetic contract without entering
        // the prefill scope, which also publishes persistent convolution state.
        let exact_arithmetic = glm53_exact_wide_prefill_active() || glm53_exact_verify_active();
        let exact_wide_row_exact = plan.rows > 1
            && exact_arithmetic
            && matches!(self, Self::Compressed(_))
            && std::env::var("ATLAS_GLM53_EXACT_WIDE_ROWEXACT").as_deref() == Ok("1");
        if native_rows_selected(
            glm53_exact_verify_active(),
            glm53_exact_wide_prefill_active() || glm53_layer_major_prefill_active(),
            plan.rows,
            !plan.compressed,
        ) {
            use spark_runtime::cublaslt::{ByteSpan, Orientation, SerialRowsRequest};
            ensure!(
                !gpu.stream_is_capturing(stream),
                "prepared native rows require eager verification"
            );
            let (tensor, orientation) = match self {
                Self::NativeBf16(tensor) => (tensor, Orientation::Nk),
                Self::NativeBf16ActWeight(tensor) => (tensor, Orientation::Kn),
                Self::Compressed(_) => {
                    bail!("native serial-row selection admitted compressed weights")
                }
            };
            let weight_bytes = usize::try_from(plan.input)?
                .checked_mul(usize::try_from(plan.output)?)
                .and_then(|elements| elements.checked_mul(2))
                .context("prepared native weight extent overflow")?;
            ensure!(
                tensor.bytes() == weight_bytes,
                "prepared native weight extent drift"
            );
            let request = SerialRowsRequest {
                rows: plan.rows,
                n: plan.output,
                k: plan.input,
                orientation,
                act: ByteSpan {
                    address: buffers.input_bf16.ptr.0,
                    bytes: buffers.input_bf16.bytes,
                },
                weight: ByteSpan {
                    address: tensor.ptr().0,
                    bytes: tensor.bytes(),
                },
                out: ByteSpan {
                    address: buffers.output_bf16.ptr.0,
                    bytes: buffers.output_bf16.bytes,
                },
                stream,
            };
            return if super::glm53_native_rows::glm53_native_rows_batched_active() {
                spark_runtime::cublaslt::bf16_gemm_batched_rows(request)
            } else {
                spark_runtime::cublaslt::bf16_gemm_serial_rows(request)
            };
        }
        if plan.rows > 1 && exact_arithmetic && !exact_wide_row_exact {
            let one = self.plan(1)?;
            for row in 0..plan.rows as usize {
                self.launch(
                    gpu,
                    one,
                    Glm53Exl3ProjectionBuffers {
                        input_bf16: Glm53Exl3Buffer {
                            ptr: buffers.input_bf16.ptr.offset(row * one.input_bytes),
                            bytes: one.input_bytes,
                        },
                        output_bf16: Glm53Exl3Buffer {
                            ptr: buffers.output_bf16.ptr.offset(row * one.output_bytes),
                            bytes: one.output_bytes,
                        },
                        scratch: buffers.scratch,
                    },
                    stream,
                )?;
            }
            return Ok(());
        }
        match self {
            Self::Compressed(linear) => {
                let scratch = exact_scratch(plan, buffers.scratch)?;
                let prepared = if exact_wide_row_exact {
                    linear.prepare_bf16_row_exact(gpu, plan.rows)?
                } else {
                    linear.prepare_bf16(gpu, plan.rows)?
                };
                prepared.launch(
                    gpu,
                    Glm53Exl3Bf16LinearBuffers {
                        input_bf16: buffers.input_bf16,
                        output_bf16: buffers.output_bf16,
                        input_f16: scratch.input_f16,
                        output_f16: scratch.output_f16,
                        locks_i32: scratch.locks_i32,
                        input_hadamard_f16: scratch.input_hadamard_f16,
                    },
                    stream,
                )
            }
            Self::NativeBf16(tensor) => {
                ensure!(
                    tensor.bytes()
                        == usize::try_from(plan.input)?
                            .checked_mul(usize::try_from(plan.output)?)
                            .and_then(|elements| elements.checked_mul(2))
                            .context("GLM EXL3 native weight extent overflow")?,
                    "GLM EXL3 native weight byte extent drift"
                );
                spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    buffers.input_bf16.ptr.0,
                    tensor.ptr().0,
                    buffers.output_bf16.ptr.0,
                    plan.rows,
                    plan.output,
                    plan.input,
                    stream,
                )
            }
            Self::NativeBf16ActWeight(tensor) => {
                ensure!(
                    tensor.bytes()
                        == usize::try_from(plan.input)?
                            .checked_mul(usize::try_from(plan.output)?)
                            .and_then(|elements| elements.checked_mul(2))
                            .context("GLM EXL3 native act-weight extent overflow")?,
                    "GLM EXL3 native act-weight byte extent drift"
                );
                spark_runtime::cublaslt::bf16_gemm_act_weight(
                    buffers.input_bf16.ptr.0,
                    tensor.ptr().0,
                    buffers.output_bf16.ptr.0,
                    plan.rows,
                    plan.output,
                    plan.input,
                    stream,
                )
            }
        }
    }
}

fn exact_scratch(
    plan: Glm53Exl3ProjectionPlan,
    scratch: Glm53Exl3ProjectionScratch,
) -> Result<Glm53Exl3ProjectionScratch> {
    ensure!(
        plan.input_bytes <= GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES
            && plan.output_bytes <= GLM53_EXL3_MAX_WIDE_OUTPUT_F16_BYTES,
        "GLM EXL3 projection exceeds target scratch maxima"
    );
    Ok(Glm53Exl3ProjectionScratch {
        input_f16: prefix("input F16", scratch.input_f16, plan.input_bytes)?,
        output_f16: prefix("output F16", scratch.output_f16, plan.output_bytes)?,
        locks_i32: exact("locks", scratch.locks_i32, GLM53_EXL3_LOCK_BYTES)?,
        input_hadamard_f16: prefix(
            "input Hadamard F16",
            scratch.input_hadamard_f16,
            plan.input_bytes,
        )?,
    })
}

fn validate_activation(name: &str, buffer: Glm53Exl3Buffer, bytes: usize) -> Result<()> {
    exact(name, buffer, bytes).map(|_| ())
}

fn exact(name: &str, buffer: Glm53Exl3Buffer, bytes: usize) -> Result<Glm53Exl3Buffer> {
    if buffer.ptr == DevicePtr::NULL || buffer.bytes != bytes {
        bail!("GLM EXL3 projection {name} is null or has the wrong exact extent");
    }
    buffer
        .ptr
        .0
        .checked_add(u64::try_from(bytes)?)
        .with_context(|| format!("GLM EXL3 projection {name} address overflow"))?;
    Ok(buffer)
}

fn prefix(name: &str, buffer: Glm53Exl3Buffer, bytes: usize) -> Result<Glm53Exl3Buffer> {
    if buffer.ptr == DevicePtr::NULL || buffer.bytes < bytes {
        bail!("GLM EXL3 projection {name} scratch is null or too small");
    }
    buffer
        .ptr
        .0
        .checked_add(u64::try_from(bytes)?)
        .with_context(|| format!("GLM EXL3 projection {name} address overflow"))?;
    Ok(Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_scratch_maxima_cover_lm_head_and_widest_input() {
        assert_eq!(GLM53_EXL3_MAX_INPUT_F16_BYTES, 32_768);
        assert_eq!(GLM53_EXL3_MAX_OUTPUT_F16_BYTES, 309_760);
        assert_eq!(GLM53_EXL3_LOCK_BYTES, 4_194_304);
    }

    #[test]
    fn exact_wide_prefill_scope_is_nested_and_thread_local() {
        assert!(!glm53_exact_wide_prefill_active());
        with_glm53_exact_wide_prefill(|| {
            assert!(glm53_exact_wide_prefill_active());
            with_glm53_exact_wide_prefill(|| assert!(glm53_exact_wide_prefill_active()));
            assert!(glm53_exact_wide_prefill_active());
        });
        assert!(!glm53_exact_wide_prefill_active());
    }

    #[test]
    fn layer_major_prefill_scope_is_nested_and_does_not_enable_exact_wide() {
        assert!(!glm53_layer_major_prefill_active());
        assert!(!glm53_exact_wide_prefill_active());
        with_glm53_layer_major_prefill(|| {
            assert!(glm53_layer_major_prefill_active());
            assert!(!glm53_exact_wide_prefill_active());
            with_glm53_layer_major_prefill(|| assert!(glm53_layer_major_prefill_active()));
            assert!(glm53_layer_major_prefill_active());
        });
        assert!(!glm53_layer_major_prefill_active());
        assert!(!glm53_exact_wide_prefill_active());
    }

    #[test]
    fn scratch_slices_exact_extents_and_refuses_short_buffers() {
        let buffer = |ptr, bytes| Glm53Exl3Buffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let plan = Glm53Exl3ProjectionPlan {
            rows: 1,
            input: 4096,
            output: 24_576,
            input_bytes: 8192,
            output_bytes: 49_152,
            compressed: true,
        };
        let scratch = Glm53Exl3ProjectionScratch {
            input_f16: buffer(1, GLM53_EXL3_MAX_INPUT_F16_BYTES),
            output_f16: buffer(2, GLM53_EXL3_MAX_OUTPUT_F16_BYTES),
            locks_i32: buffer(3, GLM53_EXL3_LOCK_BYTES),
            input_hadamard_f16: buffer(4, GLM53_EXL3_MAX_INPUT_F16_BYTES),
        };
        let exact = exact_scratch(plan, scratch).unwrap();
        assert_eq!(exact.input_f16.bytes, plan.input_bytes);
        assert_eq!(exact.output_f16.bytes, plan.output_bytes);
        assert_eq!(exact.input_hadamard_f16.bytes, plan.input_bytes);
        assert!(
            exact_scratch(
                plan,
                Glm53Exl3ProjectionScratch {
                    output_f16: buffer(2, plan.output_bytes - 2),
                    ..scratch
                }
            )
            .is_err()
        );
    }
}
