// SPDX-License-Identifier: AGPL-3.0-only

//! Typed projection binding for the admitted GLM-5.3 EXL3 device store.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Glm53Exl3DeviceStore, Glm53Exl3DeviceTensor, Glm53Exl3Dtype};
use crate::layers::ops::{
    Glm53Exl3Buffer, Glm53Exl3CastKernels, Glm53Exl3CastPlan, Glm53Exl3GemmBuffers,
    Glm53Exl3GemmKernel, Glm53Exl3GemmPlan, Glm53Exl3Output,
};

const TRELLIS_TILE: u64 = 16;

/// Non-owning, exact device views for one EXL3 linear projection.
///
/// The [`Glm53Exl3DeviceStore`] that produced this binding must outlive it.
#[derive(Clone, PartialEq, Eq)]
pub struct Glm53Exl3Linear {
    logical_name: String,
    size_k: u32,
    size_n: u32,
    bits: u8,
    trellis: Glm53Exl3Buffer,
    scale_in: Glm53Exl3Buffer,
    scale_out: Glm53Exl3Buffer,
}

/// Loaded cooperative kernel plus the immutable plan for one invocation shape.
pub struct PreparedGlm53Exl3Linear {
    plan: Glm53Exl3GemmPlan,
    kernel: Glm53Exl3GemmKernel,
    trellis: Glm53Exl3Buffer,
    scale_in: Glm53Exl3Buffer,
    scale_out: Glm53Exl3Buffer,
}

/// Prepared F16 EXL3 projection wrapped for Atlas's BF16 activation graph.
pub struct PreparedGlm53Exl3Bf16Linear {
    linear: PreparedGlm53Exl3Linear,
    casts: Glm53Exl3CastKernels,
    input_cast: Glm53Exl3CastPlan,
    output_cast: Glm53Exl3CastPlan,
}

/// Caller-owned activation and workspace buffers for an EXL3 invocation.
#[derive(Clone, Copy)]
pub struct Glm53Exl3LinearBuffers {
    pub input_f16: Glm53Exl3Buffer,
    pub output: Glm53Exl3Buffer,
    pub locks_i32: Glm53Exl3Buffer,
    pub input_hadamard_f16: Glm53Exl3Buffer,
}

/// Caller-owned graph buffers and dtype scratch for one BF16 EXL3 invocation.
#[derive(Clone, Copy)]
pub struct Glm53Exl3Bf16LinearBuffers {
    pub input_bf16: Glm53Exl3Buffer,
    pub output_bf16: Glm53Exl3Buffer,
    pub input_f16: Glm53Exl3Buffer,
    pub output_f16: Glm53Exl3Buffer,
    pub locks_i32: Glm53Exl3Buffer,
    pub input_hadamard_f16: Glm53Exl3Buffer,
}

impl Glm53Exl3Linear {
    /// Bind `<logical>.{trellis,suh,svh,mul1}` without copying or converting.
    pub fn bind(store: &Glm53Exl3DeviceStore, logical_name: &str) -> Result<Self> {
        ensure!(
            !logical_name.is_empty() && !logical_name.ends_with('.'),
            "GLM EXL3 logical projection name is empty or malformed"
        );
        let names =
            ["trellis", "suh", "svh", "mul1"].map(|suffix| format!("{logical_name}.{suffix}"));
        bind_parts(
            logical_name,
            store.get(&names[0]),
            store.get(&names[1]),
            store.get(&names[2]),
            store.get(&names[3]),
        )
    }

    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    pub fn size_k(&self) -> u32 {
        self.size_k
    }

    pub fn size_n(&self) -> u32 {
        self.size_n
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn trellis_buffer(&self) -> Glm53Exl3Buffer {
        self.trellis
    }

    pub fn scale_in_buffer(&self) -> Glm53Exl3Buffer {
        self.scale_in
    }

    pub fn scale_out_buffer(&self) -> Glm53Exl3Buffer {
        self.scale_out
    }

    /// Resolve the exact pinned kernel and invocation plan before execution.
    pub fn prepare(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        output: Glm53Exl3Output,
    ) -> Result<PreparedGlm53Exl3Linear> {
        let plan = Glm53Exl3GemmPlan::new(rows, self.size_k, self.size_n, self.bits, output)?;
        ensure!(
            plan.trellis_bytes == self.trellis.bytes
                && plan.scale_in_bytes == self.scale_in.bytes
                && plan.scale_out_bytes == self.scale_out.bytes,
            "GLM EXL3 bound projection no longer matches its execution plan"
        );
        let kernel = Glm53Exl3GemmKernel::load(gpu, &plan)?;
        Ok(PreparedGlm53Exl3Linear {
            plan,
            kernel,
            trellis: self.trellis,
            scale_in: self.scale_in,
            scale_out: self.scale_out,
        })
    }

    fn prepare_row_exact(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        output: Glm53Exl3Output,
    ) -> Result<PreparedGlm53Exl3Linear> {
        let plan =
            Glm53Exl3GemmPlan::new_regular_gemm(rows, self.size_k, self.size_n, self.bits, output)?;
        ensure!(
            plan.trellis_bytes == self.trellis.bytes
                && plan.scale_in_bytes == self.scale_in.bytes
                && plan.scale_out_bytes == self.scale_out.bytes,
            "GLM EXL3 bound projection no longer matches its row-exact plan"
        );
        let kernel = Glm53Exl3GemmKernel::load_row_exact(gpu, &plan)?;
        Ok(PreparedGlm53Exl3Linear {
            plan,
            kernel,
            trellis: self.trellis,
            scale_in: self.scale_in,
            scale_out: self.scale_out,
        })
    }

    /// Resolve the exact GEMM plus the BF16/F16 graph bridge.
    pub fn prepare_bf16(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
    ) -> Result<PreparedGlm53Exl3Bf16Linear> {
        Ok(PreparedGlm53Exl3Bf16Linear {
            linear: self.prepare(gpu, rows, Glm53Exl3Output::F16)?,
            casts: Glm53Exl3CastKernels::load(gpu)?,
            input_cast: Glm53Exl3CastPlan::new(rows, self.size_k)?,
            output_cast: Glm53Exl3CastPlan::new(rows, self.size_n)?,
        })
    }

    /// Resolve the cooperative row-exact small-M kernel plus BF16/F16 bridges.
    ///
    /// This is restricted by the kernel loader to 2..=8 rows and preserves the
    /// pinned M=1 arithmetic independently for every row.
    pub fn prepare_bf16_row_exact(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
    ) -> Result<PreparedGlm53Exl3Bf16Linear> {
        Ok(PreparedGlm53Exl3Bf16Linear {
            linear: self.prepare_row_exact(gpu, rows, Glm53Exl3Output::F16)?,
            casts: Glm53Exl3CastKernels::load(gpu)?,
            input_cast: Glm53Exl3CastPlan::new(rows, self.size_k)?,
            output_cast: Glm53Exl3CastPlan::new(rows, self.size_n)?,
        })
    }
}

impl PreparedGlm53Exl3Linear {
    pub fn plan(&self) -> &Glm53Exl3GemmPlan {
        &self.plan
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        buffers: Glm53Exl3LinearBuffers,
        stream: u64,
    ) -> Result<()> {
        self.kernel.launch(
            gpu,
            &self.plan,
            Glm53Exl3GemmBuffers {
                input_f16: buffers.input_f16,
                trellis_i16: self.trellis,
                output: buffers.output,
                locks_i32: buffers.locks_i32,
                scale_in_f16: self.scale_in,
                input_hadamard_f16: buffers.input_hadamard_f16,
                scale_out_f16: self.scale_out,
            },
            stream,
        )
    }
}

impl PreparedGlm53Exl3Bf16Linear {
    pub fn plan(&self) -> &Glm53Exl3GemmPlan {
        self.linear.plan()
    }

    /// Convert BF16 input, execute EXL3, and convert its F16 result to BF16.
    ///
    /// All three launches stay on `stream`; this method does not allocate or
    /// synchronize, so callers retain graph-capture and ownership control.
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        buffers: Glm53Exl3Bf16LinearBuffers,
        stream: u64,
    ) -> Result<()> {
        self.casts.bf16_to_f16(
            gpu,
            self.input_cast,
            buffers.input_bf16,
            buffers.input_f16,
            stream,
        )?;
        self.linear.launch(
            gpu,
            Glm53Exl3LinearBuffers {
                input_f16: buffers.input_f16,
                output: buffers.output_f16,
                locks_i32: buffers.locks_i32,
                input_hadamard_f16: buffers.input_hadamard_f16,
            },
            stream,
        )?;
        self.casts.f16_to_bf16(
            gpu,
            self.output_cast,
            buffers.output_f16,
            buffers.output_bf16,
            stream,
        )
    }
}

fn bind_parts(
    logical_name: &str,
    trellis: Option<&Glm53Exl3DeviceTensor>,
    scale_in: Option<&Glm53Exl3DeviceTensor>,
    scale_out: Option<&Glm53Exl3DeviceTensor>,
    mul1: Option<&Glm53Exl3DeviceTensor>,
) -> Result<Glm53Exl3Linear> {
    let trellis = required(logical_name, "trellis", trellis)?;
    let scale_in = required(logical_name, "suh", scale_in)?;
    let scale_out = required(logical_name, "svh", scale_out)?;
    let mul1 = required(logical_name, "mul1", mul1)?;
    ensure_tensor(trellis, Glm53Exl3Dtype::I16, 3, "trellis")?;
    ensure_tensor(scale_in, Glm53Exl3Dtype::F16, 1, "suh")?;
    ensure_tensor(scale_out, Glm53Exl3Dtype::F16, 1, "svh")?;
    ensure_tensor(mul1, Glm53Exl3Dtype::I32, 0, "mul1")?;
    ensure!(mul1.byte_len == 4, "GLM EXL3 mul1 marker must be one i32");

    let size_k = only_dimension(scale_in, "suh")?;
    let size_n = only_dimension(scale_out, "svh")?;
    let encoded_bits = *trellis
        .shape
        .get(2)
        .context("GLM EXL3 trellis is missing its encoded-bit dimension")?;
    ensure!(
        encoded_bits % TRELLIS_TILE == 0,
        "GLM EXL3 trellis bit dimension is not tile-aligned"
    );
    let bits = u8::try_from(encoded_bits / TRELLIS_TILE)?;
    ensure!(
        matches!(bits, 2..=5),
        "GLM EXL3 projection bit width is unsupported"
    );
    ensure!(
        trellis.shape == [size_k / TRELLIS_TILE, size_n / TRELLIS_TILE, encoded_bits],
        "GLM EXL3 trellis geometry does not match suh/svh dimensions"
    );

    let size_k = u32::try_from(size_k).context("GLM EXL3 input dimension exceeds u32")?;
    let size_n = u32::try_from(size_n).context("GLM EXL3 output dimension exceeds u32")?;
    let plan = Glm53Exl3GemmPlan::new(1, size_k, size_n, bits, Glm53Exl3Output::F16)?;
    ensure!(
        trellis.byte_len == plan.trellis_bytes
            && scale_in.byte_len == plan.scale_in_bytes
            && scale_out.byte_len == plan.scale_out_bytes,
        "GLM EXL3 projection tensor byte extents do not match its geometry"
    );
    Ok(Glm53Exl3Linear {
        logical_name: logical_name.to_owned(),
        size_k,
        size_n,
        bits,
        trellis: buffer(trellis),
        scale_in: buffer(scale_in),
        scale_out: buffer(scale_out),
    })
}

fn required<'a>(
    logical_name: &str,
    suffix: &str,
    tensor: Option<&'a Glm53Exl3DeviceTensor>,
) -> Result<&'a Glm53Exl3DeviceTensor> {
    tensor.with_context(|| format!("GLM EXL3 projection {logical_name} is missing .{suffix}"))
}

fn ensure_tensor(
    tensor: &Glm53Exl3DeviceTensor,
    dtype: Glm53Exl3Dtype,
    rank: usize,
    suffix: &str,
) -> Result<()> {
    if tensor.ptr == DevicePtr::NULL || tensor.dtype != dtype || tensor.shape.len() != rank {
        bail!("GLM EXL3 {suffix} device tensor has the wrong pointer, dtype, or rank");
    }
    Ok(())
}

fn only_dimension(tensor: &Glm53Exl3DeviceTensor, suffix: &str) -> Result<u64> {
    let dimension = *tensor
        .shape
        .first()
        .with_context(|| format!("GLM EXL3 {suffix} dimension is missing"))?;
    ensure!(
        dimension != 0,
        "GLM EXL3 {suffix} dimension must be nonzero"
    );
    Ok(dimension)
}

fn buffer(tensor: &Glm53Exl3DeviceTensor) -> Glm53Exl3Buffer {
    Glm53Exl3Buffer {
        ptr: tensor.ptr,
        bytes: tensor.byte_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(ptr: u64, shape: &[u64], dtype: Glm53Exl3Dtype) -> Glm53Exl3DeviceTensor {
        let elements = shape.iter().copied().product::<u64>();
        Glm53Exl3DeviceTensor {
            ptr: DevicePtr(ptr),
            shape: shape.to_vec(),
            dtype,
            byte_len: usize::try_from(elements * dtype.byte_size()).unwrap(),
            shard: 0,
        }
    }

    fn valid_parts() -> [Glm53Exl3DeviceTensor; 4] {
        [
            tensor(1, &[256, 1536, 64], Glm53Exl3Dtype::I16),
            tensor(2, &[4096], Glm53Exl3Dtype::F16),
            tensor(3, &[24576], Glm53Exl3Dtype::F16),
            tensor(4, &[], Glm53Exl3Dtype::I32),
        ]
    }

    #[test]
    fn exact_qkv_quartet_binds_to_k4_projection() {
        let parts = valid_parts();
        let linear = bind_parts(
            "model.language_model.layers.0.self_attn.qkv_proj",
            Some(&parts[0]),
            Some(&parts[1]),
            Some(&parts[2]),
            Some(&parts[3]),
        )
        .unwrap();
        assert_eq!(
            (linear.size_k(), linear.size_n(), linear.bits()),
            (4096, 24576, 4)
        );
        assert_eq!(linear.trellis.bytes, 50_331_648);
    }

    #[test]
    fn missing_or_mistyped_quartet_fails_closed() {
        let mut parts = valid_parts();
        assert!(
            bind_parts(
                "qkv",
                None,
                Some(&parts[1]),
                Some(&parts[2]),
                Some(&parts[3])
            )
            .is_err()
        );
        parts[3].dtype = Glm53Exl3Dtype::F32;
        assert!(
            bind_parts(
                "qkv",
                Some(&parts[0]),
                Some(&parts[1]),
                Some(&parts[2]),
                Some(&parts[3]),
            )
            .is_err()
        );
    }

    #[test]
    fn inconsistent_trellis_geometry_fails_closed() {
        let mut parts = valid_parts();
        parts[0].shape[1] -= 1;
        assert!(
            bind_parts(
                "qkv",
                Some(&parts[0]),
                Some(&parts[1]),
                Some(&parts[2]),
                Some(&parts[3]),
            )
            .is_err()
        );
    }
}
