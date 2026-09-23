// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
const THREADS: u32 = 128;
const L2_EPS: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaDecodePlan {
    pub batch: u32,
    pub groups: u32,
    pub vector_bytes: usize,
    pub decay_bytes: usize,
    pub beta_bytes: usize,
    pub state_bytes: usize,
}

impl Glm53KdaDecodePlan {
    pub fn new(batch: u32, heads: u32, key_dim: u32, value_dim: u32) -> Result<Self> {
        if batch == 0 {
            bail!("GLM KDA decode requires at least one sequence");
        }
        if heads != HEADS || key_dim != HEAD_DIM || value_dim != HEAD_DIM {
            bail!("GLM KDA decode requires exact 64 heads x K128 x V128");
        }
        let groups = batch
            .checked_mul(heads)
            .context("GLM KDA decode CUDA grid overflow")?;
        let vectors = usize::try_from(groups)?
            .checked_mul(usize::try_from(key_dim)?)
            .context("GLM KDA decode vector element overflow")?;
        let states = vectors
            .checked_mul(usize::try_from(value_dim)?)
            .context("GLM KDA decode state element overflow")?;
        Ok(Self {
            batch,
            groups,
            vector_bytes: vectors
                .checked_mul(2)
                .context("GLM KDA decode vector byte overflow")?,
            decay_bytes: vectors
                .checked_mul(4)
                .context("GLM KDA decode decay byte overflow")?,
            beta_bytes: usize::try_from(groups)?
                .checked_mul(2)
                .context("GLM KDA decode beta byte overflow")?,
            state_bytes: states
                .checked_mul(4)
                .context("GLM KDA decode state byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaDecodeBuffers {
    pub state_f32: GgmlIqBuffer,
    pub query_bf16: GgmlIqBuffer,
    pub key_bf16: GgmlIqBuffer,
    pub value_bf16: GgmlIqBuffer,
    pub log_decay_f32: GgmlIqBuffer,
    pub beta_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53KdaDecodeKernel {
    decode: KernelHandle,
    decode_oop: KernelHandle,
}

impl Glm53KdaDecodeKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            decode: gpu.kernel("glm53_kda", "atlas_glm53_kda_decode")?,
            decode_oop: gpu.kernel("glm53_kda", "atlas_glm53_kda_decode_oop")?,
        })
    }

    /// Out-of-place recurrence: read `state_in`, write `buffers.state_f32`.
    /// `state_in == buffers.state_f32` runs in place. Same arithmetic as
    /// [`Self::launch`]; only the source pointer differs.
    pub fn launch_oop(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaDecodePlan,
        state_in: GgmlIqBuffer,
        buffers: Glm53KdaDecodeBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        if state_in.bytes != plan.state_bytes || state_in.ptr == DevicePtr::NULL {
            bail!("GLM KDA decode state_in buffer is null or has the wrong extent");
        }
        KernelLaunch::new(gpu, self.decode_oop)
            .grid([plan.groups, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(state_in.ptr)
            .arg_ptr(buffers.state_f32.ptr)
            .arg_ptr(buffers.query_bf16.ptr)
            .arg_ptr(buffers.key_bf16.ptr)
            .arg_ptr(buffers.value_bf16.ptr)
            .arg_ptr(buffers.log_decay_f32.ptr)
            .arg_ptr(buffers.beta_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(HEAD_DIM)
            .arg_f32(L2_EPS)
            .launch(stream)
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaDecodePlan,
        buffers: Glm53KdaDecodeBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.decode)
            .grid([plan.groups, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.state_f32.ptr)
            .arg_ptr(buffers.query_bf16.ptr)
            .arg_ptr(buffers.key_bf16.ptr)
            .arg_ptr(buffers.value_bf16.ptr)
            .arg_ptr(buffers.log_decay_f32.ptr)
            .arg_ptr(buffers.beta_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(HEAD_DIM)
            .arg_f32(L2_EPS)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53KdaDecodePlan, buffers: Glm53KdaDecodeBuffers) -> Result<()> {
    let named = [
        ("state", buffers.state_f32, plan.state_bytes),
        ("query", buffers.query_bf16, plan.vector_bytes),
        ("key", buffers.key_bf16, plan.vector_bytes),
        ("value", buffers.value_bf16, plan.vector_bytes),
        ("log decay", buffers.log_decay_f32, plan.decay_bytes),
        ("beta", buffers.beta_bf16, plan.beta_bytes),
        ("output", buffers.output_bf16, plan.vector_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.bytes != expected || buffer.ptr == DevicePtr::NULL {
            bail!("GLM KDA decode {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM KDA decode {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM KDA decode device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn reference(mut state: [f32; 4], log_decay: [f32; 2]) -> ([f32; 2], [f32; 4]) {
        let query = [0.5f32, -1.0];
        let key = [1.5f32, 0.25];
        let value = [0.75f32, -0.5];
        let q_norm = (query.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
        let k_norm = (key.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
        let query = query.map(|x| x / q_norm * (2.0f32).sqrt().recip());
        let key = key.map(|x| x / k_norm);
        let decay = log_decay.map(f32::exp);
        for row in 0..2 {
            for column in 0..2 {
                state[row * 2 + column] *= decay[row];
            }
        }
        let mut memory = [0.0f32; 2];
        for column in 0..2 {
            memory[column] = state[column] * key[0] + state[2 + column] * key[1];
        }
        let delta = [
            (value[0] - memory[0]) * 0.625,
            (value[1] - memory[1]) * 0.625,
        ];
        for row in 0..2 {
            for column in 0..2 {
                state[row * 2 + column] += key[row] * delta[column];
            }
        }
        let output = [
            state[0] * query[0] + state[2] * query[1],
            state[1] * query[0] + state[3] * query[1],
        ];
        (output, state)
    }

    #[test]
    fn exact_decode_geometry_and_vector_decay_are_pinned() {
        let plan = Glm53KdaDecodePlan::new(2, 64, 128, 128).unwrap();
        assert_eq!(plan.groups, 128);
        assert_eq!(plan.vector_bytes, 2 * 64 * 128 * 2);
        assert_eq!(plan.decay_bytes, 2 * 64 * 128 * 4);
        assert_eq!(plan.beta_bytes, 2 * 64 * 2);
        assert_eq!(plan.state_bytes, 2 * 64 * 128 * 128 * 4);
        assert!(Glm53KdaDecodePlan::new(0, 64, 128, 128).is_err());
        assert!(Glm53KdaDecodePlan::new(1, 64, 64, 128).is_err());

        let state = [1.0, 2.0, 3.0, 4.0];
        let vector_decay = reference(state, [0.5f32.ln(), 0.25f32.ln()]);
        let scalar_decay = reference(state, [0.5f32.ln(), 0.5f32.ln()]);
        assert_ne!(vector_decay, scalar_decay);
    }

    #[test]
    fn launch_rejects_alias_before_effect() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53KdaDecodeKernel::load(&gpu).unwrap();
        let plan = Glm53KdaDecodePlan::new(1, 64, 128, 128).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53KdaDecodeBuffers {
            state_f32: at(0x10_0000, plan.state_bytes),
            query_bf16: at(0x60_0000, plan.vector_bytes),
            key_bf16: at(0x70_0000, plan.vector_bytes),
            value_bf16: at(0x80_0000, plan.vector_bytes),
            log_decay_f32: at(0x90_0000, plan.decay_bytes),
            beta_bf16: at(0xa0_0000, plan.beta_bytes),
            output_bf16: at(0xb0_0000, plan.vector_bytes),
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53KdaDecodeBuffers {
                        output_bf16: valid.query_bf16,
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }

    const KDA_CUDA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu"
    ));

    fn oop_fixture(gpu: &MockGpuBackend) -> (Glm53KdaDecodePlan, GgmlIqBuffer, Glm53KdaDecodeBuffers) {
        let plan = Glm53KdaDecodePlan::new(1, HEADS, HEAD_DIM, HEAD_DIM).unwrap();
        let at = |bytes: usize| GgmlIqBuffer {
            ptr: gpu.alloc(bytes).unwrap(),
            bytes,
        };
        let buffers = Glm53KdaDecodeBuffers {
            state_f32: at(plan.state_bytes),
            query_bf16: at(plan.vector_bytes),
            key_bf16: at(plan.vector_bytes),
            value_bf16: at(plan.vector_bytes),
            log_decay_f32: at(plan.decay_bytes),
            beta_bf16: at(plan.beta_bytes),
            output_bf16: at(plan.vector_bytes),
        };
        (plan, at(plan.state_bytes), buffers)
    }

    #[test]
    fn out_of_place_decode_launches_once_and_refuses_a_short_source() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53KdaDecodeKernel::load(&gpu).unwrap();
        let (plan, state_in, buffers) = oop_fixture(&gpu);
        kernel.launch_oop(&gpu, plan, state_in, buffers, 7).unwrap();
        assert_eq!(gpu.launch_count(), 1);
        // In place (source == destination) is admitted: the one-row walk uses it.
        kernel
            .launch_oop(&gpu, plan, buffers.state_f32, buffers, 7)
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
        let short = GgmlIqBuffer {
            ptr: state_in.ptr,
            bytes: state_in.bytes - 4,
        };
        assert!(kernel.launch_oop(&gpu, plan, short, buffers, 7).is_err());
        let null = GgmlIqBuffer {
            ptr: DevicePtr::NULL,
            bytes: state_in.bytes,
        };
        assert!(kernel.launch_oop(&gpu, plan, null, buffers, 7).is_err());
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn out_of_place_decode_kernel_reads_source_once_and_admits_aliasing() {
        let body = KDA_CUDA
            .split("atlas_glm53_kda_decode_oop(")
            .nth(1)
            .expect("out-of-place decode kernel present")
            .split("extern \"C\"")
            .next()
            .unwrap();
        assert!(body.starts_with("\n        const float * state_in,\n        float * state_out,"));
        assert!(!body.contains("__restrict__ state"));
        assert!(body.contains("const float decayed = state_in[index] * decay[at];"));
        assert!(body.contains("state_out[index] = decayed;"));
        assert!(body.contains("const float updated = state_out[index] + k_values[at] * delta;"));
        assert!(!body.contains("state[index]"));
        // Same normalization, decay and update expressions as the in-place kernel.
        let in_place = KDA_CUDA
            .split("atlas_glm53_kda_decode(")
            .nth(1)
            .unwrap()
            .split("extern \"C\"")
            .next()
            .unwrap();
        for expression in [
            "(q_values[column] / q_norm) * (1.0f / sqrtf(128.0f))",
            "k_values[column] = k_values[column] / k_norm;",
            "(__bfloat162float(value[vector_base + column]) - memory) * beta_value;",
            "output[vector_base + column] = __float2bfloat16_rn(result);",
        ] {
            assert!(in_place.contains(expression) && body.contains(expression));
        }
    }
}
