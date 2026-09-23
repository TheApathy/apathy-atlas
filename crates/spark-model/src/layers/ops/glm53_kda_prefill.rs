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
pub struct Glm53KdaPrefillPlan {
    pub batch: u32,
    pub tokens: u32,
    pub groups: u32,
    pub vector_bytes: usize,
    pub decay_bytes: usize,
    pub beta_bytes: usize,
    pub state_bytes: usize,
}

impl Glm53KdaPrefillPlan {
    pub fn new(batch: u32, tokens: u32, heads: u32, key_dim: u32, value_dim: u32) -> Result<Self> {
        if batch == 0 || tokens == 0 {
            bail!("GLM KDA prefill requires nonzero batch and token counts");
        }
        if heads != HEADS || key_dim != HEAD_DIM || value_dim != HEAD_DIM {
            bail!("GLM KDA prefill requires exact 64 heads x K128 x V128");
        }
        let groups = batch
            .checked_mul(heads)
            .context("GLM KDA prefill CUDA grid overflow")?;
        let token_groups = usize::try_from(batch)?
            .checked_mul(usize::try_from(tokens)?)
            .and_then(|count| count.checked_mul(usize::try_from(heads).ok()?))
            .context("GLM KDA prefill token-group overflow")?;
        let vectors = token_groups
            .checked_mul(usize::try_from(key_dim)?)
            .context("GLM KDA prefill vector element overflow")?;
        let states = usize::try_from(groups)?
            .checked_mul(usize::try_from(key_dim)?)
            .and_then(|count| count.checked_mul(usize::try_from(value_dim).ok()?))
            .context("GLM KDA prefill state element overflow")?;
        Ok(Self {
            batch,
            tokens,
            groups,
            vector_bytes: vectors
                .checked_mul(2)
                .context("GLM KDA prefill vector byte overflow")?,
            decay_bytes: vectors
                .checked_mul(4)
                .context("GLM KDA prefill decay byte overflow")?,
            beta_bytes: token_groups
                .checked_mul(2)
                .context("GLM KDA prefill beta byte overflow")?,
            state_bytes: states
                .checked_mul(4)
                .context("GLM KDA prefill state byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaPrefillBuffers {
    pub state_f32: GgmlIqBuffer,
    pub query_bf16: GgmlIqBuffer,
    pub key_bf16: GgmlIqBuffer,
    pub value_bf16: GgmlIqBuffer,
    pub log_decay_f32: GgmlIqBuffer,
    pub beta_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53KdaPrefillKernel {
    prefill: KernelHandle,
    prefill_oop: KernelHandle,
    register_resident: KernelHandle,
    register_resident_c8: KernelHandle,
    register_resident_c8_enabled: bool,
    register_resident_columns: u32,
}

impl Glm53KdaPrefillKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let columns_env = std::env::var("ATLAS_GLM53_KDA_RR_COLUMNS").ok();
        let register_resident_c8_enabled = parse_rr_c8_enabled(columns_env.as_deref())?;
        let (register_resident_columns, columns_symbol) = rr_columns(columns_env.as_deref());
        Ok(Self {
            prefill: gpu.kernel("glm53_kda", "atlas_glm53_kda_prefill")?,
            prefill_oop: gpu.kernel("glm53_kda", "atlas_glm53_kda_prefill_oop")?,
            register_resident: gpu
                .kernel("glm53_kda", "atlas_glm53_kda_prefill_register_resident")?,
            register_resident_c8: gpu.kernel(
                "glm53_kda_rr_columns",
                match std::env::var("ATLAS_GLM53_KDA_RR_PREFETCH").as_deref() {
                    Ok("1") => "atlas_glm53_kda_prefill_rr_c8_pf",
                    Ok("2") => "atlas_glm53_kda_prefill_rr_c8_pf2",
                    _ => columns_symbol,
                },
            )?,
            register_resident_c8_enabled,
            register_resident_columns,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaPrefillPlan,
        buffers: Glm53KdaPrefillBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.prefill)
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
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(HEAD_DIM)
            .arg_f32(L2_EPS)
            .launch(stream)
    }

    /// Out-of-place recurrence: token 0 reads `state_in`, every token writes
    /// `buffers.state_f32`. Same arithmetic as [`Self::launch`].
    pub fn launch_oop(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaPrefillPlan,
        state_in: GgmlIqBuffer,
        buffers: Glm53KdaPrefillBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        if state_in.bytes != plan.state_bytes || state_in.ptr == DevicePtr::NULL {
            bail!("GLM KDA prefill state_in buffer is null or has the wrong extent");
        }
        KernelLaunch::new(gpu, self.prefill_oop)
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
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(HEAD_DIM)
            .arg_f32(L2_EPS)
            .launch(stream)
    }

    pub fn launch_register_resident(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaPrefillPlan,
        buffers: Glm53KdaPrefillBuffers,
        stream: u64,
    ) -> Result<()> {
        if plan.tokens <= 8 {
            bail!("GLM KDA register-resident prefill requires more than 8 tokens");
        }
        validate_buffers(plan, buffers)?;
        let (kernel, columns) = if self.register_resident_c8_enabled {
            (self.register_resident_c8, self.register_resident_columns)
        } else {
            (self.register_resident, 1)
        };
        KernelLaunch::new(gpu, kernel)
            .grid([plan.groups, HEAD_DIM / (4 * columns), 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.state_f32.ptr)
            .arg_ptr(buffers.query_bf16.ptr)
            .arg_ptr(buffers.key_bf16.ptr)
            .arg_ptr(buffers.value_bf16.ptr)
            .arg_ptr(buffers.log_decay_f32.ptr)
            .arg_ptr(buffers.beta_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(HEAD_DIM)
            .arg_f32(L2_EPS)
            .launch(stream)
    }
}

fn parse_rr_c8_enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") | Some("4") | Some("2") => Ok(true),
        Some("0") => Ok(false),
        Some(value) => {
            bail!("ATLAS_GLM53_KDA_RR_COLUMNS must be absent or exactly 0, 1, 4 or 2; got {value:?}")
        }
    }
}

/// Columns per warp of the register-resident kernel selected by
/// `ATLAS_GLM53_KDA_RR_COLUMNS` (8 default; 4 / 2 = narrower groups, more warps).
fn rr_columns(value: Option<&str>) -> (u32, &'static str) {
    match value {
        Some("4") => (4, "atlas_glm53_kda_prefill_rr_c4"),
        Some("2") => (2, "atlas_glm53_kda_prefill_rr_c2"),
        _ => (8, "atlas_glm53_kda_prefill_rr_c8"),
    }
}

fn validate_buffers(plan: Glm53KdaPrefillPlan, buffers: Glm53KdaPrefillBuffers) -> Result<()> {
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
            bail!("GLM KDA prefill {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM KDA prefill {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM KDA prefill device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    const CUDA_SOURCE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu"
    ));
    const RR_COLUMNS_CUDA_SOURCE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/iq3/glm53_kda_rr_columns.cu"
    ));
    /// The exl3 overlay `#include`s the iq3 file above and adds the prefetch
    /// and narrow-column variants.
    const RR_COLUMNS_EXL3_CUDA_SOURCE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_kda_rr_columns.cu"
    ));

    fn original_tree(mut values: [f32; 128]) -> f32 {
        let mut stride = 64;
        while stride > 0 {
            for index in 0..stride {
                values[index] += values[index + stride];
            }
            stride >>= 1;
        }
        values[0]
    }

    fn register_tree(values: [f32; 128]) -> f32 {
        let mut lanes = [0.0f32; 32];
        for lane in 0..32 {
            lanes[lane] =
                (values[lane] + values[lane + 64]) + (values[lane + 32] + values[lane + 96]);
        }
        let mut offset = 16;
        while offset > 0 {
            for lane in 0..offset {
                lanes[lane] += lanes[lane + offset];
            }
            offset >>= 1;
        }
        lanes[0]
    }

    fn step(mut state: [f32; 4], query: [f32; 2]) -> ([f32; 2], [f32; 4]) {
        let key = [1.5f32, 0.25];
        let value = [0.75f32, -0.5];
        let decay = [0.5f32, 0.25];
        let q_norm = (query.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
        let k_norm = (key.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
        let query = query.map(|x| x / q_norm * (2.0f32).sqrt().recip());
        let key = key.map(|x| x / k_norm);
        for row in 0..2 {
            for column in 0..2 {
                state[row * 2 + column] *= decay[row];
            }
        }
        let memory: [f32; 2] =
            std::array::from_fn(|column| state[column] * key[0] + state[2 + column] * key[1]);
        let delta: [f32; 2] =
            std::array::from_fn(|column| (value[column] - memory[column]) * 0.625);
        for row in 0..2 {
            for column in 0..2 {
                state[row * 2 + column] += key[row] * delta[column];
            }
        }
        let output: [f32; 2] =
            std::array::from_fn(|column| state[column] * query[0] + state[2 + column] * query[1]);
        (output, state)
    }

    #[test]
    fn exact_prefill_geometry_and_causal_state_are_pinned() {
        let plan = Glm53KdaPrefillPlan::new(2, 3, 64, 128, 128).unwrap();
        assert_eq!(plan.groups, 128);
        assert_eq!(plan.vector_bytes, 2 * 3 * 64 * 128 * 2);
        assert_eq!(plan.decay_bytes, 2 * 3 * 64 * 128 * 4);
        assert_eq!(plan.beta_bytes, 2 * 3 * 64 * 2);
        assert_eq!(plan.state_bytes, 2 * 64 * 128 * 128 * 4);
        assert!(Glm53KdaPrefillPlan::new(0, 3, 64, 128, 128).is_err());
        assert!(Glm53KdaPrefillPlan::new(1, 0, 64, 128, 128).is_err());
        assert!(Glm53KdaPrefillPlan::new(1, 3, 63, 128, 128).is_err());
        assert!(Glm53KdaPrefillPlan::new(u32::MAX, u32::MAX, 64, 128, 128).is_err());

        let initial = [1.0, 2.0, 3.0, 4.0];
        let (_, after_first) = step(initial, [0.5, -1.0]);
        let (causal_second, _) = step(after_first, [-0.25, 0.75]);
        let (reset_second, _) = step(initial, [-0.25, 0.75]);
        assert_ne!(causal_second, reset_second);
    }

    #[test]
    fn register_resident_norm_tree_and_scope_are_pinned() {
        let squares = std::array::from_fn(|index| {
            let value = (index as f32 - 47.0) * 0.03125;
            value * value
        });
        assert_eq!(
            original_tree(squares).to_bits(),
            register_tree(squares).to_bits()
        );
        assert!(CUDA_SOURCE.contains("atlas_glm53_kda_prefill_register_resident"));
        assert!(CUDA_SOURCE.contains("tokens <= 8U"));
        assert!(CUDA_SOURCE.contains("blockIdx.y * GLM53_KDA_RR_WARPS + warp"));
        assert!(CUDA_SOURCE.contains("lane + 96U"));
        assert!(CUDA_SOURCE.contains("__shfl_down_sync"));
    }

    #[test]
    fn register_resident_c8_is_default_on_with_exact_rollback() {
        // `ATLAS_GLM53_KDA_RR_COLUMNS` is the register-resident selector: it is
        // default ON, "0" is the exact rollback to the non-register-resident
        // kernel, and "1"/"4"/"2" choose the columns-per-warp group. Anything
        // else is still a hard error -- in particular "8", which looks like it
        // ought to name the default group but never has.
        assert!(parse_rr_c8_enabled(None).unwrap());
        assert!(parse_rr_c8_enabled(Some("1")).unwrap());
        assert!(parse_rr_c8_enabled(Some("4")).unwrap());
        assert!(parse_rr_c8_enabled(Some("2")).unwrap());
        assert!(!parse_rr_c8_enabled(Some("0")).unwrap());
        for invalid in ["", "3", "8", "16", "01", "04", " 1", "1 ", "true"] {
            assert!(
                parse_rr_c8_enabled(Some(invalid)).is_err(),
                "accepted {invalid:?}"
            );
        }

        // Default (absent, or the historical "1") is still the 8-column kernel,
        // and the rollback value never selects a narrow variant.
        assert_eq!(rr_columns(None), (8, "atlas_glm53_kda_prefill_rr_c8"));
        assert_eq!(rr_columns(Some("1")), (8, "atlas_glm53_kda_prefill_rr_c8"));
        assert_eq!(rr_columns(Some("0")), (8, "atlas_glm53_kda_prefill_rr_c8"));
        assert_eq!(rr_columns(Some("4")), (4, "atlas_glm53_kda_prefill_rr_c4"));
        assert_eq!(rr_columns(Some("2")), (2, "atlas_glm53_kda_prefill_rr_c2"));

        // The base file still carries only the 8-column kernel; the narrow
        // groups and the ATLAS_GLM53_KDA_RR_PREFETCH=1|2 variants live in the
        // exl3 overlay, which includes it.
        assert!(RR_COLUMNS_CUDA_SOURCE.contains("atlas_glm53_kda_prefill_rr_c8"));
        assert!(!RR_COLUMNS_CUDA_SOURCE.contains("prefill_rr_c2"));
        assert!(!RR_COLUMNS_CUDA_SOURCE.contains("prefill_rr_c4"));
        assert!(RR_COLUMNS_EXL3_CUDA_SOURCE.contains("iq3/glm53_kda_rr_columns.cu"));
        assert!(RR_COLUMNS_EXL3_CUDA_SOURCE.contains("atlas_glm53_kda_prefill_rr_c4"));
        assert!(RR_COLUMNS_EXL3_CUDA_SOURCE.contains("atlas_glm53_kda_prefill_rr_c2"));
        assert!(RR_COLUMNS_EXL3_CUDA_SOURCE.contains("atlas_glm53_kda_prefill_rr_c8_pf"));
        assert!(RR_COLUMNS_EXL3_CUDA_SOURCE.contains("atlas_glm53_kda_prefill_rr_c8_pf2"));
        assert_eq!(HEAD_DIM % (4 * 8), 0);
        assert_eq!(HEAD_DIM % (4 * 4), 0);
        assert_eq!(HEAD_DIM % (4 * 2), 0);
        assert!(RR_COLUMNS_CUDA_SOURCE.contains("first_column"));
        assert!(RR_COLUMNS_CUDA_SOURCE.contains("__shfl_down_sync"));

        // Prefetch is a separate, default-off lever: with it absent the symbol
        // is whatever the columns selector chose.
        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/layers/ops/glm53_kda_prefill.rs"));
        assert!(source.contains("Ok(\"1\") => \"atlas_glm53_kda_prefill_rr_c8_pf\","));
        assert!(source.contains("Ok(\"2\") => \"atlas_glm53_kda_prefill_rr_c8_pf2\","));
        assert!(source.contains("_ => columns_symbol,"));
    }

    #[test]
    fn launch_rejects_alias_before_effect() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53KdaPrefillKernel::load(&gpu).unwrap();
        let plan = Glm53KdaPrefillPlan::new(1, 9, 64, 128, 128).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53KdaPrefillBuffers {
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
                    Glm53KdaPrefillBuffers {
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
        kernel
            .launch_register_resident(&gpu, plan, valid, 0)
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
        assert!(
            kernel
                .launch_register_resident(
                    &gpu,
                    Glm53KdaPrefillPlan::new(1, 8, 64, 128, 128).unwrap(),
                    valid,
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn out_of_place_prefill_launches_and_reads_the_source_only_for_token_zero() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53KdaPrefillKernel::load(&gpu).unwrap();
        let plan = Glm53KdaPrefillPlan::new(1, 4, HEADS, HEAD_DIM, HEAD_DIM).unwrap();
        let at = |bytes: usize| GgmlIqBuffer {
            ptr: gpu.alloc(bytes).unwrap(),
            bytes,
        };
        let buffers = Glm53KdaPrefillBuffers {
            state_f32: at(plan.state_bytes),
            query_bf16: at(plan.vector_bytes),
            key_bf16: at(plan.vector_bytes),
            value_bf16: at(plan.vector_bytes),
            log_decay_f32: at(plan.decay_bytes),
            beta_bf16: at(plan.beta_bytes),
            output_bf16: at(plan.vector_bytes),
        };
        let state_in = at(plan.state_bytes);
        kernel.launch_oop(&gpu, plan, state_in, buffers, 7).unwrap();
        assert_eq!(gpu.launch_count(), 1);
        let short = GgmlIqBuffer {
            ptr: state_in.ptr,
            bytes: state_in.bytes - 4,
        };
        assert!(kernel.launch_oop(&gpu, plan, short, buffers, 7).is_err());
        assert_eq!(gpu.launch_count(), 1);

        let body = CUDA_SOURCE
            .split("atlas_glm53_kda_prefill_oop(")
            .nth(1)
            .expect("out-of-place prefill kernel present")
            .split("extern \"C\"")
            .next()
            .unwrap();
        assert!(body.starts_with("\n        const float * state_in,\n        float * state_out,"));
        assert!(!body.contains("__restrict__ state"));
        assert!(body.contains("const float * source = (token == 0U) ? state_in : state_out;"));
        assert!(body.contains("const float decayed = source[index] * decay[at];"));
        assert!(body.contains("const float updated = state_out[index] + k_values[at] * delta;"));
        assert!(!body.contains("state[index]"));
    }
}
