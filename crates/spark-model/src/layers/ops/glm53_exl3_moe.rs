// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas binding for the pinned ExLlamaV3 device-routed EXL3 MoE kernel.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{GLM53_EXL3_MAX_WIDE_ROWS, Glm53Exl3Buffer, Glm53Exl3RoutePolicy};

const HIDDEN: u32 = 4096;
const INTERMEDIATE: u32 = 2048;
const EXPERTS: u32 = 288;
const TOP_K: u32 = 8;
const MAX_ROWS: u32 = GLM53_EXL3_MAX_WIDE_ROWS as u32;
// GB10 exposes 48 SMs. Match the pinned ExLlamaV3 admission exactly:
// `exl3_moe_max_concurrency = num_sms / MOE_SMS_PER_EXPERT = 48 / 8 = 6`.
// Every block in all six barrier groups must be co-resident.
const CONCURRENCY: u32 = 6;
const SMS_PER_GROUP: u32 = 8;
const SHARED_MEMORY_BYTES: u32 = 90 * 1024;
const STAGED_SCATTER_SHARED_MEMORY_BYTES: u32 = 8 * 128 * size_of::<f32>() as u32;
const STAGED_CHUNK_ROWS: u32 = 16;
const STAGED_BLOCK_THREADS: u32 = 256;
const STAGED_BASE_MODULE: &str = "glm53_exl3_moe_staged_k16";
const STAGED_GEMM_MODULE: &str = "glm53_exl3_moe_staged_n256_f1";
const STAGED_TILE_N: u32 = 256;
const STAGED_SHARED_MEMORY_BYTES: u32 = 20_992;
const FUSED_MOE_KERNEL_LAUNCHES: u32 = 2;
const STAGED_MOE_KERNEL_LAUNCHES: u32 = 7;
pub const GLM53_EXL3_MOE_LOCK_BYTES: usize = (1024 * 1024 + 2 * 1024 + 2 + 64) * size_of::<i32>();
const MOE_SYMBOL: &str = "_Z15exl3_moe_kernelILi2ELi256ELi2EEvPK6__halfPS0_S3_S3_S3_PfPPKtPS2_S8_S7_S8_S8_S7_S8_S8_PKlSA_S2_iiiiiifiiiiPi";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LargeMConfig {
    override_group: Option<(u32, u32)>,
}

fn parse_large_m_config(group: Option<&str>) -> Result<LargeMConfig> {
    let Some(group) = group else {
        return Ok(LargeMConfig {
            override_group: None,
        });
    };
    let group_width = match group {
        "8" => 8,
        "12" => 12,
        "16" => 16,
        "24" => 24,
        other => {
            bail!("ATLAS_GLM53_EXL3_MOE_LARGE_GROUP_WIDTH must be 8, 12, 16, or 24, got {other}")
        }
    };
    let concurrency = 48 / group_width;
    ensure!(
        group_width * concurrency == 48 && group_width <= 32 && concurrency <= CONCURRENCY,
        "GLM EXL3 large-M MoE config must occupy exactly 48 CTAs within scratch bounds"
    );
    Ok(LargeMConfig {
        override_group: Some((group_width, concurrency)),
    })
}

fn large_m_schedule(rows: u32, config: LargeMConfig) -> (u32, u32) {
    config
        .override_group
        .unwrap_or_else(|| if rows >= 1024 { (16, 3) } else { (12, 4) })
}

fn parse_staged_enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => bail!("ATLAS_GLM53_EXL3_MOE_STAGED must be exactly 0 or 1, got {other}"),
    }
}

fn route_private_scope(rows: u32, enabled: bool) -> bool {
    enabled && (1..=8).contains(&rows)
}

fn moe_kernel_launches(rows: u32, staged_loaded: bool, route_private_enabled: bool) -> u32 {
    if route_private_scope(rows, route_private_enabled) {
        return 2;
    }
    if rows >= 1024 && staged_loaded {
        STAGED_MOE_KERNEL_LAUNCHES
    } else {
        FUSED_MOE_KERNEL_LAUNCHES
    }
}

#[path = "glm53_exl3_moe_plan.rs"]
mod layout;
pub use layout::{
    Glm53Exl3MoeBuffers, Glm53Exl3MoePlan, Glm53Exl3MoePointerTables, Glm53Exl3MoeScratch,
};
use layout::{exact, validate_buffers};
#[path = "glm53_exl3_moe_staged.rs"]
mod staged;
#[path = "glm53_exl3_moe_verify_staged.rs"]
mod verify_staged;

pub struct Glm53Exl3MoeKernels {
    pack_routes: KernelHandle,
    pack_routes_private: KernelHandle,
    fused_moe: KernelHandle,
    fused_moe_private: KernelHandle,
    route_policy: Glm53Exl3RoutePolicy,
    large_m: LargeMConfig,
    staged: Option<Glm53Exl3StagedMoeKernels>,
    verify_staged: Option<Glm53Exl3StagedMoeKernels>,
    combine_shared: KernelHandle,
    combine_private_shared: KernelHandle,
}

struct Glm53Exl3StagedMoeKernels {
    private: bool,
    gemm_block_threads: u32,
    gemm_shared_bytes: u32,
    build_chunks: KernelHandle,
    gather: KernelHandle,
    gate_up: KernelHandle,
    activate: KernelHandle,
    down: KernelHandle,
    scatter: KernelHandle,
}

impl Glm53Exl3MoeKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let policy = Glm53Exl3RoutePolicy::parse_with_verify_group(
            std::env::var_os("ATLAS_GLM53_EXL3_ROUTE_PRIVATE").as_deref(),
            std::env::var_os("ATLAS_GLM53_EXL3_ROUTE_PRIVATE_PREFILL").as_deref(),
            std::env::var_os("ATLAS_GLM53_EXL3_MOE_VERIFY_GROUP_WIDTH").as_deref(),
            std::env::var_os("ATLAS_GLM53_EXACT_VERIFY").as_deref(),
        )?
        .with_verify_staged_k32(
            std::env::var_os("ATLAS_GLM53_EXL3_MOE_VERIFY_STAGED_K32").as_deref(),
            std::env::var_os("ATLAS_GLM53_EXACT_VERIFY").as_deref(),
        )?;
        policy.validate_moe_mode(std::env::var_os("ATLAS_GLM53_EXL3_MOE").as_deref())?;
        Self::load_with_route_policy(gpu, policy)
    }

    pub fn load_with_route_policy(
        gpu: &dyn GpuBackend,
        route_policy: Glm53Exl3RoutePolicy,
    ) -> Result<Self> {
        let verify_staged = route_policy
            .verify_staged_k32_enabled()
            .then(|| verify_staged::load(gpu))
            .transpose()?;
        let large_group = std::env::var("ATLAS_GLM53_EXL3_MOE_LARGE_GROUP_WIDTH").ok();
        let large_m = parse_large_m_config(large_group.as_deref())?;
        let staged = std::env::var("ATLAS_GLM53_EXL3_MOE_STAGED").ok();
        let staged_enabled = parse_staged_enabled(staged.as_deref())?;
        ensure!(
            std::env::var_os("ATLAS_GLM53_EXL3_MOE_STAGED_VARIANT").is_none(),
            "ATLAS_GLM53_EXL3_MOE_STAGED_VARIANT was experimental and is no longer admitted"
        );
        let staged = staged_enabled
            .then(|| {
                let private = route_policy.prefill_enabled();
                let (module, names) = if private {
                    (
                        "glm53_exl3_moe_staged_private",
                        [
                            "atlas_glm53_exl3_build_chunks_private",
                            "atlas_glm53_exl3_staged_gather_private",
                            "atlas_glm53_exl3_staged_activate_private",
                            "atlas_glm53_exl3_staged_scatter_private",
                        ],
                    )
                } else {
                    (
                        STAGED_BASE_MODULE,
                        [
                            "atlas_glm53_exl3_build_chunks",
                            "atlas_glm53_exl3_staged_gather",
                            "atlas_glm53_exl3_staged_activate",
                            "atlas_glm53_exl3_staged_scatter",
                        ],
                    )
                };
                let gate_up =
                    gpu.kernel(STAGED_GEMM_MODULE, "atlas_glm53_exl3_staged_gate_up_k16")?;
                let down = gpu.kernel(STAGED_GEMM_MODULE, "atlas_glm53_exl3_staged_down_k16")?;
                gpu.set_kernel_max_dynamic_shared_memory(gate_up, STAGED_SHARED_MEMORY_BYTES)?;
                gpu.set_kernel_max_dynamic_shared_memory(down, STAGED_SHARED_MEMORY_BYTES)?;
                Ok::<_, anyhow::Error>(Glm53Exl3StagedMoeKernels {
                    private,
                    gemm_block_threads: STAGED_BLOCK_THREADS,
                    gemm_shared_bytes: STAGED_SHARED_MEMORY_BYTES,
                    build_chunks: gpu.kernel(module, names[0])?,
                    gather: gpu.kernel(module, names[1])?,
                    gate_up,
                    activate: gpu.kernel(module, names[2])?,
                    down,
                    scatter: gpu.kernel(module, names[3])?,
                })
            })
            .transpose()?;
        let fused_moe = gpu.kernel("glm53_exl3_moe_k2_cb2", MOE_SYMBOL)?;
        gpu.set_kernel_max_dynamic_shared_memory(fused_moe, SHARED_MEMORY_BYTES)?;
        let fused_moe_private = gpu.kernel(
            "glm53_exl3_moe_private",
            "atlas_glm53_exl3_moe_private_k2_n256_cb2",
        )?;
        gpu.set_kernel_max_dynamic_shared_memory(fused_moe_private, SHARED_MEMORY_BYTES)?;
        Ok(Self {
            pack_routes: gpu.kernel("glm53_exl3_moe_route", "atlas_glm53_exl3_pack_routes")?,
            pack_routes_private: gpu.kernel(
                "glm53_exl3_moe_route",
                "atlas_glm53_exl3_pack_routes_private",
            )?,
            fused_moe,
            fused_moe_private,
            route_policy,
            large_m,
            staged,
            verify_staged,
            combine_shared: gpu
                .kernel("glm53_exl3_moe_route", "atlas_glm53_exl3_combine_shared")?,
            combine_private_shared: gpu.kernel(
                "glm53_exl3_moe_route",
                "atlas_glm53_exl3_combine_private_shared",
            )?,
        })
    }

    pub fn plan(&self, rows: u32) -> Result<Glm53Exl3MoePlan> {
        let mut plan = Glm53Exl3MoePlan::new(rows)?;
        plan.route_private_f32_bytes = self.route_policy.private_bytes(rows)?;
        if self.verify_staged_active(rows) {
            plan = plan.with_verify_staged_k32()?;
        } else if rows <= 8 {
            plan = plan.with_fused_concurrency(self.fused_schedule(rows).1)?;
        }
        Ok(plan)
    }

    fn fused_schedule(&self, rows: u32) -> (u32, u32) {
        if rows > 8 {
            return large_m_schedule(rows, self.large_m);
        }
        let selected = self.route_policy.verify_group(
            rows,
            super::glm53_exact_verify_active(),
            super::glm53_exact_wide_prefill_active() || super::glm53_layer_major_prefill_active(),
        );
        match selected {
            Some(width) => (width, 48 / width),
            None => (SMS_PER_GROUP, CONCURRENCY),
        }
    }

    pub fn validate(&self, plan: Glm53Exl3MoePlan, buffers: Glm53Exl3MoeBuffers) -> Result<()> {
        ensure!(
            plan == self.plan(plan.rows)?,
            "GLM EXL3 MoE plan differs from latched route policy"
        );
        self.selected_staged(plan.rows)?;
        validate_buffers(plan, buffers)
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Exl3MoePlan,
        buffers: Glm53Exl3MoeBuffers,
        stream: u64,
    ) -> Result<()> {
        self.validate(plan, buffers)?;
        let staged = self.selected_staged(plan.rows)?;
        let route_private = self.route_policy.private_for(plan.rows)?;
        if !route_private {
            gpu.memset_async(buffers.output_f32.ptr, 0, buffers.output_f32.bytes, stream)?;
        }
        gpu.memset_async(
            buffers.route_status_u32.ptr,
            0,
            buffers.route_status_u32.bytes,
            stream,
        )?;
        let pack_routes = if route_private {
            self.pack_routes_private
        } else {
            self.pack_routes
        };
        KernelLaunch::new(gpu, pack_routes)
            .grid([1, 1, 1])
            .block([320, 1, 1])
            .arg_ptr(buffers.route_ids_u32.ptr)
            .arg_ptr(buffers.route_weights_f32.ptr)
            .arg_ptr(buffers.expert_count_i64.ptr)
            .arg_ptr(buffers.token_sorted_i64.ptr)
            .arg_ptr(buffers.weight_sorted_f16.ptr)
            .arg_ptr(buffers.route_status_u32.ptr)
            .arg_u32(plan.rows)
            .arg_u32(EXPERTS)
            .arg_u32(TOP_K)
            .launch(stream)?;

        if let Some(staged) = staged {
            self.launch_staged(gpu, plan, buffers, stream, staged)
        } else {
            let p = buffers.pointers;
            let (group_width, concurrency) = self.fused_schedule(plan.rows);
            let fused_moe = if route_private {
                self.fused_moe_private
            } else {
                self.fused_moe
            };
            let routed_output = if route_private {
                buffers.route_private_f32
            } else {
                buffers.output_f32
            };
            KernelLaunch::new(gpu, fused_moe)
                .grid([group_width, 1, concurrency])
                .block([512, 1, 1])
                .shared_mem(SHARED_MEMORY_BYTES)
                .arg_ptr(buffers.input_f16.ptr)
                .arg_ptr(buffers.temp_state_g_f16.ptr)
                .arg_ptr(buffers.temp_state_u_f16.ptr)
                .arg_ptr(buffers.temp_intermediate_g_f16.ptr)
                .arg_ptr(buffers.temp_intermediate_u_f16.ptr)
                .arg_ptr(routed_output.ptr)
                .arg_ptr(p.gate_trellis.ptr)
                .arg_ptr(p.gate_suh.ptr)
                .arg_ptr(p.gate_svh.ptr)
                .arg_ptr(p.up_trellis.ptr)
                .arg_ptr(p.up_suh.ptr)
                .arg_ptr(p.up_svh.ptr)
                .arg_ptr(p.down_trellis.ptr)
                .arg_ptr(p.down_suh.ptr)
                .arg_ptr(p.down_svh.ptr)
                .arg_ptr(buffers.expert_count_i64.ptr)
                .arg_ptr(buffers.token_sorted_i64.ptr)
                .arg_ptr(buffers.weight_sorted_f16.ptr)
                .arg_i32(HIDDEN as i32)
                .arg_i32(INTERMEDIATE as i32)
                .arg_i32(EXPERTS as i32)
                .arg_i32(TOP_K as i32)
                .arg_i32(plan.rows as i32)
                .arg_i32(concurrency as i32)
                .arg_f32(10.0)
                .arg_i32(0)
                .arg_i32(2)
                .arg_i32(2)
                .arg_i32(2)
                .arg_ptr(buffers.locks_i32.ptr)
                .launch(stream)
        }
    }

    pub fn kernel_launches(&self, rows: u32) -> u32 {
        if self.verify_staged_active(rows) {
            return STAGED_MOE_KERNEL_LAUNCHES;
        }
        moe_kernel_launches(
            rows,
            self.staged.is_some(),
            self.route_policy
                .private_for(1)
                .expect("valid scalar geometry"),
        )
    }

    pub fn combine_shared(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Exl3MoePlan,
        routed_f32: Glm53Exl3Buffer,
        route_private_f32: Glm53Exl3Buffer,
        shared_bf16: Glm53Exl3Buffer,
        output_bf16: Glm53Exl3Buffer,
        status_u32: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            plan == self.plan(plan.rows)?,
            "GLM EXL3 combine plan differs from latched route policy"
        );
        exact("routed output", routed_f32, plan.output_f32_bytes)?;
        exact(
            "route-private output",
            route_private_f32,
            plan.route_private_f32_bytes,
        )?;
        exact("shared output", shared_bf16, plan.input_f16_bytes)?;
        exact("combined output", output_bf16, plan.input_f16_bytes)?;
        exact("route status", status_u32, 4)?;
        let elements = plan
            .rows
            .checked_mul(HIDDEN)
            .context("GLM EXL3 MoE combine overflow")?;
        let route_private = self.route_policy.private_for(plan.rows)?;
        let kernel = if route_private {
            self.combine_private_shared
        } else {
            self.combine_shared
        };
        let routed = if route_private {
            route_private_f32
        } else {
            routed_f32
        };
        KernelLaunch::new(gpu, kernel)
            .grid([elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(routed.ptr)
            .arg_ptr(shared_bf16.ptr)
            .arg_ptr(output_bf16.ptr)
            .arg_ptr(status_u32.ptr)
            .arg_u32(elements)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "glm53_exl3_moe_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm53_exl3_moe_verify_schedule_tests.rs"]
mod verify_schedule_tests;
