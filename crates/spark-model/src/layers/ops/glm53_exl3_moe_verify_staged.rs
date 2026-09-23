// SPDX-License-Identifier: AGPL-3.0-only
//! Default-off verifier schedule over unchanged private transforms and K32 GEMM.
use super::*;
use std::ffi::OsStr;

const BLOCK_THREADS: u32 = 512;
const SH3_SHARED_BYTES: u32 = 25_600;
const SH4_SHARED_BYTES: u32 = 28_672;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pipeline {
    Sh3,
    Sh4,
}

impl Pipeline {
    pub(super) fn parse(value: Option<&OsStr>) -> Result<Self> {
        match value {
            None => Ok(Self::Sh3),
            Some(value) if value == OsStr::new("0") => Ok(Self::Sh3),
            Some(value) if value == OsStr::new("1") => Ok(Self::Sh4),
            Some(_) => {
                bail!("ATLAS_GLM53_EXL3_MOE_VERIFY_K32_SH4 must be absent or exactly 0 or 1")
            }
        }
    }

    fn binding(self) -> (&'static str, &'static str, &'static str, u32) {
        match self {
            Self::Sh3 => (
                "glm53_exl3_moe_verify_k32",
                "atlas_glm53_exl3_verify_gate_up_k32",
                "atlas_glm53_exl3_verify_down_k32",
                SH3_SHARED_BYTES,
            ),
            Self::Sh4 => (
                "glm53_exl3_moe_verify_k32_sh4",
                "atlas_glm53_exl3_verify_gate_up_k32_sh4",
                "atlas_glm53_exl3_verify_down_k32_sh4",
                SH4_SHARED_BYTES,
            ),
        }
    }
}

pub(super) fn load(gpu: &dyn GpuBackend) -> Result<Glm53Exl3StagedMoeKernels> {
    let pipeline =
        Pipeline::parse(std::env::var_os("ATLAS_GLM53_EXL3_MOE_VERIFY_K32_SH4").as_deref())?;
    load_with_pipeline(gpu, pipeline)
}

pub(super) fn load_with_pipeline(
    gpu: &dyn GpuBackend,
    pipeline: Pipeline,
) -> Result<Glm53Exl3StagedMoeKernels> {
    let (module, gate_up_name, down_name, shared_bytes) = pipeline.binding();
    let gate_up = gpu.kernel(module, gate_up_name)?;
    let down = gpu.kernel(module, down_name)?;
    gpu.set_kernel_max_dynamic_shared_memory(gate_up, shared_bytes)?;
    gpu.set_kernel_max_dynamic_shared_memory(down, shared_bytes)?;
    let private = "glm53_exl3_moe_staged_private";
    Ok(Glm53Exl3StagedMoeKernels {
        private: true,
        build_chunks: gpu.kernel(private, "atlas_glm53_exl3_build_chunks_private")?,
        gather: gpu.kernel(private, "atlas_glm53_exl3_staged_gather_private")?,
        gate_up,
        activate: gpu.kernel(private, "atlas_glm53_exl3_staged_activate_private")?,
        down,
        scatter: gpu.kernel(private, "atlas_glm53_exl3_staged_scatter_private")?,
        gemm_block_threads: BLOCK_THREADS,
        gemm_shared_bytes: shared_bytes,
    })
}

impl Glm53Exl3MoeKernels {
    pub(super) fn verify_staged_active(&self, rows: u32) -> bool {
        self.route_policy.verify_staged_k32(
            rows,
            super::super::glm53_exact_verify_active(),
            super::super::glm53_exact_wide_prefill_active()
                || super::super::glm53_layer_major_prefill_active(),
        )
    }

    pub(super) fn selected_staged(&self, rows: u32) -> Result<Option<&Glm53Exl3StagedMoeKernels>> {
        if self.verify_staged_active(rows) {
            let selected = self
                .verify_staged
                .as_ref()
                .context("GLM private K32 verifier kernels were not loaded")?;
            ensure!(
                selected.private
                    && selected.gemm_block_threads == BLOCK_THREADS
                    && matches!(
                        selected.gemm_shared_bytes,
                        SH3_SHARED_BYTES | SH4_SHARED_BYTES
                    ),
                "GLM private K32 verifier launch geometry changed"
            );
            Ok(Some(selected))
        } else if rows >= 1024 {
            Ok(self.staged.as_ref())
        } else {
            Ok(None)
        }
    }
}

impl Glm53Exl3MoePlan {
    pub(super) fn with_verify_staged_k32(mut self) -> Result<Self> {
        ensure!(
            (2..=8).contains(&self.rows) && self.pairs == self.rows * TOP_K,
            "GLM private K32 staging requires verifier rows2..8"
        );
        // Every nonempty chunk contains at least one pair. Prefix existing
        // descriptors; never enlarge/rebind the model-owned M2048 arena.
        self.max_chunks = self.pairs;
        self.chunk_descriptor_u32_bytes = usize::try_from(self.pairs)? * size_of::<u32>();
        Ok(self)
    }
}
