// SPDX-License-Identifier: AGPL-3.0-only

//! Checked EXL3 routed-MoE layouts shared by all execution paths.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Exl3MoePlan {
    pub rows: u32,
    pub pairs: u32,
    pub input_f16_bytes: usize,
    pub output_f32_bytes: usize,
    pub route_private_f32_bytes: usize,
    pub expert_count_i64_bytes: usize,
    pub token_sorted_i64_bytes: usize,
    pub weight_sorted_f16_bytes: usize,
    pub temp_state_f16_bytes: usize,
    pub temp_intermediate_f16_bytes: usize,
    pub pointer_table_u64_bytes: usize,
    pub route_ids_u32_bytes: usize,
    pub route_weights_f32_bytes: usize,
    pub pair_expert_u32_bytes: usize,
    pub chunk_descriptor_u32_bytes: usize,
    pub chunk_count_u32_bytes: usize,
    pub max_chunks: u32,
}

impl Glm53Exl3MoePlan {
    /// Prefix the existing M2048 owner allocation; never resize/rebind it.
    /// Widths8/4/2 keep all K tiles of each output column in one CTA.
    pub(super) fn with_fused_concurrency(mut self, concurrency: u32) -> Result<Self> {
        ensure!(
            (1..=8).contains(&self.rows) && matches!(concurrency, 6 | 12 | 24),
            "GLM EXL3 small-M schedule requires rows1..8 and 6, 12, or 24 groups"
        );
        // Pinned locks: <=768 GEMM ints, <=48 group-barrier ints and
        // <=26 scheduler ints, in disjoint 1M/2048/66-int reservations.
        let temp_rows = concurrency
            .checked_mul(self.rows)
            .context("GLM EXL3 scheduled scratch row overflow")?
            .max(self.pairs);
        ensure!(
            temp_rows <= 8 * MAX_ROWS,
            "GLM EXL3 scheduled scratch exceeds its existing owner arena"
        );
        self.temp_state_f16_bytes = usize::try_from(temp_rows)?
            .checked_mul(HIDDEN as usize * 2)
            .context("GLM EXL3 scheduled hidden scratch overflow")?;
        self.temp_intermediate_f16_bytes = usize::try_from(temp_rows)?
            .checked_mul(INTERMEDIATE as usize * 2)
            .context("GLM EXL3 scheduled intermediate scratch overflow")?;
        Ok(self)
    }

    pub fn new(rows: u32) -> Result<Self> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows),
            "GLM EXL3 fused MoE admits 1..={MAX_ROWS} rows"
        );
        let pairs = rows
            .checked_mul(TOP_K)
            .context("GLM EXL3 MoE pair overflow")?;
        let bytes = |elements: u32, width: usize| -> Result<usize> {
            usize::try_from(elements)?
                .checked_mul(width)
                .context("GLM EXL3 MoE byte overflow")
        };
        let fused_temp_rows = CONCURRENCY
            .checked_mul(rows)
            .context("GLM EXL3 MoE temp row overflow")?;
        let temp_rows = fused_temp_rows.max(pairs);
        let max_chunks = pairs
            .div_ceil(STAGED_CHUNK_ROWS)
            .checked_add(EXPERTS)
            .context("GLM EXL3 staged MoE chunk descriptor overflow")?;
        Ok(Self {
            rows,
            pairs,
            input_f16_bytes: bytes(rows.checked_mul(HIDDEN).unwrap(), 2)?,
            output_f32_bytes: bytes(rows.checked_mul(HIDDEN).unwrap(), 4)?,
            route_private_f32_bytes: bytes(rows.min(8).checked_mul(TOP_K).unwrap(), 4_096 * 4)?,
            expert_count_i64_bytes: bytes(EXPERTS + 1, 8)?,
            token_sorted_i64_bytes: bytes(pairs, 8)?,
            weight_sorted_f16_bytes: bytes(pairs, 2)?,
            temp_state_f16_bytes: bytes(temp_rows.checked_mul(HIDDEN).unwrap(), 2)?,
            temp_intermediate_f16_bytes: bytes(temp_rows.checked_mul(INTERMEDIATE).unwrap(), 2)?,
            pointer_table_u64_bytes: bytes(EXPERTS, 8)?,
            route_ids_u32_bytes: bytes(pairs, 4)?,
            route_weights_f32_bytes: bytes(pairs, 4)?,
            pair_expert_u32_bytes: bytes(pairs, 4)?,
            chunk_descriptor_u32_bytes: bytes(max_chunks, 4)?,
            chunk_count_u32_bytes: size_of::<u32>(),
            max_chunks,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3MoePointerTables {
    pub gate_trellis: Glm53Exl3Buffer,
    pub gate_suh: Glm53Exl3Buffer,
    pub gate_svh: Glm53Exl3Buffer,
    pub up_trellis: Glm53Exl3Buffer,
    pub up_suh: Glm53Exl3Buffer,
    pub up_svh: Glm53Exl3Buffer,
    pub down_trellis: Glm53Exl3Buffer,
    pub down_suh: Glm53Exl3Buffer,
    pub down_svh: Glm53Exl3Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3MoeScratch {
    pub expert_count_i64: Glm53Exl3Buffer,
    pub token_sorted_i64: Glm53Exl3Buffer,
    pub weight_sorted_f16: Glm53Exl3Buffer,
    pub temp_state_g_f16: Glm53Exl3Buffer,
    pub temp_state_u_f16: Glm53Exl3Buffer,
    pub temp_intermediate_g_f16: Glm53Exl3Buffer,
    pub temp_intermediate_u_f16: Glm53Exl3Buffer,
    pub output_f32: Glm53Exl3Buffer,
    pub route_private_f32: Glm53Exl3Buffer,
    pub route_status_u32: Glm53Exl3Buffer,
    pub locks_i32: Glm53Exl3Buffer,
    pub pair_expert_u32: Glm53Exl3Buffer,
    pub chunk_expert_u32: Glm53Exl3Buffer,
    pub chunk_start_u32: Glm53Exl3Buffer,
    pub chunk_rows_u32: Glm53Exl3Buffer,
    pub chunk_count_u32: Glm53Exl3Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3MoeBuffers {
    pub input_f16: Glm53Exl3Buffer,
    pub output_f32: Glm53Exl3Buffer,
    pub route_private_f32: Glm53Exl3Buffer,
    pub route_ids_u32: Glm53Exl3Buffer,
    pub route_weights_f32: Glm53Exl3Buffer,
    pub expert_count_i64: Glm53Exl3Buffer,
    pub token_sorted_i64: Glm53Exl3Buffer,
    pub weight_sorted_f16: Glm53Exl3Buffer,
    pub temp_state_g_f16: Glm53Exl3Buffer,
    pub temp_state_u_f16: Glm53Exl3Buffer,
    pub temp_intermediate_g_f16: Glm53Exl3Buffer,
    pub temp_intermediate_u_f16: Glm53Exl3Buffer,
    pub route_status_u32: Glm53Exl3Buffer,
    pub locks_i32: Glm53Exl3Buffer,
    pub pair_expert_u32: Glm53Exl3Buffer,
    pub chunk_expert_u32: Glm53Exl3Buffer,
    pub chunk_start_u32: Glm53Exl3Buffer,
    pub chunk_rows_u32: Glm53Exl3Buffer,
    pub chunk_count_u32: Glm53Exl3Buffer,
    pub pointers: Glm53Exl3MoePointerTables,
}

pub(super) fn exact(name: &str, buffer: Glm53Exl3Buffer, bytes: usize) -> Result<()> {
    if buffer.ptr == DevicePtr::NULL || buffer.bytes != bytes {
        bail!("GLM EXL3 fused MoE {name} is null or has the wrong exact extent");
    }
    Ok(())
}

pub(super) fn validate_buffers(plan: Glm53Exl3MoePlan, b: Glm53Exl3MoeBuffers) -> Result<()> {
    let mut expected = vec![
        ("input F16", b.input_f16, plan.input_f16_bytes),
        ("output F32", b.output_f32, plan.output_f32_bytes),
        (
            "route-private F32",
            b.route_private_f32,
            plan.route_private_f32_bytes,
        ),
        ("route IDs", b.route_ids_u32, plan.route_ids_u32_bytes),
        (
            "route weights",
            b.route_weights_f32,
            plan.route_weights_f32_bytes,
        ),
        (
            "expert counts",
            b.expert_count_i64,
            plan.expert_count_i64_bytes,
        ),
        (
            "sorted tokens",
            b.token_sorted_i64,
            plan.token_sorted_i64_bytes,
        ),
        (
            "sorted weights",
            b.weight_sorted_f16,
            plan.weight_sorted_f16_bytes,
        ),
        (
            "temp state g",
            b.temp_state_g_f16,
            plan.temp_state_f16_bytes,
        ),
        (
            "temp state u",
            b.temp_state_u_f16,
            plan.temp_state_f16_bytes,
        ),
        (
            "temp intermediate g",
            b.temp_intermediate_g_f16,
            plan.temp_intermediate_f16_bytes,
        ),
        (
            "temp intermediate u",
            b.temp_intermediate_u_f16,
            plan.temp_intermediate_f16_bytes,
        ),
        ("route status", b.route_status_u32, 4),
        ("locks", b.locks_i32, GLM53_EXL3_MOE_LOCK_BYTES),
        (
            "pair experts",
            b.pair_expert_u32,
            plan.pair_expert_u32_bytes,
        ),
        (
            "chunk experts",
            b.chunk_expert_u32,
            plan.chunk_descriptor_u32_bytes,
        ),
        (
            "chunk starts",
            b.chunk_start_u32,
            plan.chunk_descriptor_u32_bytes,
        ),
        (
            "chunk rows",
            b.chunk_rows_u32,
            plan.chunk_descriptor_u32_bytes,
        ),
        ("chunk count", b.chunk_count_u32, plan.chunk_count_u32_bytes),
    ];
    let p = b.pointers;
    expected.extend([
        (
            "gate trellis table",
            p.gate_trellis,
            plan.pointer_table_u64_bytes,
        ),
        ("gate suh table", p.gate_suh, plan.pointer_table_u64_bytes),
        ("gate svh table", p.gate_svh, plan.pointer_table_u64_bytes),
        (
            "up trellis table",
            p.up_trellis,
            plan.pointer_table_u64_bytes,
        ),
        ("up suh table", p.up_suh, plan.pointer_table_u64_bytes),
        ("up svh table", p.up_svh, plan.pointer_table_u64_bytes),
        (
            "down trellis table",
            p.down_trellis,
            plan.pointer_table_u64_bytes,
        ),
        ("down suh table", p.down_suh, plan.pointer_table_u64_bytes),
        ("down svh table", p.down_svh, plan.pointer_table_u64_bytes),
    ]);
    for (name, buffer, bytes) in expected {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != bytes {
            bail!("GLM EXL3 fused MoE {name} is null or has the wrong exact extent");
        }
        buffer
            .ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .with_context(|| format!("GLM EXL3 fused MoE {name} address overflow"))?;
    }
    Ok(())
}
