// SPDX-License-Identifier: AGPL-3.0-only

//! Compact same-stream work-list launchers for ordinary-NVFP4 MoE prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const NVFP4_WORKLIST_N_TILE: u32 = 128;
pub const NVFP4_WORKLIST_GATE_UP_M_TILE: u32 = 64;
pub const NVFP4_WORKLIST_DOWN_M_TILE: u32 = 64;

const MAX_WORKLIST_GRID_CTAS: u32 = 16_384;

/// Conservative item bound shared by the arena sizing and both launch paths.
///
/// A single expert can own every expanded row. Distributing rows across
/// experts contributes at most one partial M tile per expert; the extra item
/// preserves the bound used by the existing proven FP8 work-list path.
pub fn nvfp4_worklist_capacity_items(
    total_expanded: usize,
    num_experts: usize,
    n_tiles: u32,
    m_tile: u32,
) -> Result<u32> {
    ensure!(total_expanded > 0, "NVFP4 MoE work-list requires rows");
    ensure!(num_experts > 0, "NVFP4 MoE work-list requires experts");
    ensure!(
        n_tiles > 0 && n_tiles < 64,
        "NVFP4 work-list n_tiles must be in 1..64, got {n_tiles}"
    );
    ensure!(
        m_tile == NVFP4_WORKLIST_GATE_UP_M_TILE,
        "NVFP4 work-list m_tile must be 64, got {m_tile}"
    );

    let row_tiles = total_expanded.div_ceil(m_tile as usize);
    ensure!(
        row_tiles < (1usize << 26),
        "NVFP4 work-list m_tile index exceeds 26-bit packing range: {row_tiles}"
    );
    let items = row_tiles
        .checked_add(num_experts)
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(n_tiles as usize))
        .ok_or_else(|| anyhow::anyhow!("NVFP4 work-list capacity overflow"))?;
    ensure!(
        items <= i32::MAX as usize,
        "NVFP4 work-list exceeds device count range: {items}"
    );
    Ok(items as u32)
}

/// Build compact `(expert, m_tile, n_tile)` items on the compute stream.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_build_tile_worklist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr,
    packed_weight_ptrs: DevicePtr,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    num_experts: u32,
    n_tiles: u32,
    m_tile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(packed_weight_ptrs)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(num_experts)
        .arg_u32(n_tiles)
        .arg_u32(m_tile)
        .launch(stream)
}

/// Compact M64 fused gate+up launch. The kernel grid-strides over the
/// builder's work-list and preserves the parent NVFP4 K64 arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_worklist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_tiles.clamp(1, MAX_WORKLIST_GRID_CTAS), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}

/// Compact M64 down-projection launch. The kernel grid-strides over the
/// builder's work-list and preserves the parent NVFP4 K64 arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_k64_worklist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_tiles.clamp(1, MAX_WORKLIST_GRID_CTAS), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_worklist(offsets: &[u32], n_tiles: u32, m_tile: u32) -> Vec<(u32, u32)> {
        let mut items = Vec::new();
        for expert in 0..offsets.len() - 1 {
            let rows = offsets[expert + 1] - offsets[expert];
            for mt in 0..rows.div_ceil(m_tile) {
                for nt in 0..n_tiles {
                    items.push((expert as u32, (mt << 6) | nt));
                }
            }
        }
        items
    }

    #[test]
    fn flash_next_16k_item_bounds_match_arena_math() {
        let rows = 16_000 * 10;
        let gate_up = nvfp4_worklist_capacity_items(rows, 512, 10, 64).unwrap();
        let down = nvfp4_worklist_capacity_items(rows, 512, 20, 64).unwrap();
        assert_eq!(gate_up as usize * 8, 241_040);
        assert_eq!(down as usize * 8, 482_080);
    }

    #[test]
    fn packed_n_tile_range_fails_closed() {
        assert!(nvfp4_worklist_capacity_items(160_000, 512, 0, 128).is_err());
        assert!(nvfp4_worklist_capacity_items(160_000, 512, 64, 128).is_err());
        assert!(nvfp4_worklist_capacity_items(160_000, 512, 10, 32).is_err());
    }

    #[test]
    fn packed_geometry_is_expert_then_m_then_n_and_decodes_exactly() {
        let items = cpu_worklist(&[0, 0, 1, 66, 194], 3, 64);
        assert_eq!(items.len(), 15);
        assert_eq!(&items[..3], &[(1, 0), (1, 1), (1, 2)]);
        assert_eq!(
            &items[3..9],
            &[(2, 0), (2, 1), (2, 2), (2, 64), (2, 65), (2, 66)]
        );
        assert_eq!(items.last().copied(), Some((3, 66)));
        for (_, packed) in items {
            assert!((packed & 0x3f) < 3);
            assert!((packed >> 6) < 2);
        }
    }
}
