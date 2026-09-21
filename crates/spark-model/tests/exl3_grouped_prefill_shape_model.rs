// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the DeepSeek-K2 fixed-shape grouped-prefill
//! entry points. The wrappers must remain thin specializations of the shared
//! kernel; runtime shape checks and host dispatch are correctness boundaries.

use std::collections::HashSet;

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill.cu");
const GU_K16: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2_gu.cu");
const GU_K64: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2_gu.cu");
const DOWN_K16: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2_down.cu");
const DOWN_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2_down.cu");
const GU_N128_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_gu.cu");
const DOWN_N128_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_down.cu");
const GENERIC_K2_K16: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2.cu");
const GENERIC_K2_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2.cu");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");

fn assert_wrapper(source: &str, kernel_name: &str, n: u32, k: u32, k_step: u32) {
    assert!(source.contains("#define EXL3_PF_FIXED_BITS 2"));
    assert!(source.contains("#define EXL3_PF_FIXED_IDENTITY_ROWS 1"));
    assert!(source.contains(&format!("#define EXL3_PF_FIXED_N {n}")));
    assert!(source.contains(&format!("#define EXL3_PF_FIXED_K {k}")));
    assert!(source.contains(&format!("#define EXL3_PF_KERNEL_NAME {kernel_name}")));
    if k_step == 64 {
        assert!(source.contains("#define EXL3_PF_K_STEP 64"));
        assert!(source.contains("#define EXL3_PF_ASYNC_STAGE 1"));
    } else {
        assert!(!source.contains("#define EXL3_PF_K_STEP 64"));
    }
    assert!(source.contains("#include \"exl3_grouped_prefill.cu\""));
}

#[test]
fn fixed_shape_wrappers_compile_identity_rows_while_generic_kernels_keep_gather() {
    assert!(KERNEL.contains("#ifndef EXL3_PF_FIXED_IDENTITY_ROWS"));
    assert!(KERNEL.contains("#define EXL3_PF_FIXED_IDENTITY_ROWS 0"));
    assert!(KERNEL.contains("static_assert(EXL3_PF_FIXED_IDENTITY_ROWS == 0 ||"));
    assert!(KERNEL.contains("EXL3_PF_FIXED_IDENTITY_ROWS == 1,"));
    assert!(KERNEL.contains("#if EXL3_PF_FIXED_IDENTITY_ROWS"));
    assert!(KERNEL.contains("if (sorted_token_ids != nullptr) return;"));
    assert!(
        KERNEL.matches("const int input_row = sorted_row;").count() >= 2,
        "both synchronous and asynchronous staging must compile to identity rows"
    );
    assert!(
        KERNEL
            .matches("sorted_token_ids ? sorted_token_ids[sorted_row] : sorted_row")
            .count()
            >= 2,
        "generic wrappers must retain pointer-based gather selection"
    );
}

#[test]
fn wrappers_lock_the_k2_gate_up_and_down_shapes() {
    assert_wrapper(GU_K16, "exl3_grouped_prefill_k2_gu", 2048, 4096, 16);
    assert_wrapper(GU_K64, "exl3_grouped_prefill_k64_k2_gu", 2048, 4096, 64);
    assert_wrapper(DOWN_K16, "exl3_grouped_prefill_k2_down", 4096, 2048, 16);
    assert_wrapper(DOWN_K64, "exl3_grouped_prefill_k64_k2_down", 4096, 2048, 64);
    assert_wrapper(
        GU_N128_K64,
        "exl3_grouped_prefill_k64_n128_k2_gu",
        2048,
        4096,
        64,
    );
    assert_wrapper(
        DOWN_N128_K64,
        "exl3_grouped_prefill_k64_n128_k2_down",
        4096,
        2048,
        64,
    );
}

#[test]
fn shared_kernel_rejects_runtime_shape_or_bitrate_mismatches() {
    assert!(KERNEL.contains("#ifndef EXL3_PF_FIXED_N"));
    assert!(KERNEL.contains("#ifndef EXL3_PF_FIXED_K"));
    assert!(KERNEL.contains("bits != EXL3_PF_FIXED_BITS"));
    assert!(KERNEL.contains("N != EXL3_PF_FIXED_N"));
    assert!(KERNEL.contains("K != EXL3_PF_FIXED_K"));
    assert!(KERNEL.contains("fixed N and K must be paired"));
    assert!(KERNEL.contains("fixed-persistent needs a fixed shape"));
}

#[test]
fn persistent_grid_stride_covers_every_fixed_shape_strip_once() {
    let num_experts = 256usize;
    for n in [2048usize, 4096] {
        let total_strips = num_experts * (n / 64);
        for grid_ctas in [1usize, 7, 48, 96, total_strips, total_strips + 13] {
            let mut seen = HashSet::new();
            for cta in 0..grid_ctas {
                for strip in (cta..total_strips).step_by(grid_ctas) {
                    assert!(seen.insert(strip), "N={n} grid={grid_ctas} strip={strip}");
                }
            }
            assert_eq!(seen.len(), total_strips, "N={n} grid={grid_ctas}");
        }
    }
    assert!(KERNEL.contains("work += work_stride"));
    assert!(KERNEL.contains("work_stride = persistent"));
}

#[test]
fn dispatch_requires_exact_k2_shapes_and_k_step() {
    for field in [
        "grouped_direct_k2_gu_k",
        "grouped_direct_k64_k2_gu_k",
        "grouped_direct_k2_down_k",
        "grouped_direct_k64_k2_down_k",
        "grouped_direct_k64_n128_k2_gu_k",
        "grouped_direct_k64_n128_k2_down_k",
    ] {
        assert!(STATE.contains(field), "missing fixed-shape handle {field}");
        assert!(DISPATCH.contains(field), "dispatch does not use {field}");
    }
    assert!(DISPATCH.contains("tab.bits == 2"));
    assert!(DISPATCH.contains("tab.n == 2048 && tab.k == 4096"));
    assert!(DISPATCH.contains("tab.n == 4096 && tab.k == 2048"));
    assert!(DISPATCH.contains("pf.direct_k64"));
    assert!(DISPATCH.contains("pf.fixed_shape && tab.bits == 2"));
    assert!(STATE.contains("&& gate.bits == 2"));
    assert!(STATE.contains("&& up.bits == 2"));
    assert!(STATE.contains("&& down.bits == 2"));
}

#[test]
fn every_nonmatching_shape_has_an_explicit_generic_k2_fallback() {
    assert!(DISPATCH.contains("pf.grouped_direct_k64_k2_k"));
    assert!(DISPATCH.contains("pf.grouped_direct_k2_k"));
    assert!(DISPATCH.contains("pf.direct_k64 && pf.fixed_k2 && tab.bits == 2"));
    assert!(DISPATCH.contains("pf.fixed_k2 && tab.bits == 2"));

    let last_fixed_shape = DISPATCH
        .find("pf.grouped_direct_k2_down_k")
        .expect("fixed-shape down dispatch");
    let first_generic_k2 = DISPATCH
        .find("pf.grouped_direct_k64_k2_k")
        .expect("generic K64/K2 fallback");
    assert!(
        last_fixed_shape < first_generic_k2,
        "generic K2 fallback must follow every exact fixed-shape arm"
    );
}

#[test]
fn host_direct_launch_proves_identity_rows_with_a_null_gather_pointer() {
    let direct_launch = DISPATCH
        .find("return KernelLaunch::new(gpu, kernel)")
        .expect("direct grouped-prefill launch");
    let launch_tail = &DISPATCH[direct_launch..];
    let null_gather = launch_tail
        .find(".arg_ptr(DevicePtr(0))")
        .expect("direct launch must pass a null sorted-token gather pointer");
    let persistent = launch_tail
        .find(".arg_u32(u32::from(persistent))")
        .expect("direct launch tail");
    assert!(null_gather < persistent);
}

#[test]
fn all_fixed_shape_wrappers_enable_exact_full_grid() {
    for source in [
        GU_K16,
        GU_K64,
        DOWN_K16,
        DOWN_K64,
        GU_N128_K64,
        DOWN_N128_K64,
    ] {
        assert!(source.contains("#define EXL3_PF_EXACT_FULL_GRID 1"));
    }
    for source in [GENERIC_K2_K16, GENERIC_K2_K64] {
        assert!(!source.contains("#define EXL3_PF_EXACT_FULL_GRID 1"));
    }
}

#[test]
fn exact_full_grid_compile_time_contract_is_fail_closed() {
    assert!(KERNEL.contains("#ifndef EXL3_PF_EXACT_FULL_GRID"));
    assert!(KERNEL.contains("#define EXL3_PF_EXACT_FULL_GRID 0"));
    assert!(KERNEL.contains("EXL3_PF_EXACT_FULL_GRID == 0 ||"));
    assert!(KERNEL.contains("EXL3_PF_EXACT_FULL_GRID == 1"));
    assert!(KERNEL.contains("!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_BITS == 2"));
    assert!(KERNEL.contains("!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_M_TILE == 64"));
    assert!(KERNEL.contains("!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_PERSISTENT == 1"));
    assert!(KERNEL.contains("!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_N != 0"));
    assert!(KERNEL.contains("fixed N-tile count must be a power of two"));
}

#[test]
fn exact_full_grid_has_one_dimensional_guards_and_constant_strip_mapping() {
    assert!(KERNEL.contains("#if EXL3_PF_EXACT_FULL_GRID"));
    assert!(KERNEL.contains("gridDim.y != 1 || gridDim.z != 1"));
    assert!(KERNEL.contains("gridDim.x != total_strips"));
    assert!(KERNEL.contains("blockIdx.x & (n_tiles - 1)"));
    assert!(KERNEL.contains("blockIdx.x >> n_tile_shift"));

    let exact = KERNEL
        .find("#if EXL3_PF_EXACT_FULL_GRID")
        .expect("exact-grid source arm");
    let generic = KERNEL[exact..]
        .find("#else")
        .map(|offset| exact + offset)
        .expect("generic mapping source arm");
    let grid_stride = KERNEL[generic..]
        .find("work += work_stride")
        .map(|offset| generic + offset)
        .expect("generic grid-stride mapping");
    assert!(exact < generic && generic < grid_stride);
}

#[test]
fn host_persistent_grid_size_is_checked_not_saturated() {
    assert!(DISPATCH.contains("let strips = num_experts"));
    assert!(DISPATCH.contains(".checked_mul(tab.n / direct_n_tile)"));
    assert!(DISPATCH.contains("EXL3 persistent prefill grid size overflow"));
    assert!(!DISPATCH.contains("num_experts.saturating_mul(tab.n / direct_n_tile)"));
}
