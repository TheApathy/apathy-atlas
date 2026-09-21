// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for fixed-shape EXL3 grouped-prefill launch bounds.

const CORE: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill.cu");
const K2_GU: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2_gu.cu");
const K2_DOWN: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2_down.cu");
const K64_K2_GU: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2_gu.cu");
const K64_K2_DOWN: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2_down.cu");
const K64_N128_K2_GU: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_gu.cu");
const K64_N128_K2_DOWN: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_down.cu");
const GENERIC_K2: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2.cu");
const M128: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_m128.cu");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");

const FIXED_WRAPPERS: [&str; 6] = [
    K2_GU,
    K2_DOWN,
    K64_K2_GU,
    K64_K2_DOWN,
    K64_N128_K2_GU,
    K64_N128_K2_DOWN,
];

#[test]
fn every_exact_m64_wrapper_declares_its_exact_thread_launch_bound() {
    for wrapper in FIXED_WRAPPERS {
        assert!(wrapper.contains("#define EXL3_PF_EXACT_FULL_GRID 1"));
        assert!(
            wrapper.contains("#define EXL3_PF_LAUNCH_BOUNDS 128")
                || wrapper.contains("#define EXL3_PF_LAUNCH_BOUNDS 256")
        );
    }
}

#[test]
fn generic_k2_and_m128_wrappers_do_not_claim_the_fixed_m64_launch_bound() {
    assert!(!GENERIC_K2.contains("#define EXL3_PF_LAUNCH_BOUNDS"));
    assert!(!M128.contains("#define EXL3_PF_LAUNCH_BOUNDS"));
    assert!(M128.contains("#define EXL3_PF_M_TILE 128"));
}

#[test]
fn core_launch_bound_is_optional_and_confined_to_exact_m64_kernels() {
    assert!(CORE.contains("#ifndef EXL3_PF_LAUNCH_BOUNDS"));
    assert!(CORE.contains("#define EXL3_PF_LAUNCH_BOUNDS 0"));
    assert!(CORE.contains("EXL3_PF_LAUNCH_BOUNDS == 0 || EXL3_PF_LAUNCH_BOUNDS == 128"));
    assert!(CORE.contains("!EXL3_PF_LAUNCH_BOUNDS || EXL3_PF_M_TILE == 64"));
    assert!(CORE.contains("!EXL3_PF_LAUNCH_BOUNDS || EXL3_PF_EXACT_FULL_GRID == 1"));
    assert!(CORE.contains("__launch_bounds__(EXL3_PF_LAUNCH_BOUNDS)"));
}

#[test]
fn exact_wrappers_fail_closed_on_nonexact_block_shape() {
    assert!(!CORE.contains("EXL3_PF_FIXED_BLOCK_THREADS"));
    assert!(CORE.contains("#if EXL3_PF_EXACT_FULL_GRID"));
    assert!(CORE.contains("mov.u32 n, %ntid.x; setp.ne.u32 p, n, 128; @p exit;"));
    assert!(CORE.contains("mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;"));
    assert!(CORE.contains("mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit;"));
    assert!(CORE.contains("mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;"));
    assert!(CORE.contains("const unsigned int m_warp = warp / EXL3_PF_N_WARPS;"));
    assert!(CORE.contains("load_vec += blockDim.x"));
}

#[test]
fn host_launches_fixed_shapes_as_m64_and_keeps_m128_on_its_generic_arm() {
    assert!(DISPATCH.contains("let direct_m_tile = if pf.direct_m128"));
    assert!(DISPATCH.contains("pf.grouped_direct_m128_k"));
    assert!(DISPATCH.contains(".block([direct_block_threads, 1, 1])"));
    assert!(STATE.contains("&& !direct_m128"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_FIXED_SHAPE"));
}
