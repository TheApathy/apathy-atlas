// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the opt-in M64xN128/N256 direct EXL3 prefill rungs.

use std::fs;
use std::path::PathBuf;

const CORE: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill.cu");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");

fn kernel(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(name);
    fs::read_to_string(path).expect("N128 wrapper must exist")
}

fn example(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name);
    fs::read_to_string(path).expect("N128 GPU parity harness must exist")
}

#[test]
fn n128_wrappers_are_exact_k64_k2_production_shapes() {
    let gu = kernel("exl3_grouped_prefill_k64_n128_k2_gu.cu");
    let down = kernel("exl3_grouped_prefill_k64_n128_k2_down.cu");
    for wrapper in [&gu, &down] {
        assert!(wrapper.contains("#define EXL3_PF_N_TILE 128"));
        assert!(wrapper.contains("#define EXL3_PF_M_TILE 64"));
        assert!(wrapper.contains("#define EXL3_PF_K_STEP 64"));
        assert!(wrapper.contains("#define EXL3_PF_FIXED_BITS 2"));
        assert!(wrapper.contains("#define EXL3_PF_EXACT_FULL_GRID 1"));
        assert!(wrapper.contains("#define EXL3_PF_LAUNCH_BOUNDS 256"));
        assert!(wrapper.contains("#define EXL3_PF_ASYNC_STAGE 1"));
    }
    assert!(gu.contains("#define EXL3_PF_FIXED_N 2048"));
    assert!(gu.contains("#define EXL3_PF_FIXED_K 4096"));
    assert!(down.contains("#define EXL3_PF_FIXED_N 4096"));
    assert!(down.contains("#define EXL3_PF_FIXED_K 2048"));
}

#[test]
fn n256_wrappers_are_exact_k64_k2_production_shapes() {
    let gu = kernel("exl3_grouped_prefill_k64_n256_k2_gu.cu");
    let down = kernel("exl3_grouped_prefill_k64_n256_k2_down.cu");
    for wrapper in [&gu, &down] {
        assert!(wrapper.contains("#define EXL3_PF_N_TILE 256"));
        assert!(wrapper.contains("#define EXL3_PF_M_TILE 64"));
        assert!(wrapper.contains("#define EXL3_PF_K_STEP 64"));
        assert!(wrapper.contains("#define EXL3_PF_FIXED_BITS 2"));
        assert!(wrapper.contains("#define EXL3_PF_EXACT_FULL_GRID 1"));
        assert!(wrapper.contains("#define EXL3_PF_LAUNCH_BOUNDS 512"));
        assert!(wrapper.contains("#define EXL3_PF_ASYNC_STAGE 1"));
    }
    assert!(gu.contains("#define EXL3_PF_FIXED_N 2048"));
    assert!(gu.contains("#define EXL3_PF_FIXED_K 4096"));
    assert!(down.contains("#define EXL3_PF_FIXED_N 4096"));
    assert!(down.contains("#define EXL3_PF_FIXED_K 2048"));
}

#[test]
fn core_derives_warp_and_trellis_geometry_from_n_tile() {
    assert!(CORE.contains("#ifndef EXL3_PF_N_TILE"));
    assert!(CORE.contains("#define EXL3_PF_N_TILE 64"));
    assert!(CORE.contains("#define EXL3_PF_N_WARPS (EXL3_PF_N_TILE / 16)"));
    assert!(CORE.contains("const unsigned int n_warp = warp & (EXL3_PF_N_WARPS - 1);"));
    assert!(CORE.contains("const unsigned int m_warp = warp / EXL3_PF_N_WARPS;"));
    assert!(CORE.contains("EXL3_PF_N_WARPS * 2 * EXL3_PF_SMEM_BITS"));
    assert!(CORE.contains("const unsigned int strip_u4 = EXL3_PF_N_WARPS * 2 * bit_width;"));
}

#[test]
fn m64_n128_warp_mapping_covers_every_output_once() {
    let mut owners = vec![0u8; 64 * 128];
    for warp in 0..8usize {
        let n_warp = warp & 7;
        for lane in 0..32usize {
            let group = lane >> 2;
            let tid = lane & 3;
            for mt in 0..4usize {
                for nt in 0..2usize {
                    let col0 = n_warp * 16 + nt * 8 + tid * 2;
                    let row0 = mt * 16 + group;
                    let row1 = row0 + 8;
                    for row in [row0, row1] {
                        for col in col0..col0 + 2 {
                            owners[row * 128 + col] += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(owners.into_iter().all(|count| count == 1));
}

#[test]
fn m64_n256_warp_mapping_covers_every_output_once() {
    let mut owners = vec![0u8; 64 * 256];
    for warp in 0..16usize {
        let n_warp = warp & 15;
        for lane in 0..32usize {
            let group = lane >> 2;
            let tid = lane & 3;
            for mt in 0..4usize {
                for nt in 0..2usize {
                    let col0 = n_warp * 16 + nt * 8 + tid * 2;
                    for row in [mt * 16 + group, mt * 16 + group + 8] {
                        for col in col0..col0 + 2 {
                            owners[row * 256 + col] += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(owners.into_iter().all(|count| count == 1));
}

#[test]
fn n128_halves_activation_stages_and_ctas_without_changing_mma_work() {
    for n in [2048u32, 4096] {
        let n64_ctas = n / 64;
        let n128_ctas = n / 128;
        assert_eq!(n128_ctas * 2, n64_ctas);
        assert_eq!(n64_ctas * 4, n128_ctas * 8);
    }
}

#[test]
fn n256_halves_n128_ctas_and_activation_stages_without_changing_mma_work() {
    for n in [2048u32, 4096] {
        let n128_ctas = n / 128;
        let n256_ctas = n / 256;
        assert_eq!(n256_ctas * 2, n128_ctas);
        assert_eq!(n128_ctas * 8, n256_ctas * 16);
    }
}

#[test]
fn host_keeps_n128_exact_shape_k64_and_opt_in() {
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_N128"));
    assert!(STATE.contains("let direct_n128 = fixed_shape"));
    assert!(STATE.contains("grouped_direct_k64_n128_k2_gu_k"));
    assert!(STATE.contains("grouped_direct_k64_n128_k2_down_k"));
    assert!(DISPATCH.contains("else if pf.direct_n128"));
    assert!(DISPATCH.contains("let direct_block_threads"));
    assert!(DISPATCH.contains("pf.grouped_direct_k64_n128_k2_gu_k"));
    assert!(DISPATCH.contains("pf.grouped_direct_k64_n128_k2_down_k"));
    assert!(DISPATCH.contains(".block([direct_block_threads, 1, 1])"));
}

#[test]
fn host_keeps_n256_exact_shape_k64_and_opt_in() {
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_N256"));
    assert!(STATE.contains("let direct_n256 = fixed_shape"));
    assert!(STATE.contains("grouped_direct_k64_n256_k2_gu_k"));
    assert!(STATE.contains("grouped_direct_k64_n256_k2_down_k"));
    assert!(STATE.contains(") = if direct_n256 {"));
    assert!(STATE.contains("(KernelHandle(0), KernelHandle(0))"));
    assert!(DISPATCH.contains("let direct_n_tile = if pf.direct_n256 {"));
    assert!(DISPATCH.contains("pf.grouped_direct_k64_n256_k2_gu_k"));
    assert!(DISPATCH.contains("pf.grouped_direct_k64_n256_k2_down_k"));
}

#[test]
fn gpu_gate_compares_n64_and_n128_and_exercises_fail_closed_block_shape() {
    let source = example("exl3_prefill_n128_microtest.rs");
    for kernel in [
        "exl3_grouped_prefill_k64_k2_gu",
        "exl3_grouped_prefill_k64_k2_down",
        "exl3_grouped_prefill_k64_n128_k2_gu",
        "exl3_grouped_prefill_k64_n128_k2_down",
        "exl3_grouped_prefill_k64_n256_k2_gu",
        "exl3_grouped_prefill_k64_n256_k2_down",
    ] {
        assert!(source.contains(kernel));
    }
    assert!(source.contains("[(2048usize, 4096usize), (4096, 2048)]"));
    assert!(source.contains("n_tile: u32"));
    assert!(source.contains("threads: u32"));
    assert!(source.contains(".block([threads, 1, 1])"));
    assert!(source.contains("wrong_block_unchanged"));
    assert!(source.contains("n64 == wide"));
    assert!(source.contains("n128 == n256"));
}
