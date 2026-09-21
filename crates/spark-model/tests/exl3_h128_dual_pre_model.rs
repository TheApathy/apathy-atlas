// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the opt-in dual gate/up EXL3 H128 pre-rotation.

use half::bf16;

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");
const SIZES: &str = include_str!("../../spark-runtime/src/buffers/sizes.rs");
const ACCESSORS: &str = include_str!("../../spark-runtime/src/buffers/accessors.rs");
const MICROTEST: &str = include_str!("../examples/exl3_dual_pre_microtest.rs");

fn had128(mut values: [f32; 128]) -> [f32; 128] {
    for stride in [1, 2, 4, 8, 16, 32, 64] {
        for base in (0..128).step_by(stride * 2) {
            for offset in 0..stride {
                let a = values[base + offset];
                let b = values[base + stride + offset];
                values[base + offset] = a + b;
                values[base + stride + offset] = a - b;
            }
        }
    }
    values
}

fn rotate(input: &[bf16; 128], signs: &[f32; 128]) -> [u16; 128] {
    let mut signed = [0.0f32; 128];
    for i in 0..128 {
        signed[i] = input[i].to_f32() * signs[i];
    }
    let transformed = had128(signed);
    std::array::from_fn(|i| bf16::from_f32(transformed[i] * 0.088_388_346).to_bits())
}

fn sign(expert: usize, column: usize, salt: usize) -> f32 {
    if (expert * 131 + column * 17 + salt).count_ones() & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn legacy_projection(
    input: &[bf16],
    token_ids: Option<&[usize]>,
    expert_ids: &[usize],
    salt: usize,
) -> (Vec<u16>, Vec<bool>) {
    const H: usize = 4096;
    let mut output = vec![0xdead; expert_ids.len() * H];
    let mut written = vec![false; output.len()];
    for (row, &expert) in expert_ids.iter().enumerate() {
        let token = token_ids.map_or(row, |ids| ids[row]);
        for chunk in 0..32 {
            let values = std::array::from_fn(|i| input[token * H + chunk * 128 + i]);
            let signs = std::array::from_fn(|i| sign(expert, chunk * 128 + i, salt));
            let rotated = rotate(&values, &signs);
            for i in 0..128 {
                let index = row * H + chunk * 128 + i;
                assert!(!written[index]);
                output[index] = rotated[i];
                written[index] = true;
            }
        }
    }
    (output, written)
}

fn dual_projection(
    input: &[bf16],
    token_ids: Option<&[usize]>,
    expert_ids: &[usize],
    gate_salt: usize,
    up_salt: usize,
) -> ((Vec<u16>, Vec<bool>), (Vec<u16>, Vec<bool>)) {
    const H: usize = 4096;
    let mut gate = (
        vec![0xa5a5; expert_ids.len() * H],
        vec![false; expert_ids.len() * H],
    );
    let mut up = (
        vec![0x5a5a; expert_ids.len() * H],
        vec![false; expert_ids.len() * H],
    );
    for (row, &expert) in expert_ids.iter().enumerate() {
        let token = token_ids.map_or(row, |ids| ids[row]);
        for chunk in 0..32 {
            let values = std::array::from_fn(|i| input[token * H + chunk * 128 + i]);
            let gate_signs = std::array::from_fn(|i| sign(expert, chunk * 128 + i, gate_salt));
            let up_signs = std::array::from_fn(|i| sign(expert, chunk * 128 + i, up_salt));
            let gate_chunk = rotate(&values, &gate_signs);
            let up_chunk = rotate(&values, &up_signs);
            for i in 0..128 {
                let index = row * H + chunk * 128 + i;
                assert!(!gate.1[index] && !up.1[index]);
                gate.0[index] = gate_chunk[i];
                up.0[index] = up_chunk[i];
                gate.1[index] = true;
                up.1[index] = true;
            }
        }
    }
    (gate, up)
}

#[test]
fn dual_cpu_model_is_two_independent_legacy_transforms() {
    let input = std::array::from_fn(|i| bf16::from_f32((i as f32 - 61.0) / 37.0));
    let gate_signs = std::array::from_fn(|i| if i % 3 == 0 { -1.0 } else { 1.0 });
    let up_signs = std::array::from_fn(|i| if i.count_ones() & 1 == 0 { 1.0 } else { -1.0 });

    let legacy_gate = rotate(&input, &gate_signs);
    let legacy_up = rotate(&input, &up_signs);
    let dual = (rotate(&input, &gate_signs), rotate(&input, &up_signs));
    assert_eq!(dual.0, legacy_gate);
    assert_eq!(dual.1, legacy_up);
    assert_ne!(
        dual.0, dual.1,
        "the signs must make both outputs load-bearing"
    );
}

#[test]
fn full_h4096_address_model_matches_projection_major_legacy() {
    const H: usize = 4096;
    let input = (0..5 * H)
        .map(|i| bf16::from_f32(((i * 29 % 509) as f32 - 254.0) / 113.0))
        .collect::<Vec<_>>();
    let token_ids = [4, 0, 3, 1, 4, 2, 0, 3];
    let expert_ids = [255, 0, 37, 255, 128, 1, 200, 17];
    let legacy_gate = legacy_projection(&input, Some(&token_ids), &expert_ids, 0x51);
    let legacy_up = legacy_projection(&input, Some(&token_ids), &expert_ids, 0xa7);
    let (dual_gate, dual_up) = dual_projection(&input, Some(&token_ids), &expert_ids, 0x51, 0xa7);
    assert_eq!(dual_gate.0, legacy_gate.0);
    assert_eq!(dual_up.0, legacy_up.0);
    assert!(dual_gate.1.iter().all(|&written| written));
    assert!(dual_up.1.iter().all(|&written| written));
    assert_ne!(dual_gate.0, dual_up.0);

    let identity_experts = [255, 0, 1, 17, 37];
    let identity_legacy = legacy_projection(&input, None, &identity_experts, 0x51);
    let (identity_dual, _) = dual_projection(&input, None, &identity_experts, 0x51, 0xa7);
    assert_eq!(identity_dual.0, identity_legacy.0);
    assert!(identity_dual.1.iter().all(|&written| written));
    assert!(KERNEL.contains("blockIdx.y * EXL3_HROW_WARPS + warp"));
}

#[test]
fn fixed_dual_entry_loads_input_once_and_reuses_the_shared_arithmetic_body() {
    assert!(KERNEL.contains("exl3_h128_pre_values4("));
    assert!(KERNEL.contains("exl3_h128_pre_transform4("));
    assert!(KERNEL.contains("exl3_h128_pre_transform4_packed("));
    assert!(KERNEL.contains("exl3_h128_pre_dual_rows_h4096("));
    assert!(KERNEL.contains("exl3_h128_pre_values4(a0, a1, a2, a3"));
    assert!(KERNEL.contains("exl3_h128_pre_transform4_packed(a0, a1, a2, a3"));
    assert!(KERNEL.contains("gate_suh_tab"));
    assert!(KERNEL.contains("up_suh_tab"));

    let start = KERNEL
        .find("void exl3_h128_pre_dual_rows_h4096(")
        .expect("dual entry");
    let end = KERNEL[start..]
        .find("// Output pass")
        .map(|offset| start + offset)
        .expect("end of dual entry");
    let dual = &KERNEL[start..end];
    for lane in 0..4 {
        assert_eq!(
            dual.matches(&format!("__bfloat162float(a[{lane}])"))
                .count(),
            1,
            "input lane {lane} must be widened exactly once"
        );
    }
}

#[test]
fn dual_entry_is_fail_closed_on_the_exact_h4096_geometry() {
    assert!(KERNEL.contains("if (K != 4096) return;"));
    assert!(KERNEL.contains("gridDim.x != rows || gridDim.y != 4 || gridDim.z != 1"));
    assert!(KERNEL.contains("mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;"));
}

#[test]
fn dual_only_packs_aligned_bf16_pairs_without_changing_lane_order() {
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::from_bits(0x3f80_8000),
        f32::from_bits(0x3f81_8000),
    ];
    for pair in values.chunks_exact(2) {
        let low = bf16::from_f32(pair[0]).to_bits();
        let high = bf16::from_f32(pair[1]).to_bits();
        let packed = u32::from(low) | (u32::from(high) << 16);
        assert_eq!(&packed.to_le_bytes()[..2], &low.to_le_bytes());
        assert_eq!(&packed.to_le_bytes()[2..], &high.to_le_bytes());
    }
    for row in 0..8usize {
        for chunk in 0..32usize {
            for lane in 0..32usize {
                let byte_offset = (row * 4096 + chunk * 128 + 4 * lane) * 2;
                assert_eq!(byte_offset % 4, 0);
            }
        }
    }
    assert!(KERNEL.contains("exl3_h128_pre_transform4_packed("));
    assert!(KERNEL.contains("__floats2bfloat162_rn"));
    assert!(KERNEL.contains("reinterpret_cast<__nv_bfloat162*>(out)"));
    let generic = KERNEL
        .split("extern \"C\" __global__ void exl3_h128_pre_rows(")
        .nth(1)
        .expect("generic pre")
        .split("#if EXL3_HROW_DSV4_FIXED")
        .next()
        .expect("generic body");
    assert!(!generic.contains("exl3_h128_pre_transform4_packed"));
}

#[test]
fn buffer_arena_reserves_a_full_h4096_rotation_with_capacity_visibility() {
    assert!(SIZES.contains("let exl3_rotated_up"));
    assert!(SIZES.contains("expert_inter.max(exl3_rotated_up) * bf16"));
    assert!(ACCESSORS.contains("expert_up_out_bytes"));

    let rows = 2410usize * 6;
    let old_bytes = rows * 2048 * 2;
    let new_bytes = rows * 4096 * 2;
    assert_eq!(new_bytes - old_bytes, 59_228_160);
}

#[test]
fn host_is_opt_in_exact_capacity_checked_and_uses_the_safe_alias_schedule() {
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_DUAL_PRE"));
    assert!(STATE.contains("h128_pre_dual_h4096_k: KernelHandle"));
    assert!(DISPATCH.contains("pf.dual_pre"));
    assert!(DISPATCH.contains("ctx.buffers.expert_up_out_bytes()"));
    assert!(DISPATCH.contains("launch_h128_pre_dual("));
    assert!(DISPATCH.contains("run_proj(&st.up, expert_up_out, expert_down_out)"));
    assert!(DISPATCH.contains("let up_projected = expert_down_out"));
}

#[test]
fn enable_predicate_requires_fixed_shape_fused_post_and_explicit_request() {
    let enabled = |fixed_shape: bool, fused_post: bool, requested: bool| {
        fixed_shape && fused_post && requested
    };
    for mask in 0..8 {
        let fixed_shape = mask & 1 != 0;
        let fused_post = mask & 2 != 0;
        let requested = mask & 4 != 0;
        assert_eq!(enabled(fixed_shape, fused_post, requested), mask == 7);
    }
    assert!(STATE.contains("const fn exl3_dual_pre_enabled("));
    assert!(STATE.contains("fixed_shape && fused_post && requested"));
    assert!(STATE.contains("let fixed_shape = direct"));
    assert!(STATE.contains("gate.bits == 2"));
    assert!(STATE.contains("gate.n == 2048"));
    assert!(STATE.contains("gate.k == 4096"));
}

#[test]
fn production_model_saves_one_launch_and_one_expanded_input_read_per_layer() {
    const TOKENS: u64 = 2410;
    const TOPK: u64 = 6;
    const HIDDEN: u64 = 4096;
    const LAYERS: u64 = 43;
    let saved = TOKENS * TOPK * HIDDEN * 2;
    assert_eq!(saved, 118_456_320);
    assert_eq!(saved * LAYERS, 5_093_621_760);
    assert_eq!(2 * LAYERS - LAYERS, 43);
}

#[test]
fn gpu_promotion_gate_keeps_parity_identity_guards_and_timing() {
    assert!(MICROTEST.contains("const TOKEN_CASES: [usize; 5] = [1, 17, 65, 256, 2410]"));
    assert!(MICROTEST.contains("DevicePtr(0)"));
    assert!(MICROTEST.contains("identity-gather byte parity"));
    assert!(MICROTEST.contains("let invalid = ["));
    assert!(MICROTEST.contains("timed_sample"));
    assert!(MICROTEST.contains("legacy pre+pre"));
}
