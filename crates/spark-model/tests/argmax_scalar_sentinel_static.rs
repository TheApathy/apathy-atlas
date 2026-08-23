// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source and host-simulation contract for scalar BF16/FP32 argmax.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/argmax_bf16.cu");
const WRAPPER: &str = include_str!("../src/layers/ops/sampling.rs");
const THREADS: usize = 1024;
const ARGMAX_ADD_FNV1A64: u64 = 0xf32c_1aa6_79ab_3d3c;
const TOPK_FNV1A64: u64 = 0x7a07_c3cf_77ba_4655;

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let end_index = source[start_index..]
        .find(end)
        .map(|offset| start_index + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start_index..end_index]
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn fnv1a64(source: &str) -> u64 {
    source.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn host_strict_argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty());
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    best
}

fn legacy_scalar_argmax(logits: &[f32]) -> usize {
    let mut values = vec![-1.0e30_f32; THREADS];
    let mut indices = vec![0usize; THREADS];
    for tid in 0..THREADS {
        for index in (tid..logits.len()).step_by(THREADS) {
            if logits[index] > values[tid] {
                values[tid] = logits[index];
                indices[tid] = index;
            }
        }
    }

    let mut stride = THREADS / 2;
    while stride > 0 {
        for tid in 0..stride {
            if values[tid + stride] > values[tid] {
                values[tid] = values[tid + stride];
                indices[tid] = indices[tid + stride];
            }
        }
        stride /= 2;
    }
    indices[0]
}

#[test]
fn legacy_bf16_sentinel_loses_the_true_extreme_negative_winner() {
    let logits = [
        f32::from_bits(u32::from(0xff7f_u16) << 16),
        f32::from_bits(u32::from(0xff7e_u16) << 16),
    ];
    assert!(logits[1] > logits[0]);
    assert_eq!(host_strict_argmax(&logits), 1);
    assert_eq!(legacy_scalar_argmax(&logits), 0);
}

#[test]
fn legacy_fp32_sentinel_loses_the_true_extreme_negative_winner() {
    let logits = [-f32::MAX, -3.0e38_f32];
    assert!(logits[1] > logits[0]);
    assert_eq!(host_strict_argmax(&logits), 1);
    assert_eq!(legacy_scalar_argmax(&logits), 0);
}

#[test]
fn legacy_fp32_value_only_tree_selects_1024_over_lower_id_1() {
    let mut logits = vec![-8.0_f32; 2_048];
    logits[1] = 7.0;
    logits[1_024] = 7.0;

    assert_eq!(host_strict_argmax(&logits), 1);
    assert_eq!(legacy_scalar_argmax(&logits), 1_024);
}

#[test]
fn scalar_candidates_and_secondary_keys_cover_both_reduction_phases() {
    let ordinary = compact(section(
        KERNEL,
        "extern \"C\" __global__ void argmax_bf16(",
        "// Argmax over the elementwise sum",
    ));
    assert!(ordinary.contains("floatlocal_max=-CUDART_INF_F;"));
    assert!(ordinary.contains("unsignedintlocal_idx=0xFFFFFFFFu;"));
    assert!(ordinary.contains("if(v>local_max||(v==local_max&&i<local_idx)){"));
    assert!(ordinary.contains(
        "if(s_val[tid+s]>s_val[tid]||(s_val[tid+s]==s_val[tid]&&s_idx[tid+s]<s_idx[tid])){"
    ));

    let fp32 = compact(section(
        KERNEL,
        "extern \"C\" __global__ void argmax_fp32(",
        "// Top-K over BF16 logits",
    ));
    assert!(fp32.contains("floatlocal_max=-CUDART_INF_F;"));
    assert!(fp32.contains("unsignedintlocal_idx=0xFFFFFFFFu;"));
    assert!(fp32.contains("if(v>local_max||(v==local_max&&i<local_idx)){"));
    assert!(fp32.contains(
        "if(s_val[tid+s]>s_val[tid]||(s_val[tid+s]==s_val[tid]&&s_idx[tid+s]<s_idx[tid])){"
    ));
}

#[test]
fn scalar_abis_and_single_block_launch_geometry_remain_exact() {
    let ordinary = section(
        KERNEL,
        "extern \"C\" __global__ void argmax_bf16(",
        "// Argmax over the elementwise sum",
    );
    let ordinary_signature_end = ordinary
        .find(") {")
        .map(|index| index + 3)
        .expect("ordinary argmax signature must remain present");
    assert_eq!(
        compact(&ordinary[..ordinary_signature_end]),
        "extern\"C\"__global__voidargmax_bf16(const__nv_bfloat16*__restrict__logits,unsignedint*__restrict__out,unsignedintn){"
    );

    let fp32 = section(
        KERNEL,
        "extern \"C\" __global__ void argmax_fp32(",
        "// Top-K over BF16 logits",
    );
    let fp32_signature_end = fp32
        .find(") {")
        .map(|index| index + 3)
        .expect("FP32 argmax signature must remain present");
    assert_eq!(
        compact(&fp32[..fp32_signature_end]),
        "extern\"C\"__global__voidargmax_fp32(constfloat*__restrict__logits,unsignedint*__restrict__out,unsignedintn){"
    );

    let wrapper = compact(section(
        WRAPPER,
        "pub fn argmax_bf16(",
        "/// GPU-side argmax over `base_logits + bias`",
    ));
    assert!(wrapper.contains(".grid([1,1,1]).block([1024,1,1])"));
    assert!(wrapper.contains(".arg_ptr(logits).arg_ptr(out).arg_u32(vocab_size).launch(stream)"));
}

#[test]
fn argmax_add_and_topk_regions_are_byte_pinned() {
    let argmax_add = section(
        KERNEL,
        "// Argmax over the elementwise sum",
        "// Argmax over FP32 logits",
    );
    let topk_start = KERNEL
        .find("// Top-K over BF16 logits")
        .expect("top-K section must remain present");
    let topk = &KERNEL[topk_start..];

    assert_eq!(fnv1a64(argmax_add), ARGMAX_ADD_FNV1A64);
    assert_eq!(fnv1a64(topk), TOPK_FNV1A64);
}
