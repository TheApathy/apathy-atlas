// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source and tie contract for the shared ordinary BF16 argmax.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/argmax_bf16.cu");
const THREADS: usize = 1024;
const ARGMAX_ADD_FNV1A64: u64 = 0x96ff_dd14_b3ae_726a;

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

fn legacy_value_only_argmax(logits: &[f32]) -> usize {
    let mut values = vec![f32::NEG_INFINITY; THREADS];
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
fn value_only_tree_exposes_the_1024_over_1_tie_failure() {
    let mut logits = vec![-8.0; 2_048];
    logits[1] = 7.0;
    logits[1_024] = 7.0;

    let mut host_lowest = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[host_lowest] {
            host_lowest = index;
        }
    }
    assert_eq!(host_lowest, 1);
    assert_eq!(legacy_value_only_argmax(&logits), 1_024);
}

#[test]
fn ordinary_argmax_abi_and_launch_contract_remain_exact() {
    assert!(KERNEL.contains("// Grid: (1, 1, 1)  Block: (1024, 1, 1)"));
    let ordinary = section(
        KERNEL,
        "extern \"C\" __global__ void argmax_bf16(",
        "// Argmax over the elementwise sum",
    );
    let signature_end = ordinary
        .find(") {")
        .map(|index| index + 3)
        .expect("ordinary argmax signature must remain present");
    assert_eq!(
        compact(&ordinary[..signature_end]),
        "extern\"C\"__global__voidargmax_bf16(const__nv_bfloat16*__restrict__logits,unsignedint*__restrict__out,unsignedintn){"
    );
}

#[test]
fn ordinary_argmax_has_lowest_id_secondary_keys_in_both_phases() {
    let ordinary = compact(section(
        KERNEL,
        "extern \"C\" __global__ void argmax_bf16(",
        "// Argmax over the elementwise sum",
    ));
    assert!(ordinary.contains("if(v>local_max||(v==local_max&&i<local_idx)){"));
    assert!(ordinary.contains(
        "if(s_val[tid+s]>s_val[tid]||(s_val[tid+s]==s_val[tid]&&s_idx[tid+s]<s_idx[tid])){"
    ));
}

#[test]
fn argmax_add_function_is_byte_pinned() {
    let argmax_add = section(
        KERNEL,
        "extern \"C\" __global__ void argmax_add_bf16(",
        "// Argmax over FP32 logits",
    );
    assert_eq!(fnv1a64(argmax_add), ARGMAX_ADD_FNV1A64);
}
