// SPDX-License-Identifier: AGPL-3.0-only

//! Offline fail-closed ABI contracts for DeepSeek-V4 TC2 prefill attention.

const CUDA: &str =
    include_str!("../../../kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu");
const DISPATCH: &str = include_str!("../src/layers/qwen3_attention/prefill/cache_skip_v4.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn tc2_body_prefix() -> &'static str {
    let signature = "extern \"C\" __global__ void prefill_attn_compressed_tc2(";
    let start = CUDA.find(signature).expect("TC2 entry must exist");
    let body = &CUDA[start..];
    let end = body
        .find("const unsigned int q_head = blockIdx.x;")
        .expect("TC2 q-head indexing must exist");
    &body[..end]
}

#[test]
fn tc2_guard_precedes_indexing_and_covers_the_complete_launch_abi() {
    let prefix = compact(tc2_body_prefix());
    for contract in [
        "blockDim.x!=128u",
        "blockDim.y!=1u",
        "blockDim.z!=1u",
        "gridDim.x!=num_q_heads",
        "gridDim.y!=required_q_blocks",
        "gridDim.z!=1u",
        "num_kv_heads==0u",
        "num_q_heads%num_kv_heads!=0u",
        "ratio==0u",
        "Q==nullptr",
        "K==nullptr",
        "V==nullptr",
        "Kc==nullptr",
        "Vc==nullptr",
        "O==nullptr",
    ] {
        assert!(
            prefix.contains(contract),
            "TC2 ABI guard omits `{contract}`"
        );
    }
    assert!(prefix.contains("if(") && prefix.contains(")return;"));

    let zero_divisor = prefix.find("num_kv_heads==0u").unwrap();
    let divisibility = prefix.find("num_q_heads%num_kv_heads!=0u").unwrap();
    assert!(
        zero_divisor < divisibility,
        "zero check must short-circuit modulo"
    );
}

#[test]
fn both_production_launches_match_the_guarded_geometry_and_head_arguments() {
    let dispatch = compact(DISPATCH);
    let geometry = ".grid([nq,n.div_ceil(16),1]).block([128,1,1])";
    assert_eq!(dispatch.matches(geometry).count(), 2);
    assert!(dispatch.contains(&format!("KernelLaunch::new(ctx.gpu,attn_k){geometry}")));
    assert!(dispatch.contains(&format!("KernelLaunch::new(ctx.gpu,tc_dense_k){geometry}")));
    assert_eq!(dispatch.matches(".arg_u32(nq).arg_u32(nkv)").count(), 2);
}

#[derive(Clone, Copy)]
struct Launch {
    grid: [u32; 3],
    block: [u32; 3],
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    ratio: u32,
    ptrs: [bool; 6],
}

fn abi_accepts(launch: Launch) -> bool {
    let required_q_blocks = launch.seq_len / 16 + u32::from(launch.seq_len % 16 != 0);
    launch.block == [128, 1, 1]
        && launch.grid == [launch.num_q_heads, required_q_blocks, 1]
        && launch.num_kv_heads > 0
        && launch.num_q_heads % launch.num_kv_heads == 0
        && launch.ratio > 0
        && launch.ptrs.into_iter().all(|ptr| ptr)
}

#[test]
fn abi_model_rejects_each_malformed_geometry_and_divisor_independently() {
    let valid = Launch {
        grid: [64, 151, 1],
        block: [128, 1, 1],
        seq_len: 2410,
        num_q_heads: 64,
        num_kv_heads: 1,
        ratio: 4,
        ptrs: [true; 6],
    };
    assert!(abi_accepts(valid));
    assert!(abi_accepts(Launch {
        num_kv_heads: 8,
        ..valid
    }));

    for block in [[127, 1, 1], [129, 1, 1], [128, 2, 1], [128, 1, 2]] {
        assert!(!abi_accepts(Launch { block, ..valid }));
    }
    for grid in [
        [63, 151, 1],
        [65, 151, 1],
        [64, 150, 1],
        [64, 152, 1],
        [64, 151, 2],
    ] {
        assert!(!abi_accepts(Launch { grid, ..valid }));
    }
    for (num_q_heads, num_kv_heads, ratio) in [(64, 0, 4), (64, 3, 4), (64, 1, 0)] {
        assert!(!abi_accepts(Launch {
            num_q_heads,
            num_kv_heads,
            ratio,
            ..valid
        }));
    }
}

#[test]
fn abi_model_rejects_each_null_mandatory_pointer() {
    let valid = Launch {
        grid: [64, 10, 1],
        block: [128, 1, 1],
        seq_len: 160,
        num_q_heads: 64,
        num_kv_heads: 1,
        ratio: 128,
        ptrs: [true; 6],
    };

    for null_index in 0..valid.ptrs.len() {
        let mut ptrs = valid.ptrs;
        ptrs[null_index] = false;
        assert!(!abi_accepts(Launch { ptrs, ..valid }));
    }
}
