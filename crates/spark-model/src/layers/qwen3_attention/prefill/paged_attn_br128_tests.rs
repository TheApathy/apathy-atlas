// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::kv_cache::KvCacheDtype;

use super::paged_attn::{
    Nvfp4PagedBr128Route as Route, nvfp4_paged_br128_route, parse_nvfp4_paged_br128,
};

const GATE_ENTRY: &str = include_str!("../../../../examples/nvfp4_paged_attn_br128_microgate.rs");
const GATE_CONTRACT: &str =
    include_str!("../../../../examples/nvfp4_paged_attn_br128_microgate/contract.rs");
const GATE_TIMING: &str =
    include_str!("../../../../examples/nvfp4_paged_attn_br128_microgate/timing.rs");

#[test]
fn selector_is_default_off_narrow_and_fail_closed() {
    assert_eq!(
        nvfp4_paged_br128_route(false, KvCacheDtype::Nvfp4, 8192, 24, 4, 256, false),
        Route::Disabled
    );
    for (dtype, q_len, nq, nkv, hd) in [
        (KvCacheDtype::Bf16, 8192, 24, 4, 256),
        (KvCacheDtype::Nvfp4, 2047, 24, 4, 256),
        (KvCacheDtype::Nvfp4, 8192, 0, 4, 256),
        (KvCacheDtype::Nvfp4, 8192, 24, 0, 256),
        (KvCacheDtype::Nvfp4, 8192, 25, 4, 256),
        (KvCacheDtype::Nvfp4, 8192, 24, 4, 128),
    ] {
        assert_eq!(
            nvfp4_paged_br128_route(true, dtype, q_len, nq, nkv, hd, true),
            Route::Ineligible
        );
    }
    assert_eq!(
        nvfp4_paged_br128_route(true, KvCacheDtype::Nvfp4, 2048, 24, 4, 256, false),
        Route::Missing
    );
    assert_eq!(
        nvfp4_paged_br128_route(true, KvCacheDtype::Nvfp4, 2048, 24, 4, 256, true),
        Route::Complete
    );
}

#[test]
fn explicit_flag_accepts_only_zero_or_one() {
    assert_eq!(parse_nvfp4_paged_br128(None), Ok(false));
    assert_eq!(parse_nvfp4_paged_br128(Some("0")), Ok(false));
    assert_eq!(parse_nvfp4_paged_br128(Some("1")), Ok(true));
    for invalid in ["", "true", "01", "2", " 1"] {
        assert!(parse_nvfp4_paged_br128(Some(invalid)).is_err());
    }
}

const fn parent_br64_limit(kv_len: u32, q_offset: u32, q_end: u32) -> u32 {
    let all = kv_len.div_ceil(32);
    let causal = (q_offset + q_end - 1) / 32 + 1;
    if all < causal { all } else { causal }
}

#[test]
fn merged_tile_retains_each_parent_causal_limit() {
    for (kv_len, q_offset, q_start, q_len, expected_lower, expected_full) in [
        (2048, 0, 0, 2048, 2, 4),
        (2048, 0, 1920, 2048, 62, 64),
        (8321, 17, 128, 8192, 7, 9),
        (9000, 33, 8064, 8192, 256, 258),
        (16384, 8192, 0, 8192, 258, 260),
        (24576, 16384, 4032, 8192, 640, 642),
        (32768, 24576, 8064, 8192, 1022, 1024),
    ] {
        let lower_end = (q_start + 64).min(q_len);
        let full_end = (q_start + 128).min(q_len);
        let lower = parent_br64_limit(kv_len, q_offset, lower_end);
        let upper = parent_br64_limit(kv_len, q_offset, full_end);
        assert!(lower <= upper);
        assert_eq!((lower, upper), (expected_lower, expected_full));
    }
}

#[test]
fn cuda_source_pins_aliasing_ownership_and_barrier_order() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cuda =
        std::fs::read_to_string(root.join("kernels/gb10/common/prefill_paged_nvfp4_br128.cuh"))
            .expect("BR128 CUDA source");

    assert!(cuda.contains("static_assert(BR128_SHARED_BYTES == 95808"));
    assert!(cuda.contains("__launch_bounds__(512, 1)"));
    assert!(cuda.contains("extern __shared__ __align__(16) unsigned char smem_raw128[]"));
    assert_eq!(
        cuda.matches("__nv_bfloat16 (*smem_KV128)[HDIM_PAD]")
            .count(),
        1,
        "K and V must use exactly one aliased tile"
    );
    assert!(cuda.contains("const unsigned int qk_warp_m = warp_id * 16"));
    assert!(cuda.contains("const unsigned int pv_warp_m = (warp_id & 7) * 16"));
    assert!(cuda.contains("const unsigned int pv_n_start = (warp_id >> 3)"));
    assert!(cuda.contains("lower_num_kv_blocks"));
    assert!(cuda.contains("warp_id < 8 && warp_tile_active"));
    assert!(cuda.contains("if (warp_tile_active)"));

    let qk = cuda.find("if (warp_id < 8 && warp_tile_active)").unwrap();
    let v_load = cuda
        .find("LOAD_KV_TILE(V_cache, block_table, smem_KV128")
        .unwrap();
    let pv = cuda.find("// === PV MMA (all 16 warps").unwrap();
    let next_k = cuda
        .find("LOAD_KV_TILE(K_cache, block_table, smem_KV128,\n                         (kv_block+1)*BC")
        .unwrap();
    assert!(qk < v_load && v_load < pv && pv < next_k);
    assert!(cuda[qk..v_load].contains("__syncthreads();"));
    assert!(cuda[v_load..pv].contains("__syncthreads();"));
    assert!(cuda[pv..next_k].contains("__syncthreads();"));
}

#[test]
fn host_launch_and_dispatch_pin_the_separate_abi() {
    let ops = include_str!("../../ops/prefill_attn_main_b.rs");
    let dispatch = include_str!("paged_attn.rs");
    let init = include_str!("../init.rs");

    let launch = ops
        .split("pub fn prefill_attention_paged_nvfp4_128")
        .nth(1)
        .expect("BR128 launcher")
        .split("/// Paged prefill (BR=64) for Bf16K")
        .next()
        .unwrap();
    assert!(launch.contains(".grid([num_q_heads, div_ceil(q_len, 128), 1])"));
    assert!(launch.contains(".block([512, 1, 1])"));
    assert!(launch.contains(".shared_mem(95_808)"));
    let abi = [
        ".arg_ptr(q)",
        ".arg_ptr(k_cache)",
        ".arg_ptr(v_cache)",
        ".arg_ptr(output)",
        ".arg_ptr(block_table)",
        ".arg_u32(q_len)",
        ".arg_u32(kv_len)",
        ".arg_u32(q_offset)",
        ".arg_u32(num_q_heads)",
        ".arg_u32(num_kv_heads)",
        ".arg_u32(head_dim)",
        ".arg_u32(cache_block_size)",
        ".arg_u32(sliding_window)",
        ".arg_u32(1u32)",
        ".arg_f32(inv_sqrt_d)",
        ".arg_u64(block_stride_bytes)",
        ".arg_u64(data_section_bytes)",
    ];
    let positions: Vec<_> = abi
        .iter()
        .map(|needle| {
            launch
                .find(needle)
                .unwrap_or_else(|| panic!("missing ABI argument {needle}"))
        })
        .collect();
    for pair in positions.windows(2) {
        assert!(pair[0] < pair[1], "BR128 launch ABI order changed");
    }
    let parent = ops
        .split("pub fn prefill_attention_paged_nvfp4_64")
        .nth(1)
        .expect("BR64 launcher")
        .split("pub fn prefill_attention_paged_nvfp4_128")
        .next()
        .unwrap();
    fn arg_lines(source: &str) -> Vec<&str> {
        source
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".arg_"))
            .collect::<Vec<_>>()
    }
    assert_eq!(
        arg_lines(parent),
        arg_lines(launch),
        "parent/candidate ABI drift"
    );
    assert!(dispatch.contains("Nvfp4PagedBr128Route::Missing"));
    assert!(dispatch.contains("ENGAGED ATLAS_PREFILL_ATTN_BR128: nvfp4-hd256-br128"));
    let missing = dispatch.find("Nvfp4PagedBr128Route::Missing").unwrap();
    let fallback = dispatch[missing..]
        .find("prefill_attention_paged_nvfp4_64")
        .map(|offset| missing + offset)
        .unwrap();
    assert!(
        missing < fallback,
        "Missing must fail before the BR64 fallback"
    );
    assert!(dispatch.contains("static ENGAGED: std::sync::Once"));
    assert!(init.contains("inferspark_prefill_paged_nvfp4_128"));
    let marker = dispatch.find("ENGAGED.call_once").unwrap();
    let launch = dispatch[..marker]
        .rfind("let outcome = ops::prefill_attention_paged_nvfp4_128(")
        .unwrap();
    assert!(dispatch[launch..marker].contains(")?;"));
    assert_eq!(dispatch.matches("ENGAGED.call_once").count(), 1);
    assert!(dispatch[marker..].contains("outcome"));
}

#[test]
fn production_route_is_paged_continuation_only() {
    let prefill = include_str!("../trait_impl/prefill_inner.rs");
    let zero = prefill.find("if seq_len_start == 0").unwrap();
    let cache_skip = prefill.find("prefill_attention_with_cache_skip").unwrap();
    let paged = prefill.find("prefill_attention_paged(").unwrap();
    assert!(zero < cache_skip && cache_skip < paged);
    assert!(prefill[cache_skip..paged].contains("} else {"));
    assert!(include_str!("paged_attn.rs").contains("chunk-1+ flash-attention path"));
}

#[test]
fn raw_gate_is_full_fail_closed_and_order_balanced() {
    let sources = [
        GATE_ENTRY,
        GATE_CONTRACT,
        GATE_TIMING,
        include_str!("../../../../examples/nvfp4_paged_attn_br128_microgate/fixtures.rs"),
        include_str!("../../../../examples/nvfp4_paged_attn_br128_microgate/guarded.rs"),
        include_str!("../../../../examples/nvfp4_paged_attn_br128_microgate/runtime.rs"),
    ];
    for source in sources {
        assert!(source.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
        assert!(source.lines().count() <= 250);
    }
    for offset in ["q_offset: 8192", "q_offset: 16384", "q_offset: 24576"] {
        assert!(GATE_CONTRACT.contains(offset));
    }
    assert!(GATE_CONTRACT.contains("TIMING_PAIRS: usize = 24"));
    assert!(GATE_ENTRY.contains("timing requires ATLAS_BR128_MICROGATE_FULL=1"));
    assert!(GATE_ENTRY.contains("BR128_SMOKE_PASS cases=2 qualified=false"));
    assert!(GATE_ENTRY.contains("resources.max_threads >= 512"));
    assert!(GATE_ENTRY.contains("function_resources(candidate)"));
    assert!(GATE_ENTRY.contains("resources.shared_bytes == 0"));
    assert!(GATE_ENTRY.contains("resources.local_bytes == 64"));
    assert!(GATE_ENTRY.contains("resources.max_dynamic_shared_bytes == 95_808"));
    assert!(GATE_CONTRACT.contains("let dynamic_shared_bytes = 95_808"));
    assert!(GATE_CONTRACT.contains("cuFuncSetAttribute(function, 8, dynamic_shared_bytes)"));
    assert!(GATE_CONTRACT.contains("max_dynamic_shared_bytes: attr(8)?"));
    assert!(!GATE_ENTRY.contains("ALL PASS:"));
    assert!(GATE_TIMING.contains("parent_first.push(delta)"));
    assert!(GATE_TIMING.contains("candidate_first.push(delta)"));
    assert!(GATE_TIMING.contains("order-bias gate failed"));
    assert!(GATE_TIMING.contains("metrics.5 > 0.0 && metrics.6 > 0.0"));
}
