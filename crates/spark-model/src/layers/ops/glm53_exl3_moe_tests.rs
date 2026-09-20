// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const ROUTE_CUDA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_route.cu"
));
const STAGED_CUDA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_staged_k16.cu"
));
const STAGED_N256_F1_CUDA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_staged_n256_f1.cu"
));
const STAGED_PRIVATE_CUDA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_staged_private.cu"
));

#[test]
fn k8_plan_pins_the_published_verify_geometry() {
    let plan = Glm53Exl3MoePlan::new(8).unwrap();
    assert_eq!(plan.pairs, 64);
    assert_eq!(plan.input_f16_bytes, 65_536);
    assert_eq!(plan.output_f32_bytes, 131_072);
    assert_eq!(plan.expert_count_i64_bytes, 2_312);
    assert_eq!(plan.token_sorted_i64_bytes, 512);
    assert_eq!(plan.weight_sorted_f16_bytes, 128);
    assert_eq!(plan.temp_state_f16_bytes, 524_288);
    assert_eq!(plan.temp_intermediate_f16_bytes, 262_144);
    assert_eq!(plan.pointer_table_u64_bytes, 2_304);
    assert_eq!(plan.pair_expert_u32_bytes, 256);
    assert_eq!(plan.max_chunks, 292);
    assert_eq!(plan.chunk_descriptor_u32_bytes, 1_168);
    assert_eq!(plan.chunk_count_u32_bytes, 4);
    assert_eq!(GLM53_EXL3_MOE_LOCK_BYTES, 4_202_760);
}

#[test]
fn verifier_width_is_fail_closed() {
    assert!(Glm53Exl3MoePlan::new(0).is_err());
    assert!(Glm53Exl3MoePlan::new(MAX_ROWS).is_ok());
    assert!(Glm53Exl3MoePlan::new(MAX_ROWS + 1).is_err());
}

#[test]
fn route_private_selector_scope_and_extent_are_exact() {
    let parse_route_private_enabled = |value: Option<&str>| {
        Glm53Exl3RoutePolicy::parse(value.map(std::ffi::OsStr::new), None)?.private_for(1)
    };
    assert!(!parse_route_private_enabled(None).unwrap());
    assert!(!parse_route_private_enabled(Some("0")).unwrap());
    assert!(parse_route_private_enabled(Some("1")).unwrap());
    assert!(parse_route_private_enabled(Some("true")).is_err());
    assert!(route_private_scope(1, true));
    assert!(route_private_scope(8, true));
    assert!(!route_private_scope(9, true));
    assert!(!route_private_scope(8, false));
    assert_eq!(
        Glm53Exl3MoePlan::new(1).unwrap().route_private_f32_bytes,
        8 * 4_096 * 4
    );
    assert_eq!(
        Glm53Exl3MoePlan::new(8).unwrap().route_private_f32_bytes,
        8 * 8 * 4_096 * 4
    );
}

#[test]
fn route_private_source_contract_is_non_atomic_and_slot_ordered() {
    const PRIVATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_private.cu"
    ));
    const ROUTE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_route.cu"
    ));
    assert!(PRIVATE.contains("atlas_glm53_exl3_moe_private_k2_n256_cb2"));
    assert!(PRIVATE.contains("source_pair / num_experts_per_tok"));
    assert!(PRIVATE.contains("output_state + source_pair * hidden_dim"));
    assert!(!PRIVATE.contains("atomicAdd(output_ptr"));
    assert!(ROUTE.contains("atlas_glm53_exl3_pack_routes_private"));
    assert!(ROUTE.contains("token_sorted[destination] = static_cast<int64_t>(source);"));
    assert!(ROUTE.contains("atlas_glm53_exl3_combine_private_shared"));
    assert!(ROUTE.contains("for (uint32_t slot = 0; slot < 8; ++slot)"));
    assert!(ROUTE.contains("sum += route_private[pair * 4096 + column];"));
}

#[test]
fn direct_combine_selector_is_strict_and_dependency_bound() {
    let private_prefill = Glm53Exl3RoutePolicy::parse(
        Some(std::ffi::OsStr::new("1")),
        Some(std::ffi::OsStr::new("1")),
    )
    .unwrap();
    let scalar_private = Glm53Exl3RoutePolicy::parse(
        Some(std::ffi::OsStr::new("1")),
        Some(std::ffi::OsStr::new("0")),
    )
    .unwrap();
    let value = |value: &'static str| Some(std::ffi::OsStr::new(value));
    assert!(!parse_direct_combine_enabled(None, private_prefill, true).unwrap());
    assert!(!parse_direct_combine_enabled(value("0"), private_prefill, true).unwrap());
    assert!(parse_direct_combine_enabled(value("1"), private_prefill, true).unwrap());
    assert!(parse_direct_combine_enabled(value("true"), private_prefill, true).is_err());
    assert!(parse_direct_combine_enabled(value("1"), scalar_private, true).is_err());
    assert!(parse_direct_combine_enabled(value("1"), private_prefill, false).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let non_utf8 = std::ffi::OsStr::from_bytes(&[0xff]);
        assert!(parse_direct_combine_enabled(Some(non_utf8), private_prefill, true).is_err());
    }
}

#[test]
fn direct_combine_scope_is_large_prefill_only() {
    assert!(!direct_combine_scope(1, true));
    assert!(!direct_combine_scope(1023, true));
    assert!(direct_combine_scope(1024, true));
    assert!(direct_combine_scope(2048, true));
    assert!(!direct_combine_scope(2048, false));
}

#[test]
fn direct_combine_source_preserves_inverse_and_slot_order() {
    assert!(ROUTE_CUDA.contains("atlas_glm53_exl3_pack_routes_private_inverse"));
    assert!(ROUTE_CUDA.contains("source_to_sorted[source] = destination;"));
    assert!(STAGED_PRIVATE_CUDA.contains("atlas_glm53_exl3_staged_combine_private"));
    assert!(STAGED_PRIVATE_CUDA.contains("for (uint32_t slot = 0; slot < 8; ++slot)"));
    assert!(STAGED_PRIVATE_CUDA.contains("const uint32_t pair = source_to_sorted[source_pair];"));
    assert!(STAGED_PRIVATE_CUDA.contains("sum0 += value0;"));
    assert!(STAGED_PRIVATE_CUDA.contains("sum1 += value1;"));
    assert!(STAGED_PRIVATE_CUDA.contains("sum2 += value2;"));
    assert!(STAGED_PRIVATE_CUDA.contains("sum3 += value3;"));
    assert!(!STAGED_PRIVATE_CUDA.contains("atomicAdd(output_state"));
}

#[test]
fn large_m_autotune_is_exact_and_default_preserving() {
    assert_eq!(
        parse_large_m_config(None).unwrap(),
        LargeMConfig {
            override_group: None,
        }
    );
    for (group, concurrency) in [("8", 6), ("12", 4), ("16", 3), ("24", 2)] {
        let config = parse_large_m_config(Some(group)).unwrap();
        let (group_width, actual_concurrency) = config.override_group.unwrap();
        assert_eq!(group_width * actual_concurrency, 48);
        assert_eq!(actual_concurrency, concurrency);
        assert!(actual_concurrency <= CONCURRENCY);
    }
    assert!(parse_large_m_config(Some("32")).is_err());
    assert!(MOE_SYMBOL.contains("Li256E"));
    let automatic = parse_large_m_config(None).unwrap();
    assert_eq!(large_m_schedule(9, automatic), (12, 4));
    assert_eq!(large_m_schedule(1023, automatic), (12, 4));
    assert_eq!(large_m_schedule(1024, automatic), (16, 3));
    assert_eq!(large_m_schedule(MAX_ROWS, automatic), (16, 3));
}

#[test]
fn staged_path_defaults_to_the_qualified_complete_tile_variant() {
    assert!(parse_staged_enabled(None).unwrap());
    assert!(parse_staged_enabled(Some("1")).unwrap());
    assert!(!parse_staged_enabled(Some("0")).unwrap());
    assert!(parse_staged_enabled(Some("true")).is_err());
    assert_eq!(STAGED_BLOCK_THREADS, 256);
    assert_eq!(STAGED_BASE_MODULE, "glm53_exl3_moe_staged_k16");
    assert_eq!(STAGED_GEMM_MODULE, "glm53_exl3_moe_staged_n256_f1");
    assert_eq!(STAGED_TILE_N, 256);
    assert_eq!(STAGED_SHARED_MEMORY_BYTES, 20_992);
    assert_eq!(STAGED_SCATTER_SHARED_MEMORY_BYTES, 4_096);
    assert_eq!(moe_kernel_launches(1, true, true), 2);
    assert_eq!(moe_kernel_launches(8, true, true), 2);
    assert_eq!(moe_kernel_launches(9, true, true), 2);
    assert_eq!(moe_kernel_launches(1023, true, false), 2);
    assert_eq!(moe_kernel_launches(1024, false, false), 2);
    assert_eq!(moe_kernel_launches(1024, true, false), 7);
    assert!(STAGED_CUDA.contains("__launch_bounds__(256, ATLAS_GLM53_STAGED_MIN_BLOCKS)"));
    assert!(STAGED_CUDA.contains("intermediate_dim / ATLAS_GLM53_STAGED_TILE_N"));
    assert!(STAGED_CUDA.contains("4096 / ATLAS_GLM53_STAGED_TILE_N"));
    assert!(STAGED_N256_F1_CUDA.contains("ATLAS_GLM53_STAGED_FRAG_STAGES 1"));
    assert!(STAGED_N256_F1_CUDA.contains("ATLAS_GLM53_STAGED_MIN_BLOCKS 3"));
    assert!(
        STAGED_CUDA
            .contains("#define barrier_acquire(lock, lock_i) ((void)(lock), (void)(lock_i))")
    );
    assert!(STAGED_CUDA.contains("atlas_glm53_exl3_staged_scatter"));
}

#[test]
fn route_packer_is_stable_tiled_through_m2048() {
    assert!(ROUTE_CUDA.contains("__shared__ uint32_t tile_experts[320]"));
    assert!(ROUTE_CUDA.contains("for (uint32_t tile = 0; tile < pairs; tile += blockDim.x)"));
    assert!(ROUTE_CUDA.contains("for (uint32_t prior = 0; prior < thread; ++prior)"));
    assert!(ROUTE_CUDA.contains("expert_offsets[selected] + local_rank"));
    assert!(ROUTE_CUDA.contains("expert_offsets[thread] += tile_count"));
    assert!(!ROUTE_CUDA.contains("for (uint32_t source = 0; source < pairs; ++source)"));
}

fn stable_reference(ids: &[usize], experts: usize) -> Vec<usize> {
    (0..experts)
        .flat_map(|expert| {
            ids.iter()
                .enumerate()
                .filter_map(move |(source, &selected)| (selected == expert).then_some(source))
        })
        .collect()
}

fn stable_tiled(ids: &[usize], experts: usize, tile_width: usize) -> Vec<usize> {
    let mut counts = vec![0usize; experts];
    for &selected in ids {
        counts[selected] += 1;
    }
    let mut offsets = vec![0usize; experts];
    for expert in 1..experts {
        offsets[expert] = offsets[expert - 1] + counts[expert - 1];
    }
    let mut output = vec![usize::MAX; ids.len()];
    for tile in (0..ids.len()).step_by(tile_width) {
        let end = (tile + tile_width).min(ids.len());
        let chunk = &ids[tile..end];
        for (lane, &selected) in chunk.iter().enumerate() {
            let local_rank = chunk[..lane]
                .iter()
                .filter(|&&prior| prior == selected)
                .count();
            output[offsets[selected] + local_rank] = tile + lane;
        }
        for &selected in chunk {
            offsets[selected] += 1;
        }
    }
    output
}

#[test]
fn tiled_route_oracle_preserves_expert_major_source_order() {
    for pairs in [64, 2_048 * 8] {
        let ids: Vec<usize> = (0..pairs)
            .map(|source| (source * 37 + source / 11 + 5) % EXPERTS as usize)
            .collect();
        assert_eq!(
            stable_tiled(&ids, EXPERTS as usize, 320),
            stable_reference(&ids, EXPERTS as usize)
        );
    }
}
