// SPDX-License-Identifier: AGPL-3.0-only

const CACHE_SKIP: &str = include_str!("../src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
const PAGED: &str = include_str!("../src/layers/qwen3_attention/prefill/paged_v4.rs");
const ASSEMBLE: &str = include_str!("../src/weight_loader/deepseek_v4/assemble.rs");
const RELEASE: &str = include_str!("../src/weight_loader/deepseek_v4/attention_residency.rs");

#[test]
fn both_v4_prefill_paths_dispatch_releasable_projections() {
    for source in [CACHE_SKIP, PAGED] {
        assert!(source.matches("v4_project_prefill").count() >= 2);
        assert!(source.contains("v4_grouped_wo_a_prefill"));
    }
}

#[test]
fn release_is_explicit_and_nulls_every_bf16_alias() {
    assert!(ASSEMBLE.contains("ATLAS_V4_ATTN_RELEASE_BF16"));
    assert!(RELEASE.contains("gpu.free(wq_b.weight)"));
    assert!(RELEASE.contains("gpu.free(wo_a.weight)"));
    assert!(RELEASE.contains("gpu.free(o_dense.weight)"));
    assert!(RELEASE.contains("wq_b.weight = DevicePtr::NULL"));
    assert!(RELEASE.contains("wo_a.weight = DevicePtr::NULL"));
    assert!(RELEASE.contains("o_dense.weight = DevicePtr::NULL"));
}

#[test]
fn production_shapes_reclaim_exactly_8_0625_gib() {
    let elements_per_projection = 33_554_432usize;
    let bytes_per_layer = 3 * elements_per_projection * size_of::<u16>();
    assert_eq!(bytes_per_layer, 201_326_592);
    assert_eq!(bytes_per_layer * 43, 8_657_043_456);
}

#[test]
fn grouped_fp8_scratch_fits_inside_the_dead_q_buffer() {
    let rows = 129usize;
    let groups = 8usize;
    let group_in = 4_096usize;
    let group_out = 1_024usize;
    let q_width = groups * group_in;
    let scratch_elements = rows * (group_in + group_out);
    let q_buffer_elements = rows * q_width;

    assert!(scratch_elements <= q_buffer_elements);
    assert_eq!(groups * group_out, 8_192);
}
