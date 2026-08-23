// SPDX-License-Identifier: AGPL-3.0-only

use super::super::ssm_pool_geometry::*;
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
fn input() -> SsmPoolGeometryInput {
    SsmPoolGeometryInput {
        max_slots: 8,
        num_ssm_layers: 2,
        h_bytes: 8,
        conv_bytes: 4,
        has_mtp: false,
        num_intermediates: 0,
        lazy_commit: false,
        num_key_heads: 0,
        key_head_dim: 0,
        num_value_heads: 0,
        value_head_dim: 0,
    }
}

fn expect_error(input: SsmPoolGeometryInput, expected: &str) {
    let error = checked_ssm_pool_geometry(input).unwrap_err();
    assert!(
        error.to_string().contains(expected),
        "wrong error for {input:?}: {error:#}"
    );
}
fn tiny_config(num_layers: usize, enabled_bytes: bool) -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.layer_types = if num_layers == 0 {
        vec![LayerType::FullAttention]
    } else {
        vec![LayerType::LinearAttention; num_layers]
    };
    config.linear_num_key_heads = usize::from(enabled_bytes);
    config.linear_key_head_dim = usize::from(enabled_bytes);
    config.linear_num_value_heads = usize::from(enabled_bytes);
    config.linear_value_head_dim = usize::from(enabled_bytes);
    config.linear_conv_kernel_dim = usize::from(enabled_bytes);
    config
}

#[test]
fn canonical_plain_geometry_is_dummy_inclusive_and_constructed_exactly() {
    let geometry = checked_ssm_pool_geometry(input()).unwrap();
    assert_eq!(geometry.total_slots, 9);
    assert_eq!(geometry.state_copies, 1);
    assert_eq!(geometry.total_bytes, 216);
    assert_eq!(geometry.kv_retain_bytes + geometry.gate_retain_bytes, 0);

    let gpu = MockGpuBackend::new();
    let config = tiny_config(2, true);
    let expected = checked_ssm_pool_geometry(
        SsmPoolGeometryInput::from_config(&config, 1, false, 0, false).unwrap(),
    )
    .unwrap();
    let pool = SsmStatePool::new(&config, 1, false, 0, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 4);
    assert_eq!(
        gpu.read_alloc(pool.h_state_pools[0]).unwrap().len(),
        expected.h_state_allocation_bytes
    );
    assert_eq!(
        gpu.read_alloc(pool.conv_state_pools[0]).unwrap().len(),
        expected.conv_state_allocation_bytes
    );
}

#[test]
fn flat_tree_and_lazy_shapes_use_the_exact_shared_formula() {
    let flat = checked_ssm_speculative_geometry(true, true, 15, None).unwrap();
    assert_eq!(flat.dflash_verify_width, 16);
    assert_eq!(flat.ddtree_capacity, 16);
    assert_eq!(flat.num_intermediates, 17);

    let wide = checked_ssm_speculative_geometry(true, true, 15, Some(31)).unwrap();
    assert_eq!(wide.num_intermediates, 32);

    let mut flat_input = input();
    flat_input.has_mtp = true;
    flat_input.num_intermediates = flat.num_intermediates;
    assert_eq!(
        checked_ssm_pool_geometry(flat_input).unwrap().total_bytes,
        4104
    );

    let mut wide_input = input();
    wide_input.has_mtp = true;
    wide_input.num_intermediates = wide.num_intermediates;
    assert_eq!(
        checked_ssm_pool_geometry(wide_input).unwrap().total_bytes,
        7344
    );

    let mut lazy = input();
    lazy.max_slots = 1;
    lazy.has_mtp = true;
    lazy.num_intermediates = 4;
    lazy.lazy_commit = true;
    lazy.num_key_heads = 1;
    lazy.key_head_dim = 2;
    lazy.num_value_heads = 1;
    lazy.value_head_dim = 2;
    let lazy = checked_ssm_pool_geometry(lazy).unwrap();
    assert_eq!(lazy.kv_retain_bytes, 48);
    assert_eq!(lazy.gate_retain_bytes, 32);
    assert_eq!(lazy.total_bytes, 608);
}

#[test]
fn disabled_and_enabled_zero_byte_shapes_are_distinct() {
    let mut disabled = input();
    disabled.max_slots = usize::MAX;
    disabled.num_ssm_layers = 0;
    disabled.h_bytes = 0;
    disabled.conv_bytes = 0;
    assert_eq!(
        checked_ssm_pool_geometry(disabled).unwrap(),
        SsmPoolGeometry::default()
    );

    let gpu = MockGpuBackend::new();
    let mut config = tiny_config(0, false);
    config.mamba_num_heads = usize::MAX;
    config.mamba_head_dim = usize::MAX;
    config.ssm_state_size = usize::MAX;
    let pool = SsmStatePool::new(&config, usize::MAX, true, usize::MAX, &gpu).unwrap();
    assert_eq!(pool.num_ssm_layers, 0);
    assert_eq!(gpu.alloc_count(), 0);

    let mut zero_h = input();
    zero_h.h_bytes = 0;
    expect_error(zero_h, "h state bytes must be positive");
    let mut zero_conv = input();
    zero_conv.conv_bytes = 0;
    expect_error(zero_conv, "conv state bytes must be positive");
    let mut zero_intermediates = input();
    zero_intermediates.has_mtp = true;
    expect_error(zero_intermediates, "intermediate count must be positive");

    let error = match SsmStatePool::new(&tiny_config(2, false), 1, false, 0, &gpu) {
        Ok(_) => panic!("enabled zero-byte state unexpectedly allocated"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("state bytes must be positive"));
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn state_and_speculative_overflow_boundaries_fail_closed() {
    let mut slots = input();
    slots.max_slots = usize::MAX;
    expect_error(slots, "dummy-inclusive slots");

    let mut sum = input();
    sum.h_bytes = usize::MAX;
    sum.conv_bytes = 1;
    expect_error(sum, "state bytes per copy");

    let mut copies = input();
    copies.has_mtp = true;
    copies.num_intermediates = usize::MAX;
    expect_error(copies, "state copies");

    let mut layer = input();
    layer.max_slots = 0;
    layer.num_ssm_layers = 2;
    layer.h_bytes = usize::MAX / 2;
    layer.conv_bytes = 2;
    expect_error(layer, "all SSM layers");

    let mut per_layer = input();
    per_layer.max_slots = 1;
    per_layer.num_ssm_layers = 1;
    per_layer.h_bytes = usize::MAX / 2;
    per_layer.conv_bytes = 1;
    expect_error(per_layer, "dummy-inclusive bytes per layer");

    assert!(
        checked_ssm_speculative_geometry(true, false, usize::MAX - 1, None)
            .unwrap_err()
            .to_string()
            .contains("drafts + 2")
    );
    assert!(
        checked_ssm_speculative_geometry(true, true, usize::MAX, None)
            .unwrap_err()
            .to_string()
            .contains("DFlash verify width")
    );
    assert!(
        checked_ssm_speculative_geometry(true, true, 32, Some(32))
            .unwrap_err()
            .to_string()
            .contains("kernel maximum")
    );
}

#[test]
fn every_lazy_retention_boundary_is_checked() {
    let lazy = |nk, kd, nv, vd, ni| SsmPoolGeometryInput {
        has_mtp: true,
        num_intermediates: ni,
        lazy_commit: true,
        num_key_heads: nk,
        key_head_dim: kd,
        num_value_heads: nv,
        value_head_dim: vd,
        ..input()
    };

    for (shape, expected) in [
        (lazy(2, usize::MAX, 1, 1, 1), "key width"),
        (lazy(1, usize::MAX / 2 + 1, 1, 1, 1), "doubled key width"),
        (lazy(1, 1, 2, usize::MAX, 1), "value width"),
        (lazy(1, usize::MAX / 2, 1, 2, 1), "retention conv width"),
        (
            lazy(0, 0, 1, usize::MAX / 2 + 1, 2),
            "KV retention elements",
        ),
        (lazy(0, 0, 1, usize::MAX / 2 + 1, 1), "KV retention bytes"),
        (lazy(1, 1, usize::MAX, 0, 1), "gate width"),
        (
            lazy(1, 1, usize::MAX / 4 + 1, 0, 2),
            "gate retention elements",
        ),
        (lazy(1, 1, usize::MAX / 8 + 1, 0, 1), "gate retention bytes"),
        (
            lazy(1, usize::MAX / 8, usize::MAX / 16 + 2, 0, 1),
            "retention bytes per slot",
        ),
    ] {
        expect_error(shape, expected);
    }
}

#[test]
fn raw_gdn_wrap_to_four_rejects_before_allocation() {
    let mut config = tiny_config(1, true);
    config.linear_num_key_heads = 0;
    config.linear_num_value_heads = usize::MAX / 4 + 2;
    let gpu = MockGpuBackend::new();
    let error = match SsmStatePool::new(&config, 1, false, 0, &gpu) {
        Ok(_) => panic!("wrapped GDN state geometry unexpectedly allocated"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("GDN h FP32 bytes"), "{error:#}");
    assert_eq!(gpu.alloc_count(), 0);
}
