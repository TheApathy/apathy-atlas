// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::cli::{Cli, Command};
use clap::Parser;
use spark_model::model::ssm_pool_geometry::{
    SsmPoolGeometryInput, checked_ssm_pool_geometry, checked_ssm_speculative_geometry,
};

const EXPECTED_SNAPSHOT_BYTES: usize = 696;

fn serve_args() -> cli::ServeArgs {
    let cli = Cli::try_parse_from(["spark", "serve", "nvidia/model"]).unwrap();
    match cli.command {
        Command::Serve(args) => args,
    }
}

fn expect_capacity_error(shape: (usize, usize, usize, usize, usize, usize), expected: &str) {
    let (max_batch, layers, h_bytes, conv_bytes, cache_slots, ring_slots) = shape;
    let error = checked_ssm_preflight_capacity(
        0,
        max_batch,
        layers,
        h_bytes,
        conv_bytes,
        cache_slots,
        ring_slots,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains(expected),
        "wrong error for {shape:?}: {error:#}"
    );
}

#[test]
fn representative_preflight_geometry_matches_the_pool_fixture() {
    let pool = checked_ssm_pool_geometry(SsmPoolGeometryInput {
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
    })
    .unwrap();
    let capacity = checked_ssm_preflight_capacity(pool.total_bytes, 8, 2, 8, 4, 5, 3).unwrap();
    assert_eq!(capacity.ssm_pool_bytes, 216);
    assert_eq!(capacity.ssm_snapshot_bytes, EXPECTED_SNAPSHOT_BYTES);

    let zero = checked_ssm_preflight_capacity(
        0,
        usize::MAX,
        0,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    assert_eq!(zero.ssm_pool_bytes, 0);
    assert_eq!(zero.ssm_snapshot_bytes, 0);
}

#[test]
fn server_and_constructor_share_flat_tree_and_lazy_geometry() {
    let flat = checked_ssm_speculative_geometry(true, true, 15, None).unwrap();
    let wide = checked_ssm_speculative_geometry(true, true, 15, Some(31)).unwrap();
    assert_eq!(flat.num_intermediates, 17);
    assert_eq!(wide.num_intermediates, 32);
}

#[test]
fn every_snapshot_checked_boundary_is_contextual() {
    expect_capacity_error((usize::MAX, 1, 1, 1, 0, 2), "decode region slots");
    expect_capacity_error((1, 1, 1, 1, usize::MAX, 1), "snapshot slots");
    expect_capacity_error((0, 2, usize::MAX / 2 + 1, 1, 1, 0), "snapshot layer bytes");
    expect_capacity_error((0, 1, usize::MAX / 2 + 1, 1, 2, 0), "snapshot total bytes");
    expect_capacity_error((1, 1, 0, 1, 1, 0), "snapshot h bytes must be positive");
    expect_capacity_error((1, 1, 1, 0, 1, 0), "snapshot conv bytes must be positive");
}

#[test]
fn final_reserve_additions_fail_closed() {
    assert_eq!(speculative_cuda_headroom(None, 4096), 512 << 20);
    assert_eq!(speculative_cuda_headroom(Some(15), 4096), 4 << 30);

    for (parts, expected) in [
        ((usize::MAX, 1, 0, 0, 0), "SSM pool + snapshot"),
        ((usize::MAX - 1, 0, 2, 0, 0), "+ GDN"),
        ((usize::MAX - 1, 0, 0, 2, 0), "+ CUDA headroom"),
        ((usize::MAX - 1, 0, 0, 0, 2), "+ buffer arena"),
    ] {
        let error =
            checked_reserve_totals(parts.0, parts.1, parts.2, parts.3, parts.4).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "wrong error: {error:#}"
        );
    }
}

#[test]
fn public_preflight_rejects_overflow_before_the_memory_comparison() {
    let mut args = serve_args();
    args.max_batch_size = usize::MAX;
    let config = ModelConfig::qwen3_next_80b_nvfp4();

    let error = match preflight_reserve(&args, &config, usize::MAX) {
        Ok(_) => panic!("overflowing public preflight unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("dummy-inclusive slots"));
}

fn mamba_config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
    config.mamba_num_heads = 1;
    config.mamba_head_dim = 1;
    config.ssm_state_size = 1;
    config.n_groups = 0;
    config.linear_conv_kernel_dim = 1;
    config
}

fn expect_public_projection_error(config: &ModelConfig, expected: &str) {
    let error = match preflight_reserve(&serve_args(), config, usize::MAX) {
        Ok(_) => panic!("overflowing SSM projection unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(expected),
        "wrong error: {error:#}"
    );
}

#[test]
fn raw_mamba_product_add_and_fp32_boundaries_fail_closed() {
    macro_rules! reject {
        ($expected:literal, $($field:ident = $value:expr),+ $(,)?) => {{
            let mut config = mamba_config();
            $(config.$field = $value;)+
            expect_public_projection_error(&config, $expected);
        }};
    }
    reject!(
        "Mamba inner width",
        mamba_num_heads = usize::MAX,
        mamba_head_dim = 2
    );
    reject!(
        "Mamba h state elements",
        mamba_num_heads = usize::MAX / 2 + 1,
        ssm_state_size = 2
    );
    reject!("Mamba h FP32 bytes", mamba_num_heads = usize::MAX / 4 + 1);
    reject!(
        "Mamba grouped state width",
        n_groups = usize::MAX,
        ssm_state_size = 2
    );
    reject!(
        "Mamba doubled grouped state width",
        n_groups = usize::MAX / 2 + 1
    );
    reject!(
        "Mamba conv input width",
        mamba_num_heads = 2,
        n_groups = usize::MAX / 2
    );
    reject!(
        "Mamba conv kernel elements",
        mamba_num_heads = 2,
        linear_conv_kernel_dim = usize::MAX
    );
    reject!(
        "Mamba conv FP32 bytes",
        linear_conv_kernel_dim = usize::MAX / 4 + 1
    );
}
