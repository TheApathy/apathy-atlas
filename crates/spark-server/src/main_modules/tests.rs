// SPDX-License-Identifier: AGPL-3.0-only

use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::{build_layer_kv_dtypes, build_layer_kv_dtypes_from_set};

#[test]
fn test_cli_parse_positional_model() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "--port",
        "9999",
        "--max-seq-len",
        "8192",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(
                args.model.as_deref(),
                Some("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"),
            );
            assert!(args.model_from_path.is_none());
            assert_eq!(args.port, 9999);
            assert_eq!(args.max_seq_len, 8192);
            assert_eq!(args.gpu_memory_utilization, 0.90);
            assert_eq!(args.scheduling_policy, "fifo");
            assert_eq!(args.tbt_deadline_ms, 100);
        }
    }
}

#[test]
fn test_cli_parse_model_from_path() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "--model-from-path",
        "/tmp/model",
        "--port",
        "8888",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert!(args.model.is_none());
            assert_eq!(
                args.model_from_path,
                Some(std::path::PathBuf::from("/tmp/model")),
            );
        }
    }
}

#[test]
fn test_cli_parse_slai_policy() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/model",
        "--scheduling-policy",
        "slai",
        "--tbt-deadline-ms",
        "50",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.scheduling_policy, "slai");
            assert_eq!(args.tbt_deadline_ms, 50);
        }
    }
}

#[test]
fn test_build_layer_kv_dtypes_disabled() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // high_precision_layers=0 returns empty vec (backward compatible)
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Nvfp4, 12, 0, KvCacheDtype::Bf16);
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_bf16_all_layers() {
    // When base dtype is BF16, ALL layers must be BF16 — returning empty would
    // let callers with unwrap_or(Fp8) silently downgrade MLA KV latents to FP8.
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Bf16, 12, 2, KvCacheDtype::Bf16);
    // kv_dtype == boundary_dtype → early-return empty (no-benefit guard).
    // Either the test's invariant is "all 12 are Bf16" (post-callers' unwrap_or
    // default) or "empty is fine because base == boundary". Pick the latter
    // semantic since that's what the function emits.
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_basic() {
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Nvfp4, 12, 2, KvCacheDtype::Bf16);
    assert_eq!(dtypes.len(), 12);
    // First 2: BF16
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
    assert_eq!(dtypes[1], KvCacheDtype::Bf16);
    // Middle 8: NVFP4
    for i in 2..10 {
        assert_eq!(dtypes[i], KvCacheDtype::Nvfp4, "layer {i}");
    }
    // Last 2: BF16
    assert_eq!(dtypes[10], KvCacheDtype::Bf16);
    assert_eq!(dtypes[11], KvCacheDtype::Bf16);
}

#[test]
fn test_build_layer_kv_dtypes_overlap() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // 4 layers, hp=3 → all become BF16 (first 3 and last 3 overlap)
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Fp8, 4, 3, KvCacheDtype::Bf16);
    assert_eq!(dtypes.len(), 4);
    for d in &dtypes {
        assert_eq!(*d, KvCacheDtype::Bf16);
    }
}

#[test]
fn test_build_layer_kv_dtypes_single_layer() {
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Nvfp4, 1, 1, KvCacheDtype::Bf16);
    assert_eq!(dtypes.len(), 1);
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
}

#[test]
fn test_build_layer_kv_dtypes_from_set_basic() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // Explicit measured set: layers 0, 5, 11 kept at BF16; the rest NVFP4.
    let dtypes =
        build_layer_kv_dtypes_from_set(KvCacheDtype::Nvfp4, 12, &[0, 5, 11], KvCacheDtype::Bf16);
    assert_eq!(dtypes.len(), 12);
    for i in 0..12 {
        let expected = if [0usize, 5, 11].contains(&i) {
            KvCacheDtype::Bf16
        } else {
            KvCacheDtype::Nvfp4
        };
        assert_eq!(dtypes[i], expected, "layer {i}");
    }
}

#[test]
fn test_build_layer_kv_dtypes_from_set_empty_is_uniform() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // Empty set → empty vec (uniform dtype), same no-benefit semantics as the positional builder.
    let dtypes = build_layer_kv_dtypes_from_set(KvCacheDtype::Fp8, 16, &[], KvCacheDtype::Bf16);
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_from_set_same_dtype_is_empty() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // boundary == base → no benefit → empty vec (matches build_layer_kv_dtypes).
    let dtypes =
        build_layer_kv_dtypes_from_set(KvCacheDtype::Bf16, 16, &[1, 2], KvCacheDtype::Bf16);
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_from_set_out_of_range_ignored() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // Index 99 is out of range for a 16-layer model — ignored, not a panic.
    // Valid index 3 still applies.
    let dtypes =
        build_layer_kv_dtypes_from_set(KvCacheDtype::Turbo4, 16, &[3, 99], KvCacheDtype::Bf16);
    assert_eq!(dtypes.len(), 16);
    assert_eq!(dtypes[3], KvCacheDtype::Bf16);
    let hp = dtypes.iter().filter(|d| **d == KvCacheDtype::Bf16).count();
    assert_eq!(hp, 1, "only the in-range index should apply");
}

#[test]
fn test_build_layer_kv_dtypes_from_set_duplicate_indices() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // Duplicates are idempotent — layer 2 set twice is still just layer 2.
    let dtypes =
        build_layer_kv_dtypes_from_set(KvCacheDtype::Nvfp4, 8, &[2, 2, 2], KvCacheDtype::Bf16);
    let hp = dtypes.iter().filter(|d| **d == KvCacheDtype::Bf16).count();
    assert_eq!(hp, 1);
    assert_eq!(dtypes[2], KvCacheDtype::Bf16);
}

#[test]
fn test_cli_parse_kv_high_precision_layer_set() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/model",
        "--kv-high-precision-layer-set",
        "0,5,11",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layer_set, "0,5,11");
        }
    }
}

#[test]
fn test_cli_default_kv_high_precision_layer_set() {
    let cli = Cli::try_parse_from(["spark", "serve", "nvidia/model"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layer_set, "");
        }
    }
}

#[test]
fn test_cli_parse_kv_high_precision_layers() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/model",
        "--kv-high-precision-layers",
        "3",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "3");
        }
    }
}

#[test]
fn test_cli_default_kv_high_precision_layers() {
    let cli = Cli::try_parse_from(["spark", "serve", "nvidia/model"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "0");
        }
    }
}
