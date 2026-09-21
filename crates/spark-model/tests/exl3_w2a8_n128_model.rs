// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated EXL3 W2A8 N128 prefill experiment.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const COMPONENT_NAME: &str = "exl3_w2a8_grouped_prefill_n128.cu";
const KERNEL_BUILD: &str = include_str!("../../atlas-kernels/build.rs");
const MODEL_REGISTRY: &str = include_str!("../../../kernels/gb10/deepseek-v4-flash/MODEL.toml");
const EXL3_STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const EXL3_DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");
const PROBE: &str =
    include_str!("../../../kernels/gb10/experiments/exl3_w2a8_grouped_prefill_probe.cu");
const PROBE_BUILD: &str = include_str!("../../../scripts/check-exl3-prefill-w2a8-probe-build.sh");

fn component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/experiments")
        .join(COMPONENT_NAME)
}

fn component() -> String {
    fs::read_to_string(component_path()).expect("isolated N128 W2A8 component must exist")
}

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

fn mma_tile_coord(lane: usize, sequence: usize) -> (usize, usize) {
    let n = 8 * (sequence / 4) + lane / 4;
    let k = 2 * (lane % 4) + sequence % 2 + 8 * ((sequence % 4) / 2);
    (k, n)
}

fn repacked_b_coords(
    group: usize,
    n_half: usize,
    tid: usize,
    k_half: usize,
) -> [(usize, usize); 4] {
    let source_tid = 2 * (tid % 2);
    let source_sequence = 4 * n_half + 2 * (tid / 2);
    let coord = |source_tid, sequence| {
        let (k, n) = mma_tile_coord(4 * group + source_tid, sequence);
        (k_half + k, n)
    };
    [
        coord(source_tid, source_sequence),
        coord(source_tid, source_sequence + 1),
        coord(source_tid + 1, source_sequence),
        coord(source_tid + 1, source_sequence + 1),
    ]
}

fn assert_tree_excludes(path: &Path) {
    for entry in fs::read_dir(path).expect("production tree must exist") {
        let path = entry.expect("production entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|value| value.to_str()) == Some("experiments") {
                continue;
            }
            assert_tree_excludes(&path);
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(
                !source.contains("exl3_w2a8_grouped_prefill_n128"),
                "{}",
                path.display()
            );
            assert!(
                !source.contains("ATLAS_EXL3_PREFILL_W2A8_N128"),
                "{}",
                path.display()
            );
        }
    }
}

#[test]
fn component_is_exact_n128_k2_and_fail_closed() {
    let source = component();
    let flat = compact(&source);
    assert!(source.contains("#define W2A8_N_TILE 128"));
    assert!(source.contains("#define W2A8_M_TILE 64"));
    assert!(source.contains("#define W2A8_SCALE_GROUP 128"));
    assert!(source.contains("#define W2A8_K_STAGE 64"));
    assert!(source.contains("#define W2A8_K_PAIR 32"));
    assert!(source.contains("__launch_bounds__(256)"));
    assert!(flat.contains("blockDim.x!=256||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.y!=1||gridDim.z!=1"));
    assert!(flat.contains("N!=W2A8_FIXED_N||K!=W2A8_FIXED_K"));
    assert!(flat.contains("bits!=2||persistent_mode!=1"));
    assert!(flat.contains("gridDim.x!=(unsignedlonglong)num_experts*n_tiles"));
    assert!(source.contains("W2A8_FIXED_N == 2048 && W2A8_FIXED_K == 4096"));
    assert!(source.contains("W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048"));
}

#[test]
fn eight_warps_own_every_n128_output_once() {
    let mut owners = HashSet::new();
    for warp in 0..8 {
        for lane in 0..32 {
            let group = lane / 4;
            let tid = lane % 4;
            for mt in 0..4 {
                for nt in 0..2 {
                    let col = warp * 16 + nt * 8 + tid * 2;
                    let row0 = mt * 16 + group;
                    let row1 = row0 + 8;
                    for row in [row0, row1] {
                        assert!(owners.insert((row, col)));
                        assert!(owners.insert((row, col + 1)));
                    }
                }
            }
        }
    }
    assert_eq!(owners.len(), 64 * 128);
    assert!(owners.iter().all(|&(m, n)| m < 64 && n < 128));
}

#[test]
fn eight_warps_repack_k32_by_n128_fragments_bijectively() {
    let mut coords = HashSet::new();
    for warp in 0..8 {
        for group in 0..8 {
            for n_half in 0..2 {
                for tid in 0..4 {
                    for k_half in [0, 16] {
                        for (k, local_n) in repacked_b_coords(group, n_half, tid, k_half) {
                            assert!(coords.insert((k, warp * 16 + local_n)));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(coords.len(), 32 * 128);
    assert!(coords.iter().all(|&(k, n)| k < 32 && n < 128));
}

#[test]
fn grid_and_logical_read_model_halves_n64_work() {
    const TOKENS: u64 = 2_410;
    const TOP_K: u64 = 6;
    const EXPERTS: u64 = 256;
    const LAYERS: u64 = 43;
    let rows = TOKENS * TOP_K;
    let per_layer = |tile: u64| 2 * rows * 4096 * (2048 / tile) + rows * 2048 * (4096 / tile);
    assert_eq!(per_layer(64), 5_685_903_360);
    assert_eq!(per_layer(128), 2_842_951_680);
    assert_eq!((per_layer(64) - per_layer(128)) * LAYERS, 122_246_922_240);

    let ctas = |tile: u64| 2 * EXPERTS * (2048 / tile) + EXPERTS * (4096 / tile);
    assert_eq!(ctas(64), 32_768);
    assert_eq!(ctas(128), 16_384);
    assert_eq!((ctas(64) - ctas(128)) * LAYERS, 704_512);
}

#[test]
fn native_k32_fragments_and_k128_scale_fold_are_retained() {
    let source = component();
    let flat = compact(&source);
    assert!(source.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"));
    assert!(source.contains("float outer[4][2][4] = {}"));
    assert!(source.contains("float inner[4][2][4] = {}"));
    assert!(flat.contains("k_block+=W2A8_SCALE_GROUP"));
    assert!(flat.contains("k_stage+=W2A8_K_STAGE"));
    assert!(source.contains("const float factor0 = scale0 * W2A8_INV_WEIGHT_SCALE"));
    assert!(source.contains("const float factor1 = scale1 * W2A8_INV_WEIGHT_SCALE"));
    assert!(source.contains("outer[mt][nt][0] += inner[mt][nt][0] * factor0"));
    assert!(source.contains("out[0] = __float2bfloat16"));
    assert!(source.contains("out[1] = __float2bfloat16"));
}

#[test]
fn n128_trellis_and_activation_staging_cover_each_stage_once() {
    let source = component();
    let flat = compact(&source);
    assert!(source.contains("uint4 smem_A[W2A8_M_TILE][5]"));
    assert!(source.contains("uint4 smem_T[4][32]"));
    assert!(flat.contains("vector<W2A8_M_TILE*4;vector+=blockDim.x"));
    assert!(flat.contains("load<128;load+=blockDim.x"));
    assert!(flat.contains("constunsignedintk_tile=load>>5"));
    assert!(flat.contains("constunsignedintword=load&31"));
    assert!(flat.contains("(constunsignedint*)smem_T[pair*2]+warp*16"));
    assert!(flat.contains("constunsignedintcol=n_base+warp*16+nt*8+tid*2"));

    let mut staged = HashSet::new();
    for load in 0..128 {
        let k_tile = load >> 5;
        let word = load & 31;
        assert!(k_tile < 4 && word < 32);
        assert!(staged.insert((k_tile, word)));
    }
    assert_eq!(staged.len(), 4 * 32);

    let staged_u32: HashSet<_> = staged
        .iter()
        .flat_map(|&(k_tile, uint4_word)| (0..4).map(move |lane| (k_tile, uint4_word * 4 + lane)))
        .collect();
    let consumed: HashSet<_> = (0..4)
        .flat_map(|k_tile| {
            (0..8).flat_map(move |warp| (0..16).map(move |word| (k_tile, warp * 16 + word)))
        })
        .collect();
    assert_eq!(consumed, staged_u32);
}

#[test]
fn component_has_no_production_or_fallback_reachability() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert!(!KERNEL_BUILD.contains("exl3_w2a8_grouped_prefill_n128"));
    assert!(!MODEL_REGISTRY.contains("exl3_w2a8_grouped_prefill_n128"));
    assert!(!EXL3_STATE.contains("exl3_w2a8_grouped_prefill_n128"));
    assert!(!EXL3_DISPATCH.contains("exl3_w2a8_grouped_prefill_n128"));
    assert_tree_excludes(&root.join("kernels/gb10"));
    assert_tree_excludes(&root.join("crates/spark-model/src"));
    let source = component();
    assert!(!source.contains("getenv("));
    assert!(!source.contains("fallback"));
}

#[test]
fn standalone_promotion_probe_builds_all_three_w2a8_strip_widths() {
    assert!(PROBE.contains("#ifndef W2A8_PROBE_N_TILE"));
    assert_eq!(PROBE.matches("W2A8_PROBE_N_TILE !=").count(), 3);
    assert!(PROBE.contains("W2A8_PROBE_N_TILE must be 64, 128, or 256"));
    assert_eq!(PROBE.matches("#if W2A8_PROBE_N_TILE == 256").count(), 7);
    assert_eq!(PROBE.matches("#elif W2A8_PROBE_N_TILE == 128").count(), 4);
    assert!(PROBE.contains("exl3_w2a8_grouped_prefill_n128.cu"));
    assert!(PROBE.contains("exl3_w2a8_grouped_prefill_n256.cu"));
    assert!(PROBE.contains("W2A8_PROBE_THREADS"));
    assert!(PROBE.contains("#define W2A8_PROBE_THREADS 512"));
    for symbol in [
        "exl3_w2a8_grouped_prefill_n256_gu",
        "exl3_w2a8_grouped_prefill_n256_down",
    ] {
        assert!(PROBE.contains(symbol), "missing N256 probe symbol {symbol}");
    }
    assert!(PROBE.contains("W2A8_FIXED_N / W2A8_PROBE_N_TILE"));
    for canary in [
        "wrong-block-x",
        "wrong-block-y",
        "wrong-block-z",
        "wrong-grid-x",
        "wrong-grid-y",
        "wrong-grid-z",
        "wrong-runtime-N",
        "wrong-runtime-K",
        "wrong-runtime-bits",
        "wrong-runtime-persistent",
    ] {
        assert!(PROBE.contains(&format!("invalid_canary(\"{canary}\"")));
    }
    assert!(PROBE.contains("run_case({0, 0, 63, 63, 129}, \"multi-with-empty\")"));
    assert!(PROBE.contains("run_case({0, 63, 129, 129, 129}, \"multi-trailing-empty\")"));
    assert!(PROBE.contains("W2A8_PROBE_DUMP"));
    assert!(PROBE.contains("std::fopen(dump_path, \"wbx\")"));
    assert!(PROBE.contains("std::fwrite"));

    assert!(PROBE_BUILD.contains("exl3_w2a8_grouped_prefill_n128.cu"));
    assert!(PROBE_BUILD.contains("exl3_w2a8_grouped_prefill_n256.cu"));
    assert!(PROBE_BUILD.contains("receipt_format=atlas-w2a8-v3"));
    assert_eq!(
        PROBE_BUILD.matches("for n_tile in 64 128 256").count(),
        2,
        "compile and individual-runner loops must both cover three widths"
    );
    assert!(PROBE_BUILD.contains("-DW2A8_PROBE_N_TILE=\"$n_tile\""));
    assert!(PROBE_BUILD.contains("run-${kind}-n${n_tile}-probe.sh"));
    assert!(PROBE_BUILD.contains("compile_command_${kind}_n${n_tile}"));
    assert!(PROBE_BUILD.contains("run-${kind}-pair-probe.sh"));
    assert!(PROBE_BUILD.contains("run-${kind}-three-width-probe.sh"));
    assert!(PROBE_BUILD.contains("cross-width W2A8 hashes"));
    assert!(PROBE_BUILD.contains("three-width W2A8 hashes"));
    assert!(PROBE_BUILD.contains("w2a8_hash="));
    assert!(PROBE_BUILD.contains("hash_count -ne 9"));
    assert!(PROBE_BUILD.contains("n128_hash_count -ne 9"));
    assert!(PROBE_BUILD.contains("n256_hash_count -ne 9"));
    assert!(PROBE_BUILD.contains("$n64_hashes != \"$n256_hashes\""));
    assert!(PROBE_BUILD.contains("$n64_input != \"$n256_input\""));
    assert!(PROBE_BUILD.contains("W2A8_PROBE_DUMP="));
    assert!(PROBE_BUILD.contains("cmp -s"));
    for dump_pair in [
        "cmp -s \"$width_tmp/n64.bin\" \"$width_tmp/n128.bin\"",
        "cmp -s \"$width_tmp/n64.bin\" \"$width_tmp/n256.bin\"",
    ] {
        assert!(
            PROBE_BUILD.contains(dump_pair),
            "three-width runner omits raw comparison `{dump_pair}`"
        );
    }
    assert!(PROBE_BUILD.contains("expect_dump_rejection"));
    assert!(PROBE_BUILD.contains("exclusive W2A8 probe dump"));
    assert!(PROBE_BUILD.contains("dump_before"));
}
