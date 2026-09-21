// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the exact M32xN256 W2A8 fused gate/up producer.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n256.cu";
const WRAPPER: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n256.cu";
const SASS_GATE: &str = "scripts/check-exl3-prefill-w2a8-fused-gu-n256-sass.sh";
const PROBE: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n256_probe.cu";
const PROBE_BUILD: &str = "scripts/check-exl3-prefill-w2a8-fused-gu-n256-probe-build.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("required M32xN256 artifact {relative}: {error}"))
}

fn compact(text: &str) -> String {
    text.chars()
        .filter(|value| !value.is_whitespace())
        .collect()
}

fn marked<'a>(text: &'a str, label: &str) -> &'a str {
    let begin = format!("// BEGIN {label}");
    let end = format!("// END {label}");
    assert_eq!(text.matches(&begin).count(), 1, "missing/duplicate {begin}");
    assert_eq!(text.matches(&end).count(), 1, "missing/duplicate {end}");
    let start = text.find(&begin).unwrap() + begin.len();
    let finish = text[start..].find(&end).unwrap() + start;
    assert!(start < finish, "reversed {label} markers");
    &text[start..finish]
}

#[test]
fn exact_shape_and_geometry_are_locked() {
    let cuda = source(COMPONENT);
    let flat = compact(&cuda);

    for contract in [
        "#define W2F_M_TILE 32",
        "#define W2F_N_TILE 256",
        "#define W2F_THREADS 256",
        "#define W2F_N_STRIPS_PER_WARP 2",
        "#define W2F_TOTAL_ROWS 14460",
        "#define W2F_GATE_UP_N W2A8_FIXED_N",
        "#define W2F_GATE_UP_K W2A8_FIXED_K",
        "W2F_GATE_UP_N == 2048 && W2F_GATE_UP_K == 4096",
        "__launch_bounds__(W2F_THREADS)",
    ] {
        assert!(cuda.contains(contract), "missing `{contract}`");
    }
    for external in ["W2A8_FIXED_N", "W2A8_FIXED_K", "W2A8_KERNEL_NAME"] {
        assert!(cuda.contains(&format!("#ifndef {external}")));
        assert!(cuda.contains(&format!("#error \"{external} must be explicit\"")));
    }
    assert!(flat.contains("blockDim.x!=W2F_THREADS||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.x!=(unsignedlonglong)num_experts*n_tiles"));
    assert!(flat.contains("num_experts!=W2F_EXPERTS||total_rows!=W2F_TOTAL_ROWS"));
    assert!(flat.contains("total_rows>(unsignedint)INT_MAX"));
}

#[test]
fn eight_warps_cover_m32_n256_exactly_once() {
    let mut owners = HashSet::new();
    for warp in 0..8 {
        for lane in 0..32 {
            let group = lane / 4;
            let tid = lane % 4;
            for mt in 0..2 {
                for nt in 0..4 {
                    let column = warp * 32 + nt * 8 + tid * 2;
                    for row in [mt * 16 + group, mt * 16 + group + 8] {
                        assert!(owners.insert((row, column)));
                        assert!(owners.insert((row, column + 1)));
                    }
                }
            }
        }
    }
    assert_eq!(owners.len(), 32 * 256);
    assert!(owners.iter().all(|&(row, col)| row < 32 && col < 256));

    let cuda = compact(&source(COMPONENT));
    assert!(cuda.contains("constunsignedintcol=warp*32+nt*8+tid*2"));
    assert!(cuda.contains("floatouter[2][4][4]={}"));
}

#[test]
fn trellis_and_activation_staging_are_bijective() {
    let mut activation = HashSet::new();
    let mut trellis = HashSet::new();
    for thread in 0..256 {
        for vector in (thread..32 * 4).step_by(256) {
            assert!(activation.insert((vector / 4, vector % 4)));
        }
        for load in (thread..4 * 64).step_by(256) {
            assert!(trellis.insert((load / 64, load % 64)));
        }
    }
    assert_eq!(activation.len(), 32 * 4);
    assert_eq!(trellis.len(), 4 * 64);

    let cuda = source(COMPONENT);
    assert!(cuda.contains("uint4 activation[W2F_M_TILE][5]"));
    assert!(cuda.contains("uint4 trellis[4][64]"));
    assert!(cuda.contains("vector < W2F_M_TILE * 4"));
    assert!(cuda.contains("load < 256"));
}

#[test]
fn k_order_and_numeric_seams_match_n128() {
    let cuda = source(COMPONENT);
    let helper = marked(&cuda, "N256 K-order contract");
    let seams = marked(&cuda, "N256 BF16 and FP8 seams");
    let flat = compact(helper);

    assert!(flat.contains("k_block+=W2F_K_GROUP"));
    assert!(flat.contains("k_stage+=W2F_K_STAGE"));
    assert!(flat.contains("pair<2"));
    assert!(helper.contains("w2f_mma"));
    assert!(cuda.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"));
    assert!(seams.contains("shared.gate") && seams.contains("shared.up"));
    assert!(seams.matches("__float2bfloat16").count() >= 6);
    assert!(seams.matches("__bfloat162float").count() >= 6);
    assert!(seams.contains("__NV_SATFINITE") && seams.contains("__NV_E4M3"));
    assert!(seams.contains("W2F_MIN_SCALE"));
}

#[test]
fn two_k128_chunks_have_unique_value_and_scale_owners() {
    let mut values = HashSet::new();
    let mut scales = HashSet::new();
    for chunk in 0..2 {
        for warp in 0..8 {
            for row in (warp..32).step_by(8) {
                for lane in 0..32 {
                    for item in 0..4 {
                        assert!(values.insert((row, chunk * 128 + 4 * lane + item)));
                    }
                    if lane == 0 {
                        assert!(scales.insert((row, chunk)));
                    }
                }
            }
        }
    }
    assert_eq!(values.len(), 32 * 256);
    assert_eq!(scales.len(), 32 * 2);

    let flat = compact(&source(COMPONENT));
    assert!(flat.contains("for(unsignedintchunk_local=0;chunk_local<2;++chunk_local)"));
    assert!(flat.contains("constunsignedintchunk=2*n_tile+chunk_local"));
    assert!(flat.contains("row_local=warp;row_local<W2F_M_TILE;row_local+=W2F_HROW_WARPS"));
    assert!(flat.contains("output_scale[(unsignedlonglong)row*16+chunk]=scale"));
}

#[test]
fn route_is_fully_validated_before_any_table_or_output_access() {
    let cuda = source(COMPONENT);
    let guards = marked(&cuda, "N256 exact-shape guards");
    let flat = compact(guards);
    for contract in [
        "routing_index<W2F_EXPERTS",
        "expert_offsets[routing_index]",
        "expert_offsets[routing_index+1]",
        "route_start<0||route_end<route_start",
        "route_end!=(int)total_rows",
        "__ballot_sync(0xffffffffu,routing_invalid)!=0",
    ] {
        assert!(flat.contains(contract), "route guard omits `{contract}`");
    }
    assert!(
        cuda.find("// END N256 exact-shape guards").unwrap()
            < cuda.find("// BEGIN N256 gate trellis pass").unwrap()
    );
}

#[test]
fn shared_model_stays_below_48_kib() {
    const OUTPUT_TILES: usize = 2 * 32 * 256 * 2;
    const ACTIVATION: usize = 32 * 5 * 16;
    const TRELLIS: usize = 4 * 64 * 16;
    const GEMM: usize = ACTIVATION + TRELLIS;
    const ABS_VALUES: usize = 8 * 128 * 4;
    const TOTAL: usize = OUTPUT_TILES + if GEMM > ABS_VALUES { GEMM } else { ABS_VALUES };
    assert_eq!(TOTAL, 39_424);
    assert!(TOTAL < 48 * 1024);

    let cuda = source(COMPONENT);
    assert!(cuda.contains("sizeof(W2FShared) == 39424"));
    assert!(cuda.contains("sizeof(W2FShared) <= 49152"));
}

#[test]
fn source_model_pins_structural_savings() {
    const ROWS: u64 = 2_410 * 6;
    const LAYERS: u64 = 43;
    const REMOVED_TILES: u64 = 16 - 8;
    let fp8 = ROWS * 2 * 4_096 * REMOVED_TILES;
    let scales = ROWS * 2 * (4_096 / 128) * 4 * REMOVED_TILES;
    assert_eq!(fp8, 947_650_560);
    assert_eq!(scales, 29_614_080);
    assert_eq!((fp8 + scales) * LAYERS, 42_022_379_520);
    assert_eq!(256 * REMOVED_TILES * LAYERS, 88_064);
}

#[test]
fn wrapper_and_sass_gate_bind_only_the_exact_component() {
    let wrapper = source(WRAPPER);
    assert!(wrapper.contains("#define W2A8_FIXED_N 2048"));
    assert!(wrapper.contains("#define W2A8_FIXED_K 4096"));
    assert!(wrapper.contains("#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n256"));
    assert_eq!(wrapper.matches("#include").count(), 1);
    assert!(!wrapper.contains("__global__"));

    let gate = source(SASS_GATE);
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "threads=256",
        "allocated_registers_per_block <= register_file_limit",
        "shared <= static_shared_limit",
        "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
        "QMMA.16832.F32.E4M3.E4M3",
        "F2FP.SATFINITE.E4M3",
        "MUFU.EX2",
        "BAR.SYNC",
        "(ATOM|RED|LDL|STL)",
    ] {
        assert!(gate.contains(contract), "SASS gate omits `{contract}`");
    }
}

#[test]
fn admission_probe_locks_production_routes_and_exact_byte_parity() {
    let probe = source(PROBE);
    for contract in [
        "exl3_w2a8_fused_gu_down_emit_n128",
        "exl3_w2a8_fused_gu_down_emit_n256",
        "kProductionRows = 14460",
        "production-balanced",
        "production-empty",
        "production-skewed",
        "production-boundaries",
        "{1, 31, 32, 33, 63, 64, 65, 127, 128, 129}",
        "fp8_mismatches",
        "scale_mismatches",
        "std::memcmp",
    ] {
        assert!(probe.contains(contract), "probe omits `{contract}`");
    }
    assert!(probe.contains("kDownFp8Bytes = 29614080"));
    assert!(probe.contains("kDownScaleBytes = 925440"));
    assert!(probe.contains("route.back() != static_cast<int>(kProductionRows)"));
}

#[test]
fn admission_route_model_keeps_every_case_at_the_production_total() {
    const ROWS: usize = 14_460;
    const EXPERTS: usize = 256;
    const BOUNDARIES: [usize; 10] = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129];

    let distribute = |total: usize, experts: usize| {
        let mut counts = vec![total / experts; experts];
        for count in counts.iter_mut().take(total % experts) {
            *count += 1;
        }
        counts
    };

    let balanced = distribute(ROWS, EXPERTS);
    assert_eq!(balanced.iter().sum::<usize>(), ROWS);
    assert_eq!(balanced.iter().filter(|&&count| count == 57).count(), 124);
    assert_eq!(balanced.iter().filter(|&&count| count == 56).count(), 132);

    let mut empty = balanced.clone();
    empty[17] = 0;
    empty[18] += balanced[17];
    assert_eq!(empty.iter().sum::<usize>(), ROWS);
    assert_eq!(empty.iter().filter(|&&count| count == 0).count(), 1);

    let mut skewed = vec![2_410];
    skewed.extend(distribute(ROWS - 2_410, EXPERTS - 1));
    assert_eq!(skewed.iter().sum::<usize>(), ROWS);
    assert_eq!(skewed[0], 2_410);

    let assigned = BOUNDARIES.iter().sum::<usize>();
    let mut boundary_route = BOUNDARIES.to_vec();
    boundary_route.extend(distribute(ROWS - assigned, EXPERTS - BOUNDARIES.len()));
    assert_eq!(boundary_route.len(), EXPERTS);
    assert_eq!(boundary_route.iter().sum::<usize>(), ROWS);
    assert_eq!(&boundary_route[..BOUNDARIES.len()], &BOUNDARIES);
}

#[test]
fn admission_probe_is_adversarial_and_immutable() {
    let probe = source(PROBE);
    for contract in [
        "kPoisonPairs",
        "{probe::kPoisonA, probe::kPoisonB}",
        "{probe::kPoisonB, probe::kPoisonA}",
        "guards_clean",
        "snapshot_inputs",
        "verify_inputs_immutable",
        "first-malformed-route",
        "late-malformed-route",
        "verify_fully_poisoned",
        "route[kMaxExperts - 8]",
    ] {
        assert!(probe.contains(contract), "probe omits `{contract}`");
    }
    assert!(probe.matches("verify_inputs_immutable").count() >= 2);
}

#[test]
fn admission_probe_uses_abba_and_receipt_bound_threshold() {
    let probe = source(PROBE);
    for contract in [
        "W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP",
        "W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT",
        "run_abba_timing",
        "time_launch(n128)",
        "time_launch(n256)",
        "const float a0",
        "const float b0",
        "const float b1",
        "const float a1",
        "speedup < kMinSpeedup",
        "input_hash=",
        "routing_hash=",
        "tables_hash=",
        "n128_output_hash=",
        "n256_output_hash=",
    ] {
        assert!(probe.contains(contract), "probe omits `{contract}`");
    }

    let build = source(PROBE_BUILD);
    for contract in [
        "decimal_pattern='^(0|[1-9][0-9]*)([.][0-9]*[1-9])?$'",
        "printf \"%.17g\"",
        "receipt_format=atlas-w2a8-fused-gu-n256-v1",
        "git_status_sha256",
        "source_sha256",
        "binary_sha256",
        "cubin_sha256",
        "runner_sha256=",
        "if [[ $# -ne 0 ]]",
        "actual_binary_sha256",
        "actual_receipt_sha256",
        "stdout_lines -ne 10",
        "exl3_w2a8_fused_gu_down_emit_n128",
        "exl3_w2a8_fused_gu_down_emit_n256",
    ] {
        assert!(build.contains(contract), "probe build omits `{contract}`");
    }
}
