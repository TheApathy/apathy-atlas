// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated EXL3 W2A8 M64xN256 experiment.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const COMPONENT_NAME: &str = "exl3_w2a8_grouped_prefill_n256.cu";
const SASS_SCRIPT_NAME: &str = "check-exl3-prefill-w2a8-n256-sass.sh";
const KERNEL_BUILD: &str = include_str!("../../atlas-kernels/build.rs");
const MODEL_REGISTRY: &str = include_str!("../../../kernels/gb10/deepseek-v4-flash/MODEL.toml");
const EXL3_STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const EXL3_DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn required_source(relative: impl AsRef<Path>, label: &str) -> String {
    let path = workspace().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "required W2A8 N256 {label} is missing at {}: {error}",
            path.display()
        )
    })
}

fn component() -> String {
    required_source(
        Path::new("kernels/gb10/experiments").join(COMPONENT_NAME),
        "component",
    )
}

fn sass_script() -> String {
    required_source(Path::new("scripts").join(SASS_SCRIPT_NAME), "SASS gate")
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
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
        } else if path.file_name().and_then(|value| value.to_str())
            == Some("exl3_w2a8_grouped_prefill_n256_k2_down.cu")
        {
            let wrapper = fs::read_to_string(&path).expect("N256 down wrapper must be UTF-8");
            assert_eq!(
                wrapper,
                concat!(
                    "// SPDX-License-Identifier: AGPL-3.0-only\n",
                    "// DeepSeek-V4 exact K2 N256 down W2A8 grouped-prefill wrapper.\n",
                    "\n",
                    "#define W2A8_FIXED_N 4096\n",
                    "#define W2A8_FIXED_K 2048\n",
                    "#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n256_k2_down\n",
                    "#ifdef W2A8_PACKED_E4M3_CANDIDATE\n",
                    "#undef W2A8_PACKED_E4M3_CANDIDATE\n",
                    "#endif\n",
                    "#define W2A8_PACKED_E4M3_CANDIDATE 1\n",
                    "#ifdef W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE\n",
                    "#undef W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE\n",
                    "#endif\n",
                    "#define W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE 1\n",
                    "#ifdef W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE\n",
                    "#undef W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE\n",
                    "#endif\n",
                    "#define W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE 1\n",
                    "#include \"../../experiments/exl3_w2a8_grouped_prefill_n256.cu\"\n",
                ),
                "only the exact include-only DeepSeek N256 down wrapper is allowed"
            );
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(
                !source.contains("exl3_w2a8_grouped_prefill_n256")
                    && !source.contains("ATLAS_EXL3_PREFILL_W2A8_N256"),
                "compile-only W2A8 N256 leaked into {}",
                path.display()
            );
        }
    }
}

#[test]
fn component_is_exact_m64_n256_k2_and_fails_closed() {
    let source = component();
    let flat = compact(&source);
    for definition in [
        "#define W2A8_M_TILE 64",
        "#define W2A8_N_TILE 256",
        "#define W2A8_THREADS 512",
        "#define W2A8_WARPS (W2A8_THREADS / 32)",
        "#define W2A8_SCALE_GROUP 128",
        "#define W2A8_K_STAGE 64",
        "#define W2A8_K_PAIR 32",
    ] {
        assert!(source.contains(definition), "missing `{definition}`");
    }
    assert!(source.contains("__launch_bounds__(W2A8_THREADS)"));
    assert!(flat.contains("blockDim.x!=W2A8_THREADS||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.y!=1||gridDim.z!=1"));
    assert!(flat.contains("N!=W2A8_FIXED_N||K!=W2A8_FIXED_K"));
    assert!(flat.contains("bits!=2||persistent_mode!=1"));
    assert!(flat.contains("num_experts==0||total_rows==0"));
    assert!(flat.contains("gridDim.x!=(unsignedlonglong)num_experts*n_tiles"));
    assert!(flat.contains(
        "A_fp8==nullptr||a_scale==nullptr||trellis_tab==nullptr||C==nullptr||expert_offsets==nullptr"
    ));
    assert!(flat.contains("total_rows>(unsignedint)INT_MAX"));
    assert!(flat.contains(
        "for(unsignedintrouting_index=route_lane;routing_index<num_experts;routing_index+=32)"
    ));
    assert!(flat.contains("expert_offsets[routing_index]"));
    assert!(flat.contains("expert_offsets[routing_index+1]"));
    assert!(
        flat.contains("route_start<0||route_end<route_start||(unsignedint)route_end>total_rows")
    );
    assert!(flat.contains("routing_index==0&&route_start!=0"));
    assert!(flat.contains("routing_index+1==num_experts&&route_end!=(int)total_rows"));
    assert!(flat.contains("__ballot_sync(0xffffffffu,routing_invalid)!=0"));
    let validation = source.find("const unsigned int route_lane").unwrap();
    let first_output_extent = source.find("const int m_start").unwrap();
    assert!(validation < first_output_extent);
    assert!(source.contains("W2A8_FIXED_N == 2048 && W2A8_FIXED_K == 4096"));
    assert!(source.contains("W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048"));
}

#[test]
fn promotion_probe_passes_row_extent_and_canaries_malformed_offsets() {
    let probe = required_source(
        "kernels/gb10/experiments/exl3_w2a8_grouped_prefill_probe.cu",
        "promotion probe",
    );
    let flat = compact(&probe);
    assert!(flat.contains("#ifW2A8_PROBE_N_TILE==256"));
    assert!(flat.contains(
        "input,scales,trellis,output,offsets,num_experts,total_rows,n,k,bits,persistent_mode"
    ));
    for canary in [
        "negative-offset",
        "oversized-positive-offset",
        "nonmonotonic-offset",
        "prefix-gap-offset",
        "final-gap-offset",
        "rows_capacity + 1",
        "null-inputs",
    ] {
        assert!(probe.contains(canary), "probe omits `{canary}`");
    }
}

#[test]
fn sixteen_warps_own_every_m64_n256_output_exactly_once() {
    let mut owners = HashSet::new();
    for warp in 0..16 {
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
    assert_eq!(owners.len(), 64 * 256);
    assert!(owners.iter().all(|&(row, col)| row < 64 && col < 256));
}

#[test]
fn sixteen_warps_cover_each_native_k32_n256_fragment_coordinate_once() {
    let mut coordinates = HashSet::new();
    for warp in 0..16 {
        for group in 0..8 {
            for n_half in 0..2 {
                for tid in 0..4 {
                    for k_half in [0, 16] {
                        for (k, local_n) in repacked_b_coords(group, n_half, tid, k_half) {
                            assert!(coordinates.insert((k, warp * 16 + local_n)));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(coordinates.len(), 32 * 256);
    assert!(coordinates.iter().all(|&(k, n)| k < 32 && n < 256));
}

#[test]
fn k128_trellis_and_activation_staging_are_complete_bijections() {
    let source = component();
    let flat = compact(&source);
    assert!(source.contains("uint4 smem_A[W2A8_M_TILE][5]"));
    assert!(source.contains("#define W2A8_A_VECTORS (W2A8_M_TILE * 4)"));
    assert!(source.contains("#define W2A8_T_WORDS (W2A8_N_TILE / 4)"));
    assert!(source.contains("#define W2A8_T_VECTORS (4 * W2A8_T_WORDS)"));
    assert!(source.contains("uint4 smem_T[4][W2A8_T_WORDS]"));
    assert!(flat.contains("vector<W2A8_A_VECTORS;vector+=blockDim.x"));
    assert!(flat.contains("load<W2A8_T_VECTORS;load+=blockDim.x"));
    assert!(flat.contains("constunsignedintk_tile=load/W2A8_T_WORDS"));
    assert!(flat.contains("constunsignedintword=load%W2A8_T_WORDS"));
    assert!(source.contains("#define W2A8_CURRENT_T smem_T"));
    assert!(flat.contains("(constunsignedint*)W2A8_CURRENT_T[pair*2]+warp*16"));
    assert!(flat.contains("constunsignedintcol=n_base+warp*16+nt*8+tid*2"));

    let activation_owners: HashSet<_> = (0..512)
        .filter(|&thread| thread < 64 * 4)
        .map(|thread| (thread, thread >> 2, thread & 3))
        .collect();
    assert_eq!(activation_owners.len(), 256);
    assert!(
        activation_owners
            .iter()
            .all(|&(thread, row, vector_k)| thread < 256 && row < 64 && vector_k < 4)
    );

    let staged: HashSet<_> = (0..256).map(|load| (load >> 6, load & 63)).collect();
    assert_eq!(staged.len(), 4 * 64);
    let staged_u32: HashSet<_> = staged
        .iter()
        .flat_map(|&(k_tile, uint4_word)| (0..4).map(move |lane| (k_tile, uint4_word * 4 + lane)))
        .collect();
    let consumed: HashSet<_> = (0..4)
        .flat_map(|k_tile| {
            (0..16).flat_map(move |warp| (0..16).map(move |word| (k_tile, warp * 16 + word)))
        })
        .collect();
    assert_eq!(staged_u32, consumed);
}

#[test]
fn native_k32_mma_and_k128_scale_fold_are_retained() {
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
fn n256_halves_n128_grid_and_logical_activation_reads() {
    const TOKENS: u64 = 2_410;
    const TOP_K: u64 = 6;
    const EXPERTS: u64 = 256;
    const LAYERS: u64 = 43;
    let rows = TOKENS * TOP_K;
    let logical_reads = |tile: u64| 2 * rows * 4096 * (2048 / tile) + rows * 2048 * (4096 / tile);
    assert_eq!(logical_reads(128), 2_842_951_680);
    assert_eq!(logical_reads(256), 1_421_475_840);
    assert_eq!(logical_reads(128), 2 * logical_reads(256));
    assert_eq!(
        (logical_reads(128) - logical_reads(256)) * LAYERS,
        61_123_461_120
    );

    let ctas = |tile: u64| 2 * EXPERTS * (2048 / tile) + EXPERTS * (4096 / tile);
    assert_eq!(ctas(128), 16_384);
    assert_eq!(ctas(256), 8_192);
    assert_eq!((ctas(128) - ctas(256)) * LAYERS, 352_256);
}

#[test]
fn production_reachability_is_limited_to_the_exact_down_wrapper() {
    let root = workspace();
    assert!(!KERNEL_BUILD.contains("exl3_w2a8_grouped_prefill_n256"));
    assert!(!MODEL_REGISTRY.contains("exl3_w2a8_grouped_prefill_n256"));
    assert!(
        !EXL3_STATE.contains("exl3_w2a8_grouped_prefill_n256\"")
            && !EXL3_STATE.contains("std::env::var(\"ATLAS_EXL3_PREFILL_W2A8_N256\")")
    );
    assert!(!EXL3_DISPATCH.contains("exl3_w2a8_grouped_prefill_n256\""));
    assert_tree_excludes(&root.join("kernels/gb10"));
    let source = component();
    assert!(!source.contains("getenv("));
    assert!(!source.contains("fallback"));
}

#[test]
fn sass_gate_compiles_both_exact_shapes_and_rejects_resource_regressions() {
    let script = sass_script();
    assert!(script.contains(COMPONENT_NAME));
    for contract in [
        "compile_and_check gu 2048 4096",
        "compile_and_check down 4096 2048",
        "expect_compile_failure 3072 4096",
        "expect_compile_failure 2048 2048",
        "expect_compile_failure 4096 4096",
        "--dump-resource-usage",
        "--fmad=false",
        "-arch=sm_121a",
        "spill stores",
        "spill loads",
    ] {
        assert!(script.contains(contract), "SASS gate omits `{contract}`");
    }
    assert!(script.contains("register_file_limit=65536"));
    assert!(script.contains("allocated_register_block <= register_file_limit"));
    assert!(script.contains("$registers == 97"));
    assert!(script.contains("$stack == 0"));
    assert!(script.contains("$local_bytes == 0"));
    assert!(script.contains("$shared == 10240"));
    assert!(script.contains("$instructions == 1093"));
    assert!(script.contains("routing_index < num_experts"));
    assert!(script.contains("__ballot_sync(0xffffffffu, routing_invalid)"));
    assert!(script.contains("grep -c 'QMMA.16832.F32.E4M3.E4M3'"));
    assert!(script.contains("== 16"));
    assert!(script.contains("grep -c 'SHFL.IDX'"));
    assert!(script.contains("grep -c 'F2FP.SATFINITE.E4M3'"));
    assert!(script.contains("ATOM|RED|LDL|STL"));
}
