// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated N128 W2A8 gate/up-to-down-A8 fusion.

use half::bf16;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const COMPONENT_RELATIVE: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";
const COMPONENT_NAME: &str = "exl3_w2a8_fused_gu_down_emit_n128";
const WRAPPER_RELATIVE: &str =
    "kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n128.cu";
const KERNEL_BUILD: &str = include_str!("../../atlas-kernels/build.rs");
const MODEL_REGISTRY: &str = include_str!("../../../kernels/gb10/deepseek-v4-flash/MODEL.toml");
const SASS_GATE: &str =
    include_str!("../../../scripts/check-exl3-prefill-w2a8-fused-gu-n128-sass.sh");

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn component() -> String {
    fs::read_to_string(workspace().join(COMPONENT_RELATIVE))
        .expect("isolated fused N128 W2A8 component must exist")
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn marked_section<'a>(source: &'a str, name: &str) -> &'a str {
    let begin = format!("// BEGIN {name}");
    let end = format!("// END {name}");
    assert_eq!(
        source.matches(&begin).count(),
        1,
        "duplicate/missing {begin}"
    );
    assert_eq!(source.matches(&end).count(), 1, "duplicate/missing {end}");
    let start = source.find(&begin).unwrap() + begin.len();
    let finish = source[start..].find(&end).unwrap() + start;
    assert!(start < finish, "reversed markers for {name}");
    &source[start..finish]
}

fn marker_position(source: &str, marker: &str) -> usize {
    source
        .find(&format!("// BEGIN {marker}"))
        .unwrap_or_else(|| panic!("missing stage marker `{marker}`"))
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn valid_expert_range(m_start: i32, m_end: i32, total_rows: u32) -> bool {
    total_rows != 0
        && m_start >= 0
        && m_end > m_start
        && u32::try_from(m_end).is_ok_and(|end| end <= total_rows)
}

#[test]
fn exact_n128_gu_shape_and_launch_fail_closed_before_compute() {
    let source = component();
    let flat = compact(&source);
    let guards = marked_section(&source, "fused exact-shape guards");

    assert!(source.contains("#define W2F_M_TILE 64"));
    assert!(source.contains("#define W2F_N_TILE 128"));
    assert!(source.contains("#define W2F_K_GROUP 128"));
    for external in ["W2A8_FIXED_N", "W2A8_FIXED_K", "W2A8_KERNEL_NAME"] {
        assert!(
            source.contains(&format!("#ifndef {external}"))
                && source.contains(&format!("#error \"{external} must be explicit\"")),
            "compile-only ABI must require `{external}`"
        );
    }
    assert!(source.contains("W2F_GATE_UP_N == 2048 && W2F_GATE_UP_K == 4096"));
    assert!(source.contains("void W2A8_KERNEL_NAME("));
    assert!(source.contains("__launch_bounds__(256)"));
    assert!(flat.contains("blockDim.x!=256||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.y!=1||gridDim.z!=1"));
    assert!(flat.contains("N!=W2F_GATE_UP_N||K!=W2F_GATE_UP_K"));
    assert!(flat.contains("bits!=2||persistent_mode!=1"));
    assert!(flat.contains("total_rows==0"));
    assert!(flat.contains("gridDim.x!=(unsignedlonglong)num_experts*n_tiles"));
    assert!(flat.contains("total_rows>(unsignedint)INT_MAX"));
    assert!(flat.contains("num_experts!=W2F_EXPERTS"));
    assert!(flat.contains(
        "for(unsignedintrouting_index=route_lane;routing_index<W2F_EXPERTS;routing_index+=32)"
    ));
    assert!(flat.contains("expert_offsets[routing_index]"));
    assert!(flat.contains("expert_offsets[routing_index+1]"));
    assert!(
        flat.contains("route_start<0||route_end<route_start||(unsignedint)route_end>total_rows")
    );
    assert!(flat.contains("routing_index==0&&route_start!=0"));
    assert!(flat.contains("route_end!=(int)total_rows"));
    assert!(flat.contains("__ballot_sync(0xffffffffu,routing_invalid)!=0"));
    assert!(flat.contains("if(m_start==m_end)return"));
    assert!(
        flat.contains("unsignedintnum_experts,unsignedinttotal_rows,unsignedintN,unsignedintK")
    );
    assert!(guards.contains("return"));
    assert!(
        source.find("// END fused exact-shape guards").unwrap()
            < marker_position(&source, "fused gate trellis pass"),
        "all launch guards must precede reads and writes"
    );
}

#[test]
fn total_rows_guard_rejects_every_malformed_signed_range() {
    assert!(valid_expert_range(0, 1, 1));
    assert!(valid_expert_range(63, 129, 129));
    assert!(!valid_expert_range(0, 1, 0));
    assert!(!valid_expert_range(-1, 1, 1));
    assert!(!valid_expert_range(0, 0, 1));
    assert!(!valid_expert_range(2, 1, 2));
    assert!(!valid_expert_range(0, 130, 129));
    assert!(!valid_expert_range(i32::MAX - 1, i32::MAX, 129));
}

#[test]
fn eight_warps_own_every_m64_n128_projection_output_once() {
    let mut owners = HashSet::new();
    for warp in 0..8 {
        for lane in 0..32 {
            let group = lane / 4;
            let tid = lane % 4;
            for mt in 0..4 {
                for nt in 0..2 {
                    let column = warp * 16 + nt * 8 + tid * 2;
                    for row in [mt * 16 + group, mt * 16 + group + 8] {
                        assert!(owners.insert((row, column)));
                        assert!(owners.insert((row, column + 1)));
                    }
                }
            }
        }
    }
    assert_eq!(owners.len(), 64 * 128);
    assert!(owners.iter().all(|&(row, column)| row < 64 && column < 128));
}

#[test]
fn gate_and_up_consume_distinct_trellises_sequentially() {
    let source = component();
    let gate = marked_section(&source, "fused gate trellis pass");
    let up = marked_section(&source, "fused up trellis pass");
    let gate_at = marker_position(&source, "fused gate trellis pass");
    let up_at = marker_position(&source, "fused up trellis pass");

    assert!(
        gate_at < up_at,
        "gate projection must precede up projection"
    );
    assert!(gate.contains("gate_trellis_tab"));
    assert!(!gate.contains("up_trellis_tab"));
    assert!(up.contains("up_trellis_tab"));
    assert!(!up.contains("gate_trellis_tab"));
    assert!(gate.contains("w2f_compute_leg"));
    assert!(gate.contains("gate_fp8") && gate.contains("gate_scale"));
    assert!(gate.contains("gate_trellis") && gate.contains("shared.gate"));
    assert!(up.contains("w2f_compute_leg"));
    assert!(up.contains("up_fp8") && up.contains("up_scale"));
    assert!(up.contains("up_trellis") && up.contains("shared.up"));
    let helper = &source[source.find("void w2f_compute_leg(").unwrap()
        ..source.find("void w2f_emit_group(").unwrap()];
    let helper_flat = compact(helper);
    assert!(helper_flat.contains("k_block+=W2F_K_GROUP"));
    assert!(helper_flat.contains("k_stage+=W2F_K_STAGE"));
    assert!(source.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"));
    assert!(helper.contains("w2f_mma"));
    assert!(helper.contains("scratch.trellis"));
    assert!(
        helper.rfind("__syncthreads()").unwrap() > helper.rfind("output[row1]").unwrap(),
        "each projection must retire its shared output before the next leg"
    );
    let between = &source[source.find("// END fused gate trellis pass").unwrap()..up_at];
    assert!(
        between.contains("__syncthreads()"),
        "gate trellis/accumulator state must retire before up reuses it"
    );
}

#[test]
fn fused_pipeline_keeps_all_bf16_boundaries_in_legacy_order() {
    let source = component();
    let boundaries = marked_section(&source, "fused bf16 round-reexpand barriers");
    let transform = marked_section(&source, "fused h128 swiglu down-pre order");

    for boundary in [
        "shared.gate",
        "shared.up",
        "const __nv_bfloat16 a0",
        "w2f_emit_group",
    ] {
        assert!(boundaries.contains(boundary), "missing `{boundary}`");
    }
    assert!(boundaries.matches("__float2bfloat16").count() >= 6);
    assert!(boundaries.matches("__bfloat162float").count() >= 6);

    let gate = marker_position(&source, "fused gate trellis pass");
    let up = marker_position(&source, "fused up trellis pass");
    let h128 = marker_position(&source, "fused h128 swiglu down-pre order");
    assert!(gate < up && up < h128);

    let flat = compact(transform);
    let gate_h128 = flat.find("gate_svh").expect("gate post-H128/SVH");
    let up_h128 = flat.find("up_svh").expect("up post-H128/SVH");
    let swiglu = flat.find("W2F_SWIGLU_LIMIT").expect("bounded SwiGLU");
    let down_pre = flat.find("down_suh").expect("down pre-H128/SUH");
    assert!(gate_h128 < swiglu && up_h128 < swiglu && swiglu < down_pre);
    assert!(transform.contains("w2f_had128"));
    assert!(transform.contains("__expf"));
    assert!(transform.contains("w2f_emit_group"));
}

#[test]
fn bf16_boundary_oracle_rejects_each_skipped_materialization() {
    fn pipeline(gate: f32, up: f32, skip: Option<usize>) -> f32 {
        let boundary = |stage: usize, value: f32| {
            if skip == Some(stage) {
                value
            } else {
                round_bf16(value)
            }
        };
        let gate = boundary(0, gate);
        let up = boundary(1, up);
        let gate = boundary(2, gate.mul_add(1.03125, -0.0078125));
        let up = boundary(3, up.mul_add(-0.9375, 0.015625));
        let gate = gate.min(10.0);
        let up = up.clamp(-10.0, 10.0);
        let activation = boundary(4, gate * (1.0 / (1.0 + (-gate).exp())) * up);
        boundary(5, activation.mul_add(0.8125, -0.00390625))
    }

    let values: Vec<f32> = (-96..=96).map(|value| value as f32 / 37.0).collect();
    let mut exposed = [false; 6];
    for &gate in &values {
        for &up in &values {
            let reference = pipeline(gate, up, None).to_bits();
            for (stage, differs) in exposed.iter_mut().enumerate() {
                *differs |= pipeline(gate, up, Some(stage)).to_bits() != reference;
            }
        }
    }
    assert!(
        exposed.into_iter().all(|value| value),
        "adversarial values must expose every removed BF16 boundary"
    );
}

#[test]
fn h128_and_down_a8_phases_cover_each_row_group_once() {
    // Eight warps process eight rows at a time. A lane owns four adjacent
    // values, matching the incumbent H128 and emitter topology.
    let mut value_owners = HashSet::new();
    let mut scale_owners = HashSet::new();
    for row_batch in 0..8 {
        for warp in 0..8 {
            let row = row_batch * 8 + warp;
            for lane in 0..32 {
                for item in 0..4 {
                    assert!(value_owners.insert((row, 4 * lane + item)));
                }
                if lane == 0 {
                    assert!(scale_owners.insert(row));
                }
            }
        }
    }
    assert_eq!(value_owners.len(), 64 * 128);
    assert_eq!(scale_owners, (0..64).collect());

    let source = component();
    let quant = marked_section(&source, "fused down-a8 quant reduction");
    let flat = compact(quant);
    let source_flat = compact(&source);
    assert!(source_flat.contains("row_local=warp"));
    assert!(source_flat.contains("row_local+=W2F_HROW_WARPS"));
    assert!(source_flat.contains("row=m_start+m_local+(int)row_local"));
    for offset in ["lane+0", "lane+32", "lane+64", "lane+96"] {
        assert!(flat.contains(offset), "missing reduction slice `{offset}`");
    }
    assert!(quant.contains("for (int offset = 16; offset > 0; offset >>= 1)"));
    assert!(quant.contains("__shfl_down_sync"));
    assert!(quant.contains("lane == 0"));
    assert!(quant.contains("W2F_E4M3_MAX"));
    assert!(quant.contains("W2F_MIN_SCALE"));
    assert!(quant.contains("__NV_SATFINITE"));
    assert!(quant.contains("__NV_E4M3"));
    for value in 0..4 {
        assert!(
            flat.contains(&format!(
                "rounded{value}=__bfloat162float(__float2bfloat16(value{value}))"
            )),
            "down-pre value {value} must cross the incumbent BF16 boundary before quantization"
        );
    }
    assert!(flat.contains("output_scale[(unsignedlonglong)row*16+chunk]=scale"));
    assert!(flat.contains("output_fp8+(unsignedlonglong)row*W2F_GATE_UP_N+chunk*W2F_K_GROUP"));
}

#[test]
fn production_model_removes_two_launches_and_gate_up_roundtrip_per_layer() {
    const TOKENS: u64 = 2_410;
    const TOP_K: u64 = 6;
    const COLUMNS: u64 = 2_048;
    const LAYERS: u64 = 43;
    const BF16_BYTES: u64 = 2;

    let rows = TOKENS * TOP_K;
    let one_tensor_pass = rows * COLUMNS * BF16_BYTES;
    // The incumbent writes gate and up once, then the separate emitter reads
    // both once. The fused CTA keeps those four logical passes in shared memory.
    let eliminated_bytes_per_layer = 4 * one_tensor_pass;
    assert_eq!(one_tensor_pass, 59_228_160);
    assert_eq!(eliminated_bytes_per_layer, 236_912_640);
    assert_eq!(eliminated_bytes_per_layer * LAYERS, 10_187_243_520);
    assert_eq!(2 * LAYERS, 86);
}

#[test]
fn shared_memory_model_is_39_kib_and_hard_capped_below_48_kib() {
    const ACTIVATION_STAGE: usize = 64 * 5 * 16;
    const TRELLIS_STAGE: usize = 4 * 32 * 16;
    const GATE_TILE: usize = 64 * 128 * 2;
    const UP_TILE: usize = 64 * 128 * 2;
    const TOTAL: usize = ACTIVATION_STAGE + TRELLIS_STAGE + GATE_TILE + UP_TILE;
    assert_eq!(TOTAL, 39 * 1024);
    assert!(TOTAL <= 48 * 1024);

    let source = component();
    assert!(source.contains("sizeof(W2FShared) == 39 * 1024"));
    assert!(source.contains("sizeof(W2FShared) <= 49152"));
    assert!(source.contains("uint4 activation[W2F_M_TILE][5]"));
    assert!(source.contains("uint4 trellis[4][32]"));
    assert!(source.contains("__nv_bfloat16 gate[W2F_M_TILE][W2F_N_TILE]"));
    assert!(source.contains("__nv_bfloat16 up[W2F_M_TILE][W2F_N_TILE]"));
}

#[test]
fn compile_gate_binds_exact_abi_and_rejects_crossed_shapes() {
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "-DW2A8_FIXED_N=2048",
        "-DW2A8_FIXED_K=4096",
        "-DW2A8_KERNEL_NAME=\"$symbol\"",
        "symbol=exl3_w2a8_fused_gu_down_emit_n128",
        "expect_compile_failure missing_all",
        "expect_compile_failure missing_symbol",
        "expect_compile_failure unsupported_n",
        "expect_compile_failure crossed_gu",
        "expect_compile_failure crossed_down",
        "allocated_registers_per_block <= register_file_limit",
        "shared <= static_shared_limit",
        "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
        "(ATOM|RED|LDL|STL)",
    ] {
        assert!(SASS_GATE.contains(contract), "SASS gate omits `{contract}`");
    }
    assert!(SASS_GATE.contains("QMMA.16832.F32.E4M3.E4M3"));
    assert!(SASS_GATE.contains("F2FP.SATFINITE.E4M3"));
    assert!(SASS_GATE.contains("MUFU.EX2"));
    assert!(SASS_GATE.contains("BAR.SYNC"));
    for exact in [
        "qmmas == 32",
        "bf16_rounds == 52",
        "fp8_converts == 36",
        "swiglu_exp == 4",
        "fma_ops == 166",
        "shuffle_ops == 193",
        "barriers == 7",
    ] {
        assert!(SASS_GATE.contains(exact), "SASS gate omits `{exact}`");
    }
}

fn assert_tree_excludes(path: &Path) {
    for entry in fs::read_dir(path).expect("production tree must exist") {
        let path = entry.expect("production entry").path();
        if path.is_dir() {
            assert_tree_excludes(&path);
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(!source.contains(COMPONENT_NAME), "{}", path.display());
            assert!(
                !source.contains("ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN"),
                "{}",
                path.display()
            );
        }
    }
}

#[test]
fn experiment_body_stays_isolated_behind_one_canonical_wrapper() {
    assert!(!KERNEL_BUILD.contains(COMPONENT_NAME));
    assert!(!MODEL_REGISTRY.contains(COMPONENT_NAME));

    let root = workspace();
    assert_tree_excludes(&root.join("kernels/gb10/common"));
    let wrapper = fs::read_to_string(root.join(WRAPPER_RELATIVE))
        .expect("canonical fused N128 W2A8 wrapper must exist");
    assert!(wrapper.contains(COMPONENT_NAME));
    assert_eq!(wrapper.matches("#include").count(), 1);
    assert!(!wrapper.contains("__global__"));
    let path = root.join(COMPONENT_RELATIVE);
    assert_eq!(path.parent().unwrap().file_name().unwrap(), "experiments");
}
