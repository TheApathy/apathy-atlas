// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the isolated EXL3 W2A8 grouped-prefill experiment.
//!
//! The component is compiled and inspected offline. DeepSeek-only wrappers
//! expose it solely through a default-off serving experiment; device numeric
//! and timing gates still block promotion.

use std::fs;
use std::path::PathBuf;

const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");
const KERNEL_BUILD: &str = include_str!("../../atlas-kernels/build.rs");
const NATIVE_FP8: &str = include_str!("../../../kernels/gb10/common/fp8_gemm_t_blockscaled.cu");
const NATIVE_W8A8: &str = include_str!("../../../kernels/gb10/common/w8a8_gemm_pipelined.cu");
const PROBE_BUILD: &str = include_str!("../../../scripts/check-exl3-prefill-w2a8-probe-build.sh");
const SASS_GATE: &str = include_str!("../../../scripts/check-exl3-prefill-w2a8-sass.sh");

fn experiment_file(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/experiments")
        .join(name);
    fs::read_to_string(path).expect("isolated W2A8 experiment source must exist")
}

fn component() -> String {
    experiment_file("exl3_w2a8_grouped_prefill.cu")
}

fn emitter() -> String {
    experiment_file("exl3_w2a8_h128_emit.cu")
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).expect("section start");
    let tail = &source[start..];
    let end = tail.find(end).expect("section end");
    &tail[..end]
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn assert_tree_excludes_w2a8(path: &std::path::Path) {
    for entry in fs::read_dir(path).expect("production source tree must exist") {
        let path = entry.expect("production source entry").path();
        if path.is_dir() {
            assert_tree_excludes_w2a8(&path);
            continue;
        }
        let extension = path.extension().and_then(|value| value.to_str());
        if matches!(extension, Some("rs" | "toml" | "cu" | "cuh")) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(
                !source.contains("exl3_w2a8_grouped_prefill"),
                "{}",
                path.display()
            );
            assert!(
                !source.contains("ATLAS_EXL3_PREFILL_W2A8"),
                "{}",
                path.display()
            );
        }
    }
}

#[test]
fn component_consumes_post_h128_a8_with_per_128_scales() {
    let source = component();
    assert!(source.contains("const unsigned char* __restrict__ A_fp8"));
    assert!(source.contains("const float* __restrict__ a_scale"));
    assert!(source.contains("W2A8_SCALE_GROUP 128"));
    assert!(source.contains("W2A8_K_STAGE 64"));
    assert!(source.contains("W2A8_K_PAIR 32"));
    assert!(source.contains("static_assert(W2A8_SCALE_GROUP == 2 * W2A8_K_STAGE"));
    assert!(source.contains("#pragma unroll 1"));
    assert!(source.contains("k_stage < W2A8_SCALE_GROUP"));
    assert!(source.contains("k_stage += W2A8_K_STAGE"));
    assert!(source.contains("(W2A8_FIXED_N == 2048 && W2A8_FIXED_K == 4096)"));
    assert!(source.contains("(W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048)"));
    assert!(source.contains("sorted, post-H128"));
    assert!(!source.contains("reinterpret_cast<const __nv_bfloat16*>(A_fp8)"));
}

#[test]
fn h128_emitter_is_exact_shape_isolated_and_keeps_the_bf16_boundary() {
    let source = emitter();
    assert!(source.contains("exl3_w2a8_h128_pre_dual_emit_h4096"));
    assert!(source.contains("exl3_w2a8_h128_post_silu_pre_emit_h2048"));
    assert!(source.contains("if (K != 4096 || rows != gridDim.x"));
    assert!(source.contains("if (N != 2048 || rows != gridDim.x"));
    assert!(source.contains("__float2bfloat16(value0)"));
    assert!(source.contains("__bfloat162float(boundary0)"));
    assert!(
        source.find("__float2bfloat16(value0)").unwrap() < source.find("fabsf(rounded0)").unwrap()
    );
    assert!(STATE.contains("exl3_w2a8_h128_pre_dual_emit_h4096"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_W2A8"));
    assert!(!source.contains("static cuda_q8_fold"));
    assert!(!source.contains("getenv("));
}

#[test]
fn h128_emitter_replays_standalone_reduction_tree_and_row_major_layout() {
    let source = emitter();
    let reduction = section(
        &source,
        "// BEGIN standalone reduction topology",
        "// END standalone reduction topology",
    );
    assert!(reduction.contains("smem_abs[warp][4 * lane + 0]"));
    for offset in ["lane + 0", "lane + 32", "lane + 64", "lane + 96"] {
        assert!(reduction.contains(offset));
    }
    assert!(reduction.contains("for (int offset = 16; offset > 0; offset >>= 1)"));
    assert_eq!(reduction.matches("__shfl_down_sync").count(), 4);
    assert!(reduction.contains("global_max = fmaxf(global_max, warp_max0)"));
    assert!(reduction.contains("global_max = fmaxf(global_max, warp_max3)"));
    assert!(source.contains("row * (K / W2A8_EMIT_GROUP_K) + chunk"));
    assert!(source.contains("row * K + chunk * W2A8_EMIT_GROUP_K + 4 * lane"));
    assert!(source.contains("__NV_SATFINITE, __NV_E4M3"));
}

#[test]
fn exact_alias_schedule_fits_and_never_compact_writes_over_a_live_input() {
    const TOKENS: usize = 2410;
    const TOPK: usize = 6;
    let rows = TOKENS * TOPK;
    let sidecar_bytes = |k: usize| rows * k + rows * (k / 128) * 4;
    let h4096_arena = rows * 4096 * 2;
    assert_eq!(sidecar_bytes(4096), 61_079_040);
    assert_eq!(sidecar_bytes(2048), 30_539_520);
    assert!(sidecar_bytes(4096) <= h4096_arena);
    assert!(sidecar_bytes(2048) <= h4096_arena);

    // Prospective host schedule: each arena is reused only after its sidecar
    // has been consumed, and the down emitter's two BF16 inputs are disjoint
    // from its compact output.
    let stages = [
        "dual emit",
        "gate W2A8",
        "up W2A8",
        "down emit",
        "down W2A8",
    ];
    let stage = |name| {
        stages
            .iter()
            .position(|candidate| *candidate == name)
            .unwrap()
    };
    assert!(stage("gate W2A8") < stage("up W2A8"));
    assert!(stage("up W2A8") < stage("down emit"));
    assert!(stage("down emit") < stage("down W2A8"));
    let gate_sidecar_arena = "expert_down_out";
    let up_sidecar_arena = "expert_up_out";
    let gate_projection_arena = "expert_gate_out";
    let up_projection_arena = "expert_down_out";
    let down_sidecar_arena = "expert_up_out";
    assert_eq!(gate_sidecar_arena, up_projection_arena);
    assert_eq!(up_sidecar_arena, down_sidecar_arena);
    assert_ne!(down_sidecar_arena, gate_projection_arena);
    assert_ne!(down_sidecar_arena, up_projection_arena);

    let source = emitter();
    assert!(source.contains("gate_fp8"));
    assert!(source.contains("up_fp8"));
    assert!(source.contains("down_fp8"));
    assert!(!source.contains("gate_fp8 = reinterpret_cast<unsigned char*>(gate)"));
}

#[test]
fn sidecar_layout_is_checked_at_chunk_and_scheduler_boundary_rows() {
    let layout = |rows: usize, k: usize| {
        let fp8_bytes = rows.checked_mul(k)?;
        let groups = k.checked_div(128)?;
        let scale_bytes = rows.checked_mul(groups)?.checked_mul(4)?;
        let total = fp8_bytes.checked_add(scale_bytes)?;
        Some((fp8_bytes, scale_bytes, total))
    };

    for (rows, expected_h4096, expected_h2048) in [
        (6_144usize, 25_952_256usize, 12_976_128usize),
        (6_150, 25_977_600, 12_988_800),
        (14_460, 61_079_040, 30_539_520),
    ] {
        let (offset_4096, scales_4096, total_4096) = layout(rows, 4096).unwrap();
        let (offset_2048, scales_2048, total_2048) = layout(rows, 2048).unwrap();
        assert_eq!(total_4096, expected_h4096);
        assert_eq!(total_2048, expected_h2048);
        assert_eq!(offset_4096 % 4, 0);
        assert_eq!(offset_2048 % 4, 0);
        assert_eq!(offset_4096 + scales_4096, total_4096);
        assert_eq!(offset_2048 + scales_2048, total_2048);
        let h4096_arena = rows.checked_mul(4096).unwrap().checked_mul(2).unwrap();
        assert!(total_4096 <= h4096_arena);
        assert!(total_2048 <= h4096_arena);
    }

    assert_eq!(layout(1, 4096), Some((4096, 128, 4224)));
    assert_eq!(layout(usize::MAX, 4096), None);
    assert_eq!(layout(usize::MAX / 4096 + 1, 4096), None);
}

#[test]
fn emitter_sass_gate_pins_exact_topology_and_resource_ceiling() {
    assert!(SASS_GATE.contains("exl3_w2a8_h128_emit.cu"));
    assert!(SASS_GATE.contains("exl3_w2a8_h128_pre_dual_emit_h4096"));
    assert!(SASS_GATE.contains("exl3_w2a8_h128_post_silu_pre_emit_h2048"));
    assert!(SASS_GATE.contains("F2F.BF16.F32"));
    assert!(SASS_GATE.contains("F2FP.SATFINITE.E4M3"));
    assert!(SASS_GATE.contains("SHFL.DOWN"));
    assert!(SASS_GATE.contains("STG.E.U16"));
    assert!(SASS_GATE.contains("4096 bytes smem"));
}

#[test]
fn b_repack_is_cross_lane_and_native_contiguous_k() {
    let source = component();
    let body = section(&source, "w2a8_repack_b(", "// END w2a8_repack_b");
    assert!(body.contains("quad_base"));
    assert!(body.contains("2 * (tid & 1)"));
    assert!(body.contains("tid >> 1"));
    assert_eq!(body.matches("__shfl_sync").count(), 2);
    assert!(body.contains("source0"));
    assert!(body.contains("source1"));
    assert!(!body.contains("return pair01 | ((unsigned int)pair23 << 16)"));
    let body = compact(body);
    assert!(body.contains("constintsrc1=src0+1;"));
    assert!(body.contains("constunsignedintsource0=__shfl_sync(0xffffffffu,source_pairs,src0);"));
    assert!(body.contains("constunsignedintsource1=__shfl_sync(0xffffffffu,source_pairs,src1);"));
    assert!(body.contains("constunsignedintshift=16*(tid>>1);"));
    assert!(body.contains("return((source0>>shift)&0xffffu)|((source1>>shift)<<16);"));

    for native in [NATIVE_FP8, NATIVE_W8A8] {
        assert!(native.contains("4 * tid") || native.contains("tid * 4"));
        assert!(
            native.contains("16 + 4 * tid")
                || native.contains("tid * 4 + 16")
                || native.contains("16 + tid * 4")
        );
    }
}

#[test]
fn a_fragments_and_mma_use_native_k32_layout() {
    let source = component();
    let body = section(
        &source,
        "// BEGIN native A fragments",
        "// END native A fragments",
    );
    assert!(body.contains("4 * tid"));
    assert!(body.contains("16 + 4 * tid"));
    assert!(!body.contains("2 * tid"));
    assert!(!body.contains("8 + 2 * tid"));
    let body = compact(body);
    assert!(body.contains("a+row0*stride+byte_k+4*tid"));
    assert!(body.contains("a+row1*stride+byte_k+4*tid"));
    assert!(body.contains("a+row0*stride+byte_k+16+4*tid"));
    assert!(body.contains("a+row1*stride+byte_k+16+4*tid"));
    assert!(body.contains("constunsignedinta0=*(constunsignedint*)(a+row0*stride+byte_k+4*tid);"));
    assert!(body.contains("constunsignedinta1=*(constunsignedint*)(a+row1*stride+byte_k+4*tid);"));
    assert!(
        body.contains("constunsignedinta2=*(constunsignedint*)(a+row0*stride+byte_k+16+4*tid);")
    );
    assert!(
        body.contains("constunsignedinta3=*(constunsignedint*)(a+row1*stride+byte_k+16+4*tid);")
    );
    let compact_source = compact(&source);
    assert!(compact_source.contains("w2a8_mma(inner[mt][0],a0,a1,a2,a3,b00,b01);"));
    assert!(compact_source.contains("w2a8_mma(inner[mt][1],a0,a1,a2,a3,b10,b11);"));
    assert!(source.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"));
    assert!(!source.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
}

#[test]
fn weight_and_activation_scales_fold_once_before_bf16_rounding() {
    let source = component();
    assert!(source.contains("W2A8_WEIGHT_SCALE 16.0f"));
    assert!(source.contains("W2A8_INV_WEIGHT_SCALE 0.0625f"));
    let encode = section(&source, "w2a8_encode_pair(", "// END w2a8_encode_pair");
    assert!(encode.contains("* W2A8_WEIGHT_SCALE"));
    assert!(encode.contains("cvt.rn.satfinite.e4m3x2.f32"));

    let fold = section(&source, "// BEGIN scale fold", "// END scale fold");
    assert_eq!(fold.matches("W2A8_INV_WEIGHT_SCALE").count(), 2);
    assert!(fold.contains("a_scale"));
    assert!(!fold.contains("__float2bfloat16"));
    let fold_pos = source.find("// BEGIN scale fold").unwrap();
    let fold_end = source.find("// END scale fold").unwrap();
    let round_pos = source.find("__float2bfloat16").unwrap();
    assert!(
        fold_pos < fold_end && fold_end < round_pos,
        "all scale folding must remain FP32 until the BF16 boundary"
    );
}

#[test]
fn fixed_shape_guards_precede_all_output_writes() {
    let source = component();
    assert!(source.contains("__launch_bounds__(128)"));
    assert!(source.contains("blockDim.x != 128"));
    assert!(source.contains("blockDim.y != 1 || blockDim.z != 1"));
    assert!(source.contains("N != W2A8_FIXED_N || K != W2A8_FIXED_K"));
    assert!(source.contains("bits != 2 || persistent_mode != 1"));
    assert!(source.contains("gridDim.y != 1 || gridDim.z != 1"));
    let guard = source.find("N != W2A8_FIXED_N").unwrap();
    let first_store = source.find("__float2bfloat16").unwrap();
    assert!(guard < first_store);
}

#[test]
fn serving_opt_in_keeps_wrappers_model_specific_and_common_tree_clean() {
    let source = component();
    assert!(source.contains("W2A8_KERNEL_NAME"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_W2A8"));
    assert!(DISPATCH.contains("try_run_exl3_w2a8_prefill"));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/experiments/exl3_w2a8_grouped_prefill.cu");
    assert_eq!(path.parent().unwrap().file_name().unwrap(), "experiments");
    let discovery = section(
        KERNEL_BUILD,
        "fn find_cu_files(",
        "#[path = \"build_codegen.rs\"]",
    );
    assert!(discovery.contains("std::fs::read_dir(kernel_dir)"));
    assert!(discovery.contains("path.extension()"));
    assert!(!discovery.contains("WalkDir"));
    assert!(!discovery.contains("is_dir()"));

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for production_tree in ["crates/atlas-kernels", "kernels/gb10/common"] {
        assert_tree_excludes_w2a8(&workspace.join(production_tree));
    }
    let model_dir = workspace.join("kernels/gb10/deepseek-v4-flash/nvfp4");
    assert!(
        model_dir
            .join("exl3_w2a8_grouped_prefill_k2_gu.cu")
            .is_file()
    );
    assert!(
        model_dir
            .join("exl3_w2a8_grouped_prefill_k2_down.cu")
            .is_file()
    );
}

#[test]
fn standalone_probe_requires_explicit_numeric_thresholds_and_tests_both_shapes() {
    let source = experiment_file("exl3_w2a8_grouped_prefill_probe.cu");
    assert!(source.contains("#if W2A8_PROBE_GU"));
    assert!(source.contains("W2A8_BUILD_ID"));
    assert!(source.contains("build_id=%s"));
    assert!(source.contains("exl3_grouped_prefill_k64_k2_gu.cu"));
    assert!(source.contains("exl3_grouped_prefill_k64_k2_down.cu"));
    assert!(source.contains("per_token_group_quant_fp8.cu"));
    assert!(source.contains("exl3_w2a8_grouped_prefill.cu"));
    assert!(source.contains("argc != 4"));
    assert!(source.contains("min_cosine"));
    assert!(source.contains("max_abs_error"));
    assert!(source.contains("min_end_to_end_speedup"));
    assert!(source.contains("min_cosine < 0.99f"));
    assert!(source.contains("max_abs_error > 1.0f"));
    assert!(source.contains("min_end_to_end_speedup <= 1.0f"));
    assert!(source.contains("{1, 63, 64, 65, 127, 128, 129}"));
    assert!(source.contains("{0, 0, 63, 63, 129}"));
    assert!(source.contains("wrong-block canary"));
    assert!(source.contains("wrong-grid canary"));
    assert!(source.contains("wrong-runtime canary"));
    assert!(source.contains("guard_bytes"));
    assert!(source.contains("0xa5"));
    assert!(source.contains("0x5a"));
    assert!(source.contains("cudaEventElapsedTime"));
    assert!(source.contains("baseline_ms0"));
    assert!(source.contains("end_to_end_ms0"));
    assert!(source.contains("end_to_end_ms1"));
    assert!(source.contains("baseline_ms1"));
    assert!(source.contains("cosine < min_cosine"));
    assert!(source.contains("max_abs > max_abs_error"));
    assert!(source.contains("speedup < min_end_to_end_speedup"));
    assert!(source.contains("cudaGetDeviceProperties"));
    assert!(source.contains("thresholds cosine="));
    assert!(source.contains("a8_saturated=%zu/%zu"));
    assert!(source.contains("floor_scales=%zu/%zu"));

    assert!(PROBE_BUILD.contains("source_sha256 $relative"));
    assert!(PROBE_BUILD.contains("check-exl3-prefill-w2a8-probe-build.sh"));
    assert!(PROBE_BUILD.contains("compile_command_${kind}"));
    assert!(PROBE_BUILD.contains("binary_sha256"));
    assert!(PROBE_BUILD.contains("cubin_sha256"));
    assert!(PROBE_BUILD.contains("grep -qx 'invalid explicit numeric threshold'"));
    assert!(PROBE_BUILD.contains("refusing nonempty W2A8_PROBE_OUTPUT_DIR"));
    assert!(PROBE_BUILD.contains("NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS"));
    assert!(PROBE_BUILD.contains("refusing unreceipted nvcc flags"));
    assert!(PROBE_BUILD.contains("build input changed during W2A8 compilation"));
    assert!(PROBE_BUILD.contains("repository identity changed during W2A8 compilation"));
    assert!(PROBE_BUILD.contains("actual_receipt_sha256"));
    assert!(PROBE_BUILD.contains("actual_binary_sha256"));
}
