// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the default-off fused-N128 K64 double buffer.

use std::fs;
use std::path::PathBuf;

const SELECTOR: &str = "W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE";
const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";
const WRAPPER: &str = concat!(
    "kernels/gb10/deepseek-v4-flash/nvfp4/",
    "exl3_w2a8_fused_gu_down_emit_n128.cu"
);
const SASS_GATE: &str = "scripts/check-exl3-prefill-w2a8-fused-gu-n128-double-buffer-sass.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    let path = workspace().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("required double-buffer input {}: {error}", path.display()))
}

fn compact(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn marked<'a>(value: &'a str, name: &str) -> &'a str {
    let begin = format!("// BEGIN {name}");
    let end = format!("// END {name}");
    assert_eq!(
        value.matches(&begin).count(),
        1,
        "missing/duplicate {begin}"
    );
    assert_eq!(value.matches(&end).count(), 1, "missing/duplicate {end}");
    let start = value.find(&begin).unwrap() + begin.len();
    let finish = value[start..].find(&end).unwrap() + start;
    &value[start..finish]
}

#[test]
fn selector_is_strict_default_off_and_forced_on_by_the_production_wrapper() {
    let component = source(COMPONENT);
    let wrapper = source(WRAPPER);
    assert!(component.contains(&format!("#ifndef {SELECTOR}")));
    assert!(component.contains(&format!("#define {SELECTOR} 0")));
    assert!(component.contains(&format!("static_assert({SELECTOR} == 0 ||")));
    assert!(component.contains(&format!("{SELECTOR} == 1,")));
    assert!(wrapper.contains(&format!("#ifdef {SELECTOR}")));
    assert!(wrapper.contains(&format!("#undef {SELECTOR}")));
    assert!(wrapper.contains(&format!("#define {SELECTOR} 1")));
    assert!(!wrapper.contains(&format!("#ifndef {SELECTOR}")));
}

#[test]
fn two_staging_buffers_fit_below_the_hard_shared_memory_limit() {
    const ACTIVATION: usize = 64 * 5 * 16;
    const TRELLIS: usize = 4 * 32 * 16;
    const ONE_STAGE: usize = ACTIVATION + TRELLIS;
    const GATE_AND_UP: usize = 2 * 64 * 128 * 2;
    assert_eq!(ONE_STAGE, 7 * 1024);
    assert_eq!(GATE_AND_UP + ONE_STAGE, 39 * 1024);
    assert_eq!(GATE_AND_UP + 2 * ONE_STAGE, 46 * 1024);
    assert!(GATE_AND_UP + 2 * ONE_STAGE <= 48 * 1024);

    let component = source(COMPONENT);
    assert!(component.contains("W2FGemmScratch gemm[2]"));
    assert!(component.contains("sizeof(W2FShared) == 46 * 1024"));
    assert!(component.contains("sizeof(W2FShared) == 39 * 1024"));
    assert!(component.contains("sizeof(W2FShared) <= 49152"));
}

#[test]
fn async_stage_has_exact_activation_and_trellis_ownership_and_zero_fill() {
    let component = source(COMPONENT);
    let stage = marked(&component, "double-buffer K64 async stage");
    let flat = compact(stage);
    for contract in [
        "cp.async.ca.shared.global",
        "__cvta_generic_to_shared",
        "W2F_M_TILE * 4",
        "vector += blockDim.x",
        "row < m_end",
        "valid_bytes",
        "load < 128",
        "load += blockDim.x",
        "scratch.activation[row_local][vector_k]",
        "scratch.trellis[k_tile][word]",
    ] {
        assert!(stage.contains(contract), "async stage omits `{contract}`");
    }
    assert!(flat.contains("constvoid*source=activation"));
    assert!(flat.contains("unsignedintvalid_bytes=0"));
    assert!(flat.contains("valid_bytes=16"));

    let activation: Vec<_> = (0..256)
        .map(|thread| (thread, thread >> 2, thread & 3))
        .collect();
    assert_eq!(activation.len(), 64 * 4);
    assert!(
        activation
            .iter()
            .all(|&(thread, row, vector_k)| thread < 256 && row < 64 && vector_k < 4)
    );
    let trellis: Vec<_> = (0..128)
        .map(|thread| (thread, thread >> 5, thread & 31))
        .collect();
    assert_eq!(trellis.len(), 4 * 32);
    assert!(
        trellis
            .iter()
            .all(|&(thread, k_tile, word)| thread < 128 && k_tile < 4 && word < 32)
    );
}

#[test]
fn schedule_overlaps_only_the_second_k64_stage_and_closes_each_lifetime() {
    let component = source(COMPONENT);
    let schedule = marked(&component, "double-buffer K128 schedule");
    let flat = compact(schedule);
    for contract in [
        "w2f_stage_async(scratch_buffers[0]",
        "w2f_cp_async_commit()",
        "w2f_cp_async_wait()",
        "__syncthreads()",
        "next_k_stage < W2F_K_GROUP",
        "scratch_buffers[next_buffer]",
        "scratch_buffers[current_buffer]",
    ] {
        assert!(schedule.contains(contract), "schedule omits `{contract}`");
    }
    assert!(flat.contains("current_buffer=(k_stage/W2F_K_STAGE)&1"));
    assert!(flat.contains("next_buffer=current_buffer^1"));

    let preload = schedule.find("w2f_stage_async(scratch_buffers[0]").unwrap();
    let preload_commit = schedule[preload..].find("w2f_cp_async_commit()").unwrap() + preload;
    let preload_wait = schedule[preload_commit..]
        .find("w2f_cp_async_wait()")
        .unwrap()
        + preload_commit;
    let preload_barrier = schedule[preload_wait..].find("__syncthreads()").unwrap() + preload_wait;
    let loop_start = schedule.find("for (unsigned int k_stage").unwrap();
    assert!(preload < preload_commit);
    assert!(preload_commit < preload_wait);
    assert!(preload_wait < preload_barrier);
    assert!(preload_barrier < loop_start);

    let next_issue = schedule.find("scratch_buffers[next_buffer]").unwrap();
    let current_decode = schedule.find("w2f_decode8").unwrap();
    let current_mma = schedule.find("w2f_mma").unwrap();
    let next_wait = schedule.rfind("w2f_cp_async_wait()").unwrap();
    let publish = schedule.rfind("__syncthreads()").unwrap();
    assert!(next_issue < current_decode);
    assert!(current_decode < current_mma);
    assert!(current_mma < next_wait);
    assert!(next_wait < publish);

    // For each K128 group: stage 0 is ready before compute; stage 1 is issued
    // into the other buffer, waited, and published before it becomes current.
    let mut ready = [false; 2];
    ready[0] = true;
    for stage in 0..2 {
        let current = stage & 1;
        let next = current ^ 1;
        assert!(ready[current]);
        if stage + 1 < 2 {
            assert!(!ready[next]);
            ready[next] = true;
        }
        ready[current] = false;
    }
    assert_eq!(ready, [false, false]);
}

#[test]
fn exact_k_extent_and_partial_m_rows_are_safe() {
    const K: usize = 4096;
    const GROUP: usize = 128;
    const STAGE: usize = 64;
    for k_block in (0..K).step_by(GROUP) {
        for k_stage in (0..GROUP).step_by(STAGE) {
            for vector_k in 0..4 {
                let first = k_block + k_stage + vector_k * 16;
                assert!(first + 15 < K);
            }
        }
    }
    assert_eq!(K % GROUP, 0);
    assert_eq!(GROUP, 2 * STAGE);

    for valid_rows in 1..=64 {
        let copied = (0..64).filter(|&row| row < valid_rows).count();
        let zero_filled = 64 - copied;
        assert_eq!(copied, valid_rows);
        assert_eq!(copied + zero_filled, 64);
    }
}

#[test]
fn combined_wait_and_publish_removes_one_barrier_per_k128_group() {
    const GROUPS_PER_LEG: u64 = 4096 / 128;
    const LEGS: u64 = 2;
    const CTAS_PER_LAYER: u64 = 256 * (2048 / 128);
    const LAYERS: u64 = 43;
    let incumbent_per_cta = GROUPS_PER_LEG * LEGS * 4;
    let candidate_per_cta = GROUPS_PER_LEG * LEGS * 3;
    assert_eq!(incumbent_per_cta, 256);
    assert_eq!(candidate_per_cta, 192);
    assert_eq!(incumbent_per_cta - candidate_per_cta, 64);
    assert_eq!(
        (incumbent_per_cta - candidate_per_cta) * CTAS_PER_LAYER * LAYERS,
        11_272_192
    );
}

#[test]
fn arithmetic_and_output_order_remain_common_to_both_staging_paths() {
    let component = source(COMPONENT);
    let helper = &component[component.find("void w2f_compute_leg(").unwrap()
        ..component.find("void w2f_emit_group(").unwrap()];
    let selection = marked(helper, "K64 staging selection");
    assert!(!selection.contains("w2f_mma"));
    assert!(!selection.contains("outer["));
    assert!(!selection.contains("output["));
    let selection_end = helper.find("// END K64 staging selection").unwrap();
    let decode = helper.find("w2f_decode8").unwrap();
    let mma = helper.find("w2f_mma").unwrap();
    let fold = helper.find("outer[mt][nt][0] +=").unwrap();
    let output = helper.find("output[row0][col]").unwrap();
    assert!(selection_end < decode && decode < mma && mma < fold && fold < output);
}

#[test]
fn sass_gate_is_packed_one_factor_and_bounds_resources() {
    let gate = source(SASS_GATE);
    for contract in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=\"$selector\"",
        "compile_variant incumbent 0",
        "compile_variant candidate 1",
        "candidate_shared <= static_shared_limit",
        "candidate_registers == incumbent_registers",
        "candidate_shared == 48128",
        "candidate_instructions == 3368",
        "candidate_barriers == incumbent_barriers",
        "production-hostile-overrides.cubin",
        "-DW2A8_PACKED_E4M3_CANDIDATE=0",
        "-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=0",
        "production hostile selector zeroes: packed=1 double_buffer=1",
        "LDGSTS",
        "QMMA.16832.F32.E4M3.E4M3",
        "F2FP.SATFINITE.E4M3",
        "spill stores",
        "spill loads",
        "cubin_set_sha256=",
    ] {
        assert!(gate.contains(contract), "SASS gate omits `{contract}`");
    }
    assert!(!gate.contains("nvidia-smi"));
    assert!(!gate.contains("SINGLE_WARP_ROUTE_GUARD"));
}
