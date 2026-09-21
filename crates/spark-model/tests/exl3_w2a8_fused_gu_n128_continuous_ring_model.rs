// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the promoted fused-N128 continuous K64 ring.

use std::fs;
use std::path::PathBuf;

const SELECTOR: &str = "W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE";
const DOUBLE_BUFFER: &str = "W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE";
const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";
const WRAPPER: &str = concat!(
    "kernels/gb10/deepseek-v4-flash/nvfp4/",
    "exl3_w2a8_fused_gu_down_emit_n128.cu"
);
const SASS_GATE: &str = "scripts/check-exl3-prefill-w2a8-fused-gu-n128-continuous-ring-sass.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    let path = workspace().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("required continuous-ring input {}: {error}", path.display())
    })
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
    assert!(component.contains(&format!("#ifndef {SELECTOR}")));
    assert!(component.contains(&format!("#define {SELECTOR} 0")));
    assert!(component.contains(&format!("static_assert({SELECTOR} == 0 ||")));
    assert!(component.contains(&format!("{SELECTOR} == 1,")));
    assert!(component.contains(&format!("#if {SELECTOR}")));
    assert!(component.contains(&format!("static_assert({DOUBLE_BUFFER} == 1,")));

    let wrapper = source(WRAPPER);
    assert!(wrapper.contains(&format!("#define {DOUBLE_BUFFER} 1")));
    assert!(wrapper.contains(&format!("#ifdef {SELECTOR}")));
    assert!(wrapper.contains(&format!("#undef {SELECTOR}")));
    assert!(wrapper.contains(&format!("#define {SELECTOR} 1")));
    assert!(!wrapper.contains(&format!("#ifndef {SELECTOR}")));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    Empty,
    Pending(usize),
    Ready(usize),
}

#[test]
fn sixty_four_stages_form_one_safe_alternating_ring_per_leg() {
    const STAGES: usize = 4096 / 64;
    let mut slots = [Slot::Empty; 2];
    let mut consumed = Vec::new();
    let mut folds = Vec::new();
    let mut issues = 0;
    let mut waits = 0;
    let mut barriers = 0;
    let mut overlapped = 0;

    slots[0] = Slot::Pending(0);
    issues += 1;
    slots[0] = Slot::Ready(0);
    waits += 1;
    barriers += 1;

    for stage in 0..STAGES {
        let current = stage & 1;
        let next_buffer = current ^ 1;
        assert_eq!(slots[current], Slot::Ready(stage));

        if stage + 1 < STAGES {
            assert_eq!(slots[next_buffer], Slot::Empty);
            slots[next_buffer] = Slot::Pending(stage + 1);
            issues += 1;
            overlapped += 1;
        }

        consumed.push(stage);
        slots[current] = Slot::Empty;

        if stage + 1 < STAGES {
            assert_eq!(slots[next_buffer], Slot::Pending(stage + 1));
            slots[next_buffer] = Slot::Ready(stage + 1);
            waits += 1;
        }
        barriers += 1;

        if stage & 1 == 1 {
            assert_eq!(&consumed[stage - 1..=stage], &[stage - 1, stage]);
            folds.push(stage / 2);
        }
    }

    assert_eq!(consumed, (0..STAGES).collect::<Vec<_>>());
    assert_eq!(folds, (0..STAGES / 2).collect::<Vec<_>>());
    assert_eq!(slots, [Slot::Empty, Slot::Empty]);
    assert_eq!(issues, 64);
    assert_eq!(waits, 64);
    assert_eq!(overlapped, 63);
    assert_eq!(barriers, 65);

    const LEGS: usize = 2;
    assert_eq!(LEGS * overlapped, 126);
    assert_eq!(LEGS * barriers, 130);
}

#[test]
fn source_orders_issue_compute_wait_publish_and_scale_fold() {
    let component = source(COMPONENT);
    let helper = &component[component.find("void w2f_compute_leg(").unwrap()
        ..component.find("void w2f_emit_group(").unwrap()];
    let initial = marked(helper, "continuous K64 ring initial publication");
    let schedule = marked(helper, "continuous K64 ring schedule");
    let flat = compact(schedule);

    assert!(compact(initial).contains(
        "w2f_stage_async(scratch_buffers[0],activation,trellis,m_start,m_end,m_local,n_base,0)"
    ));
    let initial_issue = initial.find("w2f_stage_async").unwrap();
    let initial_commit = initial.find("w2f_cp_async_commit()").unwrap();
    let initial_wait = initial.find("w2f_cp_async_wait()").unwrap();
    let initial_publish = initial.find("__syncthreads()").unwrap();
    assert!(initial_issue < initial_commit);
    assert!(initial_commit < initial_wait);
    assert!(initial_wait < initial_publish);

    for contract in [
        "absolute_stage",
        "current_buffer",
        "next_buffer",
        "next_k < W2F_GATE_UP_K",
        "scratch_buffers[next_buffer]",
        "scratch_buffers[current_buffer]",
        "w2f_cp_async_commit()",
        "w2f_cp_async_wait()",
        "__syncthreads()",
    ] {
        assert!(
            schedule.contains(contract),
            "ring schedule omits `{contract}`"
        );
    }
    assert!(flat.contains("current_buffer=absolute_stage&1"));
    assert!(flat.contains("next_buffer=current_buffer^1"));
    for expression in [
        "absolute_k=k_block+k_stage",
        "absolute_stage=absolute_k/W2F_K_STAGE",
        "next_k=absolute_k+W2F_K_STAGE",
        "if(next_k<W2F_GATE_UP_K)w2f_cp_async_wait();",
    ] {
        assert!(flat.contains(expression), "ring omits exact `{expression}`");
    }

    let next_issue = schedule.find("scratch_buffers[next_buffer]").unwrap();
    let decode = schedule.find("w2f_decode8").unwrap();
    let mma = schedule.find("w2f_mma").unwrap();
    let wait = schedule.rfind("w2f_cp_async_wait()").unwrap();
    let publish = schedule.rfind("__syncthreads()").unwrap();
    assert!(next_issue < decode && decode < mma);
    assert!(mma < wait && wait < publish);

    let schedule_end = helper.find("// END continuous K64 ring schedule").unwrap();
    let fold = helper.find("outer[mt][nt][0] +=").unwrap();
    let output = helper.find("output[row0][col]").unwrap();
    assert!(schedule_end < fold && fold < output);
}

#[test]
fn exact_k_extent_and_every_partial_m_tail_keep_zero_fill() {
    const K: usize = 4096;
    const STAGE: usize = 64;
    for next_k in (0..K).step_by(STAGE) {
        for vector_k in 0..4 {
            assert!(next_k + vector_k * 16 + 15 < K);
        }
        for k_tile in 0..4 {
            assert!((next_k / 16) + k_tile < K / 16);
        }
    }

    for valid_rows in 1..=64 {
        let copied = (0..256).filter(|vector| vector / 4 < valid_rows).count();
        assert_eq!(copied, valid_rows * 4);
        assert_eq!(256 - copied, (64 - valid_rows) * 4);
    }

    let component = source(COMPONENT);
    let stage = marked(&component, "double-buffer K64 async stage");
    assert!(stage.contains("unsigned int valid_bytes = 0"));
    assert!(stage.contains("if (row < m_end)"));
    assert!(stage.contains("valid_bytes = 16"));
}

#[test]
fn sass_gate_freezes_one_factor_arithmetic_and_resource_ceilings() {
    let gate = source(SASS_GATE);
    for contract in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1",
        "-DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1",
        "-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=\"$selector\"",
        "exl3_w2a8_composed_n128_n256_probe.cu",
        "compile_variant incumbent 0",
        "compile_variant candidate 1",
        "experiment_wrapper=\"$probe_dir/continuous-ring-experiment-wrapper.cu\"",
        "production-hostile-ring0.cubin",
        "production hostile ring zero: continuous_ring=1",
        "production_instruction_expected=2784",
        "production_sass_sha256",
        "[[ $production_sass_sha256 == \"$forced_candidate_sass_sha256\" ]]",
        "register_limit=128",
        "shared_limit=48128",
        "instruction_limit=2800",
        "candidate_registers <= register_limit",
        "candidate_shared <= shared_limit",
        "candidate_instructions <= instruction_limit",
        "[[ $candidate_shared == \"$incumbent_shared\" ]]",
        "wrapper_sha256_expected=4bd2a209b195065d63042a50183e59e60f6773524beed07fc56e4c105e8725b2",
        "compute_body_sha256_expected=7b0773744b91b8eac4b4d290ca1c3f608aa45670304678b4206624b115a0933e",
        "[[ $(metric incumbent qmma) == 32 ]]",
        "[[ $(metric incumbent bf16_rounds) == 52 ]]",
        "[[ $(metric incumbent fadd) == 116 ]]",
        "[[ $(metric incumbent ffma) == 166 ]]",
        "[[ $(metric incumbent zfill) == 2 ]]",
        "[[ $(metric candidate zfill) == 4 ]]",
        "arithmetic census exact",
        "stack=0 local=0 spills=0 atomics=0",
    ] {
        assert!(gate.contains(contract), "SASS gate omits `{contract}`");
    }
    assert!(!gate.contains("nvidia-smi"));
}
