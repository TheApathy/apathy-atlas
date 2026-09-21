// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the production N256-down continuous K64 ring.

use std::fs;
use std::path::PathBuf;

const SELECTOR: &str = "W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE";
const DOUBLE_BUFFER: &str = "W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE";
const PACKED_E4M3: &str = "W2A8_PACKED_E4M3_CANDIDATE";
const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu";
const WRAPPER: &str = concat!(
    "kernels/gb10/deepseek-v4-flash/nvfp4/",
    "exl3_w2a8_grouped_prefill_n256_k2_down.cu"
);
const SASS_GATE: &str = "scripts/check-exl3-prefill-w2a8-n256-down-continuous-ring-sass.sh";
const AB_BUILDER: &str =
    "scripts/check-exl3-prefill-w2a8-n256-down-continuous-ring-ab-probe-build.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    let path = workspace().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("required N256 ring input {}: {error}", path.display()))
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
fn raw_selector_is_default_off_and_production_forces_the_qualified_ring() {
    let component = source(COMPONENT);
    assert!(component.contains(&format!("#ifndef {SELECTOR}")));
    assert!(component.contains(&format!("#define {SELECTOR} 0")));
    assert!(component.contains(&format!("static_assert({SELECTOR} == 0 ||")));
    assert!(component.contains(&format!("{SELECTOR} == 1,")));
    assert!(component.contains(&format!("#if {SELECTOR}")));
    assert!(component.contains(&format!("static_assert({DOUBLE_BUFFER} == 1,")));
    assert!(component.contains(&format!("static_assert({PACKED_E4M3} == 1,")));
    assert!(component.contains("N256 continuous ring is an exact down-shape experiment"));
    let wrapper = source(WRAPPER);
    let production_guard =
        format!("#ifdef {SELECTOR}\n#undef {SELECTOR}\n#endif\n#define {SELECTOR} 1");
    assert_eq!(wrapper.matches(SELECTOR).count(), 3);
    assert!(wrapper.contains(&production_guard));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    Empty,
    Pending(usize),
    Ready(usize),
}

#[test]
fn thirty_two_stages_form_one_safe_ring_with_sixteen_scale_folds() {
    const STAGES: usize = 2048 / 64;
    let mut slots = [Slot::Empty; 2];
    let mut consumed = Vec::new();
    let mut folds = Vec::new();
    let mut issues = 0;
    let mut commits = 0;
    let mut waits = 0;
    let mut barriers = 0;
    let mut overlapped = 0;

    slots[0] = Slot::Pending(0);
    issues += 1;
    commits += 1;
    slots[0] = Slot::Ready(0);
    waits += 1;
    barriers += 1;

    for stage in 0..STAGES {
        let current = stage & 1;
        let next = current ^ 1;
        assert_eq!(slots[current], Slot::Ready(stage));
        if stage + 1 < STAGES {
            assert_eq!(slots[next], Slot::Empty);
            slots[next] = Slot::Pending(stage + 1);
            issues += 1;
            commits += 1;
            overlapped += 1;
        }

        consumed.push(stage);
        slots[current] = Slot::Empty;
        if stage + 1 < STAGES {
            assert_eq!(slots[next], Slot::Pending(stage + 1));
            slots[next] = Slot::Ready(stage + 1);
            waits += 1;
        }
        barriers += 1;
        if stage & 1 == 1 {
            folds.push(stage / 2);
        }
    }

    assert_eq!(consumed, (0..STAGES).collect::<Vec<_>>());
    assert_eq!(folds, (0..16).collect::<Vec<_>>());
    assert_eq!(slots, [Slot::Empty, Slot::Empty]);
    assert_eq!(
        (issues, commits, waits, overlapped, barriers),
        (32, 32, 32, 31, 33)
    );
}

#[test]
fn source_orders_issue_compute_wait_publish_across_k128_boundaries() {
    let component = source(COMPONENT);
    let initial = marked(
        &component,
        "N256 down continuous K64 ring initial publication",
    );
    let schedule = marked(&component, "N256 down continuous K64 ring schedule");
    let flat = compact(schedule);

    let initial_issue = initial.find("w2a8_stage_async").unwrap();
    let initial_commit = initial.find("w2a8_cp_async_commit()").unwrap();
    let initial_wait = initial.find("w2a8_cp_async_wait()").unwrap();
    let initial_publish = initial.find("__syncthreads()").unwrap();
    assert!(initial_issue < initial_commit);
    assert!(initial_commit < initial_wait);
    assert!(initial_wait < initial_publish);

    for expression in [
        "absolute_k=k_block+k_stage",
        "absolute_stage=absolute_k/W2A8_K_STAGE",
        "current_buffer=absolute_stage&1",
        "next_buffer=current_buffer^1",
        "next_k=absolute_k+W2A8_K_STAGE",
        "if(next_k<W2A8_FIXED_K)",
        "scratch_buffers[next_buffer]",
        "scratch_buffers[current_buffer]",
        "if(next_k<W2A8_FIXED_K)w2a8_cp_async_wait();",
    ] {
        assert!(flat.contains(expression), "ring omits exact `{expression}`");
    }

    let issue = schedule.find("scratch_buffers[next_buffer]").unwrap();
    let decode = schedule.find("w2a8_decode8").unwrap();
    let mma = schedule.find("w2a8_mma").unwrap();
    let wait = schedule.rfind("w2a8_cp_async_wait()").unwrap();
    let publish = schedule.rfind("__syncthreads()").unwrap();
    assert!(issue < decode && decode < mma);
    assert!(mma < wait && wait < publish);

    let schedule_end = component
        .find("// END N256 down continuous K64 ring schedule")
        .unwrap();
    let fold = component.find("// BEGIN scale fold").unwrap();
    let output = component.find("__nv_bfloat16* out = C").unwrap();
    assert!(schedule_end < fold && fold < output);
}

#[test]
fn exact_k_extent_and_every_partial_m_tail_keep_zero_fill() {
    const K: usize = 2048;
    for absolute_k in (0..K).step_by(64) {
        for vector_k in 0..4 {
            assert!(absolute_k + vector_k * 16 + 15 < K);
        }
        for k_tile in 0..4 {
            assert!((absolute_k / 16) + k_tile < K / 16);
        }
    }
    for valid_rows in 1..=64 {
        let copied = (0..256).filter(|vector| vector / 4 < valid_rows).count();
        assert_eq!(copied, valid_rows * 4);
        assert_eq!(256 - copied, (64 - valid_rows) * 4);
    }
    assert_eq!(32 * 256, 8192, "all T owners issue once per K64 stage");

    let component = source(COMPONENT);
    let stage = marked(&component, "N256 down double-buffer K64 async stage");
    assert!(stage.contains("unsigned int valid_bytes"));
    assert!(stage.contains("valid_bytes = 0"));
    assert!(stage.contains("if (row < m_end)"));
    assert!(stage.contains("valid_bytes = 16"));
}

#[test]
fn future_static_and_native_gates_freeze_a_single_compile_factor() {
    let sass = source(SASS_GATE);
    for contract in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1",
        "-DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=\"$selector\"",
        "compile_variant incumbent 0",
        "compile_variant candidate 1",
        "register_limit=104",
        "shared_limit=19456",
        "continuous_ring_sass=PASS",
        "production hostile ring zero: continuous_ring=1",
    ] {
        assert!(sass.contains(contract), "SASS gate omits `{contract}`");
    }

    let builder = source(AB_BUILDER);
    for contract in [
        "only_ab_compile_factor=W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE",
        "fixed_n256_down_double_buffer=1",
        "fixed_packed_e4m3=1",
        "W2A8_N256_DOWN_CONTINUOUS_RING_AB_OUTPUT_DIR",
        "continuous_ring_speedup",
        "run-n256-down-continuous-ring-ab.sh",
        "runtime_device_identity=measured_all_four_runs",
        "printf '%s\\n' \"${device_lines[0]}\"",
    ] {
        assert!(builder.contains(contract), "A/B builder omits `{contract}`");
    }
}
