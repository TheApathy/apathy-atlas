// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the default-off N256-down K64 double buffer.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

const SELECTOR: &str = "W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE";
const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu";
const WRAPPER: &str = concat!(
    "kernels/gb10/deepseek-v4-flash/nvfp4/",
    "exl3_w2a8_grouped_prefill_n256_k2_down.cu"
);
const REGISTRY: &str = "kernels/gb10/deepseek-v4-flash/MODEL.toml";
const HOST_ROUTE: &str = "crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs";
const SASS_GATE: &str = "scripts/check-exl3-prefill-w2a8-n256-down-double-buffer-sass.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    let path = workspace().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "required N256-down double-buffer input {}: {error}",
            path.display()
        )
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
fn selector_is_strict_default_off_down_only_and_forced_on_by_production_wrapper() {
    let component = source(COMPONENT);
    assert!(component.contains(&format!("#ifndef {SELECTOR}")));
    assert!(component.contains(&format!("#define {SELECTOR} 0")));
    assert!(component.contains(&format!("static_assert({SELECTOR} == 0 ||")));
    assert!(component.contains(&format!("{SELECTOR} == 1,")));
    assert!(component.contains("static_assert(W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048,"));

    let wrapper = source(WRAPPER);
    assert!(wrapper.contains(&format!("#ifdef {SELECTOR}")));
    assert!(wrapper.contains(&format!("#undef {SELECTOR}")));
    assert!(wrapper.contains(&format!("#define {SELECTOR} 1")));
    assert!(!wrapper.contains(&format!("#ifndef {SELECTOR}")));

    for relative in [REGISTRY, HOST_ROUTE] {
        assert!(
            !source(relative).contains(SELECTOR),
            "compile-time selector leaked into runtime route {relative}"
        );
    }
}

#[test]
fn every_thread_owns_one_unique_activation_or_trellis_copy() {
    let component = source(COMPONENT);
    let stage = marked(&component, "N256 down double-buffer K64 async stage");
    let flat = compact(stage);
    for contract in [
        "cp.async.ca.shared.global",
        "__cvta_generic_to_shared",
        "threadIdx.x < W2A8_A_VECTORS",
        "threadIdx.x - W2A8_A_VECTORS",
        "scratch.activation[row_local][vector_k]",
        "scratch.trellis[k_tile][word]",
        "unsigned int valid_bytes",
        "valid_bytes = 16",
    ] {
        assert!(stage.contains(contract), "async stage omits `{contract}`");
    }
    assert!(flat.contains("source=activation"));
    assert!(flat.contains("valid_bytes=0"));

    let mut activation = HashSet::new();
    let mut trellis = HashSet::new();
    for thread in 0..512 {
        if thread < 256 {
            assert!(activation.insert((thread >> 2, thread & 3)));
        } else {
            let load = thread - 256;
            assert!(trellis.insert((load / 64, load % 64)));
        }
    }
    assert_eq!(activation.len(), 64 * 4);
    assert_eq!(trellis.len(), 4 * 64);
    assert!(
        activation
            .iter()
            .all(|&(row, vector)| row < 64 && vector < 4)
    );
    assert!(
        trellis
            .iter()
            .all(|&(k_tile, word)| k_tile < 4 && word < 64)
    );
}

#[test]
fn two_scratch_stages_fit_the_source_and_cuobjdump_ceilings() {
    const ACTIVATION: usize = 64 * 5 * 16;
    const TRELLIS: usize = 4 * 64 * 16;
    const ONE_STAGE: usize = ACTIVATION + TRELLIS;
    assert_eq!(ONE_STAGE, 9 * 1024);
    assert_eq!(2 * ONE_STAGE, 18 * 1024);
    assert!(2 * ONE_STAGE <= 19 * 1024);

    let component = source(COMPONENT);
    assert!(component.contains("sizeof(W2A8GemmScratch) == 9 * 1024"));
    assert!(component.contains("W2A8GemmScratch scratch_buffers[2]"));
}

#[test]
fn all_sixteen_k128_groups_close_both_buffer_lifetimes_in_order() {
    const K: usize = 2048;
    const GROUP: usize = 128;
    const STAGE: usize = 64;
    assert_eq!(GROUP, 2 * STAGE);
    let mut consumed = Vec::new();
    let mut barriers = 0;
    let mut commits = 0;
    let mut waits = 0;

    // 0=empty, 1=async pending, 2=published and safe to consume.
    let mut state = [0_u8; 2];
    for group in 0..K / GROUP {
        assert_eq!(state, [0, 0]);
        state[0] = 1;
        commits += 1;
        state[0] = 2;
        waits += 1;
        barriers += 1;

        state[1] = 1;
        commits += 1;
        assert_eq!(state[0], 2);
        consumed.push(2 * group);
        state[0] = 0;
        state[1] = 2;
        waits += 1;
        barriers += 1;

        assert_eq!(state[1], 2);
        consumed.push(2 * group + 1);
        state[1] = 0;
        barriers += 1;
    }

    assert_eq!(consumed, (0..32).collect::<Vec<_>>());
    assert_eq!(commits, 32);
    assert_eq!(waits, 32);
    assert_eq!(barriers, 16 * 3);
    assert_eq!(16 * 4 - barriers, 16);
}

#[test]
fn source_schedule_overlaps_only_stage_one_then_publishes_before_reuse() {
    let component = source(COMPONENT);
    let schedule = marked(&component, "N256 down double-buffer K128 schedule");
    let flat = compact(schedule);
    for contract in [
        "w2a8_stage_async(scratch_buffers[0]",
        "w2a8_cp_async_commit()",
        "w2a8_cp_async_wait()",
        "scratch_buffers[1]",
        "scratch_buffers[current_buffer]",
        "k_stage == 0",
    ] {
        assert!(schedule.contains(contract), "schedule omits `{contract}`");
    }
    assert!(flat.contains("current_buffer=k_stage/W2A8_K_STAGE"));

    let preload = schedule
        .find("w2a8_stage_async(scratch_buffers[0]")
        .unwrap();
    let preload_commit = schedule[preload..].find("w2a8_cp_async_commit()").unwrap() + preload;
    let preload_wait = schedule[preload_commit..]
        .find("w2a8_cp_async_wait()")
        .unwrap()
        + preload_commit;
    let preload_barrier = schedule[preload_wait..].find("__syncthreads()").unwrap() + preload_wait;
    let loop_start = schedule.find("for (unsigned int k_stage").unwrap();
    assert!(preload < preload_commit);
    assert!(preload_commit < preload_wait);
    assert!(preload_wait < preload_barrier);
    assert!(preload_barrier < loop_start);

    let next_issue = schedule.find("scratch_buffers[1]").unwrap();
    let decode = schedule.find("w2a8_decode8").unwrap();
    let mma = schedule.find("w2a8_mma").unwrap();
    let next_wait = schedule.rfind("w2a8_cp_async_wait()").unwrap();
    let publication = schedule.rfind("__syncthreads()").unwrap();
    assert!(next_issue < decode && decode < mma);
    assert!(mma < next_wait && next_wait < publication);
}

#[test]
fn exact_k_extent_and_every_partial_m_tail_zero_fill_are_safe() {
    const K: usize = 2048;
    for k_block in (0..K).step_by(128) {
        for k_stage in [0, 64] {
            for vector_k in 0..4 {
                let first = k_block + k_stage + vector_k * 16;
                assert!(first + 15 < K);
            }
            for k_tile in 0..4 {
                let kb = (k_block + k_stage) / 16 + k_tile;
                assert!(kb < K / 16);
            }
        }
    }
    assert_eq!(K / 128, 16);

    for valid_rows in 1..=64 {
        let mut copied = 0;
        let mut zero_filled = 0;
        for vector in 0..256 {
            if vector / 4 < valid_rows {
                copied += 1;
            } else {
                zero_filled += 1;
            }
        }
        assert_eq!(copied, valid_rows * 4);
        assert_eq!(copied + zero_filled, 256);
    }
}

#[test]
fn decode_mma_scale_fold_and_output_order_are_common_to_both_staging_arms() {
    let component = source(COMPONENT);
    let selection = marked(&component, "K64 staging selection");
    assert!(!selection.contains("w2a8_decode8"));
    assert!(!selection.contains("w2a8_mma"));
    assert!(!selection.contains("outer["));
    assert!(!selection.contains("out["));

    let kernel = &component[component.find("extern \"C\" __global__").unwrap()..];
    let selection_end = kernel.find("// END K64 staging selection").unwrap();
    let decode = kernel.find("w2a8_decode8").unwrap();
    let mma = kernel.find("w2a8_mma").unwrap();
    let fold = kernel.find("outer[mt][nt][0] +=").unwrap();
    let output = kernel.find("out[0] = __float2bfloat16").unwrap();
    assert!(selection_end < decode && decode < mma && mma < fold && fold < output);
}

#[test]
fn sass_gate_is_one_factor_and_enforces_async_and_resource_contracts() {
    let gate = source(SASS_GATE);
    for contract in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0",
        "-DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=\"$selector\"",
        "compile_variant incumbent 0",
        "compile_variant candidate 1",
        "register_limit=104",
        "shared_limit=19456",
        "instruction_limit=920",
        "candidate_registers <= register_limit",
        "candidate_shared <= shared_limit",
        "candidate_instructions <= instruction_limit",
        "production-hostile-overrides.cubin",
        "production hostile selector zeroes: packed=1 n256_down_double_buffer=1",
        "candidate ldgsts",
        "LDGDEPBAR",
        "DEPBAR\\.LE SB0, 0x0",
        "overlap_qmma != 16",
        "QMMA\\.16832\\.F32\\.E4M3\\.E4M3",
        "spill stores",
        "spill loads",
        "invalid-selector.cubin",
        "invalid-gu.cubin",
        "cubin_set_sha256=",
    ] {
        assert!(gate.contains(contract), "SASS gate omits `{contract}`");
    }
    assert!(!gate.contains("nvidia-smi"));
    assert!(!gate.contains("W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE"));
}
