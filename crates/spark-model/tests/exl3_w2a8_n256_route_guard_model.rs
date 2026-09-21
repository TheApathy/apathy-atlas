// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the default-off N256 single-warp route guard.

use std::fs;
use std::path::PathBuf;

const SELECTOR: &str = "W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE";
const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu";
const WRAPPER: &str = concat!(
    "kernels/gb10/deepseek-v4-flash/nvfp4/",
    "exl3_w2a8_grouped_prefill_n256_k2_down.cu"
);
const SASS_GATE: &str = "scripts/check-exl3-prefill-w2a8-n256-route-guard-sass.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    let path = workspace().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("required route-guard input {}: {error}", path.display()))
}

fn compact(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn section<'a>(value: &'a str, name: &str) -> &'a str {
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

fn lane_scan_invalid(offsets: &[i32], total_rows: u32) -> bool {
    let experts = offsets.len() - 1;
    (0..32).any(|lane| {
        (lane..experts).step_by(32).any(|index| {
            let start = offsets[index];
            let end = offsets[index + 1];
            start < 0
                || end < start
                || end as u32 > total_rows
                || (index == 0 && start != 0)
                || (index + 1 == experts && end != total_rows as i32)
        })
    })
}

fn incumbent_invalid(offsets: &[i32], total_rows: u32) -> bool {
    (0..16).any(|_| lane_scan_invalid(offsets, total_rows))
}

fn candidate_invalid(offsets: &[i32], total_rows: u32) -> bool {
    lane_scan_invalid(offsets, total_rows)
}

#[test]
fn selector_is_strict_default_off_and_absent_from_production_wrapper() {
    let component = source(COMPONENT);
    let wrapper = source(WRAPPER);
    assert!(component.contains(&format!("#ifndef {SELECTOR}")));
    assert!(component.contains(&format!("#define {SELECTOR} 0")));
    assert!(component.contains(&format!("static_assert({SELECTOR} == 0 ||")));
    assert!(component.contains(&format!("{SELECTOR} == 1,")));
    assert_eq!(component.matches(&format!("#if {SELECTOR}")).count(), 1);
    assert!(
        !wrapper.contains(SELECTOR),
        "the production include wrapper must retain the default-off experiment"
    );
}

#[test]
fn candidate_publishes_one_warp_decision_before_any_return_or_output_work() {
    let component = source(COMPONENT);
    let candidate = section(&component, "single-warp route guard candidate");
    let flat = compact(candidate);
    for contract in [
        "__shared__ unsigned int route_invalid_block",
        "threadIdx.x < 32",
        "__ballot_sync(0xffffffffu, routing_invalid)",
        "threadIdx.x == 0",
        "route_invalid_block = invalid_mask != 0",
        "__syncthreads()",
        "if (route_invalid_block != 0) return",
    ] {
        assert!(candidate.contains(contract), "candidate omits `{contract}`");
    }
    assert!(flat.contains("if(threadIdx.x<32)"));
    assert!(
        flat.contains("w2a8_route_lane_invalid(threadIdx.x,num_experts,total_rows,expert_offsets)")
    );

    let ballot = candidate.find("__ballot_sync").unwrap();
    let publication = candidate.find("route_invalid_block =").unwrap();
    let synchronization = candidate.find("__syncthreads()").unwrap();
    let reject = candidate
        .find("if (route_invalid_block != 0) return")
        .unwrap();
    assert!(ballot < publication);
    assert!(publication < synchronization);
    assert!(synchronization < reject);

    let guarded = section(&component, "route validation selection");
    let selection_end = component.find("// END route validation selection").unwrap();
    let output_work = component.find("const unsigned int n_tile =").unwrap();
    assert!(selection_end < output_work);
    assert!(!guarded.contains("C["));
    assert!(!guarded.contains("trellis_tab[expert_id]"));
}

#[test]
fn incumbent_keeps_the_original_every_warp_fail_closed_path() {
    let component = source(COMPONENT);
    let incumbent = section(&component, "every-warp route guard incumbent");
    let flat = compact(incumbent);
    assert!(flat.contains("constunsignedintroute_lane=threadIdx.x&31"));
    assert!(flat.contains(
        "boolrouting_invalid=w2a8_route_lane_invalid(route_lane,num_experts,total_rows,expert_offsets)"
    ));
    assert!(flat.contains("if(__ballot_sync(0xffffffffu,routing_invalid)!=0)return"));
    assert!(!incumbent.contains("__syncthreads"));
    assert!(!incumbent.contains("route_invalid_block"));
}

#[test]
fn one_warp_striping_covers_every_route_exactly_once() {
    for experts in 1..=513 {
        let mut visits = vec![0_u8; experts];
        for lane in 0..32 {
            for index in (lane..experts).step_by(32) {
                visits[index] += 1;
            }
        }
        assert!(visits.iter().all(|&count| count == 1));
    }
}

#[test]
fn exact_k2_down_geometry_removes_fifteen_redundant_route_scans() {
    const EXPERTS: u64 = 256;
    const N_TILES: u64 = 4096 / 256;
    const CTAS_PER_LAYER: u64 = EXPERTS * N_TILES;
    const LAYERS: u64 = 43;
    const OFFSET_LOADS_PER_SCAN: u64 = 2 * EXPERTS;
    let incumbent_loads = 16 * OFFSET_LOADS_PER_SCAN * CTAS_PER_LAYER * LAYERS;
    let candidate_loads = OFFSET_LOADS_PER_SCAN * CTAS_PER_LAYER * LAYERS;
    assert_eq!(CTAS_PER_LAYER, 4_096);
    assert_eq!(incumbent_loads, 1_442_840_576);
    assert_eq!(candidate_loads, 90_177_536);
    assert_eq!(incumbent_loads - candidate_loads, 1_352_663_040);
    assert_eq!(CTAS_PER_LAYER * LAYERS, 176_128);
}

#[test]
fn candidate_and_sixteen_warp_incumbent_agree_for_arbitrary_routes() {
    let mut state = 0x9e37_79b9_u32;
    for experts in 1..=256 {
        for _case in 0..32 {
            let total_rows = (state % 16_385) as i32;
            let mut offsets = Vec::with_capacity(experts + 1);
            for _ in 0..=experts {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                offsets.push((state % 32_770) as i32 - 16_384);
            }
            assert_eq!(
                candidate_invalid(&offsets, total_rows as u32),
                incumbent_invalid(&offsets, total_rows as u32),
                "decision drift for experts={experts} offsets={offsets:?}"
            );
        }
    }
}

#[test]
fn malformed_first_and_late_routes_preserve_poisoned_output() {
    let valid = [0, 3, 3, 8, 13];
    assert!(!candidate_invalid(&valid, 13));
    let cases = [
        [-1, 3, 3, 8, 13],
        [1, 3, 3, 8, 13],
        [0, 3, 3, 8, 12],
        [0, 3, 3, 14, 13],
        [0, 3, 3, 8, 14],
    ];
    for offsets in cases {
        let mut incumbent = [0xa5_u8; 64];
        let mut candidate = incumbent;
        if !incumbent_invalid(&offsets, 13) {
            incumbent.fill(0);
        }
        if !candidate_invalid(&offsets, 13) {
            candidate.fill(0);
        }
        assert_eq!(candidate, incumbent);
        assert_eq!(candidate, [0xa5_u8; 64], "malformed route wrote output");
    }
}

#[test]
fn sass_gate_is_packed_one_factor_and_bounds_candidate_resources() {
    let gate = source(SASS_GATE);
    for contract in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=\"$selector\"",
        "compile_variant incumbent 0",
        "compile_variant candidate 1",
        "--dump-resource-usage",
        "-arch=sm_121a",
        "--fmad=false",
        "candidate_registers == incumbent_registers",
        "candidate_shared == incumbent_shared + 16",
        "candidate_shared <= static_shared_limit",
        "candidate_barriers == incumbent_barriers + 1",
        "spill stores",
        "spill loads",
        "cubin_set_sha256=",
    ] {
        assert!(gate.contains(contract), "SASS gate omits `{contract}`");
    }
    assert!(!gate.contains("nvidia-smi"));
    assert!(!gate.contains("ATLAS_EXL3_PREFILL_W2A8_FUSED_GU"));
}
