// SPDX-License-Identifier: AGPL-3.0-only
//! CPU contract only; row invariance requires the actual GPU operator probe.
#[allow(dead_code)]
#[path = "../src/cublaslt/diagnostic_contract.rs"]
mod contract;
use contract::{HeuristicResult, ReductionPolicy};

#[test]
fn baseline_does_not_set_a_reduction_preference() {
    assert_eq!(ReductionPolicy::Baseline.preference_mask(), None);
    for (split, reduction) in [(0, 0), (1, 0), (8, 1), (8, 2), (8, 4)] {
        assert!(
            ReductionPolicy::Baseline
                .validate_selected(split, reduction)
                .is_ok()
        );
    }
}

#[test]
fn compute_type_only_excludes_both_output_type_reductions() {
    assert_eq!(ReductionPolicy::ComputeTypeOnly.preference_mask(), Some(2));
    assert!(
        ReductionPolicy::ComputeTypeOnly
            .validate_selected(1, 0)
            .is_ok()
    );
    assert!(
        ReductionPolicy::ComputeTypeOnly
            .validate_selected(8, 2)
            .is_ok()
    );
    for (split, reduction) in [(8, 1), (8, 4), (8, 0), (-1, 2), (8, 3), (8, 8)] {
        assert!(
            ReductionPolicy::ComputeTypeOnly
                .validate_selected(split, reduction)
                .is_err()
        );
    }
}

#[test]
fn heuristic_storage_matches_local_cuda_13_c_abi() {
    assert_eq!(std::mem::size_of::<HeuristicResult>(), 96);
    assert_eq!(std::mem::align_of::<HeuristicResult>(), 8);
    assert_eq!(std::mem::offset_of!(HeuristicResult, algo), 0);
    assert_eq!(std::mem::offset_of!(HeuristicResult, workspace_bytes), 64);
    assert_eq!(std::mem::offset_of!(HeuristicResult, state), 72);
    assert_eq!(std::mem::offset_of!(HeuristicResult, waves), 76);
}

#[test]
fn heuristic_must_be_successful_finite_and_within_actual_workspace() {
    let mut result = HeuristicResult::default();
    assert!(result.admit(1, 64).is_ok());
    assert!(result.admit(0, 64).is_err());
    result.state = 7;
    assert!(result.admit(1, 64).is_err());
    result.state = 0;
    result.workspace_bytes = 65;
    assert!(result.admit(1, 64).is_err());
    result.workspace_bytes = 64;
    for waves in [f32::NAN, f32::INFINITY, -1.0] {
        result.waves = waves;
        assert!(result.admit(1, 64).is_err());
    }
}
