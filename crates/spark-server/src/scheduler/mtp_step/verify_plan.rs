// SPDX-License-Identifier: AGPL-3.0-only

//! One policy for bootstrap and steady-state speculative verification.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyPath {
    Generic,
    K2,
    K3,
    K4,
}

pub(super) fn select(
    requires_full_verify: bool,
    bootstrap: bool,
    draft_count: usize,
    requested: usize,
    generic_floor: usize,
) -> Option<VerifyPath> {
    if draft_count == 0 {
        None
    } else if requires_full_verify || (!bootstrap && draft_count >= generic_floor) {
        Some(VerifyPath::Generic)
    } else if requested >= 3 && draft_count >= 3 {
        Some(VerifyPath::K4)
    } else if requested >= 2 && draft_count >= 2 {
        Some(VerifyPath::K3)
    } else {
        Some(VerifyPath::K2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_dspark_always_gets_capture_and_compressor_restore() {
        for bootstrap in [true, false] {
            for drafts in 1..=5 {
                for floor in [1, 4, usize::MAX] {
                    assert_eq!(
                        select(true, bootstrap, drafts, 5, floor),
                        Some(VerifyPath::Generic)
                    );
                }
            }
        }
    }

    #[test]
    fn grammar_shortening_does_not_change_required_target_state_path() {
        assert_eq!(select(true, true, 1, 1, 4), Some(VerifyPath::Generic));
        assert_eq!(select(true, false, 1, 5, 4), Some(VerifyPath::Generic));
    }

    #[test]
    fn empty_drafts_are_not_verified() {
        for required in [true, false] {
            for bootstrap in [true, false] {
                assert_eq!(select(required, bootstrap, 0, 5, 0), None);
            }
        }
    }

    #[test]
    fn legacy_bootstrap_specializations_are_unchanged() {
        assert_eq!(select(false, true, 1, 5, 1), Some(VerifyPath::K2));
        assert_eq!(select(false, true, 2, 5, 1), Some(VerifyPath::K3));
        assert_eq!(select(false, true, 5, 5, 1), Some(VerifyPath::K4));
        assert_eq!(select(false, true, 5, 1, 1), Some(VerifyPath::K2));
    }

    #[test]
    fn legacy_steady_state_respects_existing_floor() {
        for (drafts, path) in [
            (1, VerifyPath::K2),
            (2, VerifyPath::K3),
            (3, VerifyPath::K4),
            (4, VerifyPath::Generic),
        ] {
            assert_eq!(select(false, false, drafts, 5, 4), Some(path));
        }
        assert_eq!(select(false, false, 1, 1, 1), Some(VerifyPath::Generic));
    }
}
