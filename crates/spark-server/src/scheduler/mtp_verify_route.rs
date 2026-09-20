// SPDX-License-Identifier: AGPL-3.0-only

//! A full DFlash block must use the same verifier at entry and steady state.

pub(super) const fn uses_dflash_gamma(returned_drafts: usize) -> bool {
    returned_drafts >= 4
}

#[cfg(test)]
mod tests {
    use super::uses_dflash_gamma;

    #[test]
    fn native_mtp_and_grammar_clamped_widths_keep_small_verifiers() {
        for drafts in 0..=3 {
            assert!(!uses_dflash_gamma(drafts));
        }
    }

    #[test]
    fn trained_dflash2_seven_drafts_never_fall_into_k4() {
        for drafts in [4, 7, 15, 31, usize::MAX] {
            assert!(uses_dflash_gamma(drafts));
        }
    }
}
