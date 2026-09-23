// SPDX-License-Identifier: AGPL-3.0-only

//! Bound target verification independently of the checkpoint's proposal block.

use anyhow::{Result, ensure};

pub(in crate::main_modules) fn resolve_glm_dflash_max_drafts(
    requested: Option<u8>,
    is_glm53_exl3: bool,
    dflash: bool,
    gamma: usize,
) -> Result<Option<usize>> {
    let Some(requested) = requested else {
        return Ok(None); // Preserve the caller's existing scheduling policy.
    };
    ensure!(
        is_glm53_exl3 && dflash && gamma == 8,
        "--glm-dflash-max-drafts requires the strict GLM-5.3 EXL3 source, \
         --dflash and checkpoint-pinned --dflash-gamma 8"
    );
    ensure!(
        (1..=7).contains(&requested),
        "--glm-dflash-max-drafts must be in 1..=7"
    );
    Ok(Some(usize::from(requested)))
}

#[cfg(test)]
mod tests {
    use super::resolve_glm_dflash_max_drafts as resolve;

    #[test]
    fn glm_dflash_prefix_all_legal_extents_are_exact() {
        for n in 1..=7 {
            assert_eq!(
                resolve(Some(n), true, true, 8).unwrap(),
                Some(usize::from(n))
            );
        }
    }

    #[test]
    fn glm_dflash_prefix_rejects_programmatic_out_of_range_values() {
        for n in [0, 8, u8::MAX] {
            assert!(resolve(Some(n), true, true, 8).is_err());
        }
    }

    #[test]
    fn glm_dflash_prefix_requires_admitted_model_and_method() {
        for (admitted, dflash) in [(false, false), (false, true), (true, false)] {
            assert!(resolve(Some(3), admitted, dflash, 8).is_err());
        }
    }

    #[test]
    fn glm_dflash_prefix_never_changes_the_checkpoint_block_size() {
        for gamma in [0, 1, 3, 7, 9, 16, usize::MAX] {
            assert!(resolve(Some(3), true, true, gamma).is_err());
        }
    }

    #[test]
    fn glm_dflash_prefix_omission_leaves_other_models_and_modes_unchanged() {
        for admitted in [false, true] {
            for dflash in [false, true] {
                for gamma in [0, 8, 16] {
                    assert_eq!(resolve(None, admitted, dflash, gamma).unwrap(), None);
                }
            }
        }
    }

    #[test]
    fn glm_dflash_prefix_is_admitted_before_gpu_and_used_by_the_scheduler() {
        let source = include_str!("../serve.rs");
        let admission = source
            .find("serve_phases::resolve_glm_dflash_max_drafts(")
            .unwrap();
        let gpu = source.find("serve_phases::init_gpu_backend(").unwrap();
        assert!(admission < gpu);
        assert_eq!(
            source
                .matches("serve_phases::resolve_glm_dflash_max_drafts(")
                .count(),
            1
        );
        assert!(source.contains("if let Some(limit) = glm_dflash_max_drafts"));
        assert!(source.contains("if args.dflash_gamma != 8"));
        // Full drafter/arena allocation remains on the existing gamma path.
        assert!(include_str!("build.rs").contains("args.dflash_gamma.saturating_sub(1).max(1)"));
    }
}
