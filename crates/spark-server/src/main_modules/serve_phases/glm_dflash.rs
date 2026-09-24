// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 speculative-decoding admission at serve time.
//!
//! GLM speculates only through its own DFlash2 drafter and policy verifier on
//! the EXL3 checkpoint. The proposal block is pinned by the checkpoint (8);
//! the verify width is bounded separately by `--glm-dflash-max-drafts`.

use anyhow::{Result, bail, ensure};

/// DFlash2 block size the GLM-5.3 drafter checkpoint was trained with.
const GLM53_DFLASH2_BLOCK: usize = 8;

/// Resolve GLM speculation for this serve. `Ok(Some(drafts))` is the verify
/// width for a GLM DFlash2 run, `Ok(None)` leaves every other model (and
/// target-only GLM) to its existing path. Anything else GLM cannot serve is
/// refused rather than handed to a generic proposer.
pub(in crate::main_modules) fn resolve_glm_dflash(
    model_type: &str,
    is_glm53_exl3: bool,
    dflash: bool,
    speculative: bool,
    gamma: Option<usize>,
    max_drafts: Option<u8>,
) -> Result<Option<usize>> {
    if model_type != "glm5_next" {
        ensure!(
            max_drafts.is_none(),
            "--glm-dflash-max-drafts applies only to GLM-5.3"
        );
        return Ok(None);
    }
    if speculative {
        bail!("GLM-5.3 has no MTP head: speculate with --dflash and its DFlash2 drafter");
    }
    if !dflash {
        ensure!(
            max_drafts.is_none(),
            "--glm-dflash-max-drafts requires --dflash"
        );
        return Ok(None);
    }
    ensure!(
        is_glm53_exl3,
        "GLM-5.3 DFlash2 requires the EXL3 checkpoint; serve the GGUF target-only"
    );
    ensure!(
        gamma.unwrap_or(GLM53_DFLASH2_BLOCK) == GLM53_DFLASH2_BLOCK,
        "GLM-5.3 DFlash2 has a checkpoint-pinned block of {GLM53_DFLASH2_BLOCK}; \
         pass --dflash-gamma {GLM53_DFLASH2_BLOCK} (bound the verify width with \
         --glm-dflash-max-drafts)"
    );
    let drafts = max_drafts.map_or(GLM53_DFLASH2_BLOCK - 1, usize::from);
    ensure!(
        (1..GLM53_DFLASH2_BLOCK).contains(&drafts),
        "--glm-dflash-max-drafts must be in 1..=7"
    );
    Ok(Some(drafts))
}

#[cfg(test)]
mod tests {
    use super::resolve_glm_dflash as resolve;

    #[test]
    fn other_models_are_untouched() {
        assert_eq!(
            resolve("qwen3_next", false, true, false, Some(15), None).unwrap(),
            None
        );
        assert_eq!(
            resolve("qwen3_next", false, false, true, None, None).unwrap(),
            None
        );
        assert!(resolve("qwen3_next", false, true, false, None, Some(3)).is_err());
    }

    #[test]
    fn target_only_glm_is_untouched() {
        assert_eq!(
            resolve("glm5_next", true, false, false, None, None).unwrap(),
            None
        );
        assert!(resolve("glm5_next", true, false, false, None, Some(3)).is_err());
    }

    #[test]
    fn glm_dflash_resolves_the_verify_width() {
        assert_eq!(
            resolve("glm5_next", true, true, false, Some(8), Some(3)).unwrap(),
            Some(3)
        );
        assert_eq!(
            resolve("glm5_next", true, true, false, None, Some(2)).unwrap(),
            Some(2)
        );
        assert_eq!(
            resolve("glm5_next", true, true, false, Some(8), None).unwrap(),
            Some(7)
        );
    }

    #[test]
    fn glm_refuses_what_it_cannot_serve() {
        assert!(resolve("glm5_next", true, false, true, None, None).is_err());
        assert!(resolve("glm5_next", false, true, false, Some(8), Some(3)).is_err());
        assert!(resolve("glm5_next", true, true, false, Some(15), None).is_err());
        assert!(resolve("glm5_next", true, true, false, Some(8), Some(0)).is_err());
        assert!(resolve("glm5_next", true, true, false, Some(8), Some(8)).is_err());
    }
}
