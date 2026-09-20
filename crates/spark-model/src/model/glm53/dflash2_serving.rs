// SPDX-License-Identifier: AGPL-3.0-only
//! Serving-only selection; the shared cached body owns all submitted work.
use super::*;

impl Glm53Dflash2Runtime {
    pub(super) fn admit_serving(
        &self,
        kv_prefix: bool,
        capturing: bool,
    ) -> Result<ServingProjection> {
        let choice = self.serving_projection.serving_choice()?;
        choice.admit_current(
            std::env::var_os("ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION").as_deref(),
            kv_prefix,
        )?;
        choice.admit_proposal(kv_prefix, capturing)?;
        Ok(choice)
    }

    pub(super) fn propose_serving_gemv(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
    ) -> Result<[u32; 7]> {
        target.ensure_dflash2_proposal_ready()?;
        ensure!(
            matches!(self.serving_projection, CommittedProjection::StableGemv(_)),
            "GEMV serving requires its startup-resolved projection"
        );
        self.preflight_gemv_projection(self.serving_projection)?;
        self.ensure_projection_ready(self.serving_projection.family())?;
        // The inner Attempt retains host staging and pending cursor authority
        // across errors. Catch only after its state lock has unwound, exactly
        // as the existing cached serving path does; never retry another family.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.propose_cached_observed_with_projection(
                target,
                anchor,
                stream,
                None,
                self.serving_projection,
            )
        })) {
            Ok(result) => result,
            Err(_) => {
                target.poison_verify(stream);
                anyhow::bail!("GLM GEMV proposal panicked; pending owner retained for reset")
            }
        }
    }
}
