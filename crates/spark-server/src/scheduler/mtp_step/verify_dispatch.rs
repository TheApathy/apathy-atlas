// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;
use super::verify_plan::{self, VerifyPath};

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    requested: usize,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    raw_argmax: bool,
    bootstrap: bool,
) {
    let path = verify_plan::select(
        model.requires_full_speculative_verify(),
        bootstrap,
        drafts.len(),
        requested,
        super::dflash_dispatch_min(),
    );
    if crate::scheduler::verify_pipeline_helper::dflash_diag_enabled() {
        tracing::info!(
            "DFLASH DIAG verify dispatch: bootstrap={bootstrap} drafts={} path={path:?}",
            drafts.len(),
        );
    }
    let step = match path {
        Some(VerifyPath::Generic) => step_verify_dflash,
        Some(VerifyPath::K2) => step_verify_k2,
        Some(VerifyPath::K3) => step_verify_k3,
        Some(VerifyPath::K4) => step_verify_k4,
        None => return,
    };
    step(model, a, drafts, num_drafts, verify_ctx, raw_argmax);
}
