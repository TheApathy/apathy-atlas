// SPDX-License-Identifier: AGPL-3.0-only

const SOURCE: &str = include_str!("mtp_step.rs");

fn ordered(source: &str, markers: &[&str]) {
    let mut rest = source;
    for marker in markers {
        let at = rest
            .find(marker)
            .unwrap_or_else(|| panic!("missing/order: {marker}"));
        rest = &rest[at + marker.len()..];
    }
}

#[test]
fn bootstrap_gamma_precedes_k4_and_keeps_the_whole_proposal() {
    let bootstrap = SOURCE.split("// ── Phase B:").next().unwrap();
    ordered(
        bootstrap,
        &[
            "Ok(init) if !init.is_empty() =>",
            "uses_dflash_gamma(init.len())",
            "step_verify_dflash(",
            "&init,",
            "else if eff >= 3 && init.len() >= 3",
            "step_verify_k4(",
        ],
    );
    assert!(!bootstrap.contains("init.truncate("));
    assert!(!bootstrap.contains("&init[.."));
}

#[test]
fn steady_and_bootstrap_share_the_same_full_block_selector() {
    assert!(SOURCE.contains("use mtp_verify_route::uses_dflash_gamma;"));
    assert_eq!(SOURCE.matches("if uses_dflash_gamma(").count(), 2);
    let steady = SOURCE.split("// ── Phase B:").nth(1).unwrap();
    ordered(
        steady,
        &[
            "uses_dflash_gamma(drafts.len())",
            "step_verify_dflash(",
            "&drafts,",
        ],
    );
}

#[test]
fn grammar_and_serial_seam_controls_remain_before_bootstrap() {
    ordered(
        SOURCE,
        &[
            "dflash_verify_raw_argmax",
            "!crate::scheduler::verify_pipeline_helper::dflash_seam_serial_enabled()",
            "crate::scheduler::adaptive_spec::spec_allowed(a)",
            "let eff = if a.grammar_state.is_some()",
            "model.run_mtp_propose_multi(",
            "Ok(init) if !init.is_empty() =>",
        ],
    );
}
