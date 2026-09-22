// SPDX-License-Identifier: AGPL-3.0-only

// Standalone source-boundary gate; numerical parity still requires real GPU
// comparisons of same-input outputs and the complete staged/persistent state.
const OPS: &str = include_str!("../ops.rs");
const PROJECTION: &str = include_str!("glm53_exl3_projection.rs");
const TRANSACTION: &str = include_str!("../../model/glm53/dsa_verify_execution.rs");
const DSA: &str = include_str!("../../model/glm53/dsa_attention.rs");
const KDA: &str = include_str!("../../model/glm53/kda_attention.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn verification_scope_is_registered_separately_from_prefill_scope() {
    assert!(OPS.contains("mod glm53_verify_scope;"));
    assert!(OPS.contains("pub(crate) use glm53_verify_scope::*;"));
}

#[test]
fn unqualified_arithmetic_is_opt_in_before_any_state_effect() {
    let body = compact(TRANSACTION);
    let flag = body
        .find("parse_exact_verify_flag(")
        .expect("strict experimental admission");
    let save = body.find("snapshot.save(&mutio)").unwrap();
    assert!(flag < save);
    assert!(body.contains("ATLAS_GLM53_EXACT_VERIFY"));
    assert!(body.contains("ifexact_verify{"));
    assert!(body.contains("}else{self.verify_tokens_staged(tokens,stream)?}"));
}

#[test]
fn transaction_admits_outside_prefill_and_scopes_only_the_staged_forward() {
    let body = compact(TRANSACTION);
    let admission = body
        .find("!glm53_exact_wide_prefill_active()&&!glm53_layer_major_prefill_active()")
        .expect("existing prefill exclusion must remain");
    let save = body.find("snapshot.save(&mutio)").unwrap();
    let forward = body
        .find("with_glm53_exact_verify(||self.verify_tokens_staged(tokens,stream))")
        .expect("only the staged target call may enter verification arithmetic scope");
    let oracle = body.find("self.argmax_rows_device(").unwrap();
    let commit = body.find("self.commit_accepted(stream)").unwrap();
    let replay = body.find("self.walk(token,stream)").unwrap();
    assert!(admission < save && save < forward && forward < oracle);
    assert!(oracle < commit && commit < replay);
    assert_eq!(body.matches("with_glm53_exact_verify(").count(), 1);
    assert!(!body.contains("with_glm53_exact_wide_prefill("));
    assert!(!body.contains("with_glm53_layer_major_prefill("));
}

#[test]
fn projection_arithmetic_accepts_verify_without_renaming_prefill_publication() {
    let body = compact(PROJECTION);
    assert!(body.contains("glm53_exact_verify_active()"));
    assert!(body.contains("linear.prepare_bf16_row_exact(gpu,plan.rows)"));
    assert!(body.contains("letone=self.plan(1)?;"));
    assert!(body.contains("forrowin0..plan.rowsasusize"));
}

#[test]
fn verification_uses_causal_dsa_rows_without_wide_precompute() {
    let stage = DSA
        .split("pub fn stage_exl3_rows(")
        .nth(1)
        .unwrap()
        .split("fn stage_inner(")
        .next()
        .unwrap();
    let body = compact(stage);
    assert!(body.contains("glm53_exact_verify_active()"));
    assert!(body.contains("!glm53_exact_verify_active()"));
    assert!(body.contains("forrowin0..rowsasusize"));
    assert!(body.contains("self.stage_exl3_rows(gpu,1,"));
}

#[test]
fn verifier_does_not_activate_kda_prefill_publication() {
    let body = compact(KDA);
    assert!(body.contains("letexact_wide=rows>1&&glm53_exact_wide_prefill_active();"));
    assert!(body.contains("ifexact_wide&&!batch_exact_carry{"));
    assert!(body.contains("ifbatch_exact_carry{"));
    assert!(!body.contains("glm53_exact_verify_active"));
    assert!(!body.contains("with_glm53_exact_verify"));
}
