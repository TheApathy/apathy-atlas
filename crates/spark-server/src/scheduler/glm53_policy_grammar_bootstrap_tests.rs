// SPDX-License-Identifier: AGPL-3.0-only

//! Include under glm53_policy_driver_tests to reuse its actual model boundary.
//! No environment/global limit mutation and no replacement sampling policy.

use super::{DriverModel, context, grammar, one, run, scores, sequence, synchronized};

#[test]
fn live_grammar_bootstrap_keeps_three_requested_drafts_for_four_policy_rows() {
    // A grammar-illegal draft is harmless to the typed verifier: the first
    // ordinary grammar-masked pick becomes the bonus and commits only anchor.
    // It must not cause the proposal interface to silently become width one.
    let model = DriverModel::new(&[scores(2); 4], vec![2, 2, 2]);
    let mut seq = sequence();
    assert!(seq.pending_drafts.is_empty()); // Exercise bootstrap, not steady verify.
    seq.grammar_state = Some(grammar());
    seq.eos_tokens = vec![3];
    let mut active = one(&model, seq);

    run(&model, &mut active, 3, true, false, None, &context());

    assert_eq!(
        model.inputs(),
        vec![vec![0, 2, 2, 2]],
        "live grammar must use all three requested drafts plus anchor"
    );
    assert_eq!(model.count("verify"), 1);
    assert_eq!(model.count("copy"), 1);
    assert_eq!(model.count("commit"), 1);
    assert_eq!(model.count("decode"), 0);
    assert_eq!(active[0].output_tokens, vec![0]); // Actual JSON mask chooses "{".
    assert_eq!(
        active[0]
            .grammar_state
            .as_ref()
            .unwrap()
            .num_history_steps(),
        1
    );
    assert_eq!(model.position(), 3);
    assert_eq!(model.emitted(), vec![0]);
    synchronized(&model, &active[0]);
}
