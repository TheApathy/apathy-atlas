// SPDX-License-Identifier: AGPL-3.0-only

const SCHEDULER: &str = include_str!("../src/scheduler/mod.rs");
const HELPERS: &str = include_str!("../src/scheduler/mod_helpers.rs");
const LIFECYCLE: &str = include_str!("../src/scheduler/lifecycle.rs");
const MTP: &str = include_str!("../src/scheduler/mtp_step.rs");

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source.find(signature).expect("function signature missing");
    let open = start
        + source[start..]
            .find('{')
            .expect("function opening brace missing");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..open + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("function closing brace missing");
}

#[test]
fn scheduler_carries_one_cache_policy_per_active_sequence() {
    let scheduler = function_body(SCHEDULER, "pub fn run(");
    assert!(scheduler.contains("let mut cache_on_finish = vec![true; active.len()];"));
    assert_eq!(scheduler.matches("step_mtp(").count(), 3);
    assert_eq!(scheduler.matches("&mut cache_on_finish").count(), 4);
    assert!(scheduler.contains("let was_verify = !active[0].pending_drafts.is_empty();"));
    assert!(scheduler.contains("if was_verify && active[0].finished"));
    assert!(scheduler.contains("step_self_spec(&*model, &mut active, num_drafts);"));
    assert_eq!(scheduler.matches("cache_on_finish[0] = false").count(), 2);
    assert!(
        scheduler.contains("retire_finished_sequences(&*model, &mut active, &mut cache_on_finish)")
    );
}

#[test]
fn every_speculative_verify_exit_suppresses_finished_prefixes() {
    let step = function_body(MTP, "pub fn step_mtp(");
    assert!(step.contains("cache_on_finish: &mut [bool]"));
    assert_eq!(step.matches("suppress_finished_verify_cache(").count(), 4);

    let helper = function_body(MTP, "fn suppress_finished_verify_cache(");
    assert!(helper.contains("for &idx in verify_idxs"));
    assert!(helper.contains("if active[idx].finished"));
    assert!(helper.contains("cache_on_finish[idx] = false"));
}

#[test]
fn retirement_swaps_policy_with_sequence_and_gates_only_cache_insert() {
    let retire = function_body(HELPERS, "pub(super) fn retire_finished_sequences(");
    assert!(retire.contains("assert_eq!(active.len(), cache_on_finish.len())"));
    assert!(retire.contains("let cache_sequence = cache_on_finish.swap_remove(i);"));
    assert!(retire.contains("finish_sequence_with_cache(model, &mut a, cache_sequence)"));

    let finish = function_body(LIFECYCLE, "pub(super) fn finish_sequence_with_cache(");
    assert!(finish.contains("if cache_sequence"));
    assert_eq!(finish.matches("model.cache_sequence(&a.seq)").count(), 1);
    assert!(finish.contains("model.free_sequence(&mut a.seq)"));
    assert!(finish.contains("model.ep_broadcast_cmd(0xFFFFFFF1)"));

    let ordinary = function_body(LIFECYCLE, "pub fn finish_sequence(");
    assert!(ordinary.contains("finish_sequence_with_cache(model, a, true)"));
}
