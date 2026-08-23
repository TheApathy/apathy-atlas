// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    ensure_swappable_victim,
    tests::{active_seq, round_trip},
};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::emit_step::update_tool_body_phase;
use std::time::Instant;

#[path = "swap_lifecycle_source_scan.rs"]
mod source_scan;
use source_scan::{guard_precedes_spill, shadow_selector, victim_selector};

const GRAMMAR_EOS: u32 = 128;

fn json_grammar(after_open_object: bool) -> GrammarState {
    let mut vocab: Vec<String> = (0u8..128).map(|byte| String::from(byte as char)).collect();
    vocab.push("<eos>".to_string());
    let mut engine = GrammarEngine::new(&vocab, &[GRAMMAR_EOS as i32]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    if after_open_object {
        assert!(state.accept_token(b'{' as u32));
    }
    state
}

fn next_token_mask(grammar: &mut GrammarState) -> Vec<i32> {
    assert!(grammar.fill_bitmask());
    grammar.bitmask_data().to_vec()
}

#[test]
fn round_trip_moves_nonterminal_grammar_and_mid_tool_phase_exactly() {
    let (mut active, _cancel, _rx) = active_seq(false);
    active.grammar_state = Some(json_grammar(true));
    let live_mask = next_token_mask(active.grammar_state.as_mut().unwrap());
    let reset_mask = next_token_mask(&mut json_grammar(false));
    assert_ne!(
        live_mask, reset_mask,
        "resetting the matcher must change next-token admissibility"
    );
    assert!(active.tool_call_opened && active.inside_tool_body && active.suppress_tool_call);
    assert!(!active.require_tool_call);
    let tool_start = active.tool_call_start_token.expect("tool start sentinel");
    let tool_end = active.tool_call_end_token.expect("tool end sentinel");
    assert_ne!(tool_start, tool_end);

    let mut resumed = round_trip(active, Instant::now());
    assert!(resumed.tool_call_opened && resumed.inside_tool_body && resumed.suppress_tool_call);
    assert!(!resumed.require_tool_call);
    assert_eq!(resumed.tool_call_start_token, Some(tool_start));
    assert_eq!(resumed.tool_call_end_token, Some(tool_end));
    let resumed_mask = next_token_mask(resumed.grammar_state.as_mut().expect("live grammar"));
    assert_eq!(resumed_mask, live_mask);
    update_tool_body_phase(&mut resumed, tool_end);
    assert!(!resumed.inside_tool_body);
}

#[test]
fn round_trip_carries_require_tool_call_in_both_directions() {
    let source = include_str!("swap_lifecycle.rs");
    assert!(source.contains("require_tool_call: a.require_tool_call"));
    assert!(source.contains("require_tool_call: s.require_tool_call"));
    assert!(!source.contains("require_tool_call: true"));
    assert!(!source.contains("require_tool_call: false"));

    let (mut cleared, _cleared_cancel, _cleared_rx) = active_seq(false);
    cleared.require_tool_call = false;
    assert!(!round_trip(cleared, Instant::now()).require_tool_call);

    let (mut required, _required_cancel, _required_rx) = active_seq(false);
    required.require_tool_call = true;
    assert!(round_trip(required, Instant::now()).require_tool_call);
}

#[test]
fn grammar_active_victim_is_rejected_at_the_side_effect_boundary() {
    for grammar_idx in 0..3 {
        let mut active = Vec::new();
        for _ in 0..3 {
            active.push(active_seq(false).0);
        }
        active[grammar_idx].grammar_state = Some(json_grammar(true));
        let before: Vec<_> = active
            .iter()
            .map(|seq| (seq.session_hash, seq.last_token, seq.seq.slot_idx))
            .collect();
        for victim_idx in [grammar_idx, active.len(), usize::MAX] {
            let error = ensure_swappable_victim(&active, victim_idx).unwrap_err();
            assert!(error.to_string().contains("missing or grammar-active"));
            assert_eq!(active.len(), 3);
            assert_eq!(
                active
                    .iter()
                    .map(|seq| (seq.session_hash, seq.last_token, seq.seq.slot_idx))
                    .collect::<Vec<_>>(),
                before
            );
            assert_eq!(
                active
                    .iter()
                    .map(|seq| seq.grammar_state.is_some())
                    .collect::<Vec<_>>(),
                (0..3).map(|index| index == grammar_idx).collect::<Vec<_>>()
            );
        }
        for victim_idx in (0..3).filter(|&index| index != grammar_idx) {
            ensure_swappable_victim(&active, victim_idx).unwrap();
        }
    }
}

#[test]
fn production_mapping_moves_grammar_without_reset_or_clone() {
    let source = include_str!("swap_lifecycle.rs");
    let scheduler = include_str!("mod.rs");
    assert!(source.contains("grammar_state: a.grammar_state"));
    assert!(source.contains("grammar_state: s.grammar_state"));
    assert!(!source.contains("grammar_state: None"));
    assert!(!source.contains("grammar_state.clone()"));
    assert!(!source.contains("grammar_state.reset()"));
    let expected = "active.iter().enumerate().filter(|(_,a)|a.grammar_state.is_none()).max_by_key(|(_,a)|a.seq.block_table.len()).map(|(i,_)|i)";
    assert_eq!(victim_selector(scheduler).as_deref(), Some(expected));

    let evil = "active.iter().enumerate().chain(active.iter().enumerate()).max_by_key(|(_, a)| a.seq.block_table.len()).map(|(i, _)| i)";
    let shadow = shadow_selector(scheduler, &format!("let victim_idx = {evil}"));
    assert_eq!(victim_selector(&shadow), None);

    let mut_shadow = shadow_selector(scheduler, &format!("let mut victim_idx = {evil}"));
    assert_eq!(victim_selector(&mut_shadow), None);

    let commented_shadow = shadow_selector(
        scheduler,
        &format!("let /* coherent */ victim_idx = {evil}"),
    );
    assert_eq!(victim_selector(&commented_shadow), None);

    let raw_shadow = shadow_selector(
        scheduler,
        &format!("let _marker = r#\"\" //\"#; let /* comment */ victim_idx = {evil}"),
    );
    assert_eq!(victim_selector(&raw_shadow), None);

    let harmless_string = scheduler.replacen(
        "No swappable sequences (all grammar-active)",
        "let victim_idx = harmless text",
        1,
    );
    assert_eq!(victim_selector(&harmless_string), None);

    let char_shadow = shadow_selector(
        scheduler,
        &format!("let _quote = '\"'; let /* c */ victim_idx = {evil}"),
    );
    assert_eq!(victim_selector(&char_shadow), None);

    let lifetime_shadow = shadow_selector(
        scheduler,
        &format!("let _x: Option<&'static str> = None; let /* c */ victim_idx = {evil}"),
    );
    assert_eq!(victim_selector(&lifetime_shadow), None);

    let raw_ident_shadow = shadow_selector(scheduler, &format!("let r#victim_idx = {evil}"));
    assert_eq!(victim_selector(&raw_ident_shadow), None);

    let pattern_shadow = shadow_selector(scheduler, &format!("let (victim_idx) = {evil}"));
    assert_eq!(victim_selector(&pattern_shadow), None);

    let flow_shadow = scheduler
        .replacen(
            "let Some(victim_idx) = victim_idx else",
            &format!("let Some(victim_idx) = {evil} else"),
            1,
        )
        .replacen(
            "match swap_out_sequence(",
            "let _ = stringify!(victim_idx); match swap_out_sequence(",
            1,
        );
    assert_eq!(victim_selector(&flow_shadow), None);

    let reordered = scheduler.replacen(
        "let Some(victim_idx) = victim_idx else",
        "active.reverse(); let Some(victim_idx) = victim_idx else",
        1,
    );
    assert_eq!(victim_selector(&reordered), None);

    let dead_decoy = scheduler
        .replacen(
            "if let Some(ref mut spill) = spill_manager {",
            "if false { if let Some(ref mut spill) = spill_manager {",
            1,
        )
        .replacen(
            "// ── Start new requests ──",
            concat!(
                "}\n",
                "if let Some(ref mut spill) = spill_manager {\n",
                "let chosen = active.iter().position(|a| a.grammar_state.is_some()).unwrap();\n",
                "let (_id, _writer) = spill.create_file()?;\n",
                "let mut victim = active.swap_remove(chosen);\n",
                "model.free_sequence(&mut victim.seq)?;\n",
                "}\n",
                "// ── Start new requests ──"
            ),
            1,
        );
    assert_eq!(victim_selector(&dead_decoy), None);

    let swap_source = include_str!("swap_lifecycle.rs");
    assert!(guard_precedes_spill(swap_source));
    let late_guard = swap_source
        .replacen(
            "ensure_swappable_victim(active, victim_idx)?;",
            "// ensure_swappable_victim(active, victim_idx)?;",
            1,
        )
        .replacen(
            "let (swap_id, mut writer) = spill.create_file()?;",
            concat!(
                "let (swap_id, mut writer) = spill.create_file()?;\n",
                "ensure_swappable_victim(active, victim_idx)?;"
            ),
            1,
        );
    assert!(!guard_precedes_spill(&late_guard));

    let early_create = swap_source.replacen(
        "ensure_swappable_victim(active, victim_idx)?;",
        concat!(
            "let _probe = spill.create_file()?;\n",
            "ensure_swappable_victim(active, victim_idx)?;"
        ),
        1,
    );
    assert!(!guard_precedes_spill(&early_create));
}
