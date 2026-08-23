// SPDX-License-Identifier: AGPL-3.0-only

use super::{guard_precedes_spill, victim_selector};

#[test]
fn guard_contract_rejects_conditional_and_cfg_split_definitions() {
    let source = include_str!("swap_lifecycle.rs");
    let conditional = source
        .replacen(
            "anyhow::ensure!(",
            "if active.len() == 3 { anyhow::ensure!(",
            1,
        )
        .replacen("    Ok(())", "    }\n    Ok(())", 1);
    assert!(!guard_precedes_spill(&conditional));

    let split = source
        .replacen(
            "fn ensure_swappable_victim(",
            "#[cfg(test)]\nfn ensure_swappable_victim(",
            1,
        )
        .replacen(
            "/// Swap out an active sequence",
            concat!(
                "#[cfg(not(test))]\n",
                "fn ensure_swappable_victim(_: &[ActiveSeq], _: usize) -> Result<()> { Ok(()) }\n\n",
                "/// Swap out an active sequence"
            ),
            1,
        );
    assert!(!guard_precedes_spill(&split));
}

#[test]
fn selector_rejects_dead_wrapper_outside_both_markers() {
    const START: &str = "// ── Swap-out: evict active sequences to disk when blocks run low ──";
    const END: &str = "// ── Start new requests ──";
    let scheduler = include_str!("mod.rs");
    let wrapped = scheduler
        .replacen(START, &format!("if false {{\n        {START}"), 1)
        .replacen(END, &format!("{END}\n        }}"), 1);
    assert_eq!(victim_selector(&wrapped), None);
}

#[test]
fn selector_rejects_executable_changes_outside_the_swap_slice() {
    let scheduler = include_str!("mod.rs");
    for changed in [
        scheduler.replacen("/tmp/atlas-swap", "/tmp", 1),
        scheduler.replacen(
            "    let dflash_spec_think =",
            "    let _literal_probe = (r#\"changed\"#, 'x');\n    let dflash_spec_think =",
            1,
        ),
        scheduler.replacen("    loop {", "    return;\n    loop {", 1),
        scheduler.replacen("    loop {", "    #[cfg(any())]\n    loop {", 1),
        scheduler.replacen(
            "        start_new_requests(",
            "        active.retain(|a| a.grammar_state.is_none());\n        start_new_requests(",
            1,
        ),
        scheduler.replacen(
            "        start_new_requests(",
            concat!(
                "        if let Some(chosen) = active.iter().position(|a| a.grammar_state.is_some()) {",
                " let _ = active.remove(chosen); }\n        start_new_requests("
            ),
            1,
        ),
    ] {
        assert_eq!(victim_selector(&changed), None);
    }
}
