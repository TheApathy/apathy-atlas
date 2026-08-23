// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::Path};

fn assert_no_raw_calls(path: &Path, source: &str, calls: &[String], is_authority: bool) {
    for call in calls {
        let count = source.matches(call).count();
        if is_authority {
            assert_eq!(count, 1, "typed adapter call count for {call}");
        } else {
            assert_eq!(count, 0, "raw model lifecycle bypass in {}", path.display());
        }
    }
}

#[test]
fn scheduler_model_proposal_and_tree_take_live_only_in_typed_adapter() {
    let scheduler = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let proposer_call = [".run_mtp_", "propose_multi("].concat();
    let tree_take = [".take_pending_", "tree_payload("].concat();

    for entry in fs::read_dir(scheduler).expect("scheduler directory") {
        let path = entry.expect("scheduler entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("Rust source");
        let is_authority =
            path.file_name().and_then(|name| name.to_str()) == Some("proposal_lifecycle.rs");
        assert_no_raw_calls(
            &path,
            &source,
            &[proposer_call.clone(), tree_take.clone()],
            is_authority,
        );
    }
}

#[test]
#[should_panic(expected = "raw model lifecycle bypass")]
fn old_generic_direct_sequence_mutant_is_rejected() {
    let calls = [
        [".run_mtp_", "propose_multi("].concat(),
        [".take_pending_", "tree_payload("].concat(),
    ];
    let mutant = format!("model{}...); model{}...);", calls[0], calls[1]);
    assert_no_raw_calls(Path::new("verify_dflash_step.rs"), &mutant, &calls, false);
}
