// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

use super::{Conflict, conflicts, process_flags, reexec_argv, reexec_env, swap_conflicts};

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// The Qwen3.5 -> Flash-Next swap that failed on hardware: FN's gates were unset when the
/// first model ran, so every one of them may have latched "unset".
#[test]
fn a_variable_the_new_model_needs_that_is_unset_conflicts() {
    let wanted = map(&[("ATLAS_W4A16_GEMV_RT2", "1")]);
    let found = conflicts(&wanted, &BTreeMap::new(), |_| None);
    assert_eq!(
        found,
        vec![Conflict {
            key: "ATLAS_W4A16_GEMV_RT2".into(),
            latched: None,
            wanted: Some("1".into())
        }]
    );
}

/// An earlier profile's variable the new model doesn't want can't be taken back.
#[test]
fn a_variable_an_earlier_profile_set_conflicts_when_the_new_model_lacks_it() {
    let set = map(&[("ATLAS_QWEN4_PREFILL_ATTN_FLASH", "1")]);
    let found = conflicts(&BTreeMap::new(), &set, |k| set.get(k).cloned());
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].wanted, None);
    assert_eq!(
        found[0].to_string(),
        "ATLAS_QWEN4_PREFILL_ATTN_FLASH: 1 -> <unset>"
    );
}

/// Control for both tests above: the SAME environment again is compatible, and an operator
/// export wins over any profile (as `apply_env` already does), so neither conflicts.
#[test]
fn a_matching_or_operator_owned_environment_stays_in_process() {
    let set = map(&[("A", "1")]);
    let current = map(&[("A", "1"), ("B", "operator")]);
    let wanted = map(&[("A", "1"), ("B", "profile")]);
    assert!(conflicts(&wanted, &set, |k| current.get(k).cloned()).is_empty());
    // ...but the same B conflicts once a PROFILE owns it.
    let set = map(&[("A", "1"), ("B", "operator")]);
    assert_eq!(
        conflicts(&wanted, &set, |k| current.get(k).cloned()).len(),
        1
    );
}

#[test]
fn nothing_has_latched_before_the_first_model() {
    let p = super::super::model_profile::profile_for(
        Some("Qwen3.8-Flash-Next-NVFP4-Offload"),
        "qwen4_exp",
    )
    .expect("the FN built-in");
    assert!(!p.env.is_empty());
    assert!(swap_conflicts(Some(&p), false).is_empty());
}

#[test]
fn process_flags_are_picked_out_of_the_original_command_line() {
    let argv = strings(&[
        "spark",
        "serve",
        "m",
        "--port",
        "8897",
        "--bind=127.0.0.1",
        "--max-batch-size",
        "1",
        "--require-auth",
        "--auth-tokens-file",
        "/t",
        "--no-tui",
    ]);
    assert_eq!(
        process_flags(&argv),
        strings(&[
            "--port",
            "8897",
            "--bind=127.0.0.1",
            "--require-auth",
            "--auth-tokens-file",
            "/t",
            "--no-tui"
        ])
    );
}

/// A built-in recipe's argv (which pins its own port and host) plus the process's listener
/// parses, and the process's listener is the one that wins.
#[test]
fn the_reexec_command_line_parses_and_keeps_the_listener() {
    use clap::Parser as _;
    let recipe = crate::recipe::builtin::recipes()
        .into_iter()
        .find(|r| r.id == "qwen3.8/qwen3.8-flash-next-offload-local")
        .unwrap();
    let argv = recipe.argv(&BTreeMap::new()).unwrap();
    assert!(
        argv.iter().any(|a| a == "--port"),
        "control: the recipe pins a port of its own"
    );
    let out = reexec_argv(
        &argv,
        &strings(&["--port", "8897", "--bind", "127.0.0.1", "--no-tui"]),
    );
    let cli = crate::cli::Cli::try_parse_from(&out).unwrap_or_else(|e| panic!("{out:?}: {e}"));
    let crate::cli::Command::Serve(args) = cli.command;
    assert_eq!(
        (args.port, args.bind.as_str(), args.no_tui),
        (8897, "127.0.0.1", true)
    );
    assert_eq!(args.kernel_target.as_deref(), Some("qwen3.8-flash-next"));
}

#[test]
fn the_reexec_environment_is_the_operators_plus_the_new_profile() {
    let current = vec![
        ("PATH".to_string(), "/bin".to_string()),
        ("OLD_PROFILE".to_string(), "1".to_string()),
        ("MINE".to_string(), "operator".to_string()),
        (
            "ATLAS_SWAP_PROFILE_ENV_KEYS".to_string(),
            "OLD_PROFILE".to_string(),
        ),
    ];
    let env = reexec_env(
        current,
        &map(&[("OLD_PROFILE", "1")]),
        &map(&[("NEW", "2"), ("MINE", "profile")]),
    );
    assert_eq!(
        env,
        map(&[
            ("ATLAS_SWAP_PROFILE_ENV_KEYS", "NEW"),
            ("MINE", "operator"),
            ("NEW", "2"),
            ("PATH", "/bin"),
        ])
    );
}
