// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model environment across a hot swap: when the swap can stay in-process, and the
//! re-exec that replaces it when it cannot.
//!
//! ## Why a swap can't always just set the new model's environment
//! Most `ATLAS_*` gates are read once into a `OnceLock` and latched for the life of the
//! process. A swap that sets `ATLAS_QWEN4_PREFILL_ATTN_FLASH=1` for Flash-Next after a model
//! already ran has no effect on a gate that latched "unset". And a variable the previous
//! profile set can't be taken back from a gate that latched it. Loading anyway serves the new
//! model with a mix of environments nobody measured. On a real swap it failed closed with
//! "ATLAS_QWEN4_PREFILL_SSM_GRID32 requires ATLAS_W4A16_GEMV_RT2=1".
//!
//! Which gates a process actually read is not recorded anywhere (there are ~460 of them,
//! read ad hoc), so the check is conservative: once a model has loaded in this process,
//! any variable the new profile needs that isn't already set to its value, or any
//! variable an earlier profile set that the new one doesn't want, counts as latched. A
//! variable the operator exported is never a conflict: it wins over every profile, exactly as
//! in [`super::model_profile::apply_env`].
//!
//! ## Re-exec
//! On a conflict the TUI replaces the process image with `spark serve` for the new recipe:
//! the recipe's argv plus the process-scoped flags this process was started with (listener,
//! auth, dump, `--no-tui`), and the operator's environment plus the new profile's. The new
//! process starts with no latched gates at all. `ATLAS_SWAP_REEXEC=0` turns re-exec off and
//! the swap fails closed with the conflicts named, before anything is released.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::model_profile::ModelProfile;

/// Carries the profile-set keys across a re-exec, so the new process can tell them apart from
/// the operator's own exports.
const PROFILE_KEYS_ENV: &str = "ATLAS_SWAP_PROFILE_ENV_KEYS";
/// `0` disables re-exec: a conflicting swap is refused instead.
const REEXEC_SWITCH: &str = "ATLAS_SWAP_REEXEC";

/// Flags that belong to the process, not to a recipe, with whether each takes a value.
/// Carried from the original argv into the re-exec, and stripped from the recipe's argv.
const PROCESS_FLAGS: &[(&str, bool)] = &[
    ("--port", true),
    ("--bind", true),
    ("--host", true),
    ("--dump", true),
    ("--auth-tokens-file", true),
    ("--auth-token", true),
    ("--require-auth", false),
    ("--no-tui", false),
];

/// Variables a launch profile set in this process, with the value it set.
fn profile_set() -> &'static Mutex<BTreeMap<String, String>> {
    static SET: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
    SET.get_or_init(|| {
        // A re-exec'd process inherits its profile's keys through PROFILE_KEYS_ENV.
        let keys = std::env::var(PROFILE_KEYS_ENV).unwrap_or_default();
        let inherited = keys
            .split(',')
            .filter(|k| !k.is_empty())
            .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
            .collect();
        Mutex::new(inherited)
    })
}

/// Record that a profile set `key=value` (called for each variable `apply_env` set).
pub(crate) fn record_profile_set(key: &str, value: &str) {
    profile_set()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key.to_string(), value.to_string());
}

/// One variable the new model needs that this process may already have latched otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Conflict {
    pub key: String,
    /// What the process has (`None` = unset).
    pub latched: Option<String>,
    /// What the new model needs (`None` = must be unset).
    pub wanted: Option<String>,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let show = |v: &Option<String>| v.clone().unwrap_or_else(|| "<unset>".into());
        write!(
            f,
            "{}: {} -> {}",
            self.key,
            show(&self.latched),
            show(&self.wanted)
        )
    }
}

/// The conflicts between the environment `wanted` needs and what the process may have
/// latched. Pure: `profile_set` holds the earlier profiles' variables and `current` reads the
/// live environment.
pub(crate) fn conflicts(
    wanted: &BTreeMap<String, String>,
    profile_set: &BTreeMap<String, String>,
    current: impl Fn(&str) -> Option<String>,
) -> Vec<Conflict> {
    let mut out = Vec::new();
    for (key, value) in wanted {
        let cur = current(key);
        let conflict = match &cur {
            Some(c) => profile_set.contains_key(key) && c != value,
            // Unset now, so any gate that read it latched "unset".
            None => true,
        };
        if conflict {
            out.push(Conflict {
                key: key.clone(),
                latched: cur,
                wanted: Some(value.clone()),
            });
        }
    }
    for key in profile_set.keys().filter(|k| !wanted.contains_key(*k)) {
        if let Some(cur) = current(key) {
            out.push(Conflict {
                key: key.clone(),
                latched: Some(cur),
                wanted: None,
            });
        }
    }
    out
}

/// The conflicts for a swap to `next`, or empty when the swap can stay in-process. Nothing
/// has latched before a process's first model load, so a modelless process never conflicts.
pub(crate) fn swap_conflicts(
    next: Option<&ModelProfile>,
    model_loaded_before: bool,
) -> Vec<Conflict> {
    if !model_loaded_before {
        return Vec::new();
    }
    let empty = BTreeMap::new();
    let wanted = next.map_or(&empty, |p| &p.env);
    let set = profile_set()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    conflicts(wanted, &set, |k| std::env::var(k).ok())
}

pub(crate) fn reexec_enabled() -> bool {
    std::env::var(REEXEC_SWITCH).map_or(true, |v| v != "0")
}

/// The refusal when a conflicting swap can't re-exec.
pub(crate) fn refusal(recipe: &str, found: &[Conflict], why: &str) -> anyhow::Error {
    let list: Vec<String> = found.iter().map(ToString::to_string).collect();
    anyhow::anyhow!(
        "{recipe} needs an environment this process can no longer give it (gates latch at first \
         read): {}. {why}. The running model is untouched; restart spark with the new model to \
         switch.",
        list.join(", ")
    )
}

/// The process-scoped flags in `argv` (the original command line), in order, with values.
pub(crate) fn process_flags(argv: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        let name = arg.split_once('=').map_or(arg.as_str(), |(n, _)| n);
        match PROCESS_FLAGS.iter().find(|(f, _)| *f == name) {
            Some((_, true)) if !arg.contains('=') => {
                out.push(arg.clone());
                if let Some(value) = argv.get(i + 1) {
                    out.push(value.clone());
                }
                i += 1;
            }
            Some(_) => out.push(arg.clone()),
            None => {}
        }
        i += 1;
    }
    out
}

/// `recipe_argv` with its own process-scoped flags removed and `process`'s appended.
pub(crate) fn reexec_argv(recipe_argv: &[String], process: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(recipe_argv.len() + process.len());
    let mut i = 0;
    while i < recipe_argv.len() {
        let arg = &recipe_argv[i];
        let name = arg.split_once('=').map_or(arg.as_str(), |(n, _)| n);
        match PROCESS_FLAGS.iter().find(|(f, _)| *f == name) {
            Some((_, true)) if !arg.contains('=') => i += 2,
            Some(_) => i += 1,
            None => {
                out.push(arg.clone());
                i += 1;
            }
        }
    }
    out.extend(process.iter().cloned());
    out
}

/// The re-exec'd process's environment: the operator's (the current one minus every
/// variable a profile set) plus `wanted` where the operator did not export it, and the
/// marker that tells the new process which variables are the profile's.
pub(crate) fn reexec_env(
    current: impl IntoIterator<Item = (String, String)>,
    profile_set: &BTreeMap<String, String>,
    wanted: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = current
        .into_iter()
        .filter(|(k, _)| !profile_set.contains_key(k) && k != PROFILE_KEYS_ENV)
        .collect();
    let mut keys = Vec::new();
    for (k, v) in wanted {
        if !env.contains_key(k) {
            env.insert(k.clone(), v.clone());
            keys.push(k.clone());
        }
    }
    if !keys.is_empty() {
        env.insert(PROFILE_KEYS_ENV.into(), keys.join(","));
    }
    env
}

/// Replace this process with `spark serve` for `recipe_argv` under `wanted`'s environment.
/// Returns only if the exec failed. The caller must have released the outgoing model: the
/// new image starts with the GPU memory the driver reclaims from this one.
pub(crate) fn exec(recipe_argv: &[String], wanted: &BTreeMap<String, String>) -> anyhow::Error {
    use std::os::unix::process::CommandExt as _;
    let original: Vec<String> = std::env::args().collect();
    let argv = reexec_argv(recipe_argv, &process_flags(&original));
    let set = profile_set()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let env = reexec_env(std::env::vars(), &set, wanted);
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => return anyhow::anyhow!("re-exec: cannot find this binary: {e}"),
    };
    tracing::warn!(
        "swap: re-exec {} {:?}",
        exe.display(),
        argv.get(1..).unwrap_or_default()
    );
    // The new process takes the terminal over from scratch.
    crate::tui::terminal_guard::restore();
    let err = std::process::Command::new(&exe)
        .args(argv.get(1..).unwrap_or_default())
        .env_clear()
        .envs(&env)
        .exec();
    anyhow::anyhow!("re-exec of {} failed: {err}", exe.display())
}

#[cfg(test)]
#[path = "swap_env_tests.rs"]
mod tests;
