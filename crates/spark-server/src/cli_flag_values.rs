// SPDX-License-Identifier: AGPL-3.0-only

//! The closed value sets of the enumerated `spark serve` string flags.
//!
//! MINIMAL PORT. Upstream's `cli/flag_values.rs` serves two consumers: its
//! `validate::check_enum`, which REFUSES a value outside the set, and the
//! dashboard's option picker, which OFFERS the set. This engine has no
//! `validate_serve_args`, so only the picker reads this today — but the module
//! exists rather than the list being inlined at the picker, for upstream's own
//! reason: two copies of a value set drift into the worst failure a picker has,
//! **offering a value the server refuses, or hiding one it accepts.**
//!
//! `--kv-cache-dtype` is deliberately NOT a literal list here. Its authority is
//! `spark_runtime::kv_cache::KvCacheDtype::ALL`, derived from the enum itself,
//! so a variant added there appears in the picker automatically instead of
//! silently missing from a hand-maintained copy.
//!
//! Deliberately NOT wired into clap as `PossibleValuesParser`: that would be a
//! second enforcement point over the same data, and two enforcement points is
//! how the two come to disagree. clap stays the authority on the flag SURFACE
//! (names, help, defaults, arity); this module is the authority on the
//! enumerated VALUES.

/// The closed value set for `long`, or `None` when the flag is free text.
///
/// `long` is the flag's long name WITHOUT the leading dashes, as clap reports
/// it — `kv-cache-dtype`, not `--kv-cache-dtype`.
pub(crate) fn options_for_flag(long: &str) -> Option<Vec<String>> {
    match long {
        // Dispatched in `main_modules::serve` with a bail on anything else —
        // AFTER the weight load. Listing it here moves the typo diagnosis to
        // the Library form, where it costs milliseconds instead of minutes.
        "scheduling-policy" => Some(vec!["fifo".to_string(), "slai".to_string()]),
        "kv-cache-dtype" => Some(
            spark_runtime::kv_cache::KvCacheDtype::ALL
                .iter()
                .map(|d| d.name().to_string())
                .collect(),
        ),
        // Everything else is free text or a number. Returning `None` rather
        // than an empty vec keeps "no closed set" distinguishable from "a
        // closed set that happens to be empty" — the picker renders the first
        // as a text field and would render the second as an empty list.
        _ => None,
    }
}

#[cfg(test)]
#[path = "cli_flag_values_tests.rs"]
mod tests;
