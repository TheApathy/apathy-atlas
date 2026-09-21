// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::ffi::OsString;

/// POSITIVE: an explicit `ATLAS_HOME` wins outright, and is reported as such.
#[test]
fn an_explicit_atlas_home_wins() {
    let h = resolve_from(Some(OsString::from("/tmp/explicit-root")), None).unwrap();
    assert_eq!(h.root, PathBuf::from("/tmp/explicit-root"));
    assert_eq!(h.source, HomeSource::Env);
}

/// An EMPTY `ATLAS_HOME` is an error, not a silent fall back to `$HOME`.
/// Falling back would place artifacts somewhere the operator did not ask for
/// while the variable they set said otherwise.
#[test]
fn an_empty_atlas_home_is_an_error_not_a_fallback() {
    let e = resolve_from(Some(OsString::new()), Some(OsString::from("/home/u"))).unwrap_err();
    assert!(e.to_string().contains("ATLAS_HOME is set but empty"), "{e}");
}

/// POSITIVE: the default is `$HOME/.atlas` — OUR name, not upstream's.
#[test]
fn the_default_is_dot_atlas() {
    let tmp = tempfile::tempdir().unwrap();
    let h = resolve_from(None, Some(tmp.path().into())).unwrap();
    assert_eq!(h.root, tmp.path().join(".atlas"));
    assert_eq!(h.source, HomeSource::HomeDefault);
}

/// The upstream layout is honoured ONLY while ours is absent, so a box that has
/// run an upstream build keeps its artifacts instead of starting empty.
#[test]
fn the_upstream_root_is_used_only_while_ours_does_not_exist() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join(".avarok")).unwrap();
    let h = resolve_from(None, Some(tmp.path().into())).unwrap();
    assert_eq!(h.root, tmp.path().join(".avarok"));
    assert_eq!(h.source, HomeSource::UpstreamHomeDefault);

    // Once ours exists it wins, even with the upstream one still present.
    std::fs::create_dir(tmp.path().join(".atlas")).unwrap();
    let h = resolve_from(None, Some(tmp.path().into())).unwrap();
    assert_eq!(h.root, tmp.path().join(".atlas"));
    assert_eq!(h.source, HomeSource::HomeDefault);
}

/// Neither variable set is an error with a message that names both.
#[test]
fn no_home_at_all_names_both_variables() {
    let e = resolve_from(None, None).unwrap_err();
    let s = e.to_string();
    assert!(s.contains("ATLAS_HOME") && s.contains("HOME"), "{s}");
}
