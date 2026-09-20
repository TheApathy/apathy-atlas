// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../examples/glm53_partial_replay_parity/artifacts.rs"]
mod artifacts;
#[path = "../examples/glm53_partial_replay_parity/compare.rs"]
mod compare;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

fn directory() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "atlas-state-proof-test-{}-{suffix}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    dir
}

#[test]
fn complete_gate_retains_both_first_mismatch_payloads_and_all_region_comparisons() {
    let dir = directory();
    let a = BTreeMap::from([("a".into(), vec![1, 2, 3]), ("z".into(), vec![4, 5])]);
    let b = BTreeMap::from([("a".into(), vec![1, 8, 3]), ("z".into(), vec![4, 9])]);
    let report = artifacts::record_comparison(&dir, "state", &a, &b).unwrap();
    assert_eq!(report["exact"], false);
    assert_eq!(report["regions"].as_array().unwrap().len(), 2);
    assert_eq!(report["retained_raw"]["region"], "a");
    let raw_a =
        fs::read(dir.join(report["retained_raw"]["reference_file"].as_str().unwrap())).unwrap();
    let raw_b =
        fs::read(dir.join(report["retained_raw"]["candidate_file"].as_str().unwrap())).unwrap();
    assert_eq!(raw_a, a["a"]);
    assert_eq!(raw_b, b["a"]);
    assert_eq!(
        compare::digest(&raw_a),
        report["regions"][0]["reference_sha256"]
    );
    assert_eq!(
        compare::digest(&raw_b),
        report["regions"][0]["candidate_sha256"]
    );
    let stored: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!(stored, report);
    assert!(
        artifacts::record_comparison(&dir, "state", &a, &b).is_err(),
        "existing evidence must not be overwritten"
    );
    assert_eq!(
        fs::read(dir.join("state.json")).unwrap(),
        serde_json::to_vec_pretty(&stored).unwrap()
    );
}

#[test]
fn exact_gate_writes_no_raw_sample_and_invalid_labels_or_shapes_write_nothing() {
    let dir = directory();
    let a = BTreeMap::from([("a".into(), vec![1, 2])]);
    for label in ["", "../escape", "state/next", "UPPER", "a.b"] {
        assert!(artifacts::record_comparison(&dir, label, &a, &a).is_err());
    }
    assert!(artifacts::record_comparison(&dir, "bad", &a, &BTreeMap::new()).is_err());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    let report = artifacts::record_comparison(&dir, "before", &a, &a).unwrap();
    assert_eq!(report["exact"], true);
    assert!(report["retained_raw"].is_null());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
}
