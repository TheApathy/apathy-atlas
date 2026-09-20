// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../examples/glm53_partial_replay_parity/compare.rs"]
mod compare;
use compare::compare_snapshots;
use std::collections::BTreeMap;

fn snapshot() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        ("kda-0".into(), vec![1, 2, 3, 4]),
        ("tail-validity-0".into(), vec![1, 0, 1]),
        ("empty-pool".into(), vec![]),
    ])
}

#[test]
fn identical_complete_regions_include_empty_spans_and_hash_every_byte() {
    let a = snapshot();
    let result = compare_snapshots(&a, &a).unwrap();
    assert_eq!(result["exact"], true);
    assert_eq!(result["regions"].as_array().unwrap().len(), 3);
    for region in result["regions"].as_array().unwrap() {
        assert_eq!(region["different_bytes"], 0);
        assert_eq!(region["reference_sha256"], region["candidate_sha256"]);
        assert_eq!(region["reference_sha256"].as_str().unwrap().len(), 64);
    }
}

#[test]
fn missing_extra_renamed_truncated_and_empty_manifests_cannot_pass_as_comparisons() {
    let a = snapshot();
    let empty = BTreeMap::new();
    assert!(compare_snapshots(&empty, &empty).is_err());
    let mut b = a.clone();
    b.remove("empty-pool");
    assert!(compare_snapshots(&a, &b).is_err());
    let mut b = a.clone();
    b.insert("extra".into(), vec![]);
    assert!(compare_snapshots(&a, &b).is_err());
    let mut b = a.clone();
    let removed = b.remove("tail-validity-0").unwrap();
    b.insert("renamed".into(), removed);
    assert!(compare_snapshots(&a, &b).is_err());
    let mut b = a.clone();
    b.get_mut("kda-0").unwrap().pop();
    assert!(compare_snapshots(&a, &b).is_err());
}

#[test]
fn last_byte_and_multiple_region_mismatches_are_retained_not_filtered() {
    let a = snapshot();
    let mut b = a.clone();
    b.get_mut("kda-0").unwrap()[3] = 99;
    b.get_mut("tail-validity-0").unwrap()[2] = 0;
    let result = compare_snapshots(&a, &b).unwrap();
    assert_eq!(result["exact"], false);
    let changed = result["regions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["different_bytes"] != 0)
        .collect::<Vec<_>>();
    assert_eq!(changed.len(), 2);
    assert_eq!(changed[0]["first_byte"], 3);
    assert_eq!(changed[1]["first_byte"], 2);
    for region in changed {
        assert_ne!(region["reference_sha256"], region["candidate_sha256"]);
    }
}
