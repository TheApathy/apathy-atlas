// SPDX-License-Identifier: AGPL-3.0-only

use super::ReductionPolicy;
use super::bf16_plan_cache::{
    BoundedKeys, GemmKey, KeySlot, MAX_PLANS, PlanRoute, parse_setting, route,
};

#[test]
fn selector_is_strict_and_defaults_off() {
    assert_eq!(parse_setting(None).unwrap(), false);
    assert_eq!(parse_setting(Some("0")).unwrap(), false);
    assert_eq!(parse_setting(Some("1")).unwrap(), true);
    assert!(parse_setting(Some("true")).is_err());
    assert!(parse_setting(Some("2")).is_err());
    assert!(parse_setting(Some("")).is_err());
}

#[test]
fn key_separates_every_descriptor_dimension_and_orientation() {
    let base = GemmKey::new(6, 15, 20, true).unwrap();
    assert_ne!(base, GemmKey::new(7, 15, 20, true).unwrap());
    assert_ne!(base, GemmKey::new(6, 16, 20, true).unwrap());
    assert_ne!(base, GemmKey::new(6, 15, 21, true).unwrap());
    assert_ne!(base, GemmKey::new(6, 15, 20, false).unwrap());
    assert!(GemmKey::new(0, 15, 20, true).is_err());
}

#[test]
fn bounded_index_reuses_keys_and_fails_closed_at_limit() {
    let mut keys = BoundedKeys::default();
    let first = GemmKey::new(1, 1, 1, true).unwrap();
    assert_eq!(keys.locate(first).unwrap(), KeySlot::Vacant);
    keys.record(first).unwrap();
    assert_eq!(keys.locate(first).unwrap(), KeySlot::Existing);

    for n in 2..=MAX_PLANS as u32 {
        keys.record(GemmKey::new(1, n, 1, true).unwrap()).unwrap();
    }
    assert_eq!(keys.len(), MAX_PLANS);
    assert!(keys.locate(GemmKey::new(2, 1, 1, true).unwrap()).is_err());
    assert!(keys.record(first).is_err());
}

#[test]
fn diagnostics_never_enter_the_production_cache() {
    assert_eq!(route(None, false), PlanRoute::Ephemeral);
    assert_eq!(route(None, true), PlanRoute::Cached);
    assert_eq!(
        route(Some(ReductionPolicy::Baseline), true),
        PlanRoute::Ephemeral
    );
    assert_eq!(
        route(Some(ReductionPolicy::ComputeTypeOnly), true),
        PlanRoute::Ephemeral
    );
}
