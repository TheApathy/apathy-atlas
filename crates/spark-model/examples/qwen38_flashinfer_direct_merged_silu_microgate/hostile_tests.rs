// SPDX-License-Identifier: AGPL-3.0-only

use super::authority::HeldFile;
use super::guarded::{disjoint_sentinel, merged_input, output_sentinel};
use super::provenance::sha256_bytes;
use super::timing::summarize;
use super::{K, Plan, SCALE2};

const ENTRY: &str = include_str!("../qwen38_flashinfer_direct_merged_silu_microgate.rs");
const GUARDED: &str = include_str!("guarded.rs");
const PROVENANCE: &str = include_str!("provenance.rs");
const AUTHORITY: &str = include_str!("authority.rs");
const OWNER: &str = include_str!("owner.rs");
const RUNTIME: &str = include_str!("runtime.rs");
const RUNTIME_HELPERS: &str = include_str!("runtime_helpers.inc.rs");
const SCHEDULER: &str = include_str!("scheduler_authority.rs");
const TIMING: &str = include_str!("timing.rs");
const RUNTIME_SHA256: &str = "4eae11c1b6efb02b9d1df49d236b010ad3d8e5def4047c1e4b6f8f7c7eee64f1";
const RUNTIME_HELPERS_SHA256: &str =
    "249b221a5d8cec4020e9ed334eb591d4af2157ade02d9e5092678f2b20cad8a2";

#[test]
fn distinct_sentinels_reject_no_write_partial_stride_and_swap() {
    let a = output_sentinel(65_537, 0x31);
    let b = output_sentinel(65_537, 0xc7);
    assert!(a.iter().zip(&b).all(|(left, right)| left != right));
    assert_ne!(a, b, "two no-write outputs must never compare equal");
    let mut partial_a = a.clone();
    let mut partial_b = b.clone();
    partial_a[..4099].fill(0x42);
    partial_b[..4099].fill(0x42);
    assert_ne!(partial_a, partial_b);
    let mut stride = a.clone();
    stride.rotate_left(257);
    assert_ne!(a, stride);
    assert_ne!(a[..32768], b[32768..65536]);
    let expected = (0..=255).cycle().take(a.len()).collect::<Vec<u8>>();
    let replay = disjoint_sentinel(&expected, 0x5b);
    assert!(
        replay
            .iter()
            .zip(expected)
            .all(|(left, right)| *left != right)
    );
}

#[test]
fn merged_fixture_is_nonperiodic_and_row_column_side_sensitive() {
    let cols = 257;
    let bytes = merged_input(17, cols);
    let row_bytes = cols as usize * 4;
    for row in 1..17usize {
        assert_ne!(
            &bytes[..row_bytes],
            &bytes[row * row_bytes..(row + 1) * row_bytes]
        );
    }
    assert_ne!(&bytes[..16], &bytes[16..32]);
    assert_ne!(
        &bytes[..cols as usize * 2],
        &bytes[cols as usize * 2..row_bytes]
    );
    let shifted = bytes[2..]
        .iter()
        .chain(&bytes[..2])
        .copied()
        .collect::<Vec<_>>();
    assert_ne!(bytes, shifted);
}

#[test]
fn timing_gate_rejects_noise_and_sign_instability() {
    let plan = Plan::checked(2079, K, 21, SCALE2).unwrap();
    let stable_parent = vec![10.0; plan.reps];
    assert!(summarize(plan, &stable_parent, &vec![9.5; plan.reps]).is_ok());
    assert!(summarize(plan, &stable_parent, &vec![9.9999; plan.reps]).is_err());
    let unstable = (0..plan.reps)
        .map(|index| if index % 2 == 0 { 9.0 } else { 11.0 })
        .collect::<Vec<_>>();
    assert!(summarize(plan, &stable_parent, &unstable).is_err());
}

#[test]
fn held_descriptor_detects_same_length_drift() {
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let path = std::env::temp_dir().join(format!("atlas-held-{unique}"));
    std::fs::write(&path, b"alpha").unwrap();
    let held = HeldFile::open(&path, None).unwrap();
    std::fs::write(&path, b"bravo").unwrap();
    assert!(held.verify_unchanged().is_err());
    std::fs::remove_file(&path).unwrap();
}

fn mutate_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "critical seam must be unique"
    );
    source.replacen(from, to, 1)
}

const PACKED: &str = "equal(\"packed\", &parent_packed_bytes, &candidate_packed_bytes)?;";
const IMMUTABLE: &str = "merged.immutable(gpu, \"merged\")?;";
const CLEANUP: &str = "owner.finish(gpu, outcome)";
const CONFIDENCE: &str = "timing.paired_lower95_ms >= timing.minimum_absolute_ms";
const ROUTE: &str = "(\"crates/spark-model/src/layers/dense_ffn.rs\", ROUTE_SHA)";

fn static_accepts(runtime: &str, helpers: &str, timing: &str, provenance: &str) -> bool {
    sha256_bytes(runtime.as_bytes()).is_ok_and(|hash| hash == RUNTIME_SHA256)
        && sha256_bytes(helpers.as_bytes()).is_ok_and(|hash| hash == RUNTIME_HELPERS_SHA256)
        && [
            runtime.matches(PACKED).count(),
            helpers.matches(IMMUTABLE).count(),
            runtime.matches(CLEANUP).count(),
            timing.matches(CONFIDENCE).count(),
            provenance.matches(ROUTE).count(),
        ] == [1; 5]
        && runtime.find("let outcome = run_owned(") < runtime.find(CLEANUP)
        && runtime.find("complete_nonalias(&[") < runtime.find("launch(\n        false")
        && helpers.find("fn postchecks(") < helpers.find(IMMUTABLE)
        && helpers.find(IMMUTABLE) < helpers.rfind("Ok(())")
}

#[test]
fn exact_static_oracle_rejects_five_previous_false_accepts() {
    assert!(static_accepts(RUNTIME, RUNTIME_HELPERS, TIMING, PROVENANCE));
    let packed = mutate_once(
        RUNTIME,
        PACKED,
        "equal(\"packed\", &parent_packed_bytes, &parent_packed_bytes)?;",
    );
    let immutable = mutate_once(RUNTIME_HELPERS, IMMUTABLE, "");
    let cleanup = mutate_once(RUNTIME, CLEANUP, "outcome.map_err(Into::into)");
    let confidence = mutate_once(TIMING, CONFIDENCE, "timing.paired_median_ms > 0.0");
    let route = mutate_once(PROVENANCE, ROUTE, "");
    let disabled_packed = mutate_once(RUNTIME, PACKED, &format!("if false {{ {PACKED} }}"));
    let disabled_immutable = mutate_once(
        RUNTIME_HELPERS,
        IMMUTABLE,
        &format!("if false {{ {IMMUTABLE} }}"),
    );
    assert!(!static_accepts(
        &packed,
        RUNTIME_HELPERS,
        TIMING,
        PROVENANCE
    ));
    assert!(!static_accepts(RUNTIME, &immutable, TIMING, PROVENANCE));
    assert!(!static_accepts(
        &cleanup,
        RUNTIME_HELPERS,
        TIMING,
        PROVENANCE
    ));
    assert!(!static_accepts(
        RUNTIME,
        RUNTIME_HELPERS,
        &confidence,
        PROVENANCE
    ));
    assert!(!static_accepts(RUNTIME, RUNTIME_HELPERS, TIMING, &route));
    assert!(!static_accepts(
        &disabled_packed,
        RUNTIME_HELPERS,
        TIMING,
        PROVENANCE
    ));
    assert!(!static_accepts(
        RUNTIME,
        &disabled_immutable,
        TIMING,
        PROVENANCE
    ));
    for required in [
        "RELEASE_SCHEDULER_MANIFEST_PATH",
        "File::from_raw_fd(ticket_fd)",
        "verify_unchanged",
        "release_with",
        "parent_samples_ms",
    ] {
        assert!(
            [
                ENTRY, GUARDED, PROVENANCE, AUTHORITY, OWNER, RUNTIME, SCHEDULER, TIMING
            ]
            .concat()
            .contains(required)
        );
    }
}

#[test]
fn ownership_and_pre_effect_census_are_exactly_ordered() {
    let register = GUARDED.find("owner.allocate(gpu, image.len())?").unwrap();
    let copy = GUARDED.find("gpu.copy_h2d(&image, base)?").unwrap();
    assert!(
        register < copy,
        "copy failure must leave a registered owner"
    );
    let reserve = OWNER.find("self.live.try_reserve(1)?").unwrap();
    let allocate = OWNER.find("let ptr = gpu.alloc(bytes)?").unwrap();
    let publish_owner = OWNER.find("self.live.push(ptr)").unwrap();
    assert!(reserve < allocate && allocate < publish_owner);
    let census = RUNTIME.find("complete_nonalias(&[").unwrap();
    let first_launch = RUNTIME.find("launch(\n        false").unwrap();
    assert!(census < first_launch);
    let census_body = &RUNTIME[census..first_launch];
    for buffer in [
        "parent_logical_scales",
        "candidate_logical_scales",
        "parent_output",
        "candidate_output",
        "weight_scales",
    ] {
        assert!(census_body.contains(buffer), "census omitted {buffer}");
    }
    assert!(OWNER.contains("while cursor != 0"));
    assert!(OWNER.contains("self.live.remove(cursor)"));
    assert!(OWNER.contains("first_error = Some(error)"));
    assert!(
        RUNTIME.find("let outcome = run_owned(").unwrap()
            < RUNTIME.find("owner.finish(gpu, outcome)").unwrap()
    );
    assert!(ENTRY.contains("failure.retry_cleanup(gpu).is_ok()"));
    assert!(ENTRY.contains("return Err(failure.into())"));
}
