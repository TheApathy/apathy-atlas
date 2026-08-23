// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source-contract guard for SSM decode-rollback constructor wiring.

const MODEL_INIT: &str = include_str!("../src/model/impl_a1.rs");
const SNAPSHOT_POOL: &str = include_str!("../src/model/ssm_snapshot.rs");
const MODEL_TRAIT_IMPL: &str = include_str!("../src/model/trait_impl/mod.rs");
const PREFLIGHT: &str =
    include_str!("../../spark-server/src/main_modules/serve_phases/preflight.rs");
const SCHEDULER_PROMOTE: &str =
    include_str!("../../spark-server/src/scheduler/phase_promote_prefills.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn parenthesized_call<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing call: {marker}"));
    let open = source[start..]
        .find('(')
        .map(|offset| start + offset)
        .expect("call must have an opening parenthesis");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("call must have a closing parenthesis: {marker}");
}

fn top_level_args(call: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (index, byte) in call.as_bytes().iter().enumerate() {
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                args.push(compact(&call[start..index]));
                start = index + 1;
            }
            _ => {}
        }
    }
    if !call[start..].trim().is_empty() {
        args.push(compact(&call[start..]));
    }
    args
}

fn has_checked_preflight_capacity_contract(source: &str) -> bool {
    let source = compact(source);
    [
        "letdecode_ring_slots=ifnum_ssm_layers>0{(atlas_kernels::ROLLBACK_RESTEER_CAPasusize).checked_add(1).context(\"SSMpreflightdecoderingslotcountoverflow\")?}else{0};",
        "letdecode_region=checked_preflight_mul(max_batch_size,decode_ring_slots,\"decoderegionslots\")?;",
        "letsnapshot_slots=checked_preflight_add(ssm_cache_slots,decode_region,\"snapshotslots\")?;",
        "letlayer_bytes=checked_preflight_mul(num_ssm_layers,bytes_per_layer,\"snapshotlayerbytes\")?;",
        "checked_preflight_mul(snapshot_slots,layer_bytes,\"snapshottotalbytes\")?",
    ]
    .iter()
    .all(|required| source.contains(required))
}

#[test]
fn model_constructor_matches_preflight_capacity_authority() {
    let args = top_level_args(parenthesized_call(MODEL_INIT, "SsmSnapshotPool::new("));
    assert_eq!(args.len(), 7, "snapshot constructor ABI drifted: {args:?}");
    assert_eq!(
        args[4], "(atlas_kernels::ROLLBACK_RESTEER_CAPasusize)+1",
        "constructor ABI must retain the shared rollback cap plus boundary slot"
    );
    assert_eq!(
        args[5], "max_batch_size",
        "decode ring must reserve one region for every admitted batch slot"
    );

    assert!(
        has_checked_preflight_capacity_contract(PREFLIGHT),
        "startup admission must own checked ring and snapshot capacity arithmetic"
    );

    let capacity_args = top_level_args(parenthesized_call(
        PREFLIGHT,
        "let ssm_capacity = checked_ssm_preflight_capacity(",
    ));
    assert_eq!(
        capacity_args,
        [
            "pool_geometry.total_bytes",
            "args.max_batch_size",
            "num_ssm_layers",
            "pool_geometry.h_bytes",
            "pool_geometry.conv_bytes",
            "args.ssm_cache_slots",
            "decode_ring_slots",
        ],
        "preflight helper inputs drifted from the snapshot constructor geometry"
    );
}

#[test]
fn unchecked_preflight_arithmetic_mutants_fail_the_contract() {
    let checked = compact(PREFLIGHT);
    let unchecked_add = checked.replace(
        "(atlas_kernels::ROLLBACK_RESTEER_CAPasusize).checked_add(1).context(\"SSMpreflightdecoderingslotcountoverflow\")?",
        "(atlas_kernels::ROLLBACK_RESTEER_CAPasusize)+1",
    );
    assert_ne!(
        unchecked_add, checked,
        "checked-add mutation must be effective"
    );
    assert!(!has_checked_preflight_capacity_contract(&unchecked_add));

    let unchecked_product = checked.replace(
        "letdecode_region=checked_preflight_mul(max_batch_size,decode_ring_slots,\"decoderegionslots\")?;",
        "letdecode_region=decode_ring_slots*max_batch_size;",
    );
    assert_ne!(
        unchecked_product, checked,
        "checked-product mutation must be effective"
    );
    assert!(!has_checked_preflight_capacity_contract(&unchecked_product));
}

#[test]
fn hybrid_models_expose_the_nonzero_ring_to_the_scheduler() {
    let pool = compact(SNAPSHOT_POOL);
    assert!(
        pool.contains(
            "letdecode_enabled=num_ssm_layers>0&&decode_ring_slots>0&&decode_max_seqs>0;"
        )
    );
    assert!(pool.contains("decode_ring_slots:ifdecode_enabled{decode_ring_slots}else{0}"));
    assert!(pool.contains("self.decode_ring_slots>0&&!self.decode_h_snapshots.is_empty()"));

    let model = compact(MODEL_TRAIT_IMPL);
    assert!(model.contains("fnhas_ssm_layers(&self)->bool{self.ssm_pool.num_ssm_layers>0}"));
    assert!(model.contains(
        "fndecode_rollback_ring_slots(&self)->usize{ifself.ssm_snapshots.decode_rollback_enabled(){self.ssm_snapshots.decode_ring_slots}else{0}}"
    ));

    let scheduler = compact(SCHEDULER_PROMOTE);
    assert!(scheduler.contains("model.decode_rollback_ring_slots(),"));
    assert!(scheduler.contains("ssm_rollback_ring:SsmDecodeRing::new(ssm_ring_capacity),"));

    let ring_slots = (atlas_kernels::ROLLBACK_RESTEER_CAP as usize)
        .checked_add(1)
        .expect("the shared cap must leave room for one boundary snapshot");
    assert!(ring_slots > 0);
    for max_batch_size in [1usize, 8, 64] {
        assert_eq!(
            ring_slots.checked_mul(max_batch_size),
            Some(ring_slots * max_batch_size)
        );
    }
}

#[test]
fn disabled_boundaries_do_not_report_a_hybrid_ring() {
    let decode_enabled = |num_ssm_layers: usize, ring_slots: usize, max_batch: usize| {
        num_ssm_layers > 0 && ring_slots > 0 && max_batch > 0
    };
    let ring_slots = (atlas_kernels::ROLLBACK_RESTEER_CAP as usize)
        .checked_add(1)
        .expect("the shared cap must fit usize");

    assert!(!decode_enabled(0, ring_slots, 8));
    assert!(!decode_enabled(1, 0, 8));
    assert!(!decode_enabled(1, ring_slots, 0));
    assert!(decode_enabled(1, ring_slots, 8));
    assert_eq!(ring_slots.checked_mul(0), Some(0));
    assert_eq!(ring_slots.checked_mul(usize::MAX), None);
}
