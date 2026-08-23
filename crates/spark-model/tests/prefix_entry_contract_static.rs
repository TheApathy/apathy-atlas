// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

#[path = "prefix_entry_contract_static/manifest.rs"]
mod manifest;
use manifest::CHILDREN;

const ROOT: &str = include_str!("../src/model/trait_impl/prefill_b.rs");
const PREFIX_SEAM: &str = include_str!("../src/model/trait_impl/prefix_hit.rs");
const MIXED: &str = include_str!("../src/model/trait_impl/decode_b.rs");
const MODEL_TRAIT: &str = include_str!("../src/model/trait_impl/mod.rs");
const STANDARD_SCHEDULER: &str =
    include_str!("../../spark-server/src/scheduler/phase_continue_prefills/run_standard.rs");
const BATCH_SCHEDULER: &str =
    include_str!("../../spark-server/src/scheduler/phase_continue_prefills/run_batched_prefill.rs");
const PARENT_DIRECT: &str = "fnprefill_batch_chunk(&self,streams:&mut[PrefillSlice<'_>],stream:u64,)->Result<Vec<DevicePtr>>{self.prefill_batch_chunk_dispatch(streams,stream)}";
const SINGLE_HOOKS: &[&str] = &[
    "self.prefill_b_embed_chunk(",
    "self.prefill_b_prefix_lookup(",
    "self.prefill_b_proc_range(",
    "self.prefill_b_upload_meta(",
    "self.prefill_b_upload_paged(",
    "self.prefill_b_forward_layers(",
    "self.prefill_b_finalize_last(",
    "self.prefill_b_save_checkpoint(",
];
const BATCH_HOOKS: &[&str] = &[
    "return self.prefill_batch_chunk_kernel_batched(",
    "self.prefill_b_embed_chunk(",
    "self.prefill_b_prefix_lookup(",
    "self.prefill_b_proc_range(",
    "self.prefill_b_upload_meta(",
    "self.prefill_b_upload_paged(",
    "self.prefill_b_forward_layers(",
    "self.prefill_b_finalize_last(",
    "self.prefill_b_save_checkpoint(",
];
const KERNEL_HOOKS: &[&str] = &[
    "if chunk_start == 0 || cached_prefix_tokens > 0 || marconi_skip_to > 0 {",
    "self.prefill_b_embed_chunk_at(",
    "self.prefill_b_prefix_lookup(",
    "self.prefill_b_proc_range(",
    "self.prefill_b_upload_meta_at(",
    "self.prefill_b_upload_paged(",
    "self.stage_batched_attn_metadata(",
    "self.prefill_ssm_batched_layer(",
    "self.prefill_attn_batched_layer(",
    "self.prefill_b_finalize_last_at(",
    "self.prefill_b_save_checkpoint(",
];
fn child(name: &str) -> &'static str {
    CHILDREN
        .iter()
        .find_map(|(candidate, source)| (*candidate == name).then_some(*source))
        .unwrap_or_else(|| panic!("missing child source: {name}"))
}
fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}
fn declared_children(root: &str) -> BTreeSet<String> {
    root.lines()
        .filter_map(|line| line.trim().strip_prefix("mod "))
        .filter_map(|line| line.strip_suffix(';'))
        .map(str::to_owned)
        .collect()
}
fn validate_manifest(root: &str, manifest: &[(&str, &str)]) -> Result<(), String> {
    let expected = declared_children(root);
    let actual: BTreeSet<String> = manifest.iter().map(|(name, _)| (*name).into()).collect();
    if actual.len() != manifest.len() {
        return Err("duplicate child in prefix manifest".into());
    }
    (actual == expected).then_some(()).ok_or_else(|| {
        format!("prefix child manifest drift: expected={expected:?} actual={actual:?}")
    })
}
fn validate_hooks(source: &str, hooks: &[&str]) -> Result<(), String> {
    let source = compact(source);
    for hook in hooks {
        if !source.contains(&compact(hook)) {
            return Err(format!("missing production call edge: {hook}"));
        }
    }
    Ok(())
}
fn validate_paired_contract(
    seam: &str,
    lookup: &str,
    finalize: &str,
    checkpoint: &str,
) -> Result<(), String> {
    let seam = compact(seam);
    let lookup = compact(lookup);
    let finalize = compact(finalize);
    let checkpoint = compact(checkpoint);
    for required in [
        "self.config.num_ssm_layers()>0",
        ".lookup_paired(tokens,block_size,session_hash)",
    ] {
        if !seam.contains(required) {
            return Err(format!("hybrid lookup seam drift: {required}"));
        }
    }
    for required in [
        "self.lookup_prefill_prefix(tokens,bs,seq.session_hash)",
        ".session_matches(snap_id,seq.session_hash)",
        "super::attention_only_prefix_skip(",
        "super::restored_prefix_skip_tokens(",
    ] {
        if !lookup.contains(required) {
            return Err(format!("paired lookup/replay drift: {required}"));
        }
    }
    if lookup.matches(".release_matched(").count() < 2 {
        return Err("EP demotion can leak an acquired prefix ref".into());
    }
    if lookup.contains("self.prefix_cache.lookup(") {
        return Err("raw KV lookup entered a hybrid-eligible route".into());
    }
    if !finalize.contains("self.prefix_cache.insert_with_snapshot(")
        || !finalize.contains("snap_id,seq.session_hash,seq.cached_prefix_tokens")
        || !checkpoint.contains("self.prefix_cache.insert_intermediate_snapshot(")
        || !checkpoint.contains("snap_id,seq.session_hash,end_token")
    {
        return Err("snapshot insertion lost its session-paired KV edge".into());
    }
    Ok(())
}

fn validate_mixed_bypass(source: &str) -> Result<(), String> {
    let fused = source
        .split_once("PREFIX_CACHE_POLICY: BYPASS")
        .map(|(_, fused)| compact(fused))
        .ok_or_else(|| "mixed prefill cache bypass is undocumented".to_string())?;
    for forbidden in [
        "lookup_prefill_prefix(",
        "prefill_b_prefix_lookup(",
        ".lookup_paired(",
        ".prefix_cache.insert(",
        ".insert_with_snapshot(",
        ".insert_intermediate_snapshot(",
    ] {
        if fused.contains(forbidden) {
            return Err(format!(
                "mixed bypass gained unsafe cache edge: {forbidden}"
            ));
        }
    }
    for required in [
        "layer.decode_multi_seq(",
        "layer.prefill(",
        "0,//kv_write_start:cachebypasswritesallcurrentrows",
        "prefill_seq.tokens.extend_from_slice(",
        "prefill_seq.seq_len=prefill_chunk_start+n_prefill;",
    ] {
        if !fused.contains(required) {
            return Err(format!("mixed bypass skips model state: {required}"));
        }
    }
    Ok(())
}

#[test]
fn manifest_and_every_production_entry_are_complete() {
    validate_manifest(ROOT, CHILDREN).unwrap();
    validate_hooks(ROOT, SINGLE_HOOKS).unwrap();
    validate_hooks(child("batch"), BATCH_HOOKS).unwrap();
    validate_hooks(child("batch_kernel"), KERNEL_HOOKS).unwrap();
    validate_paired_contract(
        PREFIX_SEAM,
        child("prefix_lookup"),
        child("finalize_last"),
        child("save_checkpoint"),
    )
    .unwrap();
    validate_mixed_bypass(MIXED).unwrap();

    let model = compact(MODEL_TRAIT);
    assert!(model.contains("self.mixed_forward_dispatch("));
    assert!(model.contains(PARENT_DIRECT));
    assert!(compact(STANDARD_SCHEDULER).contains("model.mixed_forward("));
    assert!(compact(BATCH_SCHEDULER).contains("model.prefill_batch_chunk("));
}

#[test]
fn omitted_children_and_route_edges_fail_closed() {
    for omitted in 0..CHILDREN.len() {
        let mut manifest = CHILDREN.to_vec();
        manifest.remove(omitted);
        assert!(validate_manifest(ROOT, &manifest).is_err());
    }
    assert!(validate_manifest(&format!("{ROOT}\nmod omitted_child;"), CHILDREN).is_err());

    for (source, hooks) in [
        (ROOT, SINGLE_HOOKS),
        (child("batch"), BATCH_HOOKS),
        (child("batch_kernel"), KERNEL_HOOKS),
    ] {
        let compact_source = compact(source);
        for hook in hooks {
            let mutated = compact_source.replacen(&compact(hook), "removed_edge(", 1);
            assert!(
                validate_hooks(&mutated, hooks).is_err(),
                "accepted removal of {hook}"
            );
        }
    }
    let retry = MODEL_TRAIT.replacen(
        "self.prefill_batch_chunk_dispatch(streams, stream)",
        "match self.prefill_batch_chunk_dispatch(streams, stream)",
        1,
    );
    assert!(!compact(&retry).contains(PARENT_DIRECT));
}

#[test]
fn raw_lookup_session_ref_and_mixed_cache_mutations_are_rejected() {
    let raw = PREFIX_SEAM.replacen("lookup_paired", "lookup", 1);
    assert!(
        validate_paired_contract(
            &raw,
            child("prefix_lookup"),
            child("finalize_last"),
            child("save_checkpoint")
        )
        .is_err()
    );
    for marker in [
        "session_matches",
        "release_matched",
        "restored_prefix_skip_tokens",
    ] {
        let mutated = child("prefix_lookup").replace(marker, "removed_contract");
        assert!(
            validate_paired_contract(
                PREFIX_SEAM,
                &mutated,
                child("finalize_last"),
                child("save_checkpoint")
            )
            .is_err()
        );
    }
    let injected = MIXED.replace(
        "PREFIX_CACHE_POLICY: BYPASS",
        "PREFIX_CACHE_POLICY: BYPASS\nself.lookup_prefill_prefix(tokens, bs, session_hash);",
    );
    assert!(validate_mixed_bypass(&injected).is_err());
}
