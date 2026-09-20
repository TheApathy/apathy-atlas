// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

#[path = "prefix_entry_contract_static/manifest.rs"]
mod manifest;
use manifest::CHILDREN;

const ROOT: &str = include_str!("../src/model/trait_impl/prefill_b.rs");
const DIRECT_FINALIZE: &str = include_str!("../src/model/trait_impl/prefill_d.rs");
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

fn validate_marconi_save_elision(
    direct_finalize: &str,
    chunk_finalize: &str,
    checkpoint: &str,
) -> Result<(), String> {
    const PRIMARY_SAVE: &str = "letsnap_result=matchself.ssm_snapshots.save(";
    const LEAF_GUARD: &str = "ifself.prefix_cache.is_active()&&self.ssm_snapshots.is_enabled(){letsnap_result=matchself.ssm_snapshots.save(";
    const CHECKPOINT_GUARD: &str = "ifself.ssm_checkpoint_interval==0||!self.prefix_cache.is_active()||!self.ssm_snapshots.is_enabled(){returnOk(());}";

    for (route, source, expected_saves) in [
        ("direct finalize", compact(direct_finalize), 2),
        ("chunk finalize", compact(chunk_finalize), 1),
    ] {
        let primary_saves = source.matches(PRIMARY_SAVE).count();
        if primary_saves != expected_saves {
            return Err(format!(
                "{route} primary Marconi save count drift: expected={expected_saves} actual={primary_saves}"
            ));
        }
        let guarded_saves = source.matches(LEAF_GUARD).count();
        if guarded_saves != primary_saves {
            return Err(format!(
                "{route} admits a Marconi save while prefix caching is inactive"
            ));
        }
    }

    let checkpoint = compact(checkpoint);
    if checkpoint.matches(PRIMARY_SAVE).count() != 1 {
        return Err("intermediate Marconi save count drift".into());
    }
    let guard = checkpoint.find(CHECKPOINT_GUARD).ok_or_else(|| {
        "intermediate Marconi save lacks the inactive-prefix early return".to_string()
    })?;
    let save = checkpoint
        .find(PRIMARY_SAVE)
        .ok_or_else(|| "intermediate Marconi save disappeared".to_string())?;
    if guard >= save {
        return Err("intermediate inactive-prefix guard follows its Marconi save".into());
    }
    Ok(())
}

fn replace_nth(source: &str, needle: &str, replacement: &str, nth: usize) -> String {
    let start = source
        .match_indices(needle)
        .nth(nth)
        .map(|(start, _)| start)
        .unwrap_or_else(|| panic!("missing mutation target {needle:?} occurrence {nth}"));
    let mut mutated = String::with_capacity(source.len() - needle.len() + replacement.len());
    mutated.push_str(&source[..start]);
    mutated.push_str(replacement);
    mutated.push_str(&source[start + needle.len()..]);
    mutated
}

fn leaf_save_admitted(prefix_cache_active: bool, snapshots_enabled: bool) -> bool {
    prefix_cache_active && snapshots_enabled
}

fn checkpoint_returns_early(
    checkpoint_interval: usize,
    prefix_cache_active: bool,
    snapshots_enabled: bool,
) -> bool {
    checkpoint_interval == 0 || !prefix_cache_active || !snapshots_enabled
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
    validate_marconi_save_elision(
        DIRECT_FINALIZE,
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

#[test]
fn inactive_prefix_cache_cannot_reach_any_marconi_save() {
    for occurrence in 0..2 {
        let mutated = replace_nth(
            DIRECT_FINALIZE,
            "self.prefix_cache.is_active()",
            "true",
            occurrence,
        );
        assert!(
            validate_marconi_save_elision(
                &mutated,
                child("finalize_last"),
                child("save_checkpoint")
            )
            .is_err(),
            "accepted removal of direct-finalize active-prefix guard {occurrence}"
        );
    }

    let chunk_mutated = replace_nth(
        child("finalize_last"),
        "self.prefix_cache.is_active()",
        "true",
        0,
    );
    assert!(
        validate_marconi_save_elision(DIRECT_FINALIZE, &chunk_mutated, child("save_checkpoint"))
            .is_err(),
        "accepted removal of chunk-finalize active-prefix guard"
    );

    let checkpoint_mutated = replace_nth(
        child("save_checkpoint"),
        "!self.prefix_cache.is_active()",
        "false",
        0,
    );
    assert!(
        validate_marconi_save_elision(DIRECT_FINALIZE, child("finalize_last"), &checkpoint_mutated)
            .is_err(),
        "accepted removal of intermediate active-prefix guard"
    );
}

#[test]
fn active_prefix_cache_preserves_prior_save_admission() {
    for snapshots_enabled in [false, true] {
        let prior_leaf_admission = snapshots_enabled;
        let guarded_leaf_admission = leaf_save_admitted(true, snapshots_enabled);
        assert_eq!(guarded_leaf_admission, prior_leaf_admission);

        for checkpoint_interval in [0, 1, 128] {
            let prior_checkpoint_return = checkpoint_interval == 0 || !snapshots_enabled;
            let guarded_checkpoint_return =
                checkpoint_returns_early(checkpoint_interval, true, snapshots_enabled);
            assert_eq!(guarded_checkpoint_return, prior_checkpoint_return);
        }
    }

    assert!(!leaf_save_admitted(false, true));
    assert!(checkpoint_returns_early(0, false, true));
    assert!(checkpoint_returns_early(1, false, true));
}
