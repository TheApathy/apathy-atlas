// SPDX-License-Identifier: AGPL-3.0-only

//! RED source seams, not evidence of numerical parity or completed GPU I/O.
//! Pure frame/read-plan behavior and real-state execution have separate gates.

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/model/glm53")
            .join(name),
    )
    .unwrap_or_default()
}

fn compact(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .flat_map(str::chars)
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn item<'a>(text: &'a str, needle: &str) -> &'a str {
    let start = text
        .find(needle)
        .unwrap_or_else(|| panic!("missing {needle}"));
    let open = start + text[start..].find('{').expect("item body");
    let mut depth = 0usize;
    for (at, byte) in text.as_bytes().iter().enumerate().skip(open) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[start..=at];
                }
            }
            _ => {}
        }
    }
    panic!("unclosed {needle}");
}

fn body<'a>(text: &'a str, name: &str) -> &'a str {
    item(text, &format!("fn{name}("))
}

fn before(text: &str, first: &str, second: &str) {
    let a = text
        .find(first)
        .unwrap_or_else(|| panic!("missing {first}"));
    let b = text
        .find(second)
        .unwrap_or_else(|| panic!("missing {second}"));
    assert!(a < b, "{first} must precede {second}");
}

#[test]
fn model_registers_private_read_plan_frame_and_public_typed_diagnostic() {
    let modules = compact(&source("mod.rs"));
    assert!(modules.contains("modstate_read_plan;"));
    assert!(modules.contains("modstate_probe_frame;"));
    let target = compact(&source("target_model_exl3.rs"));
    assert!(target.contains("#[path=\"target_state_probe.rs\"]modtarget_state_probe;"));
    let exports = format!("{modules}{target}");
    for name in ["Glm53StateProbe", "StateProbeRegion"] {
        assert!(
            exports
                .split(';')
                .any(|part| part.contains("pubuse") && part.contains(name)),
            "missing diagnostic re-export {name}"
        );
    }
    let probe = compact(&source("target_state_probe.rs"));
    assert!(probe.contains("#[path=\"target_state_regions.rs\"]modregions;"));
}

#[test]
fn probe_exclusively_borrows_model_and_latches_an_admitted_frame() {
    let probe = compact(&source("target_state_probe.rs"));
    let owner = item(&probe, "pubstructGlm53StateProbe<'a>");
    for field in [
        "model:&'amutGlm53Exl3Model",
        "stamp:StateProbeStamp",
        "stream:u64",
    ] {
        assert!(
            owner.contains(field),
            "missing exclusive owner field {field}"
        );
        assert!(
            !owner.contains(&format!("pub{field}")),
            "probe fields must remain private"
        );
        assert!(!owner.contains(&format!("pub(crate){field}")));
    }
    let constructor = body(&probe, "state_probe");
    assert!(constructor.contains("&mutself"));
    assert!(constructor.contains("Result<Glm53StateProbe<'_>>"));
    assert!(constructor.contains(".state_probe_frame(stream)"));
    assert!(constructor.contains("StateProbeStamp::new("));
    before(constructor, "StateProbeStamp::new(", "Glm53StateProbe{");
    assert!(!constructor.contains(".copy_d2h"));
    assert!(!probe.contains("implCloneforGlm53StateProbe"));
    assert!(!probe.contains("implCopyforGlm53StateProbe"));
}

#[test]
fn every_typed_read_rechecks_stamp_before_resolution_plan_and_owned_copy() {
    let probe = compact(&source("target_state_probe.rs"));
    let read = body(&probe, "read");
    let header = read.split('{').next().unwrap();
    assert!(header.contains("region:StateProbeRegion"));
    assert!(header.contains("offset:usize"));
    assert!(header.contains("destination:&mut[u8]"));
    for forbidden in [
        "DevicePtr",
        "GgmlIqBuffer",
        "*mut",
        "*const",
        "address:",
        "source:",
    ] {
        assert!(
            !header.contains(forbidden),
            "public diagnostic accepts raw authority: {forbidden}"
        );
    }
    assert!(read.contains(".state_probe_frame(self.stream)"));
    assert!(read.contains("self.stamp.check("));
    before(read, "self.stamp.check(", "regions::resolve(");
    before(read, "regions::resolve(", "StateReadPlan::new(");
    before(read, "StateReadPlan::new(", ".copy_state_probe_region(");
    assert!(!read.contains(".copy_d2h"));
    assert!(
        !read.contains(".copy_from_slice("),
        "crop belongs to the owned completion boundary"
    );
}

#[test]
fn frame_is_built_from_current_model_runtime_and_execution_state() {
    let probe = compact(&source("target_state_probe.rs"));
    let frame = body(&probe, "state_probe_frame");
    before(
        frame,
        "stream==self.gpu.default_stream()",
        ".stream_is_capturing(",
    );
    for state in [
        ".state.lock(",
        ".generation",
        ".nonce",
        ".position",
        ".capacity",
        ".live_sequence.load(",
        ".poisoned_stream",
        ".default_stream(",
        ".stream_is_capturing(",
        ".context_tokens(",
        ".context_capacity(",
        "glm53_exact_wide_prefill_active(",
        "glm53_layer_major_prefill_active(",
    ] {
        assert!(
            frame.contains(state),
            "frame omits current authority {state}"
        );
    }
    for field in [
        "generation:",
        "nonce:",
        "position:",
        "context:",
        "capacity:",
        "context_capacity:",
        "model_stream:",
        "live:",
        "poisoned:",
        "capturing:",
        "prefill:",
    ] {
        assert!(frame.contains(field), "frame omits {field}");
    }
    assert!(!frame.contains(".synchronize("));
    assert!(!frame.contains(".copy_d2h"));
}

#[test]
fn typed_regions_resolve_committed_storage_not_staged_verifier_scratch() {
    let probe = compact(&source("target_state_probe.rs"));
    let regions = compact(&source("target_state_regions.rs"));
    let combined = format!("{probe}{regions}");
    let variants = item(&combined, "enumStateProbeRegion");
    for variant in [
        "Kda{",
        "Conv{",
        "DsaLatent{",
        "DsaPoolKeys{",
        "DsaPoolValidity{",
        "DsaTailKeys{",
        "DsaTailGates{",
        "DsaTailValidity{",
        "Capture{",
        "ProjectedContext",
        "Logits{",
    ] {
        assert!(variants.contains(variant), "missing typed region {variant}");
    }
    for retained in [
        ".persistent()",
        ".persistent_state_f32",
        ".latent_cache_bf16",
        ".pool_keys_bf16",
        ".pool_validity_u8",
        ".prior_tail_keys_bf16",
        ".prior_tail_gates_bf16",
        ".prior_tail_validity_u8",
        ".slot_row(",
        ".state_probe_projected(",
    ] {
        assert!(
            regions.contains(retained),
            "resolver misses actual retained source {retained}"
        );
    }
    for transient in [
        ".buffer()",
        ".staged_state_f32",
        ".latent_overlay_bf16",
        ".out_tail_validity_u8",
    ] {
        assert!(
            !regions.contains(transient),
            "resolver substitutes staged scratch {transient}"
        );
    }
    let listing = body(&probe, "regions");
    assert!(!listing.contains("StateProbeRegion::Capture"));
    assert!(!listing.contains("StateProbeRegion::Logits"));
    assert!(!listing.contains(".copy_d2h"));
}

#[test]
fn typed_views_keep_real_parent_allocation_and_live_context_bounds() {
    let regions = compact(&source("target_state_regions.rs"));
    for anchor in [
        ".arena",
        ".plan.known_bytes",
        ".logits",
        "GLM53_EXL3_MAX_WIDE_ROWS",
        "VOCAB",
        "position",
        ".checked_mul(",
        "parent:",
        "view:",
    ] {
        assert!(
            regions.contains(anchor),
            "missing actual allocation/live-view seam {anchor}"
        );
    }
    let runtime = compact(&source("dflash2_runtime.rs"));
    assert!(runtime.contains("#[path=\"dflash2_state_probe.rs\"]"));
    let projected = compact(&source("dflash2_state_probe.rs"));
    let getter = body(&projected, "state_probe_projected");
    assert!(getter.contains("Result<(GgmlIqBuffer,GgmlIqBuffer)>"));
    for anchor in [
        ".arena",
        ".plan.arena_bytes",
        ".plan.projected_target",
        ".context_tokens",
        "HIDDEN",
        ".checked_mul(",
        ".ensure_capture_kv_ready()",
    ] {
        assert!(getter.contains(anchor), "projected view misses {anchor}");
    }
    for forbidden in [".alloc(", ".copy_d2h", ".copy_h2d", ".memset("] {
        assert!(
            !getter.contains(forbidden),
            "projected getter must only resolve: {forbidden}"
        );
    }
}

#[test]
fn diagnostic_copy_reuses_owned_storage_crops_only_after_completion_and_poisons_failures() {
    let readback = compact(&source("target_policy_readback.rs"));
    let copy = body(&readback, "copy_state_probe_region");
    assert!(copy.contains(".verify_readback.lock("));
    assert!(copy.contains("destination.len()"));
    assert!(copy.contains(".logical_range()"));
    before(copy, "destination.len()", ".read(");
    before(copy, "catch_unwind(", ".read(");
    assert!(copy.contains(".read(plan.physical_bytes(),stream,"));
    before(copy, ".read(", "destination.copy_from_slice(");
    assert!(copy.contains("Ok("));
    assert!(copy.contains("Err("));
    assert!(copy.contains(".poison_verify(stream)"));
    assert!(
        !copy.contains(".copy_d2h"),
        "borrowed destination must never reach physical I/O"
    );
    assert!(!copy.contains(".to_vec("));
    for name in [
        "target_state_probe.rs",
        "target_state_regions.rs",
        "dflash2_state_probe.rs",
        "target_policy_readback.rs",
    ] {
        let module = compact(&source(name));
        assert!(
            !module.contains("OwnedReadback::new("),
            "{name} creates another readback owner"
        );
    }
}

#[test]
fn existing_readback_drain_and_request_reset_remain_authoritative_control() {
    let readback = compact(&source("target_policy_readback.rs"));
    let drain = body(&readback, "drain_policy_readback");
    assert!(drain.contains(".verify_readback.lock("));
    before(drain, ".drain(", ".clear_poison()");
    assert!(body(&readback, "drain").contains(".synchronize(stream)"));
    let target = compact(&source("target_model_exl3.rs"));
    let vision = compact(&source("target_vision_exl3.rs"));
    assert!(format!("{target}{vision}").contains(".drain_policy_readback()?"));
    assert!(target.contains("verify_readback:Mutex<OwnedReadback>"));
}
