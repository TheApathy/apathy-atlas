// SPDX-License-Identifier: AGPL-3.0-only
//! Ownership source seams; these supplement real-byte/drop behavior tests.
use std::path::Path;
const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/");

fn source(path: &str) -> String {
    std::fs::read_to_string(Path::new(ROOT).join(path))
        .unwrap_or_default()
        .split_whitespace()
        .collect()
}

#[test]
fn encoder_lock_owns_upload_before_enqueue_and_rejects_poisoned_reuse() {
    let module = source("src/layers/deepseek_vision/mod.rs");
    let forward = source("src/layers/deepseek_vision/forward.rs");
    let owner = source("src/layers/deepseek_vision/encoder_ownership.rs");
    assert!(module.contains("modowned_session;") && module.contains("modencoder_ownership;"));
    assert!(module.contains("Mutex<EncoderOwnership>"));
    assert!(owner.contains("pending_upload:Option<Vec<u8>>"));
    let retain = owner
        .find("pending_upload=Some(pixels)")
        .expect("upload lacks real owner");
    let enqueue = owner
        .find("gpu.copy_h2d(")
        .expect("actual upload bypasses owner");
    assert!(retain < enqueue);
    assert!(owner.contains("OwnedSession<") && owner.contains(".run("));
    assert!(forward.contains(".with_owned_upload("));
    assert!(!forward.contains("gpu.copy_h2d(&pixels,"));
    assert!(!forward.contains("result?;fence?;"));
}

#[test]
fn encoder_release_is_nonconsuming_and_routes_through_retained_owner() {
    let module = source("src/layers/deepseek_vision/mod.rs");
    let owner = source("src/layers/deepseek_vision/encoder_ownership.rs");
    assert!(
        !module.contains("pubfnrelease(self,"),
        "consuming error loses the owner"
    );
    assert!(module.contains("pubfntry_release(&mutself,"));
    assert!(owner.contains(".try_release("));
    assert!(!owner.contains("drain(..)"));
}

#[test]
fn entire_probe_and_stage_readbacks_use_a_single_retained_session_owner() {
    let gpu = source("examples/deepseek_vision_probe/gpu.rs");
    let stages = source("examples/deepseek_vision_probe/stages.rs");
    let owner = source("examples/deepseek_vision_probe/owner.rs");
    assert!(gpu.contains("OwnedSession::new("));
    assert!(gpu.contains(".try_release("));
    assert!(!gpu.contains("encoder.release(gpu)"));
    assert!(
        !gpu.contains("fornameinstore.names()"),
        "raw frees escape session teardown"
    );
    assert!(owner.contains("structProbeResources"));
    for field in [
        "encoder:",
        "weights:",
        "backend:",
        "linear_backend:",
        "pending_readback:",
    ] {
        assert!(owner.contains(field), "complete session missing {field}");
    }
    assert!(owner.contains("fnreadback_owned("));
    assert!(stages.contains(".readback_owned("));
    assert!(!stages.contains("gpu.copy_d2h("));
    assert!(!gpu.contains("gpu.copy_d2h("));
}

#[test]
fn full_model_pending_and_pinned_teardown_need_a_successful_drain() {
    let model = source("src/model/deepseek_vision.rs");
    let drop = source("src/model/drop.rs");
    assert!(model.contains("fntry_release_deepseek_vision(&mutself)"));
    assert!(!model.contains("forimageinpending.drain(..)"));
    assert!(!model.contains("let_=self.gpu.synchronize("));
    assert!(model.contains(".encoder.try_release("));
    let gate = drop
        .find("self.try_release_deepseek_vision()")
        .expect("no release outcome gate");
    let pinned = drop
        .find("self.drop_pinned_staging()")
        .expect("pinned staging path missing");
    assert!(gate < pinned);
    assert!(
        drop.contains("is_ok()"),
        "pinned host free must require successful drain"
    );
}
