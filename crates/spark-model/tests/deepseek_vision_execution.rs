// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/factory/vision_admission.rs"]
mod admission;

use admission::validate_deepseek_vision_execution;

#[test]
fn actual_vision_remains_target_only_for_every_speculative_mode() {
    // Bits represent native MTP, self-spec, n-gram and DFlash/DSpark.
    for modes in 1u8..16 {
        let error =
            validate_deepseek_vision_execution(true, modes == 0, true, 1, false).unwrap_err();
        assert!(error.contains("speculative decoding is not yet qualified"));
        assert!(error.contains("target-only"));
    }
}

#[test]
fn existing_single_gpu_c1_uncached_target_only_recipe_is_accepted() {
    validate_deepseek_vision_execution(true, true, true, 1, false).unwrap();
}

#[test]
fn unsupported_topology_batch_and_cache_are_rejected_without_rewriting() {
    assert!(validate_deepseek_vision_execution(true, true, false, 1, false).is_err());
    for batch in [0, 2, usize::MAX] {
        assert!(validate_deepseek_vision_execution(true, true, true, batch, false).is_err());
    }
    assert!(validate_deepseek_vision_execution(true, true, true, 1, true).is_err());
}

#[test]
fn nonvision_models_do_not_inherit_vision_restrictions() {
    for target_only in [false, true] {
        validate_deepseek_vision_execution(false, target_only, false, 8, true).unwrap();
    }
}

#[test]
fn server_and_factory_reject_before_gpu_weights_or_loader_effects() {
    let server = include_str!("../../spark-server/src/main_modules/serve.rs");
    let gate = server.find("validate_deepseek_vision_execution(").unwrap();
    assert!(server.find("serve_phases::load_model_config(").unwrap() < gate);
    assert!(gate < server.find("serve_phases::init_gpu_backend(").unwrap());
    assert!(gate < server.find("serve_phases::load_weight_store(").unwrap());
    let call = server[gate..].split(".map_err(").next().unwrap();
    for flag in [
        "args.speculative",
        "args.self_speculative",
        "args.ngram_speculative",
        "args.dflash",
    ] {
        assert!(call.contains(flag));
    }
    let factory = include_str!("../src/factory/build.rs");
    let gate = factory.find("validate_deepseek_vision_execution(").unwrap();
    assert!(gate < factory.find("vision_hc_bf16.validate_kernel(").unwrap());
    assert!(gate < factory.find("let loader = loader_for_config(").unwrap());
    assert!(gate < factory.find("loader.load_embedding(").unwrap());
}

#[test]
fn vision_generic_verify_is_always_eager_like_serial_decode() {
    let verify = include_str!("../src/model/trait_impl/verify_d.rs");
    let eligibility = verify.split("let use_graphs =").nth(1).unwrap();
    assert!(
        eligibility
            .split(';')
            .next()
            .unwrap()
            .contains("self.config.deepseek_vision.is_none()")
    );
}

#[test]
fn compressed_drafters_require_full_verification_from_bootstrap() {
    let model = include_str!("../src/model/trait_impl/mod.rs");
    assert!(model.contains("fn requires_full_speculative_verify(&self) -> bool"));
    assert!(model.contains("proposer.requires_full_target_verify()"));
    let head = include_str!("../src/layers/dspark_head.rs");
    let capability = head
        .split("fn requires_full_target_verify(&self) -> bool")
        .nth(1)
        .unwrap();
    assert_eq!(
        capability.split('}').next().unwrap().trim(),
        "{\n        true"
    );
    let scheduler = include_str!("../../spark-server/src/scheduler/mtp_step.rs");
    // Both initial and subsequent chains use one dispatch policy.
    assert_eq!(scheduler.matches("verify_dispatch::dispatch(").count(), 2);
}

#[test]
fn failed_compressor_catchup_cannot_emit_from_corrupt_state() {
    let verify = include_str!("../../spark-server/src/scheduler/verify_dflash_step.rs");
    let error_arm = verify
        .split("tracing::error!(\"dspark_compress_catchup: {e:#}\");")
        .nth(1)
        .unwrap();
    let error_arm = error_arm.split('}').next().unwrap();
    assert!(error_arm.contains("a.finished = true;"));
    assert!(error_arm.contains("return;"));
}
