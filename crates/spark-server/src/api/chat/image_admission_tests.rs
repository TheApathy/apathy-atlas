// SPDX-License-Identifier: AGPL-3.0-only

#[path = "image_admission.rs"]
mod admission;

#[test]
fn text_requests_do_not_inherit_vision_limits() {
    assert!(admission::validate(0, false, 8, true).is_ok());
}

#[test]
fn images_require_real_vision_single_sequence_and_supported_context() {
    assert!(admission::validate(1, true, 1, false).is_ok());
    assert!(admission::validate(8, true, 1, false).is_ok());
    assert!(admission::validate(1, false, 1, false).is_err());
    assert!(admission::validate(1, true, 0, false).is_err());
    assert!(admission::validate(1, true, 2, false).is_err());
    assert!(admission::validate(1, true, 1, true).is_err());
    assert!(admission::validate(9, true, 1, false).is_err());
}

#[test]
fn image_rejection_precedes_failure_rewrites_and_preprocessing() {
    let source = include_str!("mod.rs");
    let guard = source.find("image_admission::validate(").unwrap();
    assert!(guard < source.find("apply_failure_guards(").unwrap());
    assert!(guard < source.find("build_msg_entries(").unwrap());
}

#[test]
fn text_only_token_counters_reject_images_instead_of_undercounting() {
    let source = include_str!("../../anthropic/handlers.rs");
    let count = source.split("pub async fn count_tokens(").nth(1).unwrap();
    assert!(
        count.find("unsupported_multimodal_token_count").unwrap()
            < count.find("flatten_content(").unwrap()
    );
    let tokenize = include_str!("../misc_handlers.rs");
    assert!(
        tokenize.find("unsupported_multimodal_token_count").unwrap()
            < tokenize
                .find("let tokens = if let Some(ref prompt)")
                .unwrap()
    );
}
