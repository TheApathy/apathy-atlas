// SPDX-License-Identifier: AGPL-3.0-only

#[path = "image_admission.rs"]
mod admission;

#[test]
fn images_require_supported_ownership_while_text_is_unchanged() {
    assert!(admission::validate(0, false, 8, true).is_ok());
    assert!(admission::validate(8, true, 1, false).is_ok());
    for (count, vision, concurrency, yarn) in [(1,false,1,false),(1,true,2,false),(1,true,0,false),(1,true,1,true),(9,true,1,false)] {
        assert!(admission::validate(count,vision,concurrency,yarn).is_err());
    }
}

#[test]
fn request_guard_precedes_actual_prompt_preparation() {
    let source = include_str!("mod.rs");
    assert!(source.find("image_admission::validate(").unwrap()
        < source.find("prepare_chat_prompt(").unwrap());
    let token = include_str!("../misc_handlers.rs");
    assert!(token.find("unsupported_multimodal_token_count").unwrap()
        < token.find("let tokens = if let Some(ref prompt)").unwrap());
}

#[test]
fn image_reset_preserves_codispatch_prepass_and_stream_fence() {
    let chunk = include_str!("../../scheduler/prefill_a_step.rs");
    let guard = chunk.find("if vision_slice.is_none() {").unwrap();
    let encode = chunk.find("model.prepare_vision_embed(&image_pixels)?;").unwrap();
    let fence = chunk[encode..].find("model.record_event(prefill_event, model.default_stream())?").unwrap();
    assert!(guard < encode);
    assert!(fence > 0);
    assert!(!chunk.contains("if vision_slice.is_none() && !image_pixels.is_empty()"));
    let full = include_str!("../../scheduler/prefill_b_step.rs");
    let encode = full.find("model.prepare_vision_embed(&image_pixels)?;").unwrap();
    assert!(!full[encode.saturating_sub(70)..encode].contains("if !image_pixels.is_empty()"));
}
