// SPDX-License-Identifier: AGPL-3.0-only

/// Current published vision embeddings are model-owned, not per-sequence.
/// Admit images only inside the supported ownership/context envelope.
pub(super) fn validate_choices(image_count: usize, choices: usize) -> Result<(), &'static str> {
    if image_count > 0 && choices != 1 {
        return Err("Image requests require n=1 (unsupported_multimodal_choices)");
    }
    Ok(())
}

pub(super) fn validate(
    image_count: usize,
    has_vision: bool,
    max_batch_size: usize,
    yarn_context: bool,
) -> Result<(), &'static str> {
    if image_count == 0 {
        return Ok(());
    }
    if !has_vision {
        return Err("This model does not have a vision encoder (unsupported_multimodal_model)");
    }
    if max_batch_size != 1 {
        return Err(
            "Image requests require max-concurrent-sequences=1 (unsupported_multimodal_concurrency)",
        );
    }
    if yarn_context {
        return Err(
            "Static-YaRN long context is currently text-only (unsupported_multimodal_context)",
        );
    }
    if image_count > 8 {
        return Err("Image requests are limited to eight images (multimodal_image_limit)");
    }
    Ok(())
}
