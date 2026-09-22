// SPDX-License-Identifier: AGPL-3.0-only

//! Checked admission for the shared input-plus-output context budget.

pub(super) fn admitted_total(
    prompt_tokens: usize,
    requested_output_tokens: usize,
    max_seq_len: usize,
) -> Option<usize> {
    prompt_tokens
        .checked_add(requested_output_tokens)
        .filter(|&total| total <= max_seq_len)
}

#[cfg(test)]
mod tests {
    use super::admitted_total;

    #[test]
    fn million_token_boundary_is_inclusive() {
        assert_eq!(admitted_total(999_999, 1, 1_000_000), Some(1_000_000));
        assert_eq!(admitted_total(999_999, 2, 1_000_000), None);
    }

    #[test]
    fn addition_overflow_is_rejected() {
        assert_eq!(admitted_total(usize::MAX, 1, usize::MAX), None);
    }

    #[test]
    fn yarn_images_are_rejected_before_message_preprocessing() {
        // The YaRN image rejection lives in `image_admission::validate`, which
        // takes `state.yarn_context` and runs before message preprocessing.
        let chat = include_str!("chat/mod.rs");
        let admission = include_str!("chat/image_admission.rs");
        let reject = chat.find("image_admission::validate(").unwrap();
        let yarn_arg = chat[reject..].find("state.yarn_context").unwrap() + reject;
        let preprocess = chat.find("msg_entry::build_msg_entries").unwrap();
        assert!(reject < yarn_arg && yarn_arg < preprocess);
        assert!(admission.contains("if yarn_context"));
        assert!(admission.contains("unsupported_multimodal_context"));
    }
}
