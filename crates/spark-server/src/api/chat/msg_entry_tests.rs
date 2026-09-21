// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(test)]
mod error_dedup_tests {
    use super::super::duplicate_error_masks;

    #[test]
    fn exact_duplicate_errors_mask_older_keep_newest() {
        // The 45k-collapse shape: identical error text repeated.
        let err = "Error: BadResource: FileSystem.readFile (/home/nologik)";
        let results = vec![
            (2, err.to_string()),
            (5, format!("  {err}\n")), // trim-equal counts as duplicate
            (9, err.to_string()),
        ];
        let masks = duplicate_error_masks(&results);
        assert_eq!(
            masks,
            vec![
                (2, "[same error as below, attempt 1 of 3]".to_string()),
                (5, "[same error as below, attempt 2 of 3]".to_string()),
            ],
            "older duplicates masked, newest (idx 9) untouched"
        );
    }

    #[test]
    fn non_error_duplicates_untouched() {
        let ok = "src\nCargo.toml\nREADME.md";
        let results = vec![
            (1, ok.to_string()),
            (3, ok.to_string()),
            (7, ok.to_string()),
        ];
        assert!(
            duplicate_error_masks(&results).is_empty(),
            "identical SUCCESS observations are legitimate — never masked"
        );
    }

    #[test]
    fn distinct_errors_untouched() {
        let results = vec![
            (1, "Error: ENOENT no such file /a/b".to_string()),
            (
                4,
                "Error: Permission denied writing /etc/passwd".to_string(),
            ),
        ];
        assert!(duplicate_error_masks(&results).is_empty());
    }

    #[test]
    fn near_duplicate_errors_group_via_jaccard() {
        // Long errors differing in one trailing token: Jaccard over
        // 4-gram shingles lands just above 0.9 at ~90 tokens.
        let body: Vec<String> = (0..90).map(|i| format!("frame{i}")).collect();
        let base = format!("Error: {}", body.join(" "));
        let mut variant_body = body.clone();
        *variant_body.last_mut().unwrap() = "different".to_string();
        let variant = format!("Error: {}", variant_body.join(" "));
        let results = vec![(0, base), (2, variant)];
        let masks = duplicate_error_masks(&results);
        assert_eq!(
            masks,
            vec![(0, "[same error as below, attempt 1 of 2]".to_string())]
        );
    }

    #[test]
    fn single_error_never_masked() {
        let results = vec![(0, "Error: something failed".to_string())];
        assert!(duplicate_error_masks(&results).is_empty());
    }
}

#[cfg(test)]
mod vacuous_system_tests {
    use super::super::is_vacuous_system_content;

    #[test]
    fn empty_or_whitespace_is_vacuous() {
        assert!(is_vacuous_system_content(""));
        assert!(is_vacuous_system_content("   \n\t  "));
    }

    #[test]
    fn open_webui_empty_context_residue_is_vacuous() {
        // The exact 2026-05-17 field artifact.
        assert!(is_vacuous_system_content("User Context:\n\n"));
        assert!(is_vacuous_system_content("Context:"));
        assert!(is_vacuous_system_content("  System:  "));
    }

    #[test]
    fn substantive_prompt_is_not_vacuous() {
        assert!(!is_vacuous_system_content(
            "User Context:\nThe user is a senior Rust engineer."
        ));
        assert!(!is_vacuous_system_content("You are a helpful assistant."));
        assert!(!is_vacuous_system_content("Always answer in French."));
        // Label-like but with a real payload after the colon.
        assert!(!is_vacuous_system_content("Role: expert chess coach"));
        // Long single line ending in ':' is unusual prose, not a bare
        // header — keep it (avoid false-strip).
        assert!(!is_vacuous_system_content(
            "Summarize the following transcript and then ask the user this:"
        ));
    }
}

#[cfg(test)]
mod build_tests {
    use super::super::build_msg_entries;
    use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role};
    use axum::http::StatusCode;

    fn assert_bad_request(msgs: &[Message], tools_active: bool) {
        match build_msg_entries(None, None, None, msgs, tools_active, false) {
            Ok(_) => panic!("expected 400, got Ok"),
            Err(resp) => assert_eq!(resp.status(), StatusCode::BAD_REQUEST),
        }
    }

    fn text(role: Role, t: &str) -> Message {
        Message {
            role,
            content: vec![ContentPart::Text(t.into())],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    fn image(role: Role) -> Message {
        Message {
            role,
            content: vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
                ContentPart::Text("result".into()),
            ],
            tool_calls: Vec::new(),
            tool_call_id: Some("c1".into()),
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    #[test]
    fn text_only_builds_without_vision_config() {
        let msgs = vec![text(Role::User, "hello")];
        let out = build_msg_entries(None, None, None, &msgs, false, false).expect("text-only ok");
        assert_eq!(out.messages.len(), 1);
        assert_eq!(out.messages[0].image_count, 0);
    }

    #[test]
    fn deepseek_vision_preserves_interleaved_image_order_and_preprocesses_pixels() {
        use base64::Engine;
        let cfg = atlas_core::config::DeepSeekVisionConfig {
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            patch_size: 14,
            downsample_ratio: 3,
            max_tokens: 384,
            max_wh_ratio: Some(8.0),
            min_pixels: 1,
            rope_theta: 10000.0,
        };
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(14, 14, image::Rgb([255, 0, 127]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png.into_inner())
        );
        let mut msg = text(Role::User, "before ");
        msg.content.push(ContentPart::Image(ImageSource {
            data: ImageData::Base64(uri),
        }));
        msg.content.push(ContentPart::Text(" after".into()));
        let out = build_msg_entries(None, Some(&cfg), None, &[msg], false, false).unwrap();
        assert_eq!(out.messages[0].content, "before <｜deepseek_image｜> after");
        assert_eq!(out.messages[0].image_count, 0);
        assert_eq!(out.image_pad_counts, vec![1]);
        assert_eq!(out.image_pixels.len(), 1);
        assert_eq!((out.image_pixels[0].1, out.image_pixels[0].2), (1, 1));
        assert_eq!(out.image_pixels[0].0.len(), 3 * 14 * 14);
    }

    #[test]
    fn user_image_without_vision_config_is_rejected() {
        // Previously: silently dropped (200). Now: fail fast.
        assert_bad_request(&[text(Role::User, "hi"), image(Role::User)], false);
    }

    #[test]
    fn tool_result_image_is_collected_and_rejected_without_vision() {
        // Proves the tool branch now COUNTS + COLLECTS images (it used to
        // hardcode image_count: 0 and `continue` before collection): the
        // fail-fast only fires when an image was actually collected.
        assert_bad_request(&[text(Role::User, "look"), image(Role::Tool)], true);
    }

    #[test]
    fn developer_role_normalized_to_system_at_build() {
        // Was previously mapped only at JSON render time, so `developer`
        // messages bypassed the cwd-hint / vacuous-system / CWD-injection
        // scans that string-compare on "system".
        let msgs = vec![text(Role::Other("developer".into()), "be terse")];
        let out = build_msg_entries(None, None, None, &msgs, false, false).expect("ok");
        assert_eq!(out.messages[0].role, "system");

        // Other unknown roles still pass through verbatim for the
        // template to handle.
        let msgs = vec![text(Role::Other("critic".into()), "hm")];
        let out = build_msg_entries(None, None, None, &msgs, false, false).expect("ok");
        assert_eq!(out.messages[0].role, "critic");
    }

    #[test]
    fn remote_url_image_rejected_with_clear_reason() {
        // A https URL used to be mislabeled as base64 and die later in
        // the vision preprocessor with "base64 decode failed". It must
        // now be rejected up front with the real reason — even when the
        // model HAS no vision config (the URL check fires first, at
        // collection time).
        let url_msg = Message {
            role: Role::User,
            content: vec![ContentPart::Image(ImageSource {
                data: ImageData::Url("https://example.com/cat.png".into()),
            })],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        };
        match build_msg_entries(None, None, None, &[url_msg], false, false) {
            Ok(_) => panic!("expected 400, got Ok"),
            Err(resp) => assert_eq!(resp.status(), StatusCode::BAD_REQUEST),
        }
    }

    #[test]
    fn model_can_disable_duplicate_cwd_hint_injection() {
        let msgs = vec![text(
            Role::System,
            "client prompt\nworking directory: /tmp/project",
        )];

        let enabled = build_msg_entries(None, None, None, &msgs, true, false).expect("enabled");
        assert_eq!(enabled.cwd_hint.as_deref(), Some("/tmp/project"));
        assert!(enabled.messages[0].content.contains("<environment>"));

        let disabled = build_msg_entries(None, None, None, &msgs, true, true).expect("disabled");
        assert_eq!(disabled.cwd_hint.as_deref(), Some("/tmp/project"));
        assert_eq!(
            disabled.messages[0].content,
            "client prompt\nworking directory: /tmp/project"
        );
    }
}
