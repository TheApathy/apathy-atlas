// SPDX-License-Identifier: AGPL-3.0-only

fn flat(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn template_binds_offsets_and_never_compacts_image_ownership() {
    let src = flat(include_str!("../api/chat/template.rs"));
    assert!(src.contains(
        "ParsedContent::marker_json(&effective_content,m.image_count,&m.image_text_offsets,"
    ));
    assert!(src.contains("if!has_images&&auto_compact_active"));
    assert!(src.contains("if!has_images&&ctx_overflow_truncate_enabled()"));
    assert!(src.contains("image_count!=image_pad_counts.len()||image_pad_counts.contains(&0)"));
    assert!(src.contains("prompt_tokens.iter().filter(|&&id|id==pad).count()!=image_count"));
}

#[test]
fn tool_images_reach_common_collection_and_empty_system_images_survive() {
    let src = include_str!("../api/chat/msg_entry.rs");
    let tool = src.find("if tools_active && m.role == \"tool\"").unwrap();
    let collect = src[tool..]
        .find("let image_count = m.content.images.len()")
        .unwrap()
        + tool;
    assert!(!src[tool..collect].contains("continue;"));
    let src = flat(src);
    assert!(src.contains("image_text_offsets:content.image_text_offsets"));
    assert!(src.contains("m.role==\"system\"&&m.image_count==0&&is_vacuous_system_content"));
    assert!(src.contains("content.prepend_text("));
    assert!(src.contains("content.validate_order()"));
}

#[test]
fn destructive_text_rewriters_preserve_image_positions() {
    let src = flat(include_str!("../api/chat/mod.rs"));
    assert!(src.contains("first.content.prepend_text("));
    assert!(src.contains("ifletSome(new_body)=replacement&&messages[i].image_count==0"));
    let stall = flat(include_str!("../api/failures/stall.rs"));
    assert_eq!(
        stall
            .matches("m.role==\"system\"&&!m.content.images.is_empty()")
            .count(),
        2
    );
}

#[test]
fn replay_and_storage_use_checked_ordered_helpers() {
    let disk = include_str!("../response_store.rs");
    assert!(disk.contains("m.content.chat_json()?"));
    let stored = include_str!("../api/stored.rs");
    assert!(stored.contains("m.content.responses_json()?"));
    let lowering = flat(include_str!("responses_lowering.rs"));
    assert!(lowering.contains("IncomingMessage::try_from_responses_input_item(it).map_err(LowerResponsesError::BadRequest)?"));
    let replay = include_str!("../api/responses.rs");
    assert!(replay.contains("IncomingMessage::try_from_responses_input_item(item)"));
    assert!(!replay.contains(".filter_map(conversation_item_to_message)"));
}
