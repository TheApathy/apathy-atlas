// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn checked_order_precedes_each_wire_to_ir_edge() {
    for source in [include_str!("../api/chat/mod.rs"), include_str!("../api/responses.rs"), include_str!("../api/responses_stream.rs")] {
        let check = source.find(".validate_content_order()").unwrap();
        let conversion = source.find("req.into()").or_else(|| source.find("chat_req.into()")).unwrap();
        assert!(check < conversion);
    }
    assert!(include_str!("to_ir.rs").contains("m.content.ordered_ir_parts()"));
}

#[test]
fn image_ownership_is_not_compacted_or_deduplicated() {
    let template: String = include_str!("../api/chat/template.rs").split_whitespace().collect();
    assert!(template.contains("lethas_images=!image_pad_counts.is_empty()"));
    assert!(template.contains("!has_images&&"));
    assert!(template.contains("ParsedContent::marker_json("));
    let entries: String = include_str!("../api/chat/msg_entry.rs").split_whitespace().collect();
    assert!(entries.contains("ifm.image_count()==0{tool_result_originals.push"));
    assert!(entries.contains("super::ordered_content::flatten"));
    assert!(entries.contains("m.image_count==0&&is_vacuous_system_content"));
}

#[test]
fn replay_and_persistence_preserve_order() {
    assert!(include_str!("../response_store.rs").contains("m.content.chat_json()?"));
    for source in [include_str!("../api/stored.rs"), include_str!("../api/responses_stream.rs"), include_str!("../api/responses_translate.rs")] {
        assert!(source.contains("m.content.responses_json()?"));
    }
    assert!(include_str!("../api/responses.rs").contains("try_from_responses_input_item(item)"));
}

#[test]
fn supported_responses_aliases_keep_image_and_developer_contracts() {
    use super::IncomingMessage;
    let m = IncomingMessage::try_from_responses_input_item(&serde_json::json!({
        "role":"developer", "content":[{"type":"input_text","text":"α"},
            {"type":"input_image","image_url":{"url":"red"}},
            {"type":"image","image_url":"blue"}]
    })).unwrap().unwrap();
    assert_eq!(m.role, "user");
    assert_eq!(m.content.image_text_offsets, [2, 2]);
    assert_eq!(m.content.images, ["red", "blue"]);
}
