// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::body::{Body, Bytes};
use axum::extract::rejection::{BytesRejection, JsonRejection};
use axum::extract::{FromRequest, State};
use axum::http::{HeaderMap, Request, StatusCode, header};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::chat_completions_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::extract_assistant_incoming_message;
use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::failures::{
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F39FailureCache,
    F39PermanentFailureMatch, F49DuplicateWrite, append_f7_reminder_to_last_user,
    build_f7_stall_reminder, bump_f12_tool_call_count, check_loop_watchdog,
    collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash, f23_refuse_threshold,
    f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f37_classify_failure, f39_build_circuit_breaker_banner, f39_build_failure_cache,
    f39_class_label, f39_detect_recent_retries, f39_extract_binary_name,
    f44_check_permanent_failure, f49_build_banner, f49_detect_duplicate_writes,
    f49_extract_write_path_and_content, f50_append_original_error, f60_disable_mtp_for_request,
    flush_content_sanitizer, prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::failures::*;
use super::inference_types::*;
use super::sanitizer::*;

#[derive(serde::Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default)]
    pub items: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(serde::Deserialize)]
pub struct UpdateConversationRequest {
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(serde::Deserialize)]
pub struct AddItemsRequest {
    pub items: Vec<serde_json::Value>,
}

/// Build the public JSON shape for a conversation snapshot.
pub(super) fn conversation_body(
    snap: &crate::conversation_store::ConversationSnapshot,
) -> serde_json::Value {
    serde_json::json!({
        "id": snap.id,
        "object": "conversation",
        "created_at": snap.created_at,
        "metadata": snap.metadata,
    })
}

fn empty_create_conversation_request() -> CreateConversationRequest {
    CreateConversationRequest {
        items: None,
        metadata: None,
    }
}

/// Parse an optional create body without treating a nonempty untyped body as absent.
async fn parse_optional_create_conversation_request(
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<CreateConversationRequest, JsonRejection> {
    if headers.get(header::CONTENT_TYPE).is_none() && body.is_empty() {
        return Ok(empty_create_conversation_request());
    }

    // Use Axum itself to enforce its exact JSON media-type rules. The actual
    // bytes were already collected through the limit-aware `Bytes` extractor.
    let mut content_type_probe = Request::new(Body::from("null"));
    *content_type_probe.headers_mut() = headers.clone();
    let _ = Json::<serde_json::Value>::from_request(content_type_probe, &()).await?;

    let Json(req) = Json::<CreateConversationRequest>::from_bytes(body)?;
    Ok(req)
}

/// POST /v1/conversations — create a conversation with optional
/// initial items + metadata.
pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(error) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request body: {error}"),
            );
        }
    };
    let req = match parse_optional_create_conversation_request(&headers, &body).await {
        Ok(req) => req,
        Err(error) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {error}"),
            );
        }
    };
    let items = req.items.unwrap_or_default();
    if items.len() > crate::conversation_store::MAX_ITEMS_PER_INSERT {
        return openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "`items` exceeds per-call cap of {} (got {}).",
                crate::conversation_store::MAX_ITEMS_PER_INSERT,
                items.len(),
            ),
            Some("items"),
            Some("items_too_many"),
        );
    }
    let snap = state
        .conversation_store
        .create_snapshot(items, req.metadata.unwrap_or_default());
    Json(conversation_body(&snap)).into_response()
}

/// GET /v1/conversations/{id}
pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match state.conversation_store.get(&id) {
        Some(snap) => Json(conversation_body(&snap)).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        ),
    }
}

/// POST /v1/conversations/{id} — update metadata.
pub async fn update_conversation(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Result<Json<UpdateConversationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match state.conversation_store.update_metadata(&id, req.metadata) {
        Some(snap) => Json(conversation_body(&snap)).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        ),
    }
}

/// DELETE /v1/conversations/{id}
pub async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if state.conversation_store.delete(&id) {
        Json(serde_json::json!({
            "id": id,
            "object": "conversation.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        )
    }
}

/// POST /v1/conversations/{id}/items — append items (≤20/call).
pub async fn add_conversation_items(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Result<Json<AddItemsRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match state.conversation_store.add_items(&id, req.items) {
        Ok(items) => Json(serde_json::json!({
            "object": "list",
            "data": items,
        }))
        .into_response(),
        Err(crate::conversation_store::AddItemsError::NotFound) => {
            openai_error_response_with_param(
                StatusCode::NOT_FOUND,
                format!("Conversation '{id}' not found."),
                Some("id"),
                Some("conversation_not_found"),
            )
        }
        Err(crate::conversation_store::AddItemsError::TooMany(n)) => {
            openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!(
                    "`items` exceeds per-call cap of {} (got {n}).",
                    crate::conversation_store::MAX_ITEMS_PER_INSERT,
                ),
                Some("items"),
                Some("items_too_many"),
            )
        }
    }
}

/// GET /v1/conversations/{id}/items — list items with `limit` + `order`
/// query parameters (OpenAI spec: default 20, max 100, order=asc).
pub async fn list_conversation_items(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(snap) = state.conversation_store.get(&id) else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        );
    };
    let mut items = snap.items;
    let order = q.get("order").map(|s| s.as_str()).unwrap_or("asc");
    if order == "desc" {
        items.reverse();
    }
    let limit: usize = q
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
        .min(100);
    if items.len() > limit {
        items.truncate(limit);
    }
    let first_id = items
        .first()
        .and_then(|v| v.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    let last_id = items
        .last()
        .and_then(|v| v.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    Json(serde_json::json!({
        "object": "list",
        "data": items,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": false,
    }))
    .into_response()
}

/// GET /v1/conversations/{id}/items/{item_id}
pub async fn get_conversation_item(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
) -> Response {
    let Some(snap) = state.conversation_store.get(&id) else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        );
    };
    for it in &snap.items {
        if it.get("id").and_then(|v| v.as_str()) == Some(item_id.as_str()) {
            return Json(it.clone()).into_response();
        }
    }
    openai_error_response_with_param(
        StatusCode::NOT_FOUND,
        format!("Item '{item_id}' not found in conversation '{id}'."),
        Some("item_id"),
        Some("item_not_found"),
    )
}

/// DELETE /v1/conversations/{id}/items/{item_id}
pub async fn delete_conversation_item(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
) -> Response {
    if state.conversation_store.remove_item(&id, &item_id) {
        Json(serde_json::json!({
            "id": item_id,
            "object": "conversation.item.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Item '{item_id}' not found in conversation '{id}'."),
            Some("item_id"),
            Some("item_not_found"),
        )
    }
}

#[cfg(test)]
mod create_request_tests {
    use super::parse_optional_create_conversation_request;
    use axum::body::Bytes;
    use axum::http::{HeaderMap, HeaderValue, header};

    fn json_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        headers
    }

    #[tokio::test]
    async fn optional_json_distinguishes_absent_valid_and_malformed_bodies() {
        let absent_headers = HeaderMap::new();
        let absent = parse_optional_create_conversation_request(&absent_headers, &Bytes::new())
            .await
            .unwrap();
        assert!(absent.items.is_none());
        assert!(absent.metadata.is_none());

        let valid_body = Bytes::from_static(br#"{"metadata":{"team":"atlas"}}"#);
        let valid = parse_optional_create_conversation_request(&json_headers(), &valid_body)
            .await
            .unwrap();
        assert_eq!(
            valid.metadata.unwrap().get("team").map(String::as_str),
            Some("atlas")
        );

        let malformed = Bytes::from_static(b"{not-json");
        assert!(
            parse_optional_create_conversation_request(&json_headers(), &malformed)
                .await
                .is_err()
        );

        // Optional means genuinely absent, not merely missing Content-Type.
        assert!(
            parse_optional_create_conversation_request(&absent_headers, &valid_body)
                .await
                .is_err()
        );
        assert!(
            parse_optional_create_conversation_request(&absent_headers, &malformed)
                .await
                .is_err()
        );
        assert!(
            parse_optional_create_conversation_request(&json_headers(), &Bytes::new())
                .await
                .is_err()
        );

        let handler = include_str!("conversations.rs");
        assert!(handler.contains("body: Result<Bytes, BytesRejection>"));
        assert!(handler.contains("headers.get(header::CONTENT_TYPE).is_none() && body.is_empty()"));
        assert!(handler.contains("Invalid request JSON: {error}"));
    }
}
