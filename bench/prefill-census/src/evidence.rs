// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
pub(super) fn parse_json_response(bytes: &[u8], status: u16) -> Result<Value> {
    check(
        status == 200,
        &format!("HTTP {status}; response saved for inspection"),
    )?;
    let v: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("incomplete/invalid response JSON: {e}"))?;
    check(
        v.is_object() && v.get("error").is_none(),
        "API error or non-object response; see saved response",
    )?;
    Ok(v)
}
pub(super) fn parse_tokens(v: &Value) -> Result<Vec<u32>> {
    let array = v["tokens"]
        .as_array()
        .ok_or("tokenize response has no tokens array")?;
    let ids = array
        .iter()
        .map(|id| {
            id.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("tokenize returned invalid token ID".into())
        })
        .collect::<Result<Vec<_>>>()?;
    check(
        !ids.is_empty() && number(v, "count")? == ids.len() as u64,
        "tokenize count mismatch/empty tokens",
    )?;
    Ok(ids)
}
pub(super) fn validate_response(
    v: &Value,
    expected: Option<usize>,
    chat: bool,
    wall: f64,
) -> Result<Value> {
    check(v.get("error").is_none(), "API error in measurement")?;
    let choices = v["choices"].as_array().ok_or("missing choices")?;
    check(choices.len() == 1, "expected exactly one completion choice")?;
    let choice = &choices[0];
    check(number(choice, "index")? == 0, "unexpected choice index")?;
    let text = if chat {
        choice["message"]["content"].as_str()
    } else {
        choice["text"].as_str()
    }
    .ok_or("missing output text/content")?;
    check(
        !text.trim().is_empty(),
        "empty output; not a valid timing sample",
    )?;
    let finish = str_field(choice, "finish_reason")?;
    check(
        ["stop", "length"].contains(&finish),
        "unfinished/unsupported finish_reason",
    )?;
    let usage = &v["usage"];
    let prompt = number(usage, "prompt_tokens")?;
    let completion = number(usage, "completion_tokens")?;
    check(
        prompt > 0 && expected.is_none_or(|n| prompt == n as u64),
        "usage.prompt_tokens differs from exact input count",
    )?;
    check(
        (1..=32).contains(&completion),
        "completion token count must be 1..=32",
    )?;
    check(
        prompt.checked_add(completion) == Some(number(usage, "total_tokens")?),
        "inconsistent total_tokens",
    )?;
    check(
        number(&usage["prompt_tokens_details"], "cached_tokens")? == 0,
        "nonzero cached_tokens invalidates uncached prefill",
    )?;
    let ttft = usage["time_to_first_token_ms"]
        .as_f64()
        .ok_or("missing numeric server TTFT")?;
    check(
        ttft.is_finite() && ttft > 0.0 && wall.is_finite() && wall > 0.0,
        "nonpositive/nonfinite timing",
    )?;
    let rate = prompt as f64 / ttft * 1000.0;
    check(rate.is_finite() && rate > 0.0, "invalid derived throughput")?;
    Ok(
        json!({"prompt_tokens":prompt,"completion_tokens":completion,"cached_tokens":0,"finish_reason":finish,
        "server_ttft_ms":ttft,"client_wall_ms":wall,"effective_prefill_tokens_per_second":rate,
        "reported_decode_tokens_per_second":usage.get("response_token/s"),"output":text,"semantic_status":"unreviewed"}),
    )
}
