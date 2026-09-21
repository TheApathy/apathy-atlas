// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
pub(super) fn canary_request(model: &str) -> Value {
    json!({"model":model,"messages":[{"role":"user","content":"Reply with exactly ATLAS_CANARY_OK and nothing else."}],
        "max_tokens":32,"temperature":0.0,"stream":false,"enable_thinking":false,
        "chat_template_kwargs":{"enable_thinking":false}})
}
pub(super) fn exact_prompt(corpus: &[u32], suffix: &[u32], length: usize) -> Result<Vec<u32>> {
    check(
        !suffix.is_empty() && suffix.len() < length && corpus.len() >= length - suffix.len(),
        "token corpus cannot form requested exact bin",
    )?;
    let mut ids = corpus[..length - suffix.len()].to_vec();
    ids.extend(suffix);
    Ok(ids)
}
pub(super) fn parse_prompt_format(raw: Option<&str>) -> Result<String> {
    let format = raw.unwrap_or("plain");
    prompt_parts(format)?;
    Ok(format.to_string())
}
pub(super) fn prompt_parts(format: &str) -> Result<(&'static str, &'static str)> {
    match format {
        "plain" => Ok((
            "<think></think>\n\nReference facts:\n",
            "\n\nWrite a short factual summary of the reference facts above.\nSummary:",
        )),
        "qwen-chatml" => Ok((
            "<|im_start|>user\nReference facts:\n",
            "\n\nWrite a short factual summary of the reference facts above.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )),
        "deepseek-chat" => Ok((
            "<｜begin▁of▁sentence｜><｜User｜>Reference facts:\n",
            "\n\nWrite a short factual summary of the reference facts above.<｜Assistant｜></think>",
        )),
        _ => Err("--prompt-format must be plain, qwen-chatml or deepseek-chat".into()),
    }
}
pub(super) fn corpus(prefix: &str) -> String {
    let mut text = String::from(prefix);
    for n in 0..4000 {
        text.push_str(&format!(
            "Fact {n:04}: shelf {} contains {} blue folders and {} red folders.\n",
            n % 97,
            n % 23 + 1,
            n % 11 + 1
        ));
    }
    text
}
