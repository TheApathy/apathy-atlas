// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::fuzzy_match::fuzzy_match_tool_name;
use super::*;

/// Fix tool call arguments: schema-aware type coercion + backfill missing params.
///
/// The qwen3_coder XML format emits all parameter values as raw text. This function:
/// 1. **Type coercion**: Converts string values to the schema-expected type
///    (number, boolean, integer, object, array). Prevents "expected number,
///    received string" errors from clients like OpenCode.
/// 2. **Backfill**: fills a missing/empty required parameter only when a real
///    value can be derived from model-authored arguments or the tool definition.
///    It never invents a placeholder.
///
/// Matches vLLM's qwen3coder_tool_parser behavior (schema-aware type conversion).
///
/// Resolves the effective type from a JSON schema property, handling `anyOf`/`oneOf`
/// wrappers (e.g., Pydantic v2's `Optional[int]` → `{"anyOf": [{"type":"integer"},{"type":"null"}]}`).
fn resolve_schema_type(schema: &serde_json::Value) -> Option<&str> {
    // Direct "type" field
    if let Some(t) = schema.get("type").and_then(|t| t.as_str()) {
        return Some(t);
    }
    // anyOf / oneOf: pick first non-null type
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(|v| v.as_array()) {
            for variant in variants {
                if let Some(t) = variant.get("type").and_then(|t| t.as_str())
                    && t != "null"
                {
                    return Some(t);
                }
            }
        }
    }
    None
}

fn parse_agent_types(description: &str) -> Vec<String> {
    description
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("- ")?;
            let (name, _) = rest.split_once(':')?;
            let name = name.trim();
            (!name.is_empty()
                && name.len() <= 64
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
            .then(|| name.to_string())
        })
        .collect()
}

fn infer_default_subagent_type(description: Option<&str>) -> String {
    let candidates = description.map(parse_agent_types).unwrap_or_default();
    candidates
        .iter()
        .find(|candidate| candidate.to_ascii_lowercase().contains("general"))
        .or_else(|| candidates.first())
        .cloned()
        .unwrap_or_else(|| "general".to_string())
}

/// Derive a real value for a required parameter, or return `None` when the
/// server cannot know the model's intent. Paths, search strings, cities, and
/// other payload-bearing values must remain absent rather than becoming `""`.
fn derive_required_param(
    key: &str,
    func_name: &str,
    args: &serde_json::Map<String, serde_json::Value>,
    tool_def: &ToolDefinition,
) -> Option<String> {
    match key {
        "description" => Some(match args.get("command").and_then(|value| value.as_str()) {
            Some(command) if command.len() > 50 => {
                let head: String = command.chars().take(47).collect();
                format!("Run: {head}...")
            }
            Some(command) => format!("Run: {command}"),
            None => format!("{func_name} operation"),
        }),
        "subagent_type" | "subagentType" => Some(infer_default_subagent_type(
            tool_def.function.description.as_deref(),
        )),
        _ => None,
    }
}

pub fn backfill_required_params(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
            continue;
        };
        let Some(ref params_schema) = tool_def.function.parameters else {
            continue;
        };
        let required = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = params_schema.get("properties").and_then(|p| p.as_object());
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;

        // 1. Coerce existing parameters to schema-expected types.
        if let Some(props) = properties {
            for (key, value) in args.iter_mut() {
                let expected_type = props.get(key).and_then(|p| resolve_schema_type(p));
                if let (Some(expected), serde_json::Value::String(s)) = (expected_type, &value) {
                    let coerced = match expected {
                        "number" => s.parse::<f64>().ok().map(|n| {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(n)
                                    .unwrap_or(serde_json::Number::from(0)),
                            )
                        }),
                        "integer" => s
                            .parse::<i64>()
                            .ok()
                            .map(|n| serde_json::Value::Number(n.into())),
                        "boolean" => match s.to_lowercase().as_str() {
                            "true" | "1" | "yes" => Some(serde_json::Value::Bool(true)),
                            "false" | "0" | "no" => Some(serde_json::Value::Bool(false)),
                            _ => None,
                        },
                        "object" | "array" => serde_json::from_str(s).ok(),
                        _ => None, // "string" or unknown — keep as-is
                    };
                    if let Some(new_val) = coerced {
                        *value = new_val;
                        changed = true;
                    }
                }
            }
        }

        // 2. Normalize parameter names to match the schema.
        // The model sometimes emits camelCase (filePath) when the schema
        // defines snake_case (file_path), or vice versa. This is a known
        // Qwen3-Coder issue (vLLM #35347, llama.cpp #19382).
        if let Some(props) = properties {
            // Build case-insensitive lookup: "filepath" → "file_path" (schema name)
            let schema_normalized: std::collections::HashMap<String, &str> = props
                .keys()
                .map(|k| (k.to_lowercase().replace('_', ""), k.as_str()))
                .collect();

            let keys_to_fix: Vec<(String, String)> = args
                .keys()
                .filter(|k| !props.contains_key(*k))
                .filter_map(|k| {
                    let norm = k.to_lowercase().replace('_', "");
                    schema_normalized
                        .get(&norm)
                        .map(|schema_key| (k.clone(), schema_key.to_string()))
                })
                .collect();

            for (wrong_key, right_key) in keys_to_fix {
                if let Some(val) = args.remove(&wrong_key) {
                    args.entry(right_key).or_insert(val);
                    changed = true;
                }
            }
        }

        // 3. Derive only values the server can determine honestly. An absent
        // key makes the client's JSON-Schema `required` check fail loudly; a
        // fabricated empty string is valid `type:string` and can execute the
        // wrong call without any downstream layer knowing Atlas authored it.
        let func_name = call.function.name.clone();
        for key in &required {
            let model_supplied = match args.get(*key) {
                Some(serde_json::Value::String(value)) => !value.trim().is_empty(),
                Some(_) => true,
                None => false,
            };
            if model_supplied {
                continue;
            }
            if let Some(derived) = derive_required_param(key, &func_name, &args, tool_def) {
                args.insert(key.to_string(), serde_json::Value::String(derived));
                changed = true;
            }
        }

        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

/// Check if a tool call has empty required parameters that can't be auto-filled.
/// Returns the names of empty required params, or empty vec if all are filled.
pub fn find_empty_required_params(call: &ToolCall, tools: &[ToolDefinition]) -> Vec<String> {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
        return Vec::new();
    };
    let Some(ref params_schema) = tool_def.function.parameters else {
        return Vec::new();
    };
    let required = params_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&call.function.arguments).unwrap_or_default();
    let mut empty = Vec::new();
    for key in &required {
        match args.get(key.as_str()) {
            None => empty.push(key.clone()),
            Some(serde_json::Value::String(s)) if s.trim().is_empty() => empty.push(key.clone()),
            Some(serde_json::Value::Null) => empty.push(key.clone()),
            _ => {}
        }
    }
    empty
}

/// Normalize file paths in tool call arguments to be relative to the working directory.
///
/// OPENCODE BUG FIX (2026-04-22): the previous behaviour stripped the leading `/`
/// of any absolute path NOT under cwd, mangling user-intended paths like
/// `/tmp/calc-test16/calc.py` into `tmp/calc-test16/calc.py`. opencode then
/// resolved that relative path under `Instance.directory` (= `$HOME`), so the
/// file ended up at `$HOME/tmp/calc-test16/calc.py` instead of
/// `/tmp/calc-test16/`. The model spent 8+ turns trying to "fix" the directory
/// before the user noticed.
///
/// New behaviour:
/// - Paths under cwd → made relative (still helpful for Claude-Code-style clients)
/// - Paths starting with `/` but NOT under cwd → **PASS THROUGH UNCHANGED**.
///   The model knew what it wanted (e.g. user said "put it in /tmp/..."); we
///   should not second-guess. If it really is wrong, the filesystem op will
///   fail with a clear error and the model can self-correct.
/// - Already relative paths → unchanged
pub fn normalize_paths(calls: &mut [ToolCall], cwd: &str) {
    // Common parameter names that contain file paths
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path", "file"];
    let cwd_slash = if cwd.ends_with('/') {
        cwd.to_string()
    } else {
        format!("{cwd}/")
    };

    for call in calls.iter_mut() {
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if !path.starts_with('/') {
                    continue; // Already relative — leave it
                }
                if !path.starts_with(&cwd_slash) {
                    // Absolute path NOT under cwd — pass through verbatim. The
                    // user explicitly asked for this location (e.g. "/tmp/..."),
                    // and trimming `/` here breaks downstream clients that
                    // resolve relative paths against THEIR own working dir.
                    continue;
                }
                let new_path = path[cwd_slash.len()..].to_string();
                if new_path != *path && !new_path.is_empty() {
                    args.insert(key.to_string(), serde_json::Value::String(new_path));
                    changed = true;
                }
            }
        }
        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

// ── Tool call validation ──

const FILE_TOOLS: &[&str] = &["Write", "write", "Edit", "edit", "Read", "read"];
const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
const WRITE_FAMILY: &[&str] = &[
    "Write",
    "write",
    "Edit",
    "edit",
    "MultiEdit",
    "multiEdit",
    "multi_edit",
    "write_file",
    "writeFile",
];
const SHELL_FAMILY: &[&str] = &[
    "bash", "Bash", "shell", "Shell", "exec", "Exec", "run", "Run", "execute", "Execute",
    "terminal", "Terminal",
];
const CMD_KEYS: &[&str] = &["command", "cmd", "script", "code"];

/// Result of validating a batch of tool calls against their schemas.
pub struct ValidatedToolCalls {
    /// Tool calls that passed all validations.
    pub valid: Vec<ToolCall>,
    /// Human-readable error messages for invalid calls.
    /// These should be injected into the response content so the model
    /// sees clear, actionable feedback instead of cryptic client errors.
    pub errors: Vec<String>,
}

/// Validation severity determines whether a parsed call can be attached to
/// the response. Missing ordinary/read-only arguments remain attached so the
/// client can return actionable schema feedback; unexecutable mutation/shell
/// calls and structurally invalid calls are withheld.
#[derive(Debug)]
pub enum ToolCallIssue {
    MissingParam(String),
    EmptyRequired(String),
    Hard(String),
}

impl ToolCallIssue {
    pub fn message(&self) -> &str {
        match self {
            Self::MissingParam(message) | Self::EmptyRequired(message) | Self::Hard(message) => {
                message
            }
        }
    }

    pub fn into_message(self) -> String {
        match self {
            Self::MissingParam(message) | Self::EmptyRequired(message) | Self::Hard(message) => {
                message
            }
        }
    }
}

/// Validate tool calls against their schemas. Returns valid calls and
/// error messages for invalid ones.
///
/// Checks:
/// 1. Tool name exists in definitions
/// 2. Arguments are valid JSON
/// 3. Required params are present, with family-specific attachment policy
/// 4. file_path params don't look like directories (end with `/`)
pub fn validate_tool_calls(
    mut calls: Vec<ToolCall>,
    tools: &[ToolDefinition],
) -> ValidatedToolCalls {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    for call in &mut calls {
        // Fuzzy name repair: if model hallucinates a close-but-wrong name,
        // map to the closest available tool (NVFP4 models often drop prefixes
        // like "get_" or use abbreviations like "weather" for "get_weather").
        if tools.iter().all(|t| t.function.name != call.function.name)
            && let Some(best) = fuzzy_match_tool_name(&call.function.name, tools)
        {
            tracing::info!(
                "Fuzzy tool name repair: '{}' -> '{}'",
                call.function.name,
                best
            );
            call.function.name = best;
        }
        match assess_tool_call(call, tools) {
            Ok(()) => valid.push(call.clone()),
            Err(ToolCallIssue::MissingParam(message)) => {
                valid.push(call.clone());
                errors.push(message);
            }
            Err(issue) => errors.push(issue.into_message()),
        }
    }

    ValidatedToolCalls { valid, errors }
}

/// Compatibility wrapper for callers that only need pass/fail.
pub fn validate_single_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    assess_tool_call(call, tools).map_err(ToolCallIssue::into_message)
}

pub fn assess_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), ToolCallIssue> {
    let name = &call.function.name;

    // 1. Check tool name exists
    let tool_def = tools.iter().find(|t| t.function.name == *name);
    if tool_def.is_none() {
        let available: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(ToolCallIssue::Hard(format!(
            "Error: Unknown tool '{}'. Available tools: {}",
            name,
            available.join(", ")
        )));
    }
    let tool_def = tool_def.unwrap();

    // 2. Check arguments are valid JSON
    let args: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str(&call.function.arguments) {
            Ok(a) => a,
            Err(_) => {
                let preview: String = call.function.arguments.chars().take(100).collect();
                return Err(ToolCallIssue::Hard(format!(
                    "Error: {name} arguments must be valid JSON. Got: {preview}"
                )));
            }
        };

    // 3. Check required params are present. Do NOT enforce non-empty strings —
    // that is the client's schema concern. Empty-string rejection here broke
    // Theia IDE's getWorkspaceFileList, which legitimately passes path="".
    if let Some(ref params_schema) = tool_def.function.parameters {
        let required: Vec<&str> = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|key| args.get(*key).is_none())
            .collect();
        if !missing.is_empty() {
            // Do not let JSON-Schema array ordering weaken the disposition.
            // If a write has neither content nor path and `content` appears
            // first, the missing path must still withhold the call; likewise
            // for a shell schema that lists metadata before its command.
            let unexecutable = missing.iter().find(|key| {
                (WRITE_FAMILY.contains(&name.as_str()) && PATH_KEYS.contains(key))
                    || (SHELL_FAMILY.contains(&name.as_str()) && CMD_KEYS.contains(key))
            });
            let key = unexecutable.copied().unwrap_or(missing[0]);
            let message =
                format!("Error: {name} requires parameter '{key}' but it was not provided.");
            return Err(if unexecutable.is_some() {
                ToolCallIssue::EmptyRequired(message)
            } else {
                ToolCallIssue::MissingParam(message)
            });
        }
    }

    // 4. Path-specific validation for file tools
    // F78 (2026-04-30): file MUTATION tools must have a non-empty
    // path. Live opencode session
    // `ses_2215a95d6ffe6gAzHMBrcXqGXX` looped 11 turns because the
    // model emitted `{"content":"...","filePath":""}` (the model
    // self-truncated the content string and grammar-completed
    // filePath with the empty default). opencode's Write tool
    // returned EISDIR; the model retried with the same empty path.
    // Rejecting here turns the malformed tool_call into a no-op so
    // the response falls through to text — the model gets a single
    // chance to recover instead of opencode echoing EISDIR forever.
    // Read/Glob/LS keep the lenient behavior (Theia's
    // getWorkspaceFileList legitimately passes path="").
    if WRITE_FAMILY.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key)
                && path.trim().is_empty()
            {
                return Err(ToolCallIssue::EmptyRequired(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide an absolute path \
                         like '/tmp/calc-test75/Cargo.toml'."
                )));
            }
        }
    }
    if SHELL_FAMILY.contains(&name.as_str()) {
        for key in CMD_KEYS {
            if let Some(serde_json::Value::String(command)) = args.get(*key)
                && command.trim().len() < 2
            {
                return Err(ToolCallIssue::EmptyRequired(format!(
                    "Error: {name} requires a non-empty '{key}'. Provide the shell command to execute."
                )));
            }
        }
    }
    if FILE_TOOLS.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if path.ends_with('/') {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path must be a FILE, not a directory. Got '{}'. Use e.g. '{}/index.ts'",
                        name,
                        path,
                        path.trim_end_matches('/')
                    )));
                }
                // Check if it looks like just a directory name (no extension, no dots, no uppercase)
                // Allow extensionless files like LICENSE, Makefile, Dockerfile, Cargo.lock etc.
                if !path.is_empty()
                    && !path.contains('.')
                    && !path.contains('/')
                    && path
                        .chars()
                        .all(|c| c.is_lowercase() || c == '-' || c == '_')
                {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path '{}' looks like a directory. Add a filename, e.g. '{}/index.ts'",
                        name, path, path
                    )));
                }
            }
        }
    }

    Ok(())
}
