// SPDX-License-Identifier: AGPL-3.0-only

//! Port of `server/tool_grammar.py::build_tool_grammar`: the EBNF of exactly
//! the legal DSML calls block for one request's tools. The output text is kept
//! BYTE-IDENTICAL to the Python builder (`tests::grammar_fixtures`), so the
//! same grammar reaches the matcher.
//!
//! Production runs with `DSV41_TOOL_GRAMMAR` unset, which means this grammar is
//! OFF there. It is ported so it can be turned on, not because parity needs it.

use serde_json::Value;

use super::encoding::{CALLS_BLOCK, DSML, INVOKE_TAG, PARAM_TAG};

pub const DEFAULT_MAX_CALLS: usize = 8;

/// "Any character except U+FF5C", written as positive ranges. Negated classes
/// in xgrammar are ASCII-only, so `[^｜]` would silently accept U+FF5C.
pub const VALUE_CHAR: &str = r"[\u0000-\uFF5B\uFF5D-\U0010FFFF]";
const NAME_CHAR: &str = r"[\x20-\x21\x23-\uFF5B\uFF5D-\U0010FFFF]";
const JSON_STRING_CHAR: &str = r"[\x20-\x21\x23-\x5b\x5d-\uFF5B\uFF5D-\U0010FFFF]";

const JSON_RULES: &str = r#"jws ::= [ \n\r\t]{0,4}
jnull ::= "null"
jbool ::= "true" | "false"
jint ::= "-"? ("0" | [1-9] [0-9]*)
jnum ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
jhex ::= [0-9a-fA-F]
jesc ::= [\"\\/bfnrt] | "u" jhex jhex jhex jhex
jchar ::= JSON_STRING_CHAR | "\\" jesc
jstr ::= "\"" jchar* "\""
jval ::= jstr | jnum | jbool | jnull | jarr | jobj
jarr ::= "[" jws (jval jws ("," jws jval jws)*)? "]"
jobj ::= "{" jws (jstr jws ":" jws jval jws ("," jws jstr jws ":" jws jval jws)*)? "}""#;

fn block_open() -> String {
    format!("\n\n<{DSML}{CALLS_BLOCK}>\n")
}
fn block_close() -> String {
    format!("</{DSML}{CALLS_BLOCK}>")
}
fn invoke_open() -> String {
    format!("<{DSML}{INVOKE_TAG} name=")
}
fn invoke_close() -> String {
    format!("</{DSML}{INVOKE_TAG}>\n")
}
fn param_open() -> String {
    format!("<{DSML}{PARAM_TAG} name=")
}
fn param_close() -> String {
    format!("</{DSML}{PARAM_TAG}>\n")
}

/// `_lit`: one EBNF string literal.
pub fn lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `dsml_safe`: can `name` sit in `name="..."` unambiguously?
pub fn dsml_safe(name: &str) -> bool {
    !name.is_empty()
        && !name
            .chars()
            .any(|c| c == '"' || c == '｜' || (c as u32) < 0x20 || c as u32 == 0x7f)
}

fn scalar_rule(t: &str) -> Option<&'static str> {
    match t {
        "integer" => Some("jint"),
        "number" => Some("jnum"),
        "boolean" => Some("jbool"),
        "null" => Some("jnull"),
        "string" => Some("jstr"),
        _ => None,
    }
}

fn type_str(schema: &Value) -> Option<&str> {
    schema.get("type").and_then(Value::as_str).map(str::trim)
}

/// `_param_value`: EBNF for a non-string parameter's JSON value.
fn param_value(schema: &Value) -> Option<String> {
    match type_str(schema) {
        Some(t @ ("integer" | "number" | "boolean" | "null")) => scalar_rule(t).map(str::to_string),
        Some("object") => Some("jobj".into()),
        Some("array") => {
            let rule = schema
                .get("items")
                .filter(|i| i.is_object())
                .and_then(type_str)
                .and_then(scalar_rule);
            Some(match rule {
                None => "jarr".into(),
                Some(r) => format!("(\"[\" jws ({r} jws (\",\" jws {r} jws)*)? \"]\")"),
            })
        }
        _ => None,
    }
}

/// `_string_value`.
fn string_value(schema: &Value) -> String {
    if let Some(Value::Array(e)) = schema.get("enum")
        && !e.is_empty()
        && e.iter()
            .all(|x| x.as_str().is_some_and(|s| !s.contains('｜')))
    {
        let alts: Vec<String> = e.iter().filter_map(Value::as_str).map(lit).collect();
        return format!("({})", alts.join(" | "));
    }
    "vany".into()
}

/// `_functions`: OpenAI `{"function": {...}}` entries or flat function dicts.
fn functions(tools: &[Value]) -> Vec<&serde_json::Map<String, Value>> {
    tools
        .iter()
        .filter_map(|t| {
            let t = t.as_object()?;
            let f = match t.get("function") {
                Some(Value::Object(f)) => f,
                _ => t,
            };
            f.get("name").is_some_and(Value::is_string).then_some(f)
        })
        .collect()
}

/// `build_tool_grammar(tools, max_calls=...)`. None when no tool is usable.
pub fn build_tool_grammar(tools: &[Value], max_calls: usize) -> Option<String> {
    let q = "\"";
    let mut calls: Vec<String> = Vec::new();
    let mut rules: Vec<String> = Vec::new();
    let (mut need_json, mut need_vany, mut need_pname) = (false, false, false);

    for (i, f) in functions(tools).into_iter().enumerate() {
        let name = f.get("name").and_then(Value::as_str).unwrap_or_default();
        if !dsml_safe(name) {
            tracing::warn!("tool {name:?} cannot be expressed in DSML; left out of the grammar");
            continue;
        }
        let params = f.get("parameters").filter(|p| p.is_object());
        let props = params.and_then(|p| p.get("properties"));
        let required: Vec<&Value> = match params.and_then(|p| p.get("required")) {
            Some(Value::Array(r)) => r.iter().collect(),
            _ => Vec::new(),
        };
        let rid = format!("t{i}");
        let mut seq: Vec<String> = Vec::new();
        if let Some(Value::Object(props)) = props {
            for (j, (pname, pschema)) in props.iter().enumerate() {
                if !dsml_safe(pname) {
                    tracing::warn!(
                        "parameter {pname:?} of tool {name:?} cannot be expressed in DSML; left out"
                    );
                    continue;
                }
                let empty = Value::Object(Default::default());
                let pschema = if pschema.is_object() { pschema } else { &empty };
                let prid = format!("{rid}p{j}");
                let head = format!("{}{q}{pname}{q} string=", param_open());
                let jrule = param_value(pschema);
                let is_str = type_str(pschema) == Some("string");
                let val = string_value(pschema);
                let body = if let Some(jrule) = jrule {
                    need_json = true;
                    format!("{} {jrule}", lit(&format!("{head}{q}false{q}>")))
                } else if is_str || val != "vany" {
                    need_vany = need_vany || val == "vany";
                    format!("{} {val}", lit(&format!("{head}{q}true{q}>")))
                } else {
                    need_vany = true;
                    format!(
                        "{} (\"true\" | \"false\") {} vany",
                        lit(&format!("{head}{q}")),
                        lit(&format!("{q}>"))
                    )
                };
                rules.push(format!("{prid} ::= {body} {}", lit(&param_close())));
                let req = required.iter().any(|r| r.as_str() == Some(pname.as_str()));
                seq.push(if req { prid } else { format!("{prid}?") });
            }
        } else if params.and_then(|p| p.get("type")).and_then(Value::as_str) == Some("object") {
            need_vany = true;
            need_pname = true;
            let prid = format!("{rid}pany");
            rules.push(format!(
                "{prid} ::= {} pname {} (\"true\" | \"false\") {} vany {}",
                lit(&format!("{}{q}", param_open())),
                lit(&format!("{q} string={q}")),
                lit(&format!("{q}>")),
                lit(&param_close()),
            ));
            seq.push(format!("{prid}{{0,16}}"));
        }
        let open_tag = lit(&format!("{}{q}{name}{q}>\n", invoke_open()));
        let body = if seq.is_empty() {
            String::new()
        } else {
            format!(" {}", seq.join(" "))
        };
        rules.push(format!(
            "{rid} ::= {open_tag}{body} {}",
            lit(&invoke_close())
        ));
        calls.push(rid);
    }

    if calls.is_empty() {
        return None;
    }
    let mut parts = vec![
        format!(
            "# DSML tool calls for {} tool(s); generated by server/tool_grammar.py",
            calls.len()
        ),
        format!(
            "root ::= {} call{{1,{}}} {}",
            lit(&block_open()),
            max_calls.max(1),
            lit(&block_close())
        ),
        format!("call ::= {}", calls.join(" | ")),
    ];
    if need_vany {
        parts.push(format!("vany ::= {VALUE_CHAR}*"));
    }
    if need_pname {
        parts.push(format!("pname ::= {NAME_CHAR}*"));
    }
    parts.extend(rules);
    if need_json {
        parts.push(JSON_RULES.replace("JSON_STRING_CHAR", JSON_STRING_CHAR));
    }
    Some(parts.join("\n") + "\n")
}
