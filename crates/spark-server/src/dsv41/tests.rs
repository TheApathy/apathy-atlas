// SPDX-License-Identifier: AGPL-3.0-only

//! Differential tests against fixtures produced by the PRODUCTION Python code
//! (scripts/dsv41_parity/gen_fixtures.py): app.py, the checkpoint's
//! encoding/encoding.py, and tool_grammar.py compiled by Python xgrammar.
//!
//! Every comparison has a negative control that is required to FAIL, so none
//! of these gates can pass vacuously.

use serde_json::Value;

use super::{encoding, grammar, parse, request};

pub(super) fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/dsv41/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).expect("fixture json")
}

pub(super) fn cases(v: &Value) -> &Vec<Value> {
    v["cases"].as_array().expect("cases")
}

/// The checkpoint tokenizer. Missing it is a hard failure, not a skip: a
/// skipped id comparison is a gate that cannot fail.
pub(super) fn tokenizer(meta: &Value) -> tokenizers::Tokenizer {
    let dir = meta["model_dir"].as_str().expect("model_dir");
    let path = std::env::var("DSV41_TOKENIZER").unwrap_or_else(|_| format!("{dir}/tokenizer.json"));
    tokenizers::Tokenizer::from_file(&path).unwrap_or_else(|e| panic!("tokenizer {path}: {e}"))
}

fn encode(tok: &tokenizers::Tokenizer, s: &str) -> Vec<u32> {
    tok.encode(s, false).expect("encode").get_ids().to_vec()
}

// ------------------------------------------------------------------ render

#[test]
fn render_fixtures_match_python_byte_and_token_exact() {
    let fx = fixture("render.json");
    let tok = tokenizer(&fx["meta"]);
    let (mut ok, mut errs) = (0, 0);
    for c in cases(&fx) {
        let name = c["name"].as_str().unwrap();
        let got = request::render_request(&c["body"]);
        if c.get("error").is_some() {
            assert!(
                got.is_err(),
                "{name}: Python refused ({}) but Rust rendered",
                c["error"]["message"]
            );
            errs += 1;
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: Rust refused: {e}"));
        assert_eq!(
            got.thinking,
            c["thinking"].as_bool().unwrap(),
            "{name}: thinking"
        );
        assert_eq!(
            got.effort as u64,
            c["effort"].as_u64().unwrap(),
            "{name}: effort"
        );
        let want = c["prompt"].as_str().unwrap();
        assert_eq!(got.prompt, want, "{name}: prompt differs");
        let want_ids: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            encode(&tok, &got.prompt),
            want_ids,
            "{name}: token ids differ"
        );
        assert_eq!(
            got.images.len() as u64,
            c["n_images"].as_u64().unwrap(),
            "{name}: images"
        );
        let placeholders = want_ids.iter().filter(|&&t| t == 129264).count();
        assert_eq!(
            placeholders,
            got.images.len(),
            "{name}: placeholder tokens vs images"
        );
        let gt = got
            .grammar_tools
            .clone()
            .map(Value::Array)
            .unwrap_or(Value::Null);
        assert_eq!(gt, c["grammar_tools"], "{name}: grammar tools");
        ok += 1;
    }
    assert!(
        ok >= 30 && errs >= 8,
        "fixture coverage shrank: {ok} rendered, {errs} errors"
    );
}

/// NEGATIVE CONTROL: each perturbation is a real mistake a port could make.
/// Every one must change the rendered prompt of at least one fixture case, or
/// the comparison above would not have caught it.
#[test]
fn render_comparison_detects_plausible_port_mistakes() {
    let fx = fixture("render.json");
    let rendered: Vec<(String, String)> = cases(&fx)
        .iter()
        .filter(|c| c.get("prompt").is_some())
        .map(|c| {
            (
                c["name"].as_str().unwrap().to_string(),
                c["prompt"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let mistakes: [(&str, fn(&str) -> String); 5] = [
        ("V4 DSML spelling (no leading space)", |p| {
            p.replace("｜DSML｜ ", "｜DSML｜")
        }),
        ("effort not rendered", |p| {
            match (p.find("Reasoning Effort: "), p.find("reasoning)\n\n")) {
                (Some(a), Some(b)) => format!("{}{}", &p[..a], &p[b + "reasoning)\n\n".len()..]),
                _ => p.to_string(),
            }
        }),
        ("no <｜System｜> token", |p| p.replace("<｜System｜>", "")),
        ("compact json separators", |p| p.replace("\", \"", "\",\"")),
        ("tool results unsorted (a<->c)", |p| {
            p.replace("<tool_result>18C</tool_result>", "@@")
                .replace(
                    "<tool_result>headline 1\nheadline 2</tool_result>",
                    "<tool_result>18C</tool_result>",
                )
                .replace("@@", "<tool_result>headline 1\nheadline 2</tool_result>")
        }),
    ];
    for (what, f) in mistakes {
        let caught = rendered.iter().filter(|(_, p)| f(p) != *p).count();
        assert!(
            caught > 0,
            "mistake '{what}' changes no fixture prompt: the gate cannot see it"
        );
    }
}

#[test]
fn thinking_resolution_contract() {
    use serde_json::json;
    let r = |b: Value| request::resolve_thinking(&b, false, 75).unwrap();
    assert_eq!(r(json!({})), (false, 75));
    assert_eq!(r(json!({"reasoning_effort": "low"})), (false, 50));
    assert_eq!(r(json!({"reasoning_effort": "medium"})), (true, 60));
    assert_eq!(r(json!({"reasoning_effort": 1})), (true, 1));
    assert_eq!(
        r(json!({"reasoning_effort": "none", "enable_thinking": true})),
        (true, 75)
    );
    assert!(request::resolve_thinking(&json!({"reasoning_effort": 101}), false, 75).is_err());
}

// ------------------------------------------------------------------ parse

fn calls_json(calls: &[parse::ParsedCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| serde_json::json!({"name": c.name, "arguments": c.arguments}))
            .collect(),
    )
}

#[test]
fn parse_fixtures_match_python() {
    let fx = fixture("parse.json");
    for c in cases(&fx) {
        let name = c["name"].as_str().unwrap();
        let text = c["text"].as_str().unwrap();
        let thinking = c["mode"] == "thinking";
        let full = if text.ends_with(encoding::EOS) {
            text.to_string()
        } else {
            format!("{text}{}", encoding::EOS)
        };

        // strict: same verdict, same result
        match (parse::parse_strict(&full, thinking), c.get("strict")) {
            (Ok(m), Some(want)) => {
                assert_eq!(
                    m.content,
                    want["content"].as_str().unwrap(),
                    "{name}: strict content"
                );
                assert_eq!(
                    m.reasoning_content,
                    want["reasoning_content"].as_str().unwrap(),
                    "{name}: reasoning"
                );
                let got: Vec<Value> = m
                    .tool_calls
                    .iter()
                    .map(|t| {
                        let mut o = serde_json::json!({"name": t.name, "arguments": t.arguments});
                        if let Some(ns) = &t.namespace {
                            o["namespace"] = Value::String(ns.clone());
                        }
                        o
                    })
                    .collect();
                assert_eq!(
                    Value::Array(got),
                    want["tool_calls"],
                    "{name}: strict calls"
                );
            }
            (Err(e), None) => assert!(c.get("strict_error").is_some(), "{name}: {e}"),
            (Ok(m), None) => panic!(
                "{name}: Python strict raised {} but Rust parsed {m:?}",
                c["strict_error"]
            ),
            (Err(e), Some(_)) => panic!("{name}: Rust strict failed ({e}) where Python parsed"),
        }

        assert_eq!(
            calls_json(&parse::parse_tolerant(&full)),
            c["tolerant"],
            "{name}: tolerant"
        );

        // full server pipeline, whole text and in awkward chunks (router must be chunk-invariant)
        for chunk in [usize::MAX, 1, 3, 7] {
            let mut router = parse::OutputRouter::new(thinking, vec![], true);
            let chars: Vec<char> = text.chars().collect();
            for piece in chars.chunks(chunk.min(chars.len().max(1))) {
                router.feed(&piece.iter().collect::<String>());
            }
            router.finish();
            let calls = router.parse_tool_calls(thinking);
            let s = &c["server"];
            assert_eq!(
                router.reasoning,
                s["reasoning"].as_str().unwrap(),
                "{name}/{chunk}: reasoning"
            );
            assert_eq!(
                router.content,
                s["content"].as_str().unwrap(),
                "{name}/{chunk}: content"
            );
            assert_eq!(calls_json(&calls), s["tool_calls"], "{name}/{chunk}: calls");
        }
    }
}

/// NEGATIVE CONTROL: the V4 (no-space) spelling must NOT parse as V4.1, and a
/// strict parser that ignored the `>\n` framing would accept `missing_newline`.
#[test]
fn parse_controls_fail_where_they_must() {
    let fx = fixture("parse.json");
    let by = |n: &str| cases(&fx).iter().find(|c| c["name"] == n).unwrap().clone();
    let v4 = by("v4_spelling_is_not_v41");
    assert!(parse::parse_tolerant(v4["text"].as_str().unwrap()).is_empty());
    let one = by("one_call");
    let mutated = one["text"]
        .as_str()
        .unwrap()
        .replace("｜DSML｜ ", "｜DSML｜");
    let mut router = parse::OutputRouter::new(false, vec![], true);
    router.feed(&mutated);
    router.finish();
    assert!(
        router.parse_tool_calls(false).is_empty(),
        "V4 spelling was parsed as a V4.1 call"
    );
    let mn = by("missing_newline");
    let t = format!("{}{}", mn["text"].as_str().unwrap(), encoding::EOS);
    assert!(parse::parse_strict(&t, false).is_err());
}

#[test]
fn stop_string_and_marker_holdback() {
    let mut r = parse::OutputRouter::new(true, vec!["STOP".into()], true);
    let mut out = String::new();
    for piece in ["reason</th", "ink>ans\n", "\n<｜DS", "ML｜ calls>\n..."] {
        for (_, t) in r.feed(piece) {
            out.push_str(&t);
        }
    }
    assert_eq!(r.reasoning, "reason");
    assert_eq!(r.content, "ans");
    assert_eq!(
        out, "reasonans",
        "the partial marker must be held back, never streamed"
    );
    assert!(r.tool_text.starts_with("\n\n<｜DSML｜ calls>"));
    let mut s = parse::OutputRouter::new(false, vec!["STOP".into()], false);
    s.feed("abc ST");
    s.feed("OP tail");
    s.finish();
    assert!(s.stopped);
    assert_eq!(s.content, "abc ");
}

// ------------------------------------------------------------------ grammar

#[test]
fn grammar_text_is_byte_identical_to_python() {
    let fx = fixture("grammar.json");
    for c in cases(&fx) {
        let name = c["name"].as_str().unwrap();
        let tools = c["tools"].as_array().unwrap();
        let got = grammar::build_tool_grammar(tools, grammar::DEFAULT_MAX_CALLS);
        assert_eq!(got.as_deref(), c["ebnf"].as_str(), "{name}: EBNF differs");
    }
}
