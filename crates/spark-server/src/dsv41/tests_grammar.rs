// SPDX-License-Identifier: AGPL-3.0-only

//! The DSML grammar through Atlas's OWN matcher (the pure-Rust xgrammar port
//! behind `crate::grammar`), step-compared with Python xgrammar.

use super::tests::{cases, fixture, tokenizer};

/// FNV-1a 64 of the comma-joined allowed ids (same digest the generator writes).
fn digest(allowed: &[u32]) -> String {
    let s: Vec<String> = allowed.iter().map(u32::to_string).collect();
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.join(",").bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Tokens whose bytes could belong to a U+F000..U+FF3F character (EF followed
/// by 80..BC, a trailing EF, or a leading continuation byte); the generator
/// uses the same definition. Python xgrammar 0.1.32 wrongly excludes that
/// codepoint range from `[\u0000-\uFF5B]`; Atlas's port FIXES it
/// (xgrammar `codepoint_ranges_are_exact_across_utf8_lengths`), so the masks
/// are compared with these tokens left out.
fn hole_tokens(tok: &tokenizers::Tokenizer) -> std::collections::HashSet<u32> {
    let mut bs: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let table: std::collections::HashMap<char, u8> = bs
        .iter()
        .zip(&cs)
        .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
        .collect();
    tok.get_vocab(true)
        .into_iter()
        .filter_map(|(piece, id)| {
            let b: Vec<u8> = piece
                .chars()
                .map(|c| table.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .unwrap_or_else(|| piece.as_bytes().to_vec());
            let hole = !b.is_empty()
                && ((0x80..=0xBF).contains(&b[0]) || *b.last().unwrap() == 0xEF)
                || b.windows(2)
                    .any(|w| w[0] == 0xEF && (0x80..=0xBC).contains(&w[1]));
            hole.then_some(id)
        })
        .collect()
}

/// The EBNF is the same text; does the RUST xgrammar port enforce it the way
/// Python xgrammar does? Walk every recorded token stream and compare the
/// allowed-token set (size and digest, hole tokens excluded) at every step,
/// and the accept verdicts. The streams include refusals (U+FF5C in a value,
/// missing required param, enum violation, bad bool), so both sides must also
/// REJECT at the same token. The one stream with a U+F000..U+FF3F character
/// ("，") is where the two DIVERGE on purpose: Python rejects it (the upstream
/// bug), Rust must accept the whole stream.
#[test]
fn rust_xgrammar_masks_match_python_xgrammar() {
    use crate::grammar::{GrammarEngine, GrammarState};
    let fx = fixture("grammar.json");
    let tok = tokenizer(&fx["meta"]);
    let hole = hole_tokens(&tok);
    let mut engine = GrammarEngine::from_tokenizer(&tok, None, &[1]).expect("engine");
    let vocab = engine.vocab_size();
    let (mut streams, mut rejections, mut fixed_streams) = (0, 0, 0);
    for c in cases(&fx) {
        let name = c["name"].as_str().unwrap();
        let Some(ebnf) = c["ebnf"].as_str() else {
            continue;
        };
        let compiled = engine
            .compile_ebnf(ebnf, "root")
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for s in c["streams"].as_array().unwrap() {
            let mut st = GrammarState::new(&compiled, vocab).expect("state");
            if s["hole_in_text"].as_bool() == Some(true) {
                // Python refused mid-value; the fixed Rust matcher takes every token.
                let py_rejected = s["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|x| x["accepted"] == false);
                assert!(
                    py_rejected,
                    "{name}: the fixture no longer shows the upstream bug"
                );
                for t in s["ids"].as_array().unwrap() {
                    st.fill_bitmask();
                    assert!(
                        st.accept_token(t.as_u64().unwrap() as u32),
                        "{name}: fixed matcher refused {t}"
                    );
                }
                fixed_streams += 1;
                continue;
            }
            for (i, step) in s["steps"].as_array().unwrap().iter().enumerate() {
                st.fill_bitmask();
                let allowed: Vec<u32> = (0..vocab as u32)
                    .filter(|&t| st.is_token_allowed(t) && !hole.contains(&t))
                    .collect();
                assert_eq!(
                    allowed.len() as u64,
                    step["n_allowed_nohole"].as_u64().unwrap(),
                    "{name} step {i}: mask size"
                );
                assert_eq!(
                    digest(&allowed),
                    step["digest_nohole"].as_str().unwrap(),
                    "{name} step {i}: mask set"
                );
                let t = step["token"].as_u64().unwrap() as u32;
                let ok = st.accept_token(t);
                assert_eq!(
                    ok,
                    step["accepted"].as_bool().unwrap(),
                    "{name} step {i}: accept({t})"
                );
                if !ok {
                    rejections += 1;
                }
            }
            assert_eq!(
                st.is_terminated(),
                s["terminated"].as_bool().unwrap(),
                "{name}: terminated"
            );
            streams += 1;
        }
    }
    assert!(
        streams >= 7 && rejections >= 4 && fixed_streams == 1,
        "{streams} streams, {rejections} rejections, {fixed_streams} fixed"
    );
}

/// Mask dump for the xgrammar range-fix before/after comparison (run on each
/// build, diff with DSV41_PORT/parity/diff_range_fix_masks.py). Ignored unless
/// MASK_DUMP names an output file. Grammars: Qwen3.5 qwen3_coder and hermes
/// tool grammars (non-DeepSeek), plus the V4.1 DSML grammar, each walked into
/// a parameter VALUE, where the fix can matter.
#[test]
#[ignore = "diagnostic dump; set MASK_DUMP=path"]
fn dump_value_masks_for_range_fix() {
    use crate::grammar::{GrammarEngine, GrammarState};
    let Ok(out) = std::env::var("MASK_DUMP") else {
        return;
    };
    let tool: crate::tool_parser::ToolDefinition = serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {"name": "write_file", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}}))
    .unwrap();
    let qwen = tokenizers::Tokenizer::from_file(
        "/home/flocka/models/Qwen3.5-27B-Text-NVFP4-MTP/tokenizer.json",
    )
    .unwrap();
    let fx = fixture("grammar.json");
    let ds = tokenizer(&fx["meta"]);
    let enc = |t: &tokenizers::Tokenizer, s: &str| t.encode(s, false).unwrap().get_ids().to_vec();
    let mut dump = serde_json::Map::new();
    let mut walk = |name: &str,
                    tok: &tokenizers::Tokenizer,
                    stop: u32,
                    compile: &dyn Fn(&mut GrammarEngine) -> xgrammar::CompiledGrammar,
                    prefix: &str| {
        let mut engine = GrammarEngine::from_tokenizer(tok, None, &[stop as i32]).unwrap();
        let vocab = engine.vocab_size();
        let compiled = compile(&mut engine);
        let mut st = GrammarState::new(&compiled, vocab).unwrap();
        let mut steps = Vec::new();
        for t in enc(tok, prefix) {
            st.fill_bitmask();
            let allowed: Vec<u32> = (0..vocab as u32)
                .filter(|&i| st.is_token_allowed(i))
                .collect();
            steps.push(serde_json::json!(allowed));
            assert!(st.accept_token(t), "{name}: prefix token {t} refused");
        }
        st.fill_bitmask();
        let allowed: Vec<u32> = (0..vocab as u32)
            .filter(|&i| st.is_token_allowed(i))
            .collect();
        steps.push(serde_json::json!(allowed));
        dump.insert(name.into(), serde_json::Value::Array(steps));
    };
    let t1 = tool.clone();
    walk(
        "qwen3_coder",
        &qwen,
        248046,
        &move |e| {
            e.compile_qwen3_coder_tool_grammar(&[t1.clone()], false, "</parameter>")
                .unwrap()
        },
        "<tool_call>\n<function=write_file>\n<parameter=path>\nx",
    );
    let t2 = tool.clone();
    walk(
        "hermes",
        &qwen,
        248046,
        &move |e| e.compile_hermes_tool_grammar(&[t2.clone()], false).unwrap(),
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"x",
    );
    let dsml =
        crate::dsv41::grammar::build_tool_grammar(&[serde_json::to_value(&tool).unwrap()], 8)
            .unwrap();
    walk(
        "dsml_v41",
        &ds,
        1,
        &move |e| e.compile_ebnf(&dsml, "root").unwrap(),
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write_file\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">x",
    );
    std::fs::write(out, serde_json::to_string(&dump).unwrap()).unwrap();
}
