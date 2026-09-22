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

/// The EBNF is the same text; does the RUST xgrammar port enforce it the way
/// Python xgrammar does? Walk every recorded token stream and compare the
/// allowed-token set (size and digest) at every step, and the accept verdicts.
/// The streams include refusals (U+FF5C in a value, missing required param,
/// enum violation, bad bool), so both sides must also REJECT at the same token.
#[test]
fn rust_xgrammar_masks_match_python_xgrammar() {
    use crate::grammar::{GrammarEngine, GrammarState};
    let fx = fixture("grammar.json");
    let tok = tokenizer(&fx["meta"]);
    let mut engine = GrammarEngine::from_tokenizer(&tok, None, &[1]).expect("engine");
    let vocab = engine.vocab_size();
    let mut streams = 0;
    let mut rejections = 0;
    for c in cases(&fx) {
        let name = c["name"].as_str().unwrap();
        let Some(ebnf) = c["ebnf"].as_str() else { continue };
        let compiled = engine.compile_ebnf(ebnf, "root").unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for s in c["streams"].as_array().unwrap() {
            let mut st = GrammarState::new(&compiled, vocab).expect("state");
            for (i, step) in s["steps"].as_array().unwrap().iter().enumerate() {
                st.fill_bitmask();
                let allowed: Vec<u32> = (0..vocab as u32).filter(|&t| st.is_token_allowed(t)).collect();
                assert_eq!(allowed.len() as u64, step["n_allowed"].as_u64().unwrap(), "{name} step {i}: mask size");
                assert_eq!(digest(&allowed), step["digest"].as_str().unwrap(), "{name} step {i}: mask set");
                let t = step["token"].as_u64().unwrap() as u32;
                let ok = st.accept_token(t);
                assert_eq!(ok, step["accepted"].as_bool().unwrap(), "{name} step {i}: accept({t})");
                if !ok {
                    rejections += 1;
                }
            }
            assert_eq!(st.is_terminated(), s["terminated"].as_bool().unwrap(), "{name}: terminated");
            streams += 1;
        }
    }
    assert!(streams >= 7 && rejections >= 4, "{streams} streams, {rejections} rejections");
}
