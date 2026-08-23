// SPDX-License-Identifier: AGPL-3.0-only

fn is_ident_char(value: char) -> bool {
    value.is_ascii_alphanumeric() || value == '_'
}

fn raw_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
        return None;
    }
    let r = if bytes.get(start) == Some(&b'r') {
        start
    } else if matches!(bytes.get(start), Some(b'b' | b'c')) && bytes.get(start + 1) == Some(&b'r') {
        start + 1
    } else {
        return None;
    };
    let mut quote = r + 1;
    while bytes.get(quote) == Some(&b'#') {
        quote += 1;
    }
    if bytes.get(quote) != Some(&b'"') {
        return None;
    }
    let hashes = quote - r - 1;
    let mut end_quote = quote + 1;
    while end_quote < bytes.len() {
        if bytes[end_quote] == b'"'
            && bytes.get(end_quote + 1..end_quote + 1 + hashes) == Some(&bytes[r + 1..quote])
        {
            return Some(end_quote + 1 + hashes);
        }
        end_quote += 1;
    }
    None
}

fn char_literal_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    if bytes.get(start) != Some(&b'\'') {
        return None;
    }
    let content = start + 1;
    let end = if bytes.get(content) == Some(&b'\\') {
        content + 2
    } else {
        let value = source.get(content..)?.chars().next()?;
        if matches!(value, '\'' | '\\' | '\n' | '\r') {
            return None;
        }
        content + value.len_utf8()
    };
    (bytes.get(end) == Some(&b'\'')).then_some(end + 1)
}

fn blank(clean: &mut Vec<u8>, bytes: &[u8]) {
    clean.extend(
        bytes
            .iter()
            .map(|&value| if value == b'\n' { b'\n' } else { b' ' }),
    );
}

fn without_rust_trivia(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut clean = Vec::with_capacity(bytes.len());
    let (mut i, mut block_depth, mut line_comment) = (0, 0usize, false);
    let (mut quoted, mut escaped) = (false, false);
    while i < bytes.len() {
        let byte = bytes[i];
        let next = bytes.get(i + 1).copied();
        if line_comment {
            clean.push(if byte == b'\n' { b'\n' } else { b' ' });
            line_comment = byte != b'\n';
        } else if block_depth > 0 {
            clean.push(if byte == b'\n' { b'\n' } else { b' ' });
            if byte == b'/' && next == Some(b'*') {
                clean.push(b' ');
                block_depth += 1;
                i += 1;
            } else if byte == b'*' && next == Some(b'/') {
                clean.push(b' ');
                block_depth -= 1;
                i += 1;
            }
        } else if quoted {
            clean.push(if byte == b'\n' { b'\n' } else { b' ' });
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if let Some(end) = char_literal_end(source, i) {
            blank(&mut clean, &bytes[i..end]);
            i = end;
            continue;
        } else if let Some(end) = raw_literal_end(bytes, i) {
            blank(&mut clean, &bytes[i..end]);
            i = end;
            continue;
        } else if byte == b'/' && next == Some(b'/') {
            clean.extend_from_slice(b"  ");
            line_comment = true;
            i += 1;
        } else if byte == b'/' && next == Some(b'*') {
            clean.extend_from_slice(b"  ");
            block_depth = 1;
            i += 1;
        } else if byte == b'"' {
            clean.push(b' ');
            quoted = true;
        } else {
            clean.push(byte);
        }
        i += 1;
    }
    String::from_utf8(clean).expect("Rust source stays UTF-8")
}

fn compact_rust(source: &str) -> String {
    without_rust_trivia(source)
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect()
}

fn source_fingerprint(source: &str) -> u64 {
    source.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn ident_tokens(clean: &str, ident: &str) -> usize {
    clean
        .match_indices(ident)
        .filter(|&(at, _)| at == 0 || !clean[..at].ends_with(is_ident_char))
        .filter(|&(at, name)| !clean[at + name.len()..].starts_with(is_ident_char))
        .count()
}

fn has_exact_swap_block(clean: &str) -> bool {
    const EXPECTED: &str = concat!(
        "ifletSome(refmutspill)=spill_manager{forreqin&new_reqs{",
        "letprompt_len=req.prompt_len();",
        "letblocks_needed=prompt_len/block_size+1;",
        "whilemodel.num_free_blocks()<blocks_needed&&!active.is_empty(){",
        "letvictim_idx=active.iter().enumerate()",
        ".filter(|(_,a)|a.grammar_state.is_none())",
        ".max_by_key(|(_,a)|a.seq.block_table.len()).map(|(i,_)|i);",
        "letSome(victim_idx)=victim_idxelse{tracing::warn!();break;};",
        "matchswap_out_sequence(&*model,&mutactive,victim_idx,spill){",
        "Ok(s)=>{tracing::info!(,s.seq_len,s.num_blocks,);swapped.push(s);}",
        "Err(e)=>{tracing::error!();break;}}}}}",
    );
    compact_rust(clean) == EXPECTED
}

pub(super) fn victim_selector(source: &str) -> Option<String> {
    const START: &str = "// ── Swap-out: evict active sequences to disk when blocks run low ──";
    const END: &str = "// ── Start new requests ──";
    const SOURCE_LEN: usize = 41_276;
    const SOURCE_FNV1A: u64 = 0xc67e9f9382b21e68;
    if source.matches(START).count() != 1 || source.matches(END).count() != 1 {
        return None;
    }
    let whole_compact = compact_rust(source);
    if source.len() != SOURCE_LEN || source_fingerprint(source) != SOURCE_FNV1A {
        return None;
    }
    let (_, swap_tail) = source.split_once(START)?;
    let (swap_out, _) = swap_tail.split_once(END)?;
    let clean = without_rust_trivia(swap_out);
    let whole_clean = without_rust_trivia(source);
    let bindings: Vec<_> = clean.match_indices("let victim_idx =").collect();
    if bindings.len() != 1
        || ident_tokens(&clean, "victim_idx") != 4
        || !has_exact_swap_block(&clean)
        || whole_compact.matches("swap_out_sequence(").count() != 1
        || [
            "create_file",
            "save_sequence_state",
            "swap_remove",
            "compact_sequence",
            "pack_swapped_seq",
            "restore_sequence_state",
        ]
        .iter()
        .any(|ident| ident_tokens(&whole_clean, ident) != 0)
        || ident_tokens(&whole_clean, "remove_file") != 1
        || ident_tokens(&whole_clean, "free_sequence") != 1
    {
        return None;
    }
    let tail = &swap_out[bindings[0].0 + "let victim_idx =".len()..];
    let (selector, _) = tail.split_once(';')?;
    Some(
        selector
            .chars()
            .filter(|value| !value.is_ascii_whitespace())
            .collect(),
    )
}

pub(super) fn shadow_selector(source: &str, binding: &str) -> String {
    source.replacen(
        "let Some(victim_idx) = victim_idx else",
        &format!("{binding};\n                    let Some(victim_idx) = victim_idx else"),
        1,
    )
}

pub(super) fn guard_precedes_spill(source: &str) -> bool {
    const GUARD: &str = "ensure_swappable_victim(active,victim_idx)?;";
    const CREATE: &str = "spill.create_file()?;";
    const GUARD_BODY: &str = concat!(
        "fnensure_swappable_victim(active:&[ActiveSeq],victim_idx:usize)->Result<()>{",
        "anyhow::ensure!(active.get(victim_idx)",
        ".is_some_and(|victim|victim.grammar_state.is_none()),);Ok(())}"
    );
    const PREFIX: &str = concat!(
        "pub(incrate::scheduler)fnswap_out_sequence(",
        "model:&dynModel,active:&mutVec<ActiveSeq>,victim_idx:usize,",
        "spill:&mutKvSpillManager,)->Result<SwappedSeq>{",
        "ensure_swappable_victim(active,victim_idx)?;",
        "let(swap_id,mutwriter)=spill.create_file()?;"
    );
    let clean = without_rust_trivia(source);
    let compact = compact_rust(source);
    compact.starts_with("usesuper::*;fnensure_swappable_victim(")
        && ident_tokens(&clean, "ensure_swappable_victim") == 2
        && compact.matches(GUARD_BODY).count() == 1
        && compact.matches(GUARD).count() == 1
        && compact.matches(CREATE).count() == 1
        && compact.matches(PREFIX).count() == 1
}

#[cfg(test)]
#[path = "swap_lifecycle_source_scan_tests.rs"]
mod tests;
