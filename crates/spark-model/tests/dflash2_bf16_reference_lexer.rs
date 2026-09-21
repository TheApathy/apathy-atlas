// SPDX-License-Identifier: AGPL-3.0-only

fn push_token(output: &mut String, kind: char, token: &str) {
    use std::fmt::Write as _;

    write!(output, "{kind}{:x}:{token};", token.len()).unwrap();
}

fn string_prefix_len(bytes: &[u8], start: usize) -> Option<usize> {
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_alphabetic() && end - start < 2 {
        end += 1;
    }
    if end == start || end >= bytes.len() || !matches!(bytes[end], b'\'' | b'"') {
        return None;
    }
    let prefix = std::str::from_utf8(&bytes[start..end]).unwrap();
    matches!(
        prefix.to_ascii_lowercase().as_str(),
        "r" | "u" | "b" | "f" | "br" | "rb" | "fr" | "rf"
    )
    .then_some(end - start)
}

fn quoted_end(bytes: &[u8], quote_start: usize) -> usize {
    let quote = bytes[quote_start];
    let triple =
        bytes.get(quote_start + 1) == Some(&quote) && bytes.get(quote_start + 2) == Some(&quote);
    let width = if triple { 3 } else { 1 };
    let mut cursor = quote_start + width;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\\' {
            cursor = cursor
                .checked_add(2)
                .filter(|next| *next <= bytes.len())
                .expect("unterminated quoted string");
            continue;
        }
        if bytes[cursor] == quote
            && (!triple
                || (bytes.get(cursor + 1) == Some(&quote) && bytes.get(cursor + 2) == Some(&quote)))
        {
            return cursor + width;
        }
        assert!(
            triple || bytes[cursor] != b'\n',
            "unterminated quoted string"
        );
        cursor += 1;
    }
    panic!("unterminated quoted string");
}

fn number_end(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    let mut seen_dot = false;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if byte.is_ascii_alphanumeric() || byte == b'_' {
            cursor += 1;
        } else if byte == b'.' && !seen_dot && bytes.get(cursor + 1) != Some(&b'.') {
            seen_dot = true;
            cursor += 1;
        } else if matches!(byte, b'+' | b'-')
            && cursor > start
            && matches!(bytes[cursor - 1], b'e' | b'E')
        {
            cursor += 1;
        } else {
            break;
        }
    }
    cursor
}

pub(super) fn python_canonical(source: &str) -> String {
    assert!(source.is_ascii(), "official Python receipt must be ASCII");
    const OPERATORS: [&str; 26] = [
        "**=", "//=", "<<=", ">>=", "...", ":=", "==", "!=", "<=", ">=", "->", "**", "//", "<<",
        ">>", "+=", "-=", "*=", "/=", "%=", "@=", "&=", "|=", "^=", "@", "~",
    ];
    let bytes = source.as_bytes();
    let mut output = String::new();
    let mut cursor = 0;
    let mut brackets = Vec::new();
    let mut indents = vec![0_usize];
    let mut line_start = true;
    let mut line_has_code = false;
    let mut last_colon = false;
    let mut suite_pending = false;
    while cursor < bytes.len() {
        if line_start {
            let start = cursor;
            while bytes.get(cursor) == Some(&b' ') {
                cursor += 1;
            }
            if cursor == bytes.len() {
                break;
            }
            if bytes[cursor] == b'\n' {
                cursor += 1;
                continue;
            }
            if brackets.is_empty() && bytes[cursor] != b'#' {
                let width = cursor - start;
                if width > *indents.last().unwrap() {
                    assert!(suite_pending, "indentation does not follow a suite");
                    indents.push(width);
                    push_token(&mut output, 'I', "");
                } else {
                    while width < *indents.last().unwrap() {
                        indents.pop();
                        push_token(&mut output, 'D', "");
                    }
                    assert_eq!(width, *indents.last().unwrap(), "inconsistent dedent");
                }
                suite_pending = false;
            }
            line_start = false;
        }
        assert!(
            !matches!(bytes[cursor], b'\t' | b'\r' | 0x0b | 0x0c),
            "unsupported structural whitespace"
        );
        if bytes[cursor] == b'\n' {
            if brackets.is_empty() {
                if line_has_code {
                    push_token(&mut output, 'L', "");
                    suite_pending = last_colon;
                    line_has_code = false;
                    last_colon = false;
                }
            }
            line_start = true;
            cursor += 1;
            continue;
        }
        if bytes[cursor] == b' ' {
            cursor += 1;
            continue;
        }
        if bytes[cursor] == b'#' {
            let start = cursor;
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            push_token(&mut output, 'c', &source[start..cursor]);
            continue;
        }
        let prefix_len = string_prefix_len(bytes, cursor).unwrap_or(0);
        let quote_start = cursor + prefix_len;
        if matches!(bytes.get(quote_start), Some(&b'\'') | Some(&b'"')) {
            let end = quoted_end(bytes, quote_start);
            push_token(&mut output, 's', &source[cursor..end]);
            cursor = end;
            line_has_code = true;
            last_colon = false;
            continue;
        }
        if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            push_token(&mut output, 'i', &source[start..cursor]);
            line_has_code = true;
            last_colon = false;
            continue;
        }
        if bytes[cursor].is_ascii_digit()
            || (bytes[cursor] == b'.' && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit))
        {
            let end = number_end(bytes, cursor);
            push_token(&mut output, 'n', &source[cursor..end]);
            cursor = end;
            line_has_code = true;
            last_colon = false;
            continue;
        }
        let remaining = &source[cursor..];
        let operator = OPERATORS
            .iter()
            .find(|operator| remaining.starts_with(**operator));
        let end = cursor + operator.map_or(1, |operator| operator.len());
        let token = &source[cursor..end];
        if token.len() == 1 {
            match token.as_bytes()[0] {
                b'(' | b'[' | b'{' => brackets.push(token.as_bytes()[0]),
                b')' | b']' | b'}' => {
                    let expected = match token.as_bytes()[0] {
                        b')' => b'(',
                        b']' => b'[',
                        _ => b'{',
                    };
                    assert_eq!(brackets.pop(), Some(expected), "unmatched delimiter");
                }
                _ => {}
            }
        }
        push_token(&mut output, 'p', token);
        line_has_code = true;
        last_colon = token == ":";
        cursor = end;
    }
    assert!(brackets.is_empty(), "unclosed delimiter");
    assert!(!suite_pending && !last_colon, "suite has no body");
    if line_has_code {
        push_token(&mut output, 'L', "");
    }
    while indents.len() > 1 {
        indents.pop();
        push_token(&mut output, 'D', "");
    }
    output
}

#[cfg(test)]
#[path = "dflash2_bf16_reference_lexer_tests.rs"]
mod tests;
