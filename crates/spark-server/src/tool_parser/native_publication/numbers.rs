// SPDX-License-Identifier: AGPL-3.0-only

//! Reject JSON numeric literals whose decimal value changes through serde's
//! bounded number representation. This also prevents a rounded fractional
//! literal from falsely satisfying an integer schema. Not arbitrary precision.

use serde_json::Value;

pub(super) fn parse(text: &str) -> Result<Value, String> {
    let value =
        serde_json::from_str(text).map_err(|_| "native typed argument is not valid JSON")?;
    let bytes = text.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        match bytes[pos] {
            b'"' => {
                pos += 1;
                while pos < bytes.len() && bytes[pos] != b'"' {
                    pos += if bytes[pos] == b'\\' { 2 } else { 1 };
                }
                pos += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = pos;
                while pos < bytes.len()
                    && matches!(bytes[pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    pos += 1;
                }
                let literal = &text[start..pos];
                let number: serde_json::Number = serde_json::from_str(literal)
                    .map_err(|_| "native numeric argument is not representable")?;
                if normalized(literal)? != normalized(&number.to_string())? {
                    return Err(
                        "native numeric argument loses decimal precision; refusing to rewrite it"
                            .into(),
                    );
                }
            }
            _ => pos += 1,
        }
    }
    Ok(value)
}

/// Exact sign/significand/base-10-exponent comparison, with insignificant
/// zeros removed. Inputs have already passed the real JSON number parser.
fn normalized(text: &str) -> Result<(bool, String, i64), String> {
    let (negative, text) = match text.strip_prefix('-') {
        Some(text) => (true, text),
        None => (false, text),
    };
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (
            &text[..at],
            text[at + 1..]
                .parse::<i64>()
                .map_err(|_| "native numeric exponent is unsupported")?,
        ),
        None => (text, 0),
    };
    let fractional = mantissa.find('.').map_or(0, |at| mantissa.len() - at - 1);
    let digits = mantissa.replace('.', "");
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return Ok((false, "0".into(), 0));
    }
    let trimmed = significant.trim_end_matches('0');
    let trailing = i64::try_from(significant.len() - trimmed.len())
        .map_err(|_| "native numeric literal is too long")?;
    let fractional = i64::try_from(fractional).map_err(|_| "native numeric literal is too long")?;
    let exponent = exponent
        .checked_sub(fractional)
        .and_then(|e| e.checked_add(trailing))
        .ok_or("native numeric exponent is unsupported")?;
    Ok((negative, trimmed.to_owned(), exponent))
}
