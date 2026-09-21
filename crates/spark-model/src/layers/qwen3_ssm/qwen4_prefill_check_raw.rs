// SPDX-License-Identifier: AGPL-3.0-only

//! Raw diagnostic equality: never tolerance-promote or accept nonfinite values.

#[derive(Clone, Copy)]
pub(super) enum Element {
    Bf16,
    F32,
}

impl Element {
    fn width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

pub(super) fn validate_check(exact: bool, check: bool) -> Result<(), &'static str> {
    if check && !exact {
        Err("ATLAS_QWEN4_PREFILL_SSM_CHECK requires ATLAS_QWEN4_PREFILL_SSM_EXACT=1")
    } else {
        Ok(())
    }
}

pub(super) fn finite(bytes: &[u8], element: Element) -> Result<(), String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(element.width()) {
        return Err("empty or partial SSM check element".into());
    }
    for (index, raw) in bytes.chunks_exact(element.width()).enumerate() {
        let nonfinite = match element {
            Element::Bf16 => u16::from_ne_bytes([raw[0], raw[1]]) & 0x7f80 == 0x7f80,
            Element::F32 => {
                u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]) & 0x7f800000 == 0x7f800000
            }
        };
        if nonfinite {
            return Err(format!("nonfinite SSM check element {index}"));
        }
    }
    Ok(())
}

pub(super) fn compare_finite(
    candidate: &[u8],
    reference: &[u8],
    element: Element,
) -> Result<(), String> {
    if candidate.len() != reference.len() {
        return Err("SSM check candidate/reference byte lengths differ".into());
    }
    finite(candidate, element).map_err(|error| format!("candidate {error}"))?;
    finite(reference, element).map_err(|error| format!("reference {error}"))?;
    if let Some((byte, (&got, &expected))) = candidate
        .iter()
        .zip(reference)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        return Err(format!(
            "element {}, byte {byte}: candidate 0x{got:02x}, reference 0x{expected:02x}",
            byte / element.width()
        ));
    }
    Ok(())
}
