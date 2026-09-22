// SPDX-License-Identifier: AGPL-3.0-only

//! Pure bounded BF16 admission/comparison for the diagnostic replay.

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Difference {
    Extent,
    NonFinite {
        element: usize,
        bits: u16,
    },
    Mismatch {
        element: usize,
        compact: u16,
        shipping: u16,
    },
}

pub(super) fn admit(check: bool, compact: bool, f8: bool) -> Result<(), &'static str> {
    if check && !(compact && f8) {
        Err("compact CHECK requires compact and F8")
    } else {
        Ok(())
    }
}

pub(super) fn extent(rows: usize, columns: usize) -> Result<usize, &'static str> {
    if !(2..=2048).contains(&rows) || ![640, 2560].contains(&columns) {
        return Err("compact CHECK requires 2..2048 rows and canonical projection width");
    }
    Ok(rows * 10 * columns * 2)
}

pub(super) fn finite(bytes: &[u8]) -> Result<(), Difference> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(2) {
        return Err(Difference::Extent);
    }
    for (element, pair) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7f80 == 0x7f80 {
            return Err(Difference::NonFinite { element, bits });
        }
    }
    Ok(())
}

pub(super) fn exact(compact: &[u8], shipping: &[u8]) -> Result<(), Difference> {
    if compact.len() != shipping.len() {
        return Err(Difference::Extent);
    }
    finite(compact)?;
    finite(shipping)?;
    for (element, (a, b)) in compact
        .chunks_exact(2)
        .zip(shipping.chunks_exact(2))
        .enumerate()
    {
        if a != b {
            return Err(Difference::Mismatch {
                element,
                compact: u16::from_le_bytes([a[0], a[1]]),
                shipping: u16::from_le_bytes([b[0], b[1]]),
            });
        }
    }
    Ok(())
}
