// SPDX-License-Identifier: AGPL-3.0-only

//! Pure admission and scratch layout for the isolated grouped-MoE experiment.

pub(crate) fn parse_selector(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("selector must be absent, 0, or 1"),
    }
}

pub(crate) fn validate_conflict(value: Option<&str>, tile: bool) -> Result<(), &'static str> {
    match (value, tile) {
        (None, _) | (Some("0"), false) | (Some("1"), true) => Ok(()),
        _ => Err("competing prefill selector must be disabled for MoE-only batching"),
    }
}

pub(crate) fn validate_ple_composition(
    ple_value: Option<&str>,
    composition_value: Option<&str>,
) -> Result<(), &'static str> {
    let ple = parse_selector(ple_value)?;
    let composition = parse_selector(composition_value)?;
    if ple == composition {
        Ok(())
    } else {
        Err("whole-prompt PLE with F12 requires both composition selectors to be exact value 1")
    }
}

pub(crate) fn validate_exact_qkv16_composition(
    value: Option<&str>,
    k16_value: Option<&str>,
    f12_selected: bool,
    exact_hyper_selected: bool,
    start: usize,
    rows: usize,
) -> Result<(), &'static str> {
    if !parse_selector(value)? {
        return Ok(());
    }
    if !parse_selector(k16_value)? || !f12_selected || !exact_hyper_selected {
        return Err("exact QKV16 prefill requires F12, exact HC, and exact K16 selectors");
    }
    if start != 0 || !(16..=2048).contains(&rows) {
        return Err("exact QKV16 prefill requires the initial 16..2048-token window");
    }
    Ok(())
}

pub(crate) fn validate_exact_o16_composition(
    value: Option<&str>,
    exact_qkv16_selected: bool,
) -> Result<(), &'static str> {
    if parse_selector(value)? && !exact_qkv16_selected {
        Err("exact O16 prefill requires the exact QKV16 route")
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub capacity: usize,
    pub qkv: usize,
    pub norm: usize,
    pub hidden: usize,
    pub residual: usize,
    pub output: usize,
}

#[derive(Debug)]
pub(crate) struct Plan {
    pub row_bytes: usize,
    pub core_bytes: usize,
    pub staging_offset: usize,
    pub input_bytes: usize,
}

impl Plan {
    pub(crate) fn new(
        rows: usize,
        seq_len_start: usize,
        hidden: usize,
        residual_width: usize,
        limits: Limits,
    ) -> Result<Self, &'static str> {
        if !(2..=2048).contains(&rows) || rows > limits.capacity {
            return Err("MoE-only prefill requires 2..2048 rows within arena capacity");
        }
        if seq_len_start.checked_add(rows).is_none_or(|end| end > 2048) {
            return Err("MoE-only prefill is restricted to the initial 2048-token dense window");
        }
        if hidden == 0 || residual_width == 0 {
            return Err("MoE-only prefill requires nonzero row widths");
        }
        let row_bytes = residual_width.checked_mul(2).ok_or("row size overflow")?;
        let core_bytes = hidden.checked_mul(2).ok_or("core size overflow")?;
        let input_bytes = rows.checked_mul(core_bytes).ok_or("input size overflow")?;
        let hidden_bytes = rows.checked_mul(row_bytes).ok_or("hidden size overflow")?;
        let end = row_bytes
            .checked_add(input_bytes)
            .ok_or("staging size overflow")?;
        if end > limits.qkv || input_bytes > limits.norm || input_bytes > limits.output {
            return Err("MoE-only staging exceeds QKV, norm, or MoE output arena");
        }
        if hidden_bytes > limits.hidden || hidden_bytes > limits.residual {
            return Err("MoE-only rows exceed hidden or residual arena");
        }
        Ok(Self {
            row_bytes,
            core_bytes,
            staging_offset: row_bytes,
            input_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_exact_o16_composition, validate_exact_qkv16_composition, validate_ple_composition,
    };

    #[test]
    fn o16_requires_the_exact_qkv16_route() {
        assert!(validate_exact_o16_composition(None, false).is_ok());
        assert!(validate_exact_o16_composition(Some("0"), true).is_ok());
        assert!(validate_exact_o16_composition(Some("1"), true).is_ok());
        for case in [(Some("1"), false), (Some("true"), true)] {
            assert!(validate_exact_o16_composition(case.0, case.1).is_err());
        }
    }

    #[test]
    fn qkv16_requires_all_exact_selectors_and_initial_window() {
        assert!(validate_exact_qkv16_composition(None, None, false, false, 0, 2048).is_ok());
        assert!(validate_exact_qkv16_composition(Some("0"), None, true, true, 17, 32).is_ok());
        assert!(
            validate_exact_qkv16_composition(Some("1"), Some("1"), true, true, 0, 2048).is_ok()
        );

        for case in [
            (Some("1"), None, true, true, 0, 2048),
            (Some("1"), Some("0"), true, true, 0, 2048),
            (Some("1"), Some("1"), false, true, 0, 2048),
            (Some("1"), Some("1"), true, false, 0, 2048),
            (Some("1"), Some("1"), true, true, 1, 2047),
            (Some("1"), Some("1"), true, true, 0, 15),
            (Some("1"), Some("1"), true, true, 0, 2049),
            (Some("true"), Some("1"), true, true, 0, 2048),
        ] {
            assert!(
                validate_exact_qkv16_composition(case.0, case.1, case.2, case.3, case.4, case.5)
                    .is_err()
            );
        }
    }

    #[test]
    fn ple_composition_requires_two_exact_opt_ins() {
        assert!(validate_ple_composition(None, None).is_ok());
        assert!(validate_ple_composition(Some("0"), Some("0")).is_ok());
        assert!(validate_ple_composition(Some("1"), Some("1")).is_ok());

        for (ple, composition) in [
            (Some("1"), None),
            (Some("1"), Some("0")),
            (None, Some("1")),
            (Some("0"), Some("1")),
            (Some("true"), Some("1")),
            (Some("1"), Some("true")),
        ] {
            assert!(validate_ple_composition(ple, composition).is_err());
        }
    }
}
