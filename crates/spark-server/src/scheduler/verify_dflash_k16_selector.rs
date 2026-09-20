// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed environment selector for the K16 acceptance fixture.

use anyhow::{Result, bail, ensure};

const ENV: &str = "ATLAS_DFLASH_K16_ACCEPTANCE_FIXTURE";
const SEQ_LEN_ENV: &str = "ATLAS_DFLASH_K16_ACCEPTANCE_FIXTURE_SEQ_LEN";
const ACCEPTED_ENV: &str = "ATLAS_DFLASH_K16_ACCEPTANCE_FIXTURE_ACCEPTED_DRAFTS";
const K16: usize = 16;
const GATE_ENVS: [&str; 6] = [
    "ATLAS_QWEN4_K16_BATCHED_VERIFY",
    "ATLAS_QWEN4_K16_EXACT",
    "ATLAS_DFLASH_SERIAL_COMMIT",
    "ATLAS_DFLASH_SKIP_REPROPOSE",
    "ATLAS_DFLASH_K1_STAGE_DIAG",
    "ATLAS_DFLASH_K16_COMMIT_PARITY",
];

#[derive(Debug, Default)]
struct Values {
    enabled: Option<String>,
    seq_len: Option<String>,
    accepted: Option<String>,
    gates: [Option<String>; 6],
    stage_seq_len: Option<String>,
    stage_tokens: Option<String>,
    commit_seq_len: Option<String>,
    commit_tokens: Option<String>,
}

#[derive(Debug)]
pub(super) struct Selector {
    pub(super) accepted: usize,
    pub(super) expected_inputs: Vec<u32>,
}

fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8"),
    }
}

fn read_enabled_values(
    enabled: Option<String>,
    seq_len: Option<String>,
    accepted: Option<String>,
) -> Result<Values> {
    Ok(Values {
        enabled,
        seq_len,
        accepted,
        gates: [
            optional_env(GATE_ENVS[0])?,
            optional_env(GATE_ENVS[1])?,
            optional_env(GATE_ENVS[2])?,
            optional_env(GATE_ENVS[3])?,
            optional_env(GATE_ENVS[4])?,
            optional_env(GATE_ENVS[5])?,
        ],
        stage_seq_len: optional_env("ATLAS_DFLASH_K1_STAGE_SEQ_LEN")?,
        stage_tokens: optional_env("ATLAS_DFLASH_K1_STAGE_TOKENS")?,
        commit_seq_len: optional_env("ATLAS_DFLASH_K16_COMMIT_PARITY_SEQ_LEN")?,
        commit_tokens: optional_env("ATLAS_DFLASH_K16_COMMIT_PARITY_TOKENS")?,
    })
}

fn canonical_usize(name: &str, raw: Option<&str>) -> Result<usize> {
    let raw = raw.ok_or_else(|| anyhow::anyhow!("{ENV}=1 requires {name}"))?;
    let value = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} must be a canonical decimal usize"))?;
    ensure!(value.to_string() == raw, "{name} must be canonical decimal");
    Ok(value)
}

fn canonical_tokens(name: &str, raw: Option<&str>) -> Result<Vec<u32>> {
    let raw = raw.ok_or_else(|| anyhow::anyhow!("{ENV}=1 requires {name}"))?;
    ensure!(!raw.is_empty(), "{name} must be a nonempty canonical CSV");
    raw.split(',')
        .map(|part| {
            let token = part
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("{name} must be a canonical u32 CSV"))?;
            ensure!(token.to_string() == part, "{name} must be canonical");
            Ok(token)
        })
        .collect()
}

fn select(values: &Values, pre_verify_len: usize, consumed: bool) -> Result<Option<Selector>> {
    match values.enabled.as_deref() {
        None | Some("0") => {
            ensure!(
                values.seq_len.is_none() && values.accepted.is_none(),
                "{SEQ_LEN_ENV}/{ACCEPTED_ENV} require {ENV}=1"
            );
            Ok(None)
        }
        Some("1") => {
            for (name, value) in GATE_ENVS.into_iter().zip(&values.gates) {
                ensure!(value.as_deref() == Some("1"), "{ENV}=1 requires {name}=1");
            }
            let selected_seq_len = canonical_usize(SEQ_LEN_ENV, values.seq_len.as_deref())?;
            let accepted = canonical_usize(ACCEPTED_ENV, values.accepted.as_deref())?;
            ensure!(accepted < K16, "{ACCEPTED_ENV} must be in 0..=15");
            let stage_seq = canonical_usize(
                "ATLAS_DFLASH_K1_STAGE_SEQ_LEN",
                values.stage_seq_len.as_deref(),
            )?;
            let commit_seq = canonical_usize(
                "ATLAS_DFLASH_K16_COMMIT_PARITY_SEQ_LEN",
                values.commit_seq_len.as_deref(),
            )?;
            ensure!(
                selected_seq_len == stage_seq && stage_seq == commit_seq,
                "K16 fixture/stage/commit sequence selectors must match"
            );
            let stage_tokens = canonical_tokens(
                "ATLAS_DFLASH_K1_STAGE_TOKENS",
                values.stage_tokens.as_deref(),
            )?;
            let commit_tokens = canonical_tokens(
                "ATLAS_DFLASH_K16_COMMIT_PARITY_TOKENS",
                values.commit_tokens.as_deref(),
            )?;
            ensure!(
                stage_tokens.len() == K16,
                "K16 fixture requires exactly 16 inputs"
            );
            ensure!(
                stage_tokens == commit_tokens,
                "stage/commit token selectors must match"
            );
            Ok(
                (!consumed && selected_seq_len == pre_verify_len).then_some(Selector {
                    accepted,
                    expected_inputs: stage_tokens,
                }),
            )
        }
        Some(other) => bail!("{ENV} must be exactly 0 or 1, got {other:?}"),
    }
}

pub(super) fn select_from_env(pre_verify_len: usize, consumed: bool) -> Result<Option<Selector>> {
    let enabled = optional_env(ENV)?;
    let seq_len = optional_env(SEQ_LEN_ENV)?;
    let accepted = optional_env(ACCEPTED_ENV)?;
    if matches!(enabled.as_deref(), None | Some("0")) {
        return select(
            &Values {
                enabled,
                seq_len,
                accepted,
                ..Values::default()
            },
            pre_verify_len,
            consumed,
        );
    }
    select(
        &read_enabled_values(enabled, seq_len, accepted)?,
        pre_verify_len,
        consumed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> Values {
        Values {
            enabled: Some("1".into()),
            seq_len: Some("38".into()),
            accepted: Some("7".into()),
            gates: std::array::from_fn(|_| Some("1".into())),
            stage_seq_len: Some("38".into()),
            stage_tokens: Some((0..16).map(|v| v.to_string()).collect::<Vec<_>>().join(",")),
            commit_seq_len: Some("38".into()),
            commit_tokens: Some((0..16).map(|v| v.to_string()).collect::<Vec<_>>().join(",")),
        }
    }

    #[test]
    fn default_off_and_exact_selector() {
        assert!(select(&Values::default(), 38, false).unwrap().is_none());
        let explicit_off = Values {
            enabled: Some("0".into()),
            ..Default::default()
        };
        assert!(select(&explicit_off, 38, false).unwrap().is_none());
        let mut values = enabled();
        assert_eq!(select(&values, 38, false).unwrap().unwrap().accepted, 7);
        assert!(select(&values, 39, false).unwrap().is_none());
        assert!(select(&values, 38, true).unwrap().is_none());
        for invalid in ["", "2", "true", "01"] {
            values.enabled = Some(invalid.into());
            assert!(select(&values, 38, false).is_err());
        }
    }

    #[test]
    fn partial_and_hostile_contracts_rejected() {
        for disabled_value in ["", "0"] {
            let disabled = Values {
                seq_len: Some(disabled_value.into()),
                ..Default::default()
            };
            assert!(select(&disabled, 38, false).is_err());
        }
        for gate in 0..GATE_ENVS.len() {
            let mut values = enabled();
            values.gates[gate] = Some("0".into());
            assert!(select(&values, 38, false).is_err(), "gate {gate} admitted");
        }
        for mutate in 0..6 {
            let mut values = enabled();
            match mutate {
                0 => values.gates[1] = None,
                1 => values.accepted = Some("16".into()),
                2 => values.accepted = Some("07".into()),
                3 => values.stage_seq_len = Some("39".into()),
                4 => values.commit_tokens = Some("0,1".into()),
                5 => values.stage_tokens = Some("0,01,2".into()),
                _ => unreachable!(),
            }
            assert!(
                select(&values, 38, false).is_err(),
                "mutation {mutate} admitted"
            );
        }
    }
}
