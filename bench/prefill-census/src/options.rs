// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
pub(super) struct Options {
    pub(super) port: u16,
    pub(super) model: String,
    pub(super) mode: String,
    pub(super) output: PathBuf,
    pub(super) provenance: PathBuf,
    pub(super) timeout: u64,
    pub(super) bins: Vec<usize>,
    pub(super) prompt_format: String,
}
impl Options {
    pub(super) fn parse() -> Result<Self> {
        let mut values = BTreeMap::new();
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            check(
                [
                    "--port",
                    "--model",
                    "--ids-mode",
                    "--output-dir",
                    "--provenance",
                    "--timeout-seconds",
                    "--bins",
                    "--prompt-format",
                ]
                .contains(&key.as_str()),
                "unknown option; see README.md",
            )?;
            let value = args.next().ok_or("missing option value")?;
            check(values.insert(key, value).is_none(), "duplicate option")?;
        }
        let get = |k: &str| {
            values
                .get(k)
                .cloned()
                .ok_or_else(|| format!("required option {k}"))
        };
        let port = get("--port")?.parse::<u16>().map_err(|_| "invalid port")?;
        let model = get("--model")?;
        let mode = get("--ids-mode")?;
        let timeout = values
            .get("--timeout-seconds")
            .map_or(Ok(600), |v| v.parse::<u64>().map_err(|_| "invalid timeout"))?;
        check(
            port > 0 && !model.is_empty() && (1..=1800).contains(&timeout),
            "port/model/timeout out of bounds",
        )?;
        check(
            ["prompt-array", "prompt_token_ids"].contains(&mode.as_str()),
            "invalid --ids-mode",
        )?;
        Ok(Self {
            port,
            model,
            mode,
            output: get("--output-dir")?.into(),
            provenance: get("--provenance")?.into(),
            timeout,
            bins: parse_bins(values.get("--bins").map(String::as_str))?,
            prompt_format: parse_prompt_format(values.get("--prompt-format").map(String::as_str))?,
        })
    }
}
pub(super) fn parse_bins(raw: Option<&str>) -> Result<Vec<usize>> {
    let bins = raw
        .unwrap_or("256,2048,8192")
        .split(',')
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "--bins requires comma-separated integer lengths".to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    check(
        !bins.is_empty()
            && bins.iter().all(|n| [256, 2048, 8192].contains(n))
            && bins.windows(2).all(|pair| pair[0] < pair[1]),
        "--bins must be a unique ascending nonempty subset of 256,2048,8192",
    )?;
    Ok(bins)
}
