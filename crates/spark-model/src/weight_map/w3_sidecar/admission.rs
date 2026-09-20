// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};

const LAYERS_ENV: &str = "ATLAS_FFN_W3_LAYERS";
const PATH_ENV: &str = "ATLAS_FFN_W3_SIDECAR";
const SHA_ENV: &str = "ATLAS_FFN_W3_SIDECAR_SHA256";
const SIZE_ENV: &str = "ATLAS_FFN_W3_SIDECAR_SIZE";
const MAX_W3_LAYERS: usize = 64;

/// Immutable operator request for an attributable W3 sidecar load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct W3SidecarRequest {
    layers: BTreeSet<usize>,
    path: PathBuf,
    sha256: [u8; 32],
    size: u64,
}

impl W3SidecarRequest {
    /// Read the four-field request. Empty values are treated as absent, but a
    /// partially specified request is rejected rather than falling back to W4.
    pub fn from_env(total_layers: usize) -> Result<Option<Self>> {
        let values = [LAYERS_ENV, PATH_ENV, SHA_ENV, SIZE_ENV].map(std::env::var_os);
        Self::from_os_values(
            total_layers,
            values[0].as_deref(),
            values[1].as_deref(),
            values[2].as_deref(),
            values[3].as_deref(),
        )
    }

    pub(crate) fn from_os_values(
        total_layers: usize,
        layers: Option<&OsStr>,
        path: Option<&OsStr>,
        sha256: Option<&OsStr>,
        size: Option<&OsStr>,
    ) -> Result<Option<Self>> {
        let layers = utf8_field(LAYERS_ENV, layers)?;
        let path = utf8_field(PATH_ENV, path)?;
        let sha256 = utf8_field(SHA_ENV, sha256)?;
        let size = utf8_field(SIZE_ENV, size)?;
        Self::from_values(total_layers, layers, path, sha256, size)
    }

    pub(crate) fn from_values(
        total_layers: usize,
        layers: Option<&str>,
        path: Option<&str>,
        sha256: Option<&str>,
        size: Option<&str>,
    ) -> Result<Option<Self>> {
        let fields = [layers, path, sha256, size].map(|value| value.filter(|raw| !raw.is_empty()));
        if fields.iter().all(Option::is_none) {
            return Ok(None);
        }
        ensure!(
            (1..=MAX_W3_LAYERS).contains(&total_layers),
            "W3 request requires total_layers in 1..={MAX_W3_LAYERS}"
        );
        ensure!(
            fields.iter().all(Option::is_some),
            "W3 request requires {LAYERS_ENV}, {PATH_ENV}, {SHA_ENV}, and {SIZE_ENV} together"
        );

        let layers = parse_layers(fields[0].unwrap(), total_layers)?;
        let path = PathBuf::from(fields[1].unwrap());
        ensure!(path.is_absolute(), "{PATH_ENV} must be an absolute path");
        let sha256 = parse_sha256(fields[2].unwrap())?;
        let size = parse_size(fields[3].unwrap())?;
        Ok(Some(Self {
            layers,
            path,
            sha256,
            size,
        }))
    }

    pub fn layers(&self) -> &BTreeSet<usize> {
        &self.layers
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

fn utf8_field<'a>(name: &str, value: Option<&'a OsStr>) -> Result<Option<&'a str>> {
    value
        .map(|raw| {
            raw.to_str()
                .ok_or_else(|| anyhow::anyhow!("{name} must contain valid UTF-8"))
        })
        .transpose()
}

fn parse_layers(raw: &str, total_layers: usize) -> Result<BTreeSet<usize>> {
    ensure!(
        !raw.is_empty(),
        "{LAYERS_ENV} must not be empty when W3 is enabled"
    );
    let mut parsed = BTreeSet::new();
    for item in raw.split(',') {
        ensure!(!item.is_empty(), "{LAYERS_ENV} contains an empty item");
        let mut bounds = item.split('-');
        let first = canonical_usize(LAYERS_ENV, bounds.next().unwrap())?;
        let last = match bounds.next() {
            Some(end) => canonical_usize(LAYERS_ENV, end)?,
            None => first,
        };
        ensure!(
            bounds.next().is_none(),
            "{LAYERS_ENV} item has multiple hyphens"
        );
        ensure!(first <= last, "{LAYERS_ENV} range is descending");
        ensure!(
            last < total_layers,
            "{LAYERS_ENV} layer {last} is out of range"
        );
        for layer in first..=last {
            ensure!(parsed.insert(layer), "{LAYERS_ENV} repeats layer {layer}");
        }
    }
    ensure!(!parsed.is_empty(), "{LAYERS_ENV} selects no layers");
    Ok(parsed)
}

fn canonical_usize(name: &str, raw: &str) -> Result<usize> {
    ensure!(
        !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()),
        "{name} must use canonical decimal integers"
    );
    let value = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} integer overflow"))?;
    ensure!(value.to_string() == raw, "{name} integer is not canonical");
    Ok(value)
}

fn parse_sha256(raw: &str) -> Result<[u8; 32]> {
    ensure!(
        raw.len() == 64
            && raw
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{SHA_ENV} must be exactly 64 lowercase hexadecimal characters"
    );
    let mut digest = [0_u8; 32];
    for (index, pair) in raw.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).expect("validated hexadecimal is ASCII");
        digest[index] = u8::from_str_radix(text, 16).expect("validated hexadecimal parses");
    }
    Ok(digest)
}

fn parse_size(raw: &str) -> Result<u64> {
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{SIZE_ENV} must be a canonical positive decimal integer");
    }
    let size = raw
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("{SIZE_ENV} integer overflow"))?;
    ensure!(
        size > 0 && size.to_string() == raw,
        "{SIZE_ENV} must be a canonical positive decimal integer"
    );
    Ok(size)
}
