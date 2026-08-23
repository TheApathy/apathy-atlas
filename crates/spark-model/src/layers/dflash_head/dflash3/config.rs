// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash3 startup configuration (guide §7).
//!
//! One external gate — `ATLAS_DFLASH3` — plus an optional TOML file pointed at
//! by `ATLAS_DFLASH3_CONFIG`. Phase 1 only ever produces `Off` or `Shadow`
//! (the scaffolding is behavior-preserving); `Flat`/`Tree` are parsed so the
//! config surface stays stable but must not be selected until later phases
//! gate them on the exact-output proofs.

use std::path::PathBuf;

/// DFlash3 operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dflash3Mode {
    /// Feature disabled; the legacy flat path is the only publisher.
    Off,
    /// Run the facade side-by-side with the legacy path, log, never publish.
    Shadow,
    /// Publish through the facade (Phase 2+, flat-only).
    Flat,
    /// Publish tree/forest plans (Phase 4+, correctness-gated).
    Tree,
}

impl Dflash3Mode {
    /// Parse a mode from env/TOML text. `""`/`"0"`/`"off"`/`"false"` are
    /// `Off`; `"1"`/`"on"`/`"true"`/`"shadow"` are `Shadow` (the safe active
    /// default). Unrecognized input returns `None`.
    pub fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => Some(Self::Off),
            "1" | "on" | "shadow" | "true" | "yes" => Some(Self::Shadow),
            "flat" => Some(Self::Flat),
            "tree" => Some(Self::Tree),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Flat => "flat",
            Self::Tree => "tree",
        }
    }

    /// Modes that may (eventually) alter proposal behavior.
    pub fn is_active(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Resolved DFlash3 startup configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dflash3Config {
    pub schema: u32,
    pub mode: Dflash3Mode,
    pub config_path: Option<PathBuf>,
    pub max_flat_width: usize,
    pub max_tree_nodes: usize,
    /// Source names requested in config order. Unvalidated in Phase 1 (the
    /// full TOML `sources = [...]` array parse is deferred with the `toml`
    /// dependency).
    pub sources: Vec<String>,
}

impl Default for Dflash3Config {
    fn default() -> Self {
        Self {
            schema: 1,
            mode: Dflash3Mode::Off,
            config_path: None,
            max_flat_width: 12,
            max_tree_nodes: 31,
            sources: Vec::new(),
        }
    }
}

impl Dflash3Config {
    /// Resolve from the environment. Never panics on malformed input — an
    /// unrecognized mode falls back to `Off` and is surfaced in
    /// [`Dflash3Config::attestation`].
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        let mode_raw = std::env::var("ATLAS_DFLASH3").unwrap_or_default();
        if !mode_raw.is_empty() {
            match Dflash3Mode::parse(&mode_raw) {
                Some(mode) => cfg.mode = mode,
                None => {
                    cfg.mode = Dflash3Mode::Off;
                    tracing::warn!(
                        "ATLAS_DFLASH3={mode_raw:?} is not a recognized mode; DFlash3 disabled"
                    );
                }
            }
        }
        if let Ok(path) = std::env::var("ATLAS_DFLASH3_CONFIG")
            && !path.is_empty()
        {
            cfg.config_path = Some(PathBuf::from(&path));
            cfg.apply_toml_file(&path);
        }
        cfg
    }

    /// Minimal TOML extraction. Phase 1 reads only `mode`, `max_flat_width`,
    /// and `max_tree_nodes` via a line scan; a full `toml` dependency is
    /// deferred until the facade can affect serving.
    fn apply_toml_file(&mut self, path: &str) {
        let Ok(text) = std::fs::read_to_string(path) else {
            tracing::warn!("ATLAS_DFLASH3_CONFIG={path:?} could not be read; ignoring");
            return;
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim().trim_matches('"');
            match key {
                "mode" => {
                    if let Some(mode) = Dflash3Mode::parse(value) {
                        self.mode = mode;
                    }
                }
                "max_flat_width" => {
                    if let Ok(w) = value.parse::<usize>() {
                        self.max_flat_width = w;
                    }
                }
                "max_tree_nodes" => {
                    if let Ok(n) = value.parse::<usize>() {
                        self.max_tree_nodes = n;
                    }
                }
                _ => {}
            }
        }
    }

    /// Canonical one-line attestation. Log once at startup so a run's DFlash3
    /// disposition is unambiguous in the benchmark receipt.
    pub fn attestation(&self) -> String {
        format!(
            "DFlash3 mode={} schema={} max_flat_width={} max_tree_nodes={} config={} sources={:?}",
            self.mode.as_str(),
            self.schema,
            self.max_flat_width,
            self.max_tree_nodes,
            self.config_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "env-only".to_string()),
            self.sources,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse_round_trips_every_mode() {
        for mode in [
            Dflash3Mode::Off,
            Dflash3Mode::Shadow,
            Dflash3Mode::Flat,
            Dflash3Mode::Tree,
        ] {
            assert_eq!(Dflash3Mode::parse(mode.as_str()), Some(mode));
        }
    }

    #[test]
    fn mode_parse_accepts_env_style_aliases() {
        assert_eq!(Dflash3Mode::parse(""), Some(Dflash3Mode::Off));
        assert_eq!(Dflash3Mode::parse("0"), Some(Dflash3Mode::Off));
        assert_eq!(Dflash3Mode::parse("off"), Some(Dflash3Mode::Off));
        assert_eq!(Dflash3Mode::parse("1"), Some(Dflash3Mode::Shadow));
        assert_eq!(Dflash3Mode::parse("on"), Some(Dflash3Mode::Shadow));
        assert_eq!(Dflash3Mode::parse("SHADOW"), Some(Dflash3Mode::Shadow));
        assert_eq!(Dflash3Mode::parse("bogus"), None);
    }

    #[test]
    fn default_is_off_and_inactive() {
        let cfg = Dflash3Config::default();
        assert_eq!(cfg.mode, Dflash3Mode::Off);
        assert!(!cfg.mode.is_active());
        assert_eq!(cfg.schema, 1);
    }

    #[test]
    fn attestation_names_the_mode() {
        let cfg = Dflash3Config {
            mode: Dflash3Mode::Shadow,
            ..Dflash3Config::default()
        };
        assert!(cfg.attestation().contains("mode=shadow"));
        assert!(cfg.attestation().contains("schema=1"));
    }
}
