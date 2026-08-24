// SPDX-License-Identifier: AGPL-3.0-only

//! Loading and validation for residual-stream direction vectors.
//!
//! A direction vector is a behavioural modification shipped **separately from
//! the model**: `hidden_size` floats, a few hundred kilobytes, applied to the
//! residual stream at serve time. Applying it at runtime is arithmetically
//! identical to the rank-1 weight edit
//!
//! ```text
//! dW = -alpha * d_hat (d_hat^T W)
//! ```
//!
//! on every matrix that writes into the residual stream, so the same artifact
//! can be baked into weights, expressed as a rank-1 LoRA, or projected at
//! inference. The runtime form is preferred here because it leaves the base
//! checkpoint byte-identical: provenance claims against a pinned upstream
//! revision survive, and the modification is reverted by removing one file.
//!
//! This module owns loading and validation only. It deliberately does not
//! derive directions — deriving one requires contrasting activations over a
//! prompt set, which is a separate offline procedure whose inputs determine
//! what the direction *means*. The engine's job is to apply what it is given
//! and to refuse anything malformed.
//!
//! # Format
//!
//! JSON, so a direction is inspectable without special tooling:
//!
//! ```json
//! {
//!   "hidden_size": 5120,
//!   "layers": [1, 2, 3],
//!   "alpha": 1.0,
//!   "direction": [0.013, -0.004, ...]
//! }
//! ```
//!
//! `layers` selects which decoder layers the projection applies to; an empty
//! list means every layer. `alpha` scales the subtraction — 1.0 removes the
//! component entirely, 0.0 is a no-op.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

/// A residual-stream direction, validated and L2-normalised.
#[derive(Debug, Clone)]
pub struct DirectionVector {
    /// Unit-norm direction, `hidden_size` elements.
    pub d_hat: Vec<f32>,
    /// Projection strength. 1.0 removes the component entirely.
    pub alpha: f32,
    /// Decoder layers this applies to. Empty means all layers.
    pub layers: Vec<usize>,
}

#[derive(Deserialize)]
struct DirectionFile {
    hidden_size: usize,
    #[serde(default)]
    layers: Vec<usize>,
    #[serde(default = "default_alpha")]
    alpha: f32,
    direction: Vec<f32>,
}

fn default_alpha() -> f32 {
    1.0
}

impl DirectionVector {
    /// Load and validate a direction file, checking it against the model's
    /// hidden size.
    ///
    /// Fails closed on every malformed input rather than silently degrading:
    /// a direction that is the wrong length, non-finite, or numerically zero
    /// would otherwise apply a meaningless transformation to every token with
    /// no visible symptom.
    pub fn load(path: impl AsRef<Path>, model_hidden_size: usize) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading direction vector {}", path.display()))?;
        let file: DirectionFile = serde_json::from_str(&raw)
            .with_context(|| format!("parsing direction vector {}", path.display()))?;

        if file.hidden_size != model_hidden_size {
            bail!(
                "direction vector {} declares hidden_size={} but the model has {}. \
                 A direction is only meaningful for the model it was derived from.",
                path.display(),
                file.hidden_size,
                model_hidden_size
            );
        }
        if file.direction.len() != file.hidden_size {
            bail!(
                "direction vector {} declares hidden_size={} but carries {} elements",
                path.display(),
                file.hidden_size,
                file.direction.len()
            );
        }
        if let Some(bad) = file.direction.iter().position(|v| !v.is_finite()) {
            bail!(
                "direction vector {} has a non-finite value at index {}",
                path.display(),
                bad
            );
        }
        if !file.alpha.is_finite() {
            bail!("direction vector {} has non-finite alpha", path.display());
        }

        // Normalise here so the kernel never has to. An unnormalised direction
        // scales the subtraction by ||d||^2 rather than ||d||, which silently
        // over- or under-applies the projection.
        let norm = file.direction.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-6 {
            bail!(
                "direction vector {} has L2 norm {:.3e} — too small to normalise. \
                 A near-zero direction describes no direction at all.",
                path.display(),
                norm
            );
        }
        let d_hat = file.direction.iter().map(|v| v / norm).collect();

        Ok(Self {
            d_hat,
            alpha: file.alpha,
            layers: file.layers,
        })
    }

    /// Whether the projection applies at `layer_idx`.
    pub fn applies_to(&self, layer_idx: usize) -> bool {
        self.layers.is_empty() || self.layers.contains(&layer_idx)
    }

    /// Whether this direction changes anything. `alpha == 0` is a valid
    /// configuration meaning "loaded but inert", useful for A/B runs that keep
    /// the load path identical across arms.
    pub fn is_active(&self) -> bool {
        self.alpha != 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Writes `body` to a uniquely-named file under the temp dir and returns
    /// the path. Deliberately avoids a `tempfile` dependency for one test
    /// helper; the counter keeps parallel test threads from colliding.
    fn write_tmp(body: &str) -> std::path::PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!(
            "atlas_dirvec_{}_{}.json",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn loads_and_normalises() {
        // [3, 4] has norm 5, so d_hat is [0.6, 0.8].
        let f = write_tmp(r#"{"hidden_size":2,"direction":[3.0,4.0]}"#);
        let d = DirectionVector::load(&f, 2).unwrap();
        assert!((d.d_hat[0] - 0.6).abs() < 1e-6);
        assert!((d.d_hat[1] - 0.8).abs() < 1e-6);
        assert_eq!(d.alpha, 1.0, "alpha defaults to full projection");
        assert!(d.applies_to(0), "empty layer list means every layer");
        assert!(d.applies_to(63));
    }

    #[test]
    fn rejects_hidden_size_mismatch() {
        // The commonest way to apply a direction to the wrong model.
        let f = write_tmp(r#"{"hidden_size":2,"direction":[1.0,0.0]}"#);
        let err = DirectionVector::load(&f, 5120).unwrap_err().to_string();
        assert!(err.contains("5120"), "error names the model's size: {err}");
    }

    #[test]
    fn rejects_length_mismatch() {
        let f = write_tmp(r#"{"hidden_size":4,"direction":[1.0,0.0]}"#);
        assert!(DirectionVector::load(&f, 4).is_err());
    }

    #[test]
    fn rejects_zero_direction() {
        // Normalising this would divide by ~0 and produce NaNs across every
        // token, so it must fail at load rather than at inference.
        let f = write_tmp(r#"{"hidden_size":2,"direction":[0.0,0.0]}"#);
        let err = DirectionVector::load(&f, 2).unwrap_err().to_string();
        assert!(err.contains("norm"), "error explains why: {err}");
    }

    #[test]
    fn rejects_non_finite() {
        let f = write_tmp(r#"{"hidden_size":2,"direction":[1.0,null]}"#);
        assert!(
            DirectionVector::load(&f, 2).is_err(),
            "null parses as a JSON error or a non-finite value; either must fail"
        );
    }

    #[test]
    fn honours_layer_selection() {
        let f = write_tmp(r#"{"hidden_size":2,"layers":[5,7],"direction":[1.0,0.0]}"#);
        let d = DirectionVector::load(&f, 2).unwrap();
        assert!(d.applies_to(5));
        assert!(d.applies_to(7));
        assert!(!d.applies_to(6), "layers outside the list are untouched");
    }

    #[test]
    fn alpha_zero_loads_but_is_inert() {
        // Keeps the load path identical between A/B arms.
        let f = write_tmp(r#"{"hidden_size":2,"alpha":0.0,"direction":[1.0,0.0]}"#);
        let d = DirectionVector::load(&f, 2).unwrap();
        assert!(!d.is_active());
    }
}
