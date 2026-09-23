// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model launch profiles: the facts about a model that a recipe's `defaults:` flags
//! cannot express — how much memory it needs, whether it must run alone on the box, and the
//! process environment its loader reads.
//!
//! Read from the `atlas:` block of a built-in recipe (`recipe::builtin`), keyed by the
//! checkpoint's `model_type`, so the hot-swap path can look a profile up from nothing but the
//! `ServeArgs` it is handed.
//!
//! ## Why admission lives in the swap, AFTER the old model is released
//! GB10 memory is unified: an over-allocation takes the HOST down, not just this process. A
//! "run alone" model (DeepSeek-V4.1: ~84 GB of ~119.7) only fits once the outgoing model's
//! memory is back, so the check is made on MemAvailable between the drain and the load — the
//! only moment that number means what the check needs. A refusal there takes the swap's
//! existing restore path: the previous model is reloaded into the memory it just freed.

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::recipe::yaml::{self, Yaml};

/// What the launcher needs to know about one model beyond its flags.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelProfile {
    pub recipe_id: String,
    pub model_type: String,
    /// Host memory the model takes once loaded, in GB (unified memory on GB10).
    pub resident_gb: f64,
    /// Extra when its speculative drafter is enabled.
    pub dspark_extra_gb: f64,
    /// Memory that must remain free AFTER the model is resident, in GB.
    pub headroom_gb: f64,
    /// Refuse to load unless the box can give it resident + headroom after the previous model
    /// is released — i.e. it cannot share the machine.
    pub run_alone: bool,
    /// Process environment its loader reads (set before the load).
    pub env: BTreeMap<String, String>,
}

impl ModelProfile {
    /// Memory to require free before the load, in bytes. DSpark counts only when its env
    /// switch is on in the environment the load will see.
    pub fn required_bytes(&self, dspark_on: bool) -> u64 {
        let gb = self.resident_gb + self.headroom_gb + if dspark_on { self.dspark_extra_gb } else { 0.0 };
        (gb * 1e9) as u64
    }
}

/// Parse the `atlas:` block of one recipe YAML. `None` when the recipe has none.
pub(crate) fn parse_profile(recipe_id: &str, text: &str) -> Result<Option<ModelProfile>> {
    let doc = yaml::parse(text).with_context(|| format!("reading recipe {recipe_id}"))?;
    let Some(block) = doc.as_map().and_then(|m| m.get("atlas")).and_then(Yaml::as_map) else {
        return Ok(None);
    };
    let s = |key: &str| block.get(key).and_then(Yaml::as_str).map(str::to_string);
    let num = |key: &str| -> Result<f64> {
        match s(key) {
            Some(v) => v.parse().with_context(|| format!("{recipe_id}: atlas.{key} {v:?} is not a number")),
            None => Ok(0.0),
        }
    };
    let mut env = BTreeMap::new();
    if let Some(map) = block.get("env").and_then(Yaml::as_map) {
        for (k, v) in map {
            let v = v.as_str().with_context(|| format!("{recipe_id}: atlas.env.{k} is not a scalar"))?;
            env.insert(k.clone(), v.to_string());
        }
    }
    Ok(Some(ModelProfile {
        recipe_id: recipe_id.to_string(),
        model_type: s("model_type").with_context(|| format!("{recipe_id}: atlas.model_type is required"))?,
        resident_gb: num("resident_gb")?,
        dspark_extra_gb: num("dspark_extra_gb")?,
        headroom_gb: num("headroom_gb")?,
        run_alone: s("run_alone").as_deref() == Some("true"),
        env,
    }))
}

/// The built-in profile for the model being launched.
///
/// Matched by the checkpoint's directory name first (a built-in recipe's `model:`): several
/// served checkpoints share a `model_type` (Qwen3.5-27B, AEON-27B and Qwen3.8-27B are all
/// `qwen3_5`) but need different kernel targets and environments. Falls back to the
/// `model_type` match only when exactly one built-in declares that type.
pub(crate) fn profile_for(model: Option<&str>, model_type: &str) -> Option<ModelProfile> {
    let dir_name = model
        .map(|m| m.trim_end_matches('/'))
        .and_then(|m| std::path::Path::new(m).file_name())
        .and_then(|n| n.to_str());
    let with_model: Vec<(String, ModelProfile)> = crate::recipe::builtin::BUILTIN_YAML
        .iter()
        .filter_map(|(id, text)| {
            let recipe_model = crate::recipe::Recipe::parse(*id, text).ok()?.model;
            match parse_profile(id, text) {
                Ok(Some(p)) => Some((recipe_model, p)),
                Ok(None) => None,
                Err(e) => {
                    tracing::error!("built-in recipe {id}: atlas block does not parse: {e:#}");
                    None
                }
            }
        })
        .collect();
    if let Some(name) = dir_name
        && let Some((_, p)) = with_model.iter().find(|(m, _)| m == name)
    {
        return Some(p.clone());
    }
    let mut by_type = with_model.into_iter().filter(|(_, p)| p.model_type == model_type);
    match (by_type.next(), by_type.next()) {
        (Some((_, p)), None) => Some(p),
        _ => None,
    }
}

/// The built-in profile for a `model_type`, if exactly one built-in declares it.
pub(crate) fn profile_for_model_type(model_type: &str) -> Option<ModelProfile> {
    profile_for(None, model_type)
}

/// `MemAvailable` from `/proc/meminfo`, in bytes. On GB10 this is the pool the GPU
/// allocates from too, so it is the number an admission check has to read.
pub(crate) fn mem_available_bytes() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    parse_mem_available(&text)
}

/// `MemTotal` from `/proc/meminfo`, in bytes.
pub(crate) fn mem_total_bytes() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    parse_meminfo_field(&text, "MemTotal:")
}

pub(crate) fn parse_mem_available(meminfo: &str) -> Result<u64> {
    parse_meminfo_field(meminfo, "MemAvailable:")
}

fn parse_meminfo_field(meminfo: &str, field: &str) -> Result<u64> {
    let line = meminfo
        .lines()
        .find(|l| l.starts_with(field))
        .with_context(|| format!("/proc/meminfo has no {field} line"))?;
    let kb: u64 = line
        .split_whitespace()
        .nth(1)
        .with_context(|| format!("{field} line has no value"))?
        .parse()
        .with_context(|| format!("{field} value is not a number"))?;
    Ok(kb * 1024)
}

/// The run-alone admission rule, pure so it is testable: refuse unless `available` covers the
/// profile's requirement. Non-run-alone profiles are admitted (their memory is the load's own
/// business, as before).
pub(crate) fn admit(profile: &ModelProfile, available: u64, dspark_on: bool) -> Result<()> {
    if !profile.run_alone {
        return Ok(());
    }
    let need = profile.required_bytes(dspark_on);
    anyhow::ensure!(
        available >= need,
        "{} ({}) must run alone: it needs {:.1} GB free ({:.0} GB resident{} + {:.0} GB headroom) \
         but only {:.1} GB is available after the previous model was released. GB10 memory is \
         unified — loading anyway would risk taking the host down. Stop other GPU work on this \
         box and retry.",
        profile.recipe_id,
        profile.model_type,
        need as f64 / 1e9,
        profile.resident_gb,
        if dspark_on { format!(" + {:.0} GB DSpark", profile.dspark_extra_gb) } else { String::new() },
        profile.headroom_gb,
        available as f64 / 1e9,
    );
    Ok(())
}

/// Set the profile's environment for the upcoming load, leaving any variable the operator
/// already set untouched (an explicit export beats a recipe default). Returns what it set.
///
/// SAFETY of `set_var`: called only from the swap, which holds `ModelHost::swap_guard` and has
/// drained and joined the scheduler, so no model code is reading the environment
/// concurrently. Other threads (the UI, tokio workers) do not read these variables.
pub(crate) fn apply_env(profile: &ModelProfile) -> Vec<(String, String)> {
    let mut set = Vec::new();
    for (k, v) in &profile.env {
        if std::env::var_os(k).is_none() {
            // SAFETY: see the function doc — serialized by the swap guard, after the drain.
            unsafe { std::env::set_var(k, v) };
            set.push((k.clone(), v.clone()));
        }
    }
    set
}

#[cfg(test)]
#[path = "model_profile_tests.rs"]
mod tests;
