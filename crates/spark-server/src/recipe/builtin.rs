// SPDX-License-Identifier: AGPL-3.0-only

//! Recipes shipped INSIDE the binary, for models the public index does not carry yet.
//!
//! The Library lists a checkpoint only as well as it knows a recipe for it; a model with no
//! recipe gets a "starting point" guessed from a donor, and for DeepSeek-V4.1 any donor guess
//! is wrong (its memory needs, its expert-pack env and its run-alone admission exist nowhere
//! else). So the measured configuration ships here, in the same YAML format as the index, and
//! [`with_builtins`] adds it beneath the fetched recipes — a published recipe with the same id
//! wins, so this steps aside the moment upstream carries one.
//!
//! The YAML may carry an `atlas:` block (model_type, memory, run-alone, env) that
//! [`Recipe::parse`] ignores and `main_modules::model_profile` reads.

use super::Recipe;

/// `(id, yaml)` for every built-in recipe. The id is `family/stem`, like the index's.
pub const BUILTIN_YAML: &[(&str, &str)] = &[(
    "deepseek/deepseek-v4.1-flash-next",
    include_str!("builtin/deepseek-v4.1-flash-next.yaml"),
)];

/// Every built-in recipe, parsed. A built-in that fails to parse is a build defect and is
/// caught by the tests below, so it is skipped with an error log rather than taking the
/// Library down.
pub fn recipes() -> Vec<Recipe> {
    BUILTIN_YAML
        .iter()
        .filter_map(|(id, text)| match Recipe::parse(*id, text) {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::error!("built-in recipe {id} does not parse: {e:#}");
                None
            }
        })
        .collect()
}

/// `fetched` plus every built-in whose id it does not already carry.
pub fn with_builtins(fetched: &[Recipe]) -> Vec<Recipe> {
    with_builtins_for(fetched, |_| true)
}

/// [`with_builtins`], restricted to built-ins whose `model` passes `present` — the Library
/// adds one only when its checkpoint is on disk.
pub fn with_builtins_for(fetched: &[Recipe], present: impl Fn(&str) -> bool) -> Vec<Recipe> {
    let mut all = fetched.to_vec();
    for b in recipes() {
        if present(&b.model) && !all.iter().any(|r| r.id == b.id) {
            all.push(b);
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_parses_and_is_atlas() {
        let all = recipes();
        assert_eq!(all.len(), BUILTIN_YAML.len(), "a built-in recipe failed to parse");
        for r in &all {
            assert!(r.is_atlas(), "{} must be runtime: atlas", r.id);
        }
    }

    #[test]
    fn deepseek_v41_builds_a_valid_serve_command() {
        let r = recipes().into_iter().find(|r| r.id == "deepseek/deepseek-v4.1-flash-next").unwrap();
        assert_eq!(r.model, "DeepSeek-V4.1-Flash-Next-DGX-Spark-512K");
        let args = r.serve_args(&Default::default()).expect("argv must pass clap and validation");
        assert_eq!(args.max_seq_len, 8192);
        assert_eq!(args.max_batch_size, 1);
    }

    /// A published recipe with the same id wins; a different id is additive.
    #[test]
    fn a_fetched_recipe_with_the_same_id_replaces_the_builtin() {
        let mut fetched = recipes();
        fetched[0].description = "published upstream".into();
        let merged = with_builtins(&fetched);
        assert_eq!(merged.len(), fetched.len());
        assert_eq!(merged[0].description, "published upstream");
        // Control: with nothing fetched, the built-ins appear.
        assert_eq!(with_builtins(&[]).len(), BUILTIN_YAML.len());
        // And only for checkpoints that are present.
        assert!(with_builtins_for(&[], |_| false).is_empty());
    }
}
