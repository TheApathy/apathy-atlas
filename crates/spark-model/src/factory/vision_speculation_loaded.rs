// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    DflashBuildArgs,
    vision_speculation::{Qualification, Request},
};
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

pub(super) fn validate(
    config: &ModelConfig,
    target: &WeightStore,
    drafter: Option<&DflashBuildArgs<'_>>,
    legacy_spec: bool,
    high_speed_swap: bool,
    max_batch_tokens: usize,
    has_adapters: bool,
) -> Result<bool> {
    let policy = Qualification::from_env().map_err(anyhow::Error::msg)?;
    if !policy.enabled() || (drafter.is_none() && !legacy_spec) {
        return Ok(false);
    }
    let native = drafter.is_some_and(|args| args.drafter_config.is_none());
    policy
        .admit(Request {
            is_vision: config.deepseek_vision.is_some(),
            target_only: false,
            dflash_only: native && !legacy_spec,
            embedded_drafter: native,
            high_speed_swap,
            verify_capacity: max_batch_tokens,
            has_adapters,
        })
        .map_err(anyhow::Error::msg)?;
    policy.validate_runtime().map_err(anyhow::Error::msg)?;
    let args = drafter.expect("admission requires a native drafter");
    ensure!(
        args.dspark_expert_subset.is_none(),
        "Vision DSpark requires every embedded expert"
    );
    ensure!(
        args.gamma == Some(6),
        "Vision DSpark qualification requires --dflash-gamma 6"
    );
    let draft = args.drafter_store;
    ensure!(
        draft.contains("mtp.0.main_proj.weight"),
        "missing embedded DSpark main projection"
    );
    // Every admitted tensor must be the target store's own unmodified view.
    for name in draft.names() {
        ensure!(
            name.starts_with("mtp."),
            "unexpected non-DSpark draft tensor {name}"
        );
        let a = draft.get(name)?;
        let b = target.get(name)?;
        ensure!(
            a.ptr == b.ptr && !a.ptr.is_null() && a.shape == b.shape && a.dtype == b.dtype,
            "Vision DSpark must alias the matching target tensor {name}"
        );
    }
    for name in target.names().filter(|name| name.starts_with("mtp.")) {
        ensure!(
            draft.contains(name),
            "embedded DSpark tensor omitted: {name}"
        );
    }
    tracing::warn!(
        "Native DeepSeek Vision DSpark qualification ARMED; experimental, C1 eager, not production-qualified"
    );
    Ok(true)
}
