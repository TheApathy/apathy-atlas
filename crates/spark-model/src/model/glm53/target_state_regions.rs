// SPDX-License-Identifier: AGPL-3.0-only

//! Typed retained-state bindings with the actual enclosing allocation.
use super::*;

pub(super) struct Resolved {
    pub parent: GgmlIqBuffer,
    pub view: GgmlIqBuffer,
}

fn prefix(buffer: GgmlIqBuffer, bytes: usize) -> Result<GgmlIqBuffer> {
    ensure!(
        bytes <= buffer.bytes,
        "state probe live extent exceeds typed storage"
    );
    Ok(GgmlIqBuffer {
        ptr: buffer.ptr,
        bytes,
    })
}

pub(super) fn resolve(
    model: &Glm53Exl3Model,
    region: StateProbeRegion,
    position: u32,
) -> Result<Resolved> {
    let parent = GgmlIqBuffer {
        ptr: model.arena,
        bytes: usize::try_from(model.plan.known_bytes)?,
    };
    let position = usize::try_from(position)?;
    let view = match region {
        StateProbeRegion::Kda { ordinal } => model
            .kda_states
            .get(ordinal)
            .context("KDA ordinal outside model")?
            .persistent(),
        StateProbeRegion::Conv { ordinal } => {
            model
                .kda_conv
                .get(ordinal)
                .context("convolution ordinal outside model")?
                .persistent_state_f32
        }
        StateProbeRegion::Capture { tap, row } => model.captures.slot_row(tap, row)?,
        StateProbeRegion::ProjectedContext => {
            let guard = model
                .dflash2
                .lock()
                .map_err(|_| anyhow::anyhow!("state probe drafter owner poisoned"))?;
            let (parent, view) = guard
                .as_ref()
                .context("state probe missing drafter")?
                .state_probe_projected()?;
            return Ok(Resolved { parent, view });
        }
        StateProbeRegion::Logits { row } => {
            ensure!(row < 8, "state probe logit row outside verifier width");
            let row_bytes = (VOCAB as usize)
                .checked_mul(2)
                .context("logit row bytes overflow")?;
            let bytes = (model.scratch.max_wide_rows() as usize)
                .checked_mul(row_bytes)
                .context("logit allocation overflow")?;
            let offset = (row as usize)
                .checked_mul(row_bytes)
                .context("logit row offset overflow")?;
            let address = model
                .logits
                .0
                .checked_add(u64::try_from(offset)?)
                .context("logit row address overflow")?;
            return Ok(Resolved {
                parent: GgmlIqBuffer {
                    ptr: model.logits,
                    bytes,
                },
                view: GgmlIqBuffer {
                    ptr: DevicePtr(address),
                    bytes: row_bytes,
                },
            });
        }
        StateProbeRegion::DsaLatent { ordinal }
        | StateProbeRegion::DsaPoolKeys { ordinal }
        | StateProbeRegion::DsaPoolValidity { ordinal }
        | StateProbeRegion::DsaTailKeys { ordinal }
        | StateProbeRegion::DsaTailGates { ordinal }
        | StateProbeRegion::DsaTailValidity { ordinal } => {
            let cache = model
                .dsa_cache
                .get(ordinal)
                .context("DSA ordinal outside model")?;
            match region {
                StateProbeRegion::DsaLatent { .. } => prefix(
                    cache.latent_cache_bf16,
                    position
                        .checked_mul(512 * 2)
                        .context("latent prefix overflow")?,
                )?,
                StateProbeRegion::DsaPoolKeys { .. } => prefix(
                    cache.pool_keys_bf16,
                    (position / 4)
                        .checked_mul(128 * 2)
                        .context("pooled prefix overflow")?,
                )?,
                StateProbeRegion::DsaPoolValidity { .. } => {
                    prefix(cache.pool_validity_u8, position / 4)?
                }
                StateProbeRegion::DsaTailKeys { .. } => cache.prior_tail_keys_bf16,
                StateProbeRegion::DsaTailGates { .. } => cache.prior_tail_gates_bf16,
                StateProbeRegion::DsaTailValidity { .. } => cache.prior_tail_validity_u8,
                _ => unreachable!("DSA variant admitted by outer match"),
            }
        }
    };
    Ok(Resolved { parent, view })
}
