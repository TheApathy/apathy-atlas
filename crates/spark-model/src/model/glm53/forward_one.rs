// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only validation of the exact one-token GLM-5.3 target event stream.

use anyhow::{Result, ensure};

use crate::layers::{
    Glm53TargetAttentionKind, Glm53TargetEvent, Glm53TargetFfnKind, Glm53TargetGeometry,
    Glm53TargetSchedule,
};

pub const GLM53_TARGET_EVENTS: usize = 234;
pub const GLM53_TARGET_LAYERS: usize = 45;
pub const GLM53_KDA_LAYERS: usize = 34;
pub const GLM53_DSA_LAYERS: usize = 11;
pub const GLM53_DENSE_LAYERS: usize = 3;
pub const GLM53_MOE_LAYERS: usize = 42;
pub const GLM53_CAPTURE_LAYERS: [u32; 5] = [5, 14, 24, 33, 42];

/// A validated schedule only. It does not launch a kernel or mutate sequence
/// state, and therefore is not an `impl Model` execution path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53ForwardOnePlan {
    schedule: Glm53TargetSchedule,
}

impl Glm53ForwardOnePlan {
    pub fn exact() -> Result<Self> {
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1))?;
        let plan = Self { schedule };
        plan.validate()?;
        Ok(plan)
    }

    pub fn schedule(&self) -> &Glm53TargetSchedule {
        &self.schedule
    }

    pub fn validate(&self) -> Result<()> {
        self.schedule.validate()?;
        let events = self.schedule.events();
        ensure!(events.len() == GLM53_TARGET_EVENTS, "GLM event-count drift");

        let mut kda = 0usize;
        let mut dsa = 0usize;
        let mut dense = 0usize;
        let mut moe = 0usize;
        let mut captures = Vec::new();
        for event in events {
            match event {
                Glm53TargetEvent::Attention {
                    kind: Glm53TargetAttentionKind::Kda,
                    ..
                } => kda += 1,
                Glm53TargetEvent::Attention {
                    kind: Glm53TargetAttentionKind::Dsa,
                    ..
                } => dsa += 1,
                Glm53TargetEvent::Ffn {
                    kind: Glm53TargetFfnKind::Dense,
                    ..
                } => dense += 1,
                Glm53TargetEvent::Ffn {
                    kind: Glm53TargetFfnKind::Moe,
                    ..
                } => moe += 1,
                Glm53TargetEvent::CaptureWidenedMhc { layer, slot } => {
                    ensure!(
                        usize::try_from(*slot)? == captures.len(),
                        "capture slot drift"
                    );
                    captures.push(*layer);
                }
                _ => {}
            }
        }
        ensure!(
            (kda, dsa, dense, moe)
                == (
                    GLM53_KDA_LAYERS,
                    GLM53_DSA_LAYERS,
                    GLM53_DENSE_LAYERS,
                    GLM53_MOE_LAYERS,
                ),
            "GLM layer-kind census drift"
        );
        ensure!(captures == GLM53_CAPTURE_LAYERS, "GLM capture-layer drift");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_token_schedule_has_exact_census_and_capture_order() {
        let plan = Glm53ForwardOnePlan::exact().unwrap();
        assert_eq!(plan.schedule().geometry.chunk_tokens, 1);
        assert_eq!(plan.schedule().events().len(), 234);
        plan.validate().unwrap();
    }
}
