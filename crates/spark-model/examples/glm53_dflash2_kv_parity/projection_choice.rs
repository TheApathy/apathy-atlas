// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit diagnostic arithmetic selection; serving defaults remain unchanged.
use anyhow::{Result, ensure};
use spark_model::model::glm53::Dflash2ProbeMode as Mode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionChoice {
    Tc,
    Gemv,
}
impl ProjectionChoice {
    pub fn select(tc: bool, gemv: bool, timing: bool, projected: bool) -> Result<Option<Self>> {
        ensure!(
            !(tc && gemv),
            "TC and GEMV projection flags are mutually exclusive"
        );
        ensure!(
            !timing || (!tc && !projected),
            "timing and raw capture flags are mutually exclusive"
        );
        Ok(if gemv {
            Some(Self::Gemv)
        } else if tc || timing {
            Some(Self::Tc)
        } else {
            None
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Tc => "stable-tc",
            Self::Gemv => "stable-gemv",
        }
    }
    pub fn modes(self) -> [Mode; 3] {
        match self {
            Self::Tc => [
                Mode::FullRecompute,
                Mode::StableFullProjection,
                Mode::StableCachedProjection,
            ],
            Self::Gemv => [
                Mode::FullRecompute,
                Mode::StableGemvFullProjection,
                Mode::StableGemvCachedProjection,
            ],
        }
    }
    /// Map the existing canonical balanced schedule, preserving its SSOT.
    pub fn map_order(self, order: [Mode; 3]) -> Result<[Mode; 3]> {
        let canonical = Self::Tc.modes();
        ensure!(
            canonical
                .iter()
                .all(|mode| order.iter().filter(|m| *m == mode).count() == 1),
            "invalid canonical three-arm timing order"
        );
        let selected = self.modes();
        Ok(order.map(|mode| selected[canonical.iter().position(|m| *m == mode).unwrap()]))
    }
}
