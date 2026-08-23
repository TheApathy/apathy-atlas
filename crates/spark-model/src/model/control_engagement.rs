// SPDX-License-Identifier: AGPL-3.0-only

//! Per-verify fail-closed attestations for controlled DFlash layer paths.

use std::sync::{Mutex, Once, OnceLock};

use anyhow::{Result, bail};

const PATHS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum ControlPath {
    AttnPaged,
    AttnOut,
    AttnFfn,
    AttnPreNorm,
    AttnPostNorm,
    SsmFfn,
    SsmPreNorm,
    SsmPostNorm,
}

impl ControlPath {
    const ALL: [Self; PATHS] = [
        Self::AttnPaged,
        Self::AttnOut,
        Self::AttnFfn,
        Self::AttnPreNorm,
        Self::AttnPostNorm,
        Self::SsmFfn,
        Self::SsmPreNorm,
        Self::SsmPostNorm,
    ];

    const fn canonical(self) -> &'static str {
        match self {
            Self::AttnPaged => "attn_paged",
            Self::AttnOut => "attn_out",
            Self::AttnFfn => "attn_ffn",
            Self::AttnPreNorm => "attn_pre_norm",
            Self::AttnPostNorm => "attn_post_norm",
            Self::SsmFfn => "ssm_ffn",
            Self::SsmPreNorm => "ssm_pre_norm",
            Self::SsmPostNorm => "ssm_post_norm",
        }
    }

    fn proof_latch(self) -> &'static Once {
        static ATTN_PAGED: Once = Once::new();
        static ATTN_OUT: Once = Once::new();
        static ATTN_FFN: Once = Once::new();
        static ATTN_PRE_NORM: Once = Once::new();
        static ATTN_POST_NORM: Once = Once::new();
        static SSM_FFN: Once = Once::new();
        static SSM_PRE_NORM: Once = Once::new();
        static SSM_POST_NORM: Once = Once::new();

        match self {
            Self::AttnPaged => &ATTN_PAGED,
            Self::AttnOut => &ATTN_OUT,
            Self::AttnFfn => &ATTN_FFN,
            Self::AttnPreNorm => &ATTN_PRE_NORM,
            Self::AttnPostNorm => &ATTN_POST_NORM,
            Self::SsmFfn => &SSM_FFN,
            Self::SsmPreNorm => &SSM_PRE_NORM,
            Self::SsmPostNorm => &SSM_POST_NORM,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ControlRequests {
    pub attn_paged: bool,
    pub attn_out: bool,
    pub ffn: bool,
    pub layer_norms: bool,
}

impl ControlRequests {
    pub(crate) const fn any(self) -> bool {
        self.attn_paged || self.attn_out || self.ffn || self.layer_norms
    }

    pub(crate) const fn attention_requested(self) -> bool {
        self.attn_paged || self.attn_out || self.ffn || self.layer_norms
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LayerCounts {
    pub attention: usize,
    pub ssm: usize,
}

#[derive(Debug)]
struct Capture {
    requested: [bool; PATHS],
    expected: [usize; PATHS],
    engaged: [usize; PATHS],
}

impl Capture {
    fn new(requests: ControlRequests, counts: LayerCounts) -> Result<Self> {
        if (requests.attn_paged || requests.attn_out) && counts.attention == 0 {
            bail!(
                "DFLASH_CONTROL_PATH_PROOF requested=true engaged=false requirement=attention \
                 control requested but verify has no attention layers"
            );
        }
        if (requests.ffn || requests.layer_norms) && counts.attention + counts.ssm == 0 {
            bail!(
                "DFLASH_CONTROL_PATH_PROOF requested=true engaged=false requirement=FFN/norm \
                 control requested but verify has no executable layers"
            );
        }

        let mut requested = [false; PATHS];
        let mut expected = [0; PATHS];
        for path in ControlPath::ALL {
            let (wants_path, expected_layers) = match path {
                ControlPath::AttnPaged => (requests.attn_paged, counts.attention),
                ControlPath::AttnOut => (requests.attn_out, counts.attention),
                ControlPath::AttnFfn => (requests.ffn, counts.attention),
                ControlPath::AttnPreNorm | ControlPath::AttnPostNorm => {
                    (requests.layer_norms, counts.attention)
                }
                ControlPath::SsmFfn => (requests.ffn, counts.ssm),
                ControlPath::SsmPreNorm | ControlPath::SsmPostNorm => {
                    (requests.layer_norms, counts.ssm)
                }
            };
            let idx = path as usize;
            requested[idx] = wants_path && expected_layers > 0;
            expected[idx] = expected_layers;
        }
        Ok(Self {
            requested,
            expected,
            engaged: [0; PATHS],
        })
    }

    fn proof_line(&self, path: ControlPath, engaged: bool) -> String {
        let idx = path as usize;
        format!(
            "DFLASH_CONTROL_PATH_PROOF path={} requested=true engaged={engaged} \
             expected_layers={} engaged_layers={}",
            path.canonical(),
            self.expected[idx],
            self.engaged[idx]
        )
    }

    fn require(&self, path: ControlPath, applicable: bool, requirement: &str) -> Result<bool> {
        if !self.requested[path as usize] {
            return Ok(false);
        }
        if !applicable {
            bail!("{} requirement={requirement}", self.proof_line(path, false));
        }
        Ok(true)
    }

    fn engage(&mut self, path: ControlPath) -> Result<()> {
        let idx = path as usize;
        if !self.requested[idx] {
            bail!(
                "DFLASH_CONTROL_PATH_PROOF path={} requested=false engaged=true",
                path.canonical()
            );
        }
        self.engaged[idx] += 1;
        if self.engaged[idx] > self.expected[idx] {
            bail!(
                "{} requirement=engagement count exceeded expected layer count",
                self.proof_line(path, false)
            );
        }
        Ok(())
    }

    fn finish(self) -> Result<Vec<(ControlPath, String)>> {
        let mut proofs = Vec::new();
        for path in ControlPath::ALL {
            let idx = path as usize;
            if !self.requested[idx] {
                continue;
            }
            if self.engaged[idx] != self.expected[idx] {
                bail!(
                    "{} requirement=engagement count did not match expected layer count",
                    self.proof_line(path, false)
                );
            }
            proofs.push((path, self.proof_line(path, true)));
        }
        Ok(proofs)
    }
}

fn state() -> &'static Mutex<Option<Capture>> {
    static STATE: OnceLock<Mutex<Option<Capture>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

pub(crate) struct EngagementGuard {
    armed: bool,
}

impl EngagementGuard {
    pub(crate) fn finish(mut self) -> Result<()> {
        let capture = state()
            .lock()
            .expect("DFlash control engagement mutex")
            .take()
            .ok_or_else(|| anyhow::anyhow!("DFLASH control engagement capture is not active"))?;
        self.armed = false;
        for (path, proof) in capture.finish()? {
            path.proof_latch().call_once(|| tracing::warn!("{proof}"));
        }
        Ok(())
    }
}

impl Drop for EngagementGuard {
    fn drop(&mut self) {
        if self.armed {
            state()
                .lock()
                .expect("DFlash control engagement mutex")
                .take();
        }
    }
}

pub(crate) fn begin(
    requests: ControlRequests,
    counts: LayerCounts,
) -> Result<Option<EngagementGuard>> {
    if !requests.any() {
        return Ok(None);
    }
    let capture = Capture::new(requests, counts)?;
    let mut active = state().lock().expect("DFlash control engagement mutex");
    if active.is_some() {
        bail!("DFLASH control engagement capture overlaps an active verify");
    }
    *active = Some(capture);
    Ok(Some(EngagementGuard { armed: true }))
}

pub(crate) fn require(path: ControlPath, applicable: bool, requirement: &str) -> Result<bool> {
    let active = state().lock().expect("DFlash control engagement mutex");
    let Some(capture) = active.as_ref() else {
        return Ok(false);
    };
    capture.require(path, applicable, requirement)
}

pub(crate) fn engage(path: ControlPath) -> Result<()> {
    let mut active = state().lock().expect("DFlash control engagement mutex");
    let Some(capture) = active.as_mut() else {
        return Ok(());
    };
    capture.engage(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_requests() -> ControlRequests {
        ControlRequests {
            attn_paged: true,
            attn_out: true,
            ffn: true,
            layer_norms: true,
        }
    }

    #[test]
    fn canonical_proofs_include_exact_layer_counts() {
        let mut capture = Capture::new(
            all_requests(),
            LayerCounts {
                attention: 2,
                ssm: 3,
            },
        )
        .unwrap();
        for path in ControlPath::ALL {
            assert!(capture.require(path, true, "unused").unwrap());
            let expected = capture.expected[path as usize];
            for _ in 0..expected {
                capture.engage(path).unwrap();
            }
        }
        let proofs = capture.finish().unwrap();
        assert_eq!(proofs.len(), PATHS);
        let expected = [
            ("attn_paged", 2),
            ("attn_out", 2),
            ("attn_ffn", 2),
            ("attn_pre_norm", 2),
            ("attn_post_norm", 2),
            ("ssm_ffn", 3),
            ("ssm_pre_norm", 3),
            ("ssm_post_norm", 3),
        ];
        for ((path, proof), (canonical, count)) in proofs.iter().zip(expected) {
            assert_eq!(path.canonical(), canonical);
            assert_eq!(
                proof,
                &format!(
                    "DFLASH_CONTROL_PATH_PROOF path={canonical} requested=true engaged=true \
                     expected_layers={count} engaged_layers={count}"
                )
            );
        }
    }

    #[test]
    fn missing_or_excess_engagement_fails_closed() {
        let mut missing = Capture::new(
            ControlRequests {
                attn_out: true,
                ..Default::default()
            },
            LayerCounts {
                attention: 2,
                ssm: 0,
            },
        )
        .unwrap();
        missing.engage(ControlPath::AttnOut).unwrap();
        assert!(
            missing
                .finish()
                .unwrap_err()
                .to_string()
                .contains("engaged_layers=1")
        );

        let mut excess = Capture::new(
            ControlRequests {
                attn_out: true,
                ..Default::default()
            },
            LayerCounts {
                attention: 1,
                ssm: 0,
            },
        )
        .unwrap();
        excess.engage(ControlPath::AttnOut).unwrap();
        assert!(excess.engage(ControlPath::AttnOut).is_err());
    }

    #[test]
    fn requested_but_inapplicable_fails_before_engagement() {
        let capture = Capture::new(
            ControlRequests {
                attn_paged: true,
                ..Default::default()
            },
            LayerCounts {
                attention: 1,
                ssm: 0,
            },
        )
        .unwrap();
        let error = capture
            .require(ControlPath::AttnPaged, false, "flat chain required")
            .unwrap_err()
            .to_string();
        assert!(error.contains("path=attn_paged requested=true engaged=false"));
        assert!(error.contains("requirement=flat chain required"));
    }

    #[test]
    fn absent_layer_class_is_not_fabricated() {
        let mut capture = Capture::new(
            ControlRequests {
                ffn: true,
                layer_norms: true,
                ..Default::default()
            },
            LayerCounts {
                attention: 2,
                ssm: 0,
            },
        )
        .unwrap();
        for path in [
            ControlPath::AttnFfn,
            ControlPath::AttnPreNorm,
            ControlPath::AttnPostNorm,
        ] {
            capture.engage(path).unwrap();
            capture.engage(path).unwrap();
        }
        let proofs = capture.finish().unwrap();
        assert_eq!(proofs.len(), 3);
        assert!(proofs.iter().all(|(path, _)| !matches!(
            path,
            ControlPath::SsmFfn | ControlPath::SsmPreNorm | ControlPath::SsmPostNorm
        )));
    }
}
