// SPDX-License-Identifier: AGPL-3.0-only

//! Success-only receipts for explicitly selected Qwen4 full-prompt paths.

use std::sync::{Mutex, OnceLock};

use crate::layers::qwen3_ssm::qwen4_prefill_gemm::Mode;
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;

use self::geometry::Geometry;
use self::selectors::Selectors;

mod geometry;
mod selectors;

const ATTN_SELECTOR: &str = "ATLAS_QWEN4_ATTN_PREFILL_BATCH";
const SSM_SELECTOR: &str = "ATLAS_QWEN4_SSM_PREFILL_BATCH";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefillPath {
    Attention,
    Ssm,
}

#[derive(Debug)]
struct Capture {
    selectors: Selectors,
    m: usize,
    geometry: Geometry,
    attention_engaged: usize,
    ssm_engaged: usize,
}

impl Capture {
    fn new(selectors: Selectors, m: usize, geometry: Geometry) -> Result<Self> {
        ensure!(
            m > 1,
            "Qwen4 full-prompt selector requires actual M > 1, got {m}"
        );
        geometry.validate(selectors)?;
        Ok(Self {
            selectors,
            m,
            geometry,
            attention_engaged: 0,
            ssm_engaged: 0,
        })
    }

    fn engage(&mut self, path: PrefillPath, m: usize) -> Result<()> {
        ensure!(
            m == self.m,
            "Qwen4 prefill receipt M drift: expected {} got {m}",
            self.m
        );
        ensure!(
            self.selectors.value(path),
            "unselected Qwen4 prefill path engaged: {path:?}"
        );
        let (engaged, expected) = match path {
            PrefillPath::Attention => (&mut self.attention_engaged, self.geometry.attention_layers),
            PrefillPath::Ssm => (&mut self.ssm_engaged, self.geometry.ssm_layers),
        };
        *engaged += 1;
        ensure!(
            *engaged <= expected,
            "Qwen4 prefill {path:?} engagement exceeds {expected}"
        );
        Ok(())
    }

    fn finish(self) -> Result<Vec<String>> {
        let mut lines = Vec::with_capacity(2);
        for path in [PrefillPath::Attention, PrefillPath::Ssm] {
            if !self.selectors.value(path) {
                continue;
            }
            let (family, selector, engaged, expected, detail) = match path {
                PrefillPath::Attention => (
                    "attention",
                    ATTN_SELECTOR,
                    self.attention_engaged,
                    self.geometry.attention_layers,
                    format!(
                        "Q={} KV={} HD={}",
                        self.geometry.q_heads, self.geometry.kv_heads, self.geometry.head_dim
                    ),
                ),
                PrefillPath::Ssm => (
                    "ssm",
                    SSM_SELECTOR,
                    self.ssm_engaged,
                    self.geometry.ssm_layers,
                    format!(
                        "NK={} KD={} NV={} VD={} D={} QKVZ={}",
                        self.geometry.key_heads,
                        self.geometry.key_dim,
                        self.geometry.value_heads,
                        self.geometry.value_dim,
                        self.geometry.conv_dim,
                        self.geometry.qkvz
                    ),
                ),
            };
            ensure!(
                engaged == expected,
                "Qwen4 prefill {family} engagement incomplete: expected {expected} got {engaged}"
            );
            let (selector, mode) = if self.selectors.moe_only {
                let core = if path == PrefillPath::Ssm && self.selectors.ssm_exact {
                    match self.selectors.ssm_gemm {
                        Mode::Off => "exact_projection_sequence_nosnap",
                        Mode::Qkvz => "bf16_mma_qkvz_exact_output_sequence_nosnap",
                        Mode::Out => "exact_qkvz_bf16_mma_output_sequence_nosnap",
                        Mode::All => "bf16_mma_projections_sequence_nosnap",
                    }
                } else {
                    "serial_token_ordered"
                };
                let hc = if self.selectors.hyper_exact {
                    "exact_m32"
                } else {
                    "serial"
                };
                let schedule = if self.selectors.moe_compact {
                    "original_compact"
                } else {
                    "original_grid"
                };
                (
                    crate::layers::qwen4_prefill_moe::SELECTOR,
                    format!(
                        " ffn=grouped core={core} hc={hc} routed_schedule={schedule} ssm_projection_gemm={}",
                        self.selectors.ssm_gemm.as_str()
                    ),
                )
            } else {
                (selector, String::new())
            };
            lines.push(format!(
                "QWEN4_PREFILL_SELECTOR_RECEIPT family={family} selector={selector} value=1 M={} \
                 attention_selector={} ssm_selector={} path_success=enqueued \
                 serialized_fallback=false expected_layers={expected} engaged_layers={engaged} \
                 H={} L={} E={} TOPK={} I={} SI={} {detail}{mode}",
                self.m,
                u8::from(self.selectors.attention),
                u8::from(self.selectors.ssm),
                self.geometry.hidden,
                self.geometry.layers,
                self.geometry.experts,
                self.geometry.top_k,
                self.geometry.routed_intermediate,
                self.geometry.shared_intermediate,
            ));
        }
        Ok(lines)
    }
}

fn state() -> &'static Mutex<Option<Capture>> {
    static STATE: OnceLock<Mutex<Option<Capture>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

pub(crate) struct ReceiptGuard {
    armed: bool,
}

impl ReceiptGuard {
    pub(crate) fn finish(mut self) -> Result<()> {
        let capture = state()
            .lock()
            .expect("Qwen4 prefill receipt mutex")
            .take()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 prefill receipt capture is not active"))?;
        self.armed = false;
        for line in capture.finish()? {
            tracing::info!("{line}");
        }
        Ok(())
    }
}

impl Drop for ReceiptGuard {
    fn drop(&mut self) {
        if self.armed {
            state().lock().expect("Qwen4 prefill receipt mutex").take();
        }
    }
}

pub(crate) fn begin(config: &ModelConfig, m: usize) -> Result<Option<ReceiptGuard>> {
    let selectors = Selectors::from_env()?;
    if !selectors.any() {
        return Ok(None);
    }
    // A one-token continuation uses unchanged decode and earns no batch receipt.
    if selectors.moe_only && m == 1 {
        return Ok(None);
    }
    let capture = Capture::new(selectors, m, Geometry::from_config(config))?;
    let mut active = state().lock().expect("Qwen4 prefill receipt mutex");
    if active.is_some() {
        bail!("Qwen4 prefill receipt capture overlaps an active prefill");
    }
    *active = Some(capture);
    Ok(Some(ReceiptGuard { armed: true }))
}

pub(crate) fn engage(path: PrefillPath, m: usize) -> Result<()> {
    let mut active = state().lock().expect("Qwen4 prefill receipt mutex");
    let Some(capture) = active.as_mut() else {
        return Ok(());
    };
    capture.engage(path, m)
}

#[cfg(test)]
mod moe_tests;
#[cfg(test)]
mod tests;
