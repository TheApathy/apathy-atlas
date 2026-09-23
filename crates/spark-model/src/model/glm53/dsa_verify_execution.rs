// SPDX-License-Identifier: AGPL-3.0-only

//! Model-owned DFlash2 transaction with separately scoped exact verification.

use super::super::dsa_verify_plan::{DsaVerifyPlan, MAX_BACKUP_BYTES};
use super::super::dsa_verify_transaction::{CopyIo, DeviceRegion, VerifySnapshot};
use super::super::phase_timing_wrappers::{TimedCommit, TimedLogits, TimedPolicy};
use super::*;
use crate::layers::ops::{
    glm53_exact_wide_prefill_active, parse_exact_verify_flag, with_glm53_exact_verify,
    with_glm53_native_rows,
};

#[path = "verify_trace.rs"]
mod verify_trace;
use verify_trace::{TracePhase, TracePoint};
#[path = "dsa_policy_execution.rs"]
mod policy_execution;

struct DeviceCopies<'a>(&'a dyn GpuBackend);

impl CopyIo for DeviceCopies<'_> {
    fn copy(&mut self, source: u64, destination: u64, bytes: usize, stream: u64) -> Result<()> {
        self.0
            .copy_d2d_async(DevicePtr(source), DevicePtr(destination), bytes, stream)
    }

    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
}

impl Glm53Exl3Model {
    pub(super) fn ensure_verify_healthy(&self) -> Result<()> {
        ensure!(
            self.state.lock().unwrap().poisoned_stream.is_none(),
            "GLM sequence is poisoned after a failed state transaction; successful reset required"
        );
        Ok(())
    }

    pub(in crate::model::glm53) fn poison_verify(&self, stream: u64) {
        self.state.lock().unwrap().poisoned_stream = Some(stream);
    }

    pub(in crate::model::glm53) fn verify_dflash2_with_policy(
        &self,
        bound: super::super::verify_policy_binding::BoundVerifyRequest,
        policy: &mut dyn super::super::verify_policy_transaction::VerifyPolicy,
    ) -> Result<super::super::verify_policy_transaction::VerifyOutcome> {
        let (request, stream) = self.consume_policy_request(bound)?;
        let generation = self
            .phase_timing
            .enabled()
            .then(|| self.state.lock().unwrap().generation);
        let clock = HostClock::default();
        let timing = PhaseRecorder::new(self.phase_timing, &clock);
        let result = timing.measure(Phase::VerifyTotal, || {
            let (mut target, logits) = timing.measure(Phase::StageTotal, || {
                policy_execution::PolicyTarget::stage(self, &request, stream, &timing)
            })?;
            let mut source = policy_execution::PolicyLogits {
                model: self,
                logits,
                bytes: request.inputs().len() * VOCAB as usize * 2,
                rows: request.inputs().len(),
                stream,
            };
            let mut source = TimedLogits::new(&mut source, &timing);
            let mut policy = TimedPolicy::new(policy, &timing);
            let mut target = TimedCommit::new(&mut target, &timing);
            super::super::verify_policy_transaction::run_verify_policy_transaction(
                &request,
                &mut source,
                &mut policy,
                &mut target,
            )
        });
        if let Some(generation) = generation {
            use super::super::verify_policy_transaction::VerifyOutcome;
            let (outcome, accepted) = match &result {
                Ok(VerifyOutcome::Committed(commit)) => (
                    if commit.accepted_drafts() + 1 == request.inputs().len() {
                        "committed_full"
                    } else {
                        "committed_partial"
                    },
                    Some(commit.accepted_drafts()),
                ),
                Ok(VerifyOutcome::RestoredForOrdinaryReplay(_)) => {
                    ("restored_for_ordinary_replay", None)
                }
                Err(_) => ("error", None),
            };
            timing.emit(TimingContext {
                kind: "verify",
                generation,
                start: request.start(),
                rows: request.inputs().len(),
                stream,
                outcome,
                accepted,
                readback_bytes: request.inputs().len() * VOCAB as usize * 2,
            });
        }
        result
    }

    fn bind_dsa_snapshot(&self, plan: &DsaVerifyPlan, stream: u64) -> Result<VerifySnapshot> {
        let regions = self
            .dsa_cache
            .iter()
            .map(|cache| {
                [
                    cache.latent_cache_bf16,
                    cache.pool_keys_bf16,
                    cache.pool_validity_u8,
                    cache.prior_tail_keys_bf16,
                    cache.prior_tail_gates_bf16,
                    cache.prior_tail_validity_u8,
                ]
                .map(|buffer| DeviceRegion {
                    address: buffer.ptr.0,
                    bytes: buffer.bytes,
                })
            })
            .collect::<Vec<_>>();
        // One allocation owned by the model, never by a stack transaction. It
        // survives every failed async operation and is explicitly freed at shutdown.
        let mut backup = self.dsa_verify_backup.lock().unwrap();
        if *backup == DevicePtr::NULL {
            *backup = self
                .gpu
                .alloc(MAX_BACKUP_BYTES)
                .context("GLM bounded DSA verify backup allocation")?;
        }
        VerifySnapshot::bind(
            plan,
            &regions,
            DeviceRegion {
                address: backup.0,
                bytes: MAX_BACKUP_BYTES,
            },
            stream,
        )
    }

    pub(super) fn verify_dflash2_transaction(
        &self,
        tokens: &[u32],
        stream: u64,
    ) -> Result<Vec<u32>> {
        let trace_selection = verify_trace::selection_from_env()?;
        let exact_verify = match std::env::var("ATLAS_GLM53_EXACT_VERIFY") {
            Ok(value) => parse_exact_verify_flag(Some(&value)),
            Err(std::env::VarError::NotPresent) => parse_exact_verify_flag(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("ATLAS_GLM53_EXACT_VERIFY must be valid UTF-8")
            }
        }
        .map_err(anyhow::Error::msg)?;
        self.ensure_verify_healthy()?;
        ensure!(
            !glm53_exact_wide_prefill_active() && !glm53_layer_major_prefill_active(),
            "GLM DFlash2 transaction cannot run inside a prefill execution scope"
        );
        ensure!(
            !self.gpu.stream_is_capturing(stream),
            "GLM DFlash2 transaction requires eager execution"
        );
        ensure!(
            (2..=DFLASH2_MAX_ROWS).contains(&tokens.len()),
            "GLM DFlash2 verifier needs 2..=8 rows"
        );
        ensure!(
            tokens.iter().all(|&token| token < VOCAB),
            "GLM DFlash2 verifier token is outside vocabulary"
        );
        let start = self.position();
        let rows = u32::try_from(tokens.len())?;
        let plan = DsaVerifyPlan::new(start, rows, self.capacity)?;
        self.dflash2
            .lock()
            .unwrap()
            .as_ref()
            .context("GLM DFlash2 verifier requires an installed drafter")?
            .preflight_target_rows(start, rows)?;
        let mut snapshot = self.bind_dsa_snapshot(&plan, stream)?;
        let mut io = DeviceCopies(self.gpu.as_ref());
        if let Err(error) = snapshot.save(&mut io) {
            if snapshot.needs_reset() {
                self.poison_verify(stream);
            }
            return Err(error);
        }

        let staged = (|| -> Result<Vec<u32>> {
            let (logits, staged_start, staged_rows) = if exact_verify {
                with_glm53_native_rows(self.native_rows, || {
                    with_glm53_exact_verify(|| self.verify_tokens_staged(tokens, stream))
                })?
            } else {
                self.verify_tokens_staged(tokens, stream)?
            };
            ensure!(
                staged_start == start && staged_rows == rows,
                "GLM verify snapshot position drift"
            );
            let oracle = if self.dflash2_device_argmax {
                self.argmax_rows_device(logits, tokens.len(), stream)?
            } else {
                self.argmax_rows_host(logits, tokens.len())?
            };
            ensure!(
                oracle.len() == tokens.len() && oracle.iter().all(|&token| token < VOCAB),
                "GLM DFlash2 verifier returned invalid oracle rows"
            );
            verify_trace::capture_row(
                self.gpu.as_ref(),
                trace_selection,
                TracePoint {
                    phase: TracePhase::Wide,
                    start,
                    rows: tokens.len(),
                    anchor: tokens[0],
                    selected_oracle: Some(oracle[0]),
                    device_selector: self.dflash2_device_argmax,
                    stream,
                },
                logits,
            )?;
            Ok(oracle)
        })();
        let oracle = match staged {
            Ok(oracle) => oracle,
            Err(error) => {
                return match snapshot.restore(&mut io) {
                    Ok(()) => Err(error.context("GLM verifier failed before commit; DSA restored")),
                    Err(restore) => {
                        self.poison_verify(stream);
                        Err(error.context(format!(
                            "GLM DSA rollback failed; sequence poisoned: {restore:#}"
                        )))
                    }
                };
            }
        };
        let drafts = &tokens[1..];
        let accepted = drafts
            .iter()
            .zip(&oracle)
            .take_while(|(draft, target)| draft == target)
            .count();
        if accepted != drafts.len() {
            if let Err(error) = snapshot.restore(&mut io) {
                self.poison_verify(stream);
                return Err(error);
            }
        }
        // From this point either commit_accepted or walk can mutate persistent
        // KDA. The DSA-only backup MUST NOT be advertised as whole-model rollback.
        if let Err(error) = snapshot.begin_commit() {
            self.poison_verify(stream);
            return Err(error);
        }
        let committed = (|| -> Result<()> {
            if accepted == drafts.len() {
                self.commit_accepted(stream)?;
                self.gpu.synchronize(stream)?;
                self.state.lock().unwrap().position = start + rows;
                self.dflash2
                    .lock()
                    .unwrap()
                    .as_mut()
                    .context("GLM DFlash2 runtime disappeared during verify")?
                    .observe_target_rows(self, rows, stream)?;
            } else {
                for (row, &token) in tokens[..=accepted].iter().enumerate() {
                    let replay_logits = self.walk(token, stream)?;
                    if row == 0 {
                        verify_trace::capture_row(
                            self.gpu.as_ref(),
                            trace_selection,
                            TracePoint {
                                phase: TracePhase::Replay,
                                start,
                                rows: tokens.len(),
                                anchor: tokens[0],
                                selected_oracle: None,
                                device_selector: self.dflash2_device_argmax,
                                stream,
                            },
                            replay_logits,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = committed {
            self.poison_verify(stream);
            return match snapshot.fail_commit(&mut io) {
                Ok(()) => Err(error.context("GLM verify commit/replay failed; sequence poisoned")),
                Err(drain) => {
                    Err(error.context(format!("GLM verify commit also failed to drain: {drain:#}")))
                }
            };
        }
        if let Err(error) = snapshot.finish(&mut io) {
            self.poison_verify(stream);
            return Err(error);
        }
        Ok(oracle)
    }
}
