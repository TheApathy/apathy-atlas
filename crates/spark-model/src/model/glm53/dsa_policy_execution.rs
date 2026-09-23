// SPDX-License-Identifier: AGPL-3.0-only

//! Real DSA save/stage and target commit adapters for policy verification.

use super::*;
use crate::model::glm53::partial_replay::ReplayPath;
use crate::model::glm53::verify_policy_transaction::{LogitsIo, VerifyCommitIo, VerifyRequest};

pub(super) struct PolicyLogits<'a> {
    pub model: &'a Glm53Exl3Model,
    pub logits: DevicePtr,
    pub bytes: usize,
    pub stream: u64,
}

impl LogitsIo for PolicyLogits<'_> {
    fn copy_logits(&mut self, destination: &mut [u8]) -> Result<usize> {
        self.model
            .copy_policy_logits(self.logits, self.bytes, destination, self.stream)?;
        Ok(self.bytes)
    }
}

pub(super) struct PolicyTarget<'a> {
    model: &'a Glm53Exl3Model,
    request: &'a VerifyRequest,
    snapshot: VerifySnapshot,
    stream: u64,
    trace_selection: Option<u32>,
    exact_verify: bool,
    closed: bool,
    timing: &'a PhaseRecorder<'a>,
}

impl<'a> PolicyTarget<'a> {
    pub fn stage(
        model: &'a Glm53Exl3Model,
        request: &'a VerifyRequest,
        stream: u64,
        timing: &'a PhaseRecorder<'a>,
    ) -> Result<(Self, DevicePtr)> {
        model.ensure_verify_healthy()?;
        ensure!(
            !glm53_exact_wide_prefill_active() && !glm53_layer_major_prefill_active(),
            "GLM policy verify cannot run inside a prefill scope"
        );
        ensure!(
            !model.gpu.stream_is_capturing(stream),
            "GLM policy verification requires eager execution"
        );
        let trace_selection = verify_trace::selection_from_env()?;
        let exact_verify = match std::env::var("ATLAS_GLM53_EXACT_VERIFY") {
            Ok(value) => parse_exact_verify_flag(Some(&value)),
            Err(std::env::VarError::NotPresent) => parse_exact_verify_flag(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("ATLAS_GLM53_EXACT_VERIFY must be valid UTF-8")
            }
        }
        .map_err(anyhow::Error::msg)?;
        let start = u32::try_from(request.start())?;
        let rows = u32::try_from(request.inputs().len())?;
        ensure!(
            model.position() == start,
            "GLM policy staged position drift"
        );
        let plan = DsaVerifyPlan::new(start, rows, model.capacity)?;
        model
            .dflash2
            .lock()
            .unwrap()
            .as_ref()
            .context("GLM policy verifier requires an installed drafter")?
            .preflight_target_rows(start, rows)?;
        let snapshot = model.bind_dsa_snapshot(&plan, stream)?;
        // Establish abandonment ownership before the first asynchronous copy,
        // including a backend panic after submission rather than an Err value.
        let mut target = Self {
            model,
            request,
            snapshot,
            stream,
            trace_selection,
            exact_verify,
            closed: false,
            timing,
        };
        if let Err(error) = timing.measure(Phase::SnapshotSaveEnqueue, || {
            target.snapshot.save(&mut DeviceCopies(model.gpu.as_ref()))
        }) {
            if target.snapshot.needs_reset() {
                model.poison_verify(stream);
            }
            return Err(error);
        }
        let staged = timing.measure(Phase::WideStage, || -> Result<DevicePtr> {
            let (logits, at, count) = if exact_verify {
                with_glm53_exact_verify(|| model.verify_tokens_staged(request.inputs(), stream))?
            } else {
                model.verify_tokens_staged(request.inputs(), stream)?
            };
            ensure!(
                at == start && count == rows,
                "GLM policy staged rows/position drift"
            );
            verify_trace::capture_row(
                model.gpu.as_ref(),
                trace_selection,
                TracePoint {
                    phase: TracePhase::Wide,
                    start,
                    rows: rows as usize,
                    anchor: request.inputs()[0],
                    selected_oracle: None,
                    device_selector: false,
                    stream,
                },
                logits,
            )?;
            Ok(logits)
        });
        match staged {
            Ok(logits) => Ok((target, logits)),
            Err(error) => match timing.measure(Phase::Abort, || target.abort_staged()) {
                Ok(()) => Err(error.context("GLM policy staging failed; DSA restored")),
                Err(restore) => {
                    Err(error.context(format!("GLM policy staging restore failed: {restore:#}")))
                }
            },
        }
    }
}

impl VerifyCommitIo for PolicyTarget<'_> {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        ensure!(
            !self.closed && std::ptr::eq(request, self.request),
            "GLM policy commit request changed"
        );
        ensure!(
            (1..=request.inputs().len()).contains(&rows),
            "GLM policy commit row extent invalid"
        );
        ensure!(
            self.model.position() as usize == request.start(),
            "GLM policy commit start drift"
        );
        let mut io = DeviceCopies(self.model.gpu.as_ref());
        let timing = self.timing;
        let full = rows == request.inputs().len();
        let replay = if full {
            ReplayPath::Scalar
        } else {
            self.model
                .partial_replay
                .path(request.inputs().len(), rows)?
        };
        if !full {
            timing.measure(Phase::PartialRestore, || self.snapshot.restore(&mut io))?;
        }
        self.snapshot.begin_commit()?;
        let committed = if full {
            timing.measure(Phase::FullCommitBody, || -> Result<()> {
                self.model.commit_accepted(self.stream)?;
                self.model.gpu.synchronize(self.stream)?;
                self.model.state.lock().unwrap().position = u32::try_from(request.start() + rows)?;
                self.model.state_hash_probe(
                    u32::try_from(request.start() + rows)?,
                    "full",
                    self.stream,
                )?;
                let mut dflash2 = self.model.dflash2.lock().unwrap();
                let runtime = dflash2
                    .as_mut()
                    .context("GLM DFlash2 runtime disappeared during policy commit")?;
                if self.exact_verify {
                    runtime.observe_target_rows_ordered(self.model, rows as u32, self.stream)?;
                } else {
                    runtime.observe_target_rows(self.model, rows as u32, self.stream)?;
                }
                Ok(())
            })
        } else if self.model.prefix_commit_active() {
            timing.measure(Phase::PartialReplay, || {
                self.model.commit_prefix_rows(
                    request.start(),
                    request.inputs().len(),
                    rows,
                    self.exact_verify,
                    self.stream,
                )
            })
        } else {
            timing.measure(Phase::PartialReplay, || -> Result<()> {
                if replay == ReplayPath::Wide {
                    let logits = self.model.replay_partial_wide(
                        request.start(),
                        &request.inputs()[..rows],
                        self.stream,
                        self.exact_verify,
                    )?;
                    verify_trace::capture_row(
                        self.model.gpu.as_ref(),
                        self.trace_selection,
                        TracePoint {
                            phase: TracePhase::Replay,
                            start: request.start() as u32,
                            rows: request.inputs().len(),
                            anchor: request.inputs()[0],
                            selected_oracle: None,
                            device_selector: false,
                            stream: self.stream,
                        },
                        logits,
                    )?;
                    return Ok(());
                }
                for (row, &token) in request.inputs()[..rows].iter().enumerate() {
                    let logits = self.model.walk(token, self.stream)?;
                    if row == 0 {
                        verify_trace::capture_row(
                            self.model.gpu.as_ref(),
                            self.trace_selection,
                            TracePoint {
                                phase: TracePhase::Replay,
                                start: request.start() as u32,
                                rows: request.inputs().len(),
                                anchor: request.inputs()[0],
                                selected_oracle: None,
                                device_selector: false,
                                stream: self.stream,
                            },
                            logits,
                        )?;
                    }
                }
                Ok(())
            })
        };
        if let Err(error) = committed {
            self.poison();
            return match timing.measure(Phase::FailureDrain, || self.snapshot.fail_commit(&mut io))
            {
                Ok(()) => Err(error),
                Err(drain) => {
                    Err(error.context(format!("GLM policy commit drain failed: {drain:#}")))
                }
            };
        }
        timing.measure(Phase::FinishFence, || self.snapshot.finish(&mut io))?;
        self.closed = true;
        Ok(self.model.position() as usize)
    }

    fn abort_staged(&mut self) -> Result<()> {
        ensure!(!self.closed, "GLM policy target transaction already closed");
        self.snapshot
            .restore(&mut DeviceCopies(self.model.gpu.as_ref()))?;
        self.closed = true;
        Ok(())
    }

    fn poison(&mut self) {
        self.model.poison_verify(self.stream);
    }
}

impl Drop for PolicyTarget<'_> {
    fn drop(&mut self) {
        // Includes policy panics/abandonment and irreversible partial commits.
        // The model owns every source/backup allocation until reset/shutdown.
        if !self.closed {
            self.model.poison_verify(self.stream);
        }
    }
}
