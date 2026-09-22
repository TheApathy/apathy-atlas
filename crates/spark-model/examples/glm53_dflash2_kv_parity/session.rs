// SPDX-License-Identifier: AGPL-3.0-only
//! Whole-session ownership, including installed candidate and observer storage.
use anyhow::{Context, Result, bail, ensure};
use spark_model::model::glm53::verify_policy_transaction::{
    PolicyAdvance, VerifyOutcome, VerifyPolicy,
};
use spark_model::model::glm53::{Glm53Dflash2Runtime, Glm53Exl3Model, ProbeCapture, ProbeIo};
use spark_model::traits::{Model, SequenceState};
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, admit_glm53_exl3_files, load_glm53_exl3_store,
    materialize_glm53_exl3_native,
};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::{mem::ManuallyDrop, path::Path};

const RESERVE: usize = 8 * 1024 * 1024 * 1024;
pub struct ProbeSession {
    pub model: ManuallyDrop<Glm53Exl3Model>,
    pub reference: Option<ManuallyDrop<Glm53Dflash2Runtime>>,
    pub candidate_capture: Option<ProbeCapture>,
    pub reference_capture: Option<ProbeCapture>,
    pub original_capture: Option<ProbeCapture>,
    pub stream: u64,
    pub seq: Option<SequenceState>,
    pub logits: DevicePtr,
    closed: bool,
}
pub struct GpuProbeIo<'a>(pub &'a dyn GpuBackend);
impl ProbeIo for GpuProbeIo<'_> {
    fn copy(&mut self, source: u64, dst: &mut [u8], stream: u64) -> Result<()> {
        self.0.copy_d2h_on_stream(DevicePtr(source), dst, stream)
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
}
impl ProbeSession {
    pub fn load(target: &Path, draft: &Path) -> Result<Self> {
        // Existing loader-before-return failure semantics are NOT strengthened
        // by this diagnostic. Envelope coverage starts at constructed model.
        let files = admit_glm53_exl3_files(target)?;
        let backend = Box::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
        let gpu: &dyn GpuBackend = backend.as_ref();
        let store = match load_glm53_exl3_store(&files, gpu, RESERVE) {
            Ok(store) => store,
            Err(error) => match error.retry_cleanup(gpu) {
                Ok(primary) => return Err(primary),
                Err(retained) => bail!("target loader cleanup failed: {retained}"),
            },
        };
        let catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
        let native =
            materialize_glm53_exl3_native(&catalog, gpu).map_err(|e| anyhow::anyhow!("{e}"))?;
        let weights = Glm53Exl3TargetWeights::new(&catalog, &native)?;
        drop(catalog);
        let model = Glm53Exl3Model::new(backend, store, native, weights, 2048)?;
        let stream = model.gpu().default_stream();
        let mut session = Self {
            model: ManuallyDrop::new(model),
            reference: None,
            candidate_capture: None,
            reference_capture: None,
            original_capture: None,
            stream,
            seq: None,
            logits: DevicePtr::NULL,
            closed: false,
        };
        session.model.install_dflash2(draft)?;
        session.reference = Some(ManuallyDrop::new(Glm53Dflash2Runtime::load(
            session.model.gpu(),
            draft,
            session.model.dflash_lm_head(),
        )?));
        session.seq = Some(session.model.alloc_sequence()?);
        session.model.gpu().synchronize(stream)?;
        Ok(session)
    }
    pub fn advance(&mut self, corpus: &[u32], end: usize) -> Result<()> {
        ensure!(
            end <= 2047 && end <= corpus.len(),
            "probe context exceeds input/capacity"
        );
        let seq = self.seq.as_mut().context("missing live probe sequence")?;
        ensure!(seq.seq_len <= end, "probe contexts regressed without reset");
        while seq.seq_len < end {
            self.logits = self.model.decode(corpus[seq.seq_len], seq, self.stream)?;
            // Candidate ingestion is the actual installed model walk observer.
            self.reference
                .as_mut()
                .context("missing reference")?
                .observe_target(&self.model, self.stream)?;
        }
        Ok(())
    }
    pub fn reset(&mut self, first: u32) -> Result<()> {
        self.reference
            .as_mut()
            .context("missing reference")?
            .reset_context(self.model.gpu())?;
        let seq = self.seq.as_mut().context("missing live probe sequence")?;
        self.logits = self.model.prefill(&[first], seq, self.stream)?;
        self.reference
            .as_mut()
            .unwrap()
            .observe_target(&self.model, self.stream)?;
        self.model.gpu().synchronize(self.stream)?;
        Ok(())
    }
    pub fn anchor(&self) -> Result<u32> {
        self.model.argmax_host(self.logits)
    }

    /// Explicit deterministic sampler exercises real target commit/rollback;
    /// these are NOT naturally accepted model predictions or a quality score.
    pub fn policy_case(&mut self, drafts: [u32; 7], accept_all: bool) -> Result<serde_json::Value> {
        let anchor = self.anchor()?;
        let inputs = std::iter::once(anchor).chain(drafts).collect::<Vec<_>>();
        let seq = self.seq.as_mut().context("missing live probe sequence")?;
        let start = seq.seq_len;
        let bound = self
            .model
            .bind_verify_policy_request(&inputs, seq, self.stream)?;
        let mut policy = ForcedPolicy {
            inputs: inputs.clone(),
            accept_all,
            cursor: 0,
            saved: 0,
        };
        let outcome = self.model.decode_verify_with_policy(bound, &mut policy)?;
        let VerifyOutcome::Committed(receipt) = outcome else {
            bail!("diagnostic policy unexpectedly requested replay");
        };
        let expected = if accept_all { 7 } else { 0 };
        ensure!(
            receipt.accepted_drafts() == expected,
            "forced state-path acceptance drift"
        );
        let rows = expected + 1;
        if rows == 8 {
            self.reference
                .as_mut()
                .unwrap()
                .observe_target_rows(&self.model, 8, self.stream)?;
        } else {
            self.reference
                .as_mut()
                .unwrap()
                .observe_target(&self.model, self.stream)?;
        }
        let published =
            receipt.publish(&mut seq.tokens, &mut seq.seq_len, &mut seq.kv_valid_tokens)?;
        ensure!(
            seq.seq_len == start + rows,
            "diagnostic host publication drift"
        );
        self.logits = self
            .model
            .logits_buffer_ptr()
            .offset((rows - 1) * 154_880 * 2);
        Ok(
            serde_json::json!({"kind":if accept_all {"forced-full-accept"} else {"forced-first-reject"},
            "start":start,"end":seq.seq_len,"accepted_drafts":expected,"inputs":inputs,
            "emitted":published.emitted_tokens(),"natural_acceptance":false,"partial_acceptance_covered":false}),
        )
    }
    pub fn close(mut self) -> Result<()> {
        let drained = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let gpu = self.model.gpu();
            if let Some(capture) = &mut self.candidate_capture {
                capture.drain(&mut GpuProbeIo(gpu))?;
            }
            if let Some(capture) = &mut self.reference_capture {
                capture.drain(&mut GpuProbeIo(gpu))?;
            }
            if let Some(capture) = &mut self.original_capture {
                capture.drain(&mut GpuProbeIo(gpu))?;
            }
            gpu.synchronize(self.stream)
        }));
        match drained {
            Ok(Ok(())) => {}
            failure => {
                let reason = match failure {
                    Ok(Err(e)) => format!("{e:#}"),
                    _ => "completion panic".into(),
                };
                std::mem::forget(self);
                bail!(
                    "probe session/model/backend/host owners quarantined until process teardown: {reason}"
                );
            }
        }
        if let Some(reference) = self.reference.take() {
            ManuallyDrop::into_inner(reference).free(self.model.gpu())?;
        }
        if let Some(mut seq) = self.seq.take() {
            self.model.free_sequence(&mut seq)?;
        }
        // All submitted work has a completion receipt before consuming owners.
        let model = unsafe { ManuallyDrop::take(&mut self.model) };
        self.closed = true;
        model.free()
    }
}
impl Drop for ProbeSession {
    fn drop(&mut self) {
        if !self.closed {
            // No GPU callbacks in Drop; ManuallyDrop retains model/backend and
            // reference. Pending capture readbacks independently retain Vecs.
            eprintln!(
                "probe session abandoned: device/backend owners retained until process teardown"
            );
        }
    }
}
struct ForcedPolicy {
    inputs: Vec<u32>,
    accept_all: bool,
    cursor: usize,
    saved: usize,
}
impl VerifyPolicy for ForcedPolicy {
    fn checkpoint(&mut self) -> Result<()> {
        self.saved = self.cursor;
        Ok(())
    }
    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32> {
        ensure!(
            row == self.cursor && logits.len() == 154_880 * 2,
            "forced policy actual logits/row drift"
        );
        if !self.accept_all {
            return Ok((self.inputs[1] + 1) % 154_880);
        }
        Ok(self.inputs.get(row + 1).copied().unwrap_or(0))
    }
    fn advance(&mut self, _: u32) -> Result<PolicyAdvance> {
        self.cursor += 1;
        Ok(PolicyAdvance::Continue)
    }
    fn restore(&mut self) -> Result<()> {
        self.cursor = self.saved;
        Ok(())
    }
}
