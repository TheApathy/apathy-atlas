// SPDX-License-Identifier: AGPL-3.0-only
//! Retain the constructed model/backend across every failed diagnostic exit.
use anyhow::{Context, Result, bail};
use spark_model::model::glm53::Glm53Exl3Model;
use spark_model::traits::{Model, SequenceState};
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, admit_glm53_exl3_files, load_glm53_exl3_store,
    materialize_glm53_exl3_native,
};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::{
    mem::{self, ManuallyDrop},
    path::Path,
};

pub struct Session {
    pub model: ManuallyDrop<Glm53Exl3Model>,
    pub seq: Option<SequenceState>,
    pub stream: u64,
    pub logits: DevicePtr,
    closed: bool,
}
impl Session {
    pub fn load(target: &Path, draft: &Path) -> Result<Self> {
        // Existing pre-construction loader cleanup is unchanged by this probe.
        let files = admit_glm53_exl3_files(target)?;
        let backend = Box::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
        let gpu: &dyn GpuBackend = backend.as_ref();
        let store = match load_glm53_exl3_store(&files, gpu, 8 * 1024 * 1024 * 1024) {
            Ok(store) => store,
            Err(error) => match error.retry_cleanup(gpu) {
                Ok(primary) => return Err(primary),
                Err(retained) => bail!("state probe loader cleanup failed: {retained}"),
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
            seq: None,
            stream,
            logits: DevicePtr::NULL,
            closed: false,
        };
        session.model.install_dflash2(draft)?;
        session.seq = Some(session.model.alloc_sequence()?);
        session.model.gpu().synchronize(stream)?;
        Ok(session)
    }

    pub fn prefix(&mut self, tokens: &[u32]) -> Result<()> {
        let seq = self.seq.as_mut().context("missing state probe sequence")?;
        self.logits = self.model.prefill(tokens, seq, self.stream)?;
        Ok(())
    }

    pub fn close(mut self) -> Result<()> {
        let drained = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.model.gpu().synchronize(self.stream)
        }));
        match drained {
            Ok(Ok(())) => {}
            failed => {
                let reason = match failed {
                    Ok(Err(e)) => format!("{e:#}"),
                    _ => "completion panic".into(),
                };
                mem::forget(self);
                bail!(
                    "state probe model/backend/readback owners retained until process teardown: {reason}"
                );
            }
        }
        if let Some(mut seq) = self.seq.take() {
            self.model.free_sequence(&mut seq)?;
        }
        // Model::free performs the existing model-owned readback drain before release.
        let model = unsafe { ManuallyDrop::take(&mut self.model) };
        self.closed = true;
        model.free()
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if !self.closed {
            eprintln!("state probe abandoned: constructed model/backend owners retained");
        }
    }
}
