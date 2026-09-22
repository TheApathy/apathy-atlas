// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use spark_model::model::glm53::{Glm53Dflash2Runtime, Glm53Exl3Model};
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, admit_glm53_exl3_files, load_glm53_exl3_store,
    materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const RESERVE_BYTES: usize = 8 * 1024 * 1024 * 1024;

fn main() -> Result<()> {
    let process_started = Instant::now();
    let target_root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .context("usage: glm53_dflash2_propose <target-exl3-dir> <dflash2-dir> [token]")?;
    let draft_root = std::env::args_os()
        .nth(2)
        .map(std::path::PathBuf::from)
        .context("missing DFlash2 directory")?;
    let tokens = std::env::args()
        .nth(3)
        .map(|value| {
            value
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<Vec<u32>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![151_331u32]);
    let files = admit_glm53_exl3_files(&target_root)?;
    eprintln!(
        "PHASE admitted_exl3 elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let backend: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend> =
        std::sync::Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let gpu: &dyn GpuBackend = backend.as_ref();
    eprintln!(
        "PHASE initialized_cuda elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    eprintln!(
        "PHASE loaded_exl3_store elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
    eprintln!(
        "PHASE built_target_catalog elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let native =
        materialize_glm53_exl3_native(&catalog, gpu).map_err(|error| anyhow::anyhow!("{error}"))?;
    eprintln!(
        "PHASE materialized_native elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let weights = Glm53Exl3TargetWeights::new(&catalog, &native)?;
    eprintln!(
        "PHASE assembled_target_weights elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    drop(catalog);
    let model = Glm53Exl3Model::new(backend, store, native, weights, 256)?;
    eprintln!(
        "PHASE constructed_target_model elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let mut draft = Glm53Dflash2Runtime::load(model.gpu(), &draft_root, model.dflash_lm_head())?;
    eprintln!(
        "PHASE loaded_dflash2 elapsed_seconds={:.6}",
        process_started.elapsed().as_secs_f64()
    );
    let stream = model.gpu().default_stream();
    let trace_prefill = std::env::var_os("ATLAS_GLM53_TRACE_PREFILL").is_some();
    let mut logits = None;
    for (position, &token) in tokens.iter().enumerate() {
        let current = model.decode_token(token, stream)?;
        draft.observe_target(&model, stream)?;
        if trace_prefill {
            let next = model.argmax_host(current)?;
            eprintln!("PREFILL position={position} input={token} argmax={next}");
        }
        logits = Some(current);
    }
    let logits = logits.context("GLM EXL3 prefill needs a token")?;
    if trace_prefill {
        let capture_bytes = 5 * 8192;
        let captures = model.gpu().alloc(capture_bytes)?;
        model.copy_dflash_capture_row(0, captures, stream)?;
        let mut bytes = vec![0u8; capture_bytes];
        model.gpu().copy_d2h(captures, &mut bytes)?;
        model.gpu().free(captures)?;
        for (slot, layer) in [4u32, 13, 23, 32, 41].into_iter().enumerate() {
            let values = bytes[slot * 8192..(slot + 1) * 8192]
                .chunks_exact(2)
                .map(|pair| half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
                .collect::<Vec<_>>();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let rms = (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32)
                .sqrt();
            let maxabs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
            eprintln!("CAPTURE layer={layer} rms={rms:.9} mean={mean:.9} maxabs={maxabs:.9}");
        }
    }
    let anchor = model.argmax_host(logits)?;
    let graph_probe = std::env::var_os("ATLAS_GLM53_K8_GRAPH_PROBE").is_some();
    let (proposed, proposal_seconds) = if graph_probe {
        draft.propose_graph_probe(&model, anchor, stream)?
    } else {
        let started = Instant::now();
        let proposed = draft.propose(&model, anchor, stream)?;
        (proposed, started.elapsed().as_secs_f64())
    };
    let predecessors = std::iter::once(anchor)
        .chain(proposed.iter().copied())
        .collect::<Vec<_>>();
    let verify_started = Instant::now();
    let (logits, decode_seconds) = if graph_probe {
        model.verify_tokens_full_graph_probe(&predecessors, stream)?
    } else {
        let logits = model.verify_tokens_full(&predecessors, stream)?;
        (logits, verify_started.elapsed().as_secs_f64())
    };
    let target_oracle = model.argmax_rows_host(logits, predecessors.len())?;
    let accepted_drafts = target_oracle
        .iter()
        .take(proposed.len())
        .zip(&proposed)
        .take_while(|(oracle, candidate)| oracle == candidate)
        .count();
    ensure!(
        accepted_drafts == proposed.len(),
        "wide verifier mismatch: oracle={target_oracle:?} proposed={proposed:?}"
    );
    let accepted_tokens = 1 + accepted_drafts;
    let cycle_seconds = proposal_seconds + decode_seconds;
    let bonus = target_oracle
        .get(proposed.len())
        .copied()
        .context("wide verifier did not produce the bonus-token oracle")?;
    println!(
        "DFLASH2 context_tokens={} anchor={anchor} proposed={proposed:?} proposal_seconds={proposal_seconds:.6} proposal_tok_s={:.6}",
        tokens.len(),
        proposed.len() as f64 / proposal_seconds
    );
    println!(
        "VERIFY mode={} rows={} target_oracle={target_oracle:?} bonus={bonus} accepted_drafts={accepted_drafts} accepted_tokens={accepted_tokens} decode_seconds={decode_seconds:.6} accepted_tok_s={:.6} cycle_seconds={cycle_seconds:.6} cycle_tok_s={:.6}",
        if graph_probe {
            "layer-major-k8-graph-probe"
        } else {
            "layer-major-k8"
        },
        predecessors.len(),
        accepted_tokens as f64 / decode_seconds,
        accepted_tokens as f64 / cycle_seconds
    );
    draft.free(model.gpu())?;
    model.free()?;
    println!("RESULT: PASS physical_drafts=7");
    Ok(())
}
