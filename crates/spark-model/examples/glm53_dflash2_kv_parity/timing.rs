// SPDX-License-Identifier: AGPL-3.0-only
//! Same-context proposal wall latency; not serving, prefill or quality evidence.
use super::{
    artifacts, projection_choice::ProjectionChoice, session::ProbeSession, sweep,
    timing_observer::TimingObserver, timing_plan,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_model::model::glm53::Dflash2ProbeMode as Mode;
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn sample(session: &ProbeSession, mode: Mode, context: u32, anchor: u32) -> Result<Value> {
    let (installed_context, layout) = session.model.installed_diagnostic_state()?;
    let reference = session
        .reference
        .as_ref()
        .context("missing timing reference")?;
    ensure!(
        installed_context == context && reference.context_tokens() == context,
        "timing committed contexts changed"
    );
    let mut observer = TimingObserver::new(layout, context, session.stream)?;
    // Final capture ingestion after advance and anchor readback is outside time.
    session.model.gpu().synchronize(session.stream)?;
    let unix_start_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .to_string();
    let started = Instant::now();
    let result = match mode {
        Mode::StableCachedProjection | Mode::StableGemvCachedProjection => session
            .model
            .propose_installed_diagnostic_mode(anchor, session.stream, mode, &mut observer),
        Mode::FullRecompute | Mode::StableFullProjection | Mode::StableGemvFullProjection => {
            reference.propose_diagnostic(
                &session.model,
                anchor,
                session.stream,
                mode,
                &mut observer,
            )
        }
        Mode::CachedPrefix => unreachable!("original cached mode not in timing schedule"),
    };
    let elapsed = started.elapsed();
    let unix_end_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .to_string();
    // Both successful diagnostic modes finish same-stream work before return.
    let observed = observer.finish();
    Ok(json!({
        "mode":mode.as_str(),"context":context,"anchor":anchor,
        "elapsed_ms":elapsed.as_secs_f64()*1000.0,
        "unix_start_ns":unix_start_ns,"unix_end_ns":unix_end_ns,
        "ids":result.as_ref().ok(),"error":result.as_ref().err().map(|e|format!("{e:#}")),
        "observed_stages":observed.as_ref().ok(),
        "observer_error":observed.as_ref().err().map(|e|format!("{e:#}")),
        "successful":result.is_ok()&&observed.is_ok()
    }))
}

fn triple(
    session: &ProbeSession,
    order: [Mode; 3],
    context: u32,
    anchor: u32,
    root: &Path,
    label: &str,
    class: &str,
    previous: u32,
    choice: ProjectionChoice,
) -> Result<Value> {
    let directory = root.join(label);
    std::fs::create_dir(&directory)?;
    let (cache_class, expected_rows) = timing_plan::workload(previous, context)?;
    let mut samples = Vec::new();
    for mode in order {
        let sample = sample(session, mode, context, anchor)?;
        // Preserve completed/error receipts before any following arm can run.
        artifacts::json_file(&directory.join(format!("{}.json", mode.as_str())), &sample)?;
        ensure!(
            sample["successful"] == true,
            "timing proposal/observer failed; retained receipt in {}",
            directory.display()
        );
        samples.push(sample);
    }
    let ids =
        |mode: Mode| samples.iter().find(|v| v["mode"] == mode.as_str()).unwrap()["ids"].clone();
    let [original_mode, full_mode, cached_mode] = choice.modes();
    let stable_ids_exact = ids(full_mode) == ids(cached_mode);
    let report = json!({
        "context":context,"anchor":anchor,"class":class,"label":label,
        "projection_family":choice.name(),
        "order":order.map(Mode::as_str),"previous_successful_cached_context":previous,
        "cache_workload":cache_class,"expected_cached_projection_rows":expected_rows,
        "projection_rows_basis":"harness successful-call history and frozen KvTail source, not device counter or cache authority",
        "stable_ids_exact":stable_ids_exact,
        "original_ids_exact":ids(original_mode)==ids(full_mode),
        "samples":samples
    });
    artifacts::json_file(&directory.join("triple.json"), &report)?;
    ensure!(
        stable_ids_exact,
        "stable timing arms returned different IDs; no timing qualification"
    );
    Ok(report)
}

fn mode_ids(report: &Value, mode: Mode) -> &Value {
    &report["samples"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["mode"] == mode.as_str())
        .unwrap()["ids"]
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

pub fn run(
    session: &mut ProbeSession,
    corpus: &[u32],
    contexts: &[usize],
    root: &Path,
    choice: ProjectionChoice,
) -> Result<bool> {
    let (_, layout) = session.model.installed_diagnostic_state()?;
    let reference_layout = session
        .reference
        .as_ref()
        .context("missing reference")?
        .probe_layout()?;
    sweep::admit(&layout, &reference_layout, contexts, false)?;
    let mut history = 0u32;
    let mut all = Vec::new();
    for (index, &end) in contexts.iter().enumerate() {
        session.advance(corpus, end)?;
        let context = u32::try_from(end)?;
        let anchor = session.anchor()?;
        let first = triple(
            session,
            choice.map_order(timing_plan::order(index))?,
            context,
            anchor,
            root,
            &format!("context-{index:02}-{end}-first"),
            "first-after-setup-single-sample",
            history,
            choice,
        )?;
        history = context;
        // Do not move this warmup before first: that would consume the new tail.
        let warmup = triple(
            session,
            choice.map_order(timing_plan::order(index + 1))?,
            context,
            anchor,
            root,
            &format!("context-{index:02}-{end}-warmup"),
            "excluded-warmup",
            history,
            choice,
        )?;
        let modes = choice.modes();
        for mode in modes {
            ensure!(
                mode_ids(&first, mode) == mode_ids(&warmup, mode),
                "same-context warmup ID drift"
            );
        }
        let mut hot = Vec::new();
        for trial in 0..12 {
            let report = triple(
                session,
                choice.map_order(timing_plan::order(trial))?,
                context,
                anchor,
                root,
                &format!("context-{index:02}-{end}-hot-{trial:02}"),
                "hot-repeat-one-row",
                history,
                choice,
            )?;
            for mode in modes {
                ensure!(
                    mode_ids(&first, mode) == mode_ids(&report, mode),
                    "same-context repeat ID drift"
                );
            }
            hot.push(report);
        }
        let medians = modes.map(|mode| {
            let samples = hot
                .iter()
                .flat_map(|v| v["samples"].as_array().unwrap())
                .filter(|v| v["mode"] == mode.as_str())
                .map(|v| v["elapsed_ms"].as_f64().unwrap())
                .collect();
            json!({"mode":mode.as_str(),"n":hot.len(),"median_ms":median(samples)})
        });
        let report = json!({"context":context,"projection_family":choice.name(),"first":first,"warmup":warmup,"hot":hot,"hot_medians":medians});
        artifacts::json_file(
            &root.join(format!("context-{index:02}-{end}-summary.json")),
            &report,
        )?;
        eprintln!("TIMING context={context} hot_medians={}", json!(medians));
        all.push(report);
    }
    artifacts::json_file(
        &root.join("timing-summary.json"),
        &json!({
            "schema":"atlas.glm53.proposal-timing.v1","contexts":all,
            "projection_family":choice.name(),
            "metric":"fenced diagnostic proposal host-plus-GPU wall milliseconds",
            "excluded":"target advancement, capture ingestion, anchor selection, observer allocation, raw activation transfers and artifact I/O",
            "included":"diagnostic admission, locks, 18 metadata callbacks, 4-byte anchor upload, 28-byte ID and 4-byte status readbacks, completion fences; full-mode metadata resets and installed-candidate mutex",
            "first_sample_limit":"one sample per context, not a median or balanced same-context new-tail benchmark",
            "hot_order":"two cycles of every permutation of three arms; 12 samples per arm after one excluded warmup",
            "original_history_limit":"original and stable full share external runtime unused KV tails; no inherited whole-pool raw parity claim",
            "raw_exact_qualified":false,"quality_qualified":false,"decode_speed_qualified":false,
            "prefill_speed_qualified":false,"isolation_qualified":false,
            "isolation_requirement":"parent measured-interval observations must independently admit samples; this child cannot certify isolation"
        }),
    )?;
    Ok(true)
}
