// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit arithmetic experiment; original outputs are never replaced.
use super::{
    artifacts, projected, projected_pair, projection_choice::ProjectionChoice,
    session::ProbeSession,
};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use spark_model::model::glm53::ProbeCapture;
use std::path::Path;

pub fn triple(
    session: &mut ProbeSession,
    root: &Path,
    label: &str,
    with_projected_target: bool,
    previous_projected_target: &mut Option<Vec<u8>>,
    choice: ProjectionChoice,
) -> Result<(bool, [u32; 7])> {
    let [original_mode, full_mode, cached_mode] = choice.modes();
    let (context, layout) = session.model.installed_diagnostic_state()?;
    let reference = session
        .reference
        .as_ref()
        .context("missing independent reference")?;
    ensure!(
        reference.context_tokens() == context,
        "three-arm capture cursor mismatch"
    );
    let anchor = session.anchor()?;
    let layout = if with_projected_target {
        layout.with_projected_target(context, 4096)?
    } else {
        layout
    };
    let reference_layout = reference.probe_layout()?;
    let reference_layout = if with_projected_target {
        reference_layout.with_projected_target(context, 4096)?
    } else {
        reference_layout
    };
    session.candidate_capture = Some(ProbeCapture::new(layout.clone(), context, session.stream)?);
    session.original_capture = Some(ProbeCapture::new(
        reference_layout.clone(),
        context,
        session.stream,
    )?);
    session.reference_capture = Some(ProbeCapture::new(
        reference_layout,
        context,
        session.stream,
    )?);
    let directory = root.join(label);
    std::fs::create_dir(&directory)?;
    artifacts::json_file(
        &directory.join("input.json"),
        &json!({
            "context":context,"anchor":anchor,
            "tokens":session.seq.as_ref().context("missing sequence")?.tokens,
            "with_projected_target":with_projected_target,
            "projection_family":choice.name(),
            "actual_modes":choice.modes().map(|mode|mode.as_str()),
            "modes":["original-full","stable-full","stable-cached"],
            "execution_order":["stable-cached","original-full","stable-full"]
        }),
    )?;
    let candidate = session.model.propose_installed_diagnostic_mode(
        anchor,
        session.stream,
        cached_mode,
        session.candidate_capture.as_mut().unwrap(),
    );
    let original = if candidate.is_ok() {
        reference.propose_diagnostic(
            &session.model,
            anchor,
            session.stream,
            original_mode,
            session.original_capture.as_mut().unwrap(),
        )
    } else {
        Err(anyhow::anyhow!(
            "original arm not run after candidate failure"
        ))
    };
    // Always leave stable-full last on this runtime. It overwrites used slots;
    // unused slots retain the same stable-family history as the candidate.
    let stable = if original.is_ok() {
        reference.propose_diagnostic(
            &session.model,
            anchor,
            session.stream,
            full_mode,
            session.reference_capture.as_mut().unwrap(),
        )
    } else {
        Err(anyhow::anyhow!("stable arm not run after original failure"))
    };
    for (name, capture) in [
        ("stable-cached", session.candidate_capture.as_ref().unwrap()),
        ("original-full", session.original_capture.as_ref().unwrap()),
        ("stable-full", session.reference_capture.as_ref().unwrap()),
    ] {
        artifacts::dump(&directory.join(name), capture)?;
    }
    if candidate.is_err() || original.is_err() || stable.is_err() {
        artifacts::json_file(
            &directory.join("comparison.json"),
            &json!({
                "stable_cache_exact":false,"original_baseline_exact":false,
                "projection_family":choice.name(),
                "candidate_error":candidate.as_ref().err().map(|e|format!("{e:#}")),
                "original_error":original.as_ref().err().map(|e|format!("{e:#}")),
                "stable_error":stable.as_ref().err().map(|e|format!("{e:#}")),
                "quality_qualified":false,"speed_qualified":false
            }),
        )?;
    }
    let candidate = candidate?;
    let original = original?;
    let stable = stable?;
    let mut report = artifacts::compare_three(
        &layout,
        session.original_capture.as_ref().unwrap(),
        session.reference_capture.as_ref().unwrap(),
        session.candidate_capture.as_ref().unwrap(),
    )?;
    report["returned_ids"] =
        json!({"original":original,"stable_full":stable,"stable_cached":candidate});
    report["projection_family"] = json!(choice.name());
    report["original_pool_history"] = json!(
        "original arithmetic over used committed/noise slots; unused tail inherits the shared reference runtime's preceding stable-full history"
    );
    report["stable_ids_exact"] = json!(stable == candidate);
    let mut exact = report["stable_cache_exact"]
        .as_bool()
        .context("missing stable cache receipt")?
        && stable == candidate;
    if with_projected_target {
        let original_inputs = projected::compare_inputs(
            projected_pair(session.original_capture.as_ref().unwrap())?,
            projected_pair(session.reference_capture.as_ref().unwrap())?,
            previous_projected_target.as_deref(),
            8192,
        )?;
        let stable_inputs = projected::compare_inputs(
            projected_pair(session.reference_capture.as_ref().unwrap())?,
            projected_pair(session.candidate_capture.as_ref().unwrap())?,
            previous_projected_target.as_deref(),
            8192,
        )?;
        let inputs_exact = original_inputs["exact"] == true && stable_inputs["exact"] == true;
        artifacts::json_file(
            &directory.join("projected-inputs.json"),
            &json!({
                "exact":inputs_exact,"original_vs_stable":original_inputs,"stable_vs_cached":stable_inputs
            }),
        )?;
        report["projected_inputs_exact"] = json!(inputs_exact);
        exact &= inputs_exact;
        if inputs_exact {
            let raw = projected_pair(session.reference_capture.as_ref().unwrap())?.before;
            let mut retained = Vec::new();
            retained.try_reserve_exact(raw.len())?;
            retained.extend_from_slice(raw);
            *previous_projected_target = Some(retained);
        }
    }
    report["stable_cache_exact"] = json!(exact);
    artifacts::json_file(&directory.join("comparison.json"), &report)?;
    eprintln!(
        "TRIPLE {label} context={context} stable_cache_exact={exact} original_exact={}",
        report["original_baseline_exact"]
    );
    Ok((exact, candidate))
}
