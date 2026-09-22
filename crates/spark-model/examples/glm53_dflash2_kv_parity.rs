// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit raw gate or proposal-latency diagnostic. No serving speed claims.
#[path = "glm53_dflash2_kv_parity/artifacts.rs"]
mod artifacts;
#[path = "glm53_dflash2_kv_parity/projected.rs"]
mod projected;
#[path = "glm53_dflash2_kv_parity/projection_choice.rs"]
mod projection_choice;
#[path = "glm53_dflash2_kv_parity/session.rs"]
mod session;
#[path = "glm53_dflash2_kv_parity/sweep.rs"]
mod sweep;
#[path = "glm53_dflash2_kv_parity/three_way.rs"]
mod three_way;
#[path = "glm53_dflash2_kv_parity/timing.rs"]
mod timing;
#[path = "glm53_dflash2_kv_parity/timing_observer.rs"]
mod timing_observer;
#[path = "glm53_dflash2_kv_parity/timing_plan.rs"]
mod timing_plan;
use anyhow::{Context, Result, bail, ensure};
use projection_choice::ProjectionChoice;
use serde_json::{Value, json};
use session::ProbeSession;
use spark_model::model::glm53::{Dflash2ProbeMode, ProbeCapture, ProbeLayout, ProbeStage};
use std::path::{Path, PathBuf};

const CONTEXTS: &[usize] = &[
    1, 1, 2, 4, 7, 11, 15, 16, 17, 22, 28, 35, 43, 60, 2039, 2047,
];

fn bounded_json(path: &Path) -> Result<Value> {
    ensure!(
        std::fs::metadata(path)?.len() <= 1024 * 1024,
        "JSON input exceeds1MiB"
    );
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn tuning_environment() -> Result<Value> {
    let mut result = serde_json::Map::new();
    for (key, value) in std::env::vars_os() {
        let Some(key) = key.to_str() else {
            continue;
        };
        if !key.starts_with("ATLAS_")
            && !key.starts_with("SPARK_")
            && !key.starts_with("NCCL_")
            && !["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS"].contains(&key)
        {
            continue;
        }
        let upper = key.to_ascii_uppercase();
        ensure!(
            ![
                "AUTH",
                "SECRET",
                "PASSWORD",
                "CREDENTIAL",
                "API_KEY",
                "ACCESS_TOKEN"
            ]
            .iter()
            .any(|v| upper.contains(v)),
            "sensitive tuning namespace variable rejected"
        );
        let value = value
            .into_string()
            .map_err(|_| anyhow::anyhow!("tuning environment must be UTF-8"))?;
        result.insert(key.into(), value.into());
    }
    Ok(Value::Object(result))
}
fn projected_pair(capture: &ProbeCapture) -> Result<projected::InputPair<'_>> {
    let frames = capture.frames()?;
    let before = frames.first().context("missing projected before frame")?;
    let after = frames.last().context("missing projected after frame")?;
    ensure!(
        before.stage == ProbeStage::ProjectedTargetBefore
            && after.stage == ProbeStage::ProjectedTargetAfter,
        "projected input frames were not explicitly selected"
    );
    Ok(projected::InputPair {
        before: &before.bytes,
        after: &after.bytes,
    })
}

fn pair(
    session: &mut ProbeSession,
    root: &Path,
    label: &str,
    with_projected_target: bool,
    previous_projected_target: &mut Option<Vec<u8>>,
) -> Result<(bool, [u32; 7])> {
    let (context, layout) = session.model.installed_diagnostic_state()?;
    let reference = session
        .reference
        .as_ref()
        .context("missing independent reference")?;
    ensure!(
        reference.context_tokens() == context,
        "target/candidate/reference capture cursors differ"
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
    session.reference_capture = Some(ProbeCapture::new(
        reference_layout,
        context,
        session.stream,
    )?);
    let directory = root.join(label);
    std::fs::create_dir(&directory)?;
    artifacts::json_file(
        &directory.join("input.json"),
        &json!({"context":context,"anchor":anchor,
        "tokens":session.seq.as_ref().context("missing sequence")?.tokens,
        "with_projected_target":with_projected_target,
        "candidate_mode":Dflash2ProbeMode::CachedPrefix.as_str(),
        "reference_mode":Dflash2ProbeMode::FullRecompute.as_str()}),
    )?;
    let candidate = session.model.propose_installed_diagnostic(
        anchor,
        session.stream,
        session.candidate_capture.as_mut().unwrap(),
    );
    let reference_result = if candidate.is_ok() {
        reference.propose_diagnostic(
            &session.model,
            anchor,
            session.stream,
            Dflash2ProbeMode::FullRecompute,
            session.reference_capture.as_mut().unwrap(),
        )
    } else {
        Err(anyhow::anyhow!(
            "reference not run after candidate runtime/observer failure"
        ))
    };
    // Persist every safely retained frame before returning either mismatch or
    // runtime failure. Pending readback bytes are never exposed as a frame.
    artifacts::dump(
        &directory.join("candidate"),
        session.candidate_capture.as_ref().unwrap(),
    )?;
    artifacts::dump(
        &directory.join("reference"),
        session.reference_capture.as_ref().unwrap(),
    )?;
    let report = match (&candidate, &reference_result) {
        (Ok(candidate), Ok(reference)) => {
            let mut compared = artifacts::compare(
                &layout,
                session.reference_capture.as_ref().unwrap(),
                session.candidate_capture.as_ref().unwrap(),
            )?;
            compared["candidate_ids"] = json!(candidate);
            compared["reference_ids"] = json!(reference);
            compared["ids_exact"] = json!(candidate == reference);
            if with_projected_target {
                let reference = projected_pair(session.reference_capture.as_ref().unwrap())?;
                let candidate = projected_pair(session.candidate_capture.as_ref().unwrap())?;
                let inputs = projected::compare_inputs(
                    reference,
                    candidate,
                    previous_projected_target.as_deref(),
                    4096 * 2,
                )?;
                artifacts::json_file(&directory.join("projected-inputs.json"), &inputs)?;
                let input_exact = inputs["exact"]
                    .as_bool()
                    .context("missing input identity receipt")?;
                compared["projected_inputs_exact"] = json!(input_exact);
                if !input_exact {
                    compared["exact"] = json!(false);
                } else {
                    let raw = projected_pair(session.reference_capture.as_ref().unwrap())?.before;
                    let mut retained = Vec::new();
                    retained.try_reserve_exact(raw.len())?;
                    retained.extend_from_slice(raw);
                    *previous_projected_target = Some(retained);
                }
            }
            compared
        }
        _ => {
            json!({"exact":false,"candidate_error":candidate.as_ref().err().map(|e|format!("{e:#}")),
            "reference_error":reference_result.as_ref().err().map(|e|format!("{e:#}"))})
        }
    };
    artifacts::json_file(&directory.join("comparison.json"), &report)?;
    let candidate = candidate?;
    reference_result?;
    let exact = report["exact"].as_bool().context("missing exact receipt")?
        && report["ids_exact"].as_bool().unwrap_or(false);
    eprintln!("PAIR {label} context={context} exact={exact}");
    Ok((exact, candidate))
}

fn selected_pair(
    session: &mut ProbeSession,
    root: &Path,
    label: &str,
    with_projected_target: bool,
    previous_projected_target: &mut Option<Vec<u8>>,
    choice: Option<ProjectionChoice>,
) -> Result<(bool, [u32; 7])> {
    if let Some(choice) = choice {
        three_way::triple(
            session,
            root,
            label,
            with_projected_target,
            previous_projected_target,
            choice,
        )
    } else {
        pair(
            session,
            root,
            label,
            with_projected_target,
            previous_projected_target,
        )
    }
}

fn run(
    session: &mut ProbeSession,
    corpus: &[u32],
    contexts: &[usize],
    root: &Path,
    with_projected_target: bool,
    choice: Option<ProjectionChoice>,
) -> Result<bool> {
    let reference_layout = session
        .reference
        .as_ref()
        .context("missing reference")?
        .probe_layout()?;
    let (_, candidate_layout) = session.model.installed_diagnostic_state()?;
    sweep::admit(
        &candidate_layout,
        &reference_layout,
        contexts,
        with_projected_target,
    )?;
    // Retain only the completed projected input, not another full 18-frame set.
    let mut previous_projected_target = None;
    for (index, &context) in contexts.iter().enumerate() {
        session.advance(corpus, context)?;
        if !selected_pair(
            session,
            root,
            &format!("case{index:02}-context{context}"),
            with_projected_target,
            &mut previous_projected_target,
            choice,
        )?
        .0
        {
            return Ok(false);
        }
    }
    session.reset(corpus[0])?;
    previous_projected_target = None;
    let (exact, drafts) = selected_pair(
        session,
        root,
        "reset-before-policy",
        with_projected_target,
        &mut previous_projected_target,
        choice,
    )?;
    if !exact {
        return Ok(false);
    }
    let accepted = session.policy_case(drafts, true)?;
    artifacts::json_file(&root.join("forced-full-accept.json"), &accepted)?;
    let (exact, drafts) = selected_pair(
        session,
        root,
        "after-forced-full-accept",
        with_projected_target,
        &mut previous_projected_target,
        choice,
    )?;
    if !exact {
        return Ok(false);
    }
    let rejected = session.policy_case(drafts, false)?;
    artifacts::json_file(&root.join("forced-first-reject.json"), &rejected)?;
    if !selected_pair(
        session,
        root,
        "after-forced-first-reject",
        with_projected_target,
        &mut previous_projected_target,
        choice,
    )?
    .0
    {
        return Ok(false);
    }
    session.reset(corpus[0])?;
    previous_projected_target = None;
    Ok(selected_pair(
        session,
        root,
        "reset-after-policy",
        with_projected_target,
        &mut previous_projected_target,
        choice,
    )?
    .0)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let (mut with_projected_target, mut stable_projection) = (false, false);
    let mut proposal_timing = false;
    let mut gemv_projection = false;
    while let Some(flag) = args.last().and_then(|v| v.to_str()) {
        match flag {
            "--gemv-projection" => {
                ensure!(!gemv_projection, "duplicate GEMV projection flag");
                gemv_projection = true;
            }
            "--proposal-timing" => {
                ensure!(!proposal_timing, "duplicate proposal-timing flag");
                proposal_timing = true;
            }
            "--with-projected-target" => {
                ensure!(!with_projected_target, "duplicate projected-input flag");
                with_projected_target = true;
            }
            "--stable-projection" => {
                ensure!(!stable_projection, "duplicate stable-projection flag");
                stable_projection = true;
            }
            _ => break,
        }
        args.pop();
    }
    ensure!(
        !proposal_timing || (!stable_projection && !with_projected_target),
        "timing and raw capture flags are mutually exclusive"
    );
    let choice = ProjectionChoice::select(
        stable_projection,
        gemv_projection,
        proposal_timing,
        with_projected_target,
    )?;
    ensure!(
        (5..=6).contains(&args.len()),
        "usage: glm53_dflash2_kv_parity <target> <draft> <tokens.json> <fresh-output> <provenance.json> [contexts-csv] [--with-projected-target] [--stable-projection | --gemv-projection] OR [--proposal-timing [--gemv-projection]]"
    );
    let target = PathBuf::from(&args[0]).canonicalize()?;
    let draft = PathBuf::from(&args[1]).canonicalize()?;
    let input = PathBuf::from(&args[2]).canonicalize()?;
    let output = PathBuf::from(&args[3]);
    let provenance_path = PathBuf::from(&args[4]).canonicalize()?;
    let corpus: Vec<u32> = serde_json::from_value(bounded_json(&input)?)?;
    let contexts = if let Some(raw) = args.get(5) {
        raw.to_str()
            .context("contexts must be UTF-8")?
            .split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        CONTEXTS.to_vec()
    };
    ensure!(
        !contexts.is_empty()
            && contexts.len() <= 64
            && contexts.windows(2).all(|v| v[0] <= v[1])
            && contexts.iter().all(|v| (1..=2047).contains(v)),
        "invalid explicit context sweep"
    );
    ensure!(
        !corpus.is_empty()
            && corpus.len() <= 2047
            && corpus.len() >= *contexts.last().unwrap()
            && corpus.iter().all(|v| *v < 154_880),
        "input token corpus does not cover admitted contexts"
    );
    let provided = bounded_json(&provenance_path)?;
    let object = provided
        .as_object()
        .context("provenance must be an object")?;
    let keys = [
        "source_sha256",
        "binary_sha256",
        "target_config_sha256",
        "draft_config_sha256",
    ];
    ensure!(
        object.len() == keys.len() && keys.iter().all(|k| object.contains_key(*k)),
        "provenance requires exactly four identity hashes"
    );
    for key in keys {
        let hash = provided[key]
            .as_str()
            .context("provenance hash must be string")?;
        ensure!(
            hash.len() == 64 && hash.bytes().all(|v| v.is_ascii_hexdigit()),
            "invalid provenance hash"
        );
    }
    for (key, path) in [
        ("binary_sha256", std::env::current_exe()?),
        ("target_config_sha256", target.join("config.json")),
        ("draft_config_sha256", draft.join("config.json")),
    ] {
        ensure!(
            provided[key].as_str() == Some(artifacts::sha256(&path)?.as_str()),
            "provenance identity mismatch: {key}"
        );
    }
    let environment = tuning_environment()?;
    std::fs::create_dir(&output).context("output must be a fresh directory")?;
    artifacts::json_file(
        &output.join("admission.json"),
        &json!({"schema":"atlas.glm53.kv-parity.v1",
        "identities":provided,"input_sha256":artifacts::sha256(&input)?,"target":target,"draft":draft,
        "contexts":contexts,"environment":environment,"source_hash_basis":"root-supplied frozen-source manifest",
        "with_projected_target":with_projected_target,
        "stable_projection":stable_projection,
        "projection_family":choice.map(ProjectionChoice::name),
        "proposal_timing":proposal_timing,
        "loader_boundary":"legacy constructors before return are not ownership-qualified by this probe"}),
    )?;
    artifacts::json_file(&output.join("tokens.json"), &json!(corpus))?;
    let mut session = ProbeSession::load(&target, &draft)?;
    let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if proposal_timing {
            return timing::run(
                &mut session,
                &corpus,
                &contexts,
                &output,
                choice.context("timing family missing")?,
            );
        }
        run(
            &mut session,
            &corpus,
            &contexts,
            &output,
            with_projected_target,
            choice,
        )
    }));
    let executed = match executed {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!(
            "probe execution panicked; session retained for drain"
        )),
    };
    let cleanup = session.close();
    let success = matches!(executed, Ok(true)) && cleanup.is_ok();
    let raw_exact = !proposal_timing && matches!(executed, Ok(true));
    artifacts::json_file(
        &output.join("result.json"),
        &json!({"status":if success {if proposal_timing {"TIMING_COMPLETE"} else {"EXACT"}} else {"FAIL"},
        "raw_exact":raw_exact,"cleanup":cleanup.as_ref().err().map(|e|format!("{e:#}")),
        "error":executed.as_ref().err().map(|e|format!("{e:#}")),"selected_contexts":contexts,
        "with_projected_target":with_projected_target,
        "stable_projection":stable_projection,
        "projection_family":choice.map(ProjectionChoice::name),
        "proposal_timing":proposal_timing,
        "raw_exact_scope":if proposal_timing {"not measured; stage metadata and returned IDs only"} else if choice.is_some() {"stable-full versus stable-cached only; original drift separately retained"} else {"original-full versus original-cached"},
        "stopped_on_first_mismatch":matches!(executed, Ok(false)),"natural_partial_acceptance_qualified":false,
        "speed_qualified":false,"full_production_qualified":false}),
    )?;
    cleanup?;
    ensure!(
        executed?,
        "raw mismatch; complete retained pair is not exact-qualified"
    );
    if !success {
        bail!("probe failed");
    }
    Ok(())
}
