// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit complete-state diagnostic, not an HTTP or throughput benchmark.
#[path = "glm53_partial_replay_parity/artifacts.rs"]
mod artifacts;
#[path = "glm53_partial_replay_parity/case.rs"]
mod case;
#[path = "glm53_partial_replay_parity/compare.rs"]
mod compare;
#[path = "glm53_partial_replay_parity/forced_policy.rs"]
mod forced_policy;
#[path = "glm53_partial_replay_parity/session.rs"]
mod session;
#[path = "glm53_partial_replay_parity/snapshots.rs"]
mod snapshots;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

fn json_file(path: &Path, value: &Value) -> Result<()> {
    artifacts::write_new(path, &serde_json::to_vec_pretty(value)?)
}
fn list(value: &str) -> Result<Vec<usize>> {
    let values = value
        .split(',')
        .map(|v| v.parse::<usize>().context("invalid case list integer"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !values.is_empty() && values.len() <= 32,
        "case list extent invalid"
    );
    Ok(values)
}
fn main() -> Result<()> {
    ensure!(
        std::env::var("ATLAS_GLM53_PARTIAL_WIDE_REPLAY").as_deref() == Ok("1"),
        "probe requires explicit ATLAS_GLM53_PARTIAL_WIDE_REPLAY=1"
    );
    ensure!(
        std::env::var("ATLAS_GLM53_DFLASH2_KV_PREFIX").as_deref() == Ok("0"),
        "probe requires explicit ATLAS_GLM53_DFLASH2_KV_PREFIX=0; cached KV is outside this gate"
    );
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 14,
        "required: --target PATH --draft PATH --corpus PATH --output PATH --positions CSV --rows CSV --repeats N"
    );
    let mut options = BTreeMap::new();
    for pair in args.chunks_exact(2) {
        ensure!(
            options.insert(pair[0].as_str(), pair[1].as_str()).is_none(),
            "duplicate option"
        );
    }
    let required = [
        "--target",
        "--draft",
        "--corpus",
        "--output",
        "--positions",
        "--rows",
        "--repeats",
    ];
    ensure!(
        options.len() == required.len() && required.iter().all(|key| options.contains_key(key)),
        "unexpected or missing option"
    );
    let target = PathBuf::from(options["--target"]);
    let draft = PathBuf::from(options["--draft"]);
    let corpus_path = PathBuf::from(options["--corpus"]);
    let output = PathBuf::from(options["--output"]);
    let positions = list(options["--positions"])?;
    let rows = list(options["--rows"])?;
    let repeats: usize = options["--repeats"].parse()?;
    ensure!((1..=3).contains(&repeats), "repeats must be1..3");
    ensure!(
        fs::metadata(&corpus_path)?.len() <= 1024 * 1024,
        "corpus JSON exceeds1MiB"
    );
    let raw = fs::read(&corpus_path)?;
    let corpus: Vec<u32> = serde_json::from_slice(&raw)?;
    ensure!(
        corpus.len() <= 4096 && corpus.iter().all(|&id| id < 154_880),
        "invalid bounded token corpus"
    );
    for &position in &positions {
        for &count in &rows {
            ensure!(
                position > 0
                    && (1..=8).contains(&count)
                    && position
                        .checked_add(8)
                        .is_some_and(|end| end <= 2047 && end <= corpus.len())
                    && position
                        .checked_add(count + 1)
                        .is_some_and(|end| end <= 2047 && end <= corpus.len()),
                "case exceeds staging/continuation capacity"
            );
        }
    }
    fs::create_dir(&output)?;
    json_file(
        &output.join("input.json"),
        &json!({"target":target,"draft":draft,"corpus":corpus_path,
        "corpus_sha256":compare::digest(&raw),"positions":positions,"rows":rows,"repeats":repeats,
        "partial_wide_replay":true,"kv_prefix":false,"natural_acceptance":false,"tokens":corpus}),
    )?;
    let mut session = session::Session::load(&target, &draft)?;
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
        let mut cases = Vec::new();
        let mut exact = true;
        for repeat in 0..repeats {
            for &start in &positions {
                for &count in &rows {
                    let name = format!("case-{:03}-p{start}-r{count}", cases.len());
                    let case_output = output.join(&name);
                    fs::create_dir(&case_output)?;
                    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        case::run_case(&mut session, &corpus, start, count, &case_output)
                    }));
                    let report = match attempt {
                        Ok(Ok(report)) => report,
                        other => {
                            let message = match other {
                                Ok(Err(error)) => format!("{error:#}"),
                                Err(_) => "case execution panicked".to_owned(),
                                Ok(Ok(_)) => unreachable!(),
                            };
                            json_file(
                                &case_output.join("error.json"),
                                &json!({"exact":false,
                                "repeat":repeat,"start":start,"rows":count,"error":message}),
                            )?;
                            bail!("case {name}: {message}");
                        }
                    };
                    json_file(&case_output.join("result.json"), &report)?;
                    exact &= report["exact"] == true;
                    let receipt = json!({"repeat":repeat,"start":start,"rows":count,"file":name,"exact":report["exact"]});
                    println!("{}", receipt);
                    cases.push(receipt);
                }
            }
        }
        Ok(
            json!({"exact":exact,"cases":cases,"qualification":"complete raw-state diagnostic only; no throughput or quality score"}),
        )
    }));
    let close = session.close();
    let report = match &run {
        Ok(Ok(report)) => report.clone(),
        Ok(Err(error)) => json!({"exact":false,"error":format!("{error:#}")}),
        Err(_) => json!({"exact":false,"error":"probe execution panicked"}),
    };
    let final_report = json!({"result":report,"close_ok":close.is_ok(),"close_error":close.as_ref().err().map(|e|format!("{e:#}"))});
    json_file(&output.join("result.json"), &final_report)?;
    close?;
    match run {
        Ok(Ok(report)) if report["exact"] == true => Ok(()),
        Ok(Err(error)) => Err(error),
        _ => bail!("state parity did not pass; all completed case receipts retained"),
    }
}
