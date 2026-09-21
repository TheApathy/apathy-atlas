// SPDX-License-Identifier: AGPL-3.0-only
use crate::{
    contract::{self, Grid, ReferenceMode},
    io, pins, weight,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

pub fn reference_hash(v: &Value, case: &str, stage: &str, input: &str) -> Result<String> {
    let cases: Vec<_> = v["cases"]
        .as_array()
        .context("reference cases")?
        .iter()
        .filter(|c| c["case"].as_str() == Some(case))
        .collect();
    ensure!(cases.len() == 1, "reference case not unique");
    let ops: Vec<_> = cases[0]["operators"]
        .as_array()
        .context("reference operators")?
        .iter()
        .filter(|o| o["stage"].as_str() == Some(stage))
        .collect();
    ensure!(ops.len() == 1, "reference stage not unique");
    let op = ops[0];
    ensure!(
        op["dtype"].as_str() == Some("torch.bfloat16")
            && op["shared_native_inputs"] == json!([input]),
        "reference dtype/shared-input contract changed"
    );
    let sha = op["reference_sha256"].as_str().context("reference hash")?;
    ensure!(contract::sha_syntax(sha), "reference SHA syntax");
    Ok(sha.to_owned())
}
pub fn validate_build(v: &Value, source_sha: &str, ptx_sha: &str) -> Result<()> {
    ensure!(v["schema"] == "atlas-dsv-p1-angle-build-v1", "build schema");
    ensure!(
        v["source_sha256"] == source_sha && v["ptx_sha256"] == ptx_sha,
        "build source/PTX changed"
    );
    ensure!(
        v["compiler_sha256"]
            .as_str()
            .is_some_and(contract::sha_syntax),
        "compiler SHA missing"
    );
    ensure!(
        v["compiler_version"]
            .as_str()
            .is_some_and(|s| s.contains("release 13.")),
        "CUDA 13 build required"
    );
    ensure!(
        v["flags"]
            == json!([
                "-ptx",
                "-O3",
                "-arch=sm_121f",
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true"
            ]),
        "exact reviewed CUDA flags required"
    );
    Ok(())
}
fn record_pinned(path: &Path, sha: &str, cap: usize, files: &mut Vec<Value>) -> Result<Vec<u8>> {
    let raw = io::pinned(path, sha, cap)?;
    files.push(io::receipt(path, &raw)?);
    Ok(raw)
}
pub fn inspect(model: &Path, corpus: &Path, out: &Path) -> Result<Value> {
    let model = io::absolute(model)?;
    let corpus = io::absolute(corpus)?;
    let bench = io::absolute(Path::new(env!("CARGO_MANIFEST_DIR")))?;
    let repo = io::absolute(&bench.join("../.."))?;
    let mut files = Vec::new();
    let config = record_pinned(
        &model.join("config.json"),
        pins::CONFIG_SHA,
        4 * 1024 * 1024,
        &mut files,
    )?;
    let index = io::json(&record_pinned(
        &model.join("model.safetensors.index.json"),
        pins::INDEX_SHA,
        io::MAX_FILE,
        &mut files,
    )?)?;
    let mut reports = Vec::new();
    for (name, sha) in pins::CORPUS_FILES {
        reports.push(io::json(&record_pinned(
            &corpus.join(name),
            sha,
            4 * 1024 * 1024,
            &mut files,
        )?)?);
    }
    let native = &reports[0];
    let checkpoint = &native["checkpoint"];
    ensure!(
        checkpoint["model_revision"] == pins::MODEL_REV
            && checkpoint["config_sha256"] == pins::CONFIG_SHA
            && checkpoint["index_sha256"] == pins::INDEX_SHA,
        "retained checkpoint identity"
    );
    contract::validate_reference(&reports[2], ReferenceMode::Default)?;
    contract::validate_reference(&reports[3], ReferenceMode::Full)?;
    record_pinned(
        Path::new(pins::OFFICIAL_PATH),
        pins::OFFICIAL_SHA,
        1024 * 1024,
        &mut files,
    )?;
    for (name, sha) in pins::REFERENCE_SCRIPTS {
        record_pinned(Path::new(name), sha, 1024 * 1024, &mut files)?;
    }
    for (name, sha) in pins::PRODUCTION_FILES {
        record_pinned(&repo.join(name), sha, 1024 * 1024, &mut files)?;
    }
    for (name, sha) in pins::PTX_FILES {
        record_pinned(
            &Path::new(pins::PTX_ROOT).join(name),
            sha,
            8 * 1024 * 1024,
            &mut files,
        )?;
    }
    let mut sources = Vec::new();
    for name in pins::SOURCE_FILES {
        let p = bench.join(name);
        sources.push(io::receipt(&p, &io::read(&p, 1024 * 1024)?)?);
    }
    let mut cases = Vec::new();
    for (i, (h, w)) in [(3, 3), (4, 5), (54, 54)].into_iter().enumerate() {
        let g = Grid::new(h, w)?;
        let name = g.name();
        let dir = corpus.join(format!("{name}-stages"));
        let manifest = io::json(&record_pinned(
            &dir.join("manifest.json"),
            pins::STAGE_MANIFESTS[i],
            1024 * 1024,
            &mut files,
        )?)?;
        ensure!(
            manifest["grid"] == json!([h, w]) && manifest["output_byte_equal"] == true,
            "retained stage grid/repeat"
        );
        let mut stages = serde_json::Map::new();
        for (stage, shape) in [
            ("qkv", [g.patches(), 3072]),
            ("query", [16 * g.patches(), 64]),
            ("key", [16 * g.patches(), 64]),
            ("value", [1024, g.patches()]),
            ("swiglu", [g.patches(), 2816]),
            ("fc2", [g.patches(), 1024]),
        ] {
            if i != 1 && matches!(stage, "swiglu" | "fc2") {
                continue;
            }
            let full = format!("block-00-{stage}");
            let entries: Vec<_> = manifest["stages"]
                .as_array()
                .context("native stages")?
                .iter()
                .filter(|s| s["name"].as_str() == Some(full.as_str()))
                .collect();
            ensure!(entries.len() == 1, "native stage not unique");
            let entry = entries[0];
            contract::check_stage(entry, &full, shape)?;
            let raw = record_pinned(
                &dir.join(format!("{full}.bf16")),
                entry["sha256"].as_str().unwrap(),
                io::MAX_FILE,
                &mut files,
            )?;
            contract::check_bf16(&raw)?;
            let input = if stage == "fc2" {
                "block-00-swiglu"
            } else {
                "block-00-qkv"
            };
            let expected = if matches!(stage, "query" | "key" | "value" | "fc2") {
                let a = reference_hash(&reports[2], &name, &full, input)?;
                ensure!(
                    a == reference_hash(&reports[3], &name, &full, input)?,
                    "reference modes differ at selected boundary"
                );
                if stage == "fc2" {
                    ensure!(a == pins::FC2_REFERENCE, "fc2 reference pin");
                }
                Some(a)
            } else {
                None
            };
            stages.insert(
                stage.to_owned(),
                json!({"file":files.last().unwrap(),"shape":shape,"reference_sha256":expected}),
            );
        }
        let original = native["cases"]
            .as_array()
            .context("native cases")?
            .iter()
            .find(|c| c["name"] == name)
            .context("native case missing")?;
        cases.push(json!({"name":name,"grid":[h,w],"stages":stages,"original":original}));
    }
    let (raw, weight_receipt) = weight::selected(&model, &index, checkpoint)?;
    let libraries = json!({"cuda":io::library_receipt(Path::new("/usr/lib/aarch64-linux-gnu/libcuda.so.1"))?,
        "cublaslt":io::library_receipt(Path::new("/usr/local/cuda/lib64/libcublasLt.so.13"))?});
    let out = io::fresh(out)?;
    let selected = io::save(&out, "fc2-weight.bf16", &raw)?;
    let admission = json!({"schema":"atlas-dsv-p1-admission-v1","status":"CPU_ADMITTED",
        "diagnostic_only":true,"full_encoder_qualified":false,"model":model,"corpus":corpus,
        "official_repository":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp","official_revision":pins::OFFICIAL_REV,
        "checkpoint":checkpoint,"config":io::json(&config)?,"cases":cases,"files":files,"sources":sources,"libraries":libraries,
        "weight":selected,"weight_selection":weight_receipt,"reference_default":reports[2],"reference_full":reports[3],
        "reference_precision_evidence":{"torch":"2.10.0+cu130","default_bf16_reduced_precision":true,
            "full_bf16_reduced_precision":false,"sdpa":"CUDA MATH","allow_tf32":false,
            "tf32_evidence":"retained script assignment; stage reports do not record a TF32 field",
            "historical_default_script_byte_identity":"not established; retained script later gained full-reduction option"},
        "missing_reference_payloads":["angles","query","key","fc2"],
        "reference_gate":"exact retained SHA256 only; no per-element reference metrics available",
        "gpu_memory_cap_bytes":134217728,"angle_build_flags":["-ptx","-O3","-arch=sm_121f","--fmad=false",
            "--ftz=false","--prec-div=true","--prec-sqrt=true"]});
    io::save_json(&out, "admission.json", &admission)?;
    Ok(admission)
}
pub fn load(path: &Path, expected: &str) -> Result<Value> {
    ensure!(
        contract::sha_syntax(expected),
        "reviewed admission SHA required"
    );
    let admission = io::json(&io::pinned(path, expected, 4 * 1024 * 1024)?)?;
    ensure!(
        admission["schema"] == "atlas-dsv-p1-admission-v1" && admission["status"] == "CPU_ADMITTED",
        "admission status/schema"
    );
    for name in ["files", "sources"] {
        for f in admission[name].as_array().context("admission receipts")? {
            io::verify_receipt(f)?;
        }
    }
    for name in ["cuda", "cublaslt"] {
        let old = &admission["libraries"][name];
        ensure!(
            io::library_receipt(Path::new(
                old["path"].as_str().context("library receipt path")?
            ))? == *old,
            "library changed"
        );
    }
    let weight = io::verify_receipt(&admission["weight"])?;
    contract::check_bf16(&weight)?;
    ensure!(
        weight.len() == 5_767_168
            && Some(io::hash(&weight)?.as_str())
                == admission["weight_selection"]["payload_sha256"].as_str(),
        "selected weight pin"
    );
    Ok(admission)
}
