// SPDX-License-Identifier: AGPL-3.0-only
//! Actual operator-only diagnostic. No model/cache/serving qualification.
use super::batchm_contract::{
    BatchMIo, BatchMLaunch, BatchMPlan, INPUT_ROW_BYTES, MAX_INPUT_ROWS, OUTPUT_ROW_BYTES, Schedule,
};
use super::batchm_input::{INPUT_DERIVATION, derive_input};
use super::batchm_session::{DEVICE_CAP_BYTES, Session};
use super::gemv_contract::Span;
use super::{Manifest, artifacts, batchm::BatchMKernels, contract};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_runtime::gpu::GpuBackend;
use std::{fs, path::Path};

const SHAPES: [u32; 14] = [1, 2, 3, 8, 9, 15, 16, 17, 31, 32, 33, 256, 1024, 2047];
const KERNEL_SOURCE: &[u8] =
    include_bytes!("../../../../kernels/gb10/common/dense_gemv_bf16_batchm.cu");
const BUILD_CONFIG: &[u8] = include_bytes!("../../../../kernels/gb10/common/KERNEL.toml");
#[derive(Clone, Copy)]
struct Operator {
    input: Span,
    weight: Span,
    output: Span,
}
struct DeviceIo<'a> {
    gpu: &'a dyn GpuBackend,
    kernels: &'a BatchMKernels,
    stream: u64,
    index: usize,
    skip: Option<usize>,
}
impl BatchMIo for DeviceIo<'_> {
    fn launch(&mut self, launch: BatchMLaunch) -> Result<()> {
        let index = self.index;
        self.index += 1;
        if self.skip == Some(index) {
            return Ok(());
        }
        self.kernels.launch(self.gpu, launch, self.stream)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_schedule(
    session: &mut Session,
    kernels: &BatchMKernels,
    operator: Operator,
    rows: u32,
    source_row: u32,
    schedule: Schedule,
    skip: Option<usize>,
    root: &Path,
) -> Result<Vec<u8>> {
    let plan = BatchMPlan::new(
        rows,
        source_row,
        MAX_INPUT_ROWS,
        operator.input,
        operator.weight,
        operator.output,
        kernels.handles(),
    )?;
    let launches = plan.launches(schedule)?;
    ensure!(
        skip.is_none_or(|index| index < launches.len()),
        "omitted launch outside schedule"
    );
    fs::create_dir(root)?;
    let receipt: Vec<_> = launches
        .iter()
        .enumerate()
        .map(|(index, l)| {
            json!({
                "index":index,"source_row":l.source_row,"output_row":l.output_row,
                "rows":l.rows,"kind":format!("{:?}",l.kind),"omitted":skip==Some(index)
            })
        })
        .collect();
    artifacts::json_file(
        &root.join("launches.json"),
        &json!({"status":"PLANNED_NOT_COMPLETION","schedule":schedule.name(),
        "rows":rows,"source_row":source_row,"launches":receipt}),
    )?;
    session.poison_output(operator.output)?;
    plan.execute(
        schedule,
        &mut DeviceIo {
            gpu: &*session.gpu,
            kernels,
            stream: session.stream,
            index: 0,
            skip,
        },
    )?;
    let raw = session.read_output(operator.output)?;
    // Persist every active byte before finite/raw comparison. Inactive storage
    // must still equal the known poison; retain its entire owner on any drift.
    let active_bytes = rows as usize * OUTPUT_ROW_BYTES;
    artifacts::write(&root.join("output.bin"), &raw[..active_bytes])?;
    let guards = session.guard_snapshot()?;
    artifacts::write(&root.join("guards.bin"), &guards)?;
    let finite = contract::validate_bf16(&raw[..active_bytes], active_bytes).is_ok();
    let inactive_exact = raw[active_bytes..]
        .chunks_exact(2)
        .all(|b| b == [0xc0, 0x7f]);
    if !inactive_exact {
        artifacts::write(&root.join("full-owner-tail-failure.bin"), &raw)?;
    }
    let guards_exact = session.check_guards(&guards).is_ok();
    artifacts::json_file(
        &root.join("result.json"),
        &json!({
            "raw_sha256":artifacts::sha256(&root.join("output.bin"))?,"raw_bytes":active_bytes,
            "owner_bytes":raw.len(),"active_bytes":active_bytes,"active_finite":finite,
            "inactive_poison_exact":inactive_exact,"inactive_bf16_bits":"0x7fc0",
            "guards_sha256":artifacts::sha256(&root.join("guards.bin"))?,"guards_exact":guards_exact,
            "omitted_launch":skip,"completion":"same default stream fenced before owned D2H and after copy"
        }),
    )?;
    ensure!(
        guards_exact && inactive_exact,
        "batchM guard/inactive extent corruption at {}",
        root.display()
    );
    Ok(raw[..active_bytes].to_vec())
}

fn compare_schedules(reference: &[u8], candidate: &[u8]) -> Value {
    match contract::compare(reference, candidate) {
        Ok(report) => report,
        Err(error) => json!({"exact":false,"finite_or_extent_error":format!("{error:#}")}),
    }
}
fn run_case(
    session: &mut Session,
    kernels: &BatchMKernels,
    operator: Operator,
    rows: u32,
    start: u32,
    reverse: bool,
    root: &Path,
) -> Result<Value> {
    fs::create_dir(root)?;
    let mut modes = vec![Schedule::PairReference];
    if rows <= 16 {
        modes.push(Schedule::BatchMDirect);
    }
    modes.push(Schedule::BatchMPartitioned);
    if reverse {
        modes.reverse();
    }
    let mut outputs = Vec::new();
    for mode in modes {
        let output = run_schedule(
            session,
            kernels,
            operator,
            rows,
            start,
            mode,
            None,
            &root.join(mode.name()),
        )?;
        outputs.push((mode, output));
    }
    let reference = &outputs
        .iter()
        .find(|(mode, _)| *mode == Schedule::PairReference)
        .context("missing pair reference")?
        .1;
    let mut exact = true;
    let mut comparisons = Vec::new();
    for (mode, bytes) in &outputs {
        let comparison = compare_schedules(reference, bytes);
        exact &= comparison["exact"] == true;
        comparisons.push(json!({"schedule":mode.name(),"comparison":comparison,
            "output_sha256":artifacts::sha256(&root.join(mode.name()).join("output.bin"))?}));
    }
    let report = json!({"rows":rows,"source_row":start,"reverse":reverse,
        "exact":exact,"comparisons":comparisons});
    artifacts::json_file(&root.join("result.json"), &report)?;
    Ok(report)
}

fn negative_control(
    session: &mut Session,
    kernels: &BatchMKernels,
    operator: Operator,
    root: &Path,
) -> Result<Value> {
    fs::create_dir(root)?;
    let reference = run_schedule(
        session,
        kernels,
        operator,
        33,
        1,
        Schedule::PairReference,
        None,
        &root.join("reference"),
    )?;
    let omitted = run_schedule(
        session,
        kernels,
        operator,
        33,
        1,
        Schedule::BatchMPartitioned,
        Some(1),
        &root.join("omitted-middle"),
    )?;
    let start = 16 * OUTPUT_ROW_BYTES;
    let end = 32 * OUTPUT_ROW_BYTES;
    let detected = contract::validate_bf16(&omitted, omitted.len()).is_err();
    let untouched = omitted[start..end]
        .chunks_exact(2)
        .all(|b| b == [0xc0, 0x7f]);
    let outside_exact =
        reference[..start] == omitted[..start] && reference[end..] == omitted[end..];
    let reference_finite = contract::validate_bf16(&reference, reference.len()).is_ok();
    let pass = detected && untouched && outside_exact && reference_finite;
    let report = json!({"pass":pass,"omitted_launch_index":1,"omitted_output_rows":[16,32],
        "nonfinite_detected":detected,"omitted_rows_still_poison":untouched,
        "outside_rows_exact":outside_exact,"reference_finite":reference_finite,
        "qualification":"intentional diagnostic omission only, never a positive parity sample"});
    artifacts::json_file(&root.join("result.json"), &report)?;
    ensure!(pass, "missing-launch negative control failed");
    Ok(report)
}

fn retain_immutable(session: &mut Session, root: &Path) -> Result<Value> {
    fs::create_dir(root)?;
    let mut items = Vec::new();
    for (index, raw, exact) in session.immutable_snapshots()? {
        let name = format!("owner-{index}.bin");
        artifacts::write(&root.join(&name), &raw)?;
        items.push(
            json!({"owner":index,"file":name,"bytes":raw.len(),"exact":exact,
            "sha256":artifacts::sha256(&root.join(&name))?}),
        );
    }
    let exact = items.len() == 3 && items.iter().all(|v| v["exact"] == true);
    let report = json!({"exact":exact,"owners":items});
    artifacts::json_file(&root.join("result.json"), &report)?;
    ensure!(exact, "batchM immutable input/weight mutation");
    Ok(report)
}

pub(super) fn run(
    root: &Path,
    manifest_path: &Path,
    manifest: &Manifest,
    source: Vec<u8>,
    key: Vec<u8>,
    value: Vec<u8>,
) -> Result<()> {
    // Parent already validated all file hashes/extents/finite bytes and ELF,
    // and exclusively created this output directory before any GPU construction.
    let derived = derive_input(&source, MAX_INPUT_ROWS)?;
    ensure!(
        std::str::from_utf8(BUILD_CONFIG)?.contains("--fmad=false"),
        "missing declared fmad policy"
    );
    artifacts::write(&root.join("source-input.bin"), &source)?;
    artifacts::write(&root.join("derived-input.bin"), &derived)?;
    artifacts::write(&root.join("kernel-source.cu"), KERNEL_SOURCE)?;
    artifacts::write(&root.join("kernel-build-config.toml"), BUILD_CONFIG)?;
    let provenance = json!({"input_derivation":INPUT_DERIVATION,
        "source_input_sha256":manifest.input.sha256,"derived_input_sha256":artifacts::sha256(&root.join("derived-input.bin"))?,
        "source_rows":2,"derived_rows":MAX_INPUT_ROWS,"row_bytes":INPUT_ROW_BYTES,
        "key_sha256":manifest.key.sha256,"value_sha256":manifest.value.sha256,
        "source_sha256":manifest.source_sha256,"binary_sha256":manifest.binary_sha256,
        "manifest_sha256":artifacts::sha256(manifest_path)?,
        "kernel_source_sha256":artifacts::sha256(&root.join("kernel-source.cu"))?,
        "kernel_build_config_sha256":artifacts::sha256(&root.join("kernel-build-config.toml"))?,
        "declared_fmad":false,"actual_nvcc_ptx_receipt":"required from native build admission",
        "donor_sha256":"91d8a64c1ce8ac32703a166599f92298c1e91b6e2dc1b4158071074f1d7ce665"});
    artifacts::json_file(&root.join("provenance.json"), &provenance)?;
    let mut session = Session::new()?;
    let work = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
        // Resolve every symbol before the first allocation/upload effect.
        let kernels = BatchMKernels::load(&*session.gpu)?;
        let input = session.upload(derived)?;
        let key = session.upload(key)?;
        let value = session.upload(value)?;
        let output = session.output()?;
        // Admit every real shape/owner combination before the first projection.
        for weight in [key, value] {
            for rows in SHAPES {
                for start in [0, 1] {
                    let plan = BatchMPlan::new(
                        rows,
                        start,
                        MAX_INPUT_ROWS,
                        input,
                        weight,
                        output,
                        kernels.handles(),
                    )?;
                    plan.launches(Schedule::PairReference)?;
                    plan.launches(Schedule::BatchMPartitioned)?;
                    if rows <= 16 {
                        plan.launches(Schedule::BatchMDirect)?;
                    }
                }
            }
        }
        let mut cases = Vec::new();
        let mut immutable = Vec::new();
        let control = negative_control(
            &mut session,
            &kernels,
            Operator {
                input,
                weight: key,
                output,
            },
            &root.join("negative-control"),
        )?;
        let mut exact = true;
        for (name, weight) in [("key", key), ("value", value)] {
            let operator = Operator {
                input,
                weight,
                output,
            };
            for rows in SHAPES {
                for start in [0, 1] {
                    let mut forward: Option<Value> = None;
                    for reverse in [false, true] {
                        let name = format!(
                            "{name}-m{rows}-start{start}-{}",
                            if reverse { "reverse" } else { "forward" }
                        );
                        let report = run_case(
                            &mut session,
                            &kernels,
                            operator,
                            rows,
                            start,
                            reverse,
                            &root.join(&name),
                        )
                        .with_context(|| name.clone())?;
                        exact &= report["exact"] == true;
                        let order_exact = if let Some(prior) = &forward {
                            let prior = prior["comparisons"]
                                .as_array()
                                .context("forward comparison schema")?;
                            let current = report["comparisons"]
                                .as_array()
                                .context("reverse comparison schema")?;
                            prior.len() == current.len()
                                && prior.iter().all(|a| {
                                    current
                                        .iter()
                                        .find(|b| a["schedule"] == b["schedule"])
                                        .is_some_and(|b| a["output_sha256"] == b["output_sha256"])
                                })
                        } else {
                            true
                        };
                        exact &= order_exact;
                        if !reverse {
                            forward = Some(report.clone());
                        }
                        cases.push(
                            json!({"case":name,"result":report,"paired_order_exact":order_exact}),
                        );
                    }
                }
            }
            immutable.push(retain_immutable(
                &mut session,
                &root.join(format!("immutable-after-{name}")),
            )?);
        }
        Ok(json!({"exact":exact,"cases":cases,"negative_control":control,"immutable":immutable}))
    }));
    let owned_bytes = session.owned_bytes();
    let close = session.close();
    let work = work
        .map_err(|_| anyhow::anyhow!("batchM diagnostic panicked"))
        .and_then(|result| result);
    let report = json!({"schema":"atlas.glm53.batchm-operator.v1","provenance":provenance,
        "owned_device_bytes":owned_bytes,"device_cap_bytes":DEVICE_CAP_BYTES,
        "cleanup_complete":close.is_ok(),"cleanup_error":close.as_ref().err().map(|e|format!("{e:#}")),
        "result":work.as_ref().ok(),"error":work.as_ref().err().map(|e|format!("{e:#}")),
        "qualification":"derived-input operator exactness only; no model/cache/serving or speed promotion"});
    artifacts::json_file(&root.join("result.json"), &report)?;
    close?;
    let result = work?;
    ensure!(
        result["exact"] == true,
        "batchM raw mismatch; all raw failures retained"
    );
    println!("batchM operator diagnostic exact; serving and model qualification remain separate");
    Ok(())
}
