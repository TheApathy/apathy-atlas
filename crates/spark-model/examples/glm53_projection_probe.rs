// SPDX-License-Identifier: AGPL-3.0-only
//! Actual captured BF16[2,4096] and original layer0 K/V[1024,4096].
//! Raw equality only: a compute-type candidate never replaces the old baseline.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use spark_model::{layers::ops, weight_map::DenseWeight};
use spark_runtime::{
    cublaslt::{self, ReductionPolicy},
    gpu::{DevicePtr, GpuBackend, KernelHandle},
};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[allow(dead_code)]
#[path = "glm53_dflash2_kv_parity/artifacts.rs"]
mod artifacts;
#[path = "glm53_projection_probe/batchm.rs"]
mod batchm;
#[path = "glm53_projection_probe/batchm_contract.rs"]
mod batchm_contract;
#[path = "glm53_projection_probe/batchm_input.rs"]
mod batchm_input;
#[path = "glm53_projection_probe/batchm_probe.rs"]
mod batchm_probe;
#[path = "glm53_projection_probe/batchm_session.rs"]
mod batchm_session;
#[path = "glm53_projection_probe/contract.rs"]
mod contract;
#[path = "glm53_projection_probe/family_report.rs"]
mod family_report;
#[path = "glm53_projection_probe/gemv.rs"]
mod gemv;
#[path = "glm53_projection_probe/gemv_contract.rs"]
mod gemv_contract;
#[path = "glm53_projection_probe/session.rs"]
mod session;
#[path = "glm53_projection_probe/timing.rs"]
mod timing;
#[path = "glm53_projection_probe/timing_order.rs"]
mod timing_order;
use gemv::{GemvKernels, GemvMode};
use gemv_contract::{GemvPlan, HIDDEN, OUTPUTS, Span};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Blob {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    source_sha256: String,
    binary_sha256: String,
    input: Blob,
    key: Blob,
    value: Blob,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Projection {
    Cublas(ReductionPolicy),
    TensorCore,
    Gemv(GemvMode),
}
impl Projection {
    fn name(self) -> &'static str {
        match self {
            Self::Cublas(ReductionPolicy::Baseline) => "baseline",
            Self::Cublas(ReductionPolicy::ComputeTypeOnly) => "compute",
            Self::TensorCore => "tensor-core",
            Self::Gemv(mode) => mode.as_str(),
        }
    }
}

struct Operator {
    input: Span,
    weight: Span,
    output: Span,
    slots: Span,
    table: Span,
    gemv: GemvKernels,
    tc: KernelHandle,
}
impl Operator {
    fn new(
        session: &mut session::Session,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
    ) -> Result<Self> {
        let gemv = GemvKernels::load(&*session.gpu)?;
        let tc = session.gpu.kernel("gemm_tc", "dense_gemm_tc")?;
        ensure!(tc.0 != 0, "tensor-core handle is null");
        let weight = Span {
            ptr: weight.0,
            bytes: OUTPUTS as usize * HIDDEN as usize * 2,
        };
        let (slots, table) = gemv_contract::metadata(weight)?;
        let slots = session.upload(slots.to_vec())?;
        let table = session.upload(table.to_vec())?;
        session.gpu.synchronize(session.stream)?;
        Ok(Self {
            input: Span {
                ptr: input.0,
                bytes: 2 * HIDDEN as usize * 2,
            },
            weight,
            output: Span {
                ptr: output.0,
                bytes: 2 * OUTPUTS as usize * 2,
            },
            slots: Span {
                ptr: slots.0,
                bytes: 8,
            },
            table: Span {
                ptr: table.0,
                bytes: 8,
            },
            gemv,
            tc,
        })
    }
    fn enqueue(
        &self,
        session: &session::Session,
        policy: Projection,
        rows: u32,
        row: u32,
        diagnostic: bool,
    ) -> Result<Value> {
        let plan = GemvPlan::new(
            rows,
            row,
            self.input,
            self.weight,
            self.output,
            self.slots,
            self.table,
        )?;
        let input = plan.input();
        let output = plan.output();
        match policy {
            Projection::Cublas(policy) if diagnostic => {
                return Ok(serde_json::to_value(
                    cublaslt::bf16_gemm_act_weight_t_diagnostic(
                        input.ptr,
                        self.weight.ptr,
                        output.ptr,
                        rows,
                        OUTPUTS,
                        HIDDEN,
                        session.stream,
                        policy,
                    )?,
                )?);
            }
            Projection::Cublas(policy) => {
                ensure!(
                    policy == ReductionPolicy::Baseline,
                    "timing admits only original cuBLAS baseline"
                );
                cublaslt::bf16_gemm_act_weight_t(
                    input.ptr,
                    self.weight.ptr,
                    output.ptr,
                    rows,
                    OUTPUTS,
                    HIDDEN,
                    session.stream,
                )?;
            }
            Projection::TensorCore => ops::dense_gemm_tc(
                &*session.gpu,
                self.tc,
                DevicePtr(input.ptr),
                &DenseWeight {
                    weight: DevicePtr(self.weight.ptr),
                },
                DevicePtr(output.ptr),
                rows,
                OUTPUTS,
                HIDDEN,
                session.stream,
            )?,
            Projection::Gemv(mode) => {
                self.gemv
                    .launch(&*session.gpu, mode, &plan, session.stream)?
            }
        }
        Ok(if diagnostic {
            json!({"operator":policy.name(),"m":rows,"n":OUTPUTS,"k":HIDDEN,
                "family":if matches!(policy,Projection::TensorCore) {"fixed-K tensor-core"} else {"independent-row GEMV"}})
        } else {
            Value::Null
        })
    }
}
fn load(blob: &Blob, size: usize) -> Result<Vec<u8>> {
    ensure!(
        fs::metadata(&blob.path)?.len() == size as u64,
        "blob file extent mismatch"
    );
    ensure!(
        artifacts::sha256(&blob.path)? == blob.sha256,
        "blob hash drift"
    );
    let bytes = fs::read(&blob.path)?;
    contract::validate_bf16(&bytes, size)?;
    ensure!(
        artifacts::sha256(&blob.path)? == blob.sha256,
        "blob changed during read"
    );
    Ok(bytes)
}
fn run_case(
    session: &mut session::Session,
    root: &Path,
    operator: &Operator,
    policy: Projection,
    reverse: bool,
) -> Result<(Vec<u8>, Vec<u8>, Value)> {
    fs::create_dir(root)?;
    session.poison_output(DevicePtr(operator.output.ptr))?;
    let calls: &[(u32, usize)] = if reverse {
        &[(1, 1), (1, 0), (2, 0)]
    } else {
        &[(2, 0), (1, 0), (1, 1)]
    };
    let mut receipts = Vec::new();
    let (mut full, mut split) = (None, None);
    for &(m, row) in calls {
        let receipt = operator.enqueue(session, policy, m, u32::try_from(row)?, true)?;
        receipts.push(json!({"row":row,"receipt":receipt}));
        if m == 2 {
            full = Some(session.read(DevicePtr(operator.output.ptr))?);
        }
        if m == 1 && row == usize::from(!reverse) {
            split = Some(session.read(DevicePtr(operator.output.ptr))?);
        }
    }
    let full = full.context("missing full projection")?;
    let split = split.context("missing split projection")?;
    let mut outputs = Vec::new();
    for (name, data) in [("full.bin", &full), ("split.bin", &split)] {
        let path = root.join(name);
        artifacts::write(&path, data)?;
        outputs.push(json!({"file":name,"bytes":data.len(),"sha256":artifacts::sha256(&path)?}));
    }
    let comparison = contract::compare(&full, &split)?;
    let report =
        json!({"reverse":reverse,"calls":receipts,"outputs":outputs,"comparison":comparison});
    artifacts::json_file(&root.join("result.json"), &report)?;
    Ok((full, split, report))
}
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let with_batchm = args.last().is_some_and(|v| v == "--batchm");
    if with_batchm {
        args.pop();
    }
    let with_timing = args.last().is_some_and(|v| v == "--operator-timing");
    if with_timing {
        args.pop();
    }
    ensure!(
        !(with_batchm && with_timing),
        "batchM is a separate untimed diagnostic"
    );
    ensure!(
        args.len() == 2,
        "usage: glm53_projection_probe MANIFEST_JSON FRESH_OUTPUT [--operator-timing | --batchm]"
    );
    let manifest_path = Path::new(&args[0]);
    ensure!(
        fs::metadata(manifest_path)?.len() <= 64 * 1024,
        "operator manifest too large"
    );
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    ensure!(
        manifest.schema == "atlas.glm53.projection-inputs.v1",
        "wrong operator schema"
    );
    ensure!(
        manifest.source_sha256.len() == 64
            && manifest
                .source_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit()),
        "invalid source identity"
    );
    ensure!(
        artifacts::sha256(&std::env::current_exe()?)? == manifest.binary_sha256,
        "operator ELF mismatch"
    );
    let x = load(&manifest.input, 2 * 4096 * 2)?;
    let key = load(&manifest.key, 1024 * 4096 * 2)?;
    let value = load(&manifest.value, 1024 * 4096 * 2)?;
    let root = Path::new(&args[1]);
    fs::create_dir(root)?;
    if with_batchm {
        return batchm_probe::run(root, manifest_path, &manifest, x, key, value);
    }
    let mut session = session::Session::new()?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
        let x = session.upload(x)?;
        let key = session.upload(key)?;
        let value = session.upload(value)?;
        let output = session.output()?;
        let mut comparisons = Vec::new();
        let mut prepared = Vec::new();
        let mut family_exact = true;
        for (name, weight) in [("key", key), ("value", value)] {
            let operator = Operator::new(&mut session, x, weight, output)?;
            let mut baseline = None;
            let mut gemv_oracle = None;
            for policy in [
                Projection::Cublas(ReductionPolicy::Baseline),
                Projection::Cublas(ReductionPolicy::ComputeTypeOnly),
                Projection::TensorCore,
                Projection::Gemv(GemvMode::Sequential),
                Projection::Gemv(GemvMode::Gather),
                Projection::Gemv(GemvMode::Batch2),
            ] {
                let policy_name = policy.name();
                let (a, b, forward) = run_case(
                    &mut session,
                    &root.join(format!("{name}-{policy_name}-forward")),
                    &operator,
                    policy,
                    false,
                )?;
                let (c, d, reverse) = run_case(
                    &mut session,
                    &root.join(format!("{name}-{policy_name}-reverse")),
                    &operator,
                    policy,
                    true,
                )?;
                let baseline_diff = baseline.as_ref().map(|(full,split):&(Vec<u8>,Vec<u8>)| -> Result<Value> {
                    Ok(json!({"full":contract::compare(full,&a)?,"split":contract::compare(split,&b)?}))
                }).transpose()?;
                let family = if matches!(policy, Projection::Gemv(_)) {
                    let oracle = gemv_oracle.get_or_insert_with(|| a.clone());
                    let report = family_report::compare(oracle, [&a, &b, &c, &d])?;
                    family_exact &= report["gemv_family_exact"] == true;
                    Some(report)
                } else {
                    None
                };
                comparisons.push(json!({"weight":name,"policy":policy_name,
                    "forward":forward,"reverse":reverse,"order_full":contract::compare(&a,&c)?,
                    "order_split":contract::compare(&b,&d)?,"against_original":baseline_diff,"gemv_family":family}));
                if policy == Projection::Cublas(ReductionPolicy::Baseline) {
                    baseline = Some((a, b));
                }
            }
            prepared.push((name, operator));
        }
        artifacts::json_file(
            &root.join("family-parity.json"),
            &json!({"comparisons":comparisons,"gemv_family_exact":family_exact}),
        )?;
        let mut timings = Vec::new();
        if with_timing && family_exact {
            for (name, operator) in &prepared {
                timings.push(timing::run(&mut session, operator, name, root)?);
            }
        }
        Ok(
            json!({"schema":"atlas.glm53.projection-result.v1","source_sha256":manifest.source_sha256,
            "binary_sha256":manifest.binary_sha256,"manifest_sha256":artifacts::sha256(manifest_path)?,
            "comparisons":comparisons,"gemv_family_exact":family_exact,"with_operator_timing":with_timing,"timings":timings,
            "qualification":"isolated two-row operator only; no model parity or serving speed promotion"}),
        )
    }));
    session.close()?;
    let report = result.map_err(|_| anyhow::anyhow!("operator panicked; owners drained"))??;
    artifacts::json_file(&root.join("result.json"), &report)?;
    ensure!(
        report["gemv_family_exact"] == true,
        "new GEMV family raw mismatch; timing skipped and evidence retained"
    );
    println!("operator comparison complete; inspect raw exactness and original-reference drift");
    Ok(())
}
