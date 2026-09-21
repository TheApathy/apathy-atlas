// SPDX-License-Identifier: AGPL-3.0-only
//! Selected FC1 same-input diagnostic; no authoritative block-8 teacher exists.
use crate::{
    abi::Blas,
    contract::{self, DeviceSpan, Fc1Plan, ReductionMode},
    lt::Lt,
    lt_plan::{self, LtMode, LtPlan},
    protocol,
    report::{CandidateMode, check_native_control, compare_repeats},
};
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{
    contract::check_bf16,
    driver::{Buffer, Driver},
    io,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub struct Inputs {
    x: Vec<u8>,
    w: Vec<u8>,
    native: Vec<u8>,
    ptx: Vec<u8>,
    cublas: PathBuf,
    cublaslt: PathBuf,
}
impl Inputs {
    pub fn load(admission: &Value) -> Result<Self> {
        let x = io::verify_receipt(&admission["input"])?;
        let w = io::verify_receipt(&admission["weight"])?;
        let native = io::verify_receipt(&admission["native_control"])?;
        let ptx = io::verify_receipt(&admission["native_ptx"])?;
        ensure!(
            x.len() == contract::INPUT_BYTES
                && w.len() == contract::WEIGHT_BYTES
                && native.len() == contract::OUTPUT_BYTES,
            "selected input extents"
        );
        ensure!(
            io::hash(&x)? == contract::INPUT_SHA
                && io::hash(&native)? == contract::NATIVE_SHA
                && io::hash(&ptx)? == contract::PTX_SHA,
            "selected input/native/PTX pins"
        );
        for raw in [&x, &w, &native] {
            check_bf16(raw)?;
        }
        check_native_control(&native, &native)?;
        let library = |name: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                admission["libraries"][name]["path"]
                    .as_str()
                    .context("library path")?,
            ))
        };
        Ok(Self {
            x,
            w,
            native,
            ptx,
            cublas: library("cublas")?,
            cublaslt: library("cublaslt")?,
        })
    }
}
fn span(b: Buffer) -> DeviceSpan {
    DeviceSpan {
        ptr: b.ptr,
        bytes: b.bytes,
    }
}
struct Operands {
    x: Buffer,
    w: Buffer,
    native: Buffer,
    workspace: Buffer,
    lt_workspace: Buffer,
}
fn immutable(d: &Driver, inputs: &Inputs, b: &Operands, include_native: bool) -> Result<()> {
    ensure!(d.read(b.x)? == inputs.x, "device input changed");
    ensure!(d.read(b.w)? == inputs.w, "device weight changed");
    if include_native {
        ensure!(
            d.read(b.native)? == inputs.native,
            "native control buffer changed"
        );
    }
    d.guards()
}

/// Attempt readback even after an API error. Save available bytes BEFORE any
/// finite, repetition, native or metric check. A failed read never gets a hash.
fn finish_attempt(
    d: &Driver,
    y: Buffer,
    operation: Result<()>,
    out: &Path,
    stem: &str,
) -> Result<(Vec<u8>, Value)> {
    let raw = d.read(y);
    let saved = match &raw {
        Ok(bytes) => io::save(out, &format!("{stem}.bf16"), bytes),
        Err(error) => Err(anyhow::anyhow!("output unavailable: {error:#}")),
    };
    let operation_error = operation.as_ref().err().map(|e| format!("{e:#}"));
    let read_error = raw.as_ref().err().map(|e| format!("{e:#}"));
    let save_error = saved.as_ref().err().map(|e| format!("{e:#}"));
    let receipt = io::save_json(
        out,
        &format!("{stem}-attempt.json"),
        &json!({"operation_error":operation_error, "read_error":read_error,
            "save_error":save_error, "output":saved.as_ref().ok(),
            "numerical_validation":"not yet performed", "teacher_reference_available":false}),
    );
    ensure!(
        operation.is_ok() && raw.is_ok() && saved.is_ok() && receipt.is_ok(),
        "attempt {stem}: operation={operation_error:?}; read={read_error:?}; save={save_error:?}; receipt={receipt:?}"
    );
    Ok((raw?, json!({"output":saved?, "attempt":receipt?})))
}
fn native_control(d: &mut Driver, inputs: &Inputs, b: &Operands, out: &Path) -> Result<Value> {
    let bound = Fc1Plan::new().bind(span(b.x), span(b.w), span(b.native), span(b.workspace))?;
    let function = d.function(&inputs.ptx, "deepseek_vision_linear")?;
    let mut receipts = Vec::new();
    for repeat in 0..2 {
        d.write(b.native, &vec![0xff; b.native.bytes])?;
        let launch = d.launch(function, bound.native_launch());
        let drain = d.sync();
        let operation = match (launch, drain) {
            (Ok(()), Ok(())) => Ok(()),
            (launch, drain) => Err(anyhow::anyhow!(
                "native launch={launch:?}; completion={drain:?}"
            )),
        };
        let (raw, receipt) = finish_attempt(
            d,
            b.native,
            operation,
            out,
            &format!("native-repeat-{repeat}"),
        )?;
        receipts.push(receipt);
        let sha = io::hash(&raw)?;
        ensure!(
            sha == contract::NATIVE_SHA,
            "native control hash {sha}; candidate modes not admitted"
        );
        check_native_control(&raw, &inputs.native)?;
        immutable(d, inputs, b, true)?;
    }
    let result = json!({"status":"EXACT_NATIVE_REPLAY", "native_sha256":contract::NATIVE_SHA,
        "repeats":receipts, "repeat_count":2, "repeat_byte_equal":true,
        "input_weight_unchanged":true, "guards_unchanged":true, "teacher_reference_available":false});
    io::save_json(out, "native-control.json", &result)?;
    Ok(result)
}
fn gemmex_mode(
    d: &Driver,
    blas: &mut Blas<'_>,
    inputs: &Inputs,
    b: &Operands,
    y: Buffer,
    reduction: ReductionMode,
    mode: CandidateMode,
    out: &Path,
) -> Result<Value> {
    let bound = Fc1Plan::new().bind(span(b.x), span(b.w), span(y), span(b.workspace))?;
    let mut outputs = Vec::new();
    let mut receipts = Vec::new();
    for repeat in 0..2 {
        d.write(y, &vec![0xff; y.bytes])?;
        d.write(b.workspace, &vec![0; b.workspace.bytes])?;
        let operation = protocol::execute_mode(blas, &bound, d.stream as u64, reduction);
        let (raw, receipt) = finish_attempt(
            d,
            y,
            operation,
            out,
            &format!("{}-repeat-{repeat}", mode.label()),
        )?;
        receipts.push(receipt);
        check_bf16(&raw)?;
        immutable(d, inputs, b, true)?;
        outputs.push(raw);
    }
    let mut result = compare_repeats(mode, &outputs, &inputs.native)?;
    result["repeats"] = json!(receipts);
    result["input_weight_native_control_unchanged"] = json!(true);
    result["guards_unchanged"] = json!(true);
    result["abi"] = json!({"function":"cublasGemmEx", "transpose":["T","N"],
        "m_n_k":[5632,20,1024], "lda_ldb_ldc":[1024,1024,5632], "A_B_C_dtype":14,
        "compute_type":68, "algorithm":99, "alpha_f32":1.0, "beta_f32":0.0,
        "math_mode":reduction.math_mode(), "math_mode_restored":0, "cublas_version":blas.version,
        "workspace_bytes":bound.workspace.bytes, "workspace_set_after_stream":true,
        "pointer_mode":"host", "stream":"owned nondefault", "tf32_operand_path":false});
    io::save_json(out, &format!("{}-mode.json", mode.label()), &result)?;
    Ok(result)
}
fn run_gemmex_modes(
    d: &Driver,
    inputs: &Inputs,
    b: &Operands,
    ys: [Buffer; 2],
    out: &Path,
) -> Result<Vec<Value>> {
    let mut blas = Blas::open(d, &inputs.cublas, &inputs.cublaslt)?;
    let result = (|| -> Result<Vec<Value>> {
        Ok(vec![
            gemmex_mode(
                d,
                &mut blas,
                inputs,
                b,
                ys[0],
                ReductionMode::Default,
                CandidateMode::GemmExDefault,
                out,
            )?,
            gemmex_mode(
                d,
                &mut blas,
                inputs,
                b,
                ys[1],
                ReductionMode::Full,
                CandidateMode::GemmExFull,
                out,
            )?,
        ])
    })();
    let close = blas.close();
    let cleanup = io::save_json(
        out,
        "gemmex-cleanup.json",
        &json!({
        "run_error":result.as_ref().err().map(|e|format!("{e:#}")),
        "cleanup_error":close.as_ref().err().map(|e|format!("{e:#}"))}),
    );
    match (result, close, cleanup) {
        (Ok(value), Ok(()), Ok(_)) => Ok(value),
        (result, close, cleanup) => {
            anyhow::bail!("GemmEx run={result:?}; cleanup={close:?}; receipt={cleanup:?}")
        }
    }
}
fn lt_mode(
    d: &Driver,
    inputs: &Inputs,
    b: &Operands,
    y: Buffer,
    policy: LtMode,
    mode: CandidateMode,
    out: &Path,
) -> Result<Value> {
    let bound = LtPlan::new(policy).bind(
        span(b.x),
        span(b.w),
        span(y),
        span(b.lt_workspace),
        d.stream as u64,
    )?;
    let mut outputs = Vec::new();
    let mut receipts = Vec::new();
    for repeat in 0..2 {
        d.write(y, &vec![0xff; y.bytes])?;
        d.write(b.lt_workspace, &vec![0; b.lt_workspace.bytes])?;
        let mut lt = Lt::new(d, &inputs.cublaslt)?;
        let selected = lt_plan::execute(&mut lt, &bound);
        let operation = selected
            .as_ref()
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("{e:#}"));
        let (raw, mut receipt) = finish_attempt(
            d,
            y,
            operation,
            out,
            &format!("{}-repeat-{repeat}", mode.label()),
        )?;
        let selected = selected?;
        let abi = lt.receipt(&selected)?;
        // Preserve the actual selected algorithm receipt before validation too.
        receipt["abi"] = io::save_json(
            out,
            &format!("{}-repeat-{repeat}-abi.json", mode.label()),
            &abi,
        )?;
        receipts.push(receipt);
        check_bf16(&raw)?;
        immutable(d, inputs, b, true)?;
        outputs.push(raw);
    }
    let mut result = compare_repeats(mode, &outputs, &inputs.native)?;
    result["repeats"] = json!(receipts);
    result["input_weight_native_control_unchanged"] = json!(true);
    result["guards_unchanged"] = json!(true);
    result["requested_reduction_mask"] = json!(policy.preference_mask());
    io::save_json(out, &format!("{}-mode.json", mode.label()), &result)?;
    Ok(result)
}
fn run_lt_modes(
    d: &Driver,
    inputs: &Inputs,
    b: &Operands,
    ys: [Buffer; 2],
    out: &Path,
) -> Result<Vec<Value>> {
    Ok(vec![
        lt_mode(
            d,
            inputs,
            b,
            ys[0],
            LtMode::Baseline,
            CandidateMode::LtBaseline,
            out,
        )?,
        lt_mode(
            d,
            inputs,
            b,
            ys[1],
            LtMode::ComputeTypeOnly,
            CandidateMode::LtComputeTypeOnly,
            out,
        )?,
    ])
}
pub fn run(d: &mut Driver, inputs: &Inputs, out: &Path) -> Result<Value> {
    let expected = contract::owned_device_bytes()?;
    ensure!(
        expected <= contract::OWNED_DEVICE_CAP && d.peak_bytes == 0,
        "fresh bounded owner required"
    );
    let x = d.upload(&inputs.x)?;
    let w = d.upload(&inputs.w)?;
    let native = d.allocate(contract::OUTPUT_BYTES, 0xff)?;
    let ys = [
        d.allocate(contract::OUTPUT_BYTES, 0xff)?,
        d.allocate(contract::OUTPUT_BYTES, 0xff)?,
        d.allocate(contract::OUTPUT_BYTES, 0xff)?,
        d.allocate(contract::OUTPUT_BYTES, 0xff)?,
    ];
    let workspace = d.allocate(Fc1Plan::new().workspace_bytes, 0)?;
    let lt_workspace = d.allocate(contract::LT_WORKSPACE, 0)?;
    ensure!(d.peak_bytes == expected, "owned device accounting drift");
    let buffers = Operands {
        x,
        w,
        native,
        workspace,
        lt_workspace,
    };
    let control = native_control(d, inputs, &buffers, out)?;
    let mut modes = run_gemmex_modes(d, inputs, &buffers, [ys[0], ys[1]], out)?;
    modes.extend(run_lt_modes(d, inputs, &buffers, [ys[2], ys[3]], out)?);
    immutable(d, inputs, &buffers, true)?;
    Ok(
        json!({"status":"DIAGNOSTIC_COMPLETE", "native_control":control, "modes":modes,
        "teacher_reference_available":false, "teacher_reference_sha256":null,
        "full_encoder_qualified":false, "performance_qualified":false}),
    )
}
