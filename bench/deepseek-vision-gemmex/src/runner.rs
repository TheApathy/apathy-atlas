// SPDX-License-Identifier: AGPL-3.0-only
//! Isolated operator diagnostic, never a model/encoder or performance gate.
use crate::{
    abi::Blas,
    contract::{self, DeviceSpan, Fc1Plan, ReductionMode},
    metrics, protocol,
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
    reference: Vec<u8>,
    ptx: Vec<u8>,
    cublas: PathBuf,
    cublaslt: PathBuf,
}
impl Inputs {
    pub fn load(admission: &Value) -> Result<Self> {
        let x = io::verify_receipt(&admission["input"])?;
        let w = io::verify_receipt(&admission["weight"])?;
        let reference = io::verify_receipt(&admission["full_reference_payload"])?;
        let ptx = io::verify_receipt(&admission["native_ptx"])?;
        ensure!(
            x.len() == contract::INPUT_BYTES
                && w.len() == contract::WEIGHT_BYTES
                && reference.len() == contract::OUTPUT_BYTES,
            "fixed input extents"
        );
        ensure!(
            io::hash(&x)? == contract::INPUT_SHA && io::hash(&reference)? == contract::NATIVE_SHA,
            "selected input/full-reference pins"
        );
        for raw in [&x, &w, &reference] {
            check_bf16(raw)?;
        }
        ensure!(
            admission["default_reference_payload"].is_null(),
            "unreviewed default teacher payload"
        );
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
            reference,
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
fn immutable(d: &Driver, inputs: &Inputs, x: Buffer, w: Buffer) -> Result<()> {
    ensure!(d.read(x)? == inputs.x, "input modified by device");
    ensure!(d.read(w)? == inputs.w, "weight modified by device");
    d.guards()
}
fn native(
    d: &mut Driver,
    inputs: &Inputs,
    x: Buffer,
    w: Buffer,
    y: Buffer,
    workspace: Buffer,
    out: &Path,
) -> Result<Value> {
    let bound = Fc1Plan::new().bind(span(x), span(w), span(y), span(workspace))?;
    let function = d.function(&inputs.ptx, "deepseek_vision_linear")?;
    let mut outputs = Vec::new();
    for repeat in 0..2 {
        d.write(y, &vec![0xff; y.bytes])?;
        let launch = d.launch(function, bound.native_launch());
        let drain = d.sync();
        ensure!(
            launch.is_ok() && drain.is_ok(),
            "native launch={launch:?}; completion={drain:?}"
        );
        let raw = d.read(y)?;
        check_bf16(&raw)?;
        immutable(d, inputs, x, w)?;
        let sha = io::hash(&raw)?;
        let receipt = io::save(out, &format!("native-repeat-{repeat}.bf16"), &raw)?;
        outputs.push(receipt);
        ensure!(
            sha == contract::NATIVE_SHA && raw == inputs.reference,
            "native control mismatch {sha}; candidate GEMMs are not admitted"
        );
    }
    Ok(
        json!({"reference_exact":true,"repeats":outputs,"repeat_byte_equal":true,
        "input_weight_unchanged":true,"guards_unchanged":true,"reference_sha256":contract::NATIVE_SHA}),
    )
}
struct Operands {
    x: Buffer,
    w: Buffer,
    native: Buffer,
    workspace: Buffer,
}
fn mode(
    d: &Driver,
    blas: &mut Blas<'_>,
    inputs: &Inputs,
    buffers: &Operands,
    y: Buffer,
    mode: ReductionMode,
    out: &Path,
) -> Result<Value> {
    let bound = Fc1Plan::new().bind(
        span(buffers.x),
        span(buffers.w),
        span(y),
        span(buffers.workspace),
    )?;
    let mut previous: Option<Vec<u8>> = None;
    let mut outputs = Vec::new();
    let mut exact = true;
    for repeat in 0..2 {
        d.write(y, &vec![0xff; y.bytes])?;
        d.write(buffers.workspace, &vec![0; buffers.workspace.bytes])?;
        protocol::execute_mode(blas, &bound, d.stream as u64, mode)?;
        let raw = d.read(y)?;
        check_bf16(&raw)?;
        immutable(d, inputs, buffers.x, buffers.w)?;
        ensure!(
            d.read(buffers.native)? == inputs.reference,
            "native control output modified"
        );
        if let Some(old) = &previous {
            ensure!(*old == raw, "{} reset-repeat mismatch", mode.label());
        }
        let sha = io::hash(&raw)?;
        exact &= sha == mode.reference_sha256();
        let report = match mode {
            ReductionMode::Default => Value::Null,
            ReductionMode::Full => metrics::compare(&raw, &inputs.reference)?,
        };
        let receipt = io::save(out, &format!("{}-repeat-{repeat}.bf16", mode.label()), &raw)?;
        outputs.push(json!({"output":receipt,"reference_exact":sha == mode.reference_sha256(),
            "reference_metrics":report,"metrics_reference":"full: retained native fc1 bytes, SHA equal to full teacher; default: unavailable"}));
        previous = Some(raw);
    }
    let result = json!({"mode":mode.label(),"math_mode":mode.math_mode(),"reference_exact":exact,
        "reference_sha256":mode.reference_sha256(),"reference_payload_available":mode == ReductionMode::Full,
        "repeat_byte_equal":true,"repeats":outputs,"input_weight_native_control_unchanged":true,
        "guards_unchanged":true,"math_mode_restored":0,"cublas_version":blas.version,
        "abi":{"function":"cublasGemmEx","transpose":["T","N"],"m_n_k":[5632,20,1024],
            "lda_ldb_ldc":[1024,1024,5632],"A_B_C_dtype":14,"compute_type":68,"algorithm":99,
            "alpha_f32":1.0,"beta_f32":0.0,"workspace_bytes":bound.workspace.bytes,
            "workspace_set_after_stream":true,"pointer_mode":"host","tf32_operand_path":false}});
    io::save_json(out, &format!("{}-mode.json", mode.label()), &result)?;
    Ok(result)
}
pub fn run(d: &mut Driver, inputs: &Inputs, out: &Path) -> Result<Value> {
    let plan = Fc1Plan::new();
    let x = d.upload(&inputs.x)?;
    let w = d.upload(&inputs.w)?;
    let control = d.allocate(contract::OUTPUT_BYTES, 0xff)?;
    let default = d.allocate(contract::OUTPUT_BYTES, 0xff)?;
    let full = d.allocate(contract::OUTPUT_BYTES, 0xff)?;
    let workspace = d.allocate(plan.workspace_bytes, 0)?;
    ensure!(
        d.peak_bytes == 20_773_888 && d.peak_bytes < 32 * 1024 * 1024,
        "owned device accounting drift"
    );
    let control_receipt = native(d, inputs, x, w, control, workspace, out)?;
    let buffers = Operands {
        x,
        w,
        native: control,
        workspace,
    };
    let mut blas = Blas::open(d, &inputs.cublas, &inputs.cublaslt)?;
    let result = (|| -> Result<Value> {
        let default = mode(
            d,
            &mut blas,
            inputs,
            &buffers,
            default,
            ReductionMode::Default,
            out,
        )?;
        let full = mode(
            d,
            &mut blas,
            inputs,
            &buffers,
            full,
            ReductionMode::Full,
            out,
        )?;
        Ok(
            json!({"native_control":control_receipt,"default":default,"full":full,
            "reference_exact":default["reference_exact"] == true && full["reference_exact"] == true}),
        )
    })();
    // Always fence, restore default math and destroy the owned cuBLAS handle;
    // the outer owner then checks guards and explicitly tears down CUDA.
    let close = blas.close();
    match (result, close) {
        (Ok(v), Ok(())) => Ok(v),
        (result, close) => anyhow::bail!("GemmEx result={result:?}; cuBLAS cleanup={close:?}"),
    }
}
