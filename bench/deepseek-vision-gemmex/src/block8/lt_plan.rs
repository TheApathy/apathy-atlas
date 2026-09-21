// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed FC1 Lt descriptors and the real adapter's fail-closed call protocol.
use crate::contract::{DeviceSpan, Fc1Plan, INPUT_BYTES, LT_WORKSPACE, OUTPUT_BYTES, WEIGHT_BYTES};
use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LtMode {
    Baseline,
    ComputeTypeOnly,
}
impl LtMode {
    pub fn preference_mask(self) -> Option<u32> {
        match self {
            Self::Baseline => None,
            Self::ComputeTypeOnly => Some(2),
        }
    }
}

pub struct LtPlan {
    mode: LtMode,
}
pub struct BoundLt {
    mode: LtMode,
    call: LtCall,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtCall {
    pub transpose: [i32; 2],
    pub m_n_k: [i32; 3],
    pub weight_layout: [u64; 3],
    pub input_layout: [u64; 3],
    pub output_layout: [u64; 3],
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
    pub a_type: i32,
    pub b_type: i32,
    pub c_type: i32,
    pub d_type: i32,
    pub compute_type: i32,
    pub scale_type: i32,
    pub alpha: f32,
    pub beta: f32,
    pub workspace: DeviceSpan,
    pub stream: u64,
    pub heuristic_count: i32,
}
impl LtPlan {
    pub fn new(mode: LtMode) -> Self {
        Self { mode }
    }
    pub fn bind(
        &self,
        x: DeviceSpan,
        w: DeviceSpan,
        y: DeviceSpan,
        workspace: DeviceSpan,
        stream: u64,
    ) -> Result<BoundLt> {
        ensure!(stream != 0, "owned nondefault stream required");
        let spans = [x, w, y, workspace];
        let mut ends = [0u64; 4];
        for (i, bytes) in [INPUT_BYTES, WEIGHT_BYTES, OUTPUT_BYTES, LT_WORKSPACE]
            .into_iter()
            .enumerate()
        {
            let s = spans[i];
            ensure!(
                s.ptr != 0 && s.ptr % 256 == 0 && s.bytes == bytes,
                "Lt operand extent/alignment {i}"
            );
            ends[i] = s
                .ptr
                .checked_add(u64::try_from(bytes)?)
                .context("Lt device pointer overflow")?;
        }
        for i in 0..4 {
            for j in i + 1..4 {
                ensure!(
                    ends[i] <= spans[j].ptr || ends[j] <= spans[i].ptr,
                    "Lt operands overlap"
                );
            }
        }
        // Derive transposes, BF16 enums, FP32 compute and dimensions from the
        // shared fixed FC1 ABI. Full Lt workspace overlap was checked above.
        let geometry = Fc1Plan::new();
        let c = geometry
            .bind(
                x,
                w,
                y,
                DeviceSpan {
                    ptr: workspace.ptr,
                    bytes: geometry.workspace_bytes,
                },
            )?
            .call();
        Ok(BoundLt {
            mode: self.mode,
            call: LtCall {
                transpose: [c.transa, c.transb],
                m_n_k: [c.m, c.n, c.k],
                weight_layout: [c.k as u64, c.m as u64, c.lda as u64],
                input_layout: [c.k as u64, c.n as u64, c.ldb as u64],
                output_layout: [c.m as u64, c.n as u64, c.ldc as u64],
                a: c.a,
                b: c.b,
                c: c.c,
                d: c.c,
                a_type: c.a_type,
                b_type: c.b_type,
                c_type: c.c_type,
                d_type: c.c_type,
                compute_type: c.compute_type,
                scale_type: 0,
                alpha: c.alpha,
                beta: c.beta,
                workspace,
                stream,
                heuristic_count: 1,
            },
        })
    }
}
impl BoundLt {
    pub fn call(&self) -> LtCall {
        self.call
    }
    pub fn mode(&self) -> LtMode {
        self.mode
    }
    fn validate_selected(&self, a: &SelectedAlgorithm) -> Result<()> {
        ensure!(
            a.returned == 1 && a.state == 0 && a.algorithm_id >= 0,
            "no valid first Lt heuristic"
        );
        ensure!(
            a.workspace_bytes <= self.call.workspace.bytes && a.waves.is_finite() && a.waves >= 0.0,
            "invalid Lt workspace/wave metadata"
        );
        ensure!(
            a.split_k >= 0 && matches!(a.reduction_scheme, 0 | 1 | 2 | 4),
            "invalid Lt split/reduction"
        );
        ensure!(
            a.split_k <= 1 || a.reduction_scheme != 0,
            "split-K lacks reduction"
        );
        ensure!(
            self.mode != LtMode::ComputeTypeOnly || matches!(a.reduction_scheme, 0 | 2),
            "Lt selected non-FP32 reduction under compute-only policy"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SelectedAlgorithm {
    pub returned: i32,
    pub state: i32,
    pub algorithm_id: i32,
    pub tile_id: u32,
    pub split_k: i32,
    pub reduction_scheme: u32,
    pub workspace_bytes: usize,
    pub waves: f32,
}
pub trait LtIo {
    fn configure(&mut self, bound: &BoundLt) -> Result<()>;
    fn select_first(&mut self) -> Result<SelectedAlgorithm>;
    fn matmul(&mut self) -> Result<()>;
    fn synchronize(&mut self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
pub fn execute(io: &mut impl LtIo, bound: &BoundLt) -> Result<SelectedAlgorithm> {
    let result: Result<SelectedAlgorithm> = (|| {
        io.configure(bound)?;
        let selected = io.select_first()?;
        bound.validate_selected(&selected)?;
        io.matmul()?;
        Ok(selected)
    })();
    // Even a failed API prefix can have effects. Never destroy the owned Lt
    // objects before attempting completion, and retain all cleanup failures.
    let drain = io.synchronize();
    let close = io.close();
    match (result, drain, close) {
        (Ok(value), Ok(()), Ok(())) => Ok(value),
        (result, drain, close) => {
            anyhow::bail!("Lt operation={result:?}; completion={drain:?}; close={close:?}")
        }
    }
}
