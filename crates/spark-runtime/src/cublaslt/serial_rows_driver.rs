// SPDX-License-Identifier: AGPL-3.0-only

//! Scoped host-resource reuse, retaining one original heuristic and M1 call
//! per row. All external operations belong to the supplied I/O adapter.

use std::fmt::{self, Debug};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use super::diagnostic_contract::HeuristicResult;
use super::serial_rows_contract::{
    ByteSpan, MatrixLayout, RowCall, SerialRowsPlan, SerialRowsRequest, WORKSPACE_BYTES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Descriptor,
    Layout,
    Preference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorSet<H> {
    pub desc: H,
    pub a: H,
    pub b: H,
    pub c: H,
    pub d: H,
    pub pref: H,
}

/// Creation returns a successfully acquired handle or retains responsibility
/// for any foreign out-handle not returned to this driver. No method may add
/// GPU allocation, synchronization, graph capture, tensor copies or retries.
pub trait SerialRowsIo {
    type Handle: Copy + Debug + Eq;
    type Error: Debug;

    fn create_matmul(&mut self, compute: i32, scale: i32) -> Result<Self::Handle, Self::Error>;
    fn set_matmul_i32(
        &mut self,
        desc: Self::Handle,
        attr: u32,
        value: i32,
    ) -> Result<(), Self::Error>;
    fn create_layout(&mut self, layout: MatrixLayout) -> Result<Self::Handle, Self::Error>;
    fn create_preference(&mut self) -> Result<Self::Handle, Self::Error>;
    fn set_preference_usize(
        &mut self,
        pref: Self::Handle,
        attr: u32,
        value: usize,
    ) -> Result<(), Self::Error>;
    fn heuristic(
        &mut self,
        set: DescriptorSet<Self::Handle>,
        requested: i32,
    ) -> Result<(HeuristicResult, i32), Self::Error>;
    fn matmul(
        &mut self,
        set: DescriptorSet<Self::Handle>,
        call: RowCall,
        result: &HeuristicResult,
    ) -> Result<(), Self::Error>;
    fn destroy(&mut self, kind: ResourceKind, handle: Self::Handle) -> Result<(), Self::Error>;
    /// Optional additive strategy. An explicit batch request rejects false
    /// before matmul; it never silently executes the serial fallback.
    fn prepare_strided_m1(
        &mut self,
        _set: DescriptorSet<Self::Handle>,
        _request: SerialRowsRequest,
        _call: RowCall,
        _result: &HeuristicResult,
    ) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[derive(Debug)]
pub enum Failure<E> {
    Io(E),
    Admission(&'static str),
}

#[derive(Debug)]
pub struct CleanupFailure<H, E> {
    pub kind: ResourceKind,
    pub handle: H,
    pub error: E,
}

#[derive(Debug)]
pub struct RunError<H, E> {
    pub primary: Option<Failure<E>>,
    pub cleanup: Vec<CleanupFailure<H, E>>,
}

impl<H: Debug, E: Debug> fmt::Display for RunError<H, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "native serial rows: primary=")?;
        match &self.primary {
            None => write!(formatter, "none")?,
            Some(Failure::Io(error)) => write!(formatter, "I/O {error:?}")?,
            Some(Failure::Admission(message)) => write!(formatter, "admission {message}")?,
        }
        write!(formatter, ", cleanup=[")?;
        for (index, failure) in self.cleanup.iter().enumerate() {
            if index != 0 {
                write!(formatter, ", ")?;
            }
            write!(
                formatter,
                "{:?} {:?}: {:?}",
                failure.kind, failure.handle, failure.error
            )?;
        }
        write!(formatter, "]")
    }
}

impl<H: Debug, E: Debug> std::error::Error for RunError<H, E> {}

/// The request is validated before the first external callback. The FFI entry
/// must also call SerialRowsPlan::new before obtaining even a cold context.
pub fn run_serial_rows<I: SerialRowsIo>(
    io: &mut I,
    request: SerialRowsRequest,
    workspace: ByteSpan,
) -> Result<(), RunError<I::Handle, I::Error>> {
    run_rows(io, request, workspace, false)
}

pub fn run_batched_rows<I: SerialRowsIo>(
    io: &mut I,
    request: SerialRowsRequest,
    workspace: ByteSpan,
) -> Result<(), RunError<I::Handle, I::Error>> {
    run_rows(io, request, workspace, true)
}

fn run_rows<I: SerialRowsIo>(
    io: &mut I,
    request: SerialRowsRequest,
    workspace: ByteSpan,
    batched: bool,
) -> Result<(), RunError<I::Handle, I::Error>> {
    let admission_error = |message| RunError {
        primary: Some(Failure::Admission(message)),
        cleanup: Vec::new(),
    };
    let plan = SerialRowsPlan::new(request).map_err(admission_error)?;
    let bound = plan.bind_workspace(workspace).map_err(admission_error)?;

    // Reserve the bounded bookkeeping before any external resource exists.
    // Each successful creation is recorded before the next fallible call.
    let mut resources = Vec::with_capacity(5);
    let mut cleanup = Vec::with_capacity(5);
    let operation = catch_unwind(AssertUnwindSafe(|| -> Result<(), Failure<I::Error>> {
        // These are the unchanged CUDA13 production descriptor attributes:
        // compute32F=68, scale32F=0, TRANSA=3, TRANSB=4, maxworkspace=1.
        let desc = io.create_matmul(68, 0).map_err(Failure::Io)?;
        resources.push((ResourceKind::Descriptor, desc));
        io.set_matmul_i32(desc, 3, plan.trans_a())
            .map_err(Failure::Io)?;
        io.set_matmul_i32(desc, 4, 0).map_err(Failure::Io)?;

        let [layout_a, layout_b, layout_d] = plan.layouts();
        let a = io.create_layout(layout_a).map_err(Failure::Io)?;
        resources.push((ResourceKind::Layout, a));
        let b = io.create_layout(layout_b).map_err(Failure::Io)?;
        resources.push((ResourceKind::Layout, b));
        let d = io.create_layout(layout_d).map_err(Failure::Io)?;
        resources.push((ResourceKind::Layout, d));
        let pref = io.create_preference().map_err(Failure::Io)?;
        resources.push((ResourceKind::Preference, pref));
        io.set_preference_usize(pref, 1, WORKSPACE_BYTES)
            .map_err(Failure::Io)?;
        let set = DescriptorSet {
            desc,
            a,
            b,
            c: d,
            d,
            pref,
        };

        if batched {
            let call = bound.row(0).map_err(Failure::Admission)?;
            let (result, returned) = io.heuristic(set, 1).map_err(Failure::Io)?;
            result
                .admit(returned, WORKSPACE_BYTES)
                .map_err(Failure::Admission)?;
            if !io
                .prepare_strided_m1(set, request, call, &result)
                .map_err(Failure::Io)?
            {
                return Err(Failure::Admission("explicit strided M1 is unsupported"));
            }
            io.matmul(set, call, &result).map_err(Failure::Io)?;
            return Ok(());
        }
        for index in 0..plan.rows() {
            let call = bound.row(index).map_err(Failure::Admission)?;
            let (result, returned) = io.heuristic(set, 1).map_err(Failure::Io)?;
            result
                .admit(returned, WORKSPACE_BYTES)
                .map_err(Failure::Admission)?;
            io.matmul(set, call, &result).map_err(Failure::Io)?;
        }
        Ok(())
    }));

    // Cleanup is outside the operational unwind. A failing or panicking
    // destructor must not prevent attempts for the remaining acquired owners.
    let mut cleanup_panic = None;
    for (kind, handle) in resources.into_iter().rev() {
        match catch_unwind(AssertUnwindSafe(|| io.destroy(kind, handle))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => cleanup.push(CleanupFailure {
                kind,
                handle,
                error,
            }),
            Err(payload) => {
                if cleanup_panic.is_none() {
                    cleanup_panic = Some(payload);
                }
            }
        }
    }

    // Preserve the original operational panic rather than converting it into
    // apparent success or hiding it behind a cleanup failure.
    let result = match operation {
        Ok(result) => result,
        Err(payload) => resume_unwind(payload),
    };
    if let Some(payload) = cleanup_panic {
        resume_unwind(payload);
    }
    if result.is_ok() && cleanup.is_empty() {
        Ok(())
    } else {
        Err(RunError {
            primary: result.err(),
            cleanup,
        })
    }
}
