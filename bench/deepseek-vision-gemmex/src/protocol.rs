// SPDX-License-Identifier: AGPL-3.0-only
//! The real cuBLAS adapter consumes exactly the sequence exercised by CPU tests.
use crate::contract::{BoundFc1, DeviceSpan, GemmCall, ReductionMode};
use anyhow::{Result, ensure};

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    SetStream(u64),
    SetWorkspace(DeviceSpan),
    SetHostPointerMode,
    SetMathMode(i32),
    Gemm(GemmCall),
    Synchronize,
}
pub trait GemmIo {
    fn execute(&mut self, command: Command) -> Result<()>;
}
pub fn execute_mode(
    io: &mut impl GemmIo,
    bound: &BoundFc1,
    stream: u64,
    mode: ReductionMode,
) -> Result<()> {
    ensure!(stream != 0, "owned nondefault stream required");
    let mut errors = Vec::new();
    for command in [
        Command::SetStream(stream),
        Command::SetWorkspace(bound.workspace),
        Command::SetHostPointerMode,
        Command::SetMathMode(mode.math_mode()),
        Command::Gemm(bound.call()),
    ] {
        if let Err(error) = io.execute(command) {
            errors.push(format!("operation: {error:#}"));
            break;
        }
    }
    // A failed API may already have enqueued work. Fence and restore on every
    // partial prefix, retaining both errors rather than publishing success.
    if let Err(error) = io.execute(Command::Synchronize) {
        errors.push(format!("completion: {error:#}"));
    }
    if let Err(error) = io.execute(Command::SetMathMode(0)) {
        errors.push(format!("restore: {error:#}"));
    }
    ensure!(errors.is_empty(), "{}", errors.join("; "));
    Ok(())
}
