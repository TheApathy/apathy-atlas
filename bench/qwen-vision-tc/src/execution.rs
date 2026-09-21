// SPDX-License-Identifier: AGPL-3.0-only

//! Orchestration only; all effects pass through the explicit I/O boundary.
use crate::contract::{BoundPlan, Launch};
use crate::numerics::validate_output;

/// An implementation owns operand/output allocations and one ordered stream.
/// Uploads/initialization must be visible to that stream before `enqueue`.
/// `fence` establishes completion of all submitted work, including work from a
/// failed enqueue. `poison` must retain potentially live allocations and forbid
/// their reuse/free until completion is independently established by the owner.
pub trait ProbeIo {
    fn is_poisoned(&self) -> bool;
    fn enqueue(&mut self, launch: &Launch) -> Result<(), String>;
    fn fence(&mut self) -> Result<(), String>;
    fn verify_inputs_and_guards(&mut self) -> Result<(), String>;
    fn read_output(&mut self) -> Result<Vec<u16>, String>;
    fn poison(&mut self);
}

/// Publish output only after every selected launch, completion, immutable
/// operand/guard verification, and exact extent/finite validation succeeds.
pub fn execute(plan: &BoundPlan, io: &mut impl ProbeIo) -> Result<Vec<u16>, String> {
    if io.is_poisoned() {
        return Err("owner is poisoned; no work may be submitted".into());
    }
    let mut submit_error = None;
    for launch in plan.launches() {
        if let Err(error) = io.enqueue(launch) {
            submit_error = Some(error);
            break;
        }
    }
    // Even the first failed enqueue may have submitted work. Never return or
    // examine output until the owning stream has drained or has been poisoned.
    if let Err(error) = io.fence() {
        io.poison();
        return Err(match submit_error {
            Some(submit) => format!("{submit}; completion could not be established: {error}"),
            None => format!("completion could not be established: {error}"),
        });
    }
    if let Some(error) = submit_error {
        return Err(error);
    }
    io.verify_inputs_and_guards()?;
    let output = io.read_output()?;
    validate_output(&output, plan.output_elements())?;
    Ok(output)
}
