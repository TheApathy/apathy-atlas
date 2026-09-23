// SPDX-License-Identifier: AGPL-3.0-only
//! Narrow completion boundary for the real encoder body and its example owner.

/// The implementation owns or borrows the full session: encoder arena, weights,
/// workspace and cuBLAS handle/library lifetime, plus pending host upload data.
/// Quarantine retains every live owner; it must not free, destroy, unload, or
/// silently clear the poison state.
pub trait CompletionIo {
    fn is_poisoned(&self) -> bool;
    fn synchronize(&mut self) -> Result<(), String>;
    fn quarantine(&mut self);
}

/// `release_owners` is called only after a successful completion fence. On a
/// partial release error, retain remaining owners and report the error; never
/// pretend the entire session was released or retry already-destroyed handles.
pub trait OwnerIo: CompletionIo {
    fn release_owners(&mut self) -> Result<(), String>;
}

/// Enclose upload, the SAME encoder run, and observation in `body`. Its return
/// is a non-owning Copy receipt (e.g. DevicePtr), so discarding it cannot drop a
/// live GPU owner. This handles fallible Results, not Rust panic recovery.
pub fn with_completion<I: CompletionIo, T: Copy>(
    io: &mut I,
    body: impl FnOnce(&mut I) -> Result<T, String>,
) -> Result<T, String> {
    if io.is_poisoned() {
        return Err("diagnostic session is quarantined".into());
    }
    let operation = body(io);
    // A failing upload or enqueue may already have submitted work.
    if let Err(fence) = io.synchronize() {
        io.quarantine();
        return Err(match operation {
            Ok(_) => format!("encoder completion failed: {fence}"),
            Err(error) => format!("{error}; encoder completion failed: {fence}"),
        });
    }
    operation
}

/// Quarantined sessions require independent recovery authority, not optimistic
/// cleanup. No GPU owner is released on an uncertain completion path.
pub fn release_owned<I: OwnerIo>(io: &mut I) -> Result<(), String> {
    if io.is_poisoned() {
        return Err("diagnostic owners are quarantined".into());
    }
    if let Err(error) = io.synchronize() {
        io.quarantine();
        return Err(format!("owner release completion failed: {error}"));
    }
    if let Err(error) = io.release_owners() {
        io.quarantine();
        return Err(format!("owner release failed: {error}"));
    }
    Ok(())
}
