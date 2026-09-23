// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded model-owned position-free graph lifecycle. All device I/O is supplied.
use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

/// Failed enqueues poison the enclosing model even if its caller restores a
/// speculative snapshot or catches a panic. No ordinary decode may slip past.
pub fn with_failure_owner<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
    poison: impl FnOnce(),
) -> Result<T, E> {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => {
            poison();
            Err(error)
        }
        Err(payload) => {
            poison();
            resume_unwind(payload)
        }
    }
}

#[derive(Clone, Copy)]
pub struct Setting(bool);
impl Setting {
    pub fn parse(value: Option<&OsStr>) -> Result<Self, String> {
        match value {
            None => Ok(Self(false)),
            Some(v) if v == "0" => Ok(Self(false)),
            Some(v) if v == "1" => Ok(Self(true)),
            _ => Err("ATLAS_GLM53_FFN_GRAPHS must be absent or exactly 0 or 1".into()),
        }
    }
    pub fn enabled(self) -> bool {
        self.0
    }
}

/// Verification keys occupy `0..294` (rows 2..=8 x MoE layers 3..=44). The
/// scalar decode walk (rows = 1) has its own 42-entry band at `294..336`, so a
/// one-row graph can never be confused with a verification graph and the
/// verification key space stays exactly what it was.
pub const VERIFY_ENTRIES: usize = 294;
pub const SCALAR_ENTRIES: usize = 42;
pub const ENTRIES: usize = VERIFY_ENTRIES + SCALAR_ENTRIES;
/// `row_counts` index of the scalar band.
pub const SCALAR_ROW_SLOT: usize = 7;

#[derive(Clone, Copy)]
pub struct Key(usize);
impl Key {
    pub fn new(rows: u32, layer: u32) -> Result<Self, String> {
        if !(2..=8).contains(&rows) || !(3..=44).contains(&layer) {
            return Err("FFN graph requires verification rows2..8 and MoE layer3..44".into());
        }
        Ok(Self(((rows - 2) * 42 + layer - 3) as usize))
    }

    /// One-row decode walk key (`ATLAS_GLM53_FFN_GRAPHS_SCALAR=1`).
    pub fn new_scalar(layer: u32) -> Result<Self, String> {
        if !(3..=44).contains(&layer) {
            return Err("scalar FFN graph requires MoE layer3..44".into());
        }
        Ok(Self(VERIFY_ENTRIES + (layer - 3) as usize))
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub stream: u64,
    /// Model-owned workspace, scratch and pointer-table allocation identities.
    pub owners: [u64; 3],
}

pub trait GraphIo {
    fn dispatch(&mut self) -> Result<(), String>;
    fn begin(&mut self, stream: u64) -> Result<(), String>;
    fn end(&mut self, stream: u64) -> Result<u64, String>;
    fn launch(&mut self, graph: u64, stream: u64) -> Result<(), String>;
    fn synchronize(&mut self, stream: u64) -> Result<(), String>;
    fn destroy(&mut self, graph: u64) -> Result<(), String>;
}

#[derive(Clone, Copy)]
enum Entry {
    Cold,
    Warm,
    Ready(u64),
}

pub struct GraphCache {
    entries: [Entry; ENTRIES],
    binding: Option<Binding>,
    poisoned: bool,
    unreclaimable: bool,
    counts: [u64; 3],
    row_counts: [[u64; 3]; 8],
}
impl GraphCache {
    pub fn new() -> Self {
        Self {
            entries: [Entry::Cold; ENTRIES],
            binding: None,
            poisoned: false,
            unreclaimable: false,
            counts: [0; 3],
            row_counts: [[0; 3]; 8],
        }
    }

    pub fn counts(&self) -> [u64; 3] {
        self.counts
    }

    pub fn row_counts(&self) -> [[u64; 3]; 8] {
        self.row_counts
    }

    pub fn execute(
        &mut self,
        key: Key,
        binding: Binding,
        io: &mut impl GraphIo,
    ) -> Result<(), String> {
        if self.poisoned {
            return Err("FFN graph owner is poisoned".into());
        }
        if binding.stream == 0 || binding.owners.contains(&0) {
            return Err("FFN graph needs a nonzero stream and live model owners".into());
        }
        if self.binding.is_some_and(|previous| previous != binding) {
            return Err("FFN graph model/stream binding changed".into());
        }
        self.binding = Some(binding);
        // Any I/O error or panic leaves the entire owner unusable, including
        // failed warm-up. Never retry an invocation that may have queued work.
        self.poisoned = true;
        match self.entries[key.0] {
            Entry::Cold => {
                io.dispatch()?;
                self.entries[key.0] = Entry::Warm;
                self.counts[0] += 1;
                self.row_counts[key.0 / 42][0] += 1;
            }
            Entry::Warm => {
                self.unreclaimable = true;
                let begin = io.begin(binding.stream);
                self.unreclaimable = false;
                begin?;
                let body = catch_unwind(AssertUnwindSafe(|| io.dispatch()));
                // A successful begin requires exactly one end even when the
                // body fails or panics. Partial captured graphs are never run.
                self.unreclaimable = true;
                let end = catch_unwind(AssertUnwindSafe(|| io.end(binding.stream)));
                if matches!(&end, Ok(Ok(_))) {
                    self.unreclaimable = false;
                }
                if !matches!(&body, Ok(Ok(()))) {
                    let cleanup = match end {
                        Ok(Ok(graph)) if graph != 0 => {
                            self.entries[key.0] = Entry::Ready(graph);
                            self.unreclaimable = true;
                            match catch_unwind(AssertUnwindSafe(|| io.destroy(graph))) {
                                Ok(Ok(())) => {
                                    self.unreclaimable = false;
                                    self.entries[key.0] = Entry::Warm;
                                    Ok(())
                                }
                                Ok(Err(error)) => {
                                    self.unreclaimable = false;
                                    Err(error)
                                }
                                Err(_) => Err("graph cleanup panicked; owner retained".into()),
                            }
                        }
                        Ok(Ok(_)) => Err("capture returned null graph".into()),
                        Ok(Err(error)) => Err(error),
                        Err(_) => Err("end capture panicked".into()),
                    };
                    match body {
                        Err(payload) => resume_unwind(payload),
                        Ok(Err(error)) => {
                            return Err(format!("capture body: {error}; cleanup: {cleanup:?}"));
                        }
                        Ok(Ok(())) => unreachable!(),
                    }
                }
                let graph = match end {
                    Ok(result) => result?,
                    Err(payload) => resume_unwind(payload),
                };
                if graph == 0 {
                    return Err("capture returned null executable graph".into());
                }
                // Retain ownership before the fallible launch. A failed launch
                // requires draining before any buffers or graph can be freed.
                self.entries[key.0] = Entry::Ready(graph);
                io.launch(graph, binding.stream)?;
                self.counts[1] += 1;
                self.row_counts[key.0 / 42][1] += 1;
            }
            Entry::Ready(graph) => {
                io.launch(graph, binding.stream)?;
                self.counts[2] += 1;
                self.row_counts[key.0 / 42][2] += 1;
            }
        }
        self.poisoned = false;
        Ok(())
    }

    /// Successful graph destruction precedes all model-buffer teardown. On
    /// fence/destroy failure the failed handle and remaining owners are kept.
    pub fn drain(&mut self, io: &mut impl GraphIo) -> Result<(), String> {
        self.poisoned = true;
        if self.unreclaimable {
            return Err(
                "FFN graph ownership indeterminate; retain model until process teardown".into(),
            );
        }
        if let Some(binding) = self.binding {
            io.synchronize(binding.stream)?;
            for entry in &mut self.entries {
                if let Entry::Ready(graph) = *entry {
                    self.unreclaimable = true;
                    let destroyed = io.destroy(graph);
                    self.unreclaimable = false;
                    destroyed?;
                    *entry = Entry::Cold;
                }
            }
        }
        Ok(())
    }
}
