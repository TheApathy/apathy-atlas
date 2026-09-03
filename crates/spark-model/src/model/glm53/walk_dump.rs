// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;

use crate::layers::ops::GgmlIqBuffer;

/// Bring-up diagnostic: when `ATLAS_GLM53_DUMP_DIR` is set, every event's
/// workspace buffers are copied to host and written raw (no header) as
/// `<dir>/p<position>/<index>-<event>.<buffer>.<bf16|f32>`, for comparison
/// against a `llama-eval-callback` trace of the same token. Synchronizes after
/// every event, so it is diagnostic-only and never on by default.
///
/// The extension names the buffer's ACTUAL element width. It used to be a
/// hardcoded `.bf16` on every dump, which was already wrong for the f32 state
/// and decay buffers; now that the KDA intra-layer chain is f32 a reader that
/// trusted the old extension would decode two f32s as four bf16s and report a
/// fabricated divergence.
/// The element width a dumped buffer actually holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Glm53DumpDtype {
    Bf16,
    F32,
    /// Unsigned 32-bit integer data: expert ids, selected indices, sequence
    /// lengths. Distinct from F32 because these are NOT floats -- writing them
    /// as `.f32` invites a consumer to reinterpret 288 expert ids as denormals.
    U32,
}

impl Glm53DumpDtype {
    fn extension(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::U32 => "u32",
        }
    }
}

pub(crate) struct Glm53WalkDump {
    dir: Option<std::path::PathBuf>,
}

impl Glm53WalkDump {
    const ENV: &'static str = "ATLAS_GLM53_DUMP_DIR";

    /// Read the dump root, and NEVER let it stay relative.
    ///
    /// A relative `ATLAS_GLM53_DUMP_DIR` resolves against whatever the process
    /// happened to be launched from, which is not the directory the person who
    /// set it was standing in. That put 12 GB of `p0/`..`p27/` in the repo root
    /// three separate times in one session, and each time the first symptom
    /// was a confusing `git status`, not a dump that failed.
    pub(crate) fn from_env() -> Self {
        let Some(raw) = std::env::var_os(Self::ENV) else {
            return Self { dir: None };
        };
        let (dir, note) = Self::resolve(std::path::PathBuf::from(raw), std::env::current_dir().ok());
        eprintln!("GLM dump: {}", note);
        Self { dir: Some(dir) }
    }

    /// The resolution itself, pure so it can be pinned without touching the
    /// process environment.
    ///
    /// The note is emitted for an ABSOLUTE setting too. The point is that no
    /// run leaves you guessing where its dumps went, and a message that only
    /// appears in the surprising case is one nobody learns to read.
    fn resolve(
        raw: std::path::PathBuf,
        working_directory: Option<std::path::PathBuf>,
    ) -> (std::path::PathBuf, String) {
        if raw.is_absolute() {
            let note = format!("{} = {}", Self::ENV, raw.display());
            return (raw, note);
        }
        match working_directory {
            Some(cwd) => {
                let resolved = cwd.join(&raw);
                let note = format!(
                    "{} is RELATIVE ({}); resolved against the process working \
                     directory to {}",
                    Self::ENV,
                    raw.display(),
                    resolved.display()
                );
                (resolved, note)
            }
            // Loud and unresolved rather than quietly relative: a dump in an
            // unknown directory is the failure this exists to prevent.
            None => {
                let note = format!(
                    "{} is RELATIVE ({}) and the working directory is unreadable; \
                     dumps will land somewhere UNKNOWN",
                    Self::ENV,
                    raw.display()
                );
                (raw, note)
            }
        }
    }

    /// Copy the running executable next to the data it is about to produce.
    ///
    /// A dump is worthless if you cannot say which binary made it, and the
    /// binary is the part that disappears: cargo relinks `deps/spark_model-*`
    /// IN PLACE, so a rebuild minutes after a run silently destroys the only
    /// artifact that could attribute 5.7 GB of dumps. That has now happened
    /// twice in one session. Archiving at SCORE time does not help -- a run
    /// that is never scored is exactly the one that loses its binary.
    ///
    /// The copy lives WITH the data rather than in a shared hash-named store,
    /// so the two cannot be separated by a later rebuild, a prune, or a naming
    /// change. Done once per process, on the first write.
    fn archive_binary(dir: &std::path::Path) {
        let snapshot = dir.join("binary-snapshot");
        if snapshot.exists() {
            return;
        }
        let Ok(exe) = std::env::current_exe() else {
            eprintln!("GLM dump: cannot resolve current_exe; dump will be UNATTRIBUTABLE");
            return;
        };
        // Loud on failure: a provenance record that fails quietly is worse than
        // none, because it reads as success.
        if let Err(error) = std::fs::copy(&exe, &snapshot) {
            eprintln!(
                "GLM dump: FAILED to archive {} -> {}: {error}; dump will be UNATTRIBUTABLE",
                exe.display(),
                snapshot.display()
            );
            return;
        }
        let meta = std::fs::metadata(&exe).ok();
        let note = format!(
            "source: {}\nbytes: {}\n",
            exe.display(),
            meta.as_ref().map(|m| m.len()).unwrap_or(0)
        );
        let _ = std::fs::write(dir.join("binary-snapshot.txt"), note);
    }

    pub(crate) fn write(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        position: u32,
        stage: &str,
        buffers: &[(&str, GgmlIqBuffer, Glm53DumpDtype)],
    ) -> Result<()> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating GLM dump root {}", dir.display()))?;
        Self::archive_binary(dir);
        let dir = dir.join(format!("p{position}"));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating GLM dump directory {}", dir.display()))?;
        gpu.synchronize(stream)?;
        let stage: String = stage
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        for (name, buffer, dtype) in buffers {
            let mut host = vec![0u8; buffer.bytes];
            gpu.copy_d2h(buffer.ptr, &mut host)?;
            let path = dir.join(format!("{stage}.{name}.{}", dtype.extension()));
            std::fs::write(&path, &host)
                .with_context(|| format!("writing GLM dump {}", path.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Glm53DumpDtype, Glm53WalkDump};
    use std::path::PathBuf;

    /// A relative dump root must never survive as relative.
    ///
    /// This is the fix for a concrete, repeated incident: a relative
    /// `ATLAS_GLM53_DUMP_DIR` resolved against the process working directory
    /// and dropped 12 GB of `p0/`..`p27/` into the repo root three times in one
    /// session. The instrument is the resolution itself, not the env read, so
    /// the pin does not have to mutate the process environment to fire.
    #[test]
    fn a_relative_dump_root_is_resolved_and_always_announced() {
        let cwd = Some(PathBuf::from("/var/tmp/atlas-bringup-logs"));

        let (dir, note) = Glm53WalkDump::resolve(PathBuf::from("dump_dsafix"), cwd.clone());
        assert_eq!(dir, PathBuf::from("/var/tmp/atlas-bringup-logs/dump_dsafix"));
        assert!(dir.is_absolute());
        assert!(note.contains("is RELATIVE"), "{note}");
        assert!(note.contains("/var/tmp/atlas-bringup-logs/dump_dsafix"), "{note}");

        // `./x` and `../x` are relative too -- the check is is_absolute, not a
        // leading-slash or leading-dot test.
        for sneaky in ["./dump", "../dump", "a/b/c"] {
            let (dir, note) = Glm53WalkDump::resolve(PathBuf::from(sneaky), cwd.clone());
            assert!(dir.is_absolute(), "{sneaky} stayed relative as {}", dir.display());
            assert!(note.contains("is RELATIVE"), "{sneaky}: {note}");
        }

        // An absolute root passes through untouched, and is STILL announced --
        // a warning that only prints in the surprising case is one nobody
        // learns to read.
        let absolute = PathBuf::from("/var/tmp/atlas-bringup-logs/dump_poolfix");
        let (dir, note) = Glm53WalkDump::resolve(absolute.clone(), cwd);
        assert_eq!(dir, absolute);
        assert!(!note.contains("is RELATIVE"), "{note}");
        assert!(note.contains("/var/tmp/atlas-bringup-logs/dump_poolfix"), "{note}");

        // No working directory: unresolved, but LOUD rather than silently
        // relative. Dumps landing somewhere unknown is the failure being cured.
        let (dir, note) = Glm53WalkDump::resolve(PathBuf::from("dump"), None);
        assert_eq!(dir, PathBuf::from("dump"));
        assert!(note.contains("UNKNOWN"), "{note}");
    }

    /// Pins the dump file extensions.
    ///
    /// These are the interface a consumer's globbing scripts read, and they are
    /// too short for `strings` to resolve out of a binary -- 3-byte literals are
    /// packed contiguously, so `strings | grep -x u32` returns 0 even when the
    /// mapping is correct. A test is the instrument that can actually assert
    /// this; a binary grep cannot.
    #[test]
    fn dump_extensions_name_the_real_element_width() {
        assert_eq!(Glm53DumpDtype::Bf16.extension(), "bf16");
        assert_eq!(Glm53DumpDtype::F32.extension(), "f32");
        assert_eq!(Glm53DumpDtype::U32.extension(), "u32");
    }
}
