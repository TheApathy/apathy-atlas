// SPDX-License-Identifier: AGPL-3.0-only

//! Exact Qwen4-Exp PLE n-gram row selection.

use anyhow::{Context, Result, ensure};

pub const QWEN4_PLE_HEADS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleRowSelection {
    pub shard: usize,
    pub row: usize,
}

/// CPU row planner. It runs as soon as the current token is known, allowing
/// the 16 sparse reads to overlap embedding lookup and layer 0.
pub struct Qwen4PleHasher {
    multipliers: [i64; 3],
    vocab_sizes: [i64; QWEN4_PLE_HEADS],
    offsets: [i64; QWEN4_PLE_HEADS],
    eos_token_id: u32,
    shard_rows: usize,
}

impl Qwen4PleHasher {
    pub fn new(
        multipliers: [i64; 3],
        vocab_sizes: [i64; QWEN4_PLE_HEADS],
        offsets: [i64; QWEN4_PLE_HEADS],
        eos_token_id: u32,
        shard_rows: usize,
    ) -> Result<Self> {
        ensure!(shard_rows > 0, "PLE shard row count must be positive");
        ensure!(
            multipliers.iter().all(|value| value % 2 != 0),
            "PLE hash multipliers must be odd"
        );
        ensure!(
            vocab_sizes.iter().all(|value| *value > 0),
            "PLE head vocabulary sizes must be positive"
        );
        ensure!(offsets[0] == 0, "PLE first head offset must be zero");
        for head in 1..QWEN4_PLE_HEADS {
            ensure!(
                offsets[head] == offsets[head - 1] + vocab_sizes[head - 1],
                "PLE head offsets are not contiguous at head {head}"
            );
        }
        Ok(Self {
            multipliers,
            vocab_sizes,
            offsets,
            eos_token_id,
            shard_rows,
        })
    }

    pub fn select_decode(
        &self,
        current_token: u32,
        prior_tokens: &[u32],
    ) -> [PleRowSelection; QWEN4_PLE_HEADS] {
        let mut segment = prior_tokens
            .iter()
            .rev()
            .take_while(|&&token| token != self.eos_token_id);
        let previous = segment.next().copied().unwrap_or(self.eos_token_id);
        let previous_2 = segment.next().copied().unwrap_or(self.eos_token_id);
        let bigram = (current_token as i64).wrapping_mul(self.multipliers[0])
            ^ (previous as i64).wrapping_mul(self.multipliers[1]);
        let trigram = bigram ^ (previous_2 as i64).wrapping_mul(self.multipliers[2]);
        std::array::from_fn(|head| {
            let mixed = if head < 8 { bigram } else { trigram };
            let global = mixed.rem_euclid(self.vocab_sizes[head]) + self.offsets[head];
            let global = global as usize;
            PleRowSelection {
                shard: global / self.shard_rows,
                row: global % self.shard_rows,
            }
        })
    }

    /// Plan every sparse row needed by a contiguous prefill chunk. The
    /// returned layout is token-major, then head-major, matching the batched
    /// dequantizer. This is deliberately the same rolling decode contract:
    /// token `t` can see the supplied prefix and chunk rows `[0, t)` only.
    pub fn select_prefill(
        &self,
        current_tokens: &[u32],
        prior_tokens: &[u32],
    ) -> Vec<PleRowSelection> {
        let mut history = Vec::with_capacity(prior_tokens.len() + current_tokens.len());
        history.extend_from_slice(prior_tokens);
        let mut selections = Vec::with_capacity(current_tokens.len() * QWEN4_PLE_HEADS);
        for &token in current_tokens {
            selections.extend_from_slice(&self.select_decode(token, &history));
            history.push(token);
        }
        selections
    }
}

#[cfg(test)]
mod planner_tests {
    use super::*;

    fn hasher() -> Qwen4PleHasher {
        Qwen4PleHasher::new(
            [10007, 10009, 10037],
            std::array::from_fn(|head| 997 + head as i64),
            std::array::from_fn(|head| (0..head).map(|prior| 997 + prior as i64).sum()),
            99,
            251,
        )
        .unwrap()
    }

    #[test]
    fn prefill_plan_is_exact_rolling_decode() {
        let hasher = hasher();
        let prefix = [8, 9, 99, 11, 12];
        let tokens = [13, 14, 99, 15, 16, 17];
        let batched = hasher.select_prefill(&tokens, &prefix);
        let mut history = prefix.to_vec();
        let mut serial = Vec::new();
        for token in tokens {
            serial.extend_from_slice(&hasher.select_decode(token, &history));
            history.push(token);
        }
        assert_eq!(batched, serial);
        assert_eq!(batched.len(), tokens.len() * QWEN4_PLE_HEADS);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod runtime {
    use super::*;
    use atlas_core::config::ModelConfig;
    use parking_lot::Mutex;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
    use spark_runtime::kernel_args::KernelLaunch;
    use spark_runtime::weights::{WeightDtype, WeightStore};
    use spark_storage::ple_offload::{PleIoMode, PleNvfp4Row, PleOffloadReader};
    use std::sync::Arc;
    use std::ffi::CString;
    use std::fs::{self, DirBuilder, File, OpenOptions};
    use std::io::{Read, Write};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    use std::os::unix::io::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::layers::ops;
    use crate::weight_map::{DenseWeight, dense};

    const PLE_EMBED: usize = 2560;
    const PLE_RECORD_BYTES: usize = 90;
    const PLE_CONV_HISTORY: usize = 9;
    const PLE_MAX_SPEC_ROWS: usize = 32;
    const PLE_IO_QUEUE_DEPTH: usize = 256;
    const PLE_IO_QUEUE_DEPTH_ENV: &str = "ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH";
    const PLE_PARITY_WIDTH: usize = 10_240;
    const PLE_PARITY_ENABLE: &str = "ATLAS_QWEN4_PLE_PARITY_CAPTURE";
    const PLE_PARITY_ROOT: &str = "ATLAS_QWEN4_PLE_PARITY_ROOT";
    const PLE_PARITY_NONCE: &str = "ATLAS_QWEN4_PLE_PARITY_NONCE";

    fn parse_ple_io_queue_depth(value: Option<&str>) -> Result<usize> {
        match value {
            None | Some("256") => Ok(PLE_IO_QUEUE_DEPTH),
            Some("32") => Ok(32),
            Some(_) => anyhow::bail!("{PLE_IO_QUEUE_DEPTH_ENV} must be exact value 32 or 256"),
        }
    }

    fn configured_ple_io_queue_depth() -> Result<usize> {
        let value = std::env::var_os(PLE_IO_QUEUE_DEPTH_ENV);
        let value = value
            .as_deref()
            .map(|value| {
                value
                    .to_str()
                    .with_context(|| format!("{PLE_IO_QUEUE_DEPTH_ENV} is not UTF-8"))
            })
            .transpose()?;
        parse_ple_io_queue_depth(value)
    }

    #[cfg(test)]
    mod io_queue_depth_tests {
        use super::*;

        #[test]
        fn absent_defaults_to_wide_queue_and_exact_controls_are_admitted() {
            assert_eq!(parse_ple_io_queue_depth(None).unwrap(), 256);
            assert_eq!(parse_ple_io_queue_depth(Some("32")).unwrap(), 32);
            assert_eq!(parse_ple_io_queue_depth(Some("256")).unwrap(), 256);
        }

        #[test]
        fn ambiguous_or_unbounded_queue_depths_reject() {
            for value in ["", "0", "031", "64", "255", "257", " 256", "256 "] {
                assert!(parse_ple_io_queue_depth(Some(value)).is_err(), "{value}");
            }
        }
    }

    unsafe extern "C" {
        fn renameat2(
            olddirfd: std::ffi::c_int,
            oldpath: *const std::ffi::c_char,
            newdirfd: std::ffi::c_int,
            newpath: *const std::ffi::c_char,
            flags: std::ffi::c_uint,
        ) -> std::ffi::c_int;
    }

    fn random_private_suffix() -> Result<String> {
        let mut bytes = [0u8; 16];
        File::open("/dev/urandom")
            .context("open OS randomness for private PLE parity staging")?
            .read_exact(&mut bytes)
            .context("read OS randomness for private PLE parity staging")?;
        Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    fn rename_noreplace(source: &Path, destination: &Path) -> Result<()> {
        const AT_FDCWD: std::ffi::c_int = -100;
        const RENAME_NOREPLACE: std::ffi::c_uint = 1;
        let source_c = CString::new(source.as_os_str().as_bytes())
            .context("PLE parity rename source contains NUL")?;
        let destination_c = CString::new(destination.as_os_str().as_bytes())
            .context("PLE parity rename destination contains NUL")?;
        let status = unsafe {
            renameat2(
                AT_FDCWD,
                source_c.as_ptr(),
                AT_FDCWD,
                destination_c.as_ptr(),
                RENAME_NOREPLACE,
            )
        };
        if status != 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| {
                format!(
                    "atomic no-replace PLE parity rename {} -> {}",
                    source.display(),
                    destination.display()
                )
            });
        }
        Ok(())
    }

    fn private_staging_path(final_path: &Path) -> Result<PathBuf> {
        let parent = final_path
            .parent()
            .context("PLE parity final frame has no parent")?;
        let final_name = final_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("PLE parity final frame name is not UTF-8")?;
        Ok(parent.join(format!(
            ".{final_name}.staging-{}",
            random_private_suffix()?
        )))
    }

    struct PleParityCaptureConfig {
        root: PathBuf,
        root_dev: u64,
        root_ino: u64,
        nonce: String,
        selector: u8,
    }

    struct PleParityArtifact {
        name: &'static str,
        bytes: usize,
        sha256: String,
        dev: u64,
        ino: u64,
    }

    fn parity_capture_config() -> Result<Option<PleParityCaptureConfig>> {
        const ALLOWED: [&[u8]; 3] = [
            PLE_PARITY_ENABLE.as_bytes(),
            PLE_PARITY_ROOT.as_bytes(),
            PLE_PARITY_NONCE.as_bytes(),
        ];
        let prefix = b"ATLAS_QWEN4_PLE_PARITY_";
        for (key, _) in std::env::vars_os() {
            let raw = key.as_os_str().as_bytes();
            ensure!(
                !raw.starts_with(prefix) || ALLOWED.contains(&raw),
                "unknown Qwen4 PLE parity environment key"
            );
        }
        let enabled = std::env::var_os(PLE_PARITY_ENABLE);
        if enabled.is_none() {
            ensure!(
                std::env::var_os(PLE_PARITY_ROOT).is_none()
                    && std::env::var_os(PLE_PARITY_NONCE).is_none(),
                "partial Qwen4 PLE parity configuration while capture is disabled"
            );
            return Ok(None);
        }
        ensure!(
            enabled
                .and_then(|value| value.into_string().ok())
                .as_deref()
                == Some("1"),
            "{PLE_PARITY_ENABLE} must be exact value 1"
        );
        let nonce = std::env::var(PLE_PARITY_NONCE)
            .with_context(|| format!("missing or non-UTF8 {PLE_PARITY_NONCE}"))?;
        ensure!(valid_parity_nonce(&nonce), "invalid PLE parity nonce");
        let root_text = std::env::var(PLE_PARITY_ROOT)
            .with_context(|| format!("missing or non-UTF8 {PLE_PARITY_ROOT}"))?;
        let root = PathBuf::from(&root_text);
        ensure!(root.is_absolute(), "PLE parity root must be absolute");
        let canonical = root
            .canonicalize()
            .with_context(|| format!("canonicalize PLE parity root {root_text}"))?;
        ensure!(canonical == root, "PLE parity root is not canonical");
        let metadata = fs::symlink_metadata(&root)
            .with_context(|| format!("stat PLE parity root {root_text}"))?;
        ensure!(metadata.is_dir(), "PLE parity root is not a directory");
        ensure!(
            metadata.permissions().mode() & 0o777 == 0o700,
            "PLE parity root mode must be 0700"
        );
        let selector = match std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH").as_deref() {
            Ok("0") => 0,
            Ok("1") => 1,
            _ => anyhow::bail!(
                "PLE parity capture requires exact ATLAS_QWEN4_PLE_PREFILL_BATCH=0 or 1"
            ),
        };
        Ok(Some(PleParityCaptureConfig {
            root,
            root_dev: metadata.dev(),
            root_ino: metadata.ino(),
            nonce,
            selector,
        }))
    }

    fn valid_parity_nonce(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn sha256_hex(bytes: &[u8]) -> Result<String> {
        const INITIAL: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        fn compress(state: &mut [u32; 8], block: &[u8]) {
            let mut words = [0u32; 64];
            for (index, word) in words[..16].iter_mut().enumerate() {
                *word = u32::from_be_bytes(
                    block[index * 4..index * 4 + 4]
                        .try_into()
                        .expect("fixed SHA256 word"),
                );
            }
            for index in 16..64 {
                let s0 = words[index - 15].rotate_right(7)
                    ^ words[index - 15].rotate_right(18)
                    ^ (words[index - 15] >> 3);
                let s1 = words[index - 2].rotate_right(17)
                    ^ words[index - 2].rotate_right(19)
                    ^ (words[index - 2] >> 10);
                words[index] = words[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(words[index - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
            for index in 0..64 {
                let upper = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let choose = (e & f) ^ (!e & g);
                let first = h
                    .wrapping_add(upper)
                    .wrapping_add(choose)
                    .wrapping_add(K[index])
                    .wrapping_add(words[index]);
                let lower = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let majority = (a & b) ^ (a & c) ^ (b & c);
                let second = lower.wrapping_add(majority);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(first);
                d = c;
                c = b;
                b = a;
                a = first.wrapping_add(second);
            }
            for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h].into_iter()) {
                *slot = slot.wrapping_add(value);
            }
        }

        let byte_len = u64::try_from(bytes.len()).context("SHA256 input length exceeds u64")?;
        let bit_len = byte_len
            .checked_mul(8)
            .context("SHA256 bit length overflow")?;
        let mut state = INITIAL;
        let mut chunks = bytes.chunks_exact(64);
        for block in &mut chunks {
            compress(&mut state, block);
        }
        let remainder = chunks.remainder();
        let mut tail = [0u8; 128];
        tail[..remainder.len()].copy_from_slice(remainder);
        tail[remainder.len()] = 0x80;
        let padded = if remainder.len() < 56 { 64 } else { 128 };
        tail[padded - 8..padded].copy_from_slice(&bit_len.to_be_bytes());
        for block in tail[..padded].chunks_exact(64) {
            compress(&mut state, block);
        }
        Ok(state.iter().map(|word| format!("{word:08x}")).collect())
    }

    fn token_sha256(tokens: &[u32]) -> Result<String> {
        token_parts_sha256(&[tokens])
    }

    fn token_parts_sha256(parts: &[&[u32]]) -> Result<String> {
        let token_count = parts.iter().try_fold(0usize, |count, part| {
            count
                .checked_add(part.len())
                .context("PLE parity token count overflow")
        })?;
        let capacity = token_count
            .checked_mul(std::mem::size_of::<u32>())
            .context("PLE parity token extent overflow")?;
        let mut bytes = Vec::with_capacity(capacity);
        for part in parts {
            for token in *part {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        sha256_hex(&bytes)
    }

    fn root_is_stable(config: &PleParityCaptureConfig) -> Result<()> {
        let metadata = fs::symlink_metadata(&config.root).context("re-stat PLE parity root")?;
        ensure!(
            metadata.is_dir()
                && metadata.dev() == config.root_dev
                && metadata.ino() == config.root_ino
                && metadata.permissions().mode() & 0o777 == 0o700,
            "PLE parity root identity or mode changed"
        );
        Ok(())
    }

    struct OwnedCreatedFile {
        path: PathBuf,
        file: Option<File>,
        identity: Option<(u64, u64)>,
        keep: bool,
    }

    impl OwnedCreatedFile {
        fn new(path: PathBuf) -> Self {
            Self {
                path,
                file: None,
                identity: None,
                keep: false,
            }
        }

        fn recover_identity(&mut self) {
            if self.identity.is_some() {
                return;
            }
            self.identity = self.file.as_ref().and_then(|file| {
                file.metadata()
                    .or_else(|_| fs::metadata(format!("/proc/self/fd/{}", file.as_raw_fd())))
                    .ok()
                    .map(|metadata| (metadata.dev(), metadata.ino()))
            });
        }
    }

    impl Drop for OwnedCreatedFile {
        fn drop(&mut self) {
            if self.keep {
                return;
            }
            self.recover_identity();
            drop(self.file.take());
            if let Some(identity) = self.identity
                && fs::symlink_metadata(&self.path)
                    .is_ok_and(|current| current.dev() == identity.0 && current.ino() == identity.1)
            {
                let _ = fs::remove_file(&self.path);
            }
        }
    }

    fn write_sealed_artifact_with_admission<F>(
        frame: &Path,
        name: &'static str,
        bytes: &[u8],
        admit: F,
    ) -> Result<PleParityArtifact>
    where
        F: FnOnce(&File) -> std::io::Result<fs::Metadata>,
    {
        let path = frame.join(name);
        let mut created = OwnedCreatedFile::new(path.clone());
        created.file = Some(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .with_context(|| format!("create PLE parity artifact {}", path.display()))?,
        );
        let opened = admit(created.file.as_ref().expect("created file is retained"))
            .with_context(|| format!("stat new PLE parity artifact {}", path.display()))?;
        ensure!(
            opened.is_file() && opened.nlink() == 1,
            "new PLE parity artifact is not a private regular inode"
        );
        let identity = (opened.dev(), opened.ino());
        created.identity = Some(identity);
        let result = (|| -> Result<PleParityArtifact> {
            let file = created.file.as_mut().expect("created file is retained");
            file.write_all(bytes)
                .with_context(|| format!("write PLE parity artifact {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("sync PLE parity artifact {}", path.display()))?;
            file.set_permissions(fs::Permissions::from_mode(0o444))
                .with_context(|| format!("seal PLE parity artifact {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("sync sealed PLE parity artifact {}", path.display()))?;
            let sealed = fs::symlink_metadata(&path)
                .with_context(|| format!("re-stat PLE parity artifact {}", path.display()))?;
            ensure!(
                sealed.is_file()
                    && sealed.dev() == identity.0
                    && sealed.ino() == identity.1
                    && sealed.nlink() == 1
                    && sealed.len() == bytes.len() as u64
                    && sealed.permissions().mode() & 0o777 == 0o444,
                "PLE parity artifact identity, length, or mode changed"
            );
            Ok(PleParityArtifact {
                name,
                bytes: bytes.len(),
                sha256: sha256_hex(bytes)?,
                dev: sealed.dev(),
                ino: sealed.ino(),
            })
        })();
        if result.is_ok() {
            created.keep = true;
        }
        result
    }

    fn write_sealed_artifact(
        frame: &Path,
        name: &'static str,
        bytes: &[u8],
    ) -> Result<PleParityArtifact> {
        write_sealed_artifact_with_admission(frame, name, bytes, File::metadata)
    }

    fn cleanup_owned_artifacts(frame: &Path, artifacts: &[PleParityArtifact]) {
        for artifact in artifacts.iter().rev() {
            let path = frame.join(artifact.name);
            if fs::symlink_metadata(&path)
                .is_ok_and(|current| current.dev() == artifact.dev && current.ino() == artifact.ino)
            {
                let _ = fs::remove_file(path);
            }
        }
    }

    struct OwnedPleParityFrame {
        path: PathBuf,
        final_path: PathBuf,
        handle: Option<File>,
        identity: Option<(u64, u64)>,
        artifacts: Vec<PleParityArtifact>,
        committed: bool,
    }

    impl OwnedPleParityFrame {
        fn new(path: PathBuf, final_path: PathBuf) -> Self {
            Self {
                path,
                final_path,
                handle: None,
                identity: None,
                artifacts: Vec::with_capacity(4),
                committed: false,
            }
        }

        fn recover_identity(&mut self) {
            if self.identity.is_some() {
                return;
            }
            self.identity = self.handle.as_ref().and_then(|handle| {
                handle
                    .metadata()
                    .or_else(|_| fs::metadata(format!("/proc/self/fd/{}", handle.as_raw_fd())))
                    .ok()
                    .map(|metadata| (metadata.dev(), metadata.ino()))
            });
        }

        fn identity(&self) -> (u64, u64) {
            self.identity.expect("created frame identity is admitted")
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn publish_noreplace(&mut self) -> Result<()> {
            let published_path = self.final_path.clone();
            rename_noreplace(&self.path, &published_path)?;
            // The rename already succeeded. This assignment performs no
            // allocation and keeps rollback bound to the published inode.
            self.path = published_path;
            Ok(())
        }

        fn handle_is_owned(&self) -> bool {
            self.handle.as_ref().is_some_and(|handle| {
                let metadata = handle
                    .metadata()
                    .or_else(|_| fs::metadata(format!("/proc/self/fd/{}", handle.as_raw_fd())));
                metadata.is_ok_and(|metadata| {
                    self.identity
                        .is_some_and(|identity| (metadata.dev(), metadata.ino()) == identity)
                })
            })
        }
    }

    impl Drop for OwnedPleParityFrame {
        fn drop(&mut self) {
            if self.committed {
                return;
            }
            self.recover_identity();
            if self.handle_is_owned()
                && let Some(handle) = &self.handle
            {
                let _ = handle.set_permissions(fs::Permissions::from_mode(0o700));
            }
            cleanup_owned_artifacts(&self.path, &self.artifacts);
            drop(self.handle.take());
            if let Some(identity) = self.identity
                && fs::symlink_metadata(&self.path)
                    .is_ok_and(|current| current.dev() == identity.0 && current.ino() == identity.1)
            {
                let _ = fs::remove_dir(&self.path);
            }
        }
    }

    fn create_owned_frame_with_open_and_admission<O, A>(
        final_path: PathBuf,
        root_dev: u64,
        open: O,
        admit: A,
    ) -> Result<OwnedPleParityFrame>
    where
        O: FnOnce(&Path) -> std::io::Result<File>,
        A: FnOnce(&Path) -> std::io::Result<fs::Metadata>,
    {
        let staging_path = private_staging_path(&final_path)?;
        let mut owned = OwnedPleParityFrame::new(staging_path.clone(), final_path);
        let mut builder = DirBuilder::new();
        builder.mode(0o700);
        builder.create(&staging_path).with_context(|| {
            format!(
                "create private PLE parity staging frame {}",
                staging_path.display()
            )
        })?;
        owned.handle = Some(open(&staging_path).with_context(|| {
            format!(
                "open private PLE parity staging frame {}",
                staging_path.display()
            )
        })?);
        let handle_metadata = owned
            .handle
            .as_ref()
            .expect("created frame handle is retained")
            .metadata()
            .context("stat created PLE parity frame descriptor")?;
        ensure!(
            handle_metadata.is_dir() && handle_metadata.dev() == root_dev,
            "created PLE parity frame descriptor is invalid"
        );
        owned.identity = Some((handle_metadata.dev(), handle_metadata.ino()));
        owned
            .handle
            .as_ref()
            .expect("created frame handle is retained")
            .set_permissions(fs::Permissions::from_mode(0o700))
            .context("set PLE parity frame mode0700")?;
        let admitted = admit(&staging_path).context("stat new PLE parity staging frame")?;
        let identity = owned.identity();
        ensure!(
            admitted.is_dir()
                && admitted.dev() == root_dev
                && admitted.dev() == identity.0
                && admitted.ino() == identity.1
                && admitted.permissions().mode() & 0o777 == 0o700,
            "new PLE parity frame identity or mode is invalid"
        );
        Ok(owned)
    }

    fn create_owned_frame_with_admission<F>(
        path: PathBuf,
        root_dev: u64,
        admit: F,
    ) -> Result<OwnedPleParityFrame>
    where
        F: FnOnce(&Path) -> std::io::Result<fs::Metadata>,
    {
        create_owned_frame_with_open_and_admission(path, root_dev, |path| File::open(path), admit)
    }

    fn create_owned_frame(path: PathBuf, root_dev: u64) -> Result<OwnedPleParityFrame> {
        create_owned_frame_with_admission(path, root_dev, |path| fs::symlink_metadata(path))
    }

    fn verify_sealed_artifact(frame: &Path, artifact: &PleParityArtifact) -> Result<()> {
        let path = frame.join(artifact.name);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("verify PLE parity artifact {}", path.display()))?;
        ensure!(
            metadata.is_file()
                && metadata.dev() == artifact.dev
                && metadata.ino() == artifact.ino
                && metadata.len() == artifact.bytes as u64
                && metadata.permissions().mode() & 0o777 == 0o444,
            "PLE parity artifact final identity, length, or mode changed"
        );
        let final_bytes = fs::read(&path)
            .with_context(|| format!("re-read PLE parity artifact {}", path.display()))?;
        ensure!(
            sha256_hex(&final_bytes)? == artifact.sha256,
            "PLE parity artifact final SHA256 changed"
        );
        Ok(())
    }

    struct Qwen4PlePrefillScratch {
        max_tokens: usize,
        records_dev: DevicePtr,
        scales_dev: DevicePtr,
        embedding_dev: DevicePtr,
        key_dev: DevicePtr,
        value_dev: DevicePtr,
        gated_value_dev: DevicePtr,
        conv_input_dev: DevicePtr,
        dequant_k: KernelHandle,
        prepare_k: KernelHandle,
        conv_inject_k: KernelHandle,
        commit_state_k: KernelHandle,
    }

    impl Qwen4PlePrefillScratch {
        fn allocate(
            gpu: &dyn GpuBackend,
            max_tokens: usize,
            residual_width: usize,
        ) -> Result<Self> {
            ensure!(
                max_tokens > 0 && max_tokens <= u32::MAX as usize,
                "Qwen4 PLE prefill arena token bound is invalid: {max_tokens}"
            );
            let dequant_items = max_tokens
                .checked_mul(PLE_EMBED)
                .context("Qwen4 PLE prefill dequant extent overflow")?;
            ensure!(
                dequant_items <= u32::MAX as usize,
                "Qwen4 PLE prefill dequant extent {dequant_items} exceeds u32 grid contract"
            );
            let bytes = |rows: usize, width: usize, element: usize| -> Result<usize> {
                rows.checked_mul(width)
                    .and_then(|value| value.checked_mul(element))
                    .context("Qwen4 PLE prefill scratch extent overflow")
            };
            let extents = [
                bytes(max_tokens, QWEN4_PLE_HEADS * PLE_RECORD_BYTES, 1)?,
                bytes(max_tokens, QWEN4_PLE_HEADS, std::mem::size_of::<f32>())?,
                bytes(max_tokens, PLE_EMBED, 2)?,
                bytes(max_tokens, residual_width, 2)?,
                bytes(max_tokens, PLE_EMBED, 2)?,
                bytes(max_tokens, residual_width, std::mem::size_of::<f32>())?,
                bytes(max_tokens, residual_width, std::mem::size_of::<f32>())?,
            ];
            let mut allocated = Vec::with_capacity(extents.len());
            let result = (|| -> Result<Self> {
                for extent in extents {
                    allocated.push(gpu.alloc(extent)?);
                }
                Ok(Self {
                    max_tokens,
                    records_dev: allocated[0],
                    scales_dev: allocated[1],
                    embedding_dev: allocated[2],
                    key_dev: allocated[3],
                    value_dev: allocated[4],
                    gated_value_dev: allocated[5],
                    conv_input_dev: allocated[6],
                    dequant_k: gpu.kernel("qwen4_hyper", "qwen4_ple_dequant_prefill")?,
                    prepare_k: gpu.kernel("qwen4_hyper", "qwen4_ple_prepare_prefill")?,
                    conv_inject_k: gpu.kernel("qwen4_hyper", "qwen4_ple_conv_inject_prefill")?,
                    commit_state_k: gpu.kernel("qwen4_hyper", "qwen4_ple_commit_prefill_state")?,
                })
            })();
            match result {
                Ok(scratch) => Ok(scratch),
                Err(error) => {
                    let mut cleanup_errors = Vec::new();
                    for ptr in allocated.into_iter().rev() {
                        if let Err(cleanup) = gpu.free(ptr) {
                            cleanup_errors.push(format!("{ptr}: {cleanup:#}"));
                        }
                    }
                    if cleanup_errors.is_empty() {
                        Err(error.context("Qwen4 PLE prefill scratch allocation rolled back"))
                    } else {
                        Err(error.context(format!(
                            "Qwen4 PLE prefill scratch rollback failures: {}",
                            cleanup_errors.join("; ")
                        )))
                    }
                }
            }
        }

        fn device_buffers(&self) -> [DevicePtr; 7] {
            [
                self.records_dev,
                self.scales_dev,
                self.embedding_dev,
                self.key_dev,
                self.value_dev,
                self.gated_value_dev,
                self.conv_input_dev,
            ]
        }

        fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
            let mut errors = Vec::new();
            for ptr in self.device_buffers().into_iter().rev() {
                if let Err(error) = gpu.free(ptr) {
                    errors.push(format!("{ptr}: {error:#}"));
                }
            }
            ensure!(
                errors.is_empty(),
                "Qwen4 PLE prefill scratch release failures: {}",
                errors.join("; ")
            );
            Ok(())
        }
    }

    /// `ATLAS_QWEN4_PLE_PREFILL_STREAM`: 1 = prefill rows read with the ring
    /// kept full (`read_rows_streamed`), 2 = that plus the read started on a
    /// helper thread before layer 0. Same bytes as the windowed read either way.
    pub fn prefill_stream_level() -> u8 {
        static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        *LEVEL.get_or_init(|| match std::env::var("ATLAS_QWEN4_PLE_PREFILL_STREAM").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        })
    }

    fn prefill_stream_selected() -> bool {
        prefill_stream_level() >= 1
    }

    fn read_prefill_rows(
        hasher: &Qwen4PleHasher,
        reader: &Mutex<PleOffloadReader>,
        current_tokens: &[u32],
        prior_tokens: &[u32],
        streamed: bool,
    ) -> Result<Vec<PleNvfp4Row>> {
        let request: Vec<_> = hasher
            .select_prefill(current_tokens, prior_tokens)
            .iter()
            .map(|selection| (selection.shard, selection.row))
            .collect();
        ensure!(
            request.len() == current_tokens.len() * QWEN4_PLE_HEADS,
            "Qwen4 PLE prefill selection extent mismatch"
        );
        let mut reader = reader.lock();
        if streamed {
            reader.read_rows_streamed(&request)
        } else {
            reader.read_rows_windowed(&request)
        }
    }

    /// PLE rows being read on a helper thread (see `begin_prefill_read`).
    pub struct PlePrefillRead(Result<std::thread::JoinHandle<Result<Vec<PleNvfp4Row>>>>);

    impl PlePrefillRead {
        fn join(self) -> Result<Vec<PleNvfp4Row>> {
            self.0?
                .join()
                .map_err(|_| anyhow::anyhow!("Qwen4 PLE prefill read thread panicked"))?
        }
    }

    /// Atlas-native sparse PLE runtime. Only the 16 selected NVFP4 rows are
    /// staged per token; the 320M-row table remains in the O_DIRECT sidecar.
    pub struct Qwen4PleLayer {
        hasher: Arc<Qwen4PleHasher>,
        reader: Arc<Mutex<PleOffloadReader>>,
        scratch_submit: Mutex<()>,
        scratch_event: u64,
        scratch_poisoned: AtomicBool,
        key_proj: DenseWeight,
        value_proj: DenseWeight,
        norm_key: DenseWeight,
        norm_query: DenseWeight,
        norm_conv: DenseWeight,
        conv_weight: DenseWeight,
        records_dev: DevicePtr,
        scales_dev: DevicePtr,
        embedding_dev: DevicePtr,
        key_dev: DevicePtr,
        value_dev: DevicePtr,
        conv_state_dev: DevicePtr,
        conv_checkpoint_dev: DevicePtr,
        conv_intermediate_dev: DevicePtr,
        dequant_k: KernelHandle,
        fuse_k: KernelHandle,
        dense_gemv_k: KernelHandle,
        prefill: Option<Qwen4PlePrefillScratch>,
        hidden_size: usize,
        hc_count: usize,
        max_batch_size: usize,
        eps: f32,
    }

    impl Qwen4PleLayer {
        pub fn load(
            store: &WeightStore,
            config: &ModelConfig,
            gpu: &dyn GpuBackend,
            max_batch_size: usize,
        ) -> Result<Option<Self>> {
            if !config.is_qwen4_exp() {
                return Ok(None);
            }
            let manifest = config.ple_offload_manifest.as_deref().context(
                "Qwen4 PLE checkpoint is missing ple-offload/manifest.json; refusing to run without the position-learning enhancement table",
            )?;
            ensure!(
                config.hidden_size == PLE_EMBED,
                "Qwen4 PLE hidden size mismatch"
            );
            ensure!(config.hc_count == 4, "Qwen4 PLE requires hc_count=4");
            let prefix = format!("{}.layers.1.ple", config.weight_prefix);
            check_tensor(
                store,
                &format!("{prefix}.key_proj.weight"),
                &[10240, 2560],
                WeightDtype::BF16,
            )?;
            check_tensor(
                store,
                &format!("{prefix}.value_proj.weight"),
                &[2560, 2560],
                WeightDtype::BF16,
            )?;
            check_tensor(
                store,
                &format!("{prefix}.conv1d.weight"),
                &[10240, 1, 4],
                WeightDtype::BF16,
            )?;
            for name in ["norm_key.weight", "norm_query.weight", "norm_conv.weight"] {
                check_tensor(
                    store,
                    &format!("{prefix}.{name}"),
                    &[10240],
                    WeightDtype::BF16,
                )?;
            }

            let ep = format!("{prefix}.ple_embedding");
            let multipliers = read_i64::<3>(store, &format!("{ep}.layer_multipliers"), gpu)?;
            let vocab_sizes =
                read_i64::<QWEN4_PLE_HEADS>(store, &format!("{ep}.ngram_heads_vocab_sizes"), gpu)?;
            let offsets =
                read_i64::<QWEN4_PLE_HEADS>(store, &format!("{ep}.ngram_heads_offsets"), gpu)?;
            let hasher = Qwen4PleHasher::new(
                multipliers,
                vocab_sizes,
                offsets,
                config.eos_token_id,
                2_500_012,
            )?;

            // The published NVFP4 recipe explicitly ignores `*.ple.*`.
            // Preserve those checkpoint projections in BF16 rather than
            // introducing a second, unsupported quantization pass.
            let key_proj = dense(store, &format!("{prefix}.key_proj.weight"))?;
            let value_proj = dense(store, &format!("{prefix}.value_proj.weight"))?;
            let page_cache = std::env::var("ATLAS_PLE_PAGE_CACHE").ok().as_deref() == Some("1");
            let io_mode = if page_cache {
                PleIoMode::PageCache
            } else {
                PleIoMode::Direct
            };
            // Flash-Next already leaves only a few GiB of unified-memory
            // headroom on one Spark. Keep the private cache available as an
            // explicit opt-in, but do not duplicate sparse PLE rows by
            // default; a 10-request Weschera gate found no decode benefit and
            // the extra pressure could stall later requests.
            let cache_mb = std::env::var("ATLAS_PLE_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let io_queue_depth = configured_ple_io_queue_depth()?;
            tracing::info!(
                ple_io_queue_depth = io_queue_depth,
                ple_io_mode = ?io_mode,
                "configured Qwen4 PLE offload reader"
            );
            let reader = PleOffloadReader::open_with_mode(
                Path::new(manifest),
                io_queue_depth,
                cache_mb * 1024 * 1024,
                io_mode,
            )
            .with_context(|| format!("open Qwen4 PLE offload manifest {manifest}"))?;
            let residual = config.residual_width();
            let layer = Self {
                hasher: Arc::new(hasher),
                reader: Arc::new(Mutex::new(reader)),
                scratch_submit: Mutex::new(()),
                scratch_event: gpu.create_event()?,
                scratch_poisoned: AtomicBool::new(false),
                key_proj,
                value_proj,
                norm_key: dense(store, &format!("{prefix}.norm_key.weight"))?,
                norm_query: dense(store, &format!("{prefix}.norm_query.weight"))?,
                norm_conv: dense(store, &format!("{prefix}.norm_conv.weight"))?,
                conv_weight: dense(store, &format!("{prefix}.conv1d.weight"))?,
                records_dev: gpu.alloc(QWEN4_PLE_HEADS * PLE_RECORD_BYTES)?,
                scales_dev: gpu.alloc(QWEN4_PLE_HEADS * 4)?,
                embedding_dev: gpu.alloc(PLE_EMBED * 2)?,
                key_dev: gpu.alloc(residual * 2)?,
                value_dev: gpu.alloc(PLE_EMBED * 2)?,
                conv_state_dev: gpu.alloc(max_batch_size * residual * PLE_CONV_HISTORY * 2)?,
                conv_checkpoint_dev: gpu.alloc(max_batch_size * residual * PLE_CONV_HISTORY * 2)?,
                conv_intermediate_dev: gpu
                    .alloc(max_batch_size * PLE_MAX_SPEC_ROWS * residual * PLE_CONV_HISTORY * 2)?,
                dequant_k: gpu.kernel("qwen4_hyper", "qwen4_ple_dequant_rows")?,
                fuse_k: gpu.kernel("qwen4_hyper", "qwen4_ple_fuse_decode")?,
                dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
                prefill: None,
                hidden_size: config.hidden_size,
                hc_count: config.hc_count,
                max_batch_size,
                eps: config.rms_norm_eps as f32,
            };
            gpu.memset(
                layer.conv_state_dev,
                0,
                max_batch_size * residual * PLE_CONV_HISTORY * 2,
            )?;
            gpu.memset(
                layer.conv_checkpoint_dev,
                0,
                max_batch_size * residual * PLE_CONV_HISTORY * 2,
            )?;
            gpu.memset(
                layer.conv_intermediate_dev,
                0,
                max_batch_size * PLE_MAX_SPEC_ROWS * residual * PLE_CONV_HISTORY * 2,
            )?;
            tracing::info!(
                manifest,
                cache_mb,
                ?io_mode,
                "Qwen4 PLE sparse NVFP4 offload enabled"
            );
            Ok(Some(layer))
        }

        /// Allocate the opt-in whole-prompt arena only after the enclosing
        /// model is fully constructed. A later factory error then runs the
        /// model's synchronized Drop path instead of leaking raw pointers.
        pub fn initialize_prefill(
            &mut self,
            gpu: &dyn GpuBackend,
            max_prefill_tokens: usize,
        ) -> Result<()> {
            if std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
                .ok()
                .as_deref()
                != Some("1")
            {
                return Ok(());
            }
            ensure!(
                self.prefill.is_none(),
                "Qwen4 PLE prefill scratch is already initialized"
            );
            self.prefill = Some(Qwen4PlePrefillScratch::allocate(
                gpu,
                max_prefill_tokens,
                self.residual_width(),
            )?);
            Ok(())
        }

        /// Fetch, project, and inject one token. `prior_tokens` excludes the
        /// current input token, matching the official n-gram shift contract.
        pub fn forward_token(
            &self,
            current_token: u32,
            prior_tokens: &[u32],
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            // records/scales/embedding/key/value are shared staging buffers.
            // Serialize host enqueue order and chain their GPU use across
            // streams; a warm host-page hit can otherwise overtake unfinished
            // work that cold O_DIRECT latency happened to hide.
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
            );
            let selections = self.hasher.select_decode(current_token, prior_tokens);
            let request: Vec<_> = selections.iter().map(|s| (s.shard, s.row)).collect();
            let rows = self.reader.lock().read_rows(&request)?;
            let mut records = [0u8; QWEN4_PLE_HEADS * PLE_RECORD_BYTES];
            let mut scales = [0u8; QWEN4_PLE_HEADS * 4];
            for (head, row) in rows.iter().enumerate() {
                let start = head * PLE_RECORD_BYTES;
                records[start..start + PLE_RECORD_BYTES].copy_from_slice(&row.record);
                scales[head * 4..head * 4 + 4].copy_from_slice(&row.scale2.to_le_bytes());
            }
            gpu.stream_wait_event(stream, self.scratch_event)?;
            gpu.copy_h2d_group_on_stream(
                &[
                    HostToDeviceCopy::new(&records, self.records_dev),
                    HostToDeviceCopy::new(&scales, self.scales_dev),
                ],
                stream,
            )?;
            KernelLaunch::new(gpu, self.dequant_k)
                .grid([10, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.records_dev)
                .arg_ptr(self.scales_dev)
                .arg_ptr(self.embedding_dev)
                .launch(stream)?;
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                self.embedding_dev,
                &self.key_proj,
                self.key_dev,
                self.residual_width() as u32,
                PLE_EMBED as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                self.embedding_dev,
                &self.value_proj,
                self.value_dev,
                PLE_EMBED as u32,
                PLE_EMBED as u32,
                stream,
            )?;
            let state_stride = self.residual_width() * PLE_CONV_HISTORY * 2;
            KernelLaunch::new(gpu, self.fuse_k)
                .grid([self.hc_count as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(self.key_dev)
                .arg_ptr(self.value_dev)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(self.conv_state_dev.offset(slot_idx * state_stride))
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .arg_u32(u32::from(reset_state))
                .launch(stream)?;
            gpu.record_event(self.scratch_event, stream)
        }

        /// Start reading a prefill chunk's PLE rows on a helper thread. The rows
        /// depend only on token ids, so the caller issues this before layer 0
        /// and the O_DIRECT reads overlap the GPU work that precedes the PLE
        /// join (embedding + layer 0). Pass the handle to [`Self::forward_prefill`].
        pub fn begin_prefill_read(&self, current_tokens: &[u32], prior_tokens: &[u32]) -> PlePrefillRead {
            let hasher = Arc::clone(&self.hasher);
            let reader = Arc::clone(&self.reader);
            let current = current_tokens.to_vec();
            let prior = prior_tokens.to_vec();
            PlePrefillRead(
                std::thread::Builder::new()
                    .name("ple-prefill-read".into())
                    .spawn(move || read_prefill_rows(&hasher, &reader, &current, &prior, true))
                    .map_err(anyhow::Error::from),
            )
        }

        /// Fetch and inject a complete contiguous prefill chunk. Sparse row
        /// I/O remains host-driven, but every dense projection and PLE join is
        /// issued once for the whole chunk instead of once per token.
        /// `prefetched` must come from [`Self::begin_prefill_read`] for the same
        /// tokens; without it the rows are read here.
        #[allow(clippy::too_many_arguments)]
        pub fn forward_prefill(
            &self,
            current_tokens: &[u32],
            prior_tokens: &[u32],
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
            prefetched: Option<PlePrefillRead>,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(!current_tokens.is_empty(), "PLE prefill chunk is empty");
            let scratch = self.prefill.as_ref().context(
                "Qwen4 PLE whole-prompt path requires ATLAS_QWEN4_PLE_PREFILL_BATCH=1 at model load",
            )?;
            let num_tokens = current_tokens.len();
            ensure!(
                num_tokens <= scratch.max_tokens,
                "Qwen4 PLE prefill chunk {num_tokens} exceeds scratch bound {}",
                scratch.max_tokens
            );
            let num_tokens_u32 =
                u32::try_from(num_tokens).context("Qwen4 PLE prefill token count exceeds u32")?;

            // Every mutable staging allocation is shared between requests.
            // Keep row reads, H2D and GPU consumption in one event-chained
            // critical section just like the single-token path.
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
            );
            let rows = match prefetched {
                Some(read) => read.join()?,
                None => read_prefill_rows(
                    &self.hasher,
                    &self.reader,
                    current_tokens,
                    prior_tokens,
                    prefill_stream_selected(),
                )?,
            };
            ensure!(
                rows.len() == num_tokens * QWEN4_PLE_HEADS,
                "Qwen4 PLE prefill read extent mismatch"
            );

            let mut records = vec![0u8; rows.len() * PLE_RECORD_BYTES];
            let mut scales = vec![0u8; rows.len() * std::mem::size_of::<f32>()];
            for (index, row) in rows.iter().enumerate() {
                let record_start = index * PLE_RECORD_BYTES;
                records[record_start..record_start + PLE_RECORD_BYTES].copy_from_slice(&row.record);
                let scale_start = index * std::mem::size_of::<f32>();
                scales[scale_start..scale_start + std::mem::size_of::<f32>()]
                    .copy_from_slice(&row.scale2.to_le_bytes());
            }

            let dequant_items = num_tokens_u32
                .checked_mul(PLE_EMBED as u32)
                .context("Qwen4 PLE dequant grid overflow")?;
            let state_stride = self.state_stride_bytes();
            let state = self.conv_state_dev.offset(slot_idx * state_stride);
            if let Err(error) = gpu.stream_wait_event(stream, self.scratch_event) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE scratch predecessor wait failed; shared scratch poisoned",
                ));
            }
            let execution = (|| -> Result<()> {
                gpu.copy_h2d_group_on_stream(
                    &[
                        HostToDeviceCopy::new(&records, scratch.records_dev),
                        HostToDeviceCopy::new(&scales, scratch.scales_dev),
                    ],
                    stream,
                )?;
                KernelLaunch::new(gpu, scratch.dequant_k)
                    .grid([dequant_items.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.records_dev)
                    .arg_ptr(scratch.scales_dev)
                    .arg_ptr(scratch.embedding_dev)
                    .arg_u32(num_tokens_u32)
                    .launch(stream)?;
                ops::cublas_bf16_proj_dense(
                    scratch.embedding_dev,
                    &self.key_proj,
                    scratch.key_dev,
                    num_tokens_u32,
                    self.residual_width() as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
                ops::cublas_bf16_proj_dense(
                    scratch.embedding_dev,
                    &self.value_proj,
                    scratch.value_dev,
                    num_tokens_u32,
                    PLE_EMBED as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
                KernelLaunch::new(gpu, scratch.prepare_k)
                    .grid([num_tokens_u32, self.hc_count as u32, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(hyper)
                    .arg_ptr(scratch.key_dev)
                    .arg_ptr(scratch.value_dev)
                    .arg_ptr(self.norm_key.weight)
                    .arg_ptr(self.norm_query.weight)
                    .arg_ptr(self.norm_conv.weight)
                    .arg_ptr(scratch.gated_value_dev)
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(self.hidden_size as u32)
                    .arg_f32(self.eps)
                    .launch(stream)?;
                KernelLaunch::new(gpu, scratch.conv_inject_k)
                    .grid([
                        num_tokens_u32,
                        self.hc_count as u32,
                        (self.hidden_size as u32).div_ceil(256),
                    ])
                    .block([256, 1, 1])
                    .arg_ptr(hyper)
                    .arg_ptr(scratch.gated_value_dev)
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_ptr(self.conv_weight.weight)
                    .arg_ptr(state)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(u32::from(reset_state))
                    .launch(stream)?;
                KernelLaunch::new(gpu, scratch.commit_state_k)
                    .grid([(self.residual_width() as u32).div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_ptr(state)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(self.residual_width() as u32)
                    .arg_u32(u32::from(reset_state))
                    .launch(stream)?;
                gpu.synchronize(stream)
            })();
            if let Err(error) = execution {
                let fence = gpu
                    .synchronize(stream)
                    .and_then(|()| gpu.record_event(self.scratch_event, stream));
                return match fence {
                    Ok(()) => Err(error.context(
                        "Qwen4 PLE whole-prompt execution failed after shared scratch was fenced",
                    )),
                    Err(fence_error) => {
                        self.scratch_poisoned.store(true, Ordering::Release);
                        Err(error.context(format!(
                            "Qwen4 PLE whole-prompt execution failed and fencing failed ({fence_error:#}); shared scratch poisoned"
                        )))
                    }
                };
            }
            if let Err(error) = gpu.record_event(self.scratch_event, stream) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE post-success scratch fence failed; shared scratch poisoned",
                ));
            }
            tracing::info!(
                target: "atlas::qwen4_prefill",
                family = "ple",
                selector = "ATLAS_QWEN4_PLE_PREFILL_BATCH",
                projection = "cublaslt_bf16_non_bit_exact",
                projection_parity_required = true,
                performance_claim_allowed = false,
                num_tokens,
                heads = QWEN4_PLE_HEADS,
                hidden_size = self.hidden_size,
                hc_count = self.hc_count,
                reset_state,
                "QWEN4_PREFILL_ENGAGED"
            );
            Ok(())
        }

        pub fn residual_width(&self) -> usize {
            self.hidden_size * self.hc_count
        }

        pub fn destroy_owned_resources(&self, gpu: &dyn GpuBackend) -> Result<()> {
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE scratch is poisoned; refusing unsafe device-buffer teardown"
            );
            let teardown_stream = gpu.default_stream();
            gpu.stream_wait_event(teardown_stream, self.scratch_event)
                .context("wait for final Qwen4 PLE scratch owner before teardown")?;
            gpu.synchronize(teardown_stream)
                .context("synchronize final Qwen4 PLE scratch owner before teardown")?;
            let mut errors = Vec::new();
            if let Some(scratch) = &self.prefill
                && let Err(error) = scratch.release(gpu)
            {
                errors.push(format!("prefill scratch: {error:#}"));
            }
            if let Err(error) = gpu.destroy_event(self.scratch_event) {
                errors.push(format!("scratch event: {error:#}"));
            }
            ensure!(
                errors.is_empty(),
                "Qwen4 PLE owned-resource release failures: {}",
                errors.join("; ")
            );
            Ok(())
        }

        fn state_stride_bytes(&self) -> usize {
            self.residual_width() * PLE_CONV_HISTORY * 2
        }

        /// Diagnostic-only exact snapshot of the slot-local live/checkpoint
        /// histories. The caller supplies the producer stream whose writes
        /// must complete before either pageable-host read.
        pub(crate) fn parity_snapshot(
            &self,
            slot_idx: usize,
            gpu: &dyn GpuBackend,
            producer_stream: u64,
        ) -> Result<(usize, Vec<u8>, Vec<u8>)> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE parity slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            ensure!(stride > 0, "PLE parity state stride is zero");
            gpu.synchronize(producer_stream)?;
            let mut live = vec![0u8; stride];
            let mut checkpoint = vec![0u8; stride];
            gpu.copy_d2h(self.conv_state_dev.offset(slot_idx * stride), &mut live)?;
            gpu.copy_d2h(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                &mut checkpoint,
            )?;
            Ok((stride, live, checkpoint))
        }

        /// Capture the exact post-PLE boundary for serial-vs-whole-prompt
        /// qualification. This is inert unless the complete parity-specific
        /// environment is present. A receipt is the final commit marker; no
        /// timing or throughput field is produced here.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn capture_post_prefill_parity(
            &self,
            hidden: DevicePtr,
            request_tokens: &[u32],
            prior_tokens: &[u32],
            current_tokens: &[u32],
            chunk_start: usize,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            producer_stream: u64,
        ) -> Result<()> {
            let Some(config) = parity_capture_config()? else {
                return Ok(());
            };
            ensure!(
                cfg!(target_endian = "little"),
                "PLE parity BF16LE capture requires a little-endian host"
            );
            ensure!(
                self.residual_width() == PLE_PARITY_WIDTH,
                "PLE parity residual width {} != {PLE_PARITY_WIDTH}",
                self.residual_width()
            );
            ensure!(!current_tokens.is_empty(), "PLE parity chunk is empty");
            let chunk_end = chunk_start
                .checked_add(current_tokens.len())
                .context("PLE parity chunk range overflow")?;
            ensure!(
                chunk_end <= request_tokens.len()
                    && request_tokens[chunk_start..chunk_end] == *current_tokens,
                "PLE parity chunk is not the exact ordered request-token slice"
            );
            ensure!(
                reset_state == (chunk_start == 0),
                "PLE parity reset/continuation marker disagrees with chunk start"
            );
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE parity slot {slot_idx} out of range"
            );
            let hidden_bytes_len = current_tokens
                .len()
                .checked_mul(PLE_PARITY_WIDTH)
                .and_then(|elements| elements.checked_mul(2))
                .context("PLE parity hidden extent overflow")?;
            let state_bytes_len = self.state_stride_bytes();
            ensure!(
                state_bytes_len == PLE_PARITY_WIDTH * PLE_CONV_HISTORY * 2,
                "PLE parity state extent {state_bytes_len} is not exact [10240,9] BF16"
            );
            let state_offset = slot_idx
                .checked_mul(state_bytes_len)
                .context("PLE parity slot state offset overflow")?;

            gpu.synchronize(producer_stream)
                .context("synchronize actual post-PLE producer stream")?;
            let mut hidden_bytes = vec![0u8; hidden_bytes_len];
            let mut live_bytes = vec![0u8; state_bytes_len];
            let mut checkpoint_bytes = vec![0u8; state_bytes_len];
            gpu.copy_d2h(hidden, &mut hidden_bytes)
                .context("capture exact post-PLE hidden BF16 bytes")?;
            gpu.copy_d2h(self.conv_state_dev.offset(state_offset), &mut live_bytes)
                .context("capture exact post-PLE live state BF16 bytes")?;
            gpu.copy_d2h(
                self.conv_checkpoint_dev.offset(state_offset),
                &mut checkpoint_bytes,
            )
            .context("capture exact post-PLE checkpoint state BF16 bytes")?;

            root_is_stable(&config)?;
            let arm = if config.selector == 0 {
                "serial0"
            } else {
                "whole_prompt1"
            };
            let phase = if reset_state { "reset" } else { "continuation" };
            let frame_name = format!(
                "frame-{}-{arm}-m{}-s{chunk_start}-n{}-{phase}",
                config.nonce,
                request_tokens.len(),
                current_tokens.len()
            );
            let frame = config.root.join(&frame_name);
            let mut owned_frame = create_owned_frame(frame.clone(), config.root_dev)?;
            let staging_frame = owned_frame.path().to_path_buf();
            let frame_identity = owned_frame.identity();
            let publication = (|| -> Result<()> {
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "post_ple_hidden.bf16le",
                    &hidden_bytes,
                )?);
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "ple_live.bf16le",
                    &live_bytes,
                )?);
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "ple_checkpoint.bf16le",
                    &checkpoint_bytes,
                )?);
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&staging_frame, artifact)?;
                }
                let hidden_artifact = &owned_frame.artifacts[0];
                let live_artifact = &owned_frame.artifacts[1];
                let checkpoint_artifact = &owned_frame.artifacts[2];
                let root_receipt = config
                    .root
                    .to_str()
                    .context("canonical PLE parity root is not UTF-8")?;
                let hidden_receipt = serde_json::json!({
                    "file": hidden_artifact.name,
                    "dtype": "bf16le",
                    "shape": [current_tokens.len(), PLE_PARITY_WIDTH],
                    "bytes": hidden_artifact.bytes,
                    "sha256": hidden_artifact.sha256.as_str(),
                    "dev": hidden_artifact.dev,
                    "ino": hidden_artifact.ino,
                    "mode": "0444"
                });
                let live_receipt = serde_json::json!({
                    "file": live_artifact.name,
                    "dtype": "bf16le",
                    "shape": [PLE_PARITY_WIDTH, PLE_CONV_HISTORY],
                    "bytes": live_artifact.bytes,
                    "sha256": live_artifact.sha256.as_str(),
                    "dev": live_artifact.dev,
                    "ino": live_artifact.ino,
                    "mode": "0444"
                });
                let checkpoint_receipt = serde_json::json!({
                    "file": checkpoint_artifact.name,
                    "dtype": "bf16le",
                    "shape": [PLE_PARITY_WIDTH, PLE_CONV_HISTORY],
                    "bytes": checkpoint_artifact.bytes,
                    "sha256": checkpoint_artifact.sha256.as_str(),
                    "dev": checkpoint_artifact.dev,
                    "ino": checkpoint_artifact.ino,
                    "mode": "0444"
                });
                let artifact_receipts = serde_json::json!({
                    "post_ple_hidden": hidden_receipt,
                    "ple_live": live_receipt,
                    "ple_checkpoint": checkpoint_receipt
                });
                let receipt = serde_json::json!({
                    "schema": "atlas-qwen38-flash-next-post-ple-parity-frame-v1",
                    "boundary": "post_ple_pre_layer1",
                    "performance_claim_allowed": false,
                    "producer_stream_synchronized": true,
                    "producer_stream": producer_stream,
                    "pid": std::process::id(),
                    "nonce": config.nonce.as_str(),
                    "capture_root": root_receipt,
                    "capture_root_dev": config.root_dev,
                    "capture_root_ino": config.root_ino,
                    "frame": frame_name.as_str(),
                    "frame_dev": frame_identity.0,
                    "frame_ino": frame_identity.1,
                    "frame_commit_mode": "0500",
                    "arm": arm,
                    "selector": config.selector,
                    "request_m": request_tokens.len(),
                    "request_tokens_encoding": "u32le",
                    "request_tokens_sha256": token_sha256(request_tokens)?,
                    "ple_prior_m": prior_tokens.len(),
                    "ple_prior_tokens_sha256": token_sha256(prior_tokens)?,
                    "ple_ordered_m": prior_tokens.len().checked_add(current_tokens.len())
                        .context("PLE parity ordered token count overflow")?,
                    "ple_ordered_tokens_sha256": token_parts_sha256(&[prior_tokens, current_tokens])?,
                    "chunk_start": chunk_start,
                    "chunk_m": current_tokens.len(),
                    "chunk_tokens_encoding": "u32le",
                    "chunk_tokens_sha256": token_sha256(current_tokens)?,
                    "slot_idx": slot_idx,
                    "reset_state": reset_state,
                    "continuation": !reset_state,
                    "artifacts": artifact_receipts
                });
                let mut receipt_bytes = serde_json::to_vec(&receipt)
                    .context("serialize canonical PLE parity receipt")?;
                receipt_bytes.push(b'\n');
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "receipt.json",
                    &receipt_bytes,
                )?);
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&staging_frame, artifact)?;
                }
                owned_frame
                    .handle
                    .as_ref()
                    .expect("created frame handle is retained")
                    .sync_all()
                    .context("sync private PLE parity staging frame")?;
                File::open(&config.root)
                    .context("open PLE parity root for durability")?
                    .sync_all()
                    .context("sync PLE parity root")?;
                root_is_stable(&config)?;
                let final_frame = fs::symlink_metadata(&staging_frame)
                    .context("re-stat private PLE parity staging frame before publish")?;
                ensure!(
                    final_frame.is_dir()
                        && final_frame.dev() == frame_identity.0
                        && final_frame.ino() == frame_identity.1
                        && final_frame.permissions().mode() & 0o777 == 0o700,
                    "PLE parity staging frame identity or pre-publish mode changed"
                );
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&staging_frame, artifact)?;
                }
                // Seal the descriptor before the sole no-replace publication.
                // Any later failure remains rollback-safe because the owned
                // path is switched to the published name immediately after
                // the atomic rename.
                owned_frame
                    .handle
                    .as_ref()
                    .expect("created frame handle is retained")
                    .set_permissions(fs::Permissions::from_mode(0o500))
                    .context("seal PLE parity staging frame mode0500")?;
                owned_frame.publish_noreplace().context(
                    "atomically publish PLE parity frame without replacing a canonical path",
                )?;
                File::open(&config.root)
                    .context("open PLE parity root after publication")?
                    .sync_all()
                    .context("sync published PLE parity directory entry")?;
                root_is_stable(&config)?;
                let published =
                    fs::symlink_metadata(&frame).context("re-stat published PLE parity frame")?;
                ensure!(
                    published.is_dir()
                        && published.dev() == frame_identity.0
                        && published.ino() == frame_identity.1
                        && published.permissions().mode() & 0o777 == 0o500,
                    "published PLE parity frame identity or mode changed"
                );
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&frame, artifact)?;
                }
                // No fallible operation follows this success assignment.
                owned_frame.committed = true;
                Ok(())
            })();
            if let Err(error) = publication {
                return Err(error).context("publish exact post-PLE parity frame");
            }
            tracing::info!(
                target: "atlas::qwen4_ple_parity",
                frame = %frame.display(),
                arm,
                request_m = request_tokens.len(),
                chunk_start,
                chunk_m = current_tokens.len(),
                reset_state,
                performance_claim_allowed = false,
                "QWEN4_PLE_PARITY_CAPTURED"
            );
            Ok(())
        }

        /// Restore only the live PLE history captured after a wide verify.
        /// The normal commit still selects and mirrors the accepted
        /// intermediate; this merely reconstructs the verifier's pre-commit
        /// live-state identity after the serial diagnostic replay.
        pub(crate) fn parity_restore_live(
            &self,
            slot_idx: usize,
            stride: usize,
            live: &[u8],
            gpu: &dyn GpuBackend,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE parity restore slot {slot_idx} out of range"
            );
            ensure!(
                stride == self.state_stride_bytes() && live.len() == stride,
                "PLE parity restore stride/extent mismatch"
            );
            gpu.copy_h2d(live, self.conv_state_dev.offset(slot_idx * stride))
        }

        /// Clear every request-local PLE history buffer for a reused slot.
        /// PLE state is allocated outside `SsmStatePool`, so the pool's slot
        /// reset cannot cover it.
        pub fn zero_slot(&self, slot_idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.memset_async(
                self.conv_state_dev.offset(slot_idx * stride),
                0,
                stride,
                stream,
            )?;
            gpu.memset_async(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                0,
                stride,
                stream,
            )?;
            gpu.memset_async(
                self.conv_intermediate_dev
                    .offset(slot_idx * PLE_MAX_SPEC_ROWS * stride),
                0,
                PLE_MAX_SPEC_ROWS * stride,
                stream,
            )
        }

        /// Save the canonical PLE convolution history at a speculative boundary.
        pub fn checkpoint(&self, slot_idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(slot_idx * stride),
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        /// Preserve the state after one verify row for partial acceptance.
        pub fn save_intermediate(
            &self,
            slot_idx: usize,
            row: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(
                row < PLE_MAX_SPEC_ROWS,
                "PLE speculative row {row} exceeds {PLE_MAX_SPEC_ROWS}"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(slot_idx * stride),
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + row) * stride),
                stride,
                stream,
            )
        }

        /// Restore the pre-verify state before a verify replay.
        pub fn restore_checkpoint(
            &self,
            slot_idx: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                self.conv_state_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        /// Commit a partial rollback and make it the next checkpoint.
        pub fn rollback_and_checkpoint(
            &self,
            slot_idx: usize,
            num_accepted: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(
                num_accepted <= PLE_MAX_SPEC_ROWS,
                "Qwen4 PLE rollback accepts at most {PLE_MAX_SPEC_ROWS} rows, got {num_accepted}"
            );
            let stride = self.state_stride_bytes();
            let src = if num_accepted == 0 {
                self.conv_checkpoint_dev.offset(slot_idx * stride)
            } else {
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + num_accepted - 1) * stride)
            };
            let live = self.conv_state_dev.offset(slot_idx * stride);
            gpu.copy_d2d_async(src, live, stride, stream)?;
            gpu.copy_d2d_async(
                live,
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        pub fn owned_device_buffers(&self) -> [DevicePtr; 2] {
            [self.records_dev, self.scales_dev]
        }

        pub fn scratch_device_buffers(&self) -> [DevicePtr; 6] {
            [
                self.embedding_dev,
                self.key_dev,
                self.value_dev,
                self.conv_state_dev,
                self.conv_checkpoint_dev,
                self.conv_intermediate_dev,
            ]
        }

        pub fn prefill_device_buffers(&self) -> Option<[DevicePtr; 7]> {
            self.prefill
                .as_ref()
                .map(Qwen4PlePrefillScratch::device_buffers)
        }
    }

    fn check_tensor(
        store: &WeightStore,
        name: &str,
        shape: &[usize],
        dtype: WeightDtype,
    ) -> Result<()> {
        let tensor = store.get(name)?;
        ensure!(
            tensor.shape == shape,
            "{name} shape {:?} != {shape:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == dtype,
            "{name} dtype {:?} != {dtype:?}",
            tensor.dtype
        );
        Ok(())
    }

    fn read_i64<const N: usize>(
        store: &WeightStore,
        name: &str,
        gpu: &dyn GpuBackend,
    ) -> Result<[i64; N]> {
        check_tensor(store, name, &[N], WeightDtype::Int64)?;
        let mut bytes = vec![0u8; N * 8];
        gpu.copy_d2h(store.get(name)?.ptr, &mut bytes)?;
        Ok(std::array::from_fn(|i| {
            i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("fixed i64 slice"))
        }))
    }

    #[cfg(test)]
    mod parity_capture_tests {
        use super::*;

        fn private_test_root(label: &str) -> PathBuf {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "atlas-qwen4-ple-{label}-{}-{stamp}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            root
        }

        #[test]
        fn dependency_free_sha256_matches_standard_vectors() {
            assert_eq!(
                sha256_hex(b"").unwrap(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            );
            assert_eq!(
                sha256_hex(b"abc").unwrap(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
            assert_eq!(
                sha256_hex(&[b'a'; 64]).unwrap(),
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
            );
            assert_eq!(
                token_sha256(&[1, 2]).unwrap(),
                "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f"
            );
            assert_eq!(
                token_parts_sha256(&[&[1], &[2]]).unwrap(),
                "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f"
            );
        }

        #[test]
        fn parity_nonce_is_exact_lowercase_hex64() {
            assert!(valid_parity_nonce(&"a5".repeat(32)));
            assert!(!valid_parity_nonce(&"A5".repeat(32)));
            assert!(!valid_parity_nonce(&"a5".repeat(31)));
            assert!(!valid_parity_nonce(&format!("{}g", "a".repeat(63))));
            assert_eq!(PLE_PARITY_WIDTH * PLE_CONV_HISTORY * 2, 184_320);
            assert_eq!(38 * PLE_PARITY_WIDTH * 2, 778_240);
            assert_eq!(2013 * PLE_PARITY_WIDTH * 2, 41_226_240);
        }

        #[test]
        fn first_artifact_metadata_failure_is_absent_and_retryable() {
            let root = private_test_root("artifact-admission");
            let frame = root.join("frame");
            fs::create_dir(&frame).unwrap();
            fs::set_permissions(&frame, fs::Permissions::from_mode(0o700)).unwrap();
            let path = frame.join("post_ple_hidden.bf16le");
            let failed = write_sealed_artifact_with_admission(
                &frame,
                "post_ple_hidden.bf16le",
                b"first",
                |_| Err(std::io::Error::other("injected first metadata failure")),
            );
            assert!(failed.is_err());
            assert!(!path.exists());
            assert!(write_sealed_artifact(&frame, "post_ple_hidden.bf16le", b"retry").is_ok());
            assert_eq!(fs::read(&path).unwrap(), b"retry");
            fs::remove_file(path).unwrap();
            fs::remove_dir(frame).unwrap();
            fs::remove_dir(root).unwrap();
        }

        #[test]
        fn frame_admission_failures_are_absent_and_retryable() {
            let root = private_test_root("frame-admission");
            let root_dev = fs::symlink_metadata(&root).unwrap().dev();
            let frame = root.join("same-frame-name");
            let failed = create_owned_frame_with_admission(frame.clone(), root_dev, |_| {
                Err(std::io::Error::other("injected frame metadata failure"))
            });
            assert!(failed.is_err());
            assert!(!frame.exists());

            let mismatch = create_owned_frame_with_admission(frame.clone(), root_dev, |_| {
                fs::symlink_metadata(&root)
            });
            assert!(mismatch.is_err());
            assert!(!frame.exists());

            let retry = create_owned_frame(frame.clone(), root_dev).unwrap();
            assert!(!frame.exists());
            assert!(retry.path().is_dir());
            drop(retry);
            assert!(!frame.exists());
            fs::remove_dir(root).unwrap();
        }

        #[test]
        fn post_mkdir_open_failure_is_absent_and_retryable() {
            let root = private_test_root("frame-open");
            let root_dev = fs::symlink_metadata(&root).unwrap().dev();
            let frame = root.join("same-frame-name");
            let failed = create_owned_frame_with_open_and_admission(
                frame.clone(),
                root_dev,
                |_| Err(std::io::Error::other("injected frame open failure")),
                |path| fs::symlink_metadata(path),
            );
            assert!(failed.is_err());
            assert!(!frame.exists());
            let quarantined: Vec<PathBuf> = fs::read_dir(&root)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            assert_eq!(quarantined.len(), 1);
            assert!(
                quarantined[0]
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(".same-frame-name.staging-")
            );
            assert!(quarantined[0].is_dir());
            let retry = create_owned_frame(frame.clone(), root_dev).unwrap();
            assert!(!frame.exists());
            assert!(retry.path().is_dir());
            drop(retry);
            assert!(!frame.exists());
            fs::remove_dir(&quarantined[0]).unwrap();
            fs::remove_dir(root).unwrap();
        }

        #[test]
        fn foreign_replacements_survive_cleanup_guards() {
            let root = private_test_root("foreign-replacement");
            let artifact_frame = root.join("artifact-frame");
            fs::create_dir(&artifact_frame).unwrap();
            fs::set_permissions(&artifact_frame, fs::Permissions::from_mode(0o700)).unwrap();
            let artifact = artifact_frame.join("receipt.json");
            let displaced_artifact = artifact_frame.join("owned-displaced");
            let failed = write_sealed_artifact_with_admission(
                &artifact_frame,
                "receipt.json",
                b"owned",
                |_| {
                    fs::rename(&artifact, &displaced_artifact)?;
                    fs::write(&artifact, b"foreign")?;
                    Err(std::io::Error::other("injected replacement"))
                },
            );
            assert!(failed.is_err());
            assert_eq!(fs::read(&artifact).unwrap(), b"foreign");
            fs::remove_file(artifact).unwrap();
            fs::remove_file(displaced_artifact).unwrap();
            fs::remove_dir(artifact_frame).unwrap();

            let root_dev = fs::symlink_metadata(&root).unwrap().dev();
            let preopen_frame = root.join("preopen-frame");
            fs::create_dir(&preopen_frame).unwrap();
            fs::set_permissions(&preopen_frame, fs::Permissions::from_mode(0o711)).unwrap();
            let mut unpublished = create_owned_frame(preopen_frame.clone(), root_dev).unwrap();
            let private_stage = unpublished.path().to_path_buf();
            assert!(unpublished.publish_noreplace().is_err());
            assert_eq!(
                fs::symlink_metadata(&preopen_frame)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o711
            );
            drop(unpublished);
            assert!(!private_stage.exists());
            fs::remove_dir(preopen_frame).unwrap();

            let frame = root.join("frame");
            let displaced_frame = root.join("owned-staging-displaced");
            let mut foreign_stage = None;
            let failed = create_owned_frame_with_admission(frame.clone(), root_dev, |stage| {
                fs::rename(stage, &displaced_frame)?;
                fs::create_dir(stage)?;
                fs::set_permissions(stage, fs::Permissions::from_mode(0o711))?;
                foreign_stage = Some(stage.to_path_buf());
                Err(std::io::Error::other("injected frame replacement"))
            });
            assert!(failed.is_err());
            assert!(!frame.exists());
            assert!(displaced_frame.is_dir());
            let foreign_stage = foreign_stage.unwrap();
            assert_eq!(
                fs::symlink_metadata(&foreign_stage)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o711
            );
            fs::remove_dir(foreign_stage).unwrap();
            fs::remove_dir(displaced_frame).unwrap();
            fs::remove_dir(root).unwrap();
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use runtime::{PlePrefillRead, Qwen4PleLayer, prefill_stream_level};

#[cfg(test)]
mod tests {
    use super::*;

    fn hasher() -> Qwen4PleHasher {
        Qwen4PleHasher::new(
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ],
            [
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ],
            248_044,
            2_500_012,
        )
        .unwrap()
    }

    #[test]
    fn selections_match_official_torch_reference() {
        let got = hasher().select_decode(3, &[1, 2]);
        let expected = [
            (5, 2_105_657),
            (9, 1_910_767),
            (19, 1_813_339),
            (28, 2_177_092),
            (34, 1_060_412),
            (41, 1_521_518),
            (52, 963_195),
            (58, 1_885_506),
            (64, 1_522_309),
            (78, 938_190),
            (87, 1_923_521),
            (88, 810_726),
            (99, 518_953),
            (110, 227_210),
            (113, 1_796_643),
            (123, 2_143_747),
        ];
        for (selection, &(shard, row)) in got.iter().zip(&expected) {
            assert_eq!(*selection, PleRowSelection { shard, row });
        }
    }

    #[test]
    fn eos_resets_ngram_context() {
        assert_eq!(
            hasher().select_decode(3, &[1, 248_044]),
            hasher().select_decode(3, &[])
        );
    }
}
