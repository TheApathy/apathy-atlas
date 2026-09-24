// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint identity: which (engine, model, recipe) produced a state.
//!
//! A checkpoint is only byte-exact under the exact binary (kernels are
//! compiled into it), the exact weights and the exact numerics-affecting
//! configuration. Everything that could change a single output bit is
//! folded into one SHA-256 model key; files under a different key are
//! never read. The key is deliberately over-inclusive: a spurious miss
//! costs one prefill, a spurious hit returns wrong logits.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use ring::digest::{Context as Sha, SHA256};

use super::format::FORMAT_VERSION;

/// Incremental SHA-256 over labelled fields.
pub struct KeyBuilder(Sha);

impl Default for KeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyBuilder {
    pub fn new() -> Self {
        let mut s = Sha::new(&SHA256);
        s.update(b"atlas-ctx-key");
        s.update(&FORMAT_VERSION.to_le_bytes());
        Self(s)
    }

    /// Add a labelled field. Lengths are framed so fields cannot alias.
    pub fn add(&mut self, label: &str, value: &[u8]) -> &mut Self {
        for part in [label.as_bytes(), value] {
            self.0.update(&(part.len() as u64).to_le_bytes());
            self.0.update(part);
        }
        self
    }

    pub fn finish(self) -> [u8; 32] {
        let d = self.0.finish();
        let mut out = [0u8; 32];
        out.copy_from_slice(d.as_ref());
        out
    }
}

/// SHA-256 of a whole file, streamed.
pub fn file_digest(path: &Path) -> Result<[u8; 32]> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut s = Sha::new(&SHA256);
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        s.update(&buf[..n]);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(s.finish().as_ref());
    Ok(out)
}

/// Identity of the running engine: the executable holds every kernel.
pub fn engine_digest() -> Result<[u8; 32]> {
    file_digest(Path::new("/proc/self/exe"))
}

const SAMPLE: usize = 1 << 16;

/// Cheap weights fingerprint: small metadata files in full, every weight
/// file by name, length, and the first 64 KiB (safetensors headers carry
/// the tensor table, so a re-quantised checkpoint changes it).
pub fn dir_fingerprint(dir: &Path) -> Result<[u8; 32]> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let mut k = KeyBuilder::new();
    for name in names {
        let p = dir.join(&name);
        let len = std::fs::metadata(&p)?.len();
        k.add("file", name.as_bytes()).add("len", &len.to_le_bytes());
        let small = name.ends_with(".json") || name.ends_with(".txt") || name.ends_with(".yaml");
        if small && len <= 64 << 20 {
            k.add("body", &std::fs::read(&p)?);
        } else {
            let mut head = vec![0u8; SAMPLE.min(len as usize)];
            File::open(&p)?.read_exact(&mut head)?;
            k.add("head", &head);
        }
    }
    Ok(k.finish())
}

/// `ATLAS_*` environment, sorted, excluding this feature's own knobs. A
/// value naming an existing file also contributes its length and mtime
/// (e.g. `ATLAS_FLASHINFER_SM121_LIB` — a rebuilt `.so` changes numerics).
pub fn recipe_env() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("ATLAS_") && !k.starts_with("ATLAS_CTX_"))
        .map(|(k, v)| {
            let stamp = std::fs::metadata(&v)
                .ok()
                .filter(|m| m.is_file())
                .map(|m| {
                    let mt = m
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_nanos())
                        .unwrap_or(0);
                    format!(" [len={} mtime={mt}]", m.len())
                })
                .unwrap_or_default();
            (k, format!("{v}{stamp}"))
        })
        .collect();
    out.sort();
    out
}

/// 64-bit digest of a token prefix: the checkpoint file name and index key.
pub fn prefix_hash(tokens: &[u32]) -> u64 {
    let mut s = Sha::new(&SHA256);
    for t in tokens {
        s.update(&t.to_le_bytes());
    }
    let d = s.finish();
    u64::from_le_bytes(d.as_ref()[..8].try_into().unwrap_or([0; 8]))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_are_framed() {
        let mut a = KeyBuilder::new();
        a.add("ab", b"c");
        let mut b = KeyBuilder::new();
        b.add("a", b"bc");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn prefix_hash_depends_on_every_token() {
        let t: Vec<u32> = (0..100).collect();
        let mut u = t.clone();
        u[57] += 1;
        assert_ne!(prefix_hash(&t), prefix_hash(&u));
        assert_eq!(prefix_hash(&t), prefix_hash(&t.clone()));
        assert_ne!(prefix_hash(&t[..99]), prefix_hash(&t));
    }

    #[test]
    fn dir_fingerprint_sees_weight_edits() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("config.json"), b"{}").unwrap();
        std::fs::write(d.path().join("model.safetensors"), vec![1u8; 1000]).unwrap();
        let a = dir_fingerprint(d.path()).unwrap();
        std::fs::write(d.path().join("model.safetensors"), vec![2u8; 1000]).unwrap();
        let b = dir_fingerprint(d.path()).unwrap();
        std::fs::write(d.path().join("config.json"), b"{ }").unwrap();
        let c = dir_fingerprint(d.path()).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }
}
