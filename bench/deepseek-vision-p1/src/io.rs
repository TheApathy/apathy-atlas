// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
pub const MAX_FILE: usize = 64 * 1024 * 1024;
pub fn absolute(path: &Path) -> Result<PathBuf> {
    ensure!(
        path.is_absolute(),
        "absolute path required: {}",
        path.display()
    );
    Ok(path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", path.display()))?)
}
pub fn read(path: &Path, cap: usize) -> Result<Vec<u8>> {
    ensure!(cap <= MAX_FILE, "read cap too large");
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let m = f.metadata()?;
    ensure!(
        m.is_file() && m.len() > 0 && m.len() <= cap as u64,
        "file bound/type: {}",
        path.display()
    );
    let mut raw = Vec::with_capacity(m.len() as usize);
    f.take(cap as u64 + 1).read_to_end(&mut raw)?;
    ensure!(
        raw.len() as u64 == m.len(),
        "file changed length while reading"
    );
    Ok(raw)
}
pub fn hash(raw: &[u8]) -> Result<String> {
    let mut child = Command::new("/usr/bin/sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let write = child.stdin.take().unwrap().write_all(raw);
    let output = child.wait_with_output()?;
    write?;
    ensure!(output.status.success(), "sha256sum failed");
    let s = std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .unwrap_or("");
    ensure!(crate::contract::sha_syntax(s), "bad sha256sum response");
    Ok(s.to_owned())
}
pub fn pinned(path: &Path, sha: &str, cap: usize) -> Result<Vec<u8>> {
    let raw = read(path, cap)?;
    ensure!(hash(&raw)? == sha, "SHA256 mismatch: {}", path.display());
    Ok(raw)
}
pub fn json(raw: &[u8]) -> Result<Value> {
    Ok(serde_json::from_slice(raw)?)
}
pub fn receipt(path: &Path, raw: &[u8]) -> Result<Value> {
    Ok(json!({"path":absolute(path)?,"bytes":raw.len(),"sha256":hash(raw)?}))
}
pub fn verify_receipt(v: &Value) -> Result<Vec<u8>> {
    let p = Path::new(v["path"].as_str().context("receipt path")?);
    let bytes = v["bytes"].as_u64().context("receipt bytes")?;
    ensure!(
        bytes > 0 && bytes <= MAX_FILE as u64,
        "receipt size out of bounds"
    );
    let canonical = absolute(p)?;
    ensure!(canonical == p, "receipt path must be canonical");
    let raw = pinned(
        p,
        v["sha256"].as_str().context("receipt SHA")?,
        bytes as usize,
    )?;
    ensure!(raw.len() as u64 == bytes, "receipt size mismatch");
    Ok(raw)
}
pub fn fresh(path: &Path) -> Result<PathBuf> {
    ensure!(path.is_absolute(), "fresh output must be absolute");
    ensure!(!path.exists(), "output already exists");
    let parent = absolute(path.parent().context("output parent")?)?;
    let out = parent.join(path.file_name().context("output filename")?);
    fs::create_dir(&out)?;
    Ok(out)
}
pub fn save(dir: &Path, name: &str, raw: &[u8]) -> Result<Value> {
    ensure!(
        !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            && name != "."
            && name != "..",
        "bad output filename"
    );
    let path = dir.join(name);
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    f.write_all(raw)?;
    f.sync_all()?;
    receipt(&path, raw)
}
pub fn save_json(dir: &Path, name: &str, v: &Value) -> Result<Value> {
    let mut raw = serde_json::to_vec_pretty(v)?;
    raw.push(b'\n');
    save(dir, name, &raw)
}
pub fn library_receipt(path: &Path) -> Result<Value> {
    let p = absolute(path)?;
    let m = fs::metadata(&p)?;
    ensure!(
        m.is_file() && m.len() > 0 && m.len() < 2 * 1024 * 1024 * 1024,
        "library file bound"
    );
    let output = Command::new("/usr/bin/sha256sum")
        .arg("--")
        .arg(&p)
        .output()?;
    ensure!(output.status.success(), "library sha256sum failed");
    let text = std::str::from_utf8(&output.stdout)?;
    let sha = text.split_whitespace().next().unwrap_or("");
    ensure!(crate::contract::sha_syntax(sha), "library SHA syntax");
    Ok(json!({"path":p,"bytes":m.len(),"sha256":sha}))
}
