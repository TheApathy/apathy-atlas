// SPDX-License-Identifier: AGPL-3.0-only
//! Standalone CPU-orchestrated local HTTP census; no model/runtime linkage.
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type Result<T> = std::result::Result<T, String>;
mod evidence;
mod options;
mod prompts;
mod provenance;
mod transport;
mod workload;
use evidence::*;
use options::*;
use prompts::*;
use provenance::*;
use transport::*;
use workload::*;

fn check(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}
fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(bytes).map_err(|e| e.to_string())
}
fn save_json(path: &Path, value: &Value) -> Result<()> {
    save(
        path,
        &serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
}
fn str_field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing/non-string {key}"))
}
fn number(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| format!("missing/non-integral {key}"))
}
fn sha256(path: &Path) -> Result<String> {
    let out = Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    check(out.status.success(), "sha256sum failed")?;
    let hash = String::from_utf8(out.stdout)
        .map_err(|e| e.to_string())?
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_owned();
    check(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid sha256 output",
    )?;
    Ok(hash)
}
fn run(options: Options) -> Result<()> {
    let provenance_bytes = read(&options.provenance)?;
    let provenance: Value = serde_json::from_slice(&provenance_bytes)
        .map_err(|e| format!("invalid provenance JSON: {e}"))?;
    let binding = Binding::new(&provenance, &options)?;
    fs::create_dir(&options.output)
        .map_err(|e| format!("output must be a fresh directory: {e}"))?;
    save(&options.output.join("provenance.json"), &provenance_bytes)?;
    save_json(
        &options.output.join("admission.json"),
        &json!({"pid_start_ticks":binding.start,"binary_sha256_verified":true,"config_sha256_verified":true,"port_owner_verified":true,
        "selected_bins":options.bins,
        "prompt_format":options.prompt_format,
        "harness_sha256":sha256(&std::env::current_exe().map_err(|e| e.to_string())?)?}),
    )?;
    match workload(&options, &binding) {
        Ok(summary) => {
            save_json(&options.output.join("summary.json"), &summary)?;
            Ok(())
        }
        Err(error) => {
            save_json(
                &options.output.join("FAILED.json"),
                &json!({"status":"INVALID_OR_INCOMPLETE","error":error}),
            )?;
            Err(error)
        }
    }
}
fn main() {
    if let Err(error) = Options::parse().and_then(run) {
        eprintln!("CENSUS FAILED: {error}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests;
