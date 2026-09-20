// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
pub(super) fn secret_name(name: &str) -> bool {
    let normalized = name.to_ascii_uppercase().replace('-', "_");
    [
        "AUTH",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "API_KEY",
        "ACCESS_TOKEN",
    ]
    .iter()
    .any(|s| normalized.contains(s))
        || normalized == "__TOKEN"
}
pub(super) fn tuning_name(name: &str) -> bool {
    ["ATLAS_", "SPARK_", "NCCL_"]
        .iter()
        .any(|p| name.starts_with(p))
        || [
            "CUDA_VISIBLE_DEVICES",
            "CUDARC_CUDA_VERSION",
            "CUDA_DEVICE_ORDER",
            "CUDA_MODULE_LOADING",
        ]
        .contains(&name)
}
fn cmdline(pid: u32) -> Result<Vec<String>> {
    read(Path::new(&format!("/proc/{pid}/cmdline")))?
        .split(|b| *b == 0)
        .filter(|v| !v.is_empty())
        .map(|v| String::from_utf8(v.to_vec()).map_err(|_| "non-UTF8 server argv".into()))
        .collect()
}
fn tuning_env(pid: u32) -> Result<Value> {
    let mut env = serde_json::Map::new();
    for item in read(Path::new(&format!("/proc/{pid}/environ")))?
        .split(|b| *b == 0)
        .filter(|v| !v.is_empty())
    {
        let text = std::str::from_utf8(item).map_err(|_| "non-UTF8 server environment")?;
        let (name, value) = text.split_once('=').ok_or("malformed server environment")?;
        if tuning_name(name) {
            check(
                !secret_name(name),
                "secret-like tuning environment key: admission denied; value not recorded",
            )?;
            env.insert(name.into(), json!(value));
        }
    }
    Ok(Value::Object(env))
}
fn start_ticks(pid: u32) -> Result<String> {
    let text = String::from_utf8(read(Path::new(&format!("/proc/{pid}/stat")))?)
        .map_err(|_| "invalid process stat")?;
    let rest = text.rsplit_once(") ").ok_or("malformed process stat")?.1;
    rest.split_whitespace()
        .nth(19)
        .map(str::to_string)
        .ok_or("missing PID start ticks".into())
}
fn owns_port(pid: u32, port: u16) -> Result<()> {
    let mut inodes = Vec::new();
    for family in ["tcp", "tcp6"] {
        let text = String::from_utf8(read(Path::new(&format!("/proc/net/{family}")))?)
            .map_err(|_| "invalid TCP table")?;
        for row in text.lines().skip(1) {
            let fields: Vec<_> = row.split_whitespace().collect();
            if fields.len() > 9
                && fields[3] == "0A"
                && fields[1]
                    .rsplit_once(':')
                    .is_some_and(|(_, p)| u16::from_str_radix(p, 16).ok() == Some(port))
            {
                inodes.push(format!("socket:[{}]", fields[9]));
            }
        }
    }
    check(!inodes.is_empty(), "no listener found for census port")?;
    let sockets = fs::read_dir(format!("/proc/{pid}/fd"))
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().and_then(|e| fs::read_link(e.path()).ok()))
        .collect::<Vec<_>>();
    check(
        inodes
            .iter()
            .all(|inode| sockets.iter().any(|path| path == Path::new(inode))),
        "census TCP port is not exclusively owned by supplied PID",
    )
}
pub(super) struct Binding {
    pid: u32,
    port: u16,
    binary: PathBuf,
    argv: Value,
    env: Value,
    pub(super) start: String,
}
impl Binding {
    pub(super) fn new(v: &Value, options: &Options) -> Result<Self> {
        check(
            str_field(v, "schema")? == "atlas-prefill-census-provenance-v1",
            "invalid provenance schema",
        )?;
        check(
            str_field(v, "model_name")? == options.model
                && number(v, "port")? == u64::from(options.port),
            "provenance model/port mismatch",
        )?;
        check(
            number(v, "context_limit")?
                >= options.bins.last().copied().ok_or("no selected bins")? as u64 + 32,
            "context_limit must support largest selected prompt bin + 32 output",
        )?;
        let pid = u32::try_from(number(v, "pid")?).map_err(|_| "invalid PID")?;
        check(pid > 1, "invalid server PID")?;
        let argv = v["argv"].as_array().ok_or("missing provenance argv")?;
        check(!argv.is_empty(), "empty provenance argv")?;
        for arg in argv {
            let argument = arg.as_str().ok_or("non-string argv entry")?;
            check(
                !secret_name(argument.split('=').next().unwrap_or("")),
                "secret-like argv entry: admission denied; value not recorded",
            )?;
        }
        for (name, value) in v["env"].as_object().ok_or("missing tuning env object")? {
            check(
                tuning_name(name) && !secret_name(name) && value.is_string(),
                "provenance env contains unsupported or secret-like key",
            )?;
        }
        let binary = fs::canonicalize(str_field(v, "binary_path")?).map_err(|e| e.to_string())?;
        check(
            sha256(&binary)? == str_field(v, "binary_sha256")?,
            "binary SHA256 mismatch",
        )?;
        check(
            sha256(Path::new(str_field(v, "config_path")?))? == str_field(v, "config_sha256")?,
            "model config SHA256 mismatch",
        )?;
        let binding = Self {
            pid,
            port: options.port,
            binary,
            argv: v["argv"].clone(),
            env: v["env"].clone(),
            start: start_ticks(pid)?,
        };
        binding.verify()?;
        Ok(binding)
    }
    pub(super) fn verify(&self) -> Result<()> {
        check(
            start_ticks(self.pid)? == self.start,
            "server PID exited/reused during census",
        )?;
        check(
            fs::read_link(format!("/proc/{}/exe", self.pid)).map_err(|e| e.to_string())?
                == self.binary,
            "server executable changed/mismatched",
        )?;
        check(
            json!(cmdline(self.pid)?) == self.argv,
            "effective server argv mismatches provenance",
        )?;
        check(
            tuning_env(self.pid)? == self.env,
            "effective tuning environment mismatches provenance",
        )?;
        owns_port(self.pid, self.port)
    }
}
