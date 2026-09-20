// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
pub(super) fn post(
    options: &Options,
    binding: &Binding,
    label: &str,
    endpoint: &str,
    request: &Value,
) -> Result<(Value, f64)> {
    binding.verify()?;
    let req = options.output.join(format!("{label}.request.json"));
    let resp = options.output.join(format!("{label}.response.json"));
    save_json(&req, request)?;
    check(!resp.exists(), "response path already exists")?;
    let start = Instant::now();
    let result = Command::new("curl")
        .args([
            "-q",
            "--silent",
            "--show-error",
            "--noproxy",
            "*",
            "--proto",
            "=http",
            "--connect-timeout",
            "5",
            "--max-filesize",
            "16777216",
            "--max-time",
            &options.timeout.to_string(),
            "--request",
            "POST",
            "--header",
            "Content-Type: application/json",
            "--data-binary",
        ])
        .arg(format!("@{}", req.display()))
        .arg("--output")
        .arg(&resp)
        .args(["--write-out", "%{http_code}"])
        .arg(format!("http://127.0.0.1:{}{endpoint}", options.port))
        .output()
        .map_err(|e| e.to_string())?;
    let wall = start.elapsed().as_secs_f64() * 1000.0;
    save(
        &options.output.join(format!("{label}.curl.stderr.txt")),
        &result.stderr,
    )?;
    let status = String::from_utf8(result.stdout)
        .map_err(|_| "invalid curl status")?
        .trim()
        .parse::<u16>()
        .map_err(|_| "invalid HTTP status")?;
    save_json(
        &options.output.join(format!("{label}.transport.json")),
        &json!({"http_status":status,"curl_exit":result.status.code(),"client_wall_ms":wall}),
    )?;
    check(
        result.status.success(),
        "curl transport failure/timeout; see saved transport and stderr",
    )?;
    binding.verify()?;
    let bytes = read(&resp)?;
    check(bytes.len() <= 16 * 1024 * 1024, "oversized response")?;
    Ok((parse_json_response(&bytes, status)?, wall))
}
