// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
fn stats(samples: &[Value], key: &str) -> Value {
    let mut values = samples
        .iter()
        .map(|v| v[key].as_f64().expect("validated metric"))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    json!({"median":values[values.len()/2],"min":values[0],"max":values[values.len()-1]})
}
fn request_sample(
    options: &Options,
    binding: &Binding,
    label: &str,
    request: &Value,
    expected: Option<usize>,
    chat: bool,
) -> Result<Value> {
    let endpoint = if chat {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    };
    let (response, wall) = post(options, binding, label, endpoint, request)?;
    let mut record = validate_response(&response, expected, chat, wall)?;
    let output_path = options.output.join(format!("{label}.output.txt"));
    save(
        &output_path,
        record["output"]
            .as_str()
            .expect("validated output")
            .as_bytes(),
    )?;
    record["output_sha256"] = json!(sha256(&output_path)?);
    record["request_sha256"] = json!(sha256(
        &options.output.join(format!("{label}.request.json"))
    )?);
    record["response_sha256"] = json!(sha256(
        &options.output.join(format!("{label}.response.json"))
    )?);
    record["label"] = json!(label);
    save_json(
        &options.output.join(format!("{label}.record.json")),
        &record,
    )?;
    Ok(record)
}
pub(super) fn workload(options: &Options, binding: &Binding) -> Result<Value> {
    let canary_request = canary_request(&options.model);
    let canary = request_sample(options, binding, "canary", &canary_request, None, true)?;
    println!(
        "Canary content (verbatim; semantic review REQUIRED):\n{}\n--- end canary ---",
        canary["output"].as_str().unwrap()
    );
    let (prefix, suffix) = prompt_parts(&options.prompt_format)?;
    let source = corpus(prefix);
    save_json(
        &options.output.join("prompt-format.json"),
        &json!({"format":options.prompt_format,"prefix":prefix,"suffix":suffix,
        "token_accounting":"Exact bins include every template marker ID; corpus prefix is truncated only before separately tokenized complete suffix."}),
    )?;
    save(&options.output.join("corpus.txt"), source.as_bytes())?;
    save(&options.output.join("suffix.txt"), suffix.as_bytes())?;
    let (encoded, _) = post(
        options,
        binding,
        "tokenize-corpus",
        "/tokenize",
        &json!({"model":options.model,"prompt":source}),
    )?;
    let (tail, _) = post(
        options,
        binding,
        "tokenize-suffix",
        "/tokenize",
        &json!({"model":options.model,"prompt":suffix}),
    )?;
    let corpus_ids = parse_tokens(&encoded)?;
    let suffix_ids = parse_tokens(&tail)?;
    let mut bins = Vec::new();
    for &length in &options.bins {
        let ids = exact_prompt(&corpus_ids, &suffix_ids, length)?;
        save_json(
            &options.output.join(format!("bin-{length}.tokens.json")),
            &json!(ids),
        )?;
        let mut request =
            json!({"model":options.model,"max_tokens":32,"temperature":0.0,"stream":false});
        if options.mode == "prompt-array" {
            request["prompt"] = json!(ids);
        } else {
            request["prompt"] = json!("");
            request["prompt_token_ids"] = json!(ids);
        }
        let warm = request_sample(
            options,
            binding,
            &format!("bin-{length}-warm"),
            &request,
            Some(length),
            false,
        )?;
        let mut samples = Vec::new();
        for trial in 1..=5 {
            samples.push(request_sample(
                options,
                binding,
                &format!("bin-{length}-trial-{trial}"),
                &request,
                Some(length),
                false,
            )?);
        }
        let summary = json!({"prompt_tokens":length,"requested_completion_tokens":32,"warm_runs":1,"measured_runs":5,
            "prompt_format":options.prompt_format,
            "effective_prefill_tokens_per_second":stats(&samples,"effective_prefill_tokens_per_second"),
            "server_ttft_ms":stats(&samples,"server_ttft_ms"),"client_wall_ms":stats(&samples,"client_wall_ms"),
            "actual_completion_tokens":samples.iter().map(|v| v["completion_tokens"].clone()).collect::<Vec<_>>(),
            "identical_measured_output_hashes":samples.iter().all(|v| v["output_sha256"] == samples[0]["output_sha256"]),
            "warm":warm,"samples":samples});
        save_json(
            &options.output.join(format!("bin-{length}.summary.json")),
            &summary,
        )?;
        println!(
            "{length} tokens: effective prefill {:.2} tok/s (server TTFT median {:.2} ms); not isolated GPU prefill",
            summary["effective_prefill_tokens_per_second"]["median"]
                .as_f64()
                .unwrap(),
            summary["server_ttft_ms"]["median"].as_f64().unwrap()
        );
        bins.push(summary);
    }
    let not_measured_bins = [256, 2048, 8192]
        .into_iter()
        .filter(|n| !options.bins.contains(n))
        .collect::<Vec<_>>();
    Ok(
        json!({"schema":"atlas-prefill-census-v4","status":"MEASURED_SELECTED_BINS_SEMANTICS_UNREVIEWED","model":options.model,"port":options.port,
        "prompt_format":options.prompt_format,
        "selected_bins":options.bins,"not_measured_bins":not_measured_bins,
        "ids_mode":options.mode,"metric_definition":"prompt_tokens / server time_to_first_token_ms * 1000; not isolated on-GPU prefill",
        "provenance_sha256":sha256(&options.output.join("provenance.json"))?,"canary":canary,"bins":bins}),
    )
}
