// SPDX-License-Identifier: AGPL-3.0-only

use super::scheduler_authority::parse_manifest;
use super::scheduler_authority_tests::{identities, manifest};
use super::scheduler_build::bind_build_receipt;

#[test]
fn reproducible_build_receipt_is_exact_and_strict() {
    let (binary, bundle, sources) = identities();
    let manifest_text = manifest(&binary, &bundle, &sources);
    let manifest_fields = parse_manifest(&manifest_text).unwrap();
    let mut receipt = format!(
        "schema=qwen38-direct-merged-silu-build-v1\n\
         binary_sha256={}\n\
         bundle_sha256={}\n\
         bundle_target={}\n\
         bundle_module_count={}\n\
         build_argv={}\n\
         build_environment={}\n",
        binary.sha256,
        bundle.sha256,
        bundle.target,
        bundle.module_count,
        manifest_fields["build_argv"],
        manifest_fields["build_environment"],
    );
    for (key, value) in &manifest_fields {
        if key.starts_with("source:")
            || key.starts_with("ptx:")
            || key.starts_with("gate_source:")
            || key.starts_with("tool:")
        {
            receipt.push_str(&format!("{key}={value}\n"));
        }
    }
    let fields = parse_manifest(&receipt).unwrap();
    assert!(bind_build_receipt(&fields, &manifest_fields, &binary, &bundle, &sources).is_ok());
    for mutant in [
        receipt.replace(&binary.sha256, &"ef".repeat(32)),
        receipt.replacen("gate_source:", "omitted_gate_source:", 1),
        receipt.replacen("tool:sha256sum:sha256", "tool:sha256sum:unbound", 1),
        format!("{receipt}unexpected=value\n"),
    ] {
        let fields = parse_manifest(&mutant).unwrap();
        assert!(bind_build_receipt(&fields, &manifest_fields, &binary, &bundle, &sources).is_err());
    }
}
