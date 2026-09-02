// SPDX-License-Identifier: AGPL-3.0-only

//! Validate a real on-disk GLM-5.3 GGUF checkpoint against its pinned profile.
//!
//! Runs the production loader path — shard identity (size + SHA-256), split
//! metadata, architecture, and the full 1,412-tensor schema including every
//! per-tensor ggml type — against actual files. A green schema unit test only
//! proves the builder is self-consistent; this proves it matches the bytes.
//!
//! Usage: cargo run --example glm53_validate_checkpoint -- <profile-dir>

use spark_runtime::weights::gguf::{Glm53QuantProfile, validate_glm53_files};

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: glm53_validate_checkpoint <profile-dir>");
    let profile = match std::path::Path::new(&dir)
        .file_name()
        .and_then(|n| n.to_str())
    {
        Some("UD-IQ2_XXS") => Glm53QuantProfile::UdIq2Xxs,
        Some("UD-Q2_K_XL") => Glm53QuantProfile::UdQ2KXl,
        Some("UD-IQ3_XXS") => Glm53QuantProfile::UdIq3Xxs,
        other => anyhow::bail!("unrecognized profile directory {other:?}"),
    };
    let paths: Vec<_> = profile
        .canonical_file_names()
        .iter()
        .map(|name| std::path::Path::new(&dir).join(name))
        .collect();

    println!("profile      {:?} ({})", profile, profile.directory_name());
    let summary = validate_glm53_files(profile, &paths)?;
    println!("shards       {}", summary.shards);
    println!("tensors      {}", summary.tensors);
    println!("tensor bytes {}", summary.tensor_bytes);
    assert_eq!(summary.tensor_bytes, profile.tensor_bytes());
    println!("\nVALIDATED: identity, metadata and full tensor schema match the pinned profile.");
    Ok(())
}
