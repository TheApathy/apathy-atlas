// SPDX-License-Identifier: AGPL-3.0-only

//! EVERY KERNEL MUST HAVE A NON-TEST CALL SITE.
//!
//! Unit tests cannot catch this class, which is why it kept happening: the
//! kernels WORK. They compile, they are declared, they are exported, they pass
//! their own tests, and nothing ever calls them. Five of one night's mysteries
//! were the same defect wearing five names -- a fix written months earlier,
//! sitting in the tree, never wired.
//!
//! The check is a grep over source, needs no GPU, and runs in milliseconds.

/// Kernels deliberately kept dormant, each with the reason and what would
/// remove it from the list. Adding a name here is a DECISION; leaving one out
/// is a bug the test catches.
const DORMANT: &[(&str, &str)] = &[
    (
        "Glm53DsaCurrentVisibilityKernel",
        "the walk performs BOTH of its halves inline: the current latent row is \
         published directly in dsa_attention.rs, and the completing pool row is \
         published pre-score. Wiring it would be a walk-level restructure that \
         duplicates work already done. NOT a trough fix -- its writes_pool is \
         true only at q = KPOOL-1 (mod KPOOL), which is the CLEAN band (3.06%); \
         every trough position has writes_pool false.",
    ),
    (
        "Glm53DsaIndexCommitKernel",
        "the receipted all-11-layer DSA commit. dsa_attention.rs bypasses it with \
         a per-layer immediate publish -- the third documented shortcut past the \
         receipts -- which is effect-equivalent only because that path never \
         rejects. Wired when speculation makes rejection possible.",
    ),
    ("Glm53Dflash2CaptureKernel", "DFlash2 drafter is not wired"),
];

fn sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for directory in [
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/layers"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/glm53"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/weight_loader"),
    ] {
        let entries = std::fs::read_dir(directory).expect("readable source directory");
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            out.push((
                name,
                std::fs::read_to_string(&path).expect("readable source"),
            ));
        }
    }
    let ops = concat!(env!("CARGO_MANIFEST_DIR"), "/src/layers/ops");
    for entry in std::fs::read_dir(ops)
        .expect("readable ops directory")
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        out.push((
            name,
            std::fs::read_to_string(&path).expect("readable source"),
        ));
    }
    out
}

/// A kernel with no non-test caller is dead code that reads as a feature.
///
/// `executor.rs` naming a seam in a STRING is not a call site, and that is
/// exactly how `Glm53DsaCurrentVisibilityKernel` looked wired while it was not:
/// declared, exported, its CUDA in KERNEL.toml, advertised in the seam
/// description, and referenced nowhere outside its own tests.
#[test]
fn every_kernel_has_a_non_test_call_site_or_a_declared_reason() {
    let files = sources();

    // Kernel type names, harvested from the source rather than listed here, so
    // a new kernel is covered the moment it is written.
    let mut kernels: Vec<String> = Vec::new();
    for (name, source) in &files {
        if !name.starts_with("glm53") || name.contains("_tests") {
            continue;
        }
        for line in source.lines() {
            let line = line.trim_start();
            if let Some(rest) = line.strip_prefix("pub struct Glm53") {
                let ident: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if ident.ends_with("Kernel") || ident.ends_with("Kernels") {
                    kernels.push(format!("Glm53{ident}"));
                }
            }
        }
    }
    kernels.sort();
    kernels.dedup();
    assert!(
        kernels.len() >= 15,
        "harvested only {} kernel types; the scan is not finding them",
        kernels.len()
    );

    let mut unwired = Vec::new();
    for kernel in &kernels {
        let declaring = format!("{}", kernel);
        let has_caller = files.iter().any(|(name, source)| {
            if name.contains("_tests") {
                return false;
            }
            // Its own module declares it; that is not a call site.
            if source.contains(&format!("pub struct {declaring}")) {
                return false;
            }
            source.contains(&format!("{kernel}::")) || source.contains(&format!("{kernel} "))
        });
        if !has_caller {
            unwired.push(kernel.clone());
        }
    }

    let dormant: Vec<&str> = DORMANT.iter().map(|(name, _)| *name).collect();
    let undeclared: Vec<&String> = unwired
        .iter()
        .filter(|k| !dormant.contains(&k.as_str()))
        .collect();
    assert!(
        undeclared.is_empty(),
        "these kernels have NO non-test call site and are not on the dormant \
         list: {undeclared:?}. Either wire them or add them to DORMANT with the \
         reason and what would remove them."
    );

    // And the list must not rot: a name that IS wired must come off it, or the
    // list stops meaning anything.
    let stale: Vec<&str> = dormant
        .iter()
        .filter(|name| {
            kernels.iter().any(|k| k == *name) && !unwired.contains(&(*name).to_string())
        })
        .copied()
        .collect();
    assert!(
        stale.is_empty(),
        "these are on the DORMANT list but now HAVE call sites: {stale:?}. Remove \
         them, so the list keeps naming only what is really dormant."
    );
}
