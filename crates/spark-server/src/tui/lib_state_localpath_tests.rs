// SPDX-License-Identifier: AGPL-3.0-only

use super::local_model_path;
use crate::tui::data::library::LibraryEntry;

fn entry(id: &str, dir: &std::path::Path) -> LibraryEntry {
    LibraryEntry {
        id: id.into(),
        snapshot_dir: dir.to_path_buf(),
        size_bytes: 0,
        has_weights: true,
        model_type: "deepseek_v41".into(),
        quant: "-".into(),
        layers: 0,
        hidden: 0,
        heads: 0,
        experts: 0,
        context: 0,
        optimized: false,
    }
}

/// A plain-root checkpoint is served by its PATH — its bare directory name only resolves when
/// the process happens to run inside that root.
#[test]
fn a_plain_root_model_is_served_by_path() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("DeepSeek-V4.1-Flash-Next-DGX-Spark-512K");
    std::fs::create_dir(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let e = entry("DeepSeek-V4.1-Flash-Next-DGX-Spark-512K", &model);
    assert_eq!(
        local_model_path("DeepSeek-V4.1-Flash-Next-DGX-Spark-512K", Some(&e)).as_deref(),
        Some(model.as_path())
    );
}

/// Controls: an HF id keeps going through the hub cache, a different model is not rewritten,
/// and a directory without config.json is not a servable path.
#[test]
fn hf_ids_mismatches_and_non_checkpoints_are_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let e = entry("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", dir.path());
    assert!(local_model_path("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", Some(&e)).is_none());

    let e = entry("SomeOtherModel", dir.path());
    assert!(local_model_path("DeepSeek-V4.1-Flash-Next-DGX-Spark-512K", Some(&e)).is_none());

    let e = entry("NoConfig", dir.path());
    assert!(local_model_path("NoConfig", Some(&e)).is_none(), "no config.json -> not a checkpoint");
    assert!(local_model_path("NoConfig", None).is_none());
}
