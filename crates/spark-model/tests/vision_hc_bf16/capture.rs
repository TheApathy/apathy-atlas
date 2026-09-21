// SPDX-License-Identifier: AGPL-3.0-only

use super::probe_inputs;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        // Exact fresh directory owned by this test; never a caller-supplied path.
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn fixture() -> (Fixture, Value) {
    let path = std::env::temp_dir().join(format!(
        "atlas-hc-bf16-fixture-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let ids = [
        0u32, 128803, 19905, 418, 9045, 20370, 305, 5760, 3006, 16, 128804, 128822,
    ];
    std::fs::write(
        path.join("token_ids.bin"),
        ids.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    let mut tensors = serde_json::Map::new();
    for (name, dtype, shape) in [
        ("embed", "BF16", vec![12, 4096]),
        ("hc_expanded", "F32", vec![12, 4, 4096]),
        ("hc_pre_attn", "BF16", vec![12, 4096]),
        ("post_attn", "F32", vec![12, 4]),
        ("comb_attn", "F32", vec![12, 4, 4]),
        ("norm_attn", "BF16", vec![12, 4096]),
        ("attention_out", "BF16", vec![12, 4096]),
        ("hc_post_attn", "F32", vec![12, 4, 4096]),
        ("hc_pre_ffn", "BF16", vec![12, 4096]),
        ("post_ffn", "F32", vec![12, 4]),
        ("comb_ffn", "F32", vec![12, 4, 4]),
        ("norm_ffn", "BF16", vec![12, 4096]),
        ("moe_out", "BF16", vec![12, 4096]),
        ("hc_post_ffn", "F32", vec![12, 4, 4096]),
    ] {
        let bytes = shape.iter().product::<usize>() * if dtype == "BF16" { 2 } else { 4 };
        let file = format!("{name}.bin");
        std::fs::write(path.join(&file), vec![0; bytes]).unwrap();
        tensors.insert(
            name.into(),
            json!({"file":file,"dtype":dtype,"shape":shape,"bytes":bytes}),
        );
    }
    let manifest = json!({"schema":"atlas-vision-l0-dump-v1","status":"COMPLETE",
        "byte_order":"little","layer_index":0,"token_count":12,"hidden_size":4096,
        "hc_mult":4,"vocab_size":129280,"payload_bytes":3049392,"token_ids":ids,
        "token_ids_file":"token_ids.bin","token_ids_bytes":48,"tensors":tensors});
    (Fixture(path), manifest)
}

fn manifest(dir: &Path, value: &Value) {
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(value).unwrap(),
    )
    .unwrap();
}

#[test]
fn capture_probe_reader_accepts_only_complete_typed_bounded_inputs() {
    let (dir, good) = fixture();
    assert!(probe_inputs::captured(&dir.0).is_err()); // no COMPLETE commit marker
    manifest(&dir.0, &good);
    let cases = probe_inputs::captured(&dir.0).unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        assert_eq!(
            case.captured_output.unwrap().len(),
            case.rows * 4 * 4096 * 4
        );
    }
    for field in 0..9 {
        let mut bad = good.clone();
        match field {
            0 => bad["status"] = json!("PARTIAL"),
            1 => bad["token_ids"][1] = json!(129280),
            2 => bad["payload_bytes"] = json!(u64::MAX),
            3 => bad["tensors"]["attention_out"]["dtype"] = json!("F32"),
            4 => bad["tensors"]["attention_out"]["shape"] = json!([1, 4096]),
            5 => bad["tensors"]["attention_out"]["file"] = json!("../other"),
            6 => bad["tensors"]["extra"] = json!({}),
            7 => {
                bad["tensors"]
                    .as_object_mut()
                    .unwrap()
                    .remove("attention_out");
            }
            _ => bad["tensors"]["attention_out"]["bytes"] = json!(u64::MAX),
        }
        manifest(&dir.0, &bad);
        assert!(probe_inputs::captured(&dir.0).is_err(), "field {field}");
    }
    manifest(&dir.0, &good);
    std::fs::write(dir.0.join("attention_out.bin"), [0u8]).unwrap();
    assert!(probe_inputs::captured(&dir.0).is_err());
    #[cfg(unix)]
    {
        let payload = dir.0.join("attention_out.bin");
        std::fs::remove_file(&payload).unwrap();
        std::os::unix::fs::symlink(dir.0.join("embed.bin"), &payload).unwrap();
        assert!(probe_inputs::captured(&dir.0).is_err());
    }
}
