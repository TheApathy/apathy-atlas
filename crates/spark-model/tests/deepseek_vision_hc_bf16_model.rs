// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../examples/vision_hc_bf16_probe/inputs.rs"]
mod probe_inputs;

#[path = "vision_hc_bf16/capture.rs"]
mod capture;

fn source(path: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|error| panic!("missing production source {path}: {error}"))
}

#[test]
fn candidate_changes_only_hc_post_final_store() {
    let baseline = source("../../kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu");
    let candidate =
        source("../../kernels/gb10/deepseek-v4-flash/nvfp4/deepseek_vision_hc_post_bf16.cu");
    let body = |text: &str, name: &str| {
        let after = text.split(&format!("void {name}(")).nth(1).unwrap();
        after.split("\n}\n").next().unwrap().to_owned()
    };
    assert_eq!(
        body(&baseline, "hc_post"),
        body(&candidate, "deepseek_vision_hc_post_bf16")
            .replace("__bfloat162float(__float2bfloat16_rn(acc))", "acc")
    );
    assert!(baseline.contains("o[j * H + d] = acc;"));
}

#[test]
fn configuration_is_checked_before_weights_and_both_passes_share_the_selected_handle() {
    let serve = source("../spark-server/src/main_modules/serve.rs");
    assert!(
        serve.find("VisionHcBf16::from_env(").unwrap()
            < serve.find("serve_phases::init_gpu_backend(").unwrap()
    );
    let factory = source("src/factory/build.rs");
    assert!(
        factory.find("VisionHcBf16::from_env(").unwrap()
            < factory.find("loader_for_config(").unwrap()
    );
    let assembly = source("src/weight_loader/deepseek_v4/assemble.rs");
    assert!(
        assembly.find("layer.set_hc_weights(").unwrap()
            < assembly.find("layer.set_vision_hc_bf16(").unwrap()
    );
    for name in ["prefill_inner.rs", "decode_inner.rs"] {
        let pass = source(&format!("src/layers/qwen3_attention/trait_impl/{name}"));
        assert_eq!(pass.matches("self.hc_post_k,").count(), 3);
    }
    let decode = source("src/model/trait_impl/decode_a.rs");
    assert!(decode.contains("deepseek_vision.is_none()"));
}

#[test]
fn independent_rne_oracle_covers_sign_ties_tiny_and_overflow() {
    for (input, expected) in [
        (0x00000000, 0x00000000),
        (0x80000000, 0x80000000),
        (0x3f800000, 0x3f800000),
        (0x3f807fff, 0x3f800000),
        (0x3f808000, 0x3f800000),
        (0x3f808001, 0x3f810000),
        (0x3f817fff, 0x3f810000),
        (0x3f818000, 0x3f820000),
        (0xbf808000, 0xbf800000),
        (0xbf818000, 0xbf820000),
        (0x00000001, 0x00000000),
        (0x00007fff, 0x00000000),
        (0x00008000, 0x00000000),
        (0x00008001, 0x00010000),
        (0x00018000, 0x00020000),
        (0x80008000, 0x80000000),
        (0x007fffff, 0x00800000),
        (0x00800000, 0x00800000),
        (0x7f7f0000, 0x7f7f0000),
        (0x7f7f7fff, 0x7f7f0000),
        (0x7f7f8000, 0x7f800000),
        (0xff7f8000, 0xff800000),
    ] {
        assert_eq!(probe_inputs::rounded_bits(input), expected, "{input:08x}");
        assert_eq!(
            half::bf16::from_f32(f32::from_bits(input))
                .to_f32()
                .to_bits(),
            expected
        );
    }
}

#[test]
fn probe_inputs_are_bounded_and_native_probe_covers_alias_and_shards() {
    for rows in [1, 12] {
        for boundary in [true, false] {
            let case = probe_inputs::synthetic(rows, boundary);
            assert_eq!(case.rows, rows);
            assert!(case.label.starts_with("synthetic-"));
            assert_eq!(case.block.len(), rows * 4096 * 2);
            assert_eq!(case.residual.len(), rows * 4 * 4096 * 4);
            assert_eq!(case.post.len(), rows * 4 * 4);
            assert_eq!(case.comb.len(), rows * 16 * 4);
            assert!(case.captured_output.is_none());
        }
    }
    assert!(probe_inputs::captured(std::path::Path::new("relative")).is_err());
    let probe = source("examples/vision_hc_bf16_probe.rs");
    assert!(probe.contains("for shards in [1, 16]"));
    assert!(probe.contains("for inplace in [false, true]"));
    assert!(probe.contains("actual == want"));
    let cargo = source("Cargo.toml");
    assert!(cargo.contains(
        "name = \"vision_hc_bf16_probe\"\nrequired-features = [\"cuda\", \"gpu-examples\"]"
    ));
}
