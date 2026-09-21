// SPDX-License-Identifier: AGPL-3.0-only

//! Isolated real-weight DeepSeek vision probe; never loads decoder experts.
//! CPU: cargo run -p spark-model --no-default-features --example deepseek_vision_probe -- inspect MODEL
//! GPU (after reservation/build): ... --features cuda,gpu-examples -- run MODEL NEW_OUTPUT_DIR 3x3,4x5,54x54
//! Selected block detail: ... -- stages-block MODEL NEW_OUTPUT_DIR 4x5 8
//! Output timings include pixel upload and encoder execution, excluding model
//! loading/download/output hashing. They are not LLM prompt-prefill throughput.

#[cfg_attr(not(all(feature = "cuda", feature = "gpu-examples")), allow(dead_code))]
#[path = "deepseek_vision_probe/contract.rs"]
mod contract;
#[cfg(all(feature = "cuda", feature = "gpu-examples"))]
#[path = "deepseek_vision_probe/gpu.rs"]
mod gpu;
#[cfg(all(feature = "cuda", feature = "gpu-examples"))]
#[path = "deepseek_vision_probe/stages.rs"]
mod stages;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "inspect" {
        let manifest = contract::inspect(std::path::Path::new(&args[1]))?;
        println!("{}", serde_json::to_string_pretty(&manifest.report)?);
        return Ok(());
    }
    #[cfg(all(feature = "cuda", feature = "gpu-examples"))]
    if args.len() == 4 && matches!(args[0].as_str(), "run" | "stages") {
        return gpu::run(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
            &args[3],
            (args[0] == "stages").then_some(0),
        );
    }
    #[cfg(all(feature = "cuda", feature = "gpu-examples"))]
    if args.len() == 5 && args[0] == "stages-block" {
        return gpu::run(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
            &args[3],
            Some(args[4].parse::<usize>()?),
        );
    }
    anyhow::bail!(
        "usage: deepseek_vision_probe inspect MODEL | run|stages MODEL NEW_OUTPUT_DIR GRID_LIST | stages-block MODEL NEW_OUTPUT_DIR GRID_LIST BLOCK (GPU mode requires cuda,gpu-examples)"
    )
}

#[cfg(test)]
mod tests {
    use super::contract::*;

    #[test]
    fn probe_selection_excludes_language_and_draft_weights() {
        assert!(visual_name("vision.blocks.0.attn.wqkv.weight"));
        assert!(visual_name("aligner.w1.weight"));
        assert!(visual_name("image_start"));
        assert!(!visual_name(
            "model.layers.0.mlp.experts.0.gate_proj.trellis"
        ));
        assert!(!visual_name("mtp.0.ffn.gate.bias_vl"));
        assert!(!visual_name("image_embedding.weight"));
    }

    #[test]
    fn probe_cases_are_bounded_unique_and_cover_edge_padding() {
        assert_eq!(
            parse_grids("3x3,4x5,54x54").unwrap(),
            vec![(3, 3), (4, 5), (54, 54)]
        );
        for bad in [
            "",
            "0x3",
            "3x3,3x3",
            "100x100",
            "99999999999999999999x3",
            "3x3x3",
            "3x3,",
        ] {
            assert!(parse_grids(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn probe_hashes_and_finite_stats_cover_complete_buffers() {
        assert_eq!(
            sha256_bytes(b"abc").unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let stats = bf16_stats(&[0x80, 0x3f, 0x00, 0xc0]).unwrap();
        assert_eq!(stats["count"], 2);
        assert_eq!(stats["max_abs"], 2.0);
        assert!(bf16_stats(&[0xc0, 0x7f]).is_err());
        assert!(bf16_stats(&[0]).is_err());
    }

    #[test]
    fn probe_pixels_are_exact_dyadic_and_image_dependent() {
        let p = make_pixels(3, 3);
        assert_eq!(p.len(), 9 * 588);
        assert!(p.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
        assert_ne!(&p[..588], &p[588..1176]);
        assert_eq!(p, make_pixels(3, 3));
    }
}
