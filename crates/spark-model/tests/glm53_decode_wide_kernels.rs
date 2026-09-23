// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the two decode-width kernels.
//!
//! Both replace a kernel that ran one block per token (so exactly one block,
//! on one SM, during decode) with a grid that spreads the same reads over many
//! SMs. Neither may change a single float operation or its order: these tests
//! pin the accumulation chains character for character against the fused
//! kernels they are derived from. The end-to-end oracle is output identity
//! against the flags-off control; this is the cheap part of that argument.

use std::path::PathBuf;

fn kernel_source(name: &str) -> String {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "..",
        "kernels",
        "gb10",
        "glm5.3-flash",
        "iq3",
        name,
    ]
    .iter()
    .collect();
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

fn section<'a>(source: &'a str, start: &str, end: Option<&str>) -> &'a str {
    let from = source
        .find(start)
        .unwrap_or_else(|| panic!("missing kernel {start}"));
    match end {
        Some(end) => {
            let to = source[from..]
                .find(end)
                .unwrap_or_else(|| panic!("missing terminator {end}"));
            &source[from..from + to]
        }
        None => &source[from..],
    }
}

#[test]
fn hc_pre_mix_keeps_the_fused_projection_chain() {
    let source = kernel_source("glm53_hyper.cu");
    let fused = compact(section(
        &source,
        "atlas_glm53_hc_pre(",
        Some("atlas_glm53_hc_post("),
    ));
    let mix = compact(section(
        &source,
        "atlas_glm53_hc_pre_mix(",
        Some("atlas_glm53_hc_pre_fold("),
    ));

    // The multiply-accumulate chain and the shuffle tree are the arithmetic.
    // Both must appear verbatim in the split kernel.
    let chain = "sum=fmaf(function[function_base+index],streams[stream_base+index],sum);";
    let tree = "for(unsignedintoffset=16U;offset>0U;offset>>=1U)sum+=__shfl_down_sync(0xffffffffU,sum,offset);";
    assert!(fused.contains(chain), "fused kernel lost its fmaf chain");
    assert!(mix.contains(chain), "wide mix changed the fmaf chain");
    assert!(fused.contains(tree), "fused kernel lost its shuffle tree");
    assert!(mix.contains(tree), "wide mix changed the shuffle tree");

    // The lane stride is what makes the two chains the same sequence.
    assert!(mix.contains("for(unsignedintindex=lane;index<flat_size;index+=32U)"));
    assert!(fused.contains("for(unsignedintindex=lane;index<flat_size;index+=32U)"));

    // One block per projection per token, one warp each.
    assert!(mix.contains("constunsignedintoutput=blockIdx.x;"));
    assert!(mix.contains("constunsignedinttoken=blockIdx.y;"));
    assert!(mix.contains("gridDim.x!=GLM53_MIX"));
    assert!(mix.contains("blockDim.x!=32U"));

    // Phase 1 must not scale: the inverse RMS is not known there.
    assert!(
        !mix.contains("inverse_rms"),
        "wide mix must leave the RMS scaling to the fold pass"
    );
}

#[test]
fn hc_pre_fold_keeps_the_fused_rms_sinkhorn_and_collapse() {
    let source = kernel_source("glm53_hyper.cu");
    let fused = compact(section(
        &source,
        "atlas_glm53_hc_pre(",
        Some("atlas_glm53_hc_post("),
    ));
    let fold = compact(section(&source, "atlas_glm53_hc_pre_fold(", None));

    // Same reduction, same thread count, same tree: glm53_block_sum is order
    // sensitive and is shared by both kernels.
    let rms = "for(unsignedintindex=tid;index<flat_size;index+=GLM53_THREADS){constfloatvalue=streams[stream_base+index];square_sum+=value*value;}";
    assert!(fused.contains(rms));
    assert!(fold.contains(rms));
    assert!(fold.contains("constfloattotal=glm53_block_sum(reduction,tid);"));
    assert!(fold.contains("inverse_rms=rsqrtf(total/(float)flat_size+norm_eps);"));
    // The fold keeps the fused kernel's block shape; the attribute precedes
    // the name, so it is checked against the whole file.
    assert!(compact(&source)
        .contains("__launch_bounds__(GLM53_THREADS,1)atlas_glm53_hc_pre_fold("));

    // The scaling moved, but it is the same product of the same two operands.
    assert!(fused.contains("mixed[output]=sum*inverse_rms;"));
    assert!(fold.contains("mixed[tid]=mixed_raw[(unsignedlonglong)token*GLM53_MIX+tid]*inverse_rms;"));

    // Sinkhorn and the collapse are copied, not reimplemented.
    let collapse = "sum=fmaf(pre[stream],streams[stream_base+(unsignedlonglong)stream*hidden_size+column],sum);";
    assert!(fused.contains(collapse));
    assert!(fold.contains(collapse));
    for fragment in [
        "pre[index]=1.0f/(1.0f+expf(-pre_logit))+hc_eps;",
        "sinkhorn[index]=expf(sinkhorn[index]-maximum);",
        "sinkhorn[index]=sinkhorn[index]/sum+hc_eps;",
    ] {
        assert!(fused.contains(fragment), "fused lost {fragment}");
        assert!(fold.contains(fragment), "fold changed {fragment}");
    }
}

#[test]
fn index_projection_wide_keeps_the_serial_ascending_chain() {
    let source = kernel_source("glm53_dsa_norm_projection.cu");
    let wide = compact(section(
        &source,
        "atlas_glm53_dsa_index_projection_f32_bf16_wide(",
        None,
    ));

    // One block per (row, head) instead of one thread per head.
    assert!(wide.contains("constunsignedinthead=blockIdx.x;"));
    assert!(wide.contains("constunsignedintrow=blockIdx.y;"));
    assert!(wide.contains("gridDim.x!=heads||gridDim.y!=rows"));
    assert!(wide.contains("blockDim.x!=32U"));

    // Exactly one lane accumulates, over ascending columns, with the same
    // fmaf(weight, input, sum) operand order the fused kernel uses.
    assert!(wide.contains("if(lane==0U){for(unsignedintslot=0U;slot<GLM53_INDEX_PROJ_TILE;++slot){sum=fmaf(tile_weight[slot],tile_input[slot],sum);}}"));
    assert!(
        wide.contains("tile_input[slot]=__bfloat162float(input[input_base+tile+slot]);"),
        "the wide kernel must convert the input exactly as the fused one does"
    );
    assert!(wide.contains("__float2bfloat16_rn(sum)"));

    // Tiles must divide the contraction exactly, or the chain would be cut.
    assert!(compact(&source).contains("#defineGLM53_INDEX_PROJ_TILE512U"));
    assert_eq!(4096 % 512, 0);

    // No cross-lane reduction anywhere: partial sums are never combined.
    assert!(
        !wide.contains("__shfl"),
        "the wide index projection must not reduce across lanes"
    );
}

#[test]
fn wide_flags_are_strict_and_default_off() {
    // Mirrors the tree's convention: absent or "0" is off, "1" is on, and any
    // other spelling is a hard error rather than a silent default.
    for (name, source) in [
        ("ATLAS_GLM53_HC_PRE_WIDE", "glm53_hyper.rs"),
        (
            "ATLAS_GLM53_DSA_INDEX_PROJ_WIDE",
            "glm53_dsa_norm_projection.rs",
        ),
    ] {
        let path: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "src",
            "layers",
            "ops",
            source,
        ]
        .iter()
        .collect();
        let text = std::fs::read_to_string(&path).unwrap();
        let compacted = compact(&text);
        assert!(
            compacted.contains(&compact("None | Some(\"0\") => Ok(false),")),
            "{name} must treat absent and 0 as off"
        );
        assert!(
            compacted.contains(&compact("Some(\"1\") => Ok(true),")),
            "{name} must treat 1 as on"
        );
        assert!(
            text.contains(&format!("{name} must be absent, 0, or 1")),
            "{name} must reject other spellings"
        );
    }
}

// ── EXL3 GEMV, m == 1 ───────────────────────────────────────────────

#[test]
fn gemv_m1_instantiations_exist_and_are_mmode_zero() {
    // The gate `(2..=8).contains(&rows)` was not a correctness guard; it
    // described the kernels that had been compiled. Upstream's header states
    // "MMODE 0 is the m == 1 fast path, MMODE 1 covers 2 <= m <= 8 with
    // row-guarded fragment loads", and the six original instantiations are all
    // MMODE 1 — so ordinary decode, which is m == 1 by definition, had no GEMV
    // to fall into and took the 90 KiB-shared, 48-CTA tile GEMM instead.
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "..",
        "kernels",
        "gb10",
        "glm5.3-flash",
        "exl3",
        "glm53_exl3_gemv_cb2.cu",
    ]
    .iter()
    .collect();
    let source = std::fs::read_to_string(&path).unwrap();
    let compacted = compact(&source);

    // Template arguments are <bits, c_fp32, cb, MMODE, CFG, SMEM_STAGE>.
    for bits in [2, 3, 4] {
        for cfg in [0, 1] {
            let mmode1 = format!("exl3_gemv_kernel<{bits},false,2,1,{cfg},false>");
            let mmode0 = format!("exl3_gemv_kernel<{bits},false,2,0,{cfg},false>");
            assert!(
                compacted.contains(&mmode1),
                "the verifier instantiation {mmode1} must be kept"
            );
            assert!(
                compacted.contains(&mmode0),
                "the m == 1 instantiation {mmode0} is missing"
            );
        }
    }
    assert_eq!(
        source.matches("template __global__ void exl3_gemv_kernel").count(),
        12,
        "six MMODE 1 verifier kernels plus six MMODE 0 decode kernels",
    );
}

#[test]
fn gemv_m1_is_flagged_off_and_selects_mmode_zero() {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "src",
        "layers",
        "ops",
        "glm53_exl3.rs",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap();
    let compacted = compact(&text);

    // Strict, default-off, like every other lever in this tree.
    assert!(compacted.contains(&compact("None | Some(\"0\") => Ok(false),")));
    assert!(compacted.contains(&compact("Some(\"1\") => Ok(true),")));
    assert!(text.contains("ATLAS_GLM53_EXL3_GEMV_M1 must be absent, 0, or 1"));

    // rows == 1 reaches the GEMV ONLY through the flag; 2..=8 is unconditional
    // so the verifier path is untouched whether the flag is set or not.
    assert!(compacted.contains(&compact("(2..=8).contains(&rows) || (rows == 1 && gemv_m1_enabled())")));

    // MMODE is derived from the single-row decision, not hardcoded.
    assert!(compacted.contains(&compact("let mmode = u8::from(!plan.gemv_m1);")));
    assert!(
        !compacted.contains(&compact("ELb0ELi2ELi1ELi{}ELb0E")),
        "the MMODE argument must no longer be pinned to 1 in the symbol"
    );
}
