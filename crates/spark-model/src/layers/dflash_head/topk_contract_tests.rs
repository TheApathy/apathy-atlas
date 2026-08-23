// SPDX-License-Identifier: AGPL-3.0-only

use std::cmp::Ordering;

use super::super::propose::parse_ddtree_top_k;
use super::{
    BlockDiffusionDraftHead, checked_topk_difference, checked_topk_layout, validate_topk_request,
};

const CUDA_SOURCE: &str = include_str!("../../../../../kernels/gb10/common/argmax_bf16.cu");
const FORWARD_SOURCE: &str = include_str!("forward_block.rs");
const NOISE_SOURCE: &str = include_str!("noise_pass.rs");
const PROPOSE_SOURCE: &str = include_str!("propose.rs");
const ASYNC_SOURCE: &str = include_str!("async_propose.rs");

fn encoded_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn encoded_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode(
    tokens: &[u32],
    logits: &[f32],
    rows: usize,
    k: usize,
    vocab: u32,
) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
    BlockDiffusionDraftHead::decode_topk_bytes(
        encoded_u32(tokens),
        encoded_f32(logits),
        rows,
        rows,
        k,
        vocab as usize,
    )
}

fn reference_topk(row: &[f32], k: usize) -> Option<Vec<(u32, f32)>> {
    if row.iter().any(|value| value.is_nan()) || !row.iter().any(|value| *value > f32::NEG_INFINITY)
    {
        return None;
    }
    let mut candidates: Vec<(u32, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .map(|(token, score)| (token as u32, score))
        .collect();
    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    candidates.truncate(k);
    Some(candidates)
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn host_oracle_freezes_special_values_ties_and_extreme_bf16() {
    let extreme = [
        f32::from_bits(u32::from(0xff7f_u16) << 16),
        f32::from_bits(u32::from(0xff7e_u16) << 16),
    ];
    assert_eq!(reference_topk(&extreme, 2).unwrap()[0].0, 1);

    let mut ties = vec![-8.0; 2_048];
    ties[1] = f32::INFINITY;
    ties[1_024] = f32::INFINITY;
    let winners = reference_topk(&ties, 2).unwrap();
    assert_eq!(winners, vec![(1, f32::INFINITY), (1_024, f32::INFINITY)]);

    assert_eq!(reference_topk(&[-0.0, 0.0], 2).unwrap()[0].0, 0);
    assert!(reference_topk(&[f32::NEG_INFINITY; 3], 2).is_none());
    assert!(reference_topk(&[3.0, f32::NAN, 2.0], 2).is_none());

    let descending: Vec<f32> = (0..16).rev().map(|value| value as f32).collect();
    assert_eq!(reference_topk(&descending, 1).unwrap(), vec![(0, 15.0)]);
    let all = reference_topk(&descending, 16).unwrap();
    assert_eq!(all.len(), 16);
    assert_eq!(all.first(), Some(&(0, 15.0)));
    assert_eq!(all.last(), Some(&(15, 0.0)));
}

#[test]
fn decoder_accepts_exact_numeric_order_and_mixed_negative_infinity_tail() {
    let expected_tokens = vec![1, 1_024, 7, 3, 0, 2, 4, 6];
    let expected_logits = vec![
        f32::INFINITY,
        f32::INFINITY,
        3.0,
        f32::NEG_INFINITY,
        7.0,
        6.0,
        5.0,
        4.0,
    ];
    let (tokens, logits) = decode(&expected_tokens, &expected_logits, 2, 4, 2_048).unwrap();
    assert_eq!(tokens, expected_tokens);
    assert_eq!(logits, expected_logits);
}

#[test]
fn decoder_rejects_marker_nan_all_negative_infinity_and_malformed_rows() {
    let invalid_cases = [
        (
            vec![u32::MAX, u32::MAX],
            vec![f32::NAN, f32::NAN],
            "invalid marker",
        ),
        (vec![u32::MAX, 1], vec![f32::NAN, 2.0], "partial marker"),
        (vec![1, 2], vec![3.0, f32::NAN], "NaN"),
        (
            vec![1, 2],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
            "no usable score",
        ),
        (vec![1, 8], vec![3.0, 2.0], "out of range"),
        (vec![1, 1], vec![3.0, 2.0], "duplicate"),
        (vec![1, 2], vec![2.0, 3.0], "score order"),
        (vec![2, 1], vec![3.0, 3.0], "token order"),
    ];
    for (tokens, logits, expected) in invalid_cases {
        let error = decode(&tokens, &logits, 1, 2, 8).expect_err(expected);
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?}, got {error:#}"
        );
    }

    let error = BlockDiffusionDraftHead::decode_topk_bytes(
        encoded_u32(&[1, 2]),
        encoded_f32(&[3.0]),
        1,
        1,
        2,
        8,
    )
    .expect_err("wrong byte length");
    assert!(error.to_string().contains("byte length"));
}

#[test]
fn checked_difference_normalizes_equal_infinities_and_rejects_nan() {
    assert_eq!(checked_topk_difference(4.0, 4.0).unwrap(), 0.0);
    assert_eq!(
        checked_topk_difference(-0.0, 0.0).unwrap().to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        checked_topk_difference(f32::INFINITY, f32::INFINITY)
            .unwrap()
            .to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        checked_topk_difference(f32::NEG_INFINITY, f32::NEG_INFINITY)
            .unwrap()
            .to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        checked_topk_difference(2.0, f32::NEG_INFINITY).unwrap(),
        f32::INFINITY
    );
    assert!(checked_topk_difference(f32::NAN, 1.0).is_err());
}

#[test]
fn requested_k_is_validated_exactly_without_clamping_or_malformed_defaults() {
    for (k, vocab, expected) in [
        (0, 248_077, "k >= 1"),
        (17, 248_077, "k <= 16"),
        (4, 3, "k <= vocab"),
    ] {
        let error = validate_topk_request(k, vocab).expect_err("invalid K must fail");
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error:#}"
        );
    }
    assert_eq!(validate_topk_request(2, 248_077).unwrap(), 2);

    let absent = || Err(std::env::VarError::NotPresent);
    let value = |raw: &str| Ok(raw.to_owned());
    assert_eq!(parse_ddtree_top_k(absent(), absent(), 248_077).unwrap(), 8);
    assert_eq!(
        parse_ddtree_top_k(value("2"), value("7"), 248_077).unwrap(),
        2
    );
    assert_eq!(
        parse_ddtree_top_k(absent(), value("3"), 248_077).unwrap(),
        3
    );
    for raw in ["", "not-a-number", "0", "17", "184467440737095516160"] {
        assert!(
            parse_ddtree_top_k(value(raw), absent(), 248_077).is_err(),
            "explicit invalid K {raw:?} must not be substituted"
        );
    }
    assert!(
        parse_ddtree_top_k(value("not-a-number"), value("2"), 248_077).is_err(),
        "present malformed primary value must not fall through to the alias"
    );
    assert!(parse_ddtree_top_k(value("4"), absent(), 3).is_err());
    assert!(
        parse_ddtree_top_k(
            Err(std::env::VarError::NotUnicode(std::ffi::OsString::from(
                "bad"
            ))),
            value("2"),
            248_077,
        )
        .is_err(),
        "present non-Unicode primary value must not fall through to the alias"
    );
}

#[test]
fn checked_layout_closes_capacity_cast_scalar_and_byte_boundaries() {
    let layout = checked_topk_layout(3, 16, 248_077, 2).unwrap();
    assert_eq!(layout.num_rows, 3);
    assert_eq!(layout.vocab, 248_077);
    assert_eq!(layout.k, 2);
    assert_eq!(layout.bytes, 24);

    for (rows, capacity, vocab, k, expected) in [
        (17, 16, 248_077, 2, "scratch capacity"),
        (usize::MAX, usize::MAX, 248_077, 2, "u32"),
        (0, 16, 0, 2, "vocab > 0"),
        (0, 16, u32::MAX as usize + 1, 2, "vocab exceeds u32"),
        (0, 16, 248_077, 0, "k >= 1"),
        (1, 16, 248_077, 17, "k <= 16"),
        (1, 16, 1, 2, "k <= vocab"),
    ] {
        let error = checked_topk_layout(rows, capacity, vocab, k).expect_err(expected);
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?}, got {error:#}"
        );
    }

    assert!(
        BlockDiffusionDraftHead::decode_topk_bytes(Vec::new(), Vec::new(), 0, 0, 0, 8).is_err(),
        "rows=0,k=0 must not decode as an empty success"
    );
}

#[test]
fn valid_direct_top2_pair_decodes_exactly() {
    let (tokens, logits) = decode(&[1, 7], &[f32::INFINITY, 3.0], 1, 2, 248_077).unwrap();
    assert_eq!(tokens, vec![1, 7]);
    assert_eq!(logits, vec![f32::INFINITY, 3.0]);
}

#[test]
fn cuda_topk_source_has_row_status_total_order_and_unchanged_abi() {
    let topk = CUDA_SOURCE
        .split_once("// Top-K over BF16 logits")
        .expect("top-K marker")
        .1;
    let compact: String = topk.chars().filter(|ch| !ch.is_whitespace()).collect();
    assert!(compact.contains("booltopk_candidate_better("));
    assert!(compact.contains("isnan(v)"));
    assert!(compact.contains("v>-CUDART_INF_F"));
    assert!(compact.contains("s_row_invalid"));
    assert!(compact.contains("s_row_usable"));
    assert!(compact.contains("top_indices[out]=0xFFFFFFFFu"));
    assert!(compact.contains("top_logits[out]=CUDART_NAN_F"));
    assert!(compact.contains("floatlocal_max=-CUDART_INF_F"));
    assert!(compact.contains("unsignedintlocal_idx=0xFFFFFFFFu"));
    assert!(!compact.contains("floatlocal_max=-1e30f"));
    let signature = topk
        .split_once("extern \"C\" __global__ void topk_bf16(")
        .expect("top-K ABI")
        .1;
    let signature = signature
        .split_once(") {")
        .map(|(body, _)| body)
        .expect("top-K ABI terminator");
    let signature: String = signature
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<String>()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    assert_eq!(
        signature,
        "const__nv_bfloat16*__restrict__logits,unsignedint*__restrict__top_indices,float*__restrict__top_logits,unsignedintnum_rows,unsignedintvocab,unsignedintk"
    );
}

#[test]
fn every_topk_score_difference_uses_the_checked_helper() {
    assert!(!FORWARD_SOURCE.contains("top2[i * 2] - top2[i * 2 + 1]"));
    assert!(!FORWARD_SOURCE.contains("let margin = top1 - top2;"));
    assert!(FORWARD_SOURCE.matches("checked_topk_difference").count() >= 3);

    for raw in [
        "topk_logits[2 * r] - topk_logits[2 * r + 1]",
        "topk_logits[base + i] - row_max",
    ] {
        assert!(!PROPOSE_SOURCE.contains(raw), "unchecked arithmetic: {raw}");
    }
    assert!(PROPOSE_SOURCE.matches("checked_topk_difference").count() >= 5);

    assert!(!ASYNC_SOURCE.contains("topk_logits[2 * r] - topk_logits[2 * r + 1]"));
    assert!(ASYNC_SOURCE.contains("checked_topk_difference"));
}

#[test]
fn invalid_rows_have_explicit_direct_sync_and_async_fallbacks() {
    let forward = compact(FORWARD_SOURCE);
    assert!(forward.contains("DFlashdenoisetop-2row{i}isinvalid"));
    assert!(forward.contains("degradingtobootstrap\");returnOk(Vec::new())"));
    assert!(forward.contains(
        "DFlashadaptivetop-2outputisinvalid({error:#});degradingtobootstrap\");returnOk(Vec::new())"
    ));
    assert!(!forward.contains("margin_top2_invalid"));
    assert!(forward.contains("drafts.truncate(ifinvalid_margin{cutoff}else{cutoff.max(1)});"));

    for fallback in [
        "DFlashCATERPILLARinvalidtop-2row{r}",
        "DFlashFREE_SLOTSinvalidtop-2result",
        "DFlashBRANCHinvalidtop-2row{r}",
    ] {
        assert!(compact(PROPOSE_SOURCE).contains(fallback));
    }
    assert!(compact(PROPOSE_SOURCE).contains("filter(|&i|topk_logits[base+i]>f32::NEG_INFINITY)"));

    let async_source = compact(ASYNC_SOURCE);
    assert!(async_source.contains("DFLASH_ASYNC:invalidtop-Kmargins"));
    assert!(async_source.contains("dstate.pending_tree_payload=None;"));
    assert!(async_source.contains(
        "DFLASH_ASYNC:top-KD2Hfailed({e:#});notreepayload\");dstate.pending_tree_payload=None;"
    ));
    assert!(async_source.contains("Ok(_)=>{dstate.pending_tree_payload=None;}"));
}

#[test]
fn consumers_validate_original_k_and_direct_token_score_pairs() {
    assert!(!FORWARD_SOURCE.contains("k.clamp(1, super::DDTREE_TOP_K_MAX)"));
    assert!(!PROPOSE_SOURCE.contains(".clamp(1, super::DDTREE_TOP_K_MAX)"));

    let enqueue = FORWARD_SOURCE
        .split_once("pub(super) fn enqueue_topk_on_stream(")
        .expect("enqueue helper")
        .1;
    let validate_at = enqueue
        .find("checked_topk_layout")
        .expect("checked layout validation");
    let memset_at = enqueue.find("memset_async").expect("top-K memset");
    assert!(
        validate_at < memset_at,
        "K must be validated before any GPU operation"
    );
    assert!(FORWARD_SOURCE.contains(".checked_mul(k)"));
    assert!(
        FORWARD_SOURCE.matches("checked_topk_layout(").count() >= 6,
        "the pure helper plus both direct paths and extract/enqueue/collect must share one layout check"
    );

    let extract = FORWARD_SOURCE
        .split_once("pub(super) fn extract_topk_from_logits(")
        .expect("extract helper")
        .1;
    assert!(
        extract.find("checked_topk_layout").unwrap()
            < extract.find("enqueue_topk_on_stream").unwrap()
    );
    let collect = FORWARD_SOURCE
        .split_once("pub(super) fn collect_topk_d2h(")
        .expect("collect helper")
        .1;
    assert!(collect.find("checked_topk_layout").unwrap() < collect.find("vec![0u8;").unwrap());

    let propose = compact(PROPOSE_SOURCE);
    assert!(propose.contains(
        "Err(error)=>{tracing::warn!(\"DDTreeinvalidrequestedtop-K({error:#});emittingflat-chainpayload\")"
    ));

    assert!(
        FORWARD_SOURCE.matches("Self::decode_topk_bytes(").count() >= 3,
        "both direct top-2 paths must use the central token+score decoder"
    );
    assert!(
        FORWARD_SOURCE
            .matches("copy_d2h(self.scratch.topk_tokens_dev")
            .count()
            >= 3,
        "both direct paths and async collection must copy token IDs"
    );
}

#[test]
fn proposal_scratch_preamble_does_not_clear_fully_produced_buffers() {
    let preamble = FORWARD_SOURCE
        .split_once("let row_layout = super::DraftRowLayout::for_family")
        .expect("row layout")
        .1
        .split_once("let target_hidden_dim")
        .expect("target hidden dimension")
        .0;
    assert!(
        !preamble.contains("gpu.memset"),
        "the proposal preamble must not clear buffers whose active regions are fully produced later"
    );
    assert!(
        NOISE_SOURCE.contains("ops::batched_embed") && NOISE_SOURCE.contains("ops::argmax_bf16"),
        "the producer overwrite contract must remain visible in the forward path"
    );
}
