// SPDX-License-Identifier: AGPL-3.0-only

//! CPU reference primitives for the RadixArk DSpark confidence projection.
//!
//! These routines are an executable shape/order oracle, not a runtime
//! planner. SGLang PR 34966's default `static` mode verifies every proposal
//! and skips the confidence head. Atlas keeps dynamic cap-accept/compact
//! fail-closed until it has the same ragged layout and profiled SPS table.

use anyhow::{Result, ensure};

/// Reference output of the learned scalar confidence affine.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkConfidenceProjection {
    pub raw_logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

/// Gather the Markov W1 features used by the confidence head.
///
/// For gamma proposal rows the predecessor sequence is exactly
/// `[anchor, draft_0, ..., draft_(gamma-2)]`. The final drafted token is not
/// a predecessor for another confidence row and is therefore not gathered.
pub fn build_markov_embedding_stack(
    anchor_token: usize,
    draft_tokens: &[usize],
    markov_w1: &[f32],
    vocab_size: usize,
    markov_rank: usize,
    gamma: usize,
) -> Result<Vec<f32>> {
    ensure!(gamma > 0, "DSpark confidence gamma must be non-zero");
    ensure!(
        markov_rank > 0,
        "DSpark confidence Markov rank must be non-zero"
    );
    ensure!(
        markov_w1.len() == vocab_size.saturating_mul(markov_rank),
        "DSpark Markov W1 has {} values; expected vocab_size*rank={}*{}={}",
        markov_w1.len(),
        vocab_size,
        markov_rank,
        vocab_size.saturating_mul(markov_rank)
    );
    ensure!(
        draft_tokens.len() >= gamma.saturating_sub(1),
        "DSpark confidence needs {} predecessor drafts for gamma={gamma}; got {}",
        gamma.saturating_sub(1),
        draft_tokens.len()
    );

    let predecessors =
        std::iter::once(anchor_token).chain(draft_tokens.iter().copied().take(gamma - 1));
    let mut stack = Vec::with_capacity(gamma * markov_rank);
    for token in predecessors {
        ensure!(
            token < vocab_size,
            "DSpark confidence predecessor token {token} is outside vocab_size={vocab_size}"
        );
        let start = token * markov_rank;
        stack.extend_from_slice(&markov_w1[start..start + markov_rank]);
    }
    Ok(stack)
}

/// Evaluate `sigmoid([draft_hidden; markov_embedding] @ W^T + bias)` in FP32.
///
/// The concatenation order is load-bearing: the public checkpoint stores
/// `[hidden_size + markov_rank]`, with hidden features first. Passing a
/// non-zero rank requires one Markov feature row per proposal.
pub fn project_confidence_reference(
    draft_hidden: &[f32],
    markov_stack: Option<&[f32]>,
    proj_weight: &[f32],
    bias: f32,
    hidden_size: usize,
    markov_rank: usize,
    gamma: usize,
) -> Result<DsparkConfidenceProjection> {
    ensure!(gamma > 0, "DSpark confidence gamma must be non-zero");
    ensure!(
        hidden_size > 0,
        "DSpark confidence hidden size must be non-zero"
    );
    ensure!(
        draft_hidden.len() == gamma.saturating_mul(hidden_size),
        "DSpark confidence hidden buffer has {} values; expected gamma*hidden={}*{}={}",
        draft_hidden.len(),
        gamma,
        hidden_size,
        gamma.saturating_mul(hidden_size)
    );
    let input_dim = hidden_size
        .checked_add(markov_rank)
        .ok_or_else(|| anyhow::anyhow!("DSpark confidence input width overflow"))?;
    ensure!(
        proj_weight.len() == input_dim,
        "DSpark confidence projection has {} values; expected input_dim={input_dim}",
        proj_weight.len()
    );
    match (markov_rank, markov_stack) {
        (0, None) => {}
        (0, Some(_)) => {
            anyhow::bail!("DSpark confidence received Markov features with markov_rank=0")
        }
        (_, None) => anyhow::bail!(
            "DSpark confidence projection requires Markov features for markov_rank={markov_rank}"
        ),
        (rank, Some(stack)) => ensure!(
            stack.len() == gamma.saturating_mul(rank),
            "DSpark confidence Markov buffer has {} values; expected gamma*rank={}*{}={}",
            stack.len(),
            gamma,
            rank,
            gamma.saturating_mul(rank)
        ),
    }

    let mut raw_logits = Vec::with_capacity(gamma);
    let mut probabilities = Vec::with_capacity(gamma);
    for row in 0..gamma {
        let hidden = &draft_hidden[row * hidden_size..(row + 1) * hidden_size];
        let mut raw = bias;
        for (value, weight) in hidden.iter().zip(&proj_weight[..hidden_size]) {
            raw += value * weight;
        }
        if let Some(stack) = markov_stack {
            let markov = &stack[row * markov_rank..(row + 1) * markov_rank];
            for (value, weight) in markov.iter().zip(&proj_weight[hidden_size..]) {
                raw += value * weight;
            }
        }
        let probability = if raw >= 0.0 {
            1.0 / (1.0 + (-raw).exp())
        } else {
            let exp = raw.exp();
            exp / (1.0 + exp)
        };
        raw_logits.push(raw);
        probabilities.push(probability);
    }
    Ok(DsparkConfidenceProjection {
        raw_logits,
        probabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markov_stack_uses_anchor_then_preceding_drafts() {
        let w1 = vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let stack = build_markov_embedding_stack(2, &[3, 1, 0], &w1, 4, 2, 3).unwrap();
        assert_eq!(stack, vec![20.0, 21.0, 30.0, 31.0, 10.0, 11.0]);
    }

    #[test]
    fn projection_concatenates_hidden_before_markov_and_applies_sigmoid() {
        let result = project_confidence_reference(
            &[1.0, 2.0, 3.0, 4.0],
            Some(&[10.0, 20.0]),
            &[1.0, 0.5, 0.1],
            -1.0,
            2,
            1,
            2,
        )
        .unwrap();
        assert_eq!(result.raw_logits, vec![2.0, 6.0]);
        assert!((result.probabilities[0] - 0.880_797).abs() < 1e-6);
        assert!((result.probabilities[1] - 0.997_527_36).abs() < 1e-6);
    }

    #[test]
    fn projection_supports_the_no_markov_checkpoint_variant() {
        let result =
            project_confidence_reference(&[2.0, -1.0], None, &[0.5, 2.0], 0.25, 2, 0, 1).unwrap();
        assert_eq!(result.raw_logits, vec![-0.75]);
        assert!((result.probabilities[0] - 0.320_821_3).abs() < 1e-6);
    }

    #[test]
    fn malformed_shapes_and_missing_markov_features_fail_closed() {
        assert!(build_markov_embedding_stack(0, &[], &[0.0; 4], 2, 2, 2).is_err());
        assert!(project_confidence_reference(&[0.0; 4], None, &[0.0; 3], 0.0, 2, 1, 2).is_err());
        assert!(
            project_confidence_reference(&[0.0; 4], Some(&[0.0; 1]), &[0.0; 3], 0.0, 2, 1, 2,)
                .is_err()
        );
    }
}
