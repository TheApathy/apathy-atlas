// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{
    DeepSeekVisionConfig, build_deepseek_image_block, validate_deepseek_image_tokens,
};

pub(crate) const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub(crate) fn reject_unprepared_tokens(
    tokens: &[u32],
    placeholder: Option<u32>,
    vocab: Option<u32>,
) -> Result<()> {
    ensure!(
        !tokens
            .iter()
            .any(|token| Some(*token) == placeholder || vocab.is_some_and(|v| *token >= v)),
        "DeepSeek image tokens require matching image pixels through the chat image API"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn expand_image_placeholders(
    tokens: &[u32],
    placeholder: u32,
    grids: &[(usize, usize)],
    cfg: &DeepSeekVisionConfig,
    vocab: u32,
    max_seq_len: usize,
    completion_tokens: usize,
    initial_prefill_tokens: usize,
) -> Result<Vec<u32>> {
    cfg.validate()?;
    ensure!(
        placeholder < vocab,
        "DeepSeek image placeholder must be in the text vocabulary"
    );
    ensure!(
        tokens.iter().all(|token| *token < vocab),
        "Unprepared DeepSeek visual sentinel in prompt"
    );
    ensure!(
        tokens.iter().filter(|token| **token == placeholder).count() == grids.len(),
        "DeepSeek image placeholder count does not match image inputs"
    );
    let mut output = Vec::with_capacity(tokens.len());
    let mut images = grids.iter();
    for &token in tokens {
        if token != placeholder {
            output.push(token);
            continue;
        }
        let &(gh, gw) = images.next().context("Missing DeepSeek image grid")?;
        ensure!(gh > 0 && gw > 0, "DeepSeek image grid must be nonempty");
        let block = build_deepseek_image_block(
            gh.div_ceil(cfg.downsample_ratio),
            gw.div_ceil(cfg.downsample_ratio),
            output.len(),
            vocab,
        )?;
        ensure!(
            block.token_ids.len() <= cfg.max_tokens,
            "DeepSeek image exceeds configured token budget"
        );
        let end = output
            .len()
            .checked_add(block.token_ids.len())
            .context("DeepSeek expanded prompt overflow")?;
        ensure!(
            end <= initial_prefill_tokens,
            "Every DeepSeek image END must lie within the initial prefill chunk; increase --max-prefill-tokens or shorten preceding text"
        );
        ensure!(
            end <= max_seq_len,
            "DeepSeek expanded image exceeds context length"
        );
        output.extend(block.token_ids);
    }
    ensure!(
        output
            .len()
            .checked_add(completion_tokens)
            .is_some_and(|total| total <= max_seq_len),
        "DeepSeek expanded prompt and completion exceed context length"
    );
    let spans = validate_deepseek_image_tokens(&output, vocab, cfg.max_tokens)?;
    ensure!(
        spans.len() == grids.len(),
        "DeepSeek image sentinel span count drift"
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_vision_preprocess::tests::config;

    #[test]
    fn deepseek_expansion_preserves_text_and_uses_absolute_n_layout() {
        let got = expand_image_placeholders(
            &[8, 90, 7, 90, 6],
            90,
            &[(3, 6), (6, 3)],
            &config(),
            100,
            1000,
            32,
            1000,
        )
        .unwrap();
        let mut expected = vec![8];
        expected.extend(
            atlas_core::config::build_deepseek_image_block(1, 2, 1, 100)
                .unwrap()
                .token_ids,
        );
        expected.push(7);
        expected.extend(
            atlas_core::config::build_deepseek_image_block(2, 1, expected.len(), 100)
                .unwrap()
                .token_ids,
        );
        expected.push(6);
        assert_eq!(got, expected);
    }

    #[test]
    fn deepseek_expansion_rejects_missing_extra_or_raw_image_tokens() {
        for (tokens, grids) in [
            (vec![90], vec![]),
            (vec![1], vec![(3, 3)]),
            (vec![90, 90], vec![(3, 3)]),
            (vec![100, 90], vec![(3, 3)]),
            (vec![90], vec![(0, 3)]),
        ] {
            assert!(
                expand_image_placeholders(&tokens, 90, &grids, &config(), 100, 1000, 1, 1000)
                    .is_err()
            );
        }
        assert!(reject_unprepared_tokens(&[90], Some(90), Some(100)).is_err());
        assert!(reject_unprepared_tokens(&[104], Some(90), Some(100)).is_err());
        assert!(reject_unprepared_tokens(&[89], Some(90), Some(100)).is_ok());
    }

    #[test]
    fn deepseek_expansion_checks_initial_chunk_and_completion_capacity() {
        let valid = expand_image_placeholders(&[90, 7], 90, &[(3, 3)], &config(), 100, 100, 1, 100)
            .unwrap();
        assert!(
            expand_image_placeholders(&[90, 7], 90, &[(3, 3)], &config(), 100, valid.len(), 1, 100)
                .is_err()
        );
        assert!(
            expand_image_placeholders(
                &[90, 7],
                90,
                &[(3, 3)],
                &config(),
                100,
                100,
                1,
                valid.len() - 2
            )
            .is_err()
        );
        assert!(
            expand_image_placeholders(
                &[90, 7],
                90,
                &[(3, 3)],
                &config(),
                100,
                100,
                1,
                valid.len() - 1
            )
            .is_ok()
        );
        assert!(
            expand_image_placeholders(
                &[90],
                90,
                &[(3, 3)],
                &config(),
                100,
                usize::MAX,
                usize::MAX,
                100
            )
            .is_err()
        );
    }
}
