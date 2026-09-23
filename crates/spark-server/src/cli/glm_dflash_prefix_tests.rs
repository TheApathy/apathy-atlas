// SPDX-License-Identifier: AGPL-3.0-only

use super::ServeArgs;
use clap::Parser;

fn parse(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
    let mut args = vec!["spark", "--model-from-path", "/unused/glm"];
    args.extend_from_slice(extra);
    ServeArgs::try_parse_from(args)
}

#[test]
fn glm_dflash_prefix_accepts_each_explicit_limit_without_changing_gamma() {
    for limit in 1..=7 {
        let value = limit.to_string();
        let args = parse(&[
            "--dflash",
            "--dflash-gamma",
            "8",
            "--glm-dflash-max-drafts",
            &value,
        ])
        .expect("a bounded verification prefix must be expressible independently of gamma");
        assert!(args.dflash);
        assert_eq!(args.dflash_gamma, 8);
        assert_eq!(args.glm_dflash_max_drafts, Some(limit));
    }
}

#[test]
fn glm_dflash_prefix_rejects_invalid_limits() {
    for value in ["0", "8", "-1", "1.5", "999999999999999999999999"] {
        assert!(
            parse(&[
                "--dflash",
                "--dflash-gamma",
                "8",
                "--glm-dflash-max-drafts",
                value,
            ])
            .is_err()
        );
    }
}

#[test]
fn glm_dflash_prefix_requires_the_dflash_method() {
    assert!(parse(&["--glm-dflash-max-drafts", "3"]).is_err());
    for method in ["--speculative", "--self-speculative", "--ngram-speculative"] {
        assert!(parse(&[method, "--glm-dflash-max-drafts", "3"]).is_err());
        assert!(parse(&["--dflash", method, "--glm-dflash-max-drafts", "3"]).is_err());
    }
}

#[test]
fn glm_dflash_prefix_omission_preserves_the_legacy_cli() {
    let args = parse(&["--dflash", "--dflash-gamma", "8"]).unwrap();
    assert!(args.dflash);
    assert_eq!(args.dflash_gamma, 8);
    assert_eq!(args.num_drafts, 1);
    assert_eq!(args.glm_dflash_max_drafts, None);
}
