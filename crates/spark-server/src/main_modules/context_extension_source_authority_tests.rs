// SPDX-License-Identifier: AGPL-3.0-only

use std::fmt::Write as _;

const CONTEXT: &str = include_str!("context_extension.rs");
const RUNTIME: &str = include_str!("context_extension_runtime.rs");
const ADMISSION: &str = include_str!("context_extension_admission.rs");
const SERVE: &str = include_str!("serve.rs");
const SERVE_PHASES: &str = include_str!("serve_phases/mod.rs");
const BUILD: &str = include_str!("serve_phases/build.rs");
const CONTEXT_SHA256: &str = "d3a53b93ed3ccf1ab30c344549676877b268eb55bc12f8d98e72186f5892f1b4";
const RUNTIME_SHA256: &str = "5a074f17f89325cf75b8cc2553d1253680e243b622324ccfcd6f7cbf1b3587dd";
const ADMISSION_SHA256: &str = "c1f12ecd43e1a8f1263033723802f2fcde0841cf503b365770c07d0804b6bfa6";
// Reviewed delta: publish effective max_batch_size for early C1 image admission.
// Removing that single AppState initializer field reproduces the prior hash.
// Re-pinned 2026-09-21 for the TUI port. The ONLY change to serve.rs is the
// `tui::start(..)` call site gaining a third argument (the dashboard's
// optional ModelHost, passed as `None`) plus the comment explaining it.
// Diffed before re-pinning: no context-extension path, admission rule or
// phase ordering moved. The point of this hash is that somebody LOOKS when
// it breaks — updating it without the diff is how it becomes a rubber stamp.
const SERVE_SHA256: &str = "2fc2a19424e6723109b77f0f82c517263a01581292569156b1480a31e388743d";
const SERVE_PHASES_SHA256: &str =
    "e3e84d068c761ff43ad49a04c67d3cd37a8107f99110c919bd016832219c9f1e";
const BUILD_SHA256: &str = "63b5663ef0880e17d3725ec8833b6d82c20ef45567bc6cfbd41a8136c2005f3e";

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const ROUNDS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn compress(state: &mut [u32; 8], chunk: &[u8; 64]) {
    let mut schedule = [0_u32; 64];
    for (word, input) in schedule[..16].iter_mut().zip(chunk.chunks_exact(4)) {
        *word = u32::from_be_bytes(input.try_into().unwrap());
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(ROUNDS[index])
            .wrapping_add(schedule[index]);
        let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        (h, g, f, e, d, c, b, a) = (
            g,
            f,
            e,
            d.wrapping_add(temp1),
            c,
            b,
            a,
            temp1.wrapping_add(temp2),
        );
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut state = INITIAL;
    let mut chunks = bytes.chunks_exact(64);
    for chunk in &mut chunks {
        compress(&mut state, chunk.try_into().unwrap());
    }
    let remainder = chunks.remainder();
    let mut tail = [0_u8; 128];
    tail[..remainder.len()].copy_from_slice(remainder);
    tail[remainder.len()] = 0x80;
    let padded_len = if remainder.len() < 56 { 64 } else { 128 };
    let bit_len = u64::try_from(bytes.len()).unwrap().checked_mul(8).unwrap();
    tail[padded_len - 8..padded_len].copy_from_slice(&bit_len.to_be_bytes());
    for chunk in tail[..padded_len].chunks_exact(64) {
        compress(&mut state, chunk.try_into().unwrap());
    }
    let mut digest = String::with_capacity(64);
    for word in state {
        write!(&mut digest, "{word:08x}").unwrap();
    }
    digest
}

fn mutate_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(source.matches(from).count(), 1, "ambiguous hostile: {from}");
    source.replacen(from, to, 1)
}

#[test]
fn sha256_padding_boundaries_and_production_sources_are_exact() {
    let vectors = [
        (
            Vec::new(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc".to_vec(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            vec![b'a'; 55],
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
        ),
        (
            vec![b'a'; 56],
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
        ),
    ];
    for (input, expected) in vectors {
        assert_eq!(sha256_hex(&input), expected);
    }
    assert_eq!(sha256_hex(CONTEXT.as_bytes()), CONTEXT_SHA256);
    assert_eq!(sha256_hex(RUNTIME.as_bytes()), RUNTIME_SHA256);
    assert_eq!(sha256_hex(ADMISSION.as_bytes()), ADMISSION_SHA256);
    assert_eq!(sha256_hex(SERVE.as_bytes()), SERVE_SHA256);
    assert_eq!(sha256_hex(SERVE_PHASES.as_bytes()), SERVE_PHASES_SHA256);
    assert_eq!(sha256_hex(BUILD.as_bytes()), BUILD_SHA256);
}

#[test]
fn consumer_and_opaque_receipt_reject_omission_forgery_and_external_drift() {
    for (source, expected, from, to) in [
        (
            BUILD,
            BUILD_SHA256,
            "        admitted.max_seq_len(),",
            "        1_048_576,",
        ),
        (
            BUILD,
            BUILD_SHA256,
            "    context_admission: ContextAdmissionReceipt,",
            "    _context_admission: ContextAdmissionReceipt,",
        ),
        (
            SERVE_PHASES,
            SERVE_PHASES_SHA256,
            "    build_high_speed_swap_config, build_model, build_prefix_cache, maybe_run_ep_worker,",
            "    build_high_speed_swap_config, build_prefix_cache, maybe_run_ep_worker,",
        ),
        (SERVE, SERVE_SHA256, "        context_admission,\n", ""),
        (
            ADMISSION,
            ADMISSION_SHA256,
            "pub(crate) struct ContextAdmissionReceipt {\n    mode: ContextRuntimeMode,",
            "pub(crate) struct ContextAdmissionReceipt {\n    pub(crate) mode: ContextRuntimeMode,",
        ),
        (
            ADMISSION,
            ADMISSION_SHA256,
            "    pub(super) fn mint(",
            "    pub(crate) fn mint(",
        ),
        (
            ADMISSION,
            ADMISSION_SHA256,
            "    pub(crate) fn consume(self, observed_config_capacity: usize)",
            "    pub(crate) fn consume(&self, observed_config_capacity: usize)",
        ),
    ] {
        let mutant = mutate_once(source, from, to);
        assert_ne!(sha256_hex(mutant.as_bytes()), expected);
    }
    assert!(!ADMISSION.contains("derive(Clone"));
    assert!(!ADMISSION.contains("derive(Copy"));
    assert!(
        BUILD
            .contains("let admitted = context_admission.consume(config.max_position_embeddings)?;")
    );
    assert!(!BUILD.contains("        args.max_seq_len,"));

    let mut omitted = BUILD.replacen(
        "use super::super::context_extension::ContextAdmissionReceipt;\n\n",
        "",
        1,
    );
    omitted = omitted.replacen("    context_admission: ContextAdmissionReceipt,\n", "", 1);
    omitted = omitted.replacen(
        "    let admitted = context_admission.consume(config.max_position_embeddings)?;\n",
        "",
        1,
    );
    for (from, to) in [
        ("admitted.block_size()", "args.block_size"),
        ("admitted.max_seq_len()", "args.max_seq_len"),
        ("admitted.max_batch_size()", "args.max_batch_size"),
        ("admitted.speculative()", "args.speculative || args.dflash"),
        (
            "admitted.self_speculative()",
            "args.self_speculative || args.ngram_speculative",
        ),
        ("admitted.dflash()", "args.dflash"),
        (
            "admitted.hss_cache_blocks_per_seq()",
            "args.high_speed_swap.then_some(args.high_speed_swap_cache_blocks_per_seq)",
        ),
    ] {
        omitted = omitted.replace(from, to);
    }
    assert_ne!(sha256_hex(omitted.as_bytes()), BUILD_SHA256);
}

#[test]
fn full_source_authority_rejects_post_validation_and_cfg_split_drift() {
    let seam = "    if let Some(ref qc) = config.quantization_config {";
    for insertion in [
        "    args.dflash = true;\n",
        "    config.max_position_embeddings = 1_048_576;\n",
        "    args.max_batch_size = 8;\n",
    ] {
        let mutant = mutate_once(SERVE, seam, &format!("{insertion}{seam}"));
        assert_ne!(sha256_hex(mutant.as_bytes()), SERVE_SHA256);
    }
    let validator_body = "    let extended = match extension {";
    let mutant = mutate_once(
        RUNTIME,
        validator_body,
        "    #[cfg(not(test))]\n    if extension.is_some() { return Ok(ContextAdmissionReceipt::mint(mode, false)); }\n    let extended = match extension {",
    );
    assert_ne!(sha256_hex(mutant.as_bytes()), RUNTIME_SHA256);
}
