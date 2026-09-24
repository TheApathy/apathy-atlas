// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The default path must refuse, and say why in terms a reader can act on.
#[test]
fn admission_refuses_by_default_and_names_both_gates() {
    // The bring-up escape must not be set in a normal test run; if it is, this
    // test is measuring the wrong thing and should say so rather than pass.
    if Glm53Model::bringup_escape_set() {
        panic!("{GLM53_BRINGUP_ENV} is set in the test environment; unset it");
    }
    for scope in [Glm53AdmissionScope::GgufTargetOnly] {
        let refusal = format!("{:#}", Glm53Model::admit(scope).unwrap_err());
        assert!(refusal.contains("admission is closed"), "{refusal}");
        assert!(refusal.contains(GLM53_BRINGUP_ENV), "{refusal}");
        assert!(refusal.contains("capabilities are incomplete"), "{refusal}");
    }
}

/// EXL3 target-only admission follows its recorded evidence and nothing else;
/// speculation never rides on it.
#[test]
fn exl3_target_only_admission_follows_the_recorded_evidence() {
    if Glm53Model::bringup_escape_set() {
        panic!("{GLM53_BRINGUP_ENV} is set in the test environment; unset it");
    }
    let evidence_ok = Glm53TargetOnlyAdmission::current().validate().is_ok();
    assert_eq!(
        Glm53Model::admit(Glm53AdmissionScope::Exl3TargetOnly).is_ok(),
        evidence_ok
    );
    // Speculation needs the target admitted AND its own bit-identity evidence.
    let speculative_ok = evidence_ok
        && crate::model::glm53::speculative_admission::GLM53_SPECULATIVE_EVIDENCE
            .validate()
            .is_ok();
    assert_eq!(
        Glm53Model::admit(Glm53AdmissionScope::Speculative).is_ok(),
        speculative_ok
    );
}

/// A speculation negative control is never admitted, whatever the evidence says.
#[test]
fn speculative_admission_refuses_its_negative_controls() {
    for name in crate::model::glm53::speculative_admission::SPECULATIVE_CONTROL_ENVS {
        assert!(
            std::env::var(name).is_err(),
            "{name} is set in the test environment; unset it"
        );
    }
    assert!(!crate::model::glm53::speculative_admission::speculative_control_active());
}

/// Runtime allocation must be accounted, not discovered at `alloc` time.
#[test]
fn runtime_allocation_is_the_arena_plus_scratch_plus_logits() {
    let total = Glm53Model::runtime_allocation_bytes();
    assert_eq!(
        total,
        GLM53_KNOWN_ARENA_BYTES + Glm53WalkScratch::required_bytes() + 154_880 * 2 + 4
    );
    // Weights plus the explicit max-M2048 prompt scratch must still clear a
    // 119 GiB box with at least five GiB of headroom.
    let weights = crate::model::glm53::arena::GLM53_IQ2_XXS_TENSOR_BYTES;
    let resident = weights + total;
    assert!(
        resident < 114 * (1u64 << 30),
        "IQ2_XXS resident total {resident} leaves less than five GiB on a 119 GiB box"
    );
}

/// End-to-end forward pass against the real checkpoint on real hardware.
///
/// Ignored by default: it needs an exclusive GB10, ~107 GiB free, the pinned
/// UD-IQ2_XXS shards on disk, and `ATLAS_GLM53_UNVALIDATED_BRINGUP=1`. Run it
/// deliberately:
///
/// ```text
/// ATLAS_GLM53_UNVALIDATED_BRINGUP=1 \
///   cargo test -p spark-model --lib --release first_token -- --ignored --nocapture
/// ```
///
/// This proves the port *executes*. It proves nothing about whether the output
/// is right — that needs the llama.cpp `glm5next` parity comparison.
#[test]
#[ignore = "needs an exclusive GB10, the pinned checkpoint, and the bring-up escape"]
fn first_token_on_real_hardware() {
    use spark_runtime::cuda_backend::AtlasCudaBackend;
    use spark_runtime::weights::gguf::{Glm53QuantProfile, load_glm53_store, open_glm53_files};

    assert!(
        Glm53Model::bringup_escape_set(),
        "set {GLM53_BRINGUP_ENV}=1 to run the unvalidated bring-up path"
    );

    let profile = Glm53QuantProfile::UdIq2Xxs;
    let dir = std::path::Path::new("/home/flocka/models/GLM-5.3-Flash-GGUF")
        .join(profile.directory_name());
    let paths: Vec<_> = profile
        .canonical_file_names()
        .iter()
        .map(|name| dir.join(name))
        .collect();

    let config_json =
        std::fs::read_to_string("/home/flocka/models/GLM-5.3-Flash-meta/config.json").unwrap();
    let mut config = atlas_core::config::parse_config(&config_json).unwrap();
    assert_eq!(config.model_type, "glm5_next");
    // `parse_config` leaves the parallelism world sizes at 0; `spark serve`
    // fills them in from the launch topology. This is single-GPU EP=1/TP=1, and
    // the GLM factory's exact-geometry gate checks both, so a harness that
    // skips this is refused for "unsupported geometry" with nothing about the
    // model actually wrong.
    config.tp_world_size = 1;
    config.ep_world_size = 1;

    let ptx = atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size)
        .expect("no GLM kernel target compiled; rebuild with ATLAS_TARGET_MODEL=glm5.3-flash");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();

    // Identity, metadata and the full 1,412-tensor schema, then the H2D load.
    let mut files = open_glm53_files(profile, &paths).unwrap();
    let reserve =
        usize::try_from(Glm53Model::runtime_allocation_bytes_for(262_144).unwrap()).unwrap();
    let store = load_glm53_store(&mut files, &gpu, reserve)
        .unwrap_or_else(|error| panic!("GLM store load failed: {error}"));
    eprintln!(
        "loaded {} tensors, {} bytes resident",
        store.len(),
        store.total_bytes()
    );

    let weights = crate::factory::Glm53TargetRuntimeWeights::new(profile, &config, store)
        .unwrap_or_else(|error| panic!("GLM runtime weights refused: {error}"));
    // 256K rather than the full 1M: the arena at 1M is 95 MiB larger than this
    // checkpoint leaves free. 256K leaves roughly 21 GiB of headroom, which
    // matters because over-allocating unified memory on GB10 takes the host
    // down rather than returning an error.
    // 64K for the parity probe: it only exercises positions 0 and 1, and the
    // arena scales with capacity. Shrinking it keeps the run well clear of the
    // host-OOM cliff when other processes hold memory. The DSA pool plan uses
    // the architectural 1M constant, not this, so geometry is unaffected.
    const CONTEXT: u32 = 65_536;
    let model = Glm53Model::new(std::sync::Arc::new(gpu), weights, CONTEXT).unwrap();

    // Two tokens: one prefill step and one decode step, so the recurrence has
    // to carry state across a commit rather than only run once.
    let stream = 0u64;
    // Reference (llama.cpp glm5next, same GGUF, same raw token, greedy):
    // [9707] -> 320 -> 16, i.e. "defined" -> " (" -> "1".
    // Operation zero, against a Python dequantization of token_embd row 9707:
    // l2=0.745936 absmax=0.046696 first=[0.002719, 0.018686, 0.022677, ...]
    let e = model.embed_probe(9707, stream).unwrap();
    let l2: f32 = e.iter().map(|v| v * v).sum::<f32>().sqrt();
    let absmax = e.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "EMBED n={} l2={l2:.6} absmax={absmax:.6} first8={:?}",
        e.len(),
        &e[..8.min(e.len())]
    );

    let logits = model.prefill_tokens(&[9707], stream).unwrap();
    let first = model.argmax_host(logits, stream).unwrap();
    eprintln!("first argmax token id = {first}");
    eprintln!(
        "first top-5 = {:?}",
        model.top_k_host(logits, 5, stream).unwrap()
    );
    assert!(first < 154_880);

    // Feed the REFERENCE token rather than Atlas's own argmax, so the second
    // step is comparable to llama.cpp even when the first step disagrees.
    let logits = model.decode_token(320, stream).unwrap();
    let second = model.argmax_host(logits, stream).unwrap();
    eprintln!("second argmax token id = {second}  (after feeding reference 320)");
    eprintln!(
        "second top-5 = {:?}",
        model.top_k_host(logits, 5, stream).unwrap()
    );
    assert!(second < 154_880);
    eprintln!("PARITY: atlas=[{first}, {second}] reference=[320, 16]");

    model.free().unwrap();
}

/// Coherence check: a real multi-token prompt, greedy, 24 tokens.
///
/// Single-token parity against llama.cpp is a harsh and ambiguous test, and the
/// remaining gap is progressive numerical drift (~1-4% per op on a 2-bit
/// checkpoint) rather than a broken kernel. What actually matters for the
/// engine is whether the model produces coherent text. Prints raw ids; decode
/// them with the checkpoint tokenizer.
#[test]
#[ignore = "requires the real GLM-5.3-Flash checkpoint on GB10"]
fn generates_real_text_on_real_hardware() {
    use spark_runtime::cuda_backend::AtlasCudaBackend;
    use spark_runtime::weights::gguf::{Glm53QuantProfile, load_glm53_store, open_glm53_files};

    assert!(
        Glm53Model::bringup_escape_set(),
        "set {GLM53_BRINGUP_ENV}=1 to run the unvalidated bring-up path"
    );

    let profile = Glm53QuantProfile::UdIq2Xxs;
    let dir = std::path::Path::new("/home/flocka/models/GLM-5.3-Flash-GGUF")
        .join(profile.directory_name());
    let paths: Vec<_> = profile
        .canonical_file_names()
        .iter()
        .map(|name| dir.join(name))
        .collect();

    let config_json =
        std::fs::read_to_string("/home/flocka/models/GLM-5.3-Flash-meta/config.json").unwrap();
    let mut config = atlas_core::config::parse_config(&config_json).unwrap();
    assert_eq!(config.model_type, "glm5_next");
    // `parse_config` leaves the parallelism world sizes at 0; `spark serve`
    // fills them in from the launch topology. This is single-GPU EP=1/TP=1, and
    // the GLM factory's exact-geometry gate checks both, so a harness that
    // skips this is refused for "unsupported geometry" with nothing about the
    // model actually wrong.
    config.tp_world_size = 1;
    config.ep_world_size = 1;

    let ptx = atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size)
        .expect("no GLM kernel target compiled; rebuild with ATLAS_TARGET_MODEL=glm5.3-flash");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();

    // Identity, metadata and the full 1,412-tensor schema, then the H2D load.
    let mut files = open_glm53_files(profile, &paths).unwrap();
    let reserve =
        usize::try_from(Glm53Model::runtime_allocation_bytes_for(262_144).unwrap()).unwrap();
    let store = load_glm53_store(&mut files, &gpu, reserve)
        .unwrap_or_else(|error| panic!("GLM store load failed: {error}"));
    eprintln!(
        "loaded {} tensors, {} bytes resident",
        store.len(),
        store.total_bytes()
    );

    let weights = crate::factory::Glm53TargetRuntimeWeights::new(profile, &config, store)
        .unwrap_or_else(|error| panic!("GLM runtime weights refused: {error}"));
    // 256K rather than the full 1M: the arena at 1M is 95 MiB larger than this
    // checkpoint leaves free. 256K leaves roughly 21 GiB of headroom, which
    // matters because over-allocating unified memory on GB10 takes the host
    // down rather than returning an error.
    // 64K for the parity probe: it only exercises positions 0 and 1, and the
    // arena scales with capacity. Shrinking it keeps the run well clear of the
    // host-OOM cliff when other processes hold memory. The DSA pool plan uses
    // the architectural 1M constant, not this, so geometry is unaffected.
    const CONTEXT: u32 = 65_536;
    let model = Glm53Model::new(std::sync::Arc::new(gpu), weights, CONTEXT).unwrap();

    let stream = 0u64;

    // "The capital of France is"
    let prompt: [u32; 5] = [785, 6722, 315, 9621, 374];
    let logits = model.prefill_tokens(&prompt, stream).unwrap();
    let mut next = model.argmax_host(logits, stream).unwrap();
    let mut out: Vec<u32> = vec![next];
    for _ in 0..23 {
        let logits = model.decode_token(next, stream).unwrap();
        next = model.argmax_host(logits, stream).unwrap();
        out.push(next);
    }
    eprintln!("PROMPT_IDS = {prompt:?}");
    eprintln!("GENERATED_IDS = {out:?}");
    assert!(out.iter().all(|t| *t < 154_880));
}

/// The reset's two spans must cover every carried byte, with `mhc_expanded_f32`
/// the ONLY thing they skip.
///
/// `reset_sequence` derives its spans from the plan rather than enumerating
/// regions, because an enumerated reset that misses one region reproduces the
/// cross-request contamination in a subtler form: the second request would look
/// correct for a while and then inherit whatever was left. This pins the
/// property that derivation depends on.
///
/// The excluded region is excluded for a reason that is not "it is later in the
/// arena": `mhc_expanded_f32` is a load-time precomputation from the weights,
/// not sequence state, and zeroing it would silently destroy the mHC
/// combination matrices for every subsequent request.
#[test]
fn the_sequence_reset_spans_cover_everything_except_the_load_time_mhc_block() {
    for positions in [4_096u32, 65_536, 262_144] {
        let plan = Glm53ArenaPlan::for_context(positions).unwrap();
        let carried_end = plan.context.total_bytes;
        let mhc = plan.mhc_expanded_f32;
        let after = plan.t1_transaction.offset_bytes;

        // The precondition reset_sequence enforces.
        assert!(carried_end <= mhc.offset_bytes, "{positions}");
        assert_eq!(
            mhc.offset_bytes + mhc.allocation_bytes,
            after,
            "{positions}"
        );
        assert!(after < plan.known_bytes, "{positions}");

        // Coverage: the ONLY unzeroed bytes are mhc's and the alignment padding
        // ahead of it.
        let zeroed = carried_end + (plan.known_bytes - after);
        let skipped = plan.known_bytes - zeroed;
        assert_eq!(
            skipped,
            mhc.offset_bytes - carried_end + mhc.allocation_bytes
        );
        assert!(
            skipped < mhc.allocation_bytes + 256,
            "{positions}: {skipped} bytes skipped for a {} byte mhc block, so \
             something other than mhc and its alignment padding is being kept",
            mhc.allocation_bytes
        );

        // Every carried region the walk binds lives inside the first span.
        assert!(
            plan.context.dsa_latent.offset_bytes < carried_end,
            "{positions}"
        );
        assert!(
            plan.context.kda_conv_f32.offset_bytes < carried_end,
            "{positions}"
        );
    }
}

/// The sequence-boundary guard FIRES, and names the fix.
///
/// This is the guard for the cross-request contamination found on a live
/// server: three unrelated prompts in one process, where request 2 trailed into
/// "...famous historical figures like Napoleon" and request 3 answered "391"
/// and then continued request 1's sentence about French cuisine. Nothing reset
/// `position`, so it advanced for the life of the process and every carried
/// region went with it.
#[test]
fn the_sequence_boundary_guard_fires_on_an_unreset_model() {
    // A model at its construction state, or freshly reset, may start.
    assert!(sequence_boundary_is_clean(0).is_ok());

    // One walk is enough: after any walk, position 0 means a missed reset.
    for walks in [1u64, 5, 28, u64::MAX] {
        let error = sequence_boundary_is_clean(walks).unwrap_err().to_string();
        assert!(error.contains("sequence reset"), "{error}");
        assert!(error.contains("reset_sequence"), "{walks}: {error}");
        assert!(error.contains(&walks.to_string()), "{walks}: {error}");
    }
}

/// NO TWO DISTINCT PER-LAYER QUANTITIES MAY SHARE A BYTE.
///
/// The T1 metadata regions are indexed by hand at a dozen call sites with
/// expressions like `(KDA_ORDINALS + ordinal) * 4`, and `buffer()` bounds-checks
/// nothing but address overflow. That is the same shape as the four `/KPOOL`
/// sites meaning three different things: an arithmetic slip aliases two live
/// quantities onto one word and produces fluent, wrong output.
///
/// This walks the ACTUAL bindings rather than restating the offsets, so it
/// catches a slip at the site that made it.
///
/// It also guards a real hazard. In the `published_ends_u32` region (45 slots)
/// the KDA conv layers take 0..33, the DSA `query_positions_u32` takes 34..44,
/// and the DSA `published_ends_u32` takes 45..55 -- past the payload, inside the
/// region's 256-byte alignment padding. Nothing is corrupted today, but
/// "correcting" the DSA published-ends index to `(KDA_ORDINALS + ordinal)`
/// would land it exactly on `query_positions_u32`, which the walk writes with
/// `position` every token while latent-append writes `position + 1`. That is a
/// live off-by-one on all eleven DSA layers, and this test is what refuses it.
#[test]
fn no_two_per_layer_state_bindings_overlap() {
    const BASE: DevicePtr = DevicePtr(0x2000_0000_0000);
    let plan = Glm53ArenaPlan::for_context(65_536).unwrap();
    let (kda_states, kda_conv, dsa_cache) = Glm53Model::bind_state(&plan, BASE).unwrap();

    let mut spans: Vec<(String, u64, u64)> = Vec::new();
    for (ordinal, conv) in kda_conv.iter().enumerate() {
        for (name, b) in [
            ("published_ends", conv.published_ends_u32),
            ("published_nonces", conv.published_nonces_u64),
            ("logical_lengths", conv.logical_lengths_u32),
            ("persistent", conv.persistent_state_f32),
            ("staged", conv.staged_state_f32),
        ] {
            spans.push((
                format!("kda{ordinal}.{name}"),
                b.ptr.0,
                b.ptr.0 + b.bytes as u64,
            ));
        }
    }
    for (ordinal, dsa) in dsa_cache.iter().enumerate() {
        for (name, b) in [
            ("sequence_lengths", dsa.sequence_lengths_u32),
            ("query_positions", dsa.query_positions_u32),
            ("published_ends", dsa.published_ends_u32),
            ("published_nonces", dsa.published_nonces_u64),
            ("query_validity", dsa.query_validity_u8),
            ("out_tail_validity", dsa.out_tail_validity_u8),
            ("prior_tail_validity", dsa.prior_tail_validity_u8),
        ] {
            spans.push((
                format!("dsa{ordinal}.{name}"),
                b.ptr.0,
                b.ptr.0 + b.bytes as u64,
            ));
        }
    }
    assert!(!kda_states.is_empty());

    spans.sort_by_key(|(_, start, _)| *start);
    for pair in spans.windows(2) {
        let (left_name, left_start, left_end) = &pair[0];
        let (right_name, right_start, _) = &pair[1];
        assert!(
            left_end <= right_start,
            "GLM per-layer state bindings OVERLAP: {left_name} is [{left_start:#x}, \
             {left_end:#x}) and {right_name} starts at {right_start:#x}. Two live \
             quantities share a word; one of them is being silently overwritten \
             every token."
        );
    }
}

/// A SECOND CONCURRENT SEQUENCE IS REFUSED, and the slot is reusable after
/// release.
///
/// `reset_sequence` makes SEQUENTIAL requests independent. It does nothing for
/// CONCURRENT ones: the arena is planned for batch 1 and there is a single
/// position counter behind one mutex, so two live sequences would interleave
/// into the same state with no error at all. Both halves are needed -- the
/// reset alone leaves concurrency corrupting silently, the refusal alone leaves
/// the cross-request contamination unfixed.
///
/// This pins the counter without a device; `claim_sequence` touches no GPU.
#[test]
fn a_second_concurrent_sequence_is_refused_and_the_slot_is_reusable() {
    let live = std::sync::atomic::AtomicUsize::new(0);
    let held = || live.load(std::sync::atomic::Ordering::Acquire);

    assert!(
        claim_only_sequence_slot(&live).is_ok(),
        "the first sequence must be admitted"
    );
    let refusal = claim_only_sequence_slot(&live).unwrap_err().to_string();
    assert!(refusal.contains("ONE sequence at a time"), "{refusal}");
    assert!(refusal.contains("batch 1"), "{refusal}");
    // The failed claim must not leak its speculative increment, or one refusal
    // poisons the slot for the life of the process.
    assert_eq!(held(), 1);

    release_only_sequence_slot(&live);
    assert_eq!(held(), 0);
    assert!(
        claim_only_sequence_slot(&live).is_ok(),
        "the slot must be reusable"
    );

    release_only_sequence_slot(&live);
    // Release at zero SATURATES rather than wrapping to usize::MAX, which would
    // refuse every subsequent request forever.
    release_only_sequence_slot(&live);
    assert_eq!(held(), 0);
    assert!(claim_only_sequence_slot(&live).is_ok());
}
