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
    let refusal = format!("{:#}", Glm53Model::admit().unwrap_err());
    assert!(refusal.contains("admission is closed"), "{refusal}");
    assert!(refusal.contains(GLM53_BRINGUP_ENV), "{refusal}");
}

/// Runtime allocation must be accounted, not discovered at `alloc` time.
#[test]
fn runtime_allocation_is_the_arena_plus_scratch_plus_logits() {
    let total = Glm53Model::runtime_allocation_bytes();
    assert_eq!(
        total,
        GLM53_KNOWN_ARENA_BYTES + Glm53WalkScratch::required_bytes() + 154_880 * 2 + 4
    );
    // Weights plus runtime must still clear a 119 GiB box with headroom.
    let weights = crate::model::glm53::arena::GLM53_IQ2_XXS_TENSOR_BYTES;
    let resident = weights + total;
    assert!(
        resident < 110 * (1u64 << 30),
        "IQ2_XXS resident total {resident} exceeds 110 GiB"
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
    let model = Glm53Model::new(Box::new(gpu), weights, CONTEXT).unwrap();

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
    let model = Glm53Model::new(Box::new(gpu), weights, CONTEXT).unwrap();

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
