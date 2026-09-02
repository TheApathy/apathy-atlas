// SPDX-License-Identifier: AGPL-3.0-only
//! Print the parsed GLM-5.3 geometry alongside what the factory demands, so a
//! refusal names the field instead of the whole struct.
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <config.json>");
    let c = atlas_core::config::parse_config(&std::fs::read_to_string(path)?)?;
    macro_rules! chk {
        ($name:expr, $got:expr, $want:expr) => {
            let got = $got;
            let want = $want;
            println!(
                "{} {:<32} got={:?} want={:?}",
                if got == want { "ok  " } else { "DIFF" },
                $name,
                got,
                want
            );
        };
    }
    chk!("model_type", c.model_type.clone(), "glm5_next".to_string());
    chk!("hidden_size", c.hidden_size, 4096);
    chk!("num_hidden_layers", c.num_hidden_layers, 45);
    chk!("intermediate_size", c.intermediate_size, 12288);
    chk!("vocab_size", c.vocab_size, 154880);
    chk!("num_attention_heads", c.num_attention_heads, 64);
    chk!("num_key_value_heads", c.num_key_value_heads, 64);
    chk!("head_dim", c.head_dim, 256);
    chk!("q_lora_rank", c.q_lora_rank, 1536);
    chk!("kv_lora_rank", c.kv_lora_rank, 512);
    chk!("qk_nope_head_dim", c.qk_nope_head_dim, 256);
    chk!("qk_rope_head_dim", c.qk_rope_head_dim, 0);
    chk!("v_head_dim", c.v_head_dim, 256);
    chk!("linear_num_key_heads", c.linear_num_key_heads, 64);
    chk!("linear_num_value_heads", c.linear_num_value_heads, 64);
    chk!("linear_key_head_dim", c.linear_key_head_dim, 128);
    chk!("linear_value_head_dim", c.linear_value_head_dim, 128);
    chk!("linear_conv_kernel_dim", c.linear_conv_kernel_dim, 4);
    chk!("num_experts", c.num_experts, 288);
    chk!("num_experts_per_tok", c.num_experts_per_tok, 8);
    chk!("moe_intermediate_size", c.moe_intermediate_size, 2048);
    chk!(
        "shared_expert_intermediate_size",
        c.shared_expert_intermediate_size,
        2048
    );
    chk!("hc_mult", c.hc_mult, 4);
    chk!("hc_sinkhorn_iters", c.hc_sinkhorn_iters, 20);
    chk!(
        "max_position_embeddings",
        c.max_position_embeddings,
        1_048_576
    );
    chk!(
        "scoring_func",
        c.scoring_func.clone(),
        "sigmoid".to_string()
    );
    chk!("norm_topk_prob", c.norm_topk_prob, true);
    chk!("use_routing_bias", c.use_routing_bias, true);
    chk!("num_mtp_modules", c.num_mtp_modules, 1);
    chk!("mtp_transformer_layers", c.mtp_transformer_layers, 1);
    chk!("mtp_num_hidden_layers", c.mtp_num_hidden_layers, 1);
    chk!("bos_token_id", c.bos_token_id, 0);
    chk!("eos_token_id", c.eos_token_id, 154820);
    chk!(
        "tie_word_embeddings(want false)",
        c.tie_word_embeddings,
        false
    );
    println!("--- tail of exact_core ---");
    chk!("partial_rotary_factor==0", c.partial_rotary_factor, 0.0);
    chk!("adapter_max_rank", c.adapter_max_rank, 0);
    chk!("nested_config(want true)", c.nested_config, true);
    chk!(
        "weight_prefix",
        c.weight_prefix.clone(),
        "model.language_model".to_string()
    );
    chk!("attn_gated(want false)", c.attn_gated, false);
    chk!("vision.is_none(want true)", c.vision.is_none(), true);
    chk!("tp_world_size", c.tp_world_size, 1);
    chk!("ep_world_size", c.ep_world_size, 1);
    chk!("layer_types.len", c.layer_types.len(), 45);
    if let Some(g) = c.glm5_next.as_ref() {
        chk!("mlp_layer_types.len", g.mlp_layer_types.len(), 45);
        chk!("indexer_types.len", g.indexer_types.len(), 45);
        chk!("index_head_dim", g.index_head_dim, 128);
        chk!("index_n_heads", g.index_n_heads, 32);
        chk!("index_topk", g.index_topk, 2048);
        chk!("index_kpool", g.index_kpool, 4);
        println!(
            "     eos_token_ids                    got={:?} want=[154820, 154827, 154829]",
            g.eos_token_ids
        );
        chk!("mla_use_nope", g.mla_use_nope, true);
        chk!(
            "index_kpool_always_select_tail",
            g.index_kpool_always_select_tail,
            true
        );
        chk!("index_kpool_compress", g.index_kpool_compress, true);
        chk!(
            "index_share_for_mtp_iteration",
            g.index_share_for_mtp_iteration,
            true
        );
        chk!("indexer_rope_interleave", g.indexer_rope_interleave, true);
    }
    println!("--- float fields ---");
    println!(
        "hc_eps={:e} (want 1e-6)  rms_norm_eps={:e} (want 1e-5)  routed_scaling_factor={} (want 2.5)",
        c.hc_eps, c.rms_norm_eps, c.routed_scaling_factor
    );
    println!("glm5_next block present: {}", c.glm5_next.is_some());
    Ok(())
}
