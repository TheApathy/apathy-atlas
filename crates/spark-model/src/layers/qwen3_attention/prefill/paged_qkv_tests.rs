// SPDX-License-Identifier: AGPL-3.0-only

use super::paged_qkv::use_prefill_kv_dual;

#[test]
fn dual_kv_is_large_original_layout_only() {
    assert_eq!(
        use_prefill_kv_dual(true, true, 8192, 5120, true, true),
        Ok(true)
    );
    assert_eq!(
        use_prefill_kv_dual(false, true, 8192, 5120, true, true),
        Ok(false)
    );
    assert_eq!(
        use_prefill_kv_dual(true, true, 32, 5120, true, true),
        Ok(false)
    );
    assert_eq!(
        use_prefill_kv_dual(true, true, 33, 5120, true, true),
        Ok(true)
    );
    assert_eq!(
        use_prefill_kv_dual(true, true, 8192, 5119, true, true),
        Ok(false)
    );
    assert_eq!(
        use_prefill_kv_dual(true, true, 8192, 5120, true, true),
        Ok(true)
    );
    assert!(use_prefill_kv_dual(true, true, 8192, 5120, false, true).is_err());
    assert!(use_prefill_kv_dual(true, true, 8192, 5120, true, false).is_err());
}

#[test]
fn requested_eligible_dual_kv_requires_the_kernel_symbol() {
    assert!(use_prefill_kv_dual(true, false, 8192, 5120, true, true).is_err());
    assert_eq!(
        use_prefill_kv_dual(false, false, 8192, 5120, true, true),
        Ok(false)
    );
}

#[test]
fn requested_large_dual_kv_rejects_conflicting_layouts_and_has_path_proof() {
    let paged = include_str!("paged_qkv.rs");
    let cache_skip = include_str!("cache_skip_qkv.rs");
    let module = include_str!("mod.rs");
    assert!(paged.contains("ATLAS_PREFILL_KV_DUAL=1 requires ATLAS_PREFILL_PROJ_FAST=0"));
    assert!(paged.contains("requires compatible original-layout NVFP4 K/V weights"));
    assert!(module.contains("ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_cache_skip"));
    assert!(module.contains("ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_paged"));
    assert_eq!(
        module
            .matches("ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_cache_skip")
            .count(),
        1
    );
    assert_eq!(
        module
            .matches("ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_paged")
            .count(),
        1
    );

    for source in [paged, cache_skip] {
        for excluded in [
            "self.k_fp8w_t.is_none()",
            "self.v_fp8w_t.is_none()",
            "self.k_fp8.is_none()",
            "self.v_fp8.is_none()",
        ] {
            assert!(source.contains(excluded));
        }
        assert!(
            source.contains("let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);")
        );
        let selector = source.find("let dual_kv =").unwrap();
        let q_projection = source[selector..]
            .find("SkipProj::Q")
            .or_else(|| source[selector..].find("Proj::Q"))
            .unwrap()
            + selector;
        let dual_launch = source.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let marker = source.find("mark_prefill_kv_dual_").unwrap();
        assert!(selector < q_projection, "dual selection must fail before Q");
        assert!(
            dual_launch < marker,
            "route proof must follow a successful launch"
        );
        assert!(
            source[dual_launch..].contains("false,"),
            "K/V dual mode must not fuse SiLU"
        );

        let ordered_abi = [
            "normed,",
            "k_nvfp4.unwrap(),",
            "v_nvfp4.unwrap(),",
            "k_contiguous,",
            "v_contiguous,",
            "false,",
            "n,",
            "nkv * hd,",
            "h,",
            "stream,",
        ];
        let mut cursor = dual_launch;
        for argument in ordered_abi {
            let relative = source[cursor..]
                .find(argument)
                .unwrap_or_else(|| panic!("missing dual-K/V ABI argument {argument}"));
            cursor += relative + argument.len();
        }
    }

    assert!(cache_skip.contains("self.cache_skip_one_proj(\n            SkipProj::K"));
    assert!(cache_skip.contains("self.cache_skip_one_proj(\n            SkipProj::V"));
    assert!(paged.contains("self.prefill_one_proj(Proj::K"));
    assert!(paged.contains("self.prefill_one_proj(Proj::V"));
}

#[test]
fn projection_pipe_call_sites_are_fail_closed_and_attributed() {
    let attention_qkv = include_str!("paged_qkv.rs");
    let attention_o = include_str!("paged_oproj.rs");
    let ssm_qkvz = include_str!("../../qwen3_ssm/trait_prefill_phase1.rs");
    let ssm_out = include_str!("../../qwen3_ssm/trait_prefill_phase3.rs");

    for source in [attention_qkv, attention_o, ssm_qkvz, ssm_out] {
        assert!(source.contains("prefill_projection_pipe_route("));
        assert!(source.contains("PrefillProjectionPipeRoute::Missing"));
        assert!(source.contains("ATLAS_PREFILL_PROJ_PIPE=1 requires w4a16_gemm_pipe"));
    }
    assert!(attention_qkv.contains("ENGAGED ATLAS_PREFILL_PROJ_PIPE: {name}"));
    for name in ["attention_q", "attention_k", "attention_v"] {
        assert!(attention_qkv.contains(name));
    }
    assert!(attention_o.contains("ENGAGED ATLAS_PREFILL_PROJ_PIPE: attention_o"));
    assert!(ssm_qkvz.contains("ENGAGED ATLAS_PREFILL_PROJ_PIPE: ssm_qkvz"));
    assert!(ssm_out.contains("ENGAGED ATLAS_PREFILL_PROJ_PIPE: ssm_out"));
}
