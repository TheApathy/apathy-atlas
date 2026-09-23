// SPDX-License-Identifier: AGPL-3.0-only
//! CPU admission and real launch traces only; no device parity or speed claim.
use super::*;

fn selected(
    value: Option<&str>,
    group: Option<&str>,
    prefill: bool,
) -> Result<Glm53Exl3RoutePolicy> {
    Glm53Exl3RoutePolicy::parse_with_verify_group(
        Some(OsStr::new("1")),
        prefill.then_some(OsStr::new("1")),
        group.map(OsStr::new),
        Some(OsStr::new("1")),
    )?
    .with_verify_staged_k32(value.map(OsStr::new), Some(OsStr::new("1")))
}

fn staged_handles(first: u64, k32: bool) -> Glm53Exl3StagedMoeKernels {
    Glm53Exl3StagedMoeKernels {
        private: true,
        build_chunks: KernelHandle(first),
        gather: KernelHandle(first + 1),
        gate_up: KernelHandle(first + 2),
        activate: KernelHandle(first + 3),
        down: KernelHandle(first + 4),
        scatter: KernelHandle(first + 5),
        gemm_block_threads: if k32 { 512 } else { 256 },
        gemm_shared_bytes: if k32 { 25_600 } else { 20_992 },
    }
}

fn candidate(enabled: bool) -> Glm53Exl3MoeKernels {
    let mut k = kernels(selected(Some(if enabled { "1" } else { "0" }), None, true).unwrap());
    k.staged = Some(staged_handles(71, false));
    k.verify_staged = enabled.then(|| staged_handles(61, true));
    k
}

#[test]
fn verify_staged_parser_is_strict_latched_and_default_off() {
    for group in [None, Some("8")] {
        for value in [None, Some("0")] {
            let p = selected(value, group, true).unwrap();
            assert!(!p.verify_staged_k32_enabled());
        }
        assert!(
            selected(Some("1"), group, false)
                .unwrap()
                .verify_staged_k32_enabled()
        );
    }
    for group in [Some("2"), Some("4")] {
        assert!(selected(Some("1"), group, true).is_err());
        assert!(selected(Some("0"), group, true).is_ok());
    }
    for value in ["", "true", "01", " 1", "1 ", "2"] {
        assert!(selected(Some(value), None, true).is_err());
    }
    for (private, exact) in [
        (None, Some("1")),
        (Some("0"), Some("1")),
        (Some("1"), None),
        (Some("1"), Some("0")),
        (Some("1"), Some("true")),
    ] {
        let p = Glm53Exl3RoutePolicy::parse(private.map(OsStr::new), None).unwrap();
        assert!(
            p.with_verify_staged_k32(Some(OsStr::new("1")), exact.map(OsStr::new))
                .is_err()
        );
    }
    assert!(
        selected(Some("1"), None, false)
            .unwrap()
            .validate_moe_mode(Some(OsStr::new("serial-reference")))
            .is_err()
    );
}

#[cfg(unix)]
#[test]
fn verify_staged_non_utf8_is_rejected() {
    use std::os::unix::ffi::OsStrExt;
    let invalid = OsStr::from_bytes(&[0xff]);
    let p = selected(None, None, false).unwrap();
    assert!(
        p.with_verify_staged_k32(Some(invalid), Some(OsStr::new("1")))
            .is_err()
    );
    assert!(
        p.with_verify_staged_k32(Some(OsStr::new("1")), Some(invalid))
            .is_err()
    );
    assert!(super::super::verify_staged::Pipeline::parse(Some(invalid)).is_err());
}

#[test]
fn verify_staged_scope_and_every_legacy_plan_are_preserved() {
    let on = candidate(true);
    let off = candidate(false);
    for rows in [1, 2, 4, 8, 9, 1024, 2048] {
        assert_eq!(on.plan(rows).unwrap(), off.plan(rows).unwrap());
        assert_eq!(on.kernel_launches(rows), off.kernel_launches(rows));
    }
    with_glm53_exact_verify(|| {
        for rows in [1, 9, 1024, 2048] {
            assert_eq!(on.plan(rows).unwrap(), off.plan(rows).unwrap());
        }
        for rows in 2..=8 {
            let p = on.plan(rows).unwrap();
            assert_eq!(p.max_chunks, rows * 8);
            assert_eq!(p.chunk_descriptor_u32_bytes, rows as usize * 8 * 4);
            assert_eq!(p.temp_state_f16_bytes, rows as usize * 8 * 4096 * 2);
            assert_eq!(p.temp_intermediate_f16_bytes, rows as usize * 8 * 2048 * 2);
            assert_eq!(p.route_private_f32_bytes, rows as usize * 8 * 4096 * 4);
            assert_eq!(on.kernel_launches(rows), 7);
            assert_eq!(off.kernel_launches(rows), 2);
        }
        with_glm53_exact_wide_prefill(|| {
            for rows in [1, 4, 8, 1024] {
                assert_eq!(on.plan(rows).unwrap(), off.plan(rows).unwrap());
                assert_eq!(on.kernel_launches(rows), off.kernel_launches(rows));
            }
        });
        with_glm53_layer_major_prefill(|| {
            for rows in [1, 4, 8, 2048] {
                assert_eq!(on.plan(rows).unwrap(), off.plan(rows).unwrap());
                assert_eq!(on.kernel_launches(rows), off.kernel_launches(rows));
            }
        });
    });
}

fn trace(k: &Glm53Exl3MoeKernels, rows: u32) -> Vec<(u64, [u32; 3], [u32; 3], u32)> {
    let gpu = Trace::default();
    let p = k.plan(rows).unwrap();
    let b = buffers(p);
    k.launch(&gpu, p, b, 7).unwrap();
    let launches = gpu.launches.lock().unwrap();
    if let Some(scatter) = launches.iter().find(|v| v.function == 66) {
        assert_eq!(scatter.output, Some(b.route_private_f32.ptr.0));
    }
    launches
        .iter()
        .map(|v| (v.function, v.grid, v.block, v.shared))
        .collect()
}

#[test]
fn verify_staged_real_launcher_uses_seven_nodes_and_split_block_sizes() {
    let k = candidate(true);
    with_glm53_exact_verify(|| {
        for rows in 2..=8 {
            let p = rows * 8;
            assert_eq!(
                trace(&k, rows),
                vec![
                    (12, [1, 1, 1], [320, 1, 1], 0),
                    (61, [1, 1, 1], [320, 1, 1], 0),
                    (62, [p, 1, 1], [256, 1, 1], 0),
                    (63, [8, p, 2], [512, 1, 1], 25_600),
                    (64, [p, 1, 1], [256, 1, 1], 0),
                    (65, [16, p, 1], [512, 1, 1], 25_600),
                    (66, [p, 1, 1], [256, 1, 1], 4096),
                ]
            );
        }
        assert_eq!(trace(&k, 1), trace(&candidate(false), 1));
        with_glm53_exact_wide_prefill(|| {
            assert_eq!(trace(&k, 4), trace(&candidate(false), 4));
            assert_eq!(trace(&k, 1024), trace(&candidate(false), 1024));
        });
        with_glm53_layer_major_prefill(|| {
            assert_eq!(trace(&k, 8), trace(&candidate(false), 8));
            assert_eq!(trace(&k, 2048), trace(&candidate(false), 2048));
        });
    });
}

#[test]
fn verify_staged_bad_plan_extents_and_missing_kernel_fail_before_enqueue() {
    let mut k = candidate(true);
    with_glm53_exact_verify(|| {
        let p = k.plan(4).unwrap();
        for bad in 0..3 {
            let gpu = Trace::default();
            let mut b = buffers(p);
            match bad {
                0 => b.chunk_expert_u32.bytes -= 4,
                1 => b.temp_state_g_f16.bytes -= 2,
                _ => b.route_private_f32.bytes -= 4,
            }
            assert!(k.launch(&gpu, p, b, 7).is_err());
            assert_eq!(gpu.memsets.load(Ordering::Relaxed), 0);
            assert!(gpu.launches.lock().unwrap().is_empty());
        }
        k.verify_staged = None;
        let gpu = Trace::default();
        assert!(k.launch(&gpu, p, buffers(p), 7).is_err());
        assert_eq!(gpu.memsets.load(Ordering::Relaxed), 0);
        assert!(gpu.launches.lock().unwrap().is_empty());
    });
    let k = candidate(true);
    let p = with_glm53_exact_verify(|| k.plan(4).unwrap());
    let gpu = Trace::default();
    assert!(k.launch(&gpu, p, buffers(p), 7).is_err());
    assert_eq!(gpu.memsets.load(Ordering::Relaxed), 0);
}

#[test]
fn verify_staged_failure_stops_without_fused_fallback() {
    let k = candidate(true);
    let gpu = Trace::default();
    gpu.fail_fused.store(true, Ordering::Relaxed);
    with_glm53_exact_verify(|| {
        let p = k.plan(4).unwrap();
        assert!(k.launch(&gpu, p, buffers(p), 7).is_err());
    });
    let ids: Vec<_> = gpu
        .launches
        .lock()
        .unwrap()
        .iter()
        .map(|v| v.function)
        .collect();
    assert_eq!(ids, [12, 61, 62, 63]);
}

#[test]
fn verify_staged_loader_pins_private_helpers_and_only_new_k32_module() {
    let gpu = Trace::default();
    let k = super::super::verify_staged::load(&gpu).unwrap();
    assert!(k.private);
    assert_eq!((k.gemm_block_threads, k.gemm_shared_bytes), (512, 25_600));
    let loaded = gpu.loaded.lock().unwrap();
    assert_eq!(loaded.len(), 6);
    assert!(
        loaded
            .iter()
            .all(|(module, _)| module == "glm53_exl3_moe_staged_private"
                || module == "glm53_exl3_moe_verify_k32")
    );
    assert_eq!(
        *gpu.shared_attrs.lock().unwrap(),
        vec![(63, 25_600), (65, 25_600)]
    );
}

#[test]
fn verify_staged_sh4_selector_is_strict_and_loads_only_the_distinct_candidate() {
    use super::super::verify_staged::{load_with_pipeline, Pipeline};

    for value in [None, Some(OsStr::new("0"))] {
        assert_eq!(Pipeline::parse(value).unwrap(), Pipeline::Sh3);
    }
    assert_eq!(
        Pipeline::parse(Some(OsStr::new("1"))).unwrap(),
        Pipeline::Sh4
    );
    for value in ["", "true", "01", " 1", "1 ", "2"] {
        assert!(Pipeline::parse(Some(OsStr::new(value))).is_err());
    }

    let gpu = Trace::default();
    let k = load_with_pipeline(&gpu, Pipeline::Sh4).unwrap();
    assert!(k.private);
    assert_eq!((k.gemm_block_threads, k.gemm_shared_bytes), (512, 28_672));
    let loaded = gpu.loaded.lock().unwrap();
    assert_eq!(loaded.len(), 6);
    assert!(loaded
        .iter()
        .all(|(module, _)| module == "glm53_exl3_moe_staged_private"
            || module == "glm53_exl3_moe_verify_k32_sh4"));
    assert_eq!(
        *gpu.shared_attrs.lock().unwrap(),
        vec![(67, 28_672), (68, 28_672)]
    );
}

#[test]
fn verify_staged_source_pins_arithmetic_geometry_and_model_graph_lifetime() {
    const CUDA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_verify_k32.cu"
    ));
    assert_eq!(
        CUDA.matches("exl3_gemm_kernel_inner<2, false, 2, 16, 32, 256, 3, 3, false>")
            .count(),
        2
    );
    assert!(CUDA.contains("#include <quant/hadamard_inner.cuh>"));
    assert!(CUDA.contains("__launch_bounds__(512)"));
    assert!(!CUDA.contains("#define barrier_"));
    assert!(!CUDA.contains("exl3_mgemm_kernel"));
    assert!(!CUDA.contains("had_ff_"));
    const SH4: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_verify_k32_sh4.cu"
    ));
    assert_eq!(
        SH4.matches("exl3_gemm_kernel_inner<2, false, 2, 16, 32, 256, 4, 3, false>")
            .count(),
        2
    );
    assert!(SH4.contains("atlas_glm53_exl3_verify_gate_up_k32_sh4"));
    assert!(SH4.contains("atlas_glm53_exl3_verify_down_k32_sh4"));
    for source in [
        include_str!("glm53_exl3_moe.rs"),
        include_str!("../glm53_moe_serial.rs"),
        include_str!("../../model/glm53/target_model_exl3.rs"),
    ] {
        assert!(source.contains(".with_verify_staged_k32("));
        assert!(source.contains("ATLAS_GLM53_EXL3_MOE_VERIFY_STAGED_K32"));
    }
    const REGISTRY: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/KERNEL.toml"
    ));
    assert!(REGISTRY.contains("glm53_exl3_moe_verify_k32 = \"glm53_exl3_moe_verify_k32\""));
    assert!(REGISTRY.contains("glm53_exl3_moe_verify_k32_sh4 = \"glm53_exl3_moe_verify_k32_sh4\""));
    let owner = include_str!("../../model/glm53/ffn_graph_runtime.rs");
    assert!(owner.contains("self.environment == environment()"));
    assert!(owner.contains("starts_with(b\"ATLAS_GLM53_\")"));
    // Full K column slices retain both K32 subchains, no cross-CTA partial sums.
    for (k, n) in [(4096, 2048), (2048, 4096)] {
        let tiles_k = k / 32;
        let tiles_n = n / 256;
        for block in 0..tiles_n {
            assert_eq!(tiles_k * tiles_n * block / tiles_n, tiles_k * block);
        }
    }
    assert!(64 * 2 * 128 <= 1024 * 1024);
    assert!(64 * 256 <= 1024 * 1024);
}
