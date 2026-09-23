// SPDX-License-Identifier: AGPL-3.0-only

//! Actual host dispatch RED. No CUDA emulation, numerical or completion claim.
#[path = "glm53_exl3_private_kernel/checking.rs"]
mod checking;
#[path = "glm53_exl3_private_kernel/fixture.rs"]
mod fixture;
use checking::{check_combine, check_fused, check_pack, check_staged};
use fixture::{Fixture, TraceGpu};
use spark_model::layers::ops::{Glm53Exl3MoeKernels, Glm53Exl3MoePlan, Glm53Exl3RoutePolicy};
use spark_runtime::gpu::DevicePtr;
use std::ffi::OsStr;

const STREAM: u64 = 73;

fn policy(private: bool, prefill: bool) -> Glm53Exl3RoutePolicy {
    Glm53Exl3RoutePolicy::parse(
        Some(OsStr::new(if private { "1" } else { "0" })),
        Some(OsStr::new(if prefill { "1" } else { "0" })),
    )
    .unwrap()
}

fn load(private: bool, prefill: bool) -> (TraceGpu, Glm53Exl3MoeKernels) {
    // No process-global mutation: root runs this target with baseline schedule.
    for name in [
        "ATLAS_GLM53_EXL3_MOE_LARGE_GROUP_WIDTH",
        "ATLAS_GLM53_EXL3_MOE_STAGED_VARIANT",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "test requires unset {name}"
        );
    }
    for name in ["ATLAS_GLM53_EXL3_MOE_STAGED", "ATLAS_DEBUG_SYNC_KERNELS"] {
        let value = std::env::var_os(name);
        let allowed = if name.ends_with("STAGED") { "1" } else { "0" };
        assert!(
            value.is_none() || value.as_deref() == Some(OsStr::new(allowed)),
            "test requires baseline {name}"
        );
    }
    let gpu = TraceGpu::new();
    let kernels =
        Glm53Exl3MoeKernels::load_with_route_policy(&gpu, policy(private, prefill)).unwrap();
    assert!(gpu.effects().is_empty(), "loading cannot submit model work");
    (gpu, kernels)
}

fn run(gpu: &TraceGpu, kernels: &Glm53Exl3MoeKernels, rows: u32) -> (Glm53Exl3MoePlan, Fixture) {
    let plan = kernels.plan(rows).unwrap();
    let f = Fixture::new(gpu, plan);
    kernels.launch(gpu, plan, f.buffers, STREAM).unwrap();
    kernels
        .combine_shared(
            gpu,
            plan,
            f.buffers.output_f32,
            f.buffers.route_private_f32,
            f.shared,
            f.combined,
            f.buffers.route_status_u32,
            STREAM,
        )
        .unwrap();
    (plan, f)
}

#[test]
fn default_and_small_private_preserve_original_large_row_dispatch() {
    for small in [false, true] {
        for rows in [1, 8, 9, 106, 1024, 2038, 2048] {
            let (gpu, k) = load(small, false);
            let (p, f) = run(&gpu, &k, rows);
            let private = small && rows <= 8;
            let ls = gpu.launches();
            assert_eq!(
                p.route_private_f32_bytes,
                rows.min(8) as usize * 8 * 4096 * 4
            );
            assert_eq!(ls.len(), if rows >= 1024 { 8 } else { 3 });
            assert_eq!(k.kernel_launches(rows) as usize + 1, ls.len());
            check_pack(&ls[0], p, &f, private);
            if rows >= 1024 {
                check_staged(&ls[1..7], p, &f, false);
            } else {
                check_fused(&ls[1], p, &f, private);
            }
            check_combine(ls.last().unwrap(), p, &f, private);
            let expected = if private {
                vec![(f.buffers.route_status_u32.ptr.0, 0, 4, STREAM)]
            } else {
                vec![
                    (f.buffers.output_f32.ptr.0, 0, p.output_f32_bytes, STREAM),
                    (f.buffers.route_status_u32.ptr.0, 0, 4, STREAM),
                ]
            };
            assert_eq!(gpu.memsets(), expected);
        }
    }
}

#[test]
fn selected_prefill_uses_full_private_fused_or_staged_dispatch_and_combine() {
    for rows in [1, 8, 9, 106, 1024, 2038, 2048] {
        let (gpu, k) = load(true, true);
        let (p, f) = run(&gpu, &k, rows);
        let ls = gpu.launches();
        assert_eq!(p.route_private_f32_bytes, rows as usize * 8 * 4096 * 4);
        assert_eq!(ls.len(), if rows >= 1024 { 8 } else { 3 });
        assert_eq!(k.kernel_launches(rows) as usize + 1, ls.len());
        check_pack(&ls[0], p, &f, true);
        if rows >= 1024 {
            check_staged(&ls[1..7], p, &f, true);
        } else {
            check_fused(&ls[1], p, &f, true);
        }
        check_combine(ls.last().unwrap(), p, &f, true);
        assert!(
            gpu.memsets()
                .iter()
                .all(|(ptr, _, _, _)| *ptr != f.buffers.output_f32.ptr.0),
            "private route must not fall back to the atomically accumulated output"
        );
    }
}

#[test]
fn selected_plan_must_match_latched_policy_even_when_all_buffers_match_forged_plan() {
    for rows in [9, 106, 1024, 2038, 2048] {
        let (gpu, k) = load(true, true);
        let (_, legacy) = load(false, false);
        let wrong = legacy.plan(rows).unwrap();
        let f = Fixture::new(&gpu, wrong);
        assert!(k.launch(&gpu, wrong, f.buffers, STREAM).is_err());
        assert!(
            k.combine_shared(
                &gpu,
                wrong,
                f.buffers.output_f32,
                f.buffers.route_private_f32,
                f.shared,
                f.combined,
                f.buffers.route_status_u32,
                STREAM
            )
            .is_err()
        );
        assert!(
            gpu.effects().is_empty(),
            "mismatched policy must fail before memset/launch/copy"
        );
        let (gpu, legacy) = load(false, false);
        let (_, selected) = load(true, true);
        let wrong = selected.plan(rows).unwrap();
        let f = Fixture::new(&gpu, wrong);
        assert!(legacy.launch(&gpu, wrong, f.buffers, STREAM).is_err());
        assert!(
            legacy
                .combine_shared(
                    &gpu,
                    wrong,
                    f.buffers.output_f32,
                    f.buffers.route_private_f32,
                    f.shared,
                    f.combined,
                    f.buffers.route_status_u32,
                    STREAM
                )
                .is_err()
        );
        assert!(gpu.effects().is_empty());
    }
}

#[test]
fn invalid_geometry_and_derived_plan_fields_fail_before_any_gpu_effect() {
    let (gpu, k) = load(true, true);
    for rows in [0, 2049, u32::MAX] {
        assert!(k.plan(rows).is_err());
    }
    for mutation in 0..7 {
        let mut p = k.plan(106).unwrap();
        match mutation {
            0 => p.rows = 0,
            1 => p.rows = 2049,
            2 => p.pairs -= 1,
            3 => p.input_f16_bytes -= 2,
            4 => p.max_chunks += 1,
            5 => p.route_private_f32_bytes -= 4,
            _ => p.chunk_count_u32_bytes += 4,
        }
        let f = Fixture::new(&gpu, p);
        assert!(
            k.launch(&gpu, p, f.buffers, STREAM).is_err(),
            "mutation {mutation}"
        );
        assert!(
            k.combine_shared(
                &gpu,
                p,
                f.buffers.output_f32,
                f.buffers.route_private_f32,
                f.shared,
                f.combined,
                f.buffers.route_status_u32,
                STREAM
            )
            .is_err(),
            "mutation {mutation}"
        );
        assert!(gpu.effects().is_empty());
    }
}

#[test]
fn private_buffer_and_pointer_table_extent_failures_precede_status_memset() {
    for rows in [9, 1024, 2048] {
        for mutation in 0..7 {
            let (gpu, k) = load(true, true);
            let p = k.plan(rows).unwrap();
            let mut f = Fixture::new(&gpu, p);
            match mutation {
                0 => f.buffers.route_private_f32.bytes -= 4,
                1 => f.buffers.route_private_f32.ptr = DevicePtr::NULL,
                2 => f.buffers.route_private_f32.ptr = DevicePtr(u64::MAX - 1),
                3 => f.buffers.route_status_u32.bytes = 3,
                4 => f.buffers.pointers.down_svh.bytes -= 8,
                5 => f.buffers.chunk_count_u32.ptr = DevicePtr::NULL,
                _ => f.buffers.token_sorted_i64.bytes -= 8,
            }
            assert!(
                k.launch(&gpu, p, f.buffers, STREAM).is_err(),
                "rows={rows} mutation={mutation}"
            );
            assert!(gpu.effects().is_empty());
        }
    }
}

#[test]
fn combine_rejects_selected_private_and_destination_extent_mismatch_before_launch() {
    for mutation in 0..5 {
        let (gpu, k) = load(true, true);
        let p = k.plan(2038).unwrap();
        let mut f = Fixture::new(&gpu, p);
        match mutation {
            0 => f.buffers.route_private_f32.bytes = 1024 * 1024,
            1 => f.buffers.route_private_f32.ptr = DevicePtr::NULL,
            2 => f.shared.bytes -= 2,
            3 => f.combined.bytes -= 2,
            _ => f.buffers.route_status_u32.bytes = 0,
        }
        assert!(
            k.combine_shared(
                &gpu,
                p,
                f.buffers.output_f32,
                f.buffers.route_private_f32,
                f.shared,
                f.combined,
                f.buffers.route_status_u32,
                STREAM
            )
            .is_err()
        );
        assert!(gpu.effects().is_empty());
    }
}

#[test]
fn failed_private_submission_stops_the_real_pipeline_without_later_kernel_attempts() {
    let expected = [
        "atlas_glm53_exl3_pack_routes_private",
        "atlas_glm53_exl3_build_chunks_private",
        "atlas_glm53_exl3_staged_gather_private",
        "atlas_glm53_exl3_staged_gate_up_k16",
        "atlas_glm53_exl3_staged_activate_private",
        "atlas_glm53_exl3_staged_down_k16",
        "atlas_glm53_exl3_staged_scatter_private",
    ];
    for (index, name) in expected.iter().enumerate() {
        let (gpu, k) = load(true, true);
        let p = k.plan(1024).unwrap();
        let f = Fixture::new(&gpu, p);
        gpu.fail_on(name);
        assert!(k.launch(&gpu, p, f.buffers, STREAM).is_err());
        let attempted: Vec<_> = gpu.launches().into_iter().map(|l| l.name).collect();
        assert_eq!(attempted, expected[..=index]);
    }
}
