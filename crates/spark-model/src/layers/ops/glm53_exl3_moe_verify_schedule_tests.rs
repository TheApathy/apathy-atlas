// SPDX-License-Identifier: AGPL-3.0-only
//! CPU-only schedule admission and real launch-wiring tests; no parity claim.
use super::*;
use crate::layers::ops::{
    with_glm53_exact_verify, with_glm53_exact_wide_prefill, with_glm53_layer_major_prefill,
};
use std::ffi::{OsStr, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn policy(group: Option<&str>) -> Glm53Exl3RoutePolicy {
    Glm53Exl3RoutePolicy::parse_with_verify_group(
        Some(OsStr::new("1")),
        Some(OsStr::new("1")),
        group.map(OsStr::new),
        Some(OsStr::new("1")),
    )
    .unwrap()
}

fn kernels(policy: Glm53Exl3RoutePolicy) -> Glm53Exl3MoeKernels {
    Glm53Exl3MoeKernels {
        pack_routes: KernelHandle(11),
        pack_routes_private: KernelHandle(12),
        fused_moe: KernelHandle(43),
        fused_moe_private: KernelHandle(44),
        route_policy: policy,
        large_m: parse_large_m_config(None).unwrap(),
        staged: None,
        verify_staged: None,
        combine_shared: KernelHandle(51),
        combine_private_shared: KernelHandle(52),
    }
}

#[test]
fn verify_group_parser_is_strict_and_default_off() {
    let absent = Glm53Exl3RoutePolicy::parse(None, None).unwrap();
    assert_eq!(
        absent,
        Glm53Exl3RoutePolicy::parse_with_verify_group(None, None, None, None).unwrap()
    );
    assert_eq!(policy(None).verify_group(4, true, false), None);
    for width in ["2", "4", "8"] {
        let selected = policy(Some(width));
        assert_eq!(
            selected.verify_group(4, true, false),
            Some(width.parse::<u32>().unwrap())
        );
        assert!(
            selected
                .validate_moe_mode(Some(OsStr::new("serial-reference")))
                .is_err()
        );
    }
    for invalid in ["", "0", "1", "3", "6", "12", "24", "04", " 4", "4 ", "true"] {
        assert!(
            Glm53Exl3RoutePolicy::parse_with_verify_group(
                Some(OsStr::new("1")),
                None,
                Some(OsStr::new(invalid)),
                Some(OsStr::new("1"))
            )
            .is_err(),
            "{invalid}"
        );
    }
    for (private, exact) in [
        (None, Some("1")),
        (Some("0"), Some("1")),
        (Some("1"), None),
        (Some("1"), Some("0")),
        (Some("1"), Some("true")),
    ] {
        assert!(
            Glm53Exl3RoutePolicy::parse_with_verify_group(
                private.map(OsStr::new),
                None,
                Some(OsStr::new("4")),
                exact.map(OsStr::new)
            )
            .is_err()
        );
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_group_and_exact_prerequisite_are_rejected() {
    use std::os::unix::ffi::OsStrExt;
    let invalid = OsStr::from_bytes(&[0xff]);
    assert!(
        Glm53Exl3RoutePolicy::parse_with_verify_group(
            Some(OsStr::new("1")),
            None,
            Some(invalid),
            Some(OsStr::new("1"))
        )
        .is_err()
    );
    assert!(
        Glm53Exl3RoutePolicy::parse_with_verify_group(
            Some(OsStr::new("1")),
            None,
            Some(OsStr::new("4")),
            Some(invalid)
        )
        .is_err()
    );
}

#[test]
fn candidate_scope_preserves_plain_and_both_prefill_modes() {
    for width in ["2", "4", "8"] {
        let p = policy(Some(width));
        for rows in [0, 1, 9, 1024, 2048, 2049] {
            assert_eq!(p.verify_group(rows, true, false), None);
        }
        for rows in 2..=8 {
            assert_eq!(p.verify_group(rows, false, false), None);
            assert_eq!(p.verify_group(rows, true, true), None);
        }
        let k = kernels(p);
        assert_eq!(k.plan(4).unwrap(), kernels(policy(None)).plan(4).unwrap());
        with_glm53_exact_verify(|| {
            assert_eq!(k.plan(1).unwrap(), kernels(policy(None)).plan(1).unwrap());
            let expected = (48 / width.parse::<usize>().unwrap()).max(8);
            for rows in 2..=8 {
                let plan = k.plan(rows).unwrap();
                assert_eq!(
                    plan.temp_state_f16_bytes,
                    expected * rows as usize * 4096 * 2
                );
                assert_eq!(
                    plan.temp_intermediate_f16_bytes,
                    expected * rows as usize * 2048 * 2
                );
                assert_eq!(plan.route_private_f32_bytes, rows as usize * 8 * 4096 * 4);
            }
            with_glm53_exact_wide_prefill(|| {
                assert_eq!(k.plan(4).unwrap(), kernels(policy(None)).plan(4).unwrap());
            });
            with_glm53_layer_major_prefill(|| {
                assert_eq!(k.plan(4).unwrap(), kernels(policy(None)).plan(4).unwrap());
            });
        });
    }
}

#[test]
fn default_and_explicit_eight_preserve_every_old_plan() {
    let default = kernels(policy(None));
    let control = kernels(policy(Some("8")));
    with_glm53_exact_verify(|| {
        for rows in [1, 2, 3, 4, 8, 9, 1023, 1024, 2048] {
            assert_eq!(default.plan(rows).unwrap(), control.plan(rows).unwrap());
            assert_eq!(default.kernel_launches(rows), control.kernel_launches(rows));
        }
    });
}

#[test]
fn schedule_has_no_split_k_and_fits_pinned_lock_and_scratch_regions() {
    for width in [2, 4, 8] {
        let groups = 48 / width;
        assert!(groups <= 24 && groups * 2 <= 2048 && groups + 2 <= 66);
        assert!(groups * 32 <= 1024 * 1024);
        for (k, n) in [(4096, 2048), (2048, 4096)] {
            let tiles_k = k / 32;
            let tiles_n = n / 256;
            for block in 0..width {
                let begin = tiles_k * tiles_n * block / width;
                let end = tiles_k * tiles_n * (block + 1) / width;
                assert_eq!(begin % tiles_k, 0);
                assert_eq!(end % tiles_k, 0);
                assert_eq!((end - begin) / tiles_k, tiles_n / width);
            }
        }
        let k = kernels(policy(Some(&width.to_string())));
        with_glm53_exact_verify(|| {
            let plan = k.plan(8).unwrap();
            assert!(plan.temp_state_f16_bytes <= 8 * 2048 * 4096 * 2);
            assert!(plan.temp_intermediate_f16_bytes <= 8 * 2048 * 2048 * 2);
        });
    }
}

#[derive(Debug)]
struct Launch {
    function: u64,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    concurrency: Option<i32>,
    output: Option<u64>,
}
#[derive(Default)]
struct Trace {
    launches: Mutex<Vec<Launch>>,
    memsets: AtomicUsize,
    fail_fused: AtomicBool,
    loaded: Mutex<Vec<(String, String)>>,
    shared_attrs: Mutex<Vec<(u64, u32)>>,
}
impl GpuBackend for Trace {
    fn alloc(&self, _: usize) -> Result<DevicePtr> {
        bail!("unused")
    }
    fn alloc_managed(&self, _: usize) -> Result<DevicePtr> {
        bail!("unused")
    }
    fn free(&self, _: DevicePtr) -> Result<()> {
        bail!("unused")
    }
    fn copy_h2d(&self, _: &[u8], _: DevicePtr) -> Result<()> {
        bail!("unused")
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        bail!("unused")
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        bail!("unused")
    }
    fn synchronize(&self, _: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        1
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        self.loaded
            .lock()
            .unwrap()
            .push((module.into(), name.into()));
        let id = match (module, name) {
            ("glm53_exl3_moe_staged_private", "atlas_glm53_exl3_build_chunks_private") => 61,
            ("glm53_exl3_moe_staged_private", "atlas_glm53_exl3_staged_gather_private") => 62,
            ("glm53_exl3_moe_verify_k32", "atlas_glm53_exl3_verify_gate_up_k32") => 63,
            ("glm53_exl3_moe_staged_private", "atlas_glm53_exl3_staged_activate_private") => 64,
            ("glm53_exl3_moe_verify_k32", "atlas_glm53_exl3_verify_down_k32") => 65,
            ("glm53_exl3_moe_staged_private", "atlas_glm53_exl3_staged_scatter_private") => 66,
            ("glm53_exl3_moe_verify_k32_sh4", "atlas_glm53_exl3_verify_gate_up_k32_sh4") => 67,
            ("glm53_exl3_moe_verify_k32_sh4", "atlas_glm53_exl3_verify_down_k32_sh4") => 68,
            _ => bail!("unexpected kernel"),
        };
        Ok(KernelHandle(id))
    }
    fn set_kernel_max_dynamic_shared_memory(&self, kernel: KernelHandle, bytes: u32) -> Result<()> {
        self.shared_attrs.lock().unwrap().push((kernel.0, bytes));
        Ok(())
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        bail!("unused")
    }
    fn memset_async(&self, _: DevicePtr, _: u8, _: usize, _: u64) -> Result<()> {
        self.memsets.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(1 << 30)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(1 << 30)
    }
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        _: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let concurrency = (func.0 == 44).then(|| unsafe { *(params[23] as *const i32) });
        self.launches.lock().unwrap().push(Launch {
            function: func.0,
            grid,
            block,
            shared,
            concurrency,
            output: (func.0 == 66).then(|| unsafe { *(params[1] as *const u64) }),
        });
        ensure!(
            !([44, 63].contains(&func.0) && self.fail_fused.load(Ordering::Relaxed)),
            "injected launch failure"
        );
        Ok(())
    }
}

fn buffers(p: Glm53Exl3MoePlan) -> Glm53Exl3MoeBuffers {
    let mut cursor = 0x10_0000_u64;
    let mut b = |bytes: usize| {
        let result = Glm53Exl3Buffer {
            ptr: DevicePtr(cursor),
            bytes,
        };
        cursor += bytes as u64 + 4096;
        result
    };
    Glm53Exl3MoeBuffers {
        input_f16: b(p.input_f16_bytes),
        output_f32: b(p.output_f32_bytes),
        route_private_f32: b(p.route_private_f32_bytes),
        route_ids_u32: b(p.route_ids_u32_bytes),
        route_weights_f32: b(p.route_weights_f32_bytes),
        expert_count_i64: b(p.expert_count_i64_bytes),
        token_sorted_i64: b(p.token_sorted_i64_bytes),
        weight_sorted_f16: b(p.weight_sorted_f16_bytes),
        temp_state_g_f16: b(p.temp_state_f16_bytes),
        temp_state_u_f16: b(p.temp_state_f16_bytes),
        temp_intermediate_g_f16: b(p.temp_intermediate_f16_bytes),
        temp_intermediate_u_f16: b(p.temp_intermediate_f16_bytes),
        route_status_u32: b(4),
        locks_i32: b(GLM53_EXL3_MOE_LOCK_BYTES),
        pair_expert_u32: b(p.pair_expert_u32_bytes),
        chunk_expert_u32: b(p.chunk_descriptor_u32_bytes),
        chunk_start_u32: b(p.chunk_descriptor_u32_bytes),
        chunk_rows_u32: b(p.chunk_descriptor_u32_bytes),
        chunk_count_u32: b(p.chunk_count_u32_bytes),
        pointers: Glm53Exl3MoePointerTables {
            gate_trellis: b(p.pointer_table_u64_bytes),
            gate_suh: b(p.pointer_table_u64_bytes),
            gate_svh: b(p.pointer_table_u64_bytes),
            up_trellis: b(p.pointer_table_u64_bytes),
            up_suh: b(p.pointer_table_u64_bytes),
            up_svh: b(p.pointer_table_u64_bytes),
            down_trellis: b(p.pointer_table_u64_bytes),
            down_suh: b(p.pointer_table_u64_bytes),
            down_svh: b(p.pointer_table_u64_bytes),
        },
    }
}

#[test]
fn real_launch_uses_same_private_kernel_and_selected_grid() {
    for (value, width, groups) in [
        (None, 8, 6),
        (Some("8"), 8, 6),
        (Some("4"), 4, 12),
        (Some("2"), 2, 24),
    ] {
        let k = kernels(policy(value));
        with_glm53_exact_verify(|| {
            for rows in [1, 2, 4, 8] {
                let gpu = Trace::default();
                let plan = k.plan(rows).unwrap();
                k.launch(&gpu, plan, buffers(plan), 7).unwrap();
                let launches = gpu.launches.lock().unwrap();
                assert_eq!(launches.len(), 2);
                assert_eq!(launches[0].function, 12);
                let actual = &launches[1];
                let (width, groups) = if rows == 1 { (8, 6) } else { (width, groups) };
                assert_eq!(actual.function, 44);
                assert_eq!(actual.grid, [width, 1, groups]);
                assert_eq!(actual.concurrency, Some(groups as i32));
                assert_eq!(actual.block, [512, 1, 1]);
                assert_eq!(actual.shared, 90 * 1024);
            }
        });
    }
}

#[test]
fn wrong_extents_or_scope_fail_before_any_enqueue() {
    let k = kernels(policy(Some("2")));
    let plan = with_glm53_exact_verify(|| {
        let plan = k.plan(4).unwrap();
        let gpu = Trace::default();
        let mut wrong = buffers(plan);
        wrong.temp_state_g_f16.bytes -= 2;
        assert!(k.launch(&gpu, plan, wrong, 7).is_err());
        assert_eq!(gpu.memsets.load(Ordering::Relaxed), 0);
        assert!(gpu.launches.lock().unwrap().is_empty());
        plan
    });
    let gpu = Trace::default();
    assert!(k.launch(&gpu, plan, buffers(plan), 7).is_err());
    assert_eq!(gpu.memsets.load(Ordering::Relaxed), 0);
    assert!(gpu.launches.lock().unwrap().is_empty());
}

#[test]
fn failed_selected_matmul_is_not_retried_or_fallen_back() {
    let k = kernels(policy(Some("4")));
    let gpu = Trace::default();
    gpu.fail_fused.store(true, Ordering::Relaxed);
    with_glm53_exact_verify(|| {
        let plan = k.plan(4).unwrap();
        assert!(k.launch(&gpu, plan, buffers(plan), 7).is_err());
    });
    let launches = gpu.launches.lock().unwrap();
    assert_eq!(launches.len(), 2);
    assert_eq!(launches[1].function, 44);
    assert_eq!(launches[1].grid, [4, 1, 12]);
}

#[test]
fn production_latch_preflight_and_graph_binding_are_wired() {
    const TARGET: &str = include_str!("../../model/glm53/target_model_exl3.rs");
    const MOE: &str = include_str!("glm53_exl3_moe.rs");
    const SERIAL: &str = include_str!("../glm53_moe_serial.rs");
    const GRAPH: &str = include_str!("../../model/glm53/ffn_graph_runtime.rs");
    const STAGED: &str = include_str!("../../model/glm53/target_staged_exl3.rs");
    for source in [TARGET, MOE, SERIAL] {
        assert!(source.contains("Glm53Exl3RoutePolicy::parse_with_verify_group("));
        assert!(source.contains("ATLAS_GLM53_EXL3_MOE_VERIFY_GROUP_WIDTH"));
    }
    assert!(MOE.contains("plan == self.plan(plan.rows)?"));
    assert!(MOE.contains("let (group_width, concurrency) = self.fused_schedule(plan.rows);"));
    let fused = &SERIAL[SERIAL.find("pub fn execute_exl3_fused_rows(").unwrap()..];
    assert!(
        fused.find("fused.validate(plan, exact)?").unwrap()
            < fused.find("casts.bf16_to_f16(").unwrap()
    );
    assert!(GRAPH.contains("self.environment == environment()"));
    assert!(GRAPH.contains("starts_with(b\"ATLAS_GLM53_\")"));
    for owner in [
        "self.wide_workspace_allocation.0",
        "self.scratch_allocation.0",
        "self.moe_tables_allocation.0",
    ] {
        assert!(STAGED.contains(owner));
    }
}

#[path = "glm53_exl3_moe_verify_staged_tests.rs"]
mod verify_staged_tests;
