// SPDX-License-Identifier: AGPL-3.0-only

//! Exact positional ABI assertions for the real production dispatcher.
use super::{
    Glm53Exl3MoePlan, STREAM,
    fixture::{Arg, Fixture, Launch},
};

const PRIVATE_MODULE: &str = "glm53_exl3_moe_staged_private";
const BASE_MODULE: &str = "glm53_exl3_moe_staged_k16";
const GEMM_MODULE: &str = "glm53_exl3_moe_staged_n256_f1";

fn launch_eq(
    l: &Launch,
    module: &str,
    name: &str,
    grid: [u32; 3],
    block: u32,
    shared: u32,
    args: Vec<Arg>,
) {
    assert_eq!((&*l.module, &*l.name), (module, name));
    assert_eq!(l.grid, grid);
    assert_eq!(l.block, [block, 1, 1]);
    assert_eq!(l.shared, shared);
    assert_eq!(l.stream, STREAM);
    assert_eq!(l.args, args);
}

pub(super) fn check_pack(l: &Launch, p: Glm53Exl3MoePlan, f: &Fixture, private: bool) {
    let b = f.buffers;
    launch_eq(
        l,
        "glm53_exl3_moe_route",
        if private {
            "atlas_glm53_exl3_pack_routes_private"
        } else {
            "atlas_glm53_exl3_pack_routes"
        },
        [1, 1, 1],
        320,
        0,
        vec![
            Arg::ptr(b.route_ids_u32),
            Arg::ptr(b.route_weights_f32),
            Arg::ptr(b.expert_count_i64),
            Arg::ptr(b.token_sorted_i64),
            Arg::ptr(b.weight_sorted_f16),
            Arg::ptr(b.route_status_u32),
            Arg::word(p.rows),
            Arg::word(288),
            Arg::word(8),
        ],
    );
}

pub(super) fn check_combine(l: &Launch, p: Glm53Exl3MoePlan, f: &Fixture, private: bool) {
    let routed = if private {
        f.buffers.route_private_f32
    } else {
        f.buffers.output_f32
    };
    launch_eq(
        l,
        "glm53_exl3_moe_route",
        if private {
            "atlas_glm53_exl3_combine_private_shared"
        } else {
            "atlas_glm53_exl3_combine_shared"
        },
        [(p.rows * 4096).div_ceil(256), 1, 1],
        256,
        0,
        vec![
            Arg::ptr(routed),
            Arg::ptr(f.shared),
            Arg::ptr(f.combined),
            Arg::ptr(f.buffers.route_status_u32),
            Arg::word(p.rows * 4096),
        ],
    );
}

pub(super) fn check_staged(ls: &[Launch], p: Glm53Exl3MoePlan, f: &Fixture, private: bool) {
    assert_eq!(ls.len(), 6);
    let b = f.buffers;
    let w = b.pointers;
    let module = if private { PRIVATE_MODULE } else { BASE_MODULE };
    let mut args = vec![
        Arg::ptr(b.expert_count_i64),
        Arg::ptr(b.pair_expert_u32),
        Arg::ptr(b.chunk_expert_u32),
        Arg::ptr(b.chunk_start_u32),
        Arg::ptr(b.chunk_rows_u32),
        Arg::ptr(b.chunk_count_u32),
        Arg::ptr(b.route_status_u32),
        Arg::word(288),
        Arg::word(p.max_chunks),
    ];
    if private {
        args.push(Arg::word(p.pairs));
    }
    launch_eq(
        &ls[0],
        module,
        if private {
            "atlas_glm53_exl3_build_chunks_private"
        } else {
            "atlas_glm53_exl3_build_chunks"
        },
        [1, 1, 1],
        320,
        0,
        args,
    );
    let mut args = vec![
        Arg::ptr(b.input_f16),
        Arg::ptr(b.temp_state_g_f16),
        Arg::ptr(b.temp_state_u_f16),
        Arg::ptr(w.gate_suh),
        Arg::ptr(w.up_suh),
        Arg::ptr(b.token_sorted_i64),
        Arg::ptr(b.pair_expert_u32),
        Arg::word(p.pairs),
    ];
    if private {
        args.push(Arg::ptr(b.route_status_u32));
    }
    launch_eq(
        &ls[1],
        module,
        if private {
            "atlas_glm53_exl3_staged_gather_private"
        } else {
            "atlas_glm53_exl3_staged_gather"
        },
        [p.pairs, 1, 1],
        256,
        0,
        args,
    );
    launch_eq(
        &ls[2],
        GEMM_MODULE,
        "atlas_glm53_exl3_staged_gate_up_k16",
        [8, p.max_chunks, 2],
        256,
        20992,
        vec![
            Arg::ptr(b.temp_state_g_f16),
            Arg::ptr(b.temp_state_u_f16),
            Arg::ptr(b.temp_intermediate_g_f16),
            Arg::ptr(b.temp_intermediate_u_f16),
            Arg::ptr(w.gate_trellis),
            Arg::ptr(w.up_trellis),
            Arg::ptr(b.chunk_expert_u32),
            Arg::ptr(b.chunk_start_u32),
            Arg::ptr(b.chunk_rows_u32),
            Arg::ptr(b.chunk_count_u32),
            Arg::word(4096),
            Arg::word(2048),
            Arg::word(128),
            Arg::ptr(b.locks_i32),
        ],
    );
    let mut args = vec![
        Arg::ptr(b.temp_intermediate_g_f16),
        Arg::ptr(b.temp_intermediate_u_f16),
        Arg::ptr(w.gate_svh),
        Arg::ptr(w.up_svh),
        Arg::ptr(w.down_suh),
        Arg::ptr(b.pair_expert_u32),
        Arg::word(p.pairs),
    ];
    if private {
        args.push(Arg::ptr(b.route_status_u32));
    }
    launch_eq(
        &ls[3],
        module,
        if private {
            "atlas_glm53_exl3_staged_activate_private"
        } else {
            "atlas_glm53_exl3_staged_activate"
        },
        [p.pairs, 1, 1],
        256,
        0,
        args,
    );
    launch_eq(
        &ls[4],
        GEMM_MODULE,
        "atlas_glm53_exl3_staged_down_k16",
        [16, p.max_chunks, 1],
        256,
        20992,
        vec![
            Arg::ptr(b.temp_intermediate_g_f16),
            Arg::ptr(b.temp_state_g_f16),
            Arg::ptr(w.down_trellis),
            Arg::ptr(b.chunk_expert_u32),
            Arg::ptr(b.chunk_start_u32),
            Arg::ptr(b.chunk_rows_u32),
            Arg::ptr(b.chunk_count_u32),
            Arg::word(256),
            Arg::ptr(b.locks_i32),
        ],
    );
    let mut args = vec![
        Arg::ptr(b.temp_state_g_f16),
        Arg::ptr(if private {
            b.route_private_f32
        } else {
            b.output_f32
        }),
        Arg::ptr(w.down_svh),
        Arg::ptr(b.token_sorted_i64),
        Arg::ptr(b.weight_sorted_f16),
        Arg::ptr(b.pair_expert_u32),
        Arg::word(p.pairs),
    ];
    if private {
        args.push(Arg::ptr(b.route_status_u32));
    }
    launch_eq(
        &ls[5],
        module,
        if private {
            "atlas_glm53_exl3_staged_scatter_private"
        } else {
            "atlas_glm53_exl3_staged_scatter"
        },
        [p.pairs, 1, 1],
        256,
        4096,
        args,
    );
}

pub(super) fn check_fused(l: &Launch, p: Glm53Exl3MoePlan, f: &Fixture, private: bool) {
    assert_eq!(
        l.module,
        if private {
            "glm53_exl3_moe_private"
        } else {
            "glm53_exl3_moe_k2_cb2"
        }
    );
    if private {
        assert_eq!(l.name, "atlas_glm53_exl3_moe_private_k2_n256_cb2");
    } else {
        assert!(l.name.starts_with("_Z15exl3_moe_kernelILi2ELi256ELi2EE"));
    }
    let (width, concurrency) = if p.rows <= 8 { (8, 6) } else { (12, 4) };
    assert_eq!(l.grid, [width, 1, concurrency]);
    assert_eq!(l.block, [512, 1, 1]);
    assert_eq!(l.shared, 90 * 1024);
    assert_eq!(l.stream, STREAM);
    assert_eq!(l.args.len(), 30);
    assert_eq!(l.args[0], Arg::ptr(f.buffers.input_f16));
    assert_eq!(
        l.args[5],
        Arg::ptr(if private {
            f.buffers.route_private_f32
        } else {
            f.buffers.output_f32
        })
    );
    assert_eq!(
        &l.args[18..29],
        &[
            Arg::word(4096),
            Arg::word(2048),
            Arg::word(288),
            Arg::word(8),
            Arg::word(p.rows),
            Arg::word(concurrency),
            Arg::word(10.0_f32.to_bits()),
            Arg::word(0),
            Arg::word(2),
            Arg::word(2),
            Arg::word(2)
        ]
    );
    assert_eq!(l.args[29], Arg::ptr(f.buffers.locks_i32));
}
