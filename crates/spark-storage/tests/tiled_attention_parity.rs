// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-2 validation gate for the streaming attention design: the same
// per-step result must come out whether the kernel is invoked once for the
// full block list or in N tiles that cover the same blocks. Online softmax is
// associative; the only differences between the two paths are float
// reordering inside `__expf` and the running-state quantization at the
// kernel boundaries (m, l, o stay fp32 — no quantization at boundaries).

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use spark_storage::attention_ref::{AttnState, finalize_ref, step_tile_ref};
use spark_storage::cuda_min::{
    CudaCtx, DeviceBuffer, HostToDeviceCopy, copy_d_to_h, copy_h_to_d_group,
};
use spark_storage::tiled_attention::{TiledAttention, TiledAttentionDims};

const NUM_SEQS: usize = 1;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 16;

fn dims(tile_capacity: usize) -> TiledAttentionDims {
    TiledAttentionDims {
        max_seqs: NUM_SEQS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
        tile_capacity,
    }
}

fn random_bf16(n: usize, rng: &mut ChaCha8Rng) -> Vec<bf16> {
    let dist = StandardNormal;
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(rng);
            bf16::from_f32(v * inv)
        })
        .collect()
}

fn run_gpu(tile_size: usize, q: &[bf16], k: &[bf16], v: &[bf16], block_table: &[i32]) -> Vec<bf16> {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let attn = TiledAttention::new(dims(tile_size)).unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let k_dev = DeviceBuffer::new(k.len() * 2).unwrap();
    let v_dev = DeviceBuffer::new(v.len() * 2).unwrap();
    let tile_blocks_dev = DeviceBuffer::new(NUM_SEQS * tile_size * 4).unwrap();
    let tile_counts_dev = DeviceBuffer::new(NUM_SEQS * 4).unwrap();
    let output_dev = DeviceBuffer::new(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM * 2).unwrap();
    copy_h_to_d_group(
        &[
            HostToDeviceCopy::new(q_dev.ptr, q),
            HostToDeviceCopy::new(k_dev.ptr, k),
            HostToDeviceCopy::new(v_dev.ptr, v),
        ],
        ctx.stream,
    )
    .unwrap();
    attn.begin_step(&ctx, NUM_SEQS).unwrap();

    let n_tiles = block_table.len().div_ceil(tile_size);
    for t in 0..n_tiles {
        let start = t * tile_size;
        let end = (start + tile_size).min(block_table.len());
        let n = end - start;
        // Pad tile to tile_size with zeros (only first `n` are valid).
        let mut tile = vec![0_i32; tile_size];
        tile[..n].copy_from_slice(&block_table[start..end]);
        let counts = vec![n as i32; NUM_SEQS];
        copy_h_to_d_group(
            &[
                HostToDeviceCopy::new(tile_blocks_dev.ptr, &tile),
                HostToDeviceCopy::new(tile_counts_dev.ptr, &counts),
            ],
            ctx.stream,
        )
        .unwrap();
        let (s_blk, s_tok, s_kvh) = attn.paged_strides();
        attn.step_tile(
            &ctx,
            q_dev.ptr,
            k_dev.ptr,
            v_dev.ptr,
            tile_blocks_dev.ptr,
            tile_counts_dev.ptr,
            NUM_SEQS,
            s_blk,
            s_tok,
            s_kvh,
            BLOCK_SIZE as i32,
        )
        .unwrap();
    }
    attn.finalize(&ctx, output_dev.ptr, NUM_SEQS).unwrap();
    let mut out = vec![bf16::from_f32(0.0); NUM_SEQS * NUM_Q_HEADS * HEAD_DIM];
    copy_d_to_h(&mut out, output_dev.ptr, ctx.stream).unwrap();
    out
}

fn run_ref(tile_size: usize, q: &[bf16], k: &[bf16], v: &[bf16], block_table: &[i32]) -> Vec<bf16> {
    let mut state = AttnState::new(NUM_SEQS, NUM_Q_HEADS, HEAD_DIM);
    let n_tiles = block_table.len().div_ceil(tile_size);
    let gqa = NUM_Q_HEADS / NUM_KV_HEADS;
    for t in 0..n_tiles {
        let start = t * tile_size;
        let end = (start + tile_size).min(block_table.len());
        let n = end - start;
        let mut tile = vec![0_i32; tile_size];
        tile[..n].copy_from_slice(&block_table[start..end]);
        let counts = vec![n as i32; NUM_SEQS];
        step_tile_ref(
            &mut state,
            q,
            k,
            v,
            &tile,
            &counts,
            NUM_SEQS,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            tile_size,
            gqa,
        );
    }
    finalize_ref(&state, NUM_SEQS, NUM_Q_HEADS, HEAD_DIM)
}

fn diff_stats(a: &[bf16], b: &[bf16]) -> (f32, f32) {
    let mut max_d = 0.0_f32;
    let mut sum_d = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x.to_f32() - y.to_f32()).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
    }
    (max_d, sum_d / a.len() as f32)
}

fn build_inputs(seed: u64) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = random_bf16(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM, &mut rng);
    let k = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let v = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let block_table: Vec<i32> = (0..NUM_BLOCKS as i32).collect();
    (q, k, v, block_table)
}

#[test]
#[ignore = "requires GPU"]
fn single_tile_matches_reference() {
    let (q, k, v, bt) = build_inputs(0xCAFE);
    let gpu = run_gpu(NUM_BLOCKS, &q, &k, &v, &bt);
    let cpu = run_ref(NUM_BLOCKS, &q, &k, &v, &bt);
    let (max_d, mean_d) = diff_stats(&gpu, &cpu);
    eprintln!("single tile: max abs diff = {max_d:.3e}, mean = {mean_d:.3e}");
    assert!(max_d < 1e-2, "single-tile gpu vs ref max_d = {max_d}");
}

#[test]
#[ignore = "requires GPU"]
fn multi_tile_matches_single_tile() {
    let (q, k, v, bt) = build_inputs(0xBEEF);
    let single = run_gpu(NUM_BLOCKS, &q, &k, &v, &bt);
    for tile_size in [1, 2, 4, 8] {
        let multi = run_gpu(tile_size, &q, &k, &v, &bt);
        let (max_d, mean_d) = diff_stats(&single, &multi);
        eprintln!("tile_size={tile_size}: max abs diff = {max_d:.3e}, mean = {mean_d:.3e}");
        assert!(max_d < 1e-2, "tile_size={tile_size} max_d = {max_d}");
        assert!(mean_d < 1e-3, "tile_size={tile_size} mean_d = {mean_d}");
    }
}
