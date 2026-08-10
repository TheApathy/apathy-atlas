// SPDX-License-Identifier: AGPL-3.0-only

//! Equivalence + speedup oracle for the **multi-row (`_m`) dedup'd split-K
//! unified-`_t` decode MoE**, at DeepSeek-V4-Flash shapes (hidden 4096,
//! moe_intermediate 2048, top-6 routed + 1 shared, routed MXFP4/E8M0 per-32,
//! shared NVFP4/E4M3 per-16).
//!
//! Why this exists: the MTP K=2 verify runs the MoE for two candidate rows.
//! The two rows' routed expert sets overlap (measured 1.28x on learned-gate
//! layers, 2.01x on hash-routed ones) and the shared expert is duplicated
//! outright, so a kernel that reads each weight byte once and FMAs it into
//! both rows should cut the verify's weight traffic by ~0.72x. The previous
//! attempt at this (`batch2_t`) *lost* — it batched on top of the pre-split-K
//! kernel shape and gave back more to narrow loads than it won on reuse. This
//! measures the rewrite that batches on top of the fast `v2s4` body instead.
//!
//! Two gates, in order:
//!   1. `_m1v2s4` at num_tokens=1 must be **BIT-IDENTICAL** to the shipping
//!      `_v2s4`. The dedup/MROW rewrite must not perturb the k order at all.
//!      If this fails, nothing below it means anything.
//!   2. `_m{MROW}v2s4` at num_tokens=`tokens` must match `tokens` independent
//!      `_v2s4` passes to a bf16-ULP bound. It is NOT bit-equal: a leader block
//!      sums its k window for every row it gathered, which is the same order
//!      per row, but the *shared* row is served by one block-set instead of
//!      `tokens`, and the partial rows differ. In practice this comes out
//!      bit-equal too; the bound is the contract.
//!
//! Then the timing leg: `tokens` `_v2s4` chains (what the verify costs today —
//! the per-token fallback, or `ATLAS_K2_MOE_PER_TOKEN=1` at K=2) vs one
//! `_m{MROW}v2s4` chain.
//!
//! V2 leg (when the `_v4s4` entries are compiled): GATE V2 requires the
//! `ATLAS_MOE_SPLITK_V2` tier — `_m{MROW}v4s4`, VEC=4 weight loads at the same
//! SPLIT plus smem-staged gate_up activations — to be **BIT-IDENTICAL** to the
//! `_m{MROW}v2s4` incumbent, then times the two chains. Sweep `tokens` over
//! 4/6/8 and `overlap` to read the row-per-leader curve.
//!
//! Usage:
//!   cargo run --release -p spark-model --example moe_unified_t_m_microtest \
//!       -- [block] [pool] [seed-hex] [top_k] [overlap] [tokens]
//! Defaults: 32 24 0xD54 6 -1 2
//!
//! `tokens` is the verify width: 2 for the MTP K=2 verify, 6 for the DSpark
//! block (5 proposed rows + the committed one). It selects the narrowest
//! compiled `_m` entry point that covers it, exactly as the Rust dispatch does.
//!
//! `overlap` forces how many of rows 1.. keep of row 0's top_k picks (0..top_k),
//! or -1 for random routing. Sweeping it is how you read the speedup curve:
//! overlap=0 is the worst case (disjoint routing, only the shared expert is
//! amortized), overlap=top_k the best (hash-routed layers, where every row
//! picks the identical top-k).

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const MODULE: &str = "moe_shared_expert_fused_t";
const HIDDEN: usize = 4096; // K for gate/up, N for down
const INTER: usize = 2048; // N for gate/up, K for down
const GS_ROUTED: usize = 32; // MXFP4 E8M0 per-32
const GS_SHARED: usize = 16; // NVFP4 E4M3 per-16
const NUM_EXPERT_SLOTS: usize = 256; // pointer-table width (production)
const SPLIT: u32 = 4; // must match `forward_phase::T_SPLIT`
const VEC: u32 = 2; // must match `ops::T_SPLIT_VEC`
const LAYERS: f64 = 43.0; // DeepSeek-V4-Flash
/// Widest compiled `_m` entry point — must track
/// `spark_runtime::buffers::MOE_DECODE_MAX_ROWS` and the `_m8v2s4` entries.
const MROW_MAX: u32 = 8;
/// The MROW=6 rung of the ladder. `MoeLayer::splitk_m_t_handles` stops here for
/// any verify that fits it, so the oracle has to as well or it would validate a
/// kernel the engine never launches at six rows.
const MROW_M6: u32 = 6;
/// The persistent Stage-0 entry points top out at `_m6_persistent_v2s4`; that
/// leg is pinned to the six-row DSpark bridge shape and is not part of the
/// MROW=8 widening.
const MROW_PERSIST: u32 = 6;
const ITERS: usize = 200;
const PERSISTENT_BLOCK: u32 = 256;
const PERSISTENT_TASKS_PER_RECORD: u32 = SPLIT * 4;
const WORK_COUNT_MASK: u32 = 0x7;
const WORK_SHARED: u32 = 1 << 8;
const WORK_UP: u32 = 1 << 9;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            v.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        v.truncate(n);
        v
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// One transposed weight `[K/2, N]` packed + `[K/GS, N]` scales.
struct Wt {
    packed: Vec<u8>,
    scales: Vec<u8>,
    s2: f32,
}
impl Wt {
    /// E8M0 routed: exponent byte in [113,141] (2^-14..2^14), never 0/255
    /// (those decode to zero and would hide a scale-indexing bug).
    fn routed(rng: &mut Rng, k: usize, n: usize) -> Self {
        Wt {
            packed: rng.bytes(k / 2 * n),
            scales: (0..k / GS_ROUTED * n)
                .map(|_| 113 + (rng.next_u64() % 29) as u8)
                .collect(),
            s2: 1.0,
        }
    }
    /// NVFP4 shared: E4M3 scale bytes, 0x7F/0xFF (NaN) remapped to 1.0.
    fn shared(rng: &mut Rng, k: usize, n: usize) -> Self {
        Wt {
            packed: rng.bytes(k / 2 * n),
            scales: (0..k / GS_SHARED * n)
                .map(|_| {
                    let b = (rng.next_u64() & 0xFF) as u8;
                    if b == 0x7F || b == 0xFF { 0x38 } else { b }
                })
                .collect(),
            s2: 0.037,
        }
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64s(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Table {
    packed: DevicePtr,
    scales: DevicePtr,
    s2: DevicePtr,
    packed_host: Vec<u64>,
    scales_host: Vec<u64>,
    s2_host: Vec<f32>,
}

fn build_table(gpu: &dyn GpuBackend, ws: &[Wt]) -> Result<Table> {
    let mut wp = vec![0u64; NUM_EXPERT_SLOTS];
    let mut sp = vec![0u64; NUM_EXPERT_SLOTS];
    let mut s2 = vec![0f32; NUM_EXPERT_SLOTS];
    for (e, w) in ws.iter().enumerate() {
        wp[e] = upload(gpu, &w.packed)?.0;
        sp[e] = upload(gpu, &w.scales)?.0;
        s2[e] = w.s2;
    }
    Ok(Table {
        packed: upload(gpu, &u64s(&wp))?,
        scales: upload(gpu, &u64s(&sp))?,
        s2: upload(gpu, &f32s(&s2))?,
        packed_host: wp,
        scales_host: sp,
        s2_host: s2,
    })
}

/// Host-built, device-consumed work descriptor. Keeping the exact 48-byte
/// layout here and in CUDA is part of the microtest contract: the persistent
/// kernels do no pointer-table lookup and no routing scan.
///
/// `slots` is sized by `MROW_PERSIST`, NOT `MROW_MAX` — 6 u32 is what makes the
/// struct 48 bytes, and the CUDA side hard-codes that. Widening the `_m` ladder
/// to MROW=8 must not silently restride this.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C, align(16))]
struct PersistentWork {
    packed: u64,
    scale: u64,
    scale2_bits: u32,
    meta: u32,
    slots: [u32; MROW_PERSIST as usize],
}

impl PersistentWork {
    fn new(
        packed: u64,
        scale: u64,
        scale2: f32,
        count: usize,
        shared: bool,
        up: bool,
        gathered: &[u32],
    ) -> Self {
        assert!((1..=MROW_PERSIST as usize).contains(&count));
        assert_eq!(count, gathered.len());
        let mut slots = [gathered[0]; MROW_PERSIST as usize];
        slots[..count].copy_from_slice(gathered);
        Self {
            packed,
            scale,
            scale2_bits: scale2.to_bits(),
            meta: count as u32
                | if shared { WORK_SHARED } else { 0 }
                | if up { WORK_UP } else { 0 },
            slots,
        }
    }

    fn count(self) -> usize {
        (self.meta & WORK_COUNT_MASK) as usize
    }

    fn to_bytes(self) -> [u8; 48] {
        let mut out = [0u8; 48];
        out[0..8].copy_from_slice(&self.packed.to_le_bytes());
        out[8..16].copy_from_slice(&self.scale.to_le_bytes());
        out[16..20].copy_from_slice(&self.scale2_bits.to_le_bytes());
        out[20..24].copy_from_slice(&self.meta.to_le_bytes());
        for (i, slot) in self.slots.iter().enumerate() {
            out[24 + i * 4..28 + i * 4].copy_from_slice(&slot.to_le_bytes());
        }
        out
    }
}

#[derive(Clone, Copy)]
enum WorkOrder {
    Leader,
    Pointer,
}

fn work_bytes(work: &[PersistentWork]) -> Vec<u8> {
    work.iter().flat_map(|record| record.to_bytes()).collect()
}

fn gathered_experts(routing: &[Vec<u32>]) -> Vec<(u32, Vec<u32>)> {
    let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
    for (slot, expert) in routing.iter().flatten().copied().enumerate() {
        if let Some((_, slots)) = groups.iter_mut().find(|(e, _)| *e == expert) {
            slots.push(slot as u32);
        } else {
            groups.push((expert, vec![slot as u32]));
        }
    }
    groups
}

fn make_work(
    routing: &[Vec<u32>],
    tokens: usize,
    order: WorkOrder,
    gate: &Table,
    up: &Table,
    down: &Table,
    shared: [(u64, u64, f32); 3],
) -> (Vec<PersistentWork>, Vec<PersistentWork>) {
    let total_routed = routing.iter().map(Vec::len).sum::<usize>() as u32;
    let groups = gathered_experts(routing);
    let shared_slots: Vec<u32> = (0..tokens)
        .map(|token| total_routed + token as u32)
        .collect();
    let mut gu = Vec::with_capacity(2 * (groups.len() + 1));
    for (is_up, table, sh) in [(false, gate, shared[0]), (true, up, shared[1])] {
        for (expert, slots) in &groups {
            let e = *expert as usize;
            gu.push(PersistentWork::new(
                table.packed_host[e],
                table.scales_host[e],
                table.s2_host[e],
                slots.len(),
                false,
                is_up,
                slots,
            ));
        }
        gu.push(PersistentWork::new(
            sh.0,
            sh.1,
            sh.2,
            tokens,
            true,
            is_up,
            &shared_slots,
        ));
    }
    let mut dn = Vec::with_capacity(groups.len() + 1);
    for (expert, slots) in &groups {
        let e = *expert as usize;
        dn.push(PersistentWork::new(
            down.packed_host[e],
            down.scales_host[e],
            down.s2_host[e],
            slots.len(),
            false,
            false,
            slots,
        ));
    }
    dn.push(PersistentWork::new(
        shared[2].0,
        shared[2].1,
        shared[2].2,
        tokens,
        true,
        false,
        &shared_slots,
    ));
    if matches!(order, WorkOrder::Pointer) {
        gu.sort_unstable_by_key(|record| (record.packed, record.scale, record.meta & WORK_UP));
        dn.sort_unstable_by_key(|record| (record.packed, record.scale));
    }
    (gu, dn)
}

struct DeviceWork {
    ptr: DevicePtr,
    count: u32,
}

struct PersistentWorkSet {
    gu: [Option<DeviceWork>; 4],
    down: [Option<DeviceWork>; 4],
    shared_gu: DeviceWork,
    shared_down: DeviceWork,
}

fn work_bucket(count: usize) -> usize {
    match count {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        5 | 6 => 3,
        _ => panic!("invalid persistent row count {count}"),
    }
}

fn upload_work(gpu: &dyn GpuBackend, records: &[PersistentWork]) -> Result<DeviceWork> {
    assert!(!records.is_empty());
    Ok(DeviceWork {
        ptr: upload(gpu, &work_bytes(records))?,
        count: records.len() as u32,
    })
}

fn upload_work_set(
    gpu: &dyn GpuBackend,
    gu: &[PersistentWork],
    down: &[PersistentWork],
) -> Result<PersistentWorkSet> {
    let mut gu_buckets: [Vec<PersistentWork>; 4] = std::array::from_fn(|_| Vec::new());
    let mut down_buckets: [Vec<PersistentWork>; 4] = std::array::from_fn(|_| Vec::new());
    let mut shared_gu = Vec::new();
    let mut shared_down = Vec::new();
    for record in gu {
        if record.meta & WORK_SHARED != 0 {
            shared_gu.push(*record);
        } else {
            gu_buckets[work_bucket(record.count())].push(*record);
        }
    }
    for record in down {
        if record.meta & WORK_SHARED != 0 {
            shared_down.push(*record);
        } else {
            down_buckets[work_bucket(record.count())].push(*record);
        }
    }
    assert_eq!(shared_gu.len(), 2);
    assert_eq!(shared_down.len(), 1);
    let upload_buckets = |buckets: &[Vec<PersistentWork>; 4]| -> Result<_> {
        let mut uploaded = [None, None, None, None];
        for (i, bucket) in buckets.iter().enumerate() {
            if !bucket.is_empty() {
                uploaded[i] = Some(upload_work(gpu, bucket)?);
            }
        }
        Ok(uploaded)
    };
    Ok(PersistentWorkSet {
        gu: upload_buckets(&gu_buckets)?,
        down: upload_buckets(&down_buckets)?,
        shared_gu: upload_work(gpu, &shared_gu)?,
        shared_down: upload_work(gpu, &shared_down)?,
    })
}

#[derive(Clone)]
struct RoutingFixture {
    label: &'static str,
    routing: Vec<Vec<u32>>,
}

fn persistent_fixtures(pool: usize, seed: u64) -> Vec<RoutingFixture> {
    assert!(
        pool >= 36,
        "persistent fixtures require at least 36 experts"
    );
    // Six rows: the persistent Stage-0 kernels top out at `_m6_persistent`.
    let bridge = (0..MROW_PERSIST as usize)
        .map(|row| {
            (0..6)
                .map(|column| ((row * 2 + column) % 13) as u32)
                .collect()
        })
        .collect();
    let distinct = (0..MROW_PERSIST as usize)
        .map(|row| (0..6).map(|column| (row * 6 + column) as u32).collect())
        .collect();
    let repeated = vec![(0..6).map(|expert| expert as u32).collect(); MROW_PERSIST as usize];
    let mut random_rng = Rng(seed ^ 0xB71D_6E5A_8C29_F043);
    let random = (0..MROW_PERSIST as usize)
        .map(|_| {
            let mut row = Vec::with_capacity(6);
            while row.len() < 6 {
                let expert = (random_rng.next_u64() % pool as u64) as u32;
                if !row.contains(&expert) {
                    row.push(expert);
                }
            }
            row
        })
        .collect();
    vec![
        RoutingFixture {
            label: "bridge-u13",
            routing: bridge,
        },
        RoutingFixture {
            label: "distinct-u36",
            routing: distinct,
        },
        RoutingFixture {
            label: "random",
            routing: random,
        },
        RoutingFixture {
            label: "repeated-u6",
            routing: repeated,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_work_wire_layout_is_exactly_48_bytes() {
        assert_eq!(std::mem::size_of::<PersistentWork>(), 48);
        assert_eq!(std::mem::align_of::<PersistentWork>(), 16);
        let record = PersistentWork::new(1, 2, 0.5, 2, true, true, &[9, 11]);
        let bytes = record.to_bytes();
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            0.5f32.to_bits()
        );
        assert_eq!(
            u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            2 | WORK_SHARED | WORK_UP
        );
        assert_eq!(record.count(), 2);
        assert_eq!(record.slots, [9, 11, 9, 9, 9, 9]);
    }

    #[test]
    fn parity_fixtures_pin_bridge_distinct_random_and_repeated_shapes() {
        let fixtures = persistent_fixtures(40, 0xD54);
        let summary: Vec<_> = fixtures
            .iter()
            .map(|fixture| {
                (
                    fixture.label,
                    fixture.routing.len(),
                    gathered_experts(&fixture.routing).len(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("bridge-u13", 6, 13),
                ("distinct-u36", 6, 36),
                ("random", 6, gathered_experts(&fixtures[2].routing).len()),
                ("repeated-u6", 6, 6),
            ]
        );
        for fixture in fixtures {
            assert!(fixture.routing.iter().all(|row| {
                row.len() == 6 && row.iter().enumerate().all(|(i, e)| !row[..i].contains(e))
            }));
        }
    }

    #[test]
    fn four_stripes_cover_gu_once_and_down_twice_per_thread() {
        let outputs_per_iteration = PERSISTENT_BLOCK as usize * VEC as usize;
        let stripe_iterations = |n: usize| {
            assert_eq!(n % 4, 0);
            let stripe_width = n / 4;
            assert_eq!(stripe_width % outputs_per_iteration, 0);
            stripe_width / outputs_per_iteration
        };
        assert_eq!(PERSISTENT_TASKS_PER_RECORD, 16);
        assert_eq!(stripe_iterations(INTER), 1);
        assert_eq!(stripe_iterations(HIDDEN), 2);
        assert_eq!(4 * (HIDDEN / 4), HIDDEN);
    }
}

/// Buffers one chain writes into. The `_m` chain and the per-row reference get
/// separate sets so a comparison can never read a value the other wrote.
struct Outs {
    gate: DevicePtr,
    up: DevicePtr,
    down: DevicePtr,
    sh_gate: DevicePtr,
    sh_up: DevicePtr,
    sh_down: DevicePtr,
}

impl Outs {
    fn alloc(gpu: &dyn GpuBackend, rows: usize, tokens: usize) -> Result<Self> {
        Ok(Outs {
            gate: gpu.alloc(rows * INTER * 2)?,
            up: gpu.alloc(rows * INTER * 2)?,
            down: gpu.alloc(rows * HIDDEN * 2)?,
            sh_gate: gpu.alloc(tokens * INTER * 2)?,
            sh_up: gpu.alloc(tokens * INTER * 2)?,
            sh_down: gpu.alloc(tokens * HIDDEN * 2)?,
        })
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let block: u32 = args.get(1).map_or(32, |s| s.parse().unwrap());
    let pool: usize = args.get(2).map_or(24, |s| s.parse().unwrap());
    let seed: u64 = args.get(3).map_or(0xD54, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0xD54)
    });
    let top_k: usize = args.get(4).map_or(6, |s| s.parse().unwrap());
    let overlap: i64 = args.get(5).map_or(-1, |s| s.parse().unwrap());
    let tokens: usize = args.get(6).map_or(2, |s| s.parse().unwrap());
    assert!(pool <= NUM_EXPERT_SLOTS && top_k <= pool);
    assert!(overlap <= top_k as i64, "overlap cannot exceed top_k");
    assert!(
        (1..=MROW_MAX as usize).contains(&tokens),
        "tokens must be in 1..={MROW_MAX} (the widest compiled _m entry point)"
    );
    // The pool must hold the forced-overlap construction below: row 0's whole
    // top_k, plus (top_k - overlap) globally fresh picks for each later row.
    let needed = top_k + (tokens - 1) * (top_k - overlap.max(0) as usize);
    assert!(
        overlap < 0 || pool >= needed,
        "pool {pool} too small for tokens={tokens} top_k={top_k} overlap={overlap}: need {needed}"
    );
    assert!(
        INTER % (block as usize * VEC as usize) == 0
            && HIDDEN % (block as usize * VEC as usize) == 0,
        "block {block} x VEC {VEC} must divide both N={INTER} and N={HIDDEN}"
    );

    // Narrowest compiled `_m` entry point that covers `tokens`, exactly as
    // `MoeLayer::splitk_m_t_handles` picks it.
    let mrow: u32 = if tokens <= 2 {
        2
    } else if tokens as u32 <= MROW_M6 {
        MROW_M6
    } else {
        MROW_MAX
    };

    println!(
        "=== V4 multi-row (_m) decode MoE: block={block} pool={pool} top_k={top_k} \
         split={SPLIT} vec={VEC} overlap={overlap} tokens={tokens} (m{mrow}) \
         seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    // One candidate row per verify slot.
    let a_bf16: Vec<u16> = (0..tokens * HIDDEN)
        .map(|_| f32_to_bf16(rng.unit() * 2.0 - 1.0))
        .collect();
    println!("generating {pool} experts x 3 projections ...");
    let gates: Vec<Wt> = (0..pool)
        .map(|_| Wt::routed(&mut rng, HIDDEN, INTER))
        .collect();
    let ups: Vec<Wt> = (0..pool)
        .map(|_| Wt::routed(&mut rng, HIDDEN, INTER))
        .collect();
    let downs: Vec<Wt> = (0..pool)
        .map(|_| Wt::routed(&mut rng, INTER, HIDDEN))
        .collect();
    let sh_gate = Wt::shared(&mut rng, HIDDEN, INTER);
    let sh_up = Wt::shared(&mut rng, HIDDEN, INTER);
    let sh_down = Wt::shared(&mut rng, INTER, HIDDEN);

    // Row 0 routes freely; every later row keeps `overlap` of row 0's picks and
    // draws the rest globally fresh, so the union is exactly
    // `top_k + (tokens-1)*(top_k-overlap)` (or is drawn independently when
    // overlap < 0). Every row's own picks stay distinct — that is the top-k
    // invariant the leader election relies on to guarantee an expert appears at
    // most once per row, hence at most `tokens` times overall.
    let mut routing: Vec<Vec<u32>> = Vec::with_capacity(tokens);
    let mut handed_out: Vec<u32> = Vec::new();
    for r in 0..tokens {
        let mut row: Vec<u32> = Vec::with_capacity(top_k);
        if r > 0 && overlap >= 0 {
            row.extend_from_slice(&routing[0][..overlap as usize]);
        }
        while row.len() < top_k {
            let e = (rng.next_u64() % pool as u64) as u32;
            if row.contains(&e) || (overlap >= 0 && r > 0 && handed_out.contains(&e)) {
                continue;
            }
            row.push(e);
            handed_out.push(e);
        }
        routing.push(row);
    }
    let mut union: Vec<u32> = Vec::new();
    for e in routing.iter().flatten() {
        if !union.contains(e) {
            union.push(*e);
        }
    }
    let distinct = union.len();
    for (r, row) in routing.iter().enumerate() {
        println!("routing row{r}={row:?}");
    }
    println!(
        "  distinct routed experts across {tokens} rows: {distinct} of {} slots",
        tokens * top_k
    );

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload(gpu, &u16s(&a_bf16))?;
    let gate_tbl = build_table(gpu, &gates)?;
    let up_tbl = build_table(gpu, &ups)?;
    let down_tbl = build_table(gpu, &downs)?;
    let sh = |w: &Wt| -> Result<(DevicePtr, DevicePtr)> {
        Ok((upload(gpu, &w.packed)?, upload(gpu, &w.scales)?))
    };
    let (sh_gate_p, sh_gate_s) = sh(&sh_gate)?;
    let (sh_up_p, sh_up_s) = sh(&sh_up)?;
    let (sh_down_p, sh_down_s) = sh(&sh_down)?;

    // Flat [tokens*top_k] indices for the `_m` chain; per-row [top_k] slices for
    // the reference chain — the same device memory, so the two chains cannot see
    // different routing.
    let flat: Vec<u32> = routing.iter().flatten().copied().collect();
    let idx_flat = upload(gpu, &u32s(&flat))?;
    let idx_row: Vec<DevicePtr> = (0..tokens)
        .map(|t| idx_flat.offset(t * top_k * 4))
        .collect();

    let rows = tokens * top_k;
    let m_out = Outs::alloc(gpu, rows, tokens)?;
    let ref_out = Outs::alloc(gpu, rows, tokens)?;
    let v2_out = Outs::alloc(gpu, rows, tokens)?;

    // Partials. Single-row: [2, SPLIT, top_k+1, INTER] / [SPLIT, top_k+1, HIDDEN].
    // Multi-row: `rows + tokens` accumulator rows instead of `top_k + 1`.
    let s = SPLIT as usize;
    let m_rows = rows + tokens;
    let partial_gu = gpu.alloc(2 * s * m_rows * INTER * 4)?;
    let partial_dn = gpu.alloc(s * m_rows * HIDDEN * 4)?;
    let persistent_out = Outs::alloc(gpu, rows, tokens)?;
    let persistent_gu = gpu.alloc(2 * s * m_rows * INTER * 4)?;
    let persistent_dn = gpu.alloc(s * m_rows * HIDDEN * 4)?;

    // ── single-row reference chain (the shipping `_v2s4` path), one row ──
    let ref_chain_stages = |row: usize, o: &Outs, stages: u32, st: u64| -> Result<()> {
        let a = a_ptr.offset(row * HIDDEN * 2);
        let idx = idx_row[row];
        let g_off = o.gate.offset(row * top_k * INTER * 2);
        let u_off = o.up.offset(row * top_k * INTER * 2);
        let d_off = o.down.offset(row * top_k * HIDDEN * 2);
        let shg = o.sh_gate.offset(row * INTER * 2);
        let shu = o.sh_up.offset(row * INTER * 2);
        let shd = o.sh_down.offset(row * HIDDEN * 2);
        if stages & 1 != 0 {
            KernelLaunch::new(
                gpu,
                gpu.kernel(MODULE, "moe_expert_gate_up_shared_t_e8m0_v2s4")?,
            )
            .grid([INTER as u32 / (block * VEC), top_k as u32 + 1, 2 * SPLIT])
            .block([block, 1, 1])
            .arg_ptr(a)
            .arg_ptr(gate_tbl.packed)
            .arg_ptr(gate_tbl.scales)
            .arg_ptr(gate_tbl.s2)
            .arg_ptr(g_off)
            .arg_ptr(up_tbl.packed)
            .arg_ptr(up_tbl.scales)
            .arg_ptr(up_tbl.s2)
            .arg_ptr(u_off)
            .arg_ptr(idx)
            .arg_ptr(sh_gate_p)
            .arg_ptr(sh_gate_s)
            .arg_f32(sh_gate.s2)
            .arg_ptr(shg)
            .arg_ptr(sh_up_p)
            .arg_ptr(sh_up_s)
            .arg_f32(sh_up.s2)
            .arg_ptr(shu)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(top_k as u32)
            .arg_ptr(partial_gu)
            .launch(st)?;
        }
        if stages & 2 != 0 {
            KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_gate_up_partial_finalize")?)
                .grid([(INTER as u32).div_ceil(block), top_k as u32 + 1, 2])
                .block([block, 1, 1])
                .arg_ptr(partial_gu)
                .arg_ptr(g_off)
                .arg_ptr(shg)
                .arg_ptr(u_off)
                .arg_ptr(shu)
                .arg_u32(INTER as u32)
                .arg_u32(top_k as u32)
                .arg_u32(SPLIT)
                .launch(st)?;
        }
        if stages & 4 != 0 {
            KernelLaunch::new(
                gpu,
                gpu.kernel(MODULE, "moe_expert_silu_down_shared_t_e8m0_v2s4")?,
            )
            .grid([HIDDEN as u32 / (block * VEC), top_k as u32 + 1, SPLIT])
            .block([block, 1, 1])
            .shared_mem(INTER as u32 * 4 / SPLIT)
            .arg_ptr(g_off)
            .arg_ptr(u_off)
            .arg_ptr(down_tbl.packed)
            .arg_ptr(down_tbl.scales)
            .arg_ptr(down_tbl.s2)
            .arg_ptr(d_off)
            .arg_ptr(idx)
            .arg_ptr(shg)
            .arg_ptr(shu)
            .arg_ptr(sh_down_p)
            .arg_ptr(sh_down_s)
            .arg_f32(sh_down.s2)
            .arg_ptr(shd)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(top_k as u32)
            .arg_ptr(partial_dn)
            .launch(st)?;
        }
        if stages & 8 != 0 {
            KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize")?)
                .grid([(HIDDEN as u32).div_ceil(block), top_k as u32 + 1, 1])
                .block([block, 1, 1])
                .arg_ptr(partial_dn)
                .arg_ptr(d_off)
                .arg_ptr(shd)
                .arg_u32(HIDDEN as u32)
                .arg_u32(top_k as u32)
                .arg_u32(SPLIT)
                .launch(st)?;
        }
        Ok(())
    };
    let ref_chain =
        |row: usize, o: &Outs, st: u64| -> Result<()> { ref_chain_stages(row, o, 0xF, st) };

    // ── multi-row chain: one launch pair covers `tokens` rows ──
    //
    // `stages` is a bitmask over {gate_up, gu-finalize, down, dn-finalize} so
    // the timing leg can attribute the win (or the loss) to a single kernel;
    // 0xF is the whole chain and the only setting that leaves valid outputs.
    let m_chain_stages =
        |mrow: u32, tokens: u32, partition: bool, o: &Outs, stages: u32, st: u64| -> Result<()> {
            // The partition/precompute arms (m1u, m*c*) only exist for the >2-row
            // verify; at tokens<=2 the chain falls back to the regular m1/m2
            // kernels even when `partition` is requested. Every downstream choice
            // that assumes precomputed activation — finalize variant, its grid.z
            // and activation arg, and the down kernel's dynamic smem — must key
            // off this EFFECTIVE flag, not the raw request. Keying the smem off
            // the raw `partition` while the regular (non-precomputed) down kernel
            // is what actually launches passes shared_mem=0 to a kernel that reads
            // `s_act_m` → CUDA_ERROR_ILLEGAL_ADDRESS.
            let partition = partition && tokens > 2;
            let regular_suffix = if mrow == 1 {
                "m1"
            } else if mrow == 2 {
                "m2"
            } else if mrow <= MROW_M6 {
                "m6"
            } else {
                "m8"
            };
            // The duplicated gate arm and the open-ended top down bucket must be
            // compiled at an MROW >= tokens, or their gather clamps and the rows
            // past MROW are never written. Everything below the top bucket is
            // width-independent: its multiplicity is bounded by the bucket.
            let wide = tokens > MROW_M6;
            let gu_suffixes: Vec<(&str, u32)> = if partition && tokens > 2 {
                vec![
                    ("m1u", 1),
                    if wide {
                        ("m8d", MROW_MAX)
                    } else {
                        ("m6d", MROW_M6)
                    },
                ]
            } else {
                vec![(regular_suffix, mrow)]
            };
            let dn_suffixes: Vec<(&str, u32)> = if partition && tokens > 2 {
                vec![
                    ("m1u", 1),
                    ("m2c2", 2),
                    ("m4c34", 4),
                    if wide {
                        ("m8c58", MROW_MAX)
                    } else {
                        ("m6c56", MROW_M6)
                    },
                ]
            } else {
                vec![(regular_suffix, mrow)]
            };
            let gu: Vec<(KernelHandle, u32)> = gu_suffixes
                .iter()
                .map(|(suffix, rows)| {
                    Ok((
                        gpu.kernel(
                            MODULE,
                            &format!("moe_expert_gate_up_shared_t_e8m0_{suffix}v2s4"),
                        )?,
                        *rows,
                    ))
                })
                .collect::<Result<_>>()?;
            let dn: Vec<(KernelHandle, u32)> = dn_suffixes
                .iter()
                .map(|(suffix, rows)| {
                    Ok((
                        gpu.kernel(
                            MODULE,
                            &format!("moe_expert_silu_down_shared_t_e8m0_{suffix}v2s4"),
                        )?,
                        *rows,
                    ))
                })
                .collect::<Result<_>>()?;
            let total_routed = tokens * top_k as u32;
            let fin_rows = total_routed + tokens;
            if stages & 1 != 0 {
                for &(gu, _) in &gu {
                    KernelLaunch::new(gpu, gu)
                        .grid([INTER as u32 / (block * VEC), total_routed + 1, 2 * SPLIT])
                        .block([block, 1, 1])
                        .arg_ptr(a_ptr)
                        .arg_ptr(gate_tbl.packed)
                        .arg_ptr(gate_tbl.scales)
                        .arg_ptr(gate_tbl.s2)
                        .arg_ptr(o.gate)
                        .arg_ptr(up_tbl.packed)
                        .arg_ptr(up_tbl.scales)
                        .arg_ptr(up_tbl.s2)
                        .arg_ptr(o.up)
                        .arg_ptr(idx_flat)
                        .arg_ptr(sh_gate_p)
                        .arg_ptr(sh_gate_s)
                        .arg_f32(sh_gate.s2)
                        .arg_ptr(o.sh_gate)
                        .arg_ptr(sh_up_p)
                        .arg_ptr(sh_up_s)
                        .arg_f32(sh_up.s2)
                        .arg_ptr(o.sh_up)
                        .arg_u32(INTER as u32)
                        .arg_u32(HIDDEN as u32)
                        .arg_u32(top_k as u32)
                        .arg_u32(tokens)
                        .arg_ptr(partial_gu)
                        .launch(st)?;
                }
            }
            if stages & 2 != 0 {
                let finalize = if partition {
                    "moe_gate_up_partial_finalize_m_act"
                } else {
                    "moe_gate_up_partial_finalize_m"
                };
                let mut launch = KernelLaunch::new(gpu, gpu.kernel(MODULE, finalize)?)
                    .grid([
                        (INTER as u32).div_ceil(block),
                        fin_rows,
                        if partition { 1 } else { 2 },
                    ])
                    .block([block, 1, 1])
                    .arg_ptr(partial_gu)
                    .arg_ptr(o.gate)
                    .arg_ptr(o.sh_gate)
                    .arg_ptr(o.up)
                    .arg_ptr(o.sh_up);
                if partition {
                    launch = launch.arg_ptr(partial_gu);
                }
                launch
                    .arg_u32(INTER as u32)
                    .arg_u32(total_routed)
                    .arg_u32(tokens)
                    .arg_u32(SPLIT)
                    .launch(st)?;
            }
            if stages & 4 != 0 {
                for &(dn, arm_mrow) in &dn {
                    KernelLaunch::new(gpu, dn)
                        .grid([HIDDEN as u32 / (block * VEC), total_routed + 1, SPLIT])
                        .block([block, 1, 1])
                        // One s_act k-slice per row the leader may gather: MROW, not
                        // `tokens` — the kernel strides slices by the compile-time MROW.
                        .shared_mem(if partition {
                            0
                        } else {
                            arm_mrow * INTER as u32 * 4 / SPLIT
                        })
                        .arg_ptr(o.gate)
                        .arg_ptr(o.up)
                        .arg_ptr(partial_gu)
                        .arg_ptr(down_tbl.packed)
                        .arg_ptr(down_tbl.scales)
                        .arg_ptr(down_tbl.s2)
                        .arg_ptr(o.down)
                        .arg_ptr(idx_flat)
                        .arg_ptr(o.sh_gate)
                        .arg_ptr(o.sh_up)
                        .arg_ptr(sh_down_p)
                        .arg_ptr(sh_down_s)
                        .arg_f32(sh_down.s2)
                        .arg_ptr(o.sh_down)
                        .arg_u32(HIDDEN as u32)
                        .arg_u32(INTER as u32)
                        .arg_u32(top_k as u32)
                        .arg_u32(tokens)
                        .arg_ptr(partial_dn)
                        .launch(st)?;
                }
            }
            if stages & 8 != 0 {
                KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize_m")?)
                    .grid([(HIDDEN as u32).div_ceil(block), fin_rows, 1])
                    .block([block, 1, 1])
                    .arg_ptr(partial_dn)
                    .arg_ptr(o.down)
                    .arg_ptr(o.sh_down)
                    .arg_u32(HIDDEN as u32)
                    .arg_u32(total_routed)
                    .arg_u32(tokens)
                    .arg_u32(SPLIT)
                    .launch(st)?;
            }
            Ok(())
        };
    let m_chain = |mrow: u32, tokens: u32, o: &Outs, st: u64| -> Result<()> {
        m_chain_stages(mrow, tokens, false, o, 0xF, st)
    };
    let p_chain = |mrow: u32, tokens: u32, o: &Outs, st: u64| -> Result<()> {
        m_chain_stages(mrow, tokens, true, o, 0xF, st)
    };

    // ── V2 wide-load chain (`ATLAS_MOE_SPLITK_V2` tier): `_m{mrow}v4s4` ──
    //
    // Same SPLIT as the incumbent, VEC=4 weight loads (128-byte warp requests
    // instead of 64) and smem-staged activations on the gate_up side. The
    // whole point of keeping SPLIT at 4 is that the tier must be BIT-IDENTICAL
    // to `_m{mrow}v2s4` — GATE V2 below enforces exactly that. The finalize
    // kernels and partial layout are the incumbent's.
    const VEC_V2: u32 = 4;
    let v2_chain_stages = |mrow: u32, tokens: u32, o: &Outs, stages: u32, st: u64| -> Result<()> {
        let suffix = if mrow == 1 {
            "m1"
        } else if mrow == 2 {
            "m2"
        } else if mrow <= MROW_M6 {
            "m6"
        } else {
            "m8"
        };
        let gu = gpu.kernel(
            MODULE,
            &format!("moe_expert_gate_up_shared_t_e8m0_{suffix}v4s4"),
        )?;
        let dn = gpu.kernel(
            MODULE,
            &format!("moe_expert_silu_down_shared_t_e8m0_{suffix}v4s4"),
        )?;
        let total_routed = tokens * top_k as u32;
        let fin_rows = total_routed + tokens;
        if stages & 1 != 0 {
            KernelLaunch::new(gpu, gu)
                .grid([INTER as u32 / (block * VEC_V2), total_routed + 1, 2 * SPLIT])
                .block([block, 1, 1])
                // One bf16 activation slice of K/SPLIT elements per row the
                // compiled entry may gather (MROW, not `tokens`).
                .shared_mem(mrow * (HIDDEN as u32 / SPLIT) * 2)
                .arg_ptr(a_ptr)
                .arg_ptr(gate_tbl.packed)
                .arg_ptr(gate_tbl.scales)
                .arg_ptr(gate_tbl.s2)
                .arg_ptr(o.gate)
                .arg_ptr(up_tbl.packed)
                .arg_ptr(up_tbl.scales)
                .arg_ptr(up_tbl.s2)
                .arg_ptr(o.up)
                .arg_ptr(idx_flat)
                .arg_ptr(sh_gate_p)
                .arg_ptr(sh_gate_s)
                .arg_f32(sh_gate.s2)
                .arg_ptr(o.sh_gate)
                .arg_ptr(sh_up_p)
                .arg_ptr(sh_up_s)
                .arg_f32(sh_up.s2)
                .arg_ptr(o.sh_up)
                .arg_u32(INTER as u32)
                .arg_u32(HIDDEN as u32)
                .arg_u32(top_k as u32)
                .arg_u32(tokens)
                .arg_ptr(partial_gu)
                .launch(st)?;
        }
        if stages & 2 != 0 {
            KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_gate_up_partial_finalize_m")?)
                .grid([(INTER as u32).div_ceil(block), fin_rows, 2])
                .block([block, 1, 1])
                .arg_ptr(partial_gu)
                .arg_ptr(o.gate)
                .arg_ptr(o.sh_gate)
                .arg_ptr(o.up)
                .arg_ptr(o.sh_up)
                .arg_u32(INTER as u32)
                .arg_u32(total_routed)
                .arg_u32(tokens)
                .arg_u32(SPLIT)
                .launch(st)?;
        }
        if stages & 4 != 0 {
            KernelLaunch::new(gpu, dn)
                .grid([HIDDEN as u32 / (block * VEC_V2), total_routed + 1, SPLIT])
                .block([block, 1, 1])
                // f32 s_act slices, exactly the incumbent's contract.
                .shared_mem(mrow * INTER as u32 * 4 / SPLIT)
                .arg_ptr(o.gate)
                .arg_ptr(o.up)
                .arg_ptr(partial_gu)
                .arg_ptr(down_tbl.packed)
                .arg_ptr(down_tbl.scales)
                .arg_ptr(down_tbl.s2)
                .arg_ptr(o.down)
                .arg_ptr(idx_flat)
                .arg_ptr(o.sh_gate)
                .arg_ptr(o.sh_up)
                .arg_ptr(sh_down_p)
                .arg_ptr(sh_down_s)
                .arg_f32(sh_down.s2)
                .arg_ptr(o.sh_down)
                .arg_u32(HIDDEN as u32)
                .arg_u32(INTER as u32)
                .arg_u32(top_k as u32)
                .arg_u32(tokens)
                .arg_ptr(partial_dn)
                .launch(st)?;
        }
        if stages & 8 != 0 {
            KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize_m")?)
                .grid([(HIDDEN as u32).div_ceil(block), fin_rows, 1])
                .block([block, 1, 1])
                .arg_ptr(partial_dn)
                .arg_ptr(o.down)
                .arg_ptr(o.sh_down)
                .arg_u32(HIDDEN as u32)
                .arg_u32(total_routed)
                .arg_u32(tokens)
                .arg_u32(SPLIT)
                .launch(st)?;
        }
        Ok(())
    };
    let v2_chain = |mrow: u32, tokens: u32, o: &Outs, st: u64| -> Result<()> {
        v2_chain_stages(mrow, tokens, o, 0xF, st)
    };
    let v2_available = gpu
        .kernel(MODULE, "moe_expert_gate_up_shared_t_e8m0_m6v4s4")
        .is_ok()
        && INTER % (block as usize * VEC_V2 as usize) == 0
        && HIDDEN % (block as usize * VEC_V2 as usize) == 0;

    // ── host-prebuilt persistent chain (Stage-0 microtest only) ──
    let persistent_chain_stages =
        |work: &PersistentWorkSet, o: &Outs, stages: u32, st: u64| -> Result<()> {
            let total_routed = (tokens * top_k) as u32;
            let fin_rows = total_routed + tokens as u32;
            let bucket_rows = [1u32, 2, 4, 6];
            if stages & 1 != 0 {
                for (bucket, rows_cap) in work.gu.iter().zip(bucket_rows) {
                    let Some(records) = bucket else { continue };
                    let name = format!("moe_expert_gate_up_t_e8m0_m{rows_cap}_persistent_v2s4");
                    KernelLaunch::new(gpu, gpu.kernel(MODULE, &name)?)
                        .grid([records.count * PERSISTENT_TASKS_PER_RECORD, 1, 1])
                        .block([PERSISTENT_BLOCK, 1, 1])
                        .arg_ptr(a_ptr)
                        .arg_ptr(records.ptr)
                        .arg_u32(records.count)
                        .arg_ptr(persistent_gu)
                        .arg_u32(INTER as u32)
                        .arg_u32(HIDDEN as u32)
                        .arg_u32(top_k as u32)
                        .arg_u32(total_routed)
                        .arg_u32(tokens as u32)
                        .launch(st)?;
                }
                let records = &work.shared_gu;
                KernelLaunch::new(
                    gpu,
                    gpu.kernel(MODULE, "moe_shared_gate_up_t_m6_persistent_v2s4")?,
                )
                .grid([records.count * PERSISTENT_TASKS_PER_RECORD, 1, 1])
                .block([PERSISTENT_BLOCK, 1, 1])
                .arg_ptr(a_ptr)
                .arg_ptr(records.ptr)
                .arg_u32(records.count)
                .arg_ptr(persistent_gu)
                .arg_u32(INTER as u32)
                .arg_u32(HIDDEN as u32)
                .arg_u32(top_k as u32)
                .arg_u32(total_routed)
                .arg_u32(tokens as u32)
                .launch(st)?;
            }
            if stages & 2 != 0 {
                KernelLaunch::new(
                    gpu,
                    gpu.kernel(MODULE, "moe_gate_up_partial_finalize_m_act")?,
                )
                .grid([(INTER as u32).div_ceil(PERSISTENT_BLOCK), fin_rows, 1])
                .block([PERSISTENT_BLOCK, 1, 1])
                .arg_ptr(persistent_gu)
                .arg_ptr(o.gate)
                .arg_ptr(o.sh_gate)
                .arg_ptr(o.up)
                .arg_ptr(o.sh_up)
                .arg_ptr(persistent_gu)
                .arg_u32(INTER as u32)
                .arg_u32(total_routed)
                .arg_u32(tokens as u32)
                .arg_u32(SPLIT)
                .launch(st)?;
            }
            if stages & 4 != 0 {
                for (bucket, rows_cap) in work.down.iter().zip(bucket_rows) {
                    let Some(records) = bucket else { continue };
                    let name = format!("moe_expert_silu_down_t_e8m0_m{rows_cap}_persistent_v2s4");
                    KernelLaunch::new(gpu, gpu.kernel(MODULE, &name)?)
                        .grid([records.count * PERSISTENT_TASKS_PER_RECORD, 1, 1])
                        .block([PERSISTENT_BLOCK, 1, 1])
                        .arg_ptr(persistent_gu)
                        .arg_ptr(records.ptr)
                        .arg_u32(records.count)
                        .arg_ptr(persistent_dn)
                        .arg_u32(HIDDEN as u32)
                        .arg_u32(INTER as u32)
                        .arg_u32(total_routed)
                        .arg_u32(tokens as u32)
                        .launch(st)?;
                }
                let records = &work.shared_down;
                KernelLaunch::new(
                    gpu,
                    gpu.kernel(MODULE, "moe_shared_silu_down_t_m6_persistent_v2s4")?,
                )
                .grid([records.count * PERSISTENT_TASKS_PER_RECORD, 1, 1])
                .block([PERSISTENT_BLOCK, 1, 1])
                .arg_ptr(persistent_gu)
                .arg_ptr(records.ptr)
                .arg_u32(records.count)
                .arg_ptr(persistent_dn)
                .arg_u32(HIDDEN as u32)
                .arg_u32(INTER as u32)
                .arg_u32(total_routed)
                .arg_u32(tokens as u32)
                .launch(st)?;
            }
            if stages & 8 != 0 {
                KernelLaunch::new(gpu, gpu.kernel(MODULE, "moe_down_partial_finalize_m")?)
                    .grid([(HIDDEN as u32).div_ceil(PERSISTENT_BLOCK), fin_rows, 1])
                    .block([PERSISTENT_BLOCK, 1, 1])
                    .arg_ptr(persistent_dn)
                    .arg_ptr(o.down)
                    .arg_ptr(o.sh_down)
                    .arg_u32(HIDDEN as u32)
                    .arg_u32(total_routed)
                    .arg_u32(tokens as u32)
                    .arg_u32(SPLIT)
                    .launch(st)?;
            }
            Ok(())
        };
    let persistent_chain = |work: &PersistentWorkSet, o: &Outs, st: u64| -> Result<()> {
        persistent_chain_stages(work, o, 0xF, st)
    };

    let read = |ptr: DevicePtr, count: usize| -> Result<Vec<u16>> {
        let mut buf = vec![0u8; count * 2];
        gpu.copy_d2h(ptr, &mut buf)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    };
    let read_raw = |ptr: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
        let mut buf = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut buf)?;
        Ok(buf)
    };
    let require_raw_equal =
        |case: &str, label: &str, a: DevicePtr, b: DevicePtr, bytes: usize| -> Result<()> {
            let (lhs, rhs) = (read_raw(a, bytes)?, read_raw(b, bytes)?);
            if let Some(offset) = lhs.iter().zip(&rhs).position(|(x, y)| x != y) {
                bail!(
                    "PERSISTENT PARITY FAILED {case} {label}: byte {offset}/{bytes}, \
                     control=0x{:02x} candidate=0x{:02x}",
                    lhs[offset],
                    rhs[offset]
                );
            }
            Ok(())
        };
    // Poison every output so an unwritten element cannot fake a match.
    let poison = |o: &Outs| -> Result<()> {
        let junk = |n: usize| -> Vec<u16> {
            (0..n)
                .map(|i| (0x7F80u16 ^ (i as u16)).rotate_left(3))
                .collect()
        };
        gpu.copy_h2d(&u16s(&junk(rows * INTER)), o.gate)?;
        gpu.copy_h2d(&u16s(&junk(rows * INTER)), o.up)?;
        gpu.copy_h2d(&u16s(&junk(rows * HIDDEN)), o.down)?;
        gpu.copy_h2d(&u16s(&junk(tokens * INTER)), o.sh_gate)?;
        gpu.copy_h2d(&u16s(&junk(tokens * INTER)), o.sh_up)?;
        gpu.copy_h2d(&u16s(&junk(tokens * HIDDEN)), o.sh_down)?;
        Ok(())
    };
    let poison_partial = |ptr: DevicePtr, words: usize, salt: u32| -> Result<()> {
        let junk: Vec<f32> = (0..words)
            .map(|i| f32::from_bits(0x7FC0_0000u32 ^ (i as u32).rotate_left(9) ^ salt))
            .collect();
        gpu.copy_h2d(&f32s(&junk), ptr)?;
        Ok(())
    };

    // ── GATE 1: MROW=1, one token, must be bit-identical to `_v2s4` ──
    poison(&ref_out)?;
    ref_chain(0, &ref_out, stream)?;
    poison(&m_out)?;
    m_chain(1, 1, &m_out, stream)?;
    gpu.synchronize(stream)?;
    {
        let mut bad = 0usize;
        for (label, a, b, n) in [
            ("gate", ref_out.gate, m_out.gate, top_k * INTER),
            ("up", ref_out.up, m_out.up, top_k * INTER),
            ("down", ref_out.down, m_out.down, top_k * HIDDEN),
            ("sh_gate", ref_out.sh_gate, m_out.sh_gate, INTER),
            ("sh_up", ref_out.sh_up, m_out.sh_up, INTER),
            ("sh_down", ref_out.sh_down, m_out.sh_down, HIDDEN),
        ] {
            let (x, y) = (read(a, n)?, read(b, n)?);
            let diff = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            if diff > 0 {
                let i = x.iter().zip(&y).position(|(p, q)| p != q).unwrap();
                println!(
                    "  GATE1 {label}: {diff}/{n} differ, first at [{i}] \
                     ref={:.6} m1={:.6}",
                    bf16_to_f32(x[i]),
                    bf16_to_f32(y[i])
                );
                bad += diff;
            }
        }
        if bad > 0 {
            bail!("GATE 1 FAILED: _m1v2s4 is not bit-identical to _v2s4 ({bad} elems)");
        }
        println!("  GATE 1 PASS: _m1v2s4 == _v2s4, bit-identical (all 6 outputs)");
    }

    // ── GATE 2: `_m{mrow}`, `tokens` rows, vs `tokens` independent passes ──
    poison(&ref_out)?;
    for r in 0..tokens {
        ref_chain(r, &ref_out, stream)?;
    }
    poison(&m_out)?;
    p_chain(mrow, tokens as u32, &m_out, stream)?;
    gpu.synchronize(stream)?;
    {
        // bf16 carries 8 mantissa bits; 1/64 is 3 ULP of headroom over the
        // reassociation the shared-row block-set merge can introduce.
        const REL_MAX: f32 = 1.0 / 64.0;
        let mut worst = 0.0f32;
        let mut worst_at = String::new();
        let mut exact = true;
        for (label, a, b, n) in [
            ("gate", ref_out.gate, m_out.gate, rows * INTER),
            ("up", ref_out.up, m_out.up, rows * INTER),
            ("down", ref_out.down, m_out.down, rows * HIDDEN),
            ("sh_gate", ref_out.sh_gate, m_out.sh_gate, tokens * INTER),
            ("sh_up", ref_out.sh_up, m_out.sh_up, tokens * INTER),
            ("sh_down", ref_out.sh_down, m_out.sh_down, tokens * HIDDEN),
        ] {
            let (x, y) = (read(a, n)?, read(b, n)?);
            for (i, (p, q)) in x.iter().zip(&y).enumerate() {
                if p != q {
                    exact = false;
                }
                let (fp, fq) = (bf16_to_f32(*p), bf16_to_f32(*q));
                let denom = fp.abs().max(fq.abs()).max(1e-6);
                let rel = (fp - fq).abs() / denom;
                if rel > worst {
                    worst = rel;
                    worst_at = format!("{label}[{i}] ref={fp:.6} m{mrow}={fq:.6}");
                }
            }
        }
        if worst > REL_MAX {
            bail!("GATE 2 FAILED: worst rel {worst:.5} > {REL_MAX:.5} at {worst_at}");
        }
        println!(
            "  GATE 2 PASS: partitioned gate m1u+{dup} / down multiplicity buckets == \
             {tokens}x _v2s4, worst rel {worst:.2e} \
             (bound {REL_MAX:.2e}){exact}",
            dup = if tokens as u32 > MROW_M6 { "m8d" } else { "m6d" },
            exact = if exact { ", bit-identical" } else { "" }
        );
    }

    // ── GATE V2: `_m{mrow}v4s4` must be BIT-IDENTICAL to `_m{mrow}v2s4` ──
    //
    // The V2 tier keeps SPLIT=4, so the split points and every per-output FMA
    // order are the incumbent's; VEC=4 only remaps thread→output and the smem
    // activation staging feeds the same bf16 bytes through the same
    // conversion. Any mismatch here is a bug, not reassociation — raw bits.
    if v2_available && tokens >= 2 {
        poison(&m_out)?;
        m_chain(mrow, tokens as u32, &m_out, stream)?;
        poison(&v2_out)?;
        v2_chain(mrow, tokens as u32, &v2_out, stream)?;
        gpu.synchronize(stream)?;
        let mut bad = 0usize;
        for (label, a, b, n) in [
            ("gate", m_out.gate, v2_out.gate, rows * INTER),
            ("up", m_out.up, v2_out.up, rows * INTER),
            ("down", m_out.down, v2_out.down, rows * HIDDEN),
            ("sh_gate", m_out.sh_gate, v2_out.sh_gate, tokens * INTER),
            ("sh_up", m_out.sh_up, v2_out.sh_up, tokens * INTER),
            ("sh_down", m_out.sh_down, v2_out.sh_down, tokens * HIDDEN),
        ] {
            let (x, y) = (read(a, n)?, read(b, n)?);
            let diff = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            if diff > 0 {
                let i = x.iter().zip(&y).position(|(p, q)| p != q).unwrap();
                println!(
                    "  GATE V2 {label}: {diff}/{n} differ, first at [{i}] \
                     v2s4={:.6} v4s4={:.6}",
                    bf16_to_f32(x[i]),
                    bf16_to_f32(y[i])
                );
                bad += diff;
            }
        }
        if bad > 0 {
            bail!("GATE V2 FAILED: _m{mrow}v4s4 is not bit-identical to _m{mrow}v2s4 ({bad} elems)");
        }
        println!("  GATE V2 PASS: _m{mrow}v4s4 == _m{mrow}v2s4, bit-identical (all 6 outputs)");
    } else if !v2_available {
        println!("  GATE V2 skipped: _v4s4 entries not compiled or block*4 doesn't divide N");
    }

    // ── timing ──
    let time = |f: &dyn Fn(u64) -> Result<()>| -> Result<f64> {
        for _ in 0..20 {
            f(stream)?;
        }
        gpu.synchronize(stream)?;
        let (mut e0, mut e1) = (0u64, 0u64);
        unsafe {
            cuEventCreate(&mut e0, 0);
            cuEventCreate(&mut e1, 0);
            cuEventRecord(e0, stream);
        }
        for _ in 0..ITERS {
            f(stream)?;
        }
        let mut ms = 0f32;
        unsafe {
            cuEventRecord(e1, stream);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        Ok(ms as f64 / ITERS as f64)
    };

    // Weight bytes a single row streams: (gate+up+down) routed + shared, packed
    // at 0.5 B/elem plus the scale table.
    let routed_b = |k: usize, n: usize| (k / 2 * n) + (k / GS_ROUTED * n);
    let shared_b = |k: usize, n: usize| (k / 2 * n) + (k / GS_SHARED * n);
    let per_row_bytes = (top_k * (2 * routed_b(HIDDEN, INTER) + routed_b(INTER, HIDDEN))
        + 2 * shared_b(HIDDEN, INTER)
        + shared_b(INTER, HIDDEN)) as f64;

    // Persistent Stage-0 is deliberately pinned to the production DSpark
    // bridge shape: six rows, top-6, and enough uploaded experts for the fully
    // distinct fixture. Other invocations retain the original `_m` oracle.
    if tokens == MROW_PERSIST as usize && top_k == 6 && pool >= 36 {
        println!();
        println!("=== persistent host-work Stage-0: raw-bit parity + timing ===");
        let gu_words = 2 * s * m_rows * INTER;
        let dn_words = s * m_rows * HIDDEN;
        let shared_records = [
            (sh_gate_p.0, sh_gate_s.0, sh_gate.s2),
            (sh_up_p.0, sh_up_s.0, sh_up.s2),
            (sh_down_p.0, sh_down_s.0, sh_down.s2),
        ];
        let mut bridge_go = false;
        let mut secondary_ok = true;
        for fixture in persistent_fixtures(pool, seed) {
            let flat_fixture: Vec<u32> = fixture.routing.iter().flatten().copied().collect();
            gpu.copy_h2d(&u32s(&flat_fixture), idx_flat)?;
            let union_count = gathered_experts(&fixture.routing).len();
            let (leader_gu, leader_down) = make_work(
                &fixture.routing,
                tokens,
                WorkOrder::Leader,
                &gate_tbl,
                &up_tbl,
                &down_tbl,
                shared_records,
            );
            let (pointer_gu, pointer_down) = make_work(
                &fixture.routing,
                tokens,
                WorkOrder::Pointer,
                &gate_tbl,
                &up_tbl,
                &down_tbl,
                shared_records,
            );
            let leader = upload_work_set(gpu, &leader_gu, &leader_down)?;
            let pointer = upload_work_set(gpu, &pointer_gu, &pointer_down)?;

            poison(&m_out)?;
            poison_partial(partial_gu, gu_words, 0x1357_0000)?;
            poison_partial(partial_dn, dn_words, 0x2468_0000)?;
            p_chain(MROW_PERSIST, MROW_PERSIST, &m_out, stream)?;
            poison(&persistent_out)?;
            poison_partial(persistent_gu, gu_words, 0x9ABC_0000)?;
            poison_partial(persistent_dn, dn_words, 0xDEF0_0000)?;
            persistent_chain(&leader, &persistent_out, stream)?;
            gpu.synchronize(stream)?;
            for (label, control, candidate, bytes) in [
                ("gate", m_out.gate, persistent_out.gate, rows * INTER * 2),
                ("up", m_out.up, persistent_out.up, rows * INTER * 2),
                ("down", m_out.down, persistent_out.down, rows * HIDDEN * 2),
                (
                    "shared-gate",
                    m_out.sh_gate,
                    persistent_out.sh_gate,
                    tokens * INTER * 2,
                ),
                (
                    "shared-up",
                    m_out.sh_up,
                    persistent_out.sh_up,
                    tokens * INTER * 2,
                ),
                (
                    "shared-down",
                    m_out.sh_down,
                    persistent_out.sh_down,
                    tokens * HIDDEN * 2,
                ),
                (
                    "gate-up-partial+act",
                    partial_gu,
                    persistent_gu,
                    gu_words * 4,
                ),
                ("down-partial", partial_dn, persistent_dn, dn_words * 4),
            ] {
                require_raw_equal(fixture.label, label, control, candidate, bytes)?;
            }

            poison(&persistent_out)?;
            poison_partial(persistent_gu, gu_words, 0x55AA_0000)?;
            poison_partial(persistent_dn, dn_words, 0xAA55_0000)?;
            persistent_chain(&pointer, &persistent_out, stream)?;
            gpu.synchronize(stream)?;
            for (label, control, candidate, bytes) in [
                (
                    "pointer-gate",
                    m_out.gate,
                    persistent_out.gate,
                    rows * INTER * 2,
                ),
                ("pointer-up", m_out.up, persistent_out.up, rows * INTER * 2),
                (
                    "pointer-down",
                    m_out.down,
                    persistent_out.down,
                    rows * HIDDEN * 2,
                ),
                (
                    "pointer-shared-gate",
                    m_out.sh_gate,
                    persistent_out.sh_gate,
                    tokens * INTER * 2,
                ),
                (
                    "pointer-shared-up",
                    m_out.sh_up,
                    persistent_out.sh_up,
                    tokens * INTER * 2,
                ),
                (
                    "pointer-shared-down",
                    m_out.sh_down,
                    persistent_out.sh_down,
                    tokens * HIDDEN * 2,
                ),
                (
                    "pointer-gate-up-partial+act",
                    partial_gu,
                    persistent_gu,
                    gu_words * 4,
                ),
                (
                    "pointer-down-partial",
                    partial_dn,
                    persistent_dn,
                    dn_words * 4,
                ),
            ] {
                require_raw_equal(fixture.label, label, control, candidate, bytes)?;
            }

            let t_control = time(&|st| p_chain(MROW_PERSIST, MROW_PERSIST, &m_out, st))?;
            let t_leader = time(&|st| persistent_chain(&leader, &persistent_out, st))?;
            let t_pointer = time(&|st| persistent_chain(&pointer, &persistent_out, st))?;
            let best = t_leader.min(t_pointer);
            let actual_bytes = (union_count
                * (2 * routed_b(HIDDEN, INTER) + routed_b(INTER, HIDDEN))
                + 2 * shared_b(HIDDEN, INTER)
                + shared_b(INTER, HIDDEN)) as f64;
            let actual_gbs = |elapsed_ms: f64| actual_bytes / (elapsed_ms * 1e-3) / 1e9;
            println!(
                "  {:<12} U={union_count:>2}: control {t_control:.4} ms | \
                 leader {t_leader:.4} ms ({:.1} GB/s) | pointer {t_pointer:.4} ms \
                 ({:.1} GB/s) | raw-bit PASS",
                fixture.label,
                actual_gbs(t_leader),
                actual_gbs(t_pointer)
            );
            if fixture.label == "bridge-u13" {
                bridge_go = actual_gbs(best) >= 213.0 && best <= 0.883 && best <= t_control * 0.65;
            }
            if matches!(fixture.label, "distinct-u36" | "repeated-u6") {
                secondary_ok &= best <= t_control * 1.03;
            }
        }
        gpu.copy_h2d(&u32s(&flat), idx_flat)?;
        let persistent_hard_go = bridge_go && secondary_ok;
        println!(
            "  PERSISTENT PERFORMANCE GATE: {} (bridge >=213 GB/s, <=0.883 ms, \
             >=35% faster; distinct/repeated <=3% regression)",
            if persistent_hard_go { "GO" } else { "NO-GO" }
        );
    } else {
        println!("  persistent Stage-0 skipped: run with [top_k]=6 [tokens]=6 [pool]>=36");
    }

    let t_ref = time(&|st| {
        for r in 0..tokens {
            ref_chain(r, &ref_out, st)?;
        }
        Ok(())
    })?;
    let t_m = time(&|st| m_chain(mrow, tokens as u32, &m_out, st))?;
    let t_p = time(&|st| p_chain(mrow, tokens as u32, &m_out, st))?;
    let t_v2 = if v2_available && tokens >= 2 {
        Some(time(&|st| v2_chain(mrow, tokens as u32, &v2_out, st))?)
    } else {
        None
    };

    // The reference reads every row's weights in full; the `_m` chain reads the
    // union, so its effective GB/s is quoted against the SAME nominal byte
    // count — the ratio, not the absolute, is the result.
    let nominal = tokens as f64 * per_row_bytes;
    let gbs = |t: f64| nominal / (t * 1e-3) / 1e9;
    println!();
    println!(
        "  {tokens}x _v2s4 (today, per-row verify): {t_ref:.4} ms  ({:.0} GB/s nominal)",
        gbs(t_ref)
    );
    println!(
        "  1x _m{mrow}v2s4 (dedup'd multi-row):     {t_m:.4} ms  ({:.0} GB/s nominal)  \
         [{:+.1}%]",
        gbs(t_m),
        (t_ref / t_m - 1.0) * 100.0
    );
    println!(
        "  partitioned + bucketed down:             {t_p:.4} ms  ({:.0} GB/s nominal)  \
         [{:+.1}% vs m{mrow}]",
        gbs(t_p),
        (t_m / t_p - 1.0) * 100.0
    );
    if let Some(t_v2) = t_v2 {
        println!(
            "  1x _m{mrow}v4s4 V2 (wide-load, staged):  {t_v2:.4} ms  ({:.0} GB/s nominal)  \
             [{:+.1}% vs m{mrow}v2s4]",
            gbs(t_v2),
            (t_m / t_v2 - 1.0) * 100.0
        );
    }
    // Ceiling: only the union of the rows' experts plus ONE shared expert has to
    // be read; the duplicate picks collapse.
    let union_bytes = (distinct * (2 * routed_b(HIDDEN, INTER) + routed_b(INTER, HIDDEN))
        + 2 * shared_b(HIDDEN, INTER)
        + shared_b(INTER, HIDDEN)) as f64;
    println!(
        "  byte-traffic ceiling at this routing: {:.3}x (union {:.2} MB vs {:.2} MB)",
        nominal / union_bytes,
        union_bytes / 1e6,
        nominal / 1e6
    );
    println!(
        "  verify MoE across {LAYERS:.0} layers: {:.2} ms -> {:.2} ms",
        t_ref * LAYERS,
        t_p * LAYERS
    );

    // Per-stage attribution. The two GEMVs carry all the weight traffic; the
    // finalizes only sum SPLIT float partials. If the dedup win lands on one
    // GEMV and not the other, that GEMV is the one to tune.
    println!();
    println!(
        "  stage                {:>10} {:>10} {:>9} {:>10} {:>9}",
        "per-row", "dedup", "gain", "v2", "v2 gain"
    );
    for (label, bit) in [
        ("gate_up GEMV", 1u32),
        ("gate_up finalize", 2),
        ("silu_down GEMV", 4),
        ("down finalize", 8),
    ] {
        let r = time(&|st| {
            for row in 0..tokens {
                ref_chain_stages(row, &ref_out, bit, st)?;
            }
            Ok(())
        })?;
        let m = time(&|st| m_chain_stages(mrow, tokens as u32, true, &m_out, bit, st))?;
        // The incumbent (non-partitioned) dedup chain is the V2 comparison
        // point — same kernels the engine launches without extra env flags.
        let mi = time(&|st| m_chain_stages(mrow, tokens as u32, false, &m_out, bit, st))?;
        if v2_available && tokens >= 2 {
            let v = time(&|st| v2_chain_stages(mrow, tokens as u32, &v2_out, bit, st))?;
            println!(
                "  {label:<20} {r:>8.4} ms {m:>8.4} ms {:>+8.1}% {v:>8.4} ms {:>+8.1}%",
                (r / m - 1.0) * 100.0,
                (mi / v - 1.0) * 100.0
            );
        } else {
            println!(
                "  {label:<20} {r:>8.4} ms {m:>8.4} ms {:>+8.1}%",
                (r / m - 1.0) * 100.0
            );
        }
    }
    Ok(())
}
