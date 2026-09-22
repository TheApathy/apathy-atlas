// SPDX-License-Identifier: AGPL-3.0-only
//! **The control the CB3 expert arena owed.**
//!
//! `Cb3ExpertArena::load` allocates device memory and uploads the CB3 expert pack. Until
//! this example ran, "the pack is resident" was a claim about code that compiled, not
//! about bytes that arrived — a distinction that has cost this project real debugging time
//! more than once. This reads every uploaded byte BACK off the device and compares it to
//! the mapped shard.
//!
//! ## What is checked
//! 1. **Byte-exact readback.** For every resident slot of every layer, all twelve planes
//!    are copied D2H and compared to the shard bytes the loader read. Not a checksum over
//!    the whole arena — a per-plane comparison, so a mismatch names the layer, slot and
//!    plane rather than just saying "something differs".
//! 2. **Slot ORDER.** The arena's slot `i` must hold the expert the pack says is at slot
//!    `i`. This is the failure that would not look like a failure: every byte present,
//!    every plane the right length, experts merely in the wrong places. The model would
//!    run and produce fluent, wrong tokens.
//! 3. **NEGATIVE CONTROL** (`--control`): re-runs check (1) against a DELIBERATELY WRONG
//!    expectation — one byte flipped in the middle of one plane. It MUST fail. A gate
//!    nobody has watched fail is not evidence, and a readback gate is especially easy to
//!    write in a way that cannot fail (comparing a buffer against itself).
//!
//! ## Why it defaults to a small `packed_keep`
//! The served `packed_keep = 124` is 71.7 GB and takes minutes plus the whole box. The
//! MECHANISM — offsets, strides, slot order, the twelve-planes layout — is identical at
//! any keep, so the default is 6 (`MIN_PACKED_KEEP`, ~3.5 GB) and runs in seconds. Pass
//! `--keep N` for a larger run; `--keep 124` is the real thing and is what a pre-serve
//! check should use.
//!
//! ```text
//! cargo run -p spark-model --release --example cb3_arena_residency_microtest \
//!   --features cuda,gpu-examples -- --keep 6
//! cargo run -p spark-model --release --example cb3_arena_residency_microtest \
//!   --features cuda,gpu-examples -- --keep 6 --control
//! ```
//!
//! ## The throughput number is NOT disk throughput — read this before quoting it
//! `Cb3ExpertArena::load` reads the shards through `Mmap` and issues one `copy_h2d` per
//! plane, SINGLE-THREADED. So the figure printed below is end-to-end (page-cache-or-NVMe
//! read + H2D copy), not an NVMe measurement, and it is page-cache dependent: a second run
//! over the same shards reads mostly from RAM.
//!
//! The engram lane measured this box's NVMe at **11.53 GB/s** (O_DIRECT, 4 MB blocks, 4
//! threads; cross-checked against /proc/diskstats). That makes the ceiling a DISCRIMINATOR
//! rather than just context: if this example reports more than 11.53 GB/s, the read cannot
//! have come off the device and the page cache served it. The run says so explicitly rather
//! than leaving a reader to quote a warm number as disk performance — which is exactly the
//! misattribution that produced a bogus figure elsewhere in this project today.
//!
//! Their tuning does NOT transfer here either, in either direction: the expert pack is
//! ~14.45 MB sequential records (bandwidth-bound, saturates ~4 threads, REGRESSES by 16),
//! while engram is 264 B random reads (latency-bound, wants ~128). This loader is currently
//! 1 thread and makes no O_DIRECT claim, so it has none of the alignment traps that come
//! with sharding — and if it is ever threaded, 4 is the starting point, not 128.
//!
//! ## Host memory
//! GB10 is unified: a GPU over-allocation takes the HOST down, not just this process, and
//! cgroup MemoryMax does not contain GPU memory. At `--keep 124` the arena is 71.7 GB of
//! device memory while the mmap'd shards also populate page cache, so the run samples
//! MemAvailable throughout and reports the low-water mark.
//!
//! Take the GPU lock first: append to `/home/flocka/atlas/.gb10-queue`, then `flock` on
//! `/home/flocka/atlas/.gb10.lock`.

use anyhow::{Context, Result, bail};
use std::path::Path;

use atlas_core::config::{CB3_TENSORS, ExpertPack, SERVED_PACKED_KEEP};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::deepseek_v41_pack::LayerShard;
use spark_model::weight_loader::deepseek_v41::cb3_arena::Cb3ExpertArena;

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const PACK_SUBDIR: &str = "k154-cb3";

/// This box's NVMe ceiling, measured by the engram lane with O_DIRECT at 4 MB blocks and
/// 4 threads, cross-checked against /proc/diskstats (15.93 GB reported vs 15.9 GB on the
/// device counter). Used here as a DISCRIMINATOR: a reported rate above this cannot have
/// come off the device.
const NVME_CEILING_GB_S: f64 = 11.53;

/// Host MemAvailable in bytes, or 0 if /proc/meminfo cannot be read.
fn mem_available_bytes() -> u64 {
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    meminfo
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix("MemAvailable:")?;
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            Some(kb * 1024)
        })
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let mut keep = 6usize;
    let mut control = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--keep" => {
                keep = args
                    .next()
                    .context("--keep needs a value")?
                    .parse()
                    .context("--keep must be an integer")?
            }
            "--control" => control = true,
            other => bail!("unknown argument {other}"),
        }
    }

    let pack_dir = Path::new(MODEL_DIR).join(PACK_SUBDIR);
    if !pack_dir.is_dir() {
        bail!("CB3 pack not present at {}", pack_dir.display());
    }
    let manifest = std::fs::read_to_string(pack_dir.join("manifest.json"))
        .context("reading the CB3 manifest")?;
    let pack = ExpertPack::parse(&manifest, keep)?;

    println!(
        "CB3 arena residency control: packed_keep={keep} (served is {SERVED_PACKED_KEEP}), \
         {} layers, {:.2} GB planned",
        pack.num_layers(),
        pack.resident_bytes() as f64 / 1e9
    );

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let free_before = gpu.free_memory()?;
    let mem_available_before = mem_available_bytes();

    // Sample host MemAvailable while the upload runs. GB10 is unified memory and an
    // over-allocation takes the HOST down, so the low-water mark is the number that says
    // whether --keep 124 is safe, and it cannot be recovered after the fact.
    let low_water = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
    let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let sampler = {
        let low_water = std::sync::Arc::clone(&low_water);
        let sampling = std::sync::Arc::clone(&sampling);
        std::thread::spawn(move || {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                let now = mem_available_bytes();
                low_water.fetch_min(now, std::sync::atomic::Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        })
    };

    let started = std::time::Instant::now();
    let arena = Cb3ExpertArena::load(&pack_dir, &pack, &gpu)?;
    let upload_secs = started.elapsed().as_secs_f64();
    sampling.store(false, std::sync::atomic::Ordering::Relaxed);
    let _ = sampler.join();
    let free_after = gpu.free_memory()?;

    // Device memory must ACTUALLY have moved. An arena that allocated nothing would sail
    // through a readback of its own empty expectations.
    let consumed = free_before.saturating_sub(free_after) as u64;
    let throughput = arena.resident_bytes() as f64 / 1e9 / upload_secs.max(1e-9);
    println!(
        "uploaded {:.2} GB in {:.1}s = {:.2} GB/s END-TO-END (mmap read + H2D, 1 thread); \
         device free {:.1} -> {:.1} GB (consumed {:.2} GB)",
        arena.resident_bytes() as f64 / 1e9,
        upload_secs,
        throughput,
        free_before as f64 / 1e9,
        free_after as f64 / 1e9,
        consumed as f64 / 1e9,
    );
    // Use the measured NVMe ceiling as a discriminator, not as decoration.
    if throughput > NVME_CEILING_GB_S {
        println!(
            "  NOTE: {throughput:.2} GB/s EXCEEDS this box's measured NVMe ceiling of \
             {NVME_CEILING_GB_S} GB/s (O_DIRECT, 4 MB blocks, 4 threads), so the shards were \
             served from PAGE CACHE, not the device. This is a warm number. Do NOT quote it \
             as disk or as cold-start performance."
        );
    } else {
        println!(
            "  {throughput:.2} GB/s is at or below the {NVME_CEILING_GB_S} GB/s NVMe ceiling, \
             so this run may have touched the device. Still END-TO-END, not a disk \
             measurement — it includes the H2D copy and is single-threaded."
        );
    }
    let low = low_water.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  host MemAvailable {:.1} GB before -> {:.1} GB low-water during upload",
        mem_available_before as f64 / 1e9,
        low as f64 / 1e9,
    );
    if low < 10_000_000_000 {
        println!(
            "  WARNING: host MemAvailable fell below 10 GB. On GB10 an over-allocation takes \
             the HOST down, not just this process. Do not scale this run up without headroom."
        );
    }
    if consumed < arena.resident_bytes() / 2 {
        bail!(
            "device free memory fell by only {:.2} GB for a {:.2} GB arena — the allocation \
             did not happen as claimed",
            consumed as f64 / 1e9,
            arena.resident_bytes() as f64 / 1e9
        );
    }

    // ---------------------------------------------------------------- readback
    let mut compared_bytes: u64 = 0;
    let mut planes_compared: u64 = 0;
    let mut scratch: Vec<u8> = Vec::new();

    for layer in 0..pack.num_layers() {
        let shard = LayerShard::open(&pack_dir, layer, &pack)?;
        let residency = arena.layer(layer)?;
        let resident = pack.resident_ids(layer)?;

        for (slot, &expert_id) in resident.iter().enumerate() {
            // Slot ORDER: the pack must agree that this expert lives at this slot.
            let pack_slot = pack.slot_of(layer, expert_id)?;
            if pack_slot != slot {
                bail!(
                    "layer {layer}: expert {expert_id} is arena slot {slot} but the pack \
                     resolves it to {pack_slot} — slot order disagrees"
                );
            }
            let bytes = shard.expert(&pack, expert_id)?;

            for tensor in CB3_TENSORS {
                let want = bytes.plane(tensor);
                let device = residency.plane_ptr(tensor, slot, keep)?;
                scratch.clear();
                scratch.resize(want.len(), 0);
                gpu.copy_d2h(device, &mut scratch)?;

                // The control corrupts the EXPECTATION, not the device, so the arena is
                // never left in a bad state and the run can continue afterwards.
                let mut expected = want.to_vec();
                if control && layer == 0 && slot == 0 && tensor == CB3_TENSORS[0] {
                    let middle = expected.len() / 2;
                    expected[middle] ^= 0x01;
                    println!(
                        "  [control] flipped one bit at byte {middle} of layer 0 slot 0 \
                         plane {} — this comparison MUST fail",
                        tensor.name()
                    );
                }

                if scratch != expected {
                    let first = scratch
                        .iter()
                        .zip(&expected)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    let differing =
                        scratch.iter().zip(&expected).filter(|(a, b)| a != b).count();
                    if control {
                        println!(
                            "CONTROL FAILED AS REQUIRED: layer {layer} slot {slot} plane {} \
                             differs in {differing} of {} bytes, first at {first} \
                             (device {:#04x} vs expected {:#04x})",
                            tensor.name(),
                            expected.len(),
                            scratch[first],
                            expected[first],
                        );
                        println!(
                            "\nThe readback comparison CAN fail. The PASS below is therefore \
                             evidence, not a structural guarantee."
                        );
                        return Ok(());
                    }
                    bail!(
                        "layer {layer} slot {slot} (expert {expert_id}) plane {}: {differing} \
                         of {} bytes differ, first at offset {first} (device {:#04x} vs shard \
                         {:#04x})",
                        tensor.name(),
                        expected.len(),
                        scratch[first],
                        expected[first],
                    );
                }
                compared_bytes += want.len() as u64;
                planes_compared += 1;
            }
        }
        if layer % 8 == 0 || layer + 1 == pack.num_layers() {
            println!(
                "  layer {}/{} verified ({:.2} GB read back)",
                layer + 1,
                pack.num_layers(),
                compared_bytes as f64 / 1e9
            );
        }
    }

    if control {
        bail!(
            "CONTROL DID NOT FIRE. A deliberately corrupted expectation compared EQUAL, so \
             this readback gate cannot fail and proves nothing. Check that the comparison \
             is not against the device buffer itself."
        );
    }

    if compared_bytes != arena.resident_bytes() {
        bail!(
            "read back {compared_bytes} bytes but the arena holds {} — the comparison did \
             not cover the whole pack",
            arena.resident_bytes()
        );
    }

    println!(
        "\nPASS: {:.2} GB byte-exact across {planes_compared} planes \
         ({} layers x {keep} experts x 12), slot order confirmed against the pack.",
        compared_bytes as f64 / 1e9,
        pack.num_layers(),
    );
    println!(
        "Run again with --control to watch the comparison fail; a readback gate that has \
         not been watched fail is not evidence."
    );
    Ok(())
}
