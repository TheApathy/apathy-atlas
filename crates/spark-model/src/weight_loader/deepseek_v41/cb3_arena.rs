// SPDX-License-Identifier: AGPL-3.0-only

//! The **resident CB3 expert arena** for DeepSeek-V4.1-Flash-Next.
//!
//! This is the thing `deepseek_v41.rs`'s hard stop named as missing: "GPU residency for
//! the 83 GB K154 expert pack". It allocates once and uploads the pack's resident prefix,
//! so the MoE forward reads device memory rather than a page cache.
//!
//! ## Resident, not streamed — and why that is a measurement, not a preference
//! At the served `packed_keep = 124` the pack is 71.7 GB over 40 layers, against ~102.9 GB
//! usable at `--gpu-mem 0.86` on this 119.7 GB box. It FITS, so there is no streaming or
//! RDMA tier here and none is wanted: a tier that is never short is pure overhead. Sizing
//! from the config's 384 experts instead would ask for 222.0 GB, and on GB10 a GPU
//! over-allocation takes the HOST down rather than just this process. That is why
//! [`ExpertPack::check_residency`] runs BEFORE the first `alloc`, not after.
//!
//! ## Twelve planes, not one blob
//! A CB3 layer shard is twelve separate expert-major tensors (`[154, rows, bytes_per_row]`),
//! so one expert lives at twelve distinct offsets and `bytes_per_expert` (14,454,784) is
//! their SUM, not a contiguous run. The arena mirrors that shape exactly: twelve device
//! allocations per layer, each expert-major over the resident slots. Flattening them into
//! one per-expert blob would have to re-derive twelve offsets on every access and would put
//! the `slot * bytes_per_expert` mistake back within reach.
//!
//! ## What consumes it
//! [`CB3_RECONSTRUCT_MODULE`] / [`CB3_RECONSTRUCT_FN`], the kernel at
//! `kernels/gb10/deepseek-v4.1/cb3/cb3_reconstruct_bf16.cu`, which is the production form
//! of the harness that measured rel_l2 2.58e-07 with a wrong-K-order control at 1.41.

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

use atlas_core::config::{CB3_TENSORS, Cb3Tensor, ExpertPack, ROUTED_EXPERTS, SERVED_PACKED_KEEP};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::deepseek_v41_pack::LayerShard;

/// Module name of the CB3 reconstruct kernel, as the build system derives it from the
/// file stem `cb3_reconstruct_bf16.cu`. Spelled ONCE, here.
pub const CB3_RECONSTRUCT_MODULE: &str = "cb3_reconstruct_bf16";
/// The `extern "C" __global__` entry point inside that module.
pub const CB3_RECONSTRUCT_FN: &str = "cb3_reconstruct_bf16";

/// Environment override for `packed_keep`.
///
/// `packed_keep` is a RUNTIME KNOB that is **not discoverable from the checkpoint** — the
/// Python engine takes it from `PACKED_KEEP="124"` in its entrypoint script. Anything that
/// reads only the model directory lands on the pack's 154 and is then quietly wrong in
/// production by ~25% of expert-weight traffic: a gap that never errors and surfaces only
/// as an unexplained throughput difference between engines. So it is explicit here, it
/// defaults to what production serves, and [`Cb3ExpertArena::load`] LOGS it.
pub const PACKED_KEEP_ENV: &str = "ATLAS_DSV41_PACKED_KEEP";

/// Fraction of *free* device memory the arena is allowed to claim.
///
/// The residency check is against this, not against total memory: the embedding, the dense
/// attention tensors and the KV cache are allocated around the arena, and a check that
/// ignores them would pass and then take the host down at the next allocation.
const DEFAULT_ARENA_BUDGET_FRACTION: f64 = 0.80;

/// Absolute device-memory headroom that must remain AFTER the arena, in bytes.
///
/// A fraction alone is not enough at full scale. At `packed_keep = 124` the arena is
/// 71.7 GB; with ~96 GB free the 0.80 fraction yields a 77 GB budget and the plan is
/// admitted with ~25 GB to spare — before the mmap'd shards (83 GB of file) start filling
/// page cache, and before the KV cache and activations are allocated. The kernel will evict
/// page cache under pressure, but "should evict" is not a guarantee, and on GB10 the
/// failure is not an `OutOfMemory` error: the pool is UNIFIED, so an over-allocation takes
/// the HOST down and kills every other process on the box. cgroup MemoryMax does not
/// contain GPU memory, so nothing else catches it either.
///
/// So the floor is absolute and checked independently of the fraction. 16 GB is chosen to
/// cover the KV cache, activations and transient staging with room for page cache the
/// kernel has not yet reclaimed. Override with [`ARENA_BUDGET_GB_ENV`] only when you have
/// measured the real headroom for the configuration in front of you.
const MIN_HEADROOM_BYTES: u64 = 16_000_000_000;
/// Override for the above, in bytes' worth of GB (e.g. `72.5`).
pub const ARENA_BUDGET_GB_ENV: &str = "ATLAS_DSV41_ARENA_BUDGET_GB";

/// One layer's twelve resident CB3 planes.
///
/// Each plane is `[packed_keep, rows, bytes_per_row]` in device memory, expert-major, in
/// the pack's slot order — the SAME order as the shard, so a slot index means the same
/// thing on disk and on the device.
pub struct Cb3LayerResidency {
    planes: [DevicePtr; 12],
    /// Bytes one expert occupies in each plane, in [`CB3_TENSORS`] order.
    strides: [usize; 12],
    layer: usize,
}

impl Cb3LayerResidency {
    /// Device base of one plane for one resident slot.
    ///
    /// Returns the address the reconstruct kernel takes for `lo` / `hi` / `cb` / `scale`.
    pub fn plane_ptr(&self, tensor: Cb3Tensor, slot: usize, packed_keep: usize) -> Result<DevicePtr> {
        ensure!(
            slot < packed_keep,
            "CB3 layer {}: slot {slot} is outside the resident prefix of {packed_keep}",
            self.layer
        );
        let index = tensor_index(tensor);
        Ok(DevicePtr(
            self.planes[index].0 + (slot as u64) * (self.strides[index] as u64),
        ))
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Slot-0 base of one plane and its per-slot stride in bytes, for kernels that address
    /// every resident expert of the layer themselves (the fused grouped GEMM).
    pub fn plane_base(&self, tensor: Cb3Tensor) -> (DevicePtr, u64) {
        let index = tensor_index(tensor);
        (self.planes[index], self.strides[index] as u64)
    }
}

/// The resident expert pack: 40 layers x 12 planes x `packed_keep` experts.
pub struct Cb3ExpertArena {
    layers: Vec<Cb3LayerResidency>,
    /// Per layer, the router allow-list over the 384-space id, PAIRED WITH ITS REAL LAYER
    /// INDEX. Applied BEFORE top-k, which is what makes non-residency a routing decision
    /// rather than a lookup failure.
    ///
    /// Keyed by `(layer, ...)` rather than by position because the single-layer validation
    /// arena holds one entry whose layer is NOT 0. A positional lookup there returns that
    /// entry for every index — a plausible wrong answer, which is the failure mode this
    /// whole module is written against. Found by `cb3_moe_oracle_microtest` erroring on
    /// layer 2 after `layer()` had already been fixed and its two siblings had not.
    routing_masks: Vec<(usize, Vec<bool>)>,
    /// Per layer, a 384-wide inverse table: routed id -> resident slot, or [`NO_SLOT`].
    ///
    /// [`ExpertPack::slot_of`] is a linear scan over <= 154 ids, which its own doc comment
    /// says is fine at load time and must not land on a hot path. Routing IS a hot path, so
    /// the table is built once here — the inverse table that comment asks for.
    slot_of_routed: Vec<(usize, Vec<i32>)>,
    packed_keep: usize,
    resident_bytes: u64,
}

/// `slot_of_routed` entry for an expert that is not resident in that layer.
pub const NO_SLOT: i32 = -1;

fn tensor_index(tensor: Cb3Tensor) -> usize {
    CB3_TENSORS
        .iter()
        .position(|candidate| *candidate == tensor)
        .expect("CB3_TENSORS is exhaustive over Cb3Tensor")
}

/// Resolve `packed_keep` from the environment, defaulting to what production serves.
pub fn resolve_packed_keep() -> Result<usize> {
    match std::env::var(PACKED_KEEP_ENV) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{PACKED_KEEP_ENV}={raw:?} is not an integer")),
        Err(_) => Ok(SERVED_PACKED_KEEP),
    }
}

impl Cb3ExpertArena {
    /// Allocate and upload the resident prefix of the CB3 pack.
    ///
    /// `pack_dir` is the `k154-cb3` directory. `pack` must already carry the intended
    /// `packed_keep` (see [`ExpertPack::parse`]).
    pub fn load(pack_dir: &Path, pack: &ExpertPack, gpu: &dyn GpuBackend) -> Result<Self> {
        let packed_keep = pack.packed_keep();
        let need = pack.resident_bytes();

        // BUDGET FIRST. Nothing is allocated until the whole plan is known to fit; a
        // partial upload that dies at layer 31 would already have taken the host with it.
        let budget = arena_budget_bytes(gpu)?;
        pack.check_residency(budget)?;

        // ABSOLUTE HEADROOM FLOOR, independent of the fraction above. See
        // MIN_HEADROOM_BYTES: on GB10 the memory pool is unified, so over-allocating does
        // not raise OutOfMemory — it takes the host down with every other process on it.
        let free = gpu.free_memory().context("CB3 arena: querying free device memory")? as u64;
        let remaining = free.saturating_sub(need);
        ensure!(
            remaining >= MIN_HEADROOM_BYTES,
            "DeepSeek-V4.1 CB3 arena REFUSED: {:.1} GB resident at packed_keep={} would leave \
             only {:.1} GB of the {:.1} GB free, below the {:.1} GB floor. GB10 memory is \
             UNIFIED — an over-allocation takes the HOST down, not just this process, and \
             cgroup MemoryMax does not contain GPU memory. The mmap'd shards also fill page \
             cache as they are read. Free memory first, lower packed_keep (currently {}), or \
             set {} deliberately after measuring.",
            need as f64 / 1e9,
            packed_keep,
            remaining as f64 / 1e9,
            free as f64 / 1e9,
            MIN_HEADROOM_BYTES as f64 / 1e9,
            packed_keep,
            ARENA_BUDGET_GB_ENV,
        );

        tracing::info!(
            "DeepSeek-V4.1 CB3 arena: packed_keep={} (pack holds {}, router space {}), \
             {} layers, {:.1} GB resident, budget {:.1} GB. packed_keep is a RUNTIME KNOB \
             not present in the checkpoint — set {} to change it.",
            packed_keep,
            pack.pack_experts(),
            ROUTED_EXPERTS,
            pack.num_layers(),
            need as f64 / 1e9,
            budget as f64 / 1e9,
            PACKED_KEEP_ENV,
        );

        let mut layers = Vec::with_capacity(pack.num_layers());
        let mut routing_masks = Vec::with_capacity(pack.num_layers());
        let mut slot_of_routed = Vec::with_capacity(pack.num_layers());
        let mut uploaded: u64 = 0;

        for layer in 0..pack.num_layers() {
            let shard = LayerShard::open(pack_dir, layer, pack)
                .with_context(|| format!("CB3 arena: opening layer {layer}"))?;

            let mut planes = [DevicePtr::NULL; 12];
            let mut strides = [0usize; 12];
            for (index, tensor) in CB3_TENSORS.iter().enumerate() {
                let stride = pack.slot_stride_in(*tensor) as usize;
                let bytes = stride
                    .checked_mul(packed_keep)
                    .context("CB3 plane extent overflow")?;
                planes[index] = gpu.alloc(bytes).with_context(|| {
                    format!(
                        "CB3 arena: allocating {:.1} MB for layer {layer} plane {}",
                        bytes as f64 / 1e6,
                        tensor.name()
                    )
                })?;
                strides[index] = stride;
            }

            // UPLOAD ONE PLANE AT A TIME, not one expert at a time.
            //
            // Each plane is expert-major, so the resident prefix (slots 0..packed_keep) is
            // ONE contiguous run inside it. That is 12 copies per layer instead of
            // packed_keep * 12 — at the served keep of 124, 12 instead of 1488, and 480
            // instead of 59,520 across the model.
            //
            // This is not a micro-optimisation: measured expert-by-expert, the upload ran
            // at 0.24-0.43 GB/s end-to-end, which extrapolates to ~28 minutes at --keep 124.
            // The per-copy overhead dominated; the bytes never did. dsv41-engram measured
            // this box's NVMe at ~8.5 GB/s single-threaded, so the disk was never the
            // constraint.
            //
            // CORRECTNESS IS NOT ASSUMED HERE. The residency microtest still reads every
            // byte back and compares it PER EXPERT via `LayerShard::expert`, which resolves
            // each expert's twelve offsets independently. So the batched write is verified
            // against the unbatched read: if this span arithmetic were wrong, the readback
            // would differ. The optimisation is checked by the control that already exists.
            let resident = pack.resident_ids(layer)?;
            ensure!(
                resident.len() == packed_keep,
                "CB3 layer {layer}: pack reports {} resident ids for packed_keep {packed_keep}",
                resident.len()
            );
            for (index, tensor) in CB3_TENSORS.iter().enumerate() {
                let span = shard.plane_span(*tensor, packed_keep)?;
                let expected = strides[index]
                    .checked_mul(packed_keep)
                    .context("CB3 plane span extent overflow")?;
                ensure!(
                    span.len() == expected,
                    "CB3 layer {layer} plane {}: span is {} bytes, expected {expected}",
                    tensor.name(),
                    span.len()
                );
                gpu.copy_h2d(span, planes[index])?;
                uploaded += span.len() as u64;
            }

            routing_masks.push((layer, pack.routing_mask(layer)?));
            let mut inverse = vec![NO_SLOT; ROUTED_EXPERTS];
            for (slot, &expert_id) in resident.iter().enumerate() {
                inverse[expert_id as usize] = slot as i32;
            }
            slot_of_routed.push((layer, inverse));
            layers.push(Cb3LayerResidency {
                planes,
                strides,
                layer,
            });

            if layer % 8 == 0 || layer + 1 == pack.num_layers() {
                tracing::info!(
                    "DeepSeek-V4.1 CB3 arena: layer {}/{} resident ({:.1} GB uploaded)",
                    layer + 1,
                    pack.num_layers(),
                    uploaded as f64 / 1e9,
                );
            }
        }

        // The upload must account for every byte the plan asked for. A short upload means
        // some plane was skipped, and a skipped plane decodes to whatever `alloc` left
        // behind — finite, correctly shaped, and meaningless.
        ensure!(
            uploaded == need,
            "CB3 arena uploaded {uploaded} bytes but the plan needs {need}"
        );

        Ok(Self {
            layers,
            routing_masks,
            slot_of_routed,
            packed_keep,
            resident_bytes: need,
        })
    }

    /// Make **one layer** resident, for validation against a per-layer oracle tap.
    ///
    /// The full arena is 71.7 GB at the served keep and needs a clean machine. A single
    /// layer is 1.79 GB, which is safe to run alongside anything — and it is all the
    /// `moe_routed` comparison needs, because that tap is per layer. Without this, checking
    /// the MoE output against the engine would have been gated behind the full-scale run.
    ///
    /// `layer(l)` is valid ONLY for the layer loaded; every other index errors rather than
    /// returning a plausible wrong residency.
    pub fn load_one_layer(
        pack_dir: &Path,
        pack: &ExpertPack,
        layer: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::load_layer_subset(pack_dir, pack, &[layer], gpu)
    }

    /// Make a SUBSET of layers resident (1.79 GB each at the served keep), for teacher-forced
    /// multi-layer validation without the 71.7 GB full-scale run. Lookups are keyed by the
    /// real layer index; any layer not listed errors.
    pub fn load_layer_subset(
        pack_dir: &Path,
        pack: &ExpertPack,
        wanted: &[usize],
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let packed_keep = pack.packed_keep();
        let per_layer = packed_keep as u64 * pack.bytes_per_expert();
        ensure!(!wanted.is_empty(), "CB3 layer subset is empty");
        for (i, layer) in wanted.iter().enumerate() {
            ensure!(*layer < pack.num_layers(), "CB3 pack has no layer {layer}");
            ensure!(!wanted[..i].contains(layer), "CB3 layer subset lists layer {layer} twice");
        }
        let need = per_layer * wanted.len() as u64;

        let budget = arena_budget_bytes(gpu)?;
        ensure!(
            need <= budget,
            "CB3 arena subset {wanted:?} needs {:.1} GB but the budget is {:.1} GB",
            need as f64 / 1e9,
            budget as f64 / 1e9
        );
        let free = gpu.free_memory().context("CB3 arena: querying free device memory")? as u64;
        ensure!(
            free.saturating_sub(need) >= MIN_HEADROOM_BYTES,
            "CB3 arena subset {wanted:?} would leave under the {:.1} GB headroom floor",
            MIN_HEADROOM_BYTES as f64 / 1e9
        );
        tracing::info!(
            "DeepSeek-V4.1 CB3 arena: LAYER SUBSET {wanted:?} at packed_keep={packed_keep}, \
             {:.2} GB. `layer()` errors for any other index.",
            need as f64 / 1e9,
        );

        let mut layers = Vec::with_capacity(wanted.len());
        let mut routing_masks = Vec::with_capacity(wanted.len());
        let mut slot_of_routed = Vec::with_capacity(wanted.len());
        let mut uploaded = 0u64;
        for &layer in wanted {
            let shard = LayerShard::open(pack_dir, layer, pack)
                .with_context(|| format!("CB3 arena subset: opening layer {layer}"))?;
            let mut planes = [DevicePtr::NULL; 12];
            let mut strides = [0usize; 12];
            for (index, tensor) in CB3_TENSORS.iter().enumerate() {
                let stride = pack.slot_stride_in(*tensor) as usize;
                let bytes = stride
                    .checked_mul(packed_keep)
                    .context("CB3 plane extent overflow")?;
                planes[index] = gpu.alloc(bytes)?;
                strides[index] = stride;
                let span = shard.plane_span(*tensor, packed_keep)?;
                ensure!(span.len() == bytes, "CB3 plane span disagrees with the stride table");
                gpu.copy_h2d(span, planes[index])?;
                uploaded += span.len() as u64;
            }
            let resident = pack.resident_ids(layer)?;
            let mut inverse = vec![NO_SLOT; ROUTED_EXPERTS];
            for (slot, &expert_id) in resident.iter().enumerate() {
                inverse[expert_id as usize] = slot as i32;
            }
            layers.push(Cb3LayerResidency { planes, strides, layer });
            routing_masks.push((layer, pack.routing_mask(layer)?));
            slot_of_routed.push((layer, inverse));
        }
        ensure!(uploaded == need, "subset upload moved {uploaded} bytes, planned {need}");
        Ok(Self {
            layers,
            routing_masks,
            slot_of_routed,
            packed_keep,
            resident_bytes: need,
        })
    }

    pub fn layer(&self, layer: usize) -> Result<&Cb3LayerResidency> {
        // Matched on the layer's OWN index, not its position in the vec. For the
        // single-layer validation arena those differ, and a positional lookup would return
        // layer 0's residency for any request — a plausible wrong answer.
        self.layers
            .iter()
            .find(|residency| residency.layer == layer)
            .with_context(|| {
                format!(
                    "CB3 arena has no layer {layer} (holds {:?})",
                    self.layers.iter().map(|r| r.layer).collect::<Vec<_>>()
                )
            })
    }

    /// The router allow-list for `layer`. Apply BEFORE top-k.
    pub fn routing_mask(&self, layer: usize) -> Result<&[bool]> {
        self.routing_masks
            .iter()
            .find(|(index, _)| *index == layer)
            .map(|(_, mask)| mask.as_slice())
            .with_context(|| format!("CB3 arena has no layer {layer}"))
    }

    /// Routed 384-space id -> resident slot, or [`NO_SLOT`]. O(1), safe on a hot path.
    pub fn slot_of(&self, layer: usize, expert_id: u32) -> Result<i32> {
        let table = self
            .slot_of_routed
            .iter()
            .find(|(index, _)| *index == layer)
            .map(|(_, table)| table)
            .with_context(|| format!("CB3 arena has no layer {layer}"))?;
        let slot = *table
            .get(expert_id as usize)
            .with_context(|| format!("Expert id {expert_id} is outside the {ROUTED_EXPERTS}-wide router space"))?;
        Ok(slot)
    }

    pub fn packed_keep(&self) -> usize {
        self.packed_keep
    }
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// How many bytes the arena may claim.
///
/// Explicit override wins; otherwise a fraction of FREE device memory, which already
/// accounts for whatever the embedding and dense tensors have taken.
fn arena_budget_bytes(gpu: &dyn GpuBackend) -> Result<u64> {
    if let Ok(raw) = std::env::var(ARENA_BUDGET_GB_ENV) {
        let gb: f64 = raw
            .trim()
            .parse()
            .with_context(|| format!("{ARENA_BUDGET_GB_ENV}={raw:?} is not a number"))?;
        if !(gb.is_finite() && gb > 0.0) {
            bail!("{ARENA_BUDGET_GB_ENV}={raw:?} must be finite and positive");
        }
        return Ok((gb * 1e9) as u64);
    }
    let free = gpu.free_memory().context("CB3 arena: querying free device memory")? as u64;
    Ok((free as f64 * DEFAULT_ARENA_BUDGET_FRACTION) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K\
/k154-cb3/manifest.json";

    fn served_pack() -> Option<ExpertPack> {
        let raw = std::fs::read_to_string(MANIFEST).ok()?;
        Some(ExpertPack::parse(&raw, SERVED_PACKED_KEEP).expect("manifest parses"))
    }

    /// The default must be what production serves, not what is on disk.
    ///
    /// These are different numbers (124 vs 154) and the difference never errors — it shows
    /// up only as ~25% more expert-weight traffic than the Python engine.
    #[test]
    fn packed_keep_defaults_to_what_production_serves() {
        if std::env::var(PACKED_KEEP_ENV).is_ok() {
            eprintln!("skipping: {PACKED_KEEP_ENV} is set in this environment");
            return;
        }
        assert_eq!(resolve_packed_keep().unwrap(), SERVED_PACKED_KEEP);
        assert_eq!(SERVED_PACKED_KEEP, 124);
    }

    /// The module and function names must be the ones the BUILD actually produces.
    ///
    /// The build derives a module name from the `.cu` file stem unless `KERNEL.toml`
    /// overrides it, and `cb3_reconstruct_bf16` is deliberately not overridden. Asserting
    /// the file exists ties this constant to the artifact rather than to my memory of it;
    /// a rename then fails here instead of at `gpu.kernel()` during a serve.
    #[test]
    fn the_reconstruct_kernel_names_match_a_file_the_build_compiles() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .to_path_buf();
        let cu = root
            .join("kernels/gb10/deepseek-v4.1/cb3")
            .join(format!("{CB3_RECONSTRUCT_MODULE}.cu"));
        assert!(cu.is_file(), "{} must exist", cu.display());
        let source = std::fs::read_to_string(&cu).expect("kernel source readable");
        assert!(
            source.contains(&format!("__global__ void {CB3_RECONSTRUCT_FN}(")),
            "{CB3_RECONSTRUCT_FN} must be defined in {}",
            cu.display()
        );
        assert!(
            source.contains("extern \"C\""),
            "the kernel must have C linkage or `gpu.kernel()` cannot find it"
        );
        // And the quant dir must be a real build target, or none of it compiles.
        assert!(
            root.join("kernels/gb10/deepseek-v4.1/MODEL.toml").is_file(),
            "without MODEL.toml the model dir is skipped entirely"
        );
        assert!(
            root.join("kernels/gb10/deepseek-v4.1/cb3/KERNEL.toml").is_file(),
            "without KERNEL.toml the quant dir is not a target"
        );
    }

    /// Sizing must come from `packed_keep`, never from the config's 384.
    ///
    /// NEGATIVE CONTROL: the same check against a 384-sized plan must FAIL. Without it this
    /// is a gate on a structurally-guaranteed input — the failure mode that hid three real
    /// bugs in this tree this week.
    #[test]
    fn residency_check_admits_the_served_plan_and_refuses_the_384_one() {
        let Some(pack) = served_pack() else {
            eprintln!("skipping: {MANIFEST} not present");
            return;
        };
        let need = pack.resident_bytes();
        // ~71.7 GB at keep=124 over 40 layers.
        assert!(
            (70e9..73e9).contains(&(need as f64)),
            "served plan is {:.1} GB, expected ~71.7",
            need as f64 / 1e9
        );
        // Admitted against the real ~102.9 GB usable.
        pack.check_residency(102_900_000_000).expect("served plan must fit");

        // THE CONTROL. Sizing by the router's 384 asks for ~222 GB and must be refused.
        let per_keep = need / SERVED_PACKED_KEEP as u64;
        let wrong = per_keep * ROUTED_EXPERTS as u64;
        assert!(
            (215e9..230e9).contains(&(wrong as f64)),
            "384-sized plan is {:.1} GB, expected ~222",
            wrong as f64 / 1e9
        );
        let refusal = pack
            .check_residency(102_900_000_000)
            .err()
            .map(|e| e.to_string());
        assert!(
            refusal.is_none(),
            "the SERVED plan must not be refused; got {refusal:?}"
        );
        // A plan that genuinely does not fit must be refused, with the shortfall named.
        let err = pack
            .check_residency(wrong / 4)
            .expect_err("a 55 GB budget cannot hold 71.7 GB")
            .to_string();
        assert!(err.contains("short by"), "refusal must name the shortfall: {err}");
    }

    /// Every per-layer lookup must match on the layer's OWN index, not its vec position.
    ///
    /// `load_one_layer` holds a single entry whose layer is not 0. `layer()` was fixed for
    /// this first; `routing_mask()` and `slot_of()` still indexed by position and answered
    /// layer 0 (or any index) with layer 2's tables — found only when the GPU microtest
    /// errored. This covers all three siblings with no GPU: the arena is built directly.
    ///
    /// NEGATIVE CONTROL built in: layers 0 and 1 are asked for too and MUST error. Under the
    /// positional bug, index 0 returned the only entry, so those asserts are the ones that
    /// would have caught it.
    #[test]
    fn single_layer_arena_lookups_match_the_real_layer_not_the_position() {
        let held = 2usize;
        let mut mask = vec![false; ROUTED_EXPERTS];
        let mut inverse = vec![NO_SLOT; ROUTED_EXPERTS];
        for (slot, id) in [7u32, 40, 301].into_iter().enumerate() {
            mask[id as usize] = true;
            inverse[id as usize] = slot as i32;
        }
        let arena = Cb3ExpertArena {
            layers: vec![Cb3LayerResidency {
                planes: [DevicePtr::NULL; 12],
                strides: [0; 12],
                layer: held,
            }],
            routing_masks: vec![(held, mask)],
            slot_of_routed: vec![(held, inverse)],
            packed_keep: 3,
            resident_bytes: 0,
        };

        assert_eq!(arena.layer(held).unwrap().layer(), held);
        assert!(arena.routing_mask(held).unwrap()[40]);
        assert_eq!(arena.slot_of(held, 301).unwrap(), 2);
        assert_eq!(arena.slot_of(held, 8).unwrap(), NO_SLOT);

        for absent in [0usize, 1, 3] {
            assert!(arena.layer(absent).is_err(), "layer({absent}) must error");
            assert!(arena.routing_mask(absent).is_err(), "routing_mask({absent}) must error");
            assert!(arena.slot_of(absent, 301).is_err(), "slot_of({absent}, ..) must error");
        }
    }

    /// The inverse table and the mask must agree with `ExpertPack::slot_of` exactly,
    /// including on the ids that are NOT resident.
    ///
    /// The point of the table is to take a linear scan off the routing path; a table that
    /// disagreed with the scan on absent ids would route a token to a real-but-wrong
    /// expert, which produces plausible output and no error.
    #[test]
    fn the_inverse_slot_table_agrees_with_the_pack_on_every_routed_id() {
        let Some(pack) = served_pack() else {
            eprintln!("skipping: {MANIFEST} not present");
            return;
        };
        let mut absent_seen = 0usize;
        for layer in 0..pack.num_layers() {
            let resident = pack.resident_ids(layer).unwrap();
            let mask = pack.routing_mask(layer).unwrap();
            let mut inverse = vec![NO_SLOT; ROUTED_EXPERTS];
            for (slot, &id) in resident.iter().enumerate() {
                inverse[id as usize] = slot as i32;
            }
            for id in 0..ROUTED_EXPERTS as u32 {
                match pack.slot_of(layer, id) {
                    Ok(slot) => {
                        assert_eq!(inverse[id as usize], slot as i32);
                        assert!(mask[id as usize], "resident id {id} must be routable");
                    }
                    Err(_) => {
                        assert_eq!(
                            inverse[id as usize], NO_SLOT,
                            "layer {layer} id {id}: the table must not invent a slot the \
                             pack refuses"
                        );
                        assert!(!mask[id as usize]);
                        absent_seen += 1;
                    }
                }
            }
        }
        // The absent branch must actually have been exercised: 384 - 124 per layer.
        assert_eq!(
            absent_seen,
            (ROUTED_EXPERTS - SERVED_PACKED_KEEP) * pack.num_layers(),
            "the not-resident path must be covered, or this test proves nothing"
        );
    }
}
