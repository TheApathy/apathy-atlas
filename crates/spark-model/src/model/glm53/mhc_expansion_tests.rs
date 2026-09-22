// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::gguf::{GgmlType, GgufDeviceTensor};

use super::*;
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrix, Glm53HyperBranchWeights};

const BASE: u64 = 0x2_0000_0000;
/// Distinct from BASE so a test can tell an expanded slot from a source weight.
const WEIGHT_BASE: u64 = 0x4_0000_0000;

/// Records the (source, destination) pair of every dequantization so a test can
/// prove each layer-branch went to its own slot rather than trusting a count.
struct RecordingGpu {
    inner: MockGpuBackend,
    launches: Mutex<Vec<(u64, u64)>>,
}

impl RecordingGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            launches: Mutex::new(Vec::new()),
        }
    }
    fn pairs(&self) -> Vec<(u64, u64)> {
        self.launches.lock().unwrap().clone()
    }
}

impl GpuBackend for RecordingGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared: u32,
        _stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        // The q8->f32 kernel takes (source, destination) as its first two args.
        // SAFETY: the launch builder wrote two DevicePtr-sized args here.
        let read = |i: usize| -> u64 { unsafe { *(params[i] as *const u64) } };
        self.launches.lock().unwrap().push((read(0), read(1)));
        Ok(())
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _module: &str, _name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
}

/// Q8_0 `[16384, 24]`, the on-disk shape of `hc_*_fn.weight`.
fn function_tensor(address: u64) -> GgufDeviceTensor {
    GgufDeviceTensor {
        ptr: DevicePtr(address),
        dimensions: vec![16384, 24],
        ggml_type: GgmlType::Q8_0,
        byte_len: 16384 * 24 / 32 * 34_usize,
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(
            GgmlType::Q8_0,
            &[16384, 24],
            16384 * 24 / 32 * 34_usize,
        )
        .expect("test tensor slack"),
    }
}

fn f32_tensor(address: u64, elements: u64) -> GgufDeviceTensor {
    GgufDeviceTensor {
        ptr: DevicePtr(address),
        dimensions: vec![elements],
        ggml_type: GgmlType::F32,
        byte_len: (elements * 4) as usize,
        // Rank-1 F32: never an MMQ weight, so the slack is zero -- routed
        // through the loader's own helper rather than hardcoding 0.
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(
            GgmlType::F32,
            &[elements],
            (elements * 4) as usize,
        )
        .expect("F32 fixture slack"),
    }
}

fn branch(slot: u64) -> Glm53HyperBranchWeights {
    let stride = 0x10_0000u64;
    Glm53HyperBranchWeights {
        function: Glm53GgufMatrix::new(&function_tensor(WEIGHT_BASE + slot * stride)).unwrap(),
        base: Glm53GgufF32::new(
            &f32_tensor(WEIGHT_BASE + stride * 1000 + slot * 256, 24),
            &[24],
        )
        .unwrap(),
        scale: Glm53GgufF32::new(
            &f32_tensor(WEIGHT_BASE + stride * 2000 + slot * 256, 3),
            &[3],
        )
        .unwrap(),
    }
}

fn refs(layers: &[Glm53HyperWeights]) -> Vec<&Glm53HyperWeights> {
    layers.iter().collect()
}

fn hyper_layers(count: usize) -> Vec<Glm53HyperWeights> {
    (0..count)
        .map(|layer| Glm53HyperWeights {
            attention: branch(layer as u64 * 2),
            ffn: branch(layer as u64 * 2 + 1),
        })
        .collect()
}

#[test]
fn slots_tile_the_region_exactly_and_never_overlap() {
    let expanded = Glm53MhcExpanded::bind(DevicePtr(BASE), GLM53_MHC_EXPANDED_F32_BYTES).unwrap();
    let mut spans = Vec::new();
    for layer in 0..45 {
        for branch in [Glm53HyperBranch::Attention, Glm53HyperBranch::Ffn] {
            let slot = expanded.slot(layer, branch).unwrap();
            spans.push((slot.ptr.0, slot.ptr.0 + slot.bytes as u64));
        }
    }
    assert_eq!(spans.len(), 90);
    spans.sort();
    // Contiguous, in order, and exactly filling the reserved region.
    assert_eq!(spans[0].0, BASE);
    assert_eq!(spans[89].1, BASE + GLM53_MHC_EXPANDED_F32_BYTES);
    for pair in spans.windows(2) {
        assert_eq!(pair[0].1, pair[1].0, "mHC slots must tile without gaps");
    }
}

#[test]
fn layer_45_and_out_of_range_layers_have_no_mhc_slot() {
    let expanded = Glm53MhcExpanded::bind(DevicePtr(BASE), GLM53_MHC_EXPANDED_F32_BYTES).unwrap();
    // Layer 45 is the NextN block and carries no hyper connections.
    assert!(expanded.slot(45, Glm53HyperBranch::Attention).is_err());
    assert!(expanded.slot(1_000, Glm53HyperBranch::Ffn).is_err());
    assert!(expanded.slot(44, Glm53HyperBranch::Ffn).is_ok());
}

#[test]
fn bind_refuses_null_misaligned_or_undersized_regions() {
    assert!(Glm53MhcExpanded::bind(DevicePtr(0), GLM53_MHC_EXPANDED_F32_BYTES).is_err());
    assert!(Glm53MhcExpanded::bind(DevicePtr(BASE + 1), GLM53_MHC_EXPANDED_F32_BYTES).is_err());
    assert!(Glm53MhcExpanded::bind(DevicePtr(BASE), GLM53_MHC_EXPANDED_F32_BYTES - 1).is_err());
}

/// Every layer-branch must be dequantized into its own slot. A skipped branch
/// leaves zeroed mixing coefficients, which is fluent-wrong-output, not a crash.
#[test]
fn expansion_covers_every_layer_branch_into_a_distinct_slot() {
    let gpu = RecordingGpu::new();
    let kernel = GgmlQ8F32Kernel::load(&gpu).unwrap();
    let expanded = Glm53MhcExpanded::bind(DevicePtr(BASE), GLM53_MHC_EXPANDED_F32_BYTES).unwrap();
    let layers = hyper_layers(45);

    let refs: Vec<&Glm53HyperWeights> = layers.iter().collect();
    let launches = expanded.expand(&gpu, &kernel, &refs, 0).unwrap();
    assert_eq!(launches, 90);

    let pairs = gpu.pairs();
    assert_eq!(pairs.len(), 90);

    let sources: std::collections::BTreeSet<_> = pairs.iter().map(|(s, _)| *s).collect();
    let destinations: std::collections::BTreeSet<_> = pairs.iter().map(|(_, d)| *d).collect();
    assert_eq!(sources.len(), 90, "each branch must read its own weight");
    assert_eq!(destinations.len(), 90, "each branch must own its slot");

    // Destinations must be exactly the 90 slots, in schedule order.
    for (index, (_, destination)) in pairs.iter().enumerate() {
        let layer = index / 2;
        let branch = if index % 2 == 0 {
            Glm53HyperBranch::Attention
        } else {
            Glm53HyperBranch::Ffn
        };
        assert_eq!(
            *destination,
            expanded.slot(layer, branch).unwrap().ptr.0,
            "layer {layer} {branch:?} went to the wrong slot"
        );
    }
}

#[test]
fn expansion_refuses_a_wrong_layer_count_or_a_mis_sized_function() {
    let gpu = RecordingGpu::new();
    let kernel = GgmlQ8F32Kernel::load(&gpu).unwrap();
    let expanded = Glm53MhcExpanded::bind(DevicePtr(BASE), GLM53_MHC_EXPANDED_F32_BYTES).unwrap();

    assert!(
        expanded
            .expand(&gpu, &kernel, &refs(&hyper_layers(44)), 0)
            .is_err()
    );
    assert!(
        expanded
            .expand(&gpu, &kernel, &refs(&hyper_layers(46)), 0)
            .is_err()
    );
    assert!(
        gpu.pairs().is_empty(),
        "a refused expansion must not launch"
    );
}
