// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

const SOURCE: &str = include_str!("b1t1_bootstrap.rs");

#[path = "b1t1_bootstrap_sha256.rs"]
mod source_sha256;

fn at(ptr: u64, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: DevicePtr(ptr),
        bytes,
    }
}

fn valid_buffers(token_ids_u32: DevicePtr) -> Glm53B1T1BootstrapBuffers {
    let plan = Glm53B1T1BootstrapPlan::new(17).unwrap();
    Glm53B1T1BootstrapBuffers {
        source_q5_k: at(0x2000_0000, plan.embedding.source_bytes),
        token_ids_u32: at(token_ids_u32.0, plan.embedding.token_ids_bytes),
        hidden_bf16: at(0x4000_0000, plan.embedding.destination_bytes),
        streams_bf16: at(0x5000_0000, plan.hyper.streams_bytes),
    }
}

fn kernels(gpu: &dyn GpuBackend) -> (GgmlQ5EmbeddingKernel, Glm53HyperKernels) {
    (
        GgmlQ5EmbeddingKernel::load(gpu).unwrap(),
        Glm53HyperKernels::load(gpu).unwrap(),
    )
}

fn copy_receipt(receipt: &Glm53B1T1BootstrapReceipt) -> Glm53B1T1BootstrapReceipt {
    Glm53B1T1BootstrapReceipt {
        owner_generation: receipt.owner_generation,
        transaction_nonce: receipt.transaction_nonce,
        token_id: receipt.token_id,
        stream: receipt.stream,
        buffer_ranges: receipt.buffer_ranges,
        stage: receipt.stage,
        kernel_launches: receipt.kernel_launches,
    }
}

#[test]
fn exact_plan_pins_b1t1_q5_and_hc4_geometry() {
    let plan = Glm53B1T1BootstrapPlan::new(154_879).unwrap();
    assert_eq!((plan.embedding.tokens, plan.hyper.tokens), (1, 1));
    assert_eq!(plan.embedding.vocab, 154_880);
    assert_eq!(plan.embedding.hidden, 4_096);
    assert_eq!(plan.embedding.source_bytes, 436_142_080);
    assert_eq!(plan.embedding.token_ids_bytes, 4);
    assert_eq!(plan.embedding.destination_bytes, 8_192);
    assert_eq!(plan.hyper.hidden_bytes, 8_192);
    assert_eq!(plan.hyper.streams_bytes, 65_536);
    assert_eq!(plan.kernel_launches, 2);
    assert!(Glm53B1T1BootstrapPlan::new(154_880).is_err());
}

#[test]
fn valid_execution_writes_exact_token_then_launches_gather_and_expand() {
    let gpu = MockGpuBackend::new();
    let token_ptr = gpu.alloc(4).unwrap();
    let (embedding, hyper) = kernels(&gpu);
    let mut executor = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
    let plan = Glm53B1T1BootstrapPlan::new(0x10203).unwrap();
    let buffers = valid_buffers(token_ptr);
    let receipt = executor.execute(&gpu, plan, buffers, 9).unwrap();
    assert_eq!(gpu.read_alloc(token_ptr).unwrap(), 0x10203u32.to_le_bytes());
    assert_eq!(gpu.launch_count(), 2);
    assert_eq!(receipt.token_id(), 0x10203);
    assert_eq!(receipt.stream(), 9);
    assert_ne!(receipt.owner_generation(), 0);
    assert_ne!(receipt.transaction_nonce(), 0);
    assert_eq!(
        receipt
            .buffer_ranges()
            .map(|(start, end)| (start, end - start)),
        [
            (buffers.source_q5_k.ptr.0, 436_142_080),
            (buffers.token_ids_u32.ptr.0, 4),
            (buffers.hidden_bf16.ptr.0, 8_192),
            (buffers.streams_bf16.ptr.0, 65_536),
        ]
    );
    assert_eq!(
        format!("{buffers:?}"),
        "Glm53B1T1BootstrapBuffers { source_q5_k_bytes: 436142080, token_ids_u32_bytes: 4, hidden_bf16_bytes: 8192, streams_bf16_bytes: 65536 }"
    );
    assert_eq!(receipt.stage(), Glm53B1T1BootstrapStage::MhcExpandEnqueued);
    assert_eq!(receipt.kernel_launches(), 2);
    executor.consume_receipt(receipt).unwrap();
}

#[test]
fn checked_identity_rejects_corruption_stale_replay_and_cross_owner() {
    let gpu = MockGpuBackend::new();
    let token_ptr = gpu.alloc(4).unwrap();
    let (embedding, hyper) = kernels(&gpu);
    let mut first = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
    let mut second = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
    let plan = Glm53B1T1BootstrapPlan::new(17).unwrap();
    let buffers = valid_buffers(token_ptr);
    let receipt = first.execute(&gpu, plan, buffers, 7).unwrap();
    let stale = copy_receipt(&receipt);
    let mutations: &[fn(&mut Glm53B1T1BootstrapReceipt)] = &[
        |value: &mut Glm53B1T1BootstrapReceipt| value.owner_generation = 0,
        |value| value.transaction_nonce = 0,
        |value| value.transaction_nonce += 1,
        |value| value.token_id += 1,
        |value| value.stream += 1,
        |value| value.buffer_ranges[2].1 -= 2,
        |value| value.kernel_launches = 1,
    ];
    let launches = gpu.launch_count();
    assert!(first.execute(&gpu, plan, buffers, 7).is_err());
    assert_eq!(gpu.launch_count(), launches);
    for mutate in mutations.iter().copied() {
        let mut corrupted = copy_receipt(&receipt);
        mutate(&mut corrupted);
        assert!(first.consume_receipt(corrupted).is_err());
    }
    let generation = receipt.owner_generation();
    let nonce = receipt.transaction_nonce();
    first.consume_receipt(receipt).unwrap();
    assert!(first.consume_receipt(stale).is_err());
    let next = first.execute(&gpu, plan, buffers, 7).unwrap();
    assert_eq!(next.owner_generation(), generation);
    assert!(next.transaction_nonce() > nonce);
    let cross_owner = copy_receipt(&next);
    first.consume_receipt(next).unwrap();
    let other = second.execute(&gpu, plan, buffers, 7).unwrap();
    assert_ne!(other.owner_generation(), generation);
    assert!(second.consume_receipt(cross_owner).is_err());
    second.consume_receipt(other).unwrap();
    let staged_token = gpu.read_alloc(token_ptr).unwrap();
    let launches = gpu.launch_count();
    first.next_nonce = u64::MAX;
    let next_plan = Glm53B1T1BootstrapPlan::new(18).unwrap();
    assert!(first.execute(&gpu, next_plan, buffers, 7).is_err());
    assert_eq!(gpu.read_alloc(token_ptr).unwrap(), staged_token);
    assert_eq!(gpu.launch_count(), launches);
    assert!(first.active.is_none());
}

#[test]
fn forged_plan_and_buffer_hostiles_fail_before_copy_or_launch() {
    let mutations: &[fn(&mut Glm53B1T1BootstrapPlan, &mut Glm53B1T1BootstrapBuffers)] = &[
        |plan, _| plan.embedding.tokens = 2,
        |plan, _| plan.embedding.vocab = 154_879,
        |plan, _| plan.hyper.tokens = 2,
        |plan, _| plan.kernel_launches = 1,
        |_, buffers| buffers.source_q5_k.bytes -= 1,
        |_, buffers| buffers.token_ids_u32.bytes = 8,
        |_, buffers| buffers.hidden_bf16.bytes -= 2,
        |_, buffers| buffers.streams_bf16.bytes -= 2,
        |_, buffers| buffers.source_q5_k.ptr.0 += 1,
        |_, buffers| buffers.source_q5_k.ptr.0 += 2,
        |_, buffers| buffers.token_ids_u32.ptr.0 += 1,
        |_, buffers| buffers.token_ids_u32.ptr.0 += 2,
        |_, buffers| buffers.hidden_bf16.ptr.0 += 1,
        |_, buffers| buffers.streams_bf16.ptr.0 += 1,
        |_, buffers| buffers.streams_bf16.ptr = DevicePtr::NULL,
        |_, buffers| buffers.token_ids_u32.ptr = buffers.source_q5_k.ptr,
        |_, buffers| buffers.hidden_bf16.ptr = buffers.source_q5_k.ptr,
        |_, buffers| buffers.streams_bf16.ptr = buffers.source_q5_k.ptr,
        |_, buffers| buffers.hidden_bf16.ptr = buffers.token_ids_u32.ptr,
        |_, buffers| buffers.streams_bf16.ptr = buffers.token_ids_u32.ptr,
        |_, buffers| buffers.streams_bf16.ptr = buffers.hidden_bf16.ptr,
        |plan, buffers| {
            let end = buffers.source_q5_k.ptr.0 + plan.embedding.source_bytes as u64;
            buffers.hidden_bf16.ptr.0 = end - 2
        },
        |_, buffers| buffers.source_q5_k.ptr = DevicePtr(u64::MAX - 3),
    ];
    for mutate in mutations.iter().copied() {
        let gpu = MockGpuBackend::new();
        let token_ptr = gpu.alloc(4).unwrap();
        let (embedding, hyper) = kernels(&gpu);
        let mut executor = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
        let mut plan = Glm53B1T1BootstrapPlan::new(17).unwrap();
        let mut buffers = valid_buffers(token_ptr);
        mutate(&mut plan, &mut buffers);
        assert!(executor.execute(&gpu, plan, buffers, 5).is_err());
        assert_eq!(gpu.read_alloc(token_ptr).unwrap(), [0; 4]);
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[test]
fn token_staging_copy_failure_prevents_both_launches() {
    let gpu = MockGpuBackend::new();
    let (embedding, hyper) = kernels(&gpu);
    let mut executor = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
    let buffers = valid_buffers(DevicePtr(0x3c00_0000));
    let plan = Glm53B1T1BootstrapPlan::new(17).unwrap();
    assert!(executor.execute(&gpu, plan, buffers, 5).is_err());
    assert_eq!(gpu.launch_count(), 0);

    let token_ptr = gpu.alloc(4).unwrap();
    assert!(
        executor
            .execute(&gpu, plan, valid_buffers(token_ptr), 0)
            .is_err()
    );
    assert_eq!(gpu.read_alloc(token_ptr).unwrap(), [0; 4]);
    assert_eq!(gpu.launch_count(), 0);
}

struct FailingLaunchGpu {
    inner: MockGpuBackend,
    fail_at: usize,
    capturing: bool,
    attempts: AtomicUsize,
    attempted_launches: Mutex<Vec<(u64, u64)>>,
}

impl FailingLaunchGpu {
    fn new(fail_at: usize) -> Self {
        Self {
            inner: MockGpuBackend::new(),
            fail_at,
            capturing: false,
            attempts: AtomicUsize::new(0),
            attempted_launches: Mutex::new(Vec::new()),
        }
    }

    fn capturing() -> Self {
        Self {
            capturing: true,
            ..Self::new(usize::MAX)
        }
    }
}

impl GpuBackend for FailingLaunchGpu {
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

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.attempted_launches
            .lock()
            .unwrap()
            .push((func.0, stream));
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt == self.fail_at {
            bail!("injected launch failure");
        }
        self.inner
            .launch(func, grid, block, shared_mem, stream, params)
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.inner.synchronize(stream)
    }

    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }

    fn kernel(&self, module: &str, function: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(match (module, function) {
            ("ggml_q5_embedding", "atlas_q5_k_embedding_gather_bf16") => 0xe001,
            ("glm53_hyper", "atlas_glm53_hc_expand") => 0xe002,
            _ => 0xd000,
        }))
    }

    fn stream_is_capturing(&self, _stream: u64) -> bool {
        self.capturing
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }

    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }

    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
}

#[test]
fn successful_backend_trace_is_exact_gather_then_expand() {
    let gpu = FailingLaunchGpu::new(usize::MAX);
    let token_ptr = gpu.alloc(4).unwrap();
    let (embedding, hyper) = kernels(&gpu);
    let mut executor = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
    let plan = Glm53B1T1BootstrapPlan::new(17).unwrap();
    let receipt = executor
        .execute(&gpu, plan, valid_buffers(token_ptr), 11)
        .unwrap();
    assert_eq!(
        gpu.attempted_launches.lock().unwrap().as_slice(),
        &[(0xe001, 11), (0xe002, 11)]
    );
    assert_eq!(gpu.attempts.load(Ordering::SeqCst), 2);
    executor.consume_receipt(receipt).unwrap();
}

#[test]
fn either_launch_failure_returns_no_receipt_and_stops_in_order() {
    for fail_at in [1, 2] {
        let gpu = FailingLaunchGpu::new(fail_at);
        let token_ptr = gpu.alloc(4).unwrap();
        let (embedding, hyper) = kernels(&gpu);
        let mut executor = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
        let result = executor.execute(
            &gpu,
            Glm53B1T1BootstrapPlan::new(17).unwrap(),
            valid_buffers(token_ptr),
            11,
        );
        assert!(result.is_err());
        assert_eq!(gpu.attempts.load(Ordering::SeqCst), fail_at);
        assert_eq!(gpu.inner.launch_count(), fail_at - 1);
        assert!(executor.active.is_none());
    }
}

#[test]
fn graph_capture_rejects_before_token_copy_or_kernel_effect() {
    let gpu = FailingLaunchGpu::capturing();
    let token_ptr = gpu.alloc(4).unwrap();
    let (embedding, hyper) = kernels(&gpu);
    let mut executor = Glm53B1T1BootstrapKernels::new(&embedding, &hyper).unwrap();
    let plan = Glm53B1T1BootstrapPlan::new(17).unwrap();
    assert!(
        executor
            .execute(&gpu, plan, valid_buffers(token_ptr), 3)
            .is_err()
    );
    assert_eq!(gpu.inner.read_alloc(token_ptr).unwrap(), [0; 4]);
    assert_eq!(gpu.attempts.load(Ordering::SeqCst), 0);
}

fn ordered(source: &str, needles: &[&str]) -> bool {
    let mut cursor = 0usize;
    for needle in needles {
        let Some(found) = source[cursor..].find(needle) else {
            return false;
        };
        cursor += found + needle.len();
    }
    true
}

fn production_contract(source: &str) -> bool {
    source_sha256::matches(source)
        && ordered(
            source,
            &[
                "plan.validate()?;",
                "validate_buffers(plan, buffers)?;",
                "if gpu.stream_is_capturing(stream)",
                "gpu.copy_h2d(",
                "&plan.token_id.to_le_bytes()",
                "buffers.token_ids_u32.ptr",
                "self.embedding",
                ".launch(",
                "source_q5_k: buffers.source_q5_k",
                "token_ids_u32: buffers.token_ids_u32",
                "destination_bf16: buffers.hidden_bf16",
                "self.hyper",
                ".expand(",
                "buffers.hidden_bf16",
                "buffers.streams_bf16",
                "self.active = Some(ActiveBootstrap",
                "Ok(Glm53B1T1BootstrapReceipt",
            ],
        )
        && !source.contains("#[derive(Debug, Clone, Copy)]")
        && !source.contains(".field(\"ptr\"")
        && !source.contains(".field(\"base\"")
        && !source.contains(".field(\"buffer\"")
        && !source.contains("Glm53Gguf")
        && !source.contains("::load(")
        && !source.contains(".alloc(")
        && !source.contains("impl Model for")
        && !source.contains("Glm53RuntimeCapability")
}

#[test]
fn source_contract_rejects_admission_launch_abi_and_order_mutants() {
    assert!(production_contract(SOURCE));
    const MUTATIONS: &str = r#"const TOKENS: u32 = 1;|const TOKENS: u32 = 2;
const VOCAB: u32 = 154_880;|const VOCAB: u32 = 154_879;
const HIDDEN: u32 = 4_096;|const HIDDEN: u32 = 4_095;
const HC: u32 = 4;|const HC: u32 = 3;
const SINKHORN_ITERS: u32 = 20;|const SINKHORN_ITERS: u32 = 19;
const KERNEL_LAUNCHES: u32 = 2;|const KERNEL_LAUNCHES: u32 = 1;
#[derive(PartialEq, Eq)]\npub(crate) struct Glm53B1T1BootstrapReceipt|#[derive(Clone, Copy, PartialEq, Eq)]\npub(crate) struct Glm53B1T1BootstrapReceipt
pub(crate) source_q5_k: GgmlIqBuffer|pub source_q5_k: GgmlIqBuffer
plan.validate()?;|
validate_buffers(plan, buffers)?;|
if gpu.stream_is_capturing(stream)|if false
gpu.copy_h2d(&plan.token_id.to_le_bytes(), buffers.token_ids_u32.ptr)?;|gpu.copy_h2d(&0u32.to_le_bytes(), buffers.token_ids_u32.ptr)?;
gpu.copy_h2d(&plan.token_id.to_le_bytes(), buffers.token_ids_u32.ptr)?;|gpu.copy_h2d(&plan.token_id.to_le_bytes(), buffers.hidden_bf16.ptr)?;
source_q5_k: buffers.source_q5_k|source_q5_k: buffers.hidden_bf16
token_ids_u32: buffers.token_ids_u32|token_ids_u32: buffers.source_q5_k
destination_bf16: buffers.hidden_bf16|destination_bf16: buffers.streams_bf16
buffers.hidden_bf16,\n            buffers.streams_bf16|buffers.streams_bf16,\n            buffers.hidden_bf16
for right in left + 1..ranges.len()|for right in ranges.len()..ranges.len()
ranges[left].0 < ranges[right].1|ranges[left].0 > ranges[right].1
self.embedding.launch(|self.hyper.launch(
token_id: plan.token_id,\n            stream,\n            buffer_ranges,\n            stage:|token_id: plan.token_id,\n            stream: 0,\n            buffer_ranges,\n            stage:
stage: Glm53B1T1BootstrapStage::MhcExpandEnqueued,\n            kernel_launches: KERNEL_LAUNCHES,|stage: Glm53B1T1BootstrapStage::MhcExpandEnqueued,\n            kernel_launches: 1,
if stream == 0|if false
current.checked_add(1)|current.wrapping_add(1)
transaction_nonce\n            .checked_add(1)|transaction_nonce\n            .wrapping_add(1)
plan.embedding.source_bytes,\n            4u64,|plan.embedding.source_bytes,\n            2u64,
plan.embedding.token_ids_bytes,\n            4,|plan.embedding.token_ids_bytes,\n            2,
.checked_add(u64::try_from(buffer.bytes)?)|.checked_add(u64::try_from(buffer.bytes / 2)?)
ranges[slot] = (\n            buffer.ptr.0,|ranges[slot] = (\n            0,
destination_bf16: buffers.hidden_bf16,\n            },\n            stream,\n        )?;|destination_bf16: buffers.hidden_bf16,\n            },\n            0,\n        )?;
buffers.streams_bf16,\n            stream,\n        )?;|buffers.streams_bf16,\n            0,\n        )?;
buffers.streams_bf16,\n            stream,\n        )?;|buffers.streams_bf16,\n            stream,\n        );
token_id: plan.token_id,\n            stream,\n            buffer_ranges,\n            stage:|token_id: 0,\n            stream,\n            buffer_ranges,\n            stage:"#;
    for encoded in MUTATIONS.lines() {
        let (from, to) = encoded.split_once('|').unwrap();
        let from = from.replace("\\n", "\n");
        let to = to.replace("\\n", "\n");
        assert_eq!(
            SOURCE.matches(&from).count(),
            1,
            "ambiguous mutation: {from}"
        );
        assert!(
            !production_contract(&SOURCE.replacen(&from, &to, 1)),
            "accepted mutation: {from}"
        );
    }
}
