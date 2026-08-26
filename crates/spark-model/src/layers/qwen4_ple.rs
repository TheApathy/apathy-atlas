// SPDX-License-Identifier: AGPL-3.0-only

//! Exact Qwen4-Exp PLE n-gram row selection.

use anyhow::{Context, Result, ensure};

pub const QWEN4_PLE_HEADS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleRowSelection {
    pub shard: usize,
    pub row: usize,
}

/// CPU row planner. It runs as soon as the current token is known, allowing
/// the 16 sparse reads to overlap embedding lookup and layer 0.
pub struct Qwen4PleHasher {
    multipliers: [i64; 3],
    vocab_sizes: [i64; QWEN4_PLE_HEADS],
    offsets: [i64; QWEN4_PLE_HEADS],
    eos_token_id: u32,
    shard_rows: usize,
}

impl Qwen4PleHasher {
    pub fn new(
        multipliers: [i64; 3],
        vocab_sizes: [i64; QWEN4_PLE_HEADS],
        offsets: [i64; QWEN4_PLE_HEADS],
        eos_token_id: u32,
        shard_rows: usize,
    ) -> Result<Self> {
        ensure!(shard_rows > 0, "PLE shard row count must be positive");
        ensure!(
            multipliers.iter().all(|value| value % 2 != 0),
            "PLE hash multipliers must be odd"
        );
        ensure!(
            vocab_sizes.iter().all(|value| *value > 0),
            "PLE head vocabulary sizes must be positive"
        );
        ensure!(offsets[0] == 0, "PLE first head offset must be zero");
        for head in 1..QWEN4_PLE_HEADS {
            ensure!(
                offsets[head] == offsets[head - 1] + vocab_sizes[head - 1],
                "PLE head offsets are not contiguous at head {head}"
            );
        }
        Ok(Self {
            multipliers,
            vocab_sizes,
            offsets,
            eos_token_id,
            shard_rows,
        })
    }

    pub fn select_decode(
        &self,
        current_token: u32,
        prior_tokens: &[u32],
    ) -> [PleRowSelection; QWEN4_PLE_HEADS] {
        let mut segment = prior_tokens
            .iter()
            .rev()
            .take_while(|&&token| token != self.eos_token_id);
        let previous = segment.next().copied().unwrap_or(self.eos_token_id);
        let previous_2 = segment.next().copied().unwrap_or(self.eos_token_id);
        let bigram = (current_token as i64).wrapping_mul(self.multipliers[0])
            ^ (previous as i64).wrapping_mul(self.multipliers[1]);
        let trigram = bigram ^ (previous_2 as i64).wrapping_mul(self.multipliers[2]);
        std::array::from_fn(|head| {
            let mixed = if head < 8 { bigram } else { trigram };
            let global = mixed.rem_euclid(self.vocab_sizes[head]) + self.offsets[head];
            let global = global as usize;
            PleRowSelection {
                shard: global / self.shard_rows,
                row: global % self.shard_rows,
            }
        })
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod runtime {
    use super::*;
    use atlas_core::config::ModelConfig;
    use parking_lot::Mutex;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
    use spark_runtime::kernel_args::KernelLaunch;
    use spark_runtime::weights::{WeightDtype, WeightStore};
    use spark_storage::ple_offload::PleOffloadReader;
    use std::path::Path;

    use crate::layers::ops;
    use crate::weight_map::{DenseWeight, dense};

    const PLE_EMBED: usize = 2560;
    const PLE_RECORD_BYTES: usize = 90;
    const PLE_CONV_HISTORY: usize = 9;
    const PLE_MAX_SPEC_ROWS: usize = 32;

    /// Atlas-native sparse PLE runtime. Only the 16 selected NVFP4 rows are
    /// staged per token; the 320M-row table remains in the O_DIRECT sidecar.
    pub struct Qwen4PleLayer {
        hasher: Qwen4PleHasher,
        reader: Mutex<PleOffloadReader>,
        key_proj: DenseWeight,
        value_proj: DenseWeight,
        norm_key: DenseWeight,
        norm_query: DenseWeight,
        norm_conv: DenseWeight,
        conv_weight: DenseWeight,
        records_dev: DevicePtr,
        scales_dev: DevicePtr,
        embedding_dev: DevicePtr,
        key_dev: DevicePtr,
        value_dev: DevicePtr,
        conv_state_dev: DevicePtr,
        conv_checkpoint_dev: DevicePtr,
        conv_intermediate_dev: DevicePtr,
        dequant_k: KernelHandle,
        fuse_k: KernelHandle,
        dense_gemv_k: KernelHandle,
        hidden_size: usize,
        hc_count: usize,
        max_batch_size: usize,
        eps: f32,
    }

    impl Qwen4PleLayer {
        pub fn load(
            store: &WeightStore,
            config: &ModelConfig,
            gpu: &dyn GpuBackend,
            max_batch_size: usize,
        ) -> Result<Option<Self>> {
            if !config.is_qwen4_exp() {
                return Ok(None);
            }
            let manifest = config.ple_offload_manifest.as_deref().context(
                "Qwen4 PLE checkpoint is missing ple-offload/manifest.json; refusing to run without the position-learning enhancement table",
            )?;
            ensure!(
                config.hidden_size == PLE_EMBED,
                "Qwen4 PLE hidden size mismatch"
            );
            ensure!(config.hc_count == 4, "Qwen4 PLE requires hc_count=4");
            let prefix = format!("{}.layers.1.ple", config.weight_prefix);
            check_tensor(
                store,
                &format!("{prefix}.key_proj.weight"),
                &[10240, 2560],
                WeightDtype::BF16,
            )?;
            check_tensor(
                store,
                &format!("{prefix}.value_proj.weight"),
                &[2560, 2560],
                WeightDtype::BF16,
            )?;
            check_tensor(
                store,
                &format!("{prefix}.conv1d.weight"),
                &[10240, 1, 4],
                WeightDtype::BF16,
            )?;
            for name in ["norm_key.weight", "norm_query.weight", "norm_conv.weight"] {
                check_tensor(
                    store,
                    &format!("{prefix}.{name}"),
                    &[10240],
                    WeightDtype::BF16,
                )?;
            }

            let ep = format!("{prefix}.ple_embedding");
            let multipliers = read_i64::<3>(store, &format!("{ep}.layer_multipliers"), gpu)?;
            let vocab_sizes =
                read_i64::<QWEN4_PLE_HEADS>(store, &format!("{ep}.ngram_heads_vocab_sizes"), gpu)?;
            let offsets =
                read_i64::<QWEN4_PLE_HEADS>(store, &format!("{ep}.ngram_heads_offsets"), gpu)?;
            let hasher = Qwen4PleHasher::new(
                multipliers,
                vocab_sizes,
                offsets,
                config.eos_token_id,
                2_500_012,
            )?;

            // The published NVFP4 recipe explicitly ignores `*.ple.*`.
            // Preserve those checkpoint projections in BF16 rather than
            // introducing a second, unsupported quantization pass.
            let key_proj = dense(store, &format!("{prefix}.key_proj.weight"))?;
            let value_proj = dense(store, &format!("{prefix}.value_proj.weight"))?;
            let cache_mb = std::env::var("ATLAS_PLE_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(512);
            let reader = PleOffloadReader::open(Path::new(manifest), 32, cache_mb * 1024 * 1024)
                .with_context(|| format!("open Qwen4 PLE offload manifest {manifest}"))?;
            let residual = config.residual_width();
            let layer = Self {
                hasher,
                reader: Mutex::new(reader),
                key_proj,
                value_proj,
                norm_key: dense(store, &format!("{prefix}.norm_key.weight"))?,
                norm_query: dense(store, &format!("{prefix}.norm_query.weight"))?,
                norm_conv: dense(store, &format!("{prefix}.norm_conv.weight"))?,
                conv_weight: dense(store, &format!("{prefix}.conv1d.weight"))?,
                records_dev: gpu.alloc(QWEN4_PLE_HEADS * PLE_RECORD_BYTES)?,
                scales_dev: gpu.alloc(QWEN4_PLE_HEADS * 4)?,
                embedding_dev: gpu.alloc(PLE_EMBED * 2)?,
                key_dev: gpu.alloc(residual * 2)?,
                value_dev: gpu.alloc(PLE_EMBED * 2)?,
                conv_state_dev: gpu.alloc(max_batch_size * residual * PLE_CONV_HISTORY * 2)?,
                conv_checkpoint_dev: gpu.alloc(max_batch_size * residual * PLE_CONV_HISTORY * 2)?,
                conv_intermediate_dev: gpu
                    .alloc(max_batch_size * PLE_MAX_SPEC_ROWS * residual * PLE_CONV_HISTORY * 2)?,
                dequant_k: gpu.kernel("qwen4_hyper", "qwen4_ple_dequant_rows")?,
                fuse_k: gpu.kernel("qwen4_hyper", "qwen4_ple_fuse_decode")?,
                dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
                hidden_size: config.hidden_size,
                hc_count: config.hc_count,
                max_batch_size,
                eps: config.rms_norm_eps as f32,
            };
            gpu.memset(
                layer.conv_state_dev,
                0,
                max_batch_size * residual * PLE_CONV_HISTORY * 2,
            )?;
            gpu.memset(
                layer.conv_checkpoint_dev,
                0,
                max_batch_size * residual * PLE_CONV_HISTORY * 2,
            )?;
            gpu.memset(
                layer.conv_intermediate_dev,
                0,
                max_batch_size * PLE_MAX_SPEC_ROWS * residual * PLE_CONV_HISTORY * 2,
            )?;
            tracing::info!(manifest, cache_mb, "Qwen4 PLE sparse NVFP4 offload enabled");
            Ok(Some(layer))
        }

        /// Fetch, project, and inject one token. `prior_tokens` excludes the
        /// current input token, matching the official n-gram shift contract.
        pub fn forward_token(
            &self,
            current_token: u32,
            prior_tokens: &[u32],
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let selections = self.hasher.select_decode(current_token, prior_tokens);
            let request: Vec<_> = selections.iter().map(|s| (s.shard, s.row)).collect();
            let rows = self.reader.lock().read_rows(&request)?;
            let mut records = [0u8; QWEN4_PLE_HEADS * PLE_RECORD_BYTES];
            let mut scales = [0u8; QWEN4_PLE_HEADS * 4];
            for (head, row) in rows.iter().enumerate() {
                let start = head * PLE_RECORD_BYTES;
                records[start..start + PLE_RECORD_BYTES].copy_from_slice(&row.record);
                scales[head * 4..head * 4 + 4].copy_from_slice(&row.scale2.to_le_bytes());
            }
            gpu.copy_h2d_group_on_stream(
                &[
                    HostToDeviceCopy::new(&records, self.records_dev),
                    HostToDeviceCopy::new(&scales, self.scales_dev),
                ],
                stream,
            )?;
            KernelLaunch::new(gpu, self.dequant_k)
                .grid([10, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.records_dev)
                .arg_ptr(self.scales_dev)
                .arg_ptr(self.embedding_dev)
                .launch(stream)?;
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                self.embedding_dev,
                &self.key_proj,
                self.key_dev,
                self.residual_width() as u32,
                PLE_EMBED as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                self.embedding_dev,
                &self.value_proj,
                self.value_dev,
                PLE_EMBED as u32,
                PLE_EMBED as u32,
                stream,
            )?;
            let state_stride = self.residual_width() * PLE_CONV_HISTORY * 2;
            KernelLaunch::new(gpu, self.fuse_k)
                .grid([self.hc_count as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(self.key_dev)
                .arg_ptr(self.value_dev)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(self.conv_state_dev.offset(slot_idx * state_stride))
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .arg_u32(u32::from(reset_state))
                .launch(stream)
        }

        pub fn residual_width(&self) -> usize {
            self.hidden_size * self.hc_count
        }

        fn state_stride_bytes(&self) -> usize {
            self.residual_width() * PLE_CONV_HISTORY * 2
        }

        /// Clear every request-local PLE history buffer for a reused slot.
        /// PLE state is allocated outside `SsmStatePool`, so the pool's slot
        /// reset cannot cover it.
        pub fn zero_slot(&self, slot_idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.memset_async(
                self.conv_state_dev.offset(slot_idx * stride),
                0,
                stride,
                stream,
            )?;
            gpu.memset_async(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                0,
                stride,
                stream,
            )?;
            gpu.memset_async(
                self.conv_intermediate_dev
                    .offset(slot_idx * PLE_MAX_SPEC_ROWS * stride),
                0,
                PLE_MAX_SPEC_ROWS * stride,
                stream,
            )
        }

        /// Save the canonical PLE convolution history at a speculative boundary.
        pub fn checkpoint(&self, slot_idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(slot_idx * stride),
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        /// Preserve the state after one verify row for partial acceptance.
        pub fn save_intermediate(
            &self,
            slot_idx: usize,
            row: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(
                row < PLE_MAX_SPEC_ROWS,
                "PLE speculative row {row} exceeds {PLE_MAX_SPEC_ROWS}"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(slot_idx * stride),
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + row) * stride),
                stride,
                stream,
            )
        }

        /// Restore the pre-verify state before a verify replay.
        pub fn restore_checkpoint(
            &self,
            slot_idx: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                self.conv_state_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        /// Commit a partial rollback and make it the next checkpoint.
        pub fn rollback_and_checkpoint(
            &self,
            slot_idx: usize,
            num_accepted: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(
                num_accepted <= PLE_MAX_SPEC_ROWS,
                "Qwen4 PLE rollback accepts at most {PLE_MAX_SPEC_ROWS} rows, got {num_accepted}"
            );
            let stride = self.state_stride_bytes();
            let src = if num_accepted == 0 {
                self.conv_checkpoint_dev.offset(slot_idx * stride)
            } else {
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + num_accepted - 1) * stride)
            };
            let live = self.conv_state_dev.offset(slot_idx * stride);
            gpu.copy_d2d_async(src, live, stride, stream)?;
            gpu.copy_d2d_async(
                live,
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        pub fn owned_device_buffers(&self) -> [DevicePtr; 2] {
            [self.records_dev, self.scales_dev]
        }

        pub fn scratch_device_buffers(&self) -> [DevicePtr; 6] {
            [
                self.embedding_dev,
                self.key_dev,
                self.value_dev,
                self.conv_state_dev,
                self.conv_checkpoint_dev,
                self.conv_intermediate_dev,
            ]
        }
    }

    fn check_tensor(
        store: &WeightStore,
        name: &str,
        shape: &[usize],
        dtype: WeightDtype,
    ) -> Result<()> {
        let tensor = store.get(name)?;
        ensure!(
            tensor.shape == shape,
            "{name} shape {:?} != {shape:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == dtype,
            "{name} dtype {:?} != {dtype:?}",
            tensor.dtype
        );
        Ok(())
    }

    fn read_i64<const N: usize>(
        store: &WeightStore,
        name: &str,
        gpu: &dyn GpuBackend,
    ) -> Result<[i64; N]> {
        check_tensor(store, name, &[N], WeightDtype::Int64)?;
        let mut bytes = vec![0u8; N * 8];
        gpu.copy_d2h(store.get(name)?.ptr, &mut bytes)?;
        Ok(std::array::from_fn(|i| {
            i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("fixed i64 slice"))
        }))
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use runtime::Qwen4PleLayer;

#[cfg(test)]
mod tests {
    use super::*;

    fn hasher() -> Qwen4PleHasher {
        Qwen4PleHasher::new(
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ],
            [
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ],
            248_044,
            2_500_012,
        )
        .unwrap()
    }

    #[test]
    fn selections_match_official_torch_reference() {
        let got = hasher().select_decode(3, &[1, 2]);
        let expected = [
            (5, 2_105_657),
            (9, 1_910_767),
            (19, 1_813_339),
            (28, 2_177_092),
            (34, 1_060_412),
            (41, 1_521_518),
            (52, 963_195),
            (58, 1_885_506),
            (64, 1_522_309),
            (78, 938_190),
            (87, 1_923_521),
            (88, 810_726),
            (99, 518_953),
            (110, 227_210),
            (113, 1_796_643),
            (123, 2_143_747),
        ];
        for (selection, &(shard, row)) in got.iter().zip(&expected) {
            assert_eq!(*selection, PleRowSelection { shard, row });
        }
    }

    #[test]
    fn eos_resets_ngram_context() {
        assert_eq!(
            hasher().select_decode(3, &[1, 248_044]),
            hasher().select_decode(3, &[])
        );
    }
}
