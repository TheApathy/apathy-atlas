// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnComponent};
use crate::weight_map::w3_sidecar::{W3SidecarRequest, W3SidecarSession};
use crate::weight_map::{Nvfp4Variant, load_dense_ffn};

pub(super) struct DenseFfnLoadContext<'a> {
    store: &'a WeightStore,
    config: &'a ModelConfig,
    gpu: &'a dyn GpuBackend,
    variant: Nvfp4Variant,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    flashinfer_requested: bool,
}

impl<'a> DenseFfnLoadContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        store: &'a WeightStore,
        config: &'a ModelConfig,
        gpu: &'a dyn GpuBackend,
        variant: Nvfp4Variant,
        absmax_k: KernelHandle,
        quantize_k: KernelHandle,
        stream: u64,
        flashinfer_requested: bool,
    ) -> Self {
        Self {
            store,
            config,
            gpu,
            variant,
            absmax_k,
            quantize_k,
            stream,
            flashinfer_requested,
        }
    }

    pub(super) fn load(
        &self,
        layer: usize,
        layer_prefix: &str,
        w3_session: &mut Option<W3SidecarSession>,
    ) -> Result<FfnComponent> {
        let hidden = self.config.hidden_size;
        let intermediate = self.config.intermediate_size;
        let weights = load_dense_ffn(
            self.store,
            layer_prefix,
            self.gpu,
            self.variant,
            self.absmax_k,
            self.quantize_k,
            self.stream,
            self.config,
        )?;
        let mut ffn_layer = DenseFfnLayer::new(weights, self.gpu)?;

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.flashinfer_requested {
            ffn_layer.prepare_flashinfer_prefill(self.gpu, layer, hidden, intermediate)?;
            if layer + 1 == self.config.num_hidden_layers {
                tracing::info!(
                    layers = self.config.num_hidden_layers,
                    hidden,
                    intermediate,
                    "FlashInfer SM121 FFN prefill construction complete"
                );
            }
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        ensure!(
            !self.flashinfer_requested,
            "ATLAS_PREFILL_FFN_FLASHINFER=1 requires a Linux CUDA build"
        );

        // Retain the incumbent optional W4 transforms. They remain fallback
        // operands for prefill and every route not covered by W3.
        if crate::layers::ffn_m16_transposed_enabled() {
            let gate_t = ffn_layer.weights.gate_proj.transpose_for_gemm_cached(
                self.gpu,
                intermediate,
                hidden,
                &format!("L{layer}.mlp.gate_proj.t"),
            )?;
            let up_t = ffn_layer.weights.up_proj.transpose_for_gemm_cached(
                self.gpu,
                intermediate,
                hidden,
                &format!("L{layer}.mlp.up_proj.t"),
            )?;
            let down_t = ffn_layer.weights.down_proj.transpose_for_gemm_cached(
                self.gpu,
                hidden,
                intermediate,
                &format!("L{layer}.mlp.down_proj.t"),
            )?;
            ffn_layer.set_transposed_weights(gate_t, up_t, down_t);
            ffn_layer.alloc_splitk_workspace(self.gpu, hidden.max(intermediate) as u32)?;
            if layer == 0 {
                tracing::info!(
                    "Dense FFN M_TILE=16 transposed-weight path enabled \
                     (ATLAS_FFN_M16_TRANSPOSED=1): \
                     transposed gate/up/down per layer for w4a16_gemm_n128_m16"
                );
            }
        }

        if crate::layers::prefill_ffn_fp8_enabled() {
            ffn_layer.predequant_for_prefill(self.gpu, hidden, intermediate, self.stream)?;
            if layer == 0 {
                tracing::info!(
                    "Dense FFN FP8 predequant prefill path enabled \
                     (ATLAS_FFN_PREDEQUANT_FP8=1): \
                     pre-dequanted gate/up/down per layer for fp8_gemm_t_m128"
                );
            }
        }

        if let Some(session) = w3_session.as_mut() {
            if let Some(w3) = session.upload_layer(layer, self.gpu)? {
                let gemv = DenseFfnWeights {
                    gate_proj: w3.gate,
                    up_proj: w3.up,
                    down_proj: w3.down,
                };
                let gemm_t = DenseFfnWeights {
                    gate_proj: w3.gate_t,
                    up_proj: w3.up_t,
                    down_proj: w3.down_t,
                };
                ffn_layer.set_w3_weights(gemv, gemm_t);
                session.mark_installed(layer)?;
                tracing::info!(layer, "W3 FFN sidecar weights installed");
            }
        }
        Ok(FfnComponent::Dense(ffn_layer))
    }
}

pub(super) fn prepare_w3_session(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<W3SidecarSession>> {
    let Some(request) = W3SidecarRequest::from_env(config.num_hidden_layers)? else {
        return Ok(None);
    };
    gpu.kernel("w3a16_gemm", "w3a16_gemm_t_m32_n64")?;
    let layer_prefixes = (0..config.num_hidden_layers)
        .map(|layer| config.layer_prefix(layer))
        .collect::<Vec<_>>();
    W3SidecarSession::prepare(
        request,
        &layer_prefixes,
        config.hidden_size,
        config.intermediate_size,
    )
    .map(Some)
}

pub(super) fn finish_w3_session(session: Option<W3SidecarSession>) -> Result<()> {
    let Some(session) = session else {
        return Ok(());
    };
    let receipt = session.finish()?;
    ensure!(
        receipt.requested_layers == receipt.validated_layers
            && receipt.requested_layers == receipt.uploaded_layers
            && receipt.requested_layers == receipt.installed_layers,
        "W3 sidecar layer census differs at publication boundary"
    );
    ensure!(
        receipt.requested_count == receipt.validated_count
            && receipt.requested_count == receipt.uploaded_count
            && receipt.requested_count == receipt.installed_count
            && receipt.requested_count == receipt.requested_layers.len(),
        "W3 sidecar count census differs at publication boundary"
    );
    tracing::info!(
        layers = receipt.installed_count,
        size = receipt.size,
        device = receipt.device,
        inode = receipt.inode,
        "W3 FFN sidecar installation census complete"
    );
    Ok(())
}
