// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit DeepSeek visual-loader entry point; shared loader dispatch owns admission.

use crate::layers::deepseek_vision::DeepSeekVisionEncoder;
use anyhow::Result;
use atlas_core::config::DeepSeekVisionConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

pub fn load_vision_encoder(
    store: &WeightStore,
    config: &DeepSeekVisionConfig,
    text_hidden_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<DeepSeekVisionEncoder> {
    DeepSeekVisionEncoder::load(store, config, text_hidden_size, gpu)
}
