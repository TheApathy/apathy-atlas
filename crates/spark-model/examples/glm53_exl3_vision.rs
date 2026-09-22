// SPDX-License-Identifier: AGPL-3.0-only

//! Exact physical image-to-language gate for GLM-5.3-Flash EXL3.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::VisionConfig;
use base64::Engine;
use half::bf16;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use spark_model::layers::{forward_glm53_exl3_image, ops::Glm53Exl3Buffer};
use spark_model::model::glm53::Glm53Exl3Model;
use spark_model::vision_preprocess::preprocess_image;
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, Glm53Exl3VisionCatalog, admit_glm53_exl3_files,
    load_glm53_exl3_store, materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const RESERVE_BYTES: usize = 16 * 1024 * 1024 * 1024;
const CONTEXT: u32 = 128;
const PREFIX: &[u32] = &[
    154_822, 154_824, 154_826, // [gMASK] <sop> <|system|>
    25_062, 287, 29_905, 371, 25, 12_035, // Reasoning Effort: Low
    154_827, 154_830, // <|user|> <|begin_of_image|>
];
const SUFFIX: &[u32] = &[
    154_831, 74_198, 279, 2_168, 26_667, 13, // image closer + user text
    154_828, 154_841, // <|assistant|> <think>
];

fn vision_config() -> VisionConfig {
    VisionConfig {
        model_type: "glm5_next_vision".into(),
        depth: 24,
        hidden_size: 1024,
        num_heads: 16,
        patch_size: 14,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 4096,
        out_hidden_size: 4096,
        in_channels: 3,
        image_size: 448,
        projection_intermediate_size: 10240,
        rms_norm_eps: 1e-5,
        swiglu_limit: Some(10.0),
        deepstack_visual_indexes: Vec::new(),
        image_pad_token_id: 154854,
        video_pad_token_id: 154855,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn gradient_png_data_uri() -> Result<String> {
    let image = RgbImage::from_fn(20, 20, |x, y| {
        Rgb([
            (x * 11 + y * 3) as u8,
            (x * 5 + y * 7) as u8,
            (x * 2 + y * 13) as u8,
        ])
    });
    let mut encoded = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image).write_to(&mut encoded, ImageFormat::Png)?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(encoded.into_inner())
    ))
}

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .context("usage: glm53_exl3_vision <exact-checkpoint-directory>")?;
    let files = admit_glm53_exl3_files(&root)?;
    let backend: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend> =
        std::sync::Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let gpu: &dyn GpuBackend = backend.as_ref();
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let target_catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
    let vision_catalog = Glm53Exl3VisionCatalog::new(&files, &store)?;
    let native = materialize_glm53_exl3_native(&target_catalog, gpu).map_err(|error| {
        let retained = error.retained_bytes();
        match error.retry_cleanup(gpu) {
            Ok(primary) => primary.context(format!(
                "materialize GLM EXL3 native operands; retained before cleanup={retained}"
            )),
            Err(error) => anyhow::anyhow!("{error}; retained={}", error.retained_bytes()),
        }
    })?;
    let weights = Glm53Exl3TargetWeights::new(&target_catalog, &native)?;
    drop(target_catalog);
    let model = Glm53Exl3Model::new(backend, store, native, weights, CONTEXT)?;

    let image = gradient_png_data_uri()?;
    let (pixels, grid_h, grid_w) = preprocess_image(&image, &vision_config())?;
    ensure!((grid_h, grid_w) == (8, 8), "GLM minimum image grid drift");
    let output_bytes = (grid_h * grid_w / 4) * 4096 * 2;
    let output_ptr = model.gpu().alloc(output_bytes)?;
    let output = Glm53Exl3Buffer {
        ptr: output_ptr,
        bytes: output_bytes,
    };

    let result = (|| {
        let stream = model.gpu().default_stream();
        let receipt = forward_glm53_exl3_image(
            model.gpu(),
            &vision_catalog,
            &pixels,
            grid_h,
            grid_w,
            output,
            stream,
        )?;
        let mut raw = vec![0u8; output_bytes];
        model.gpu().copy_d2h(output_ptr, &mut raw)?;
        let mut nonzero = 0usize;
        let mut max_abs = 0.0f32;
        let mut square = 0.0f64;
        for pair in raw.chunks_exact(2) {
            let value = bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
            ensure!(
                value.is_finite(),
                "GLM vision produced a non-finite embedding"
            );
            nonzero += usize::from(value != 0.0);
            max_abs = max_abs.max(value.abs());
            square += f64::from(value) * f64::from(value);
        }
        ensure!(
            nonzero != 0 && max_abs > 0.0,
            "GLM vision produced only zero embeddings"
        );
        let rms = (square / (raw.len() / 2) as f64).sqrt();

        model.prefill_tokens(PREFIX, stream)?;
        model.prefill_embeddings(output, stream)?;
        let logits = model.prefill_tokens(SUFFIX, stream)?;
        let prompt_position = (PREFIX.len() + 16 + SUFFIX.len()) as u32;
        ensure!(
            model.position() == prompt_position,
            "GLM multimodal position accounting drift"
        );
        let mut token = model.argmax_host(logits)?;
        let mut generated = vec![token];
        for _ in 1..8 {
            token = model.argmax_host(model.decode_token(token, stream)?)?;
            generated.push(token);
        }
        Ok((
            receipt,
            fnv1a64(&raw),
            nonzero,
            max_abs,
            rms,
            prompt_position,
            generated,
        ))
    })();

    let output_cleanup = model
        .gpu()
        .free(output_ptr)
        .context("free GLM vision output");
    drop(vision_catalog);
    let model_cleanup = model.free();
    let (receipt, hash, nonzero, max_abs, rms, prompt_position, generated) =
        match (result, output_cleanup, model_cleanup) {
            (Ok(value), Ok(()), Ok(())) => value,
            (Err(error), Ok(()), Ok(())) => return Err(error),
            (Ok(_), Err(error), Ok(())) | (Ok(_), Ok(()), Err(error)) => return Err(error),
            (Err(error), output, model) => {
                return Err(error).context(format!(
                    "multimodal cleanup also failed: output={output:?}; model={model:?}"
                ));
            }
            (Ok(_), Err(output), Err(model)) => {
                return Err(output).context(format!("model cleanup also failed: {model:#}"));
            }
        };
    println!(
        "RESULT: PASS image=png grid=8x8 rows={} width={} bytes={} nonzero={} max_abs={:.8} rms={:.8} fnv1a64={:016x} prompt_position={} generated={generated:?}",
        receipt.rows, receipt.width, receipt.bytes, nonzero, max_abs, rms, hash, prompt_position,
    );
    Ok(())
}
