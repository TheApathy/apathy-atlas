// SPDX-License-Identifier: AGPL-3.0-only

//! Physical GLM-5.3 EXL3 image request through Atlas's production `Model` API.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::VisionConfig;
use base64::Engine;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use spark_model::model::glm53::Glm53Exl3Model;
use spark_model::traits::Model;
use spark_model::vision_preprocess::preprocess_image;
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, Glm53Exl3VisionCatalog, admit_glm53_exl3_files,
    load_glm53_exl3_store, materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;

const RESERVE_BYTES: usize = 16 * 1024 * 1024 * 1024;
const CONTEXT: u32 = 128;
const IMAGE_PAD: u32 = 154_854;
const PREFIX: &[u32] = &[
    154_822, 154_824, 154_826, 25_062, 287, 29_905, 371, 25, 12_035, 154_827, 154_830,
];
const SUFFIX: &[u32] = &[154_831, 74_198, 279, 2_168, 26_667, 13, 154_828, 154_841];
const EXPECTED: &[u32] = &[785, 2_168, 374, 264, 2_613, 20_129, 2_168, 448];

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
        image_pad_token_id: IMAGE_PAD,
        video_pad_token_id: 154_855,
    }
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
        .context("usage: glm53_exl3_vision_model <exact-checkpoint-directory>")?;
    let files = admit_glm53_exl3_files(&root)?;
    let backend = Box::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let store = match load_glm53_exl3_store(&files, backend.as_ref(), RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(backend.as_ref()) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let target_catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
    let vision_catalog = Glm53Exl3VisionCatalog::new(&files, &store)?;
    let native =
        materialize_glm53_exl3_native(&target_catalog, backend.as_ref()).map_err(|error| {
            match error.retry_cleanup(backend.as_ref()) {
                Ok(primary) => primary.context("materialize GLM EXL3 native operands"),
                Err(retained) => {
                    anyhow::anyhow!("native materialization cleanup failed: {retained}")
                }
            }
        })?;
    let weights = Glm53Exl3TargetWeights::new(&target_catalog, &native)?;
    drop(target_catalog);
    let model =
        Glm53Exl3Model::new_multimodal(backend, store, native, weights, vision_catalog, CONTEXT)?;

    let result = (|| -> Result<Vec<u32>> {
        let image = gradient_png_data_uri()?;
        let (pixels, grid_h, grid_w) = preprocess_image(&image, &vision_config())?;
        ensure!((grid_h, grid_w) == (8, 8), "GLM minimum image grid drift");
        let interface: &dyn Model = &model;
        interface.prepare_vision_embed(&[(pixels, grid_h, grid_w)])?;
        let mut prompt = Vec::with_capacity(PREFIX.len() + 16 + SUFFIX.len());
        prompt.extend_from_slice(PREFIX);
        prompt.extend(std::iter::repeat_n(IMAGE_PAD, 16));
        prompt.extend_from_slice(SUFFIX);
        let mut sequence = interface.alloc_sequence()?;
        let inference = (|| -> Result<Vec<u32>> {
            let stream = interface.default_stream();
            let logits = interface.prefill(&prompt, &mut sequence, stream)?;
            ensure!(
                model.position() == 35 && sequence.seq_len == 35,
                "GLM production multimodal position accounting drift"
            );
            let mut token = interface.argmax_on_device(logits, stream)?;
            let mut generated = vec![token];
            for _ in 1..8 {
                token = interface
                    .argmax_on_device(interface.decode(token, &mut sequence, stream)?, stream)?;
                generated.push(token);
            }
            ensure!(
                generated == EXPECTED,
                "GLM production multimodal semantic output drift: {generated:?}"
            );
            Ok(generated)
        })();
        let sequence_cleanup = interface.free_sequence(&mut sequence);
        match (inference, sequence_cleanup) {
            (Ok(generated), Ok(())) => Ok(generated),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error).context("free GLM EXL3 sequence"),
            (Err(error), Err(cleanup)) => {
                Err(error.context(format!("sequence cleanup also failed: {cleanup:#}")))
            }
        }
    })();
    let model_cleanup = model.free();
    let generated = match (result, model_cleanup) {
        (Ok(generated), Ok(())) => generated,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error).context("free GLM EXL3 model"),
        (Err(error), Err(cleanup)) => {
            return Err(error.context(format!("model cleanup also failed: {cleanup:#}")));
        }
    };
    println!(
        "RESULT: PASS interface=dyn-Model image=png grid=8x8 prompt_position=35 generated={generated:?}"
    );
    Ok(())
}
