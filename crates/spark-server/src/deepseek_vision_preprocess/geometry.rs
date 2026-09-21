// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use atlas_core::config::DeepSeekVisionConfig;

#[derive(Debug, PartialEq)]
pub(crate) struct ResizePlan {
    pub height: u32,
    pub width: u32,
    pub grid_h: usize,
    pub grid_w: usize,
    pub grid_llm_h: usize,
    pub grid_llm_w: usize,
    pub stretch: bool,
}

fn grid(height: usize, width: usize, p: usize, down: usize) -> Result<(usize, usize, usize)> {
    let h = (height / p).div_ceil(down);
    let w = (width / p).div_ceil(down);
    let tokens = atlas_core::config::deepseek_image_block_len(h, w, 3)?;
    Ok((h, w, tokens))
}

pub(crate) fn resize_plan(
    orig_h: u32,
    orig_w: u32,
    cfg: &DeepSeekVisionConfig,
) -> Result<ResizePlan> {
    cfg.validate()?;
    super::validate_source_size(orig_w, orig_h)?;
    let p = cfg.patch_size;
    let down = cfg.downsample_ratio;
    let mut width = f64::from(orig_w);
    let mut height = f64::from(orig_h);
    let stretch = cfg
        .max_wh_ratio
        .is_some_and(|ratio| width >= height * ratio);
    if let Some(ratio) = cfg.max_wh_ratio {
        width = width.min(height * ratio);
    }
    if width * height < cfg.min_pixels as f64 {
        let scale = (cfg.min_pixels as f64 / (width * height)).sqrt();
        width = (width * scale).floor();
        height = (height * scale).floor();
    }
    let mut best_w = ((width / p as f64).ceil() as usize) * p;
    let mut best_h = ((height / p as f64).ceil() as usize) * p;
    let limit = cfg.max_tokens - 3;
    let mut budget = limit;
    let (h, w) = loop {
        let (h, w, tokens) = grid(best_h, best_w, p, down)?;
        if tokens <= limit && h > 0 && w > 0 {
            break (h, w);
        }
        ensure!(
            budget > 4,
            "DeepSeek token budget cannot represent this image aspect ratio"
        );
        let ratio = height / width;
        let fw = (((budget - 2) as f64) / ratio + 0.25).sqrt() - 0.5;
        let fh = fw * ratio;
        if fw < 1.0 {
            let mw = 1;
            let mut mh = (budget - 2) / (mw + 1);
            mh -= mh % 2;
            best_w = mw * p * down;
            best_h = mh * p * down;
        } else if fh < 2.0 {
            let mh = 2;
            let mw = (budget - 2) / mh - 1;
            ensure!(mw > 1, "DeepSeek token budget cannot represent wide image");
            best_w = mw * p * down;
            best_h = mh * p * down;
        } else {
            let mw = fw.floor() as usize;
            let mut mh = fh.floor() as usize;
            mh -= mh % 2;
            let beta = ((mw * p * down) as f64 / width).min((mh * p * down) as f64 / height);
            best_w = ((width * beta / p as f64).floor() as usize) * p;
            best_h = ((height * beta / p as f64).floor() as usize) * p;
        }
        budget -= 1;
    };
    ensure!(
        best_h > 0 && best_w > 0 && best_h <= 32768 && best_w <= 32768,
        "DeepSeek resized image dimensions invalid"
    );
    Ok(ResizePlan {
        height: best_h as u32,
        width: best_w as u32,
        grid_h: best_h / p,
        grid_w: best_w / p,
        grid_llm_h: h,
        grid_llm_w: w,
        stretch,
    })
}
