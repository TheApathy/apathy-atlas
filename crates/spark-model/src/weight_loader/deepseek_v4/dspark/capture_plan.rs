// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed three-stage DSpark capture contract. This does not enable speculation.

use std::ffi::OsStr;

/// Order of the post-layer HC means concatenated into the shipped main_proj.
pub const CAPTURE_LAYERS: [usize; 3] = [40, 41, 42];

pub fn validate_capture_layers(layers: &[usize], target_depth: usize) -> Result<(), &'static str> {
    if layers != CAPTURE_LAYERS {
        return Err("DSpark requires exactly capture layers 40,41,42 in that order");
    }
    if layers.iter().any(|&layer| layer >= target_depth) {
        return Err("DSpark capture layers exceed the target decoder depth");
    }
    Ok(())
}

pub fn parse_capture_layers(
    value: Option<&OsStr>,
    target_depth: usize,
) -> Result<[usize; 3], &'static str> {
    let mut layers = CAPTURE_LAYERS;
    if let Some(value) = value {
        let text = value
            .to_str()
            .ok_or("ATLAS_DSPARK_CAPTURE_LAYERS must be valid UTF-8")?;
        let mut fields = text.split(',');
        for layer in &mut layers {
            let field = fields
                .next()
                .ok_or("DSpark capture requires exactly three layers")?
                .trim();
            if field.is_empty() || !field.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(
                    "ATLAS_DSPARK_CAPTURE_LAYERS requires three unsigned decimal layer IDs",
                );
            }
            *layer = field
                .parse()
                .map_err(|_| "DSpark capture layer ID overflow")?;
        }
        if fields.next().is_some() {
            return Err("DSpark capture requires exactly three layers");
        }
    }
    validate_capture_layers(&layers, target_depth)?;
    Ok(layers)
}

/// Explicit malformed layer settings fail even when capture is dormant. An
/// ordinary run with no capture/dump/layer setting remains completely inert.
pub fn capture_layers_plan(
    capture: bool,
    dump: bool,
    value: Option<&OsStr>,
    target_depth: usize,
) -> Result<Option<[usize; 3]>, &'static str> {
    if !capture && !dump && value.is_none() {
        return Ok(None);
    }
    let layers = parse_capture_layers(value, target_depth)?;
    Ok((capture || dump).then_some(layers))
}

/// Environment adapter; syntax and policy live in the pure functions above.
pub fn capture_layers_from_env(target_depth: usize) -> Result<Option<[usize; 3]>, &'static str> {
    let capture = std::env::var("ATLAS_DSPARK_CAPTURE").as_deref() == Ok("1");
    let dump = std::env::var("ATLAS_DSPARK_DUMP").is_ok_and(|value| !value.is_empty());
    let value = std::env::var_os("ATLAS_DSPARK_CAPTURE_LAYERS");
    capture_layers_plan(capture, dump, value.as_deref(), target_depth)
}
