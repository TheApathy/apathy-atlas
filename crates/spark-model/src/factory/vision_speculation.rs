// SPDX-License-Identifier: AGPL-3.0-only

//! Restricted opt-in for native Vision DSpark qualification, not promotion.

use std::ffi::OsStr;

pub const FLAG: &str = "ATLAS_DEEPSEEK_VISION_DSPARK";

/// The factory's common switch is `--speculative || --dflash`, not a legacy
/// MTP selector. A supplied drafter owns that switch; self/ngram stays separate.
/// The server rejects explicit mixed CLI modes before loading any weights.
pub fn factory_legacy_speculation(common: bool, self_speculative: bool, has_drafter: bool) -> bool {
    self_speculative || (common && !has_drafter)
}

pub const INCOMPATIBLE_FLAGS: &[&str] = &[
    "ATLAS_DFLASH_ALL_BATCHED",
    "ATLAS_DFLASH_BLOCKFORK",
    "ATLAS_DFLASH_TREE",
    "ATLAS_DSPARK_TREE",
    "ATLAS_MTP_DRAFTER_PREFILL",
    "ATLAS_MTP_CATCHUP",
    "ATLAS_DSPARK_REF_DRAFT",
    "ATLAS_VISION_HC_BF16",
    "ATLAS_DFLASH_LOW_GEAR",
    "ATLAS_V4_FORCE_NO_COMP",
    "ATLAS_V4_NO_PERROW_COMP",
    "ATLAS_MLA_NO_BATCH",
];

#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub is_vision: bool,
    pub target_only: bool,
    pub dflash_only: bool,
    pub embedded_drafter: bool,
    pub high_speed_swap: bool,
    pub verify_capacity: usize,
    pub has_adapters: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Qualification(bool);

impl Qualification {
    pub fn parse(value: Option<&OsStr>) -> Result<Self, &'static str> {
        match value.map(OsStr::to_str) {
            None | Some(Some("0")) => Ok(Self(false)),
            Some(Some("1")) => Ok(Self(true)),
            _ => Err("ATLAS_DEEPSEEK_VISION_DSPARK must be absent, 0 or 1"),
        }
    }

    pub fn from_env() -> Result<Self, &'static str> {
        Self::parse(std::env::var_os(FLAG).as_deref())
    }

    pub fn enabled(self) -> bool {
        self.0
    }

    pub fn admit(self, request: Request) -> Result<bool, &'static str> {
        if !self.0 || request.target_only {
            return Ok(false);
        }
        if !request.is_vision || !request.dflash_only || !request.embedded_drafter {
            return Err(
                "Vision DSpark qualification requires actual Vision and its own embedded DSpark via --dflash; other speculative modes/drafters remain unsupported",
            );
        }
        if request.high_speed_swap {
            return Err("Vision DSpark qualification does not support high-speed swap");
        }
        if request.verify_capacity < 6 {
            return Err("Vision DSpark requires scratch capacity for all six verification rows");
        }
        if request.has_adapters {
            return Err("Vision DSpark qualification does not support LoRA adapters");
        }
        Ok(true)
    }

    pub fn validate_runtime(self) -> Result<(), String> {
        if !self.0 {
            return Ok(());
        }
        for flag in ["ATLAS_DSPARK_CAPTURE", "ATLAS_DFLASH_MASKED_VERIFY"] {
            validate_required_runtime_value(flag, std::env::var_os(flag).as_deref())?;
        }
        for flag in INCOMPATIBLE_FLAGS {
            validate_runtime_value(flag, std::env::var_os(flag).as_deref())?;
        }
        Ok(())
    }
}

pub fn validate_required_runtime_value(key: &str, value: Option<&OsStr>) -> Result<(), String> {
    if value == Some(OsStr::new("1")) {
        Ok(())
    } else {
        Err(format!("Vision DSpark qualification requires {key}=1"))
    }
}

pub fn validate_runtime_value(key: &str, value: Option<&OsStr>) -> Result<(), String> {
    if value.is_none() || value == Some(OsStr::new("0")) {
        Ok(())
    } else {
        Err(format!(
            "Vision DSpark qualification requires {key} absent or 0"
        ))
    }
}
