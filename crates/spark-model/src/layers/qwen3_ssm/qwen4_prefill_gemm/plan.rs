// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::OsStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mode {
    #[default]
    Off,
    Qkvz,
    Out,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Projection {
    Qkvz,
    Output,
}

pub(crate) fn parse_mode(value: Option<&OsStr>) -> Result<Mode, &'static str> {
    match value.map(OsStr::to_str) {
        None | Some(Some("0")) => Ok(Mode::Off),
        Some(Some("qkvz")) => Ok(Mode::Qkvz),
        Some(Some("out")) => Ok(Mode::Out),
        Some(Some("all")) => Ok(Mode::All),
        _ => Err("expected absent/0, qkvz, out, or all"),
    }
}

impl Mode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Qkvz => "qkvz",
            Self::Out => "out",
            Self::All => "all",
        }
    }

    pub(crate) fn uses_gemm(self, projection: Projection) -> bool {
        matches!(
            (self, projection),
            (Self::All, _) | (Self::Qkvz, Projection::Qkvz) | (Self::Out, Projection::Output)
        )
    }

    pub(crate) fn admit(self, exact: bool, check: bool) -> Result<(), &'static str> {
        if self == Self::Off {
            return Ok(());
        }
        if !exact {
            return Err("requires ATLAS_QWEN4_PREFILL_SSM_EXACT=1");
        }
        if check {
            return Err(
                "numerical GEMM candidate is incompatible with strict ATLAS_QWEN4_PREFILL_SSM_CHECK=1",
            );
        }
        Ok(())
    }

    pub(crate) fn validate_handle(self, handle: u64) -> Result<(), &'static str> {
        if self != Self::Off && handle == 0 {
            return Err("selected original-layout w4a16_gemm kernel is unavailable");
        }
        Ok(())
    }
}
