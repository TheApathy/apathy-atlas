// SPDX-License-Identifier: AGPL-3.0-only

//! Value-only startup policy and restoring scope for prepared native rows.

use std::cell::Cell;
use std::ffi::OsStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeRowsSetting(bool, bool);

impl NativeRowsSetting {
    pub(crate) fn parse(value: Option<&OsStr>) -> Result<Self, &'static str> {
        match value {
            None => Ok(Self(false, false)),
            Some(value) if value == OsStr::new("0") => Ok(Self(false, false)),
            Some(value) if value == OsStr::new("1") => Ok(Self(true, false)),
            Some(_) => Err("ATLAS_GLM53_NATIVE_ROW_PREPARED must be absent or exactly 0 or 1"),
        }
    }

    pub(crate) fn enabled(self) -> bool {
        self.0
    }

    pub(crate) fn parse_with_batched(
        prepared: Option<&OsStr>,
        batched: Option<&OsStr>,
    ) -> Result<Self, &'static str> {
        let mut setting = Self::parse(prepared)?;
        setting.1 = match batched {
            None => false,
            Some(value) if value == OsStr::new("0") => false,
            Some(value) if value == OsStr::new("1") => true,
            Some(_) => {
                return Err("ATLAS_GLM53_NATIVE_ROW_BATCHED must be absent or exactly 0 or 1");
            }
        };
        if setting.1 && !setting.0 {
            return Err(
                "ATLAS_GLM53_NATIVE_ROW_BATCHED=1 requires ATLAS_GLM53_NATIVE_ROW_PREPARED=1",
            );
        }
        Ok(setting)
    }

    pub(crate) fn batched(self) -> bool {
        self.1
    }

    pub(crate) fn validate_exact_mode(self, value: Option<&OsStr>) -> Result<(), &'static str> {
        if self.0 && value != Some(OsStr::new("1")) {
            return Err("prepared native rows require ATLAS_GLM53_EXACT_VERIFY=1");
        }
        Ok(())
    }
}

thread_local! {
    static NATIVE_ROWS_ACTIVE: Cell<NativeRowsSetting> =
        const { Cell::new(NativeRowsSetting(false, false)) };
}

impl From<bool> for NativeRowsSetting {
    fn from(enabled: bool) -> Self {
        Self(enabled, false)
    }
}

struct NativeRowsGuard(NativeRowsSetting);

impl Drop for NativeRowsGuard {
    fn drop(&mut self) {
        NATIVE_ROWS_ACTIVE.with(|active| active.set(self.0));
    }
}

pub(crate) fn with_glm53_native_rows<T>(
    setting: impl Into<NativeRowsSetting>,
    operation: impl FnOnce() -> T,
) -> T {
    let previous = NATIVE_ROWS_ACTIVE.with(|active| active.replace(setting.into()));
    let _guard = NativeRowsGuard(previous);
    operation()
}

pub(crate) fn glm53_native_rows_active() -> bool {
    NATIVE_ROWS_ACTIVE.with(|active| active.get().enabled())
}

pub(crate) fn glm53_native_rows_batched_active() -> bool {
    NATIVE_ROWS_ACTIVE.with(|active| active.get().batched())
}

pub(crate) fn native_rows_selected(exact: bool, prefill: bool, rows: u32, native: bool) -> bool {
    glm53_native_rows_active() && exact && !prefill && native && (2..=8).contains(&rows)
}
