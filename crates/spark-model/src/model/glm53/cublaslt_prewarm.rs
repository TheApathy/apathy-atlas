// SPDX-License-Identifier: AGPL-3.0-only

//! Optional load-time cuBLASLt matmul prewarm in disposable GLM scratch.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::ffi::OsStr;

const PROBE_SIDE: u32 = 16;
const PROBE_MATRIX_BYTES: u64 = PROBE_SIDE as u64 * PROBE_SIDE as u64 * 2;
const PROBE_BYTES: usize = PROBE_MATRIX_BYTES as usize * 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProbePlan {
    act: u64,
    weight: u64,
    out: u64,
    end: u64,
}

impl ProbePlan {
    fn bind(base: DevicePtr, bytes: usize) -> Result<Self> {
        ensure!(
            base != DevicePtr::NULL,
            "GLM cuBLASLt prewarm scratch is null"
        );
        ensure!(
            base.0.is_multiple_of(256),
            "GLM cuBLASLt prewarm scratch is not 256-byte aligned"
        );
        ensure!(
            bytes >= PROBE_BYTES,
            "GLM cuBLASLt prewarm scratch is smaller than {PROBE_BYTES} bytes"
        );
        let weight = base
            .0
            .checked_add(PROBE_MATRIX_BYTES)
            .context("GLM cuBLASLt prewarm weight pointer overflow")?;
        let out = weight
            .checked_add(PROBE_MATRIX_BYTES)
            .context("GLM cuBLASLt prewarm output pointer overflow")?;
        let end = out
            .checked_add(PROBE_MATRIX_BYTES)
            .context("GLM cuBLASLt prewarm extent overflow")?;
        Ok(Self {
            act: base.0,
            weight,
            out,
            end,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Setting(bool);

impl Setting {
    fn parse(value: Option<&OsStr>) -> Result<Self> {
        match value.map(OsStr::as_encoded_bytes) {
            None | Some(b"0") => Ok(Self(false)),
            Some(b"1") => Ok(Self(true)),
            Some(_) => bail!("ATLAS_GLM53_CUBLASLT_PREWARM must be 0 or 1"),
        }
    }

    pub(super) fn from_env() -> Result<Self> {
        Self::parse(std::env::var_os("ATLAS_GLM53_CUBLASLT_PREWARM").as_deref())
    }

    pub(super) fn initialize(
        self,
        gpu: &dyn GpuBackend,
        scratch: DevicePtr,
        scratch_bytes: usize,
    ) -> Result<()> {
        if self.0 {
            let plan = ProbePlan::bind(scratch, scratch_bytes)?;
            let stream = gpu.default_stream();
            spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                plan.act,
                plan.weight,
                plan.out,
                PROBE_SIDE,
                PROBE_SIDE,
                PROBE_SIDE,
                stream,
            )
            .context("executing the GLM cuBLASLt prewarm matmul")?;
            gpu.synchronize(stream)
                .context("synchronizing the GLM cuBLASLt prewarm matmul")?;
            tracing::info!(
                matrix_side = PROBE_SIDE,
                scratch_bytes = PROBE_BYTES,
                "GLM cuBLASLt matmul prewarmed before serving"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selector_is_explicit_default_off_and_fail_closed() {
        assert_eq!(Setting::parse(None).unwrap(), Setting(false));
        assert_eq!(
            Setting::parse(Some(OsStr::new("0"))).unwrap(),
            Setting(false)
        );
        assert_eq!(
            Setting::parse(Some(OsStr::new("1"))).unwrap(),
            Setting(true)
        );
        assert!(Setting::parse(Some(OsStr::new("true"))).is_err());
        assert!(Setting::parse(Some(OsStr::new(""))).is_err());
    }

    #[test]
    fn probe_plan_uses_three_disjoint_aligned_bf16_matrices() {
        let plan = ProbePlan::bind(DevicePtr(0x1_0000), PROBE_BYTES).unwrap();
        assert_eq!(plan.act, 0x1_0000);
        assert_eq!(plan.weight, 0x1_0200);
        assert_eq!(plan.out, 0x1_0400);
        assert_eq!(plan.end, 0x1_0600);
    }

    #[test]
    fn probe_plan_rejects_invalid_scratch_without_wrapping() {
        assert!(ProbePlan::bind(DevicePtr::NULL, PROBE_BYTES).is_err());
        assert!(ProbePlan::bind(DevicePtr(0x1_0001), PROBE_BYTES).is_err());
        assert!(ProbePlan::bind(DevicePtr(0x1_0000), PROBE_BYTES - 1).is_err());
        assert!(ProbePlan::bind(DevicePtr(u64::MAX - 255), PROBE_BYTES).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn selector_rejects_non_utf8_without_echoing_it() {
        use std::os::unix::ffi::OsStrExt;

        let error = Setting::parse(Some(OsStr::from_bytes(&[0xff]))).unwrap_err();
        assert_eq!(
            error.to_string(),
            "ATLAS_GLM53_CUBLASLT_PREWARM must be 0 or 1"
        );
    }
}
