// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded host retirement for one GLM-5.3 DFlash2 proposal.

use std::ffi::OsStr;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::GgmlIqBuffer;

const PATH_TOKENS: usize = 7;
const PATH_BYTES: usize = PATH_TOKENS * std::mem::size_of::<u32>();
const STATUS_BYTES: usize = std::mem::size_of::<u32>();
const MAX_SPAN_BYTES: usize = 4096;
const VOCAB_SIZE: u32 = 154_880;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model::glm53) struct CoalescedReadbackSpan {
    pub source: DevicePtr,
    pub bytes: usize,
    pub status_offset: usize,
    pub path_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model::glm53) enum ProposalReadback {
    Separate,
    Coalesced,
}

impl ProposalReadback {
    pub fn parse(value: Option<&OsStr>) -> Result<Self> {
        let value = value
            .map(|value| {
                value
                    .to_str()
                    .context("ATLAS_GLM53_DFLASH2_COALESCED_READBACK must be valid UTF-8")
            })
            .transpose()?;
        match value {
            None | Some("0") => Ok(Self::Separate),
            Some("1") => Ok(Self::Coalesced),
            Some(_) => {
                anyhow::bail!("ATLAS_GLM53_DFLASH2_COALESCED_READBACK must be absent, 0, or 1")
            }
        }
    }

    pub fn is_coalesced(self) -> bool {
        self == Self::Coalesced
    }

    pub fn read(
        self,
        gpu: &dyn GpuBackend,
        path: GgmlIqBuffer,
        status: GgmlIqBuffer,
        stream: u64,
    ) -> Result<[u32; PATH_TOKENS]> {
        let mut path_bytes = [0u8; PATH_BYTES];
        let mut status_bytes = [0u8; STATUS_BYTES];
        match self {
            Self::Separate => {
                gpu.copy_d2h_on_stream(path.ptr, &mut path_bytes, stream)?;
                gpu.copy_d2h_on_stream(status.ptr, &mut status_bytes, stream)?;
            }
            Self::Coalesced => {
                let span = coalesced_readback_span(status, path)?;
                let mut staging = [0u8; MAX_SPAN_BYTES];
                gpu.copy_d2h_on_stream(span.source, &mut staging[..span.bytes], stream)?;
                status_bytes.copy_from_slice(
                    &staging[span.status_offset..span.status_offset + STATUS_BYTES],
                );
                path_bytes
                    .copy_from_slice(&staging[span.path_offset..span.path_offset + PATH_BYTES]);
            }
        }
        ensure!(
            u32::from_le_bytes(status_bytes) == 0,
            "GLM DFlash2 selector rejected device output"
        );
        let mut result = [0u32; PATH_TOKENS];
        for (index, chunk) in path_bytes.chunks_exact(4).enumerate() {
            result[index] = u32::from_le_bytes(chunk.try_into().unwrap());
            ensure!(
                result[index] < VOCAB_SIZE,
                "GLM DFlash2 proposed out-of-vocabulary token"
            );
        }
        Ok(result)
    }
}

pub(in crate::model::glm53) fn coalesced_readback_span(
    status: GgmlIqBuffer,
    path: GgmlIqBuffer,
) -> Result<CoalescedReadbackSpan> {
    ensure!(
        status.ptr != DevicePtr::NULL
            && path.ptr != DevicePtr::NULL
            && status.bytes == STATUS_BYTES
            && path.bytes == PATH_BYTES,
        "GLM DFlash2 coalesced readback requires exact status/path extents"
    );
    let status_end = status
        .ptr
        .0
        .checked_add(u64::try_from(status.bytes)?)
        .context("GLM DFlash2 coalesced status address overflow")?;
    ensure!(
        path.ptr.0 >= status_end,
        "GLM DFlash2 coalesced readback requires status before non-overlapping path"
    );
    let path_offset = usize::try_from(path.ptr.0 - status.ptr.0)?;
    let bytes = path_offset
        .checked_add(path.bytes)
        .context("GLM DFlash2 coalesced readback span overflow")?;
    ensure!(
        bytes <= MAX_SPAN_BYTES,
        "GLM DFlash2 coalesced readback span exceeds bounded staging"
    );
    Ok(CoalescedReadbackSpan {
        source: status.ptr,
        bytes,
        status_offset: 0,
        path_offset,
    })
}
