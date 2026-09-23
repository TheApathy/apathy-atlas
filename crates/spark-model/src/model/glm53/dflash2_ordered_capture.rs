// SPDX-License-Identifier: AGPL-3.0-only

//! Preserve scalar FC/RMS arithmetic while ingesting a wide capture bank.

use super::*;
use crate::model::glm53::ordered_capture::{OrderedCaptureIo, OrderedCapturePlan};

struct CaptureIo<'a> {
    runtime: &'a Glm53Dflash2Runtime,
    target: &'a Glm53Exl3Model,
}

impl OrderedCaptureIo for CaptureIo<'_> {
    fn project_row(
        &mut self,
        capture_row: u32,
        destination_position: u32,
        stream: u64,
    ) -> Result<()> {
        self.runtime
            .project_capture_rows(self.target, capture_row, 1, destination_position, stream)
    }

    fn fence(&mut self, stream: u64) -> Result<()> {
        self.target.gpu().synchronize(stream)
    }
}

impl Glm53Dflash2Runtime {
    pub(in crate::model::glm53) fn observe_target_rows_ordered(
        &mut self,
        target: &Glm53Exl3Model,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        self.ensure_capture_kv_ready()?;
        let plan = OrderedCapturePlan::new(
            self.context_tokens,
            rows,
            target.position(),
            self.context_tokens,
            self.context_capacity(),
        )?;
        let end = plan.execute(
            &mut CaptureIo {
                runtime: self,
                target,
            },
            stream,
        )?;
        self.context_tokens = end;
        Ok(())
    }
}
