// SPDX-License-Identifier: AGPL-3.0-only
//! Stage metadata only: no activation transfers or numerical-parity claim.
use anyhow::{Context, Result, ensure};
use spark_model::layers::ops::GgmlIqBuffer;
use spark_model::model::glm53::{Glm53Dflash2ProbeObserver, ProbeLayout, ProbeStage};
use spark_runtime::gpu::GpuBackend;

pub struct TimingObserver {
    layout: ProbeLayout,
    context: u32,
    stream: u64,
    cursor: usize,
}
impl TimingObserver {
    pub fn new(layout: ProbeLayout, context: u32, stream: u64) -> Result<Self> {
        ensure!(context > 0, "timing requires nonempty committed context");
        ensure!(
            layout.projected_context().is_none(),
            "timing rejects projected-input capture"
        );
        Ok(Self {
            layout,
            context,
            stream,
            cursor: 0,
        })
    }
    pub fn record(
        &mut self,
        stage: ProbeStage,
        source: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(stream == self.stream, "timing observer stream changed");
        ensure!(
            self.layout.stages().get(self.cursor) == Some(&stage),
            "timing stage missing, repeated or out of order"
        );
        ensure!(
            bytes == self.layout.bytes(stage)?,
            "timing stage extent changed"
        );
        ensure!(
            source != 0 && source % 2 == 0,
            "timing stage pointer invalid"
        );
        source
            .checked_add(u64::try_from(bytes)?)
            .context("timing stage pointer overflow")?;
        self.cursor += 1;
        Ok(())
    }
    pub fn finish(&self) -> Result<usize> {
        ensure!(
            self.cursor == self.layout.stages().len(),
            "timing observer incomplete"
        );
        Ok(self.cursor)
    }
}
impl Glm53Dflash2ProbeObserver for TimingObserver {
    fn admit(&self, layout: &ProbeLayout, context: u32, stream: u64) -> Result<()> {
        ensure!(
            self.cursor == 0 && context == self.context && stream == self.stream,
            "timing admission history/context/stream drift"
        );
        ensure!(
            layout.projected_context().is_none()
                && layout.vocab() == self.layout.vocab()
                && layout.stages() == self.layout.stages(),
            "timing layout identity changed"
        );
        for &stage in layout.stages() {
            ensure!(
                layout.bytes(stage)? == self.layout.bytes(stage)?,
                "timing layout extent changed"
            );
        }
        Ok(())
    }
    fn observe(
        &mut self,
        stage: ProbeStage,
        source: GgmlIqBuffer,
        _gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.record(stage, source.ptr.0, source.bytes, stream)
    }
}
