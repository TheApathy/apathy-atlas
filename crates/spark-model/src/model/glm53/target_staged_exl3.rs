// SPDX-License-Identifier: AGPL-3.0-only

//! Shared verifier staging. The token wrapper preserves the shipping embedding API.

use super::*;
use crate::model::glm53::prefill_capture_ingest::CaptureReceipt;

impl Glm53Exl3Model {
    pub(super) fn verify_inputs_staged(
        &self,
        input_rows: usize,
        stream: u64,
        fill: impl FnOnce(GgmlIqBuffer, u32) -> Result<()>,
    ) -> Result<(DevicePtr, u32, u32)> {
        self.verify_inputs_staged_with_capture(input_rows, stream, None, fill)
    }

    pub(super) fn verify_inputs_staged_with_capture(
        &self,
        input_rows: usize,
        stream: u64,
        mut capture: Option<&mut CaptureReceipt>,
        fill: impl FnOnce(GgmlIqBuffer, u32) -> Result<()>,
    ) -> Result<(DevicePtr, u32, u32)> {
        self.ensure_verify_healthy()?;
        ensure!(
            capture.is_none() || glm53_layer_major_prefill_active(),
            "large capture binding requires the explicit layer-major prefill scope"
        );
        let max_rows = if glm53_layer_major_prefill_active() {
            GLM53_EXL3_MAX_WIDE_ROWS
        } else {
            8
        };
        ensure!(
            (2..=max_rows).contains(&input_rows),
            "GLM EXL3 verifier needs 2..={max_rows} rows in this execution scope"
        );
        let rows = u32::try_from(input_rows)?;
        let graph_groups =
            self.ffn_graphs
                .eligible(self.gpu.as_ref(), rows, stream, capture.is_some())?;
        let (position, nonce) = {
            let mut state = self.state.lock().unwrap();
            let position = state.position;
            if let Some(receipt) = capture.as_deref() {
                receipt.validate_staged(rows, position)?;
            }
            ensure!(
                position
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
                "GLM EXL3 verifier chunk exceeds sequence capacity"
            );
            state.nonce = state.nonce.wrapping_add(1).max(1);
            (position, state.nonce)
        };
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(rows))?;
        let base = DevicePtr((self.wide_workspace_allocation.0 + 255) & !255);
        let workspace =
            Glm53BoundWorkspace::bind(&schedule.workspace, base, schedule.workspace.arena_bytes)?;
        fill(workspace.collapsed, position)?;
        let dispatcher = Glm53Dispatcher::new_exl3(
            self.gpu.as_ref(),
            rows,
            workspace,
            self.hyper.clone(),
            GgmlIqBuffer {
                ptr: self.weights.output_norm.ptr(),
                bytes: self.weights.output_norm.bytes(),
            },
            self.scratch,
            self.captures,
            &self.weights.layers,
            &self.moe_tables,
            Glm53AttentionBinding {
                kda_states: self.kda_states.clone(),
                kda_conv: self.kda_conv.clone(),
                dsa_cache: self.dsa_cache.clone(),
                geometry: Glm53DsaLayerGeometry {
                    position,
                    capacity: self.capacity,
                    nonce,
                },
                prefix: self.prefix_commit,
            },
            &self.weights.lm_head,
            GgmlIqBuffer {
                ptr: self.logits,
                bytes: rows as usize * VOCAB as usize * 2,
            },
        )?;
        let time_events = std::env::var_os("ATLAS_GLM53_WIDE_TIMING").is_some();
        let mut events = schedule.events().iter();
        while let Some(event) = events.next() {
            if graph_groups
                && let crate::layers::Glm53TargetEvent::PostAttention { layer } = event
                && (3..=44).contains(layer)
            {
                use crate::layers::{Glm53TargetEvent, Glm53TargetFfnKind};
                ensure!(
                    events.next()
                        == Some(&Glm53TargetEvent::Ffn {
                            layer: *layer,
                            kind: Glm53TargetFfnKind::Moe
                        })
                        && events.next() == Some(&Glm53TargetEvent::PostFfn { layer: *layer }),
                    "GLM FFN graph schedule group changed"
                );
                super::super::ffn_graph::with_failure_owner(
                    || {
                        self.ffn_graphs.execute(
                            &dispatcher,
                            self.gpu.as_ref(),
                            rows,
                            *layer,
                            super::super::ffn_graph::Binding {
                                stream,
                                owners: [
                                    self.wide_workspace_allocation.0,
                                    self.scratch_allocation.0,
                                    self.moe_tables_allocation.0,
                                ],
                            },
                        )
                    },
                    || self.poison_verify(stream),
                )?;
                continue;
            }
            let event_started = std::time::Instant::now();
            dispatcher
                .dispatch_with_prefill_capture(
                    self.gpu.as_ref(),
                    event,
                    stream,
                    capture.as_deref_mut(),
                )
                .with_context(|| format!("GLM EXL3 wide verify failed at {event:?}"))?;
            if time_events {
                self.gpu.synchronize(stream)?;
                eprintln!(
                    "WIDE_EVENT event={event:?} seconds={:.9}",
                    event_started.elapsed().as_secs_f64()
                );
            }
        }
        self.gpu.synchronize(stream)?;
        dump_last_row_logits(self.gpu.as_ref(), self.logits, rows)?;
        Ok((self.logits, position, rows))
    }
}

/// Correctness oracle (private tree): when `ATLAS_GLM53_LOGITS_DUMP=<dir>` is
/// set, write the last row's BF16 logits (154,880 x 2 bytes, raw) after every
/// staged forward and every decode walk as `<dir>/logits-<n>.bf16`. One
/// synchronous 310 KB D2H copy after the existing completion fence; never on by
/// default.
pub(super) fn dump_last_row_logits(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    rows: u32,
) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let Some(dir) = std::env::var_os("ATLAS_GLM53_LOGITS_DUMP") else {
        return Ok(());
    };
    let row_bytes = VOCAB as usize * 2;
    // ATLAS_GLM53_LOGITS_DUMP_ALL=1 dumps every row (teacher-forced gate);
    // needs ATLAS_GLM53_EXL3_LAST_ROW_HEAD=0 so the head is computed for all rows.
    let all = std::env::var("ATLAS_GLM53_LOGITS_DUMP_ALL").as_deref() == Ok("1");
    let (offset, bytes, suffix) = if all {
        (0, rows as usize * row_bytes, format!("-r{rows}"))
    } else {
        ((rows as usize - 1) * row_bytes, row_bytes, String::new())
    };
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(logits.offset(offset), &mut host)?;
    let index = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::path::Path::new(&dir).join(format!("logits-{index}{suffix}.bf16"));
    std::fs::create_dir_all(&dir).context("GLM logits dump dir")?;
    std::fs::write(&path, &host).with_context(|| format!("GLM logits dump {}", path.display()))?;
    Ok(())
}
