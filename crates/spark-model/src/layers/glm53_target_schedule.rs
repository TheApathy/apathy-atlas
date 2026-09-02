// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only forward ordering and bounded workspace for GLM-5.3-Flash.
//!
//! This is a sealed plan, not a `TransformerLayer` or runtime implementation.
//! It deliberately does not reuse the generic DeepSeek FP32 hidden buffers.

use anyhow::{Context, Result, bail};

const ALIGNMENT: u64 = 256;
const BF16_BYTES: u32 = 2;
/// The mHC residual streams are the one workspace region held in F32. They are
/// rewritten and re-read once per layer for 45 layers by a fold that amplifies
/// its input error ~1.3x/layer, so the two bf16 roundings a layer spent there
/// dominated the divergence. `hidden_a`/`hidden_b`/`collapsed` stay BF16.
const STREAM_BYTES: u32 = 4;
const MAX_CHUNK_TOKENS: u32 = 65_520;
const HIDDEN_SIZE: u32 = 4_096;
const HC_STREAMS: u32 = 4;
const TARGET_LAYERS: u32 = 45;
const CAPTURE_LAYERS: [u32; 5] = [5, 14, 24, 33, 42];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetAttentionKind {
    Kda,
    Dsa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetFfnKind {
    Dense,
    Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetGeometry {
    pub chunk_tokens: u32,
    pub hidden_size: u32,
    pub hc_streams: u32,
    pub target_layers: u32,
    pub stream_element_bytes: u32,
}

impl Glm53TargetGeometry {
    pub fn exact(chunk_tokens: u32) -> Self {
        Self {
            chunk_tokens,
            hidden_size: HIDDEN_SIZE,
            hc_streams: HC_STREAMS,
            target_layers: TARGET_LAYERS,
            stream_element_bytes: STREAM_BYTES,
        }
    }

    fn validate(self) -> Result<()> {
        if self.chunk_tokens == 0 || self.chunk_tokens > MAX_CHUNK_TOKENS {
            bail!("GLM target workspace requires chunk tokens in 1..=65520");
        }
        if self.hidden_size != HIDDEN_SIZE
            || self.hc_streams != HC_STREAMS
            || self.target_layers != TARGET_LAYERS
        {
            bail!("GLM target forward geometry drift");
        }
        if self.stream_element_bytes != STREAM_BYTES {
            bail!("GLM target workspace mHC streams must be FP32, never BF16");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetWorkspaceRegion {
    pub offset_bytes: u64,
    pub payload_bytes: u64,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetWorkspace {
    pub geometry: Glm53TargetGeometry,
    pub hidden_a: Glm53TargetWorkspaceRegion,
    pub hidden_b: Glm53TargetWorkspaceRegion,
    pub collapsed: Glm53TargetWorkspaceRegion,
    pub widened_hc: Glm53TargetWorkspaceRegion,
    pub hyper_post: Glm53TargetWorkspaceRegion,
    pub hyper_comb: Glm53TargetWorkspaceRegion,
    pub arena_bytes: u64,
}

impl Glm53TargetWorkspace {
    pub fn validate(self) -> Result<()> {
        self.geometry.validate()?;
        let expected = build_workspace(self.geometry)?;
        if self != expected {
            bail!("GLM target BF16 workspace layout or extent drift");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetEvent {
    ExpandMhc,
    PreAttention {
        layer: u32,
    },
    Attention {
        layer: u32,
        kind: Glm53TargetAttentionKind,
    },
    PostAttention {
        layer: u32,
    },
    Ffn {
        layer: u32,
        kind: Glm53TargetFfnKind,
    },
    PostFfn {
        layer: u32,
    },
    CaptureWidenedMhc {
        layer: u32,
        slot: u32,
    },
    OrderedMean,
    FinalNormF32,
    LmHeadF32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53TargetSchedule {
    pub geometry: Glm53TargetGeometry,
    pub workspace: Glm53TargetWorkspace,
    events: Vec<Glm53TargetEvent>,
}

impl Glm53TargetSchedule {
    pub fn new(geometry: Glm53TargetGeometry) -> Result<Self> {
        geometry.validate()?;
        let workspace = build_workspace(geometry)?;
        let events = build_events()?;
        let schedule = Self {
            geometry,
            workspace,
            events,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn events(&self) -> &[Glm53TargetEvent] {
        &self.events
    }

    pub fn validate(&self) -> Result<()> {
        self.geometry.validate()?;
        if self.workspace.geometry != self.geometry {
            bail!("GLM target workspace geometry does not match its schedule");
        }
        self.workspace.validate()?;
        validate_events(&self.events)
    }
}

fn attention_kind(layer: u32) -> Glm53TargetAttentionKind {
    if layer % 4 == 3 {
        Glm53TargetAttentionKind::Dsa
    } else {
        Glm53TargetAttentionKind::Kda
    }
}

fn ffn_kind(layer: u32) -> Glm53TargetFfnKind {
    if layer < 3 {
        Glm53TargetFfnKind::Dense
    } else {
        Glm53TargetFfnKind::Moe
    }
}

fn build_events() -> Result<Vec<Glm53TargetEvent>> {
    let mut events = Vec::with_capacity(234);
    events.push(Glm53TargetEvent::ExpandMhc);
    let mut capture_slot = 0usize;
    for layer in 0..TARGET_LAYERS {
        events.push(Glm53TargetEvent::PreAttention { layer });
        events.push(Glm53TargetEvent::Attention {
            layer,
            kind: attention_kind(layer),
        });
        events.push(Glm53TargetEvent::PostAttention { layer });
        events.push(Glm53TargetEvent::Ffn {
            layer,
            kind: ffn_kind(layer),
        });
        events.push(Glm53TargetEvent::PostFfn { layer });
        if CAPTURE_LAYERS.get(capture_slot) == Some(&layer) {
            events.push(Glm53TargetEvent::CaptureWidenedMhc {
                layer,
                slot: u32::try_from(capture_slot)?,
            });
            capture_slot += 1;
        }
    }
    events.push(Glm53TargetEvent::OrderedMean);
    events.push(Glm53TargetEvent::FinalNormF32);
    events.push(Glm53TargetEvent::LmHeadF32);
    Ok(events)
}

fn validate_events(events: &[Glm53TargetEvent]) -> Result<()> {
    let mut cursor = 0usize;
    expect_event(events, &mut cursor, Glm53TargetEvent::ExpandMhc)?;
    let mut capture_slot = 0usize;
    for layer in 0..TARGET_LAYERS {
        for expected in [
            Glm53TargetEvent::PreAttention { layer },
            Glm53TargetEvent::Attention {
                layer,
                kind: attention_kind(layer),
            },
            Glm53TargetEvent::PostAttention { layer },
            Glm53TargetEvent::Ffn {
                layer,
                kind: ffn_kind(layer),
            },
            Glm53TargetEvent::PostFfn { layer },
        ] {
            expect_event(events, &mut cursor, expected)?;
        }
        if CAPTURE_LAYERS.get(capture_slot) == Some(&layer) {
            expect_event(
                events,
                &mut cursor,
                Glm53TargetEvent::CaptureWidenedMhc {
                    layer,
                    slot: u32::try_from(capture_slot)?,
                },
            )?;
            capture_slot += 1;
        }
    }
    expect_event(events, &mut cursor, Glm53TargetEvent::OrderedMean)?;
    expect_event(events, &mut cursor, Glm53TargetEvent::FinalNormF32)?;
    expect_event(events, &mut cursor, Glm53TargetEvent::LmHeadF32)?;
    if cursor != events.len() || capture_slot != CAPTURE_LAYERS.len() {
        bail!("GLM target forward schedule has extra events or missing captures");
    }
    Ok(())
}

fn expect_event(
    events: &[Glm53TargetEvent],
    cursor: &mut usize,
    expected: Glm53TargetEvent,
) -> Result<()> {
    if events.get(*cursor) != Some(&expected) {
        bail!("GLM target forward event {cursor} is missing or reordered");
    }
    *cursor = cursor
        .checked_add(1)
        .context("GLM target forward event cursor overflow")?;
    Ok(())
}

fn build_workspace(geometry: Glm53TargetGeometry) -> Result<Glm53TargetWorkspace> {
    geometry.validate()?;
    let rows = u64::from(geometry.chunk_tokens);
    let hidden = checked_bytes(rows, u64::from(HIDDEN_SIZE), u64::from(BF16_BYTES))?;
    let widened = checked_bytes(
        rows,
        u64::from(HC_STREAMS)
            .checked_mul(u64::from(HIDDEN_SIZE))
            .context("GLM target widened width overflow")?,
        u64::from(STREAM_BYTES),
    )?;
    let post = checked_bytes(rows, u64::from(HC_STREAMS), u64::from(BF16_BYTES))?;
    let comb = checked_bytes(
        rows,
        u64::from(HC_STREAMS)
            .checked_mul(u64::from(HC_STREAMS))
            .context("GLM target comb width overflow")?,
        u64::from(BF16_BYTES),
    )?;
    let mut cursor = 0u64;
    let hidden_a = place(&mut cursor, hidden)?;
    let hidden_b = place(&mut cursor, hidden)?;
    let collapsed = place(&mut cursor, hidden)?;
    let widened_hc = place(&mut cursor, widened)?;
    let hyper_post = place(&mut cursor, post)?;
    let hyper_comb = place(&mut cursor, comb)?;
    Ok(Glm53TargetWorkspace {
        geometry,
        hidden_a,
        hidden_b,
        collapsed,
        widened_hc,
        hyper_post,
        hyper_comb,
        arena_bytes: align_up(cursor)?,
    })
}

fn checked_bytes(rows: u64, width: u64, element_bytes: u64) -> Result<u64> {
    rows.checked_mul(width)
        .and_then(|values| values.checked_mul(element_bytes))
        .context("GLM target workspace byte overflow")
}

fn align_up(value: u64) -> Result<u64> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|end| end & !(ALIGNMENT - 1))
        .context("GLM target workspace alignment overflow")
}

fn place(cursor: &mut u64, payload_bytes: u64) -> Result<Glm53TargetWorkspaceRegion> {
    let offset_bytes = align_up(*cursor)?;
    let allocation_bytes = align_up(payload_bytes)?;
    *cursor = offset_bytes
        .checked_add(allocation_bytes)
        .context("GLM target workspace placement overflow")?;
    Ok(Glm53TargetWorkspaceRegion {
        offset_bytes,
        payload_bytes,
        allocation_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_layer_order_attention_ffn_and_capture_schedule_is_pinned() {
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(8)).unwrap();
        assert_eq!(schedule.events().len(), 234);
        assert_eq!(schedule.events()[0], Glm53TargetEvent::ExpandMhc);
        assert_eq!(
            schedule
                .events()
                .iter()
                .filter(|event| matches!(
                    event,
                    Glm53TargetEvent::Attention {
                        kind: Glm53TargetAttentionKind::Kda,
                        ..
                    }
                ))
                .count(),
            34
        );
        assert_eq!(
            schedule
                .events()
                .iter()
                .filter(|event| matches!(
                    event,
                    Glm53TargetEvent::Attention {
                        kind: Glm53TargetAttentionKind::Dsa,
                        ..
                    }
                ))
                .count(),
            11
        );
        let captures: Vec<_> = schedule
            .events()
            .iter()
            .filter_map(|event| match event {
                Glm53TargetEvent::CaptureWidenedMhc { layer, slot } => Some((*layer, *slot)),
                _ => None,
            })
            .collect();
        assert_eq!(captures, [(5, 0), (14, 1), (24, 2), (33, 3), (42, 4)]);
        assert_eq!(
            &schedule.events()[231..],
            &[
                Glm53TargetEvent::OrderedMean,
                Glm53TargetEvent::FinalNormF32,
                Glm53TargetEvent::LmHeadF32,
            ]
        );
        schedule.validate().unwrap();
    }

    #[test]
    fn workspace_is_chunk_bounded_f32_streams_and_aligned() {
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(65_520)).unwrap();
        assert_eq!(schedule.workspace.hidden_a.payload_bytes, 536_739_840);
        assert_eq!(schedule.workspace.widened_hc.payload_bytes, 4_293_918_720);
        assert_eq!(schedule.workspace.hyper_post.payload_bytes, 524_160);
        assert_eq!(schedule.workspace.hyper_comb.payload_bytes, 2_096_640);
        assert_eq!(schedule.workspace.arena_bytes, 5_906_759_168);
        for region in [
            schedule.workspace.hidden_a,
            schedule.workspace.hidden_b,
            schedule.workspace.collapsed,
            schedule.workspace.widened_hc,
            schedule.workspace.hyper_post,
            schedule.workspace.hyper_comb,
        ] {
            assert_eq!(region.offset_bytes % ALIGNMENT, 0);
            assert_eq!(region.allocation_bytes % ALIGNMENT, 0);
        }
        assert!(Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1_048_576)).is_err());
    }

    #[test]
    fn forged_geometry_and_bf16_stream_workspace_fail_closed() {
        assert!(Glm53TargetSchedule::new(Glm53TargetGeometry::exact(0)).is_err());
        assert!(Glm53TargetSchedule::new(Glm53TargetGeometry::exact(65_521)).is_err());
        let mutations: [fn(&mut Glm53TargetGeometry); 4] = [
            |geometry: &mut Glm53TargetGeometry| geometry.hidden_size = 2_048,
            |geometry: &mut Glm53TargetGeometry| geometry.hc_streams = 2,
            |geometry: &mut Glm53TargetGeometry| geometry.target_layers = 44,
            |geometry: &mut Glm53TargetGeometry| geometry.stream_element_bytes = 2,
        ];
        for mutate in mutations {
            let mut geometry = Glm53TargetGeometry::exact(8);
            mutate(&mut geometry);
            assert!(Glm53TargetSchedule::new(geometry).is_err());
        }
        let mut schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(8)).unwrap();
        schedule.workspace.widened_hc.payload_bytes *= 2;
        assert!(schedule.validate().is_err());
    }

    #[test]
    fn missing_reordered_and_duplicate_captures_are_rejected() {
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(8)).unwrap();
        let capture = schedule
            .events
            .iter()
            .position(|event| matches!(event, Glm53TargetEvent::CaptureWidenedMhc { .. }))
            .unwrap();
        let mut missing = schedule.clone();
        missing.events.remove(capture);
        assert!(missing.validate().is_err());
        let mut reordered = schedule.clone();
        reordered.events.swap(capture - 1, capture);
        assert!(reordered.validate().is_err());
        let mut duplicate = schedule;
        duplicate.events.insert(capture, duplicate.events[capture]);
        assert!(duplicate.validate().is_err());
    }
}
