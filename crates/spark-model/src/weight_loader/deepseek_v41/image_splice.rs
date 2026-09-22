// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 image input on the model side: the vision tower plus the
//! prompt-embedding splice (`merge_prompt_embeddings` in engine/vision.py).
//!
//! Python embeds the prompt with `ids.masked_fill(ids == IMAGE_PAD_ID,
//! IMAGE_SENTINEL_ID)`, then overwrites each image span with the tower output:
//! START, then per aligner row `w` IMAGE rows and a NEWLINE, then END. Here
//! that happens per prefill chunk, right after the token embedding:
//!   * a PAD position (129265) gets the SENTINEL (129264) embedding row;
//!   * a position inside a span gets that span's row.
//! Spans are located from the token ids themselves: each run of sentinels is
//! consumed image by image, using the span lengths known from `encode`, so
//! two adjacent images with no pad between them still split correctly. The
//! mapping is by ABSOLUTE position, so a span that crosses a chunk boundary
//! is completed in the next chunk, and re-running a range (e.g. a replay
//! pass) reproduces the same rows.

use std::sync::Mutex;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::deepseek_vision::DeepSeekVisionEncoder;

pub const IMAGE_SENTINEL_ID: u32 = 129_264;
pub const IMAGE_PAD_ID: u32 = 129_265;

/// Slot roles within a span (engine/vision.py `image_token_types`).
const START: u8 = 0;
const IMAGE: u8 = 1;
const NEW_LINE: u8 = 2;
const END: u8 = 3;

/// `image_token_types(llm_h, llm_w)`.
pub fn span_types(llm_h: usize, llm_w: usize) -> Vec<u8> {
    let mut t = vec![START];
    for _ in 0..llm_h {
        t.extend(std::iter::repeat_n(IMAGE, llm_w));
        t.push(NEW_LINE);
    }
    t.push(END);
    t
}

/// One contiguous copy into the chunk's embedding rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceOp {
    /// Chunk row `row` <- the SENTINEL embedding.
    Pad { row: usize },
    /// Chunk rows `row..row+count` <- span `image`'s rows `slot..slot+count`.
    Span {
        row: usize,
        image: usize,
        slot: usize,
        count: usize,
    },
}

/// Plan the copies for chunk `[start, start + t)` of a prompt whose ids at
/// absolute positions `0..seen.len()` are `seen` (the chunk must be covered).
/// `span_lens[i]` is image i's span length. Errors if the sentinels in the
/// ids do not add up to the images (a placeholder count mismatch).
pub fn plan_splice(
    seen: &[u32],
    span_lens: &[usize],
    start: usize,
    t: usize,
) -> Result<Vec<SpliceOp>> {
    ensure!(
        start + t <= seen.len(),
        "splice: chunk [{start}, {}) past the ids seen ({})",
        start + t,
        seen.len()
    );
    let mut ops = Vec::new();
    // walk from 0 so the image index and in-span offset are absolute
    let (mut image, mut slot) = (0usize, 0usize);
    let mut in_span = false;
    for (p, &id) in seen.iter().enumerate().take(start + t) {
        let in_chunk = p >= start;
        if id == IMAGE_SENTINEL_ID {
            if !in_span {
                ensure!(
                    image < span_lens.len(),
                    "more image spans in the prompt than images ({})",
                    span_lens.len()
                );
                in_span = true;
                slot = 0;
            }
            if in_chunk {
                let row = p - start;
                match ops.last_mut() {
                    Some(SpliceOp::Span {
                        row: r,
                        image: i,
                        slot: s,
                        count,
                    }) if *i == image && *r + *count == row && *s + *count == slot => *count += 1,
                    _ => ops.push(SpliceOp::Span {
                        row,
                        image,
                        slot,
                        count: 1,
                    }),
                }
            }
            slot += 1;
            if slot == span_lens[image] {
                in_span = false;
                image += 1;
            }
        } else {
            ensure!(!in_span, "image span {image} cut short at position {p}");
            if id == IMAGE_PAD_ID && in_chunk {
                ops.push(SpliceOp::Pad { row: p - start });
            }
        }
    }
    Ok(ops)
}

/// The per-chunk decision `splice` acts on. `None` = nothing to do: no image
/// was encoded for this request (the text-only path, which must stay
/// byte-identical, so it issues no GPU work at all). Image tokens without
/// encoded images are an error, never a silent no-op.
pub fn chunk_plan(
    span_lens: &[usize],
    seen: &mut Vec<u32>,
    ids: &[u32],
    start: usize,
) -> Result<Option<Vec<SpliceOp>>> {
    if span_lens.is_empty() {
        ensure!(
            !ids.iter()
                .any(|&i| i == IMAGE_SENTINEL_ID || i == IMAGE_PAD_ID),
            "image tokens in the prompt but no images were encoded"
        );
        return Ok(None);
    }
    // Record the ids by absolute position (a re-run range must agree).
    if start == seen.len() {
        seen.extend_from_slice(ids);
    } else {
        ensure!(
            start + ids.len() <= seen.len() && seen[start..start + ids.len()] == *ids,
            "splice: ids at [{start}, {}) disagree with the prompt seen so far",
            start + ids.len()
        );
    }
    plan_splice(seen, span_lens, start, ids.len()).map(Some)
}

struct Span {
    rows: DevicePtr,
    len: usize,
}

struct Request {
    spans: Vec<Span>,
    /// Prompt ids by absolute position, as far as prefill has seen them.
    seen: Vec<u32>,
}

/// The V4.1 tower and the current request's encoded spans.
pub struct V41ImageSplice {
    /// Frees the current request's span buffers on drop (the encoder frees
    /// its own arena). None = never frees (non-owning callers).
    owner: Option<super::device_allocs::SharedGpu>,
    encoder: DeepSeekVisionEncoder,
    hidden: usize,
    /// START, NEWLINE, END rows (bf16, device).
    start_row: DevicePtr,
    newline_row: DevicePtr,
    end_row: DevicePtr,
    req: Mutex<Request>,
}

impl V41ImageSplice {
    pub fn new(
        encoder: DeepSeekVisionEncoder,
        hidden: usize,
        owner: Option<super::device_allocs::SharedGpu>,
    ) -> Self {
        let sp = encoder.image_special_embeddings();
        Self {
            owner,
            encoder,
            hidden,
            start_row: sp[0],
            newline_row: sp[3],
            end_row: sp[4],
            req: Mutex::new(Request {
                spans: Vec::new(),
                seen: Vec::new(),
            }),
        }
    }

    /// Free the previous request's spans and encode this request's images
    /// (`(bf16-valued patches, vit_h, vit_w)` in prompt order). Empty = text.
    pub fn encode(&self, gpu: &dyn GpuBackend, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        let mut req = self.req.lock().expect("image splice lock");
        if images.is_empty() && req.spans.is_empty() {
            req.seen.clear();
            return Ok(()); // text after text: nothing to free, no GPU call at all
        }
        for s in req.spans.drain(..) {
            gpu.free(s.rows)?;
        }
        req.seen.clear();
        let row_bytes = self.hidden * 2;
        for (patches, gh, gw) in images {
            let (lh, lw) = (gh.div_ceil(3), gw.div_ceil(3));
            let types = span_types(lh, lw);
            let aligned = self.encoder.forward(gpu, patches, *gh, *gw)?; // [lh*lw, hidden], synchronized
            let rows = gpu.alloc(types.len() * row_bytes)?;
            let mut k = 0usize;
            for (j, ty) in types.iter().enumerate() {
                let src = match *ty {
                    START => self.start_row,
                    NEW_LINE => self.newline_row,
                    END => self.end_row,
                    _ => {
                        k += 1;
                        aligned.offset((k - 1) * row_bytes)
                    }
                };
                gpu.copy_d2d(src, rows.offset(j * row_bytes), row_bytes)?;
            }
            ensure!(
                k == lh * lw,
                "span has {k} IMAGE slots for a {lh}x{lw} aligner grid"
            );
            req.spans.push(Span {
                rows,
                len: types.len(),
            });
        }
        gpu.synchronize(gpu.default_stream())?;
        Ok(())
    }

    pub fn has_images(&self) -> bool {
        !self.req.lock().expect("image splice lock").spans.is_empty()
    }

    /// Overwrite this chunk's embedding rows `x` ([t, hidden] bf16) for
    /// positions `[start, start+t)` whose ids are `ids`. `embed` is the token
    /// embedding table (for the SENTINEL row). A no-op for text requests.
    pub fn splice(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        embed: DevicePtr,
        ids: &[u32],
        start: usize,
        x: DevicePtr,
    ) -> Result<()> {
        let mut guard = self.req.lock().expect("image splice lock");
        let req = &mut *guard;
        let lens: Vec<usize> = req.spans.iter().map(|s| s.len).collect();
        let Some(ops) = chunk_plan(&lens, &mut req.seen, ids, start)? else {
            return Ok(()); // text request: not one GPU operation
        };
        let row = self.hidden * 2;
        let sentinel = embed.offset(IMAGE_SENTINEL_ID as usize * row);
        for op in ops {
            match op {
                SpliceOp::Pad { row: r } => {
                    gpu.copy_d2d_async(sentinel, x.offset(r * row), row, stream)?
                }
                SpliceOp::Span {
                    row: r,
                    image,
                    slot,
                    count,
                } => gpu.copy_d2d_async(
                    req.spans[image].rows.offset(slot * row),
                    x.offset(r * row),
                    count * row,
                    stream,
                )?,
            }
        }
        Ok(())
    }
}

/// Free span buffers through `owner`; with no owner, leave them (non-owning).
fn free_spans(owner: Option<&super::device_allocs::SharedGpu>, spans: &mut Vec<Span>) {
    let Some(gpu) = owner else { return };
    if let Err(e) = gpu.bind_to_thread() {
        tracing::warn!("V41ImageSplice: bind_to_thread failed before freeing spans: {e}");
    }
    for s in spans.drain(..) {
        if let Err(e) = gpu.free(s.rows) {
            tracing::warn!("V41ImageSplice: freeing span rows failed: {e}");
        }
    }
}

/// Dropping the splice frees the last request's span buffers; the encoder
/// (dropped with it) frees its own arena through its `DeviceAllocs`.
impl Drop for V41ImageSplice {
    fn drop(&mut self) {
        if let Ok(mut req) = self.req.lock() {
            free_spans(self.owner.as_ref(), &mut req.spans);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u32 = IMAGE_SENTINEL_ID;
    const P: u32 = IMAGE_PAD_ID;

    fn layout() -> (Vec<u32>, Vec<usize>) {
        // engine/vision.py expand_image_placeholders([5, SENT, 7, SENT], types 2x3 then 1x1):
        // span lengths 10 and 4; the second needs one pad to start at an odd position.
        let mut ids = vec![5];
        ids.extend(std::iter::repeat_n(S, 10));
        ids.extend([7, P]);
        ids.extend(std::iter::repeat_n(S, 4));
        (ids, vec![span_types(2, 3).len(), span_types(1, 1).len()])
    }

    #[test]
    fn whole_prompt_plan() {
        let (ids, lens) = layout();
        assert_eq!(lens, vec![10, 4]);
        let ops = plan_splice(&ids, &lens, 0, ids.len()).unwrap();
        assert_eq!(
            ops,
            vec![
                SpliceOp::Span {
                    row: 1,
                    image: 0,
                    slot: 0,
                    count: 10
                },
                SpliceOp::Pad { row: 12 },
                SpliceOp::Span {
                    row: 13,
                    image: 1,
                    slot: 0,
                    count: 4
                },
            ]
        );
    }

    #[test]
    fn chunked_plans_cover_exactly_what_the_whole_plan_does() {
        let (ids, lens) = layout();
        // expand every op of the whole-prompt plan into (abs position -> source)
        let flat = |ops: &[SpliceOp], base: usize| -> Vec<(usize, Option<(usize, usize)>)> {
            ops.iter()
                .flat_map(|op| match *op {
                    SpliceOp::Pad { row } => vec![(base + row, None)],
                    SpliceOp::Span {
                        row,
                        image,
                        slot,
                        count,
                    } => (0..count)
                        .map(|k| (base + row + k, Some((image, slot + k))))
                        .collect(),
                })
                .collect()
        };
        let whole = flat(&plan_splice(&ids, &lens, 0, ids.len()).unwrap(), 0);
        for chunk in 1..=ids.len() {
            let mut got = Vec::new();
            let mut start = 0;
            while start < ids.len() {
                let t = chunk.min(ids.len() - start);
                got.extend(flat(&plan_splice(&ids, &lens, start, t).unwrap(), start));
                start += t;
            }
            assert_eq!(got, whole, "chunk size {chunk}");
        }
    }

    #[test]
    fn adjacent_images_split_by_length() {
        // two 1x1 images back to back (span 4 each), no pad between them
        let ids: Vec<u32> = std::iter::once(9)
            .chain(std::iter::repeat_n(S, 8))
            .collect();
        let ops = plan_splice(&ids, &[4, 4], 0, ids.len()).unwrap();
        assert_eq!(
            ops,
            vec![
                SpliceOp::Span {
                    row: 1,
                    image: 0,
                    slot: 0,
                    count: 4
                },
                SpliceOp::Span {
                    row: 5,
                    image: 1,
                    slot: 0,
                    count: 4
                },
            ]
        );
    }

    #[test]
    fn text_only_chunks_plan_nothing() {
        let mut seen = Vec::new();
        for (start, chunk) in [
            (0usize, vec![0u32, 128803, 42, 128804, 128822]),
            (5, vec![7, 8]),
        ] {
            assert_eq!(chunk_plan(&[], &mut seen, &chunk, start).unwrap(), None);
        }
        assert!(seen.is_empty(), "the text path must not even record ids");
        // control: the same call WITH an encoded image does plan work
        let (ids, lens) = layout();
        assert!(
            chunk_plan(&lens, &mut Vec::new(), &ids, 0)
                .unwrap()
                .is_some_and(|ops| !ops.is_empty())
        );
        // and image tokens with no encoded image are refused
        assert!(chunk_plan(&[], &mut Vec::new(), &[5, S], 0).is_err());
    }

    /// The drop path frees every span buffer through an owned backend.
    /// NEGATIVE CONTROL: with no owner nothing is freed (the old leak).
    #[test]
    fn span_buffers_are_freed_with_an_owner_and_leak_without() {
        use spark_runtime::gpu::mock::MockGpuBackend;
        let mock = std::sync::Arc::new(MockGpuBackend::new());
        let shared: super::super::device_allocs::SharedGpu = mock.clone();
        let mk = |n: usize| -> Vec<Span> {
            (0..n)
                .map(|_| Span {
                    rows: shared.alloc(4096).unwrap(),
                    len: 4,
                })
                .collect()
        };
        let mut spans = mk(3);
        assert_eq!(mock.alloc_count(), 3);
        free_spans(Some(&shared), &mut spans);
        assert_eq!(mock.alloc_count(), 0, "owned drop must free all span rows");
        let mut leaky = mk(2);
        free_spans(None, &mut leaky);
        assert_eq!(
            mock.alloc_count(),
            2,
            "without an owner the spans are left (and leak)"
        );
    }

    /// NEGATIVE CONTROLS: a sentinel count that does not match the images must
    /// be refused, never silently spliced.
    #[test]
    fn mismatched_spans_are_refused() {
        let (ids, _) = layout();
        assert!(
            plan_splice(&ids, &[10], 0, ids.len()).is_err(),
            "extra span"
        );
        assert!(
            plan_splice(&ids, &[11, 4], 0, ids.len()).is_err(),
            "span cut short"
        );
    }
}
