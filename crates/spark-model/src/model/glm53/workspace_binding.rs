// SPDX-License-Identifier: AGPL-3.0-only

//! Resolves the GLM target workspace plan into device buffers.
//!
//! `Glm53TargetSchedule` describes the workspace as byte offsets and extents
//! (`hidden_a`, `hidden_b`, `collapsed`, `widened_hc`, `hyper_post`,
//! `hyper_comb`) inside one arena allocation. Every op in the walk takes
//! `GgmlIqBuffer { ptr, bytes }`. This is the layer between: arena base +
//! region -> buffer.
//!
//! It is the foundation for all 234 events, so it is deliberately the piece
//! that is hardest to get silently wrong:
//!
//! * every region is bounds-checked against the arena extent, so a plan/arena
//!   mismatch fails here rather than as a stray write during a walk;
//! * regions are checked pairwise for overlap, because two ops sharing a buffer
//!   they should not is the corruption class that produces fluent, wrong output
//!   rather than a crash (F42/F56 on Flash-Next);
//! * alignment is asserted, since the ops index these as BF16/F32 arrays.
//!
//! All of this is CPU-side pointer arithmetic and is therefore testable without
//! a GPU, which is how the rest of this module is qualified.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layers::ops::GgmlIqBuffer;
use crate::layers::{Glm53TargetWorkspace, Glm53TargetWorkspaceRegion};

/// Ops index these buffers as BF16/F32 arrays; the schedule aligns regions to
/// 256 bytes and the binder refuses anything weaker.
const REQUIRED_ALIGNMENT: u64 = 256;

/// Device buffers for one walk, resolved from the arena base.
// No PartialEq: the underlying GgmlIqBuffer does not derive it, and adding an
// impl here would invite comparing buffers by value when identity is what
// matters. Tests compare (ptr, bytes) explicitly.
#[derive(Debug, Clone, Copy)]
pub struct Glm53BoundWorkspace {
    pub hidden_a: GgmlIqBuffer,
    pub hidden_b: GgmlIqBuffer,
    pub collapsed: GgmlIqBuffer,
    pub widened_hc: GgmlIqBuffer,
    pub hyper_post: GgmlIqBuffer,
    pub hyper_comb: GgmlIqBuffer,
}

fn bind_region(
    base: DevicePtr,
    arena_bytes: u64,
    name: &'static str,
    region: Glm53TargetWorkspaceRegion,
) -> Result<GgmlIqBuffer> {
    ensure!(
        region.payload_bytes > 0,
        "GLM workspace region {name} is empty"
    );
    ensure!(
        region.payload_bytes <= region.allocation_bytes,
        "GLM workspace region {name} payload {} exceeds its allocation {}",
        region.payload_bytes,
        region.allocation_bytes
    );
    let end = region
        .offset_bytes
        .checked_add(region.allocation_bytes)
        .with_context(|| format!("GLM workspace region {name} extent overflow"))?;
    ensure!(
        end <= arena_bytes,
        "GLM workspace region {name} ends at {end} beyond the {arena_bytes}-byte arena"
    );
    let ptr = base
        .0
        .checked_add(region.offset_bytes)
        .with_context(|| format!("GLM workspace region {name} address overflow"))?;
    ensure!(
        ptr % REQUIRED_ALIGNMENT == 0,
        "GLM workspace region {name} is {REQUIRED_ALIGNMENT}-byte misaligned"
    );
    Ok(GgmlIqBuffer {
        ptr: DevicePtr(ptr),
        // Ops are handed the payload, never the padded allocation: a kernel
        // sized from the padding would read past the data it was given.
        bytes: usize::try_from(region.payload_bytes)?,
    })
}

impl Glm53BoundWorkspace {
    /// Bind every region against one arena allocation.
    ///
    /// `base` must be the arena start and `arena_bytes` its true extent —
    /// passing a larger extent than allocated would move the bounds check off
    /// the real memory and defeat the purpose.
    pub fn bind(
        workspace: &Glm53TargetWorkspace,
        base: DevicePtr,
        arena_bytes: u64,
    ) -> Result<Self> {
        workspace.validate()?;
        ensure!(!base.is_null(), "GLM workspace arena base is NULL");
        ensure!(
            base.0 % REQUIRED_ALIGNMENT == 0,
            "GLM workspace arena base is {REQUIRED_ALIGNMENT}-byte misaligned"
        );
        ensure!(
            arena_bytes >= workspace.arena_bytes,
            "GLM workspace needs {} bytes but the arena is {arena_bytes}",
            workspace.arena_bytes
        );

        let named: [(&'static str, Glm53TargetWorkspaceRegion); 6] = [
            ("hidden_a", workspace.hidden_a),
            ("hidden_b", workspace.hidden_b),
            ("collapsed", workspace.collapsed),
            ("widened_hc", workspace.widened_hc),
            ("hyper_post", workspace.hyper_post),
            ("hyper_comb", workspace.hyper_comb),
        ];

        // Overlap is checked on ALLOCATIONS, not payloads: two regions whose
        // padding overlaps still alias once a kernel writes a full tile.
        for left in 0..named.len() {
            for right in left + 1..named.len() {
                let (ln, l) = named[left];
                let (rn, r) = named[right];
                let l_end = l.offset_bytes + l.allocation_bytes;
                let r_end = r.offset_bytes + r.allocation_bytes;
                if l.offset_bytes < r_end && r.offset_bytes < l_end {
                    bail!("GLM workspace regions {ln} and {rn} overlap");
                }
            }
        }

        Ok(Self {
            hidden_a: bind_region(base, arena_bytes, "hidden_a", workspace.hidden_a)?,
            hidden_b: bind_region(base, arena_bytes, "hidden_b", workspace.hidden_b)?,
            collapsed: bind_region(base, arena_bytes, "collapsed", workspace.collapsed)?,
            widened_hc: bind_region(base, arena_bytes, "widened_hc", workspace.widened_hc)?,
            hyper_post: bind_region(base, arena_bytes, "hyper_post", workspace.hyper_post)?,
            hyper_comb: bind_region(base, arena_bytes, "hyper_comb", workspace.hyper_comb)?,
        })
    }
}

#[cfg(test)]
#[path = "workspace_binding_tests.rs"]
mod tests;
