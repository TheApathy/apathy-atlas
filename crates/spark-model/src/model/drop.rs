// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

fn destroy_graph_cache<K>(
    gpu: &dyn GpuBackend,
    cache_name: &str,
    cache: &mut HashMap<K, GraphHandle>,
) {
    for graph in cache.drain().map(|(_, graph)| graph) {
        if graph.0 != 0
            && let Err(error) = gpu.destroy_graph(graph)
        {
            tracing::error!(
                "TransformerModel::drop: destroy_graph({cache_name}, {}) failed: {error:#}",
                graph.0
            );
            // Automatic field teardown would free pinned graph sources while
            // the failed graph may still retain them. Fail closed first.
            std::process::abort();
        }
    }
}

impl Drop for TransformerModel {
    fn drop(&mut self) {
        // Every pinned-host source must remain allocated and immutable until
        // its eager async copy completes, and captured H2D source addresses
        // must remain allocated until their graph is destroyed. Drop runs
        // before automatic field teardown, so both the GPU backend and all
        // layer-owned Qwen pointer tables are still alive here.
        let default_stream = self.gpu.default_stream();
        if let Err(error) = self.gpu.synchronize(default_stream) {
            tracing::error!(
                "TransformerModel::drop: default-stream synchronization failed: {error:#}"
            );
            std::process::abort();
        }
        if self.secondary_stream != default_stream
            && let Err(error) = self.gpu.synchronize(self.secondary_stream)
        {
            tracing::error!(
                "TransformerModel::drop: secondary-stream synchronization failed: {error:#}"
            );
            std::process::abort();
        }

        // Piecewise graphs contain Qwen H2D nodes whose source pointer is
        // owned by `layers`; destroy them first, before any layer field drops.
        destroy_graph_cache(
            self.gpu.as_ref(),
            "piecewise_decode_graphs",
            self.piecewise_decode_graphs.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "decode_graph",
            self.decode_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "batch_decode_graphs",
            self.batch_decode_graphs.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify2_graph",
            self.verify2_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify3_graph",
            self.verify3_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify4_graph",
            self.verify4_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify_kgamma_graph",
            self.verify_kgamma_graph.get_mut(),
        );

        if self.secondary_event != 0
            && let Err(error) = self.gpu.destroy_event(self.secondary_event)
        {
            tracing::error!(
                "TransformerModel::drop: destroy_event({}) failed: {error:#}",
                self.secondary_event
            );
        }
    }
}
