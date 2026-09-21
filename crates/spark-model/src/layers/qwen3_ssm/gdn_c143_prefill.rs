// SPDX-License-Identifier: AGPL-3.0-only

//! Strict default-off production adapter for the sealed GDN c143 SM121 v3 ABI.

use std::ffi::OsStr;

use super::*;

const QWEN38_GDN_KEY_HEADS: usize = 16;
const QWEN38_GDN_VALUE_HEADS: usize = 48;
const QWEN38_GDN_HEAD_DIM: usize = 128;
const QWEN38_GDN_CONV_WIDTH: usize = 10_240;
const QWEN38_GDN_GATE_BETA_WIDTH: usize = 96;
pub(super) const QWEN38_GDN_QKVZ_WIDTH: usize = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GdnC143PrefillRoute {
    Disabled,
    Ineligible,
    Required,
}

pub(super) const fn gdn_c143_prefill_route(
    requested: bool,
    exact_geometry: bool,
) -> GdnC143PrefillRoute {
    if !requested {
        GdnC143PrefillRoute::Disabled
    } else if !exact_geometry {
        GdnC143PrefillRoute::Ineligible
    } else {
        GdnC143PrefillRoute::Required
    }
}

pub(super) const fn gdn_c143_exact_geometry(
    rows: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_width: usize,
    gate_beta_width: usize,
    qkvz_width: usize,
) -> bool {
    matches!(rows, 2_079 | 8_192)
        && key_heads == QWEN38_GDN_KEY_HEADS
        && value_heads == QWEN38_GDN_VALUE_HEADS
        && key_head_dim == QWEN38_GDN_HEAD_DIM
        && value_head_dim == QWEN38_GDN_HEAD_DIM
        && conv_width == QWEN38_GDN_CONV_WIDTH
        && gate_beta_width == QWEN38_GDN_GATE_BETA_WIDTH
        && qkvz_width == QWEN38_GDN_QKVZ_WIDTH
}

pub(super) fn parse_gdn_c143_selector(value: Option<&OsStr>) -> Result<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    match value.to_str() {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => {
            anyhow::bail!("ATLAS_GDN_C143_PREFILL must be absent, 0, or 1; got {value:?}")
        }
        None => anyhow::bail!("ATLAS_GDN_C143_PREFILL must be valid UTF-8"),
    }
}

fn gdn_c143_prefill_enabled() -> Result<bool> {
    static ENABLED: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    match ENABLED.get_or_init(|| {
        parse_gdn_c143_selector(std::env::var_os("ATLAS_GDN_C143_PREFILL").as_deref())
            .map_err(|error| error.to_string())
    }) {
        Ok(enabled) => Ok(*enabled),
        Err(error) => anyhow::bail!("{error}"),
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GdnC143ExecutionIdentity {
    stream: u64,
    workspace: DevicePtr,
    workspace_bytes: usize,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct GdnC143Runtime {
    library: ops::gdn_c143_sm121::GdnC143Sm121,
    execution: std::sync::OnceLock<GdnC143ExecutionIdentity>,
    launch_lock: std::sync::Mutex<()>,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl GdnC143Runtime {
    fn bind_execution(&self, proposed: GdnC143ExecutionIdentity) -> Result<()> {
        ensure!(
            proposed.stream != 0,
            "GDN c143 requires a nonzero CUDA stream"
        );
        if let Some(existing) = self.execution.get() {
            ensure!(
                *existing == proposed,
                "GDN c143 process-shared workspace/stream identity changed"
            );
            return Ok(());
        }
        if self.execution.set(proposed).is_err() {
            ensure!(
                self.execution.get() == Some(&proposed),
                "GDN c143 process-shared workspace/stream identity raced"
            );
        }
        Ok(())
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn gdn_c143_runtime() -> Result<&'static GdnC143Runtime> {
    static RUNTIME: std::sync::OnceLock<Result<GdnC143Runtime, String>> =
        std::sync::OnceLock::new();
    match RUNTIME.get_or_init(|| {
        let path = std::env::var_os("ATLAS_GDN_C143_SM121_LIB").ok_or_else(|| {
            "ATLAS_GDN_C143_PREFILL=1 requires ATLAS_GDN_C143_SM121_LIB".to_owned()
        })?;
        let library = ops::gdn_c143_sm121::GdnC143Sm121::open_exact(std::path::Path::new(&path))
            .map_err(|error| error.to_string())?;
        Ok(GdnC143Runtime {
            library,
            execution: std::sync::OnceLock::new(),
            launch_lock: std::sync::Mutex::new(()),
        })
    }) {
        Ok(runtime) => Ok(runtime),
        Err(error) => anyhow::bail!("GDN c143 runtime admission failed: {error}"),
    }
}

pub(super) struct GdnC143PrefillPlan {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    runtime: &'static GdnC143Runtime,
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    shape: ops::gdn_c143_sm121::GdnC143Shape,
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    buffers: ops::gdn_c143_sm121::GdnC143Buffers,
    stream: u64,
}

impl GdnC143PrefillPlan {
    pub(super) fn launch(&self, gpu: &dyn GpuBackend) -> Result<()> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            // The native call enqueues several dependent kernels that share
            // this process-wide arena. Serialize the host enqueue sequence;
            // the kernels remain asynchronous and ordered on the same stream.
            let _launch_guard = self
                .runtime
                .launch_lock
                .lock()
                .map_err(|_| anyhow::anyhow!("GDN c143 process-shared launch lock poisoned"))?;
            let mut prepared = self.runtime.library.prepare_borrowed(
                gpu,
                self.shape,
                self.stream,
                self.buffers.workspace_bytes,
            )?;
            prepared.launch_eager(self.buffers, self.stream)?;
            let (source_device, source_inode) = self.runtime.library.source_file_identity();
            let receipt = match self.shape.seq_len() {
                2_079 => &GDN_C143_M2079_ENGAGED,
                8_192 => &GDN_C143_M8192_ENGAGED,
                _ => unreachable!("preflight admitted a non-production GDN c143 shape"),
            };
            receipt.call_once(|| {
                tracing::info!(
                    rows = self.shape.seq_len(),
                    workspace_bytes = self.buffers.workspace_bytes,
                    required_workspace_bytes = self.shape.required_workspace_bytes(),
                    library = %self.runtime.library.path().display(),
                    library_sha256 = %self.runtime.library.sha256_hex(),
                    source_device,
                    source_inode,
                    "ENGAGED ATLAS_GDN_C143_PREFILL: sealed c143 v3 launch succeeded"
                );
            });
            Ok(())
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (gpu, self.stream);
            anyhow::bail!("GDN c143 production route requires CUDA on Linux")
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
static GDN_C143_M2079_ENGAGED: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", target_os = "linux"))]
static GDN_C143_M8192_ENGAGED: std::sync::Once = std::sync::Once::new();

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn preflight_gdn_c143_prefill(
        &self,
        ctx: &ForwardContext,
        rows: usize,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_width: usize,
        gate_beta_width: usize,
        qkvz_width: usize,
        state_fp32: DevicePtr,
        qkv_bf16: DevicePtr,
        gate_beta_fp32: DevicePtr,
        output_bf16: DevicePtr,
        stream: u64,
    ) -> Result<Option<GdnC143PrefillPlan>> {
        let requested = gdn_c143_prefill_enabled()?;
        let exact_geometry = gdn_c143_exact_geometry(
            rows,
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            conv_width,
            gate_beta_width,
            qkvz_width,
        );
        match gdn_c143_prefill_route(requested, exact_geometry) {
            GdnC143PrefillRoute::Disabled | GdnC143PrefillRoute::Ineligible => return Ok(None),
            GdnC143PrefillRoute::Required => {}
        }
        ensure!(
            !crate::layers::gdn_prefill_gatecache_enabled(),
            "ATLAS_GDN_C143_PREFILL=1 is mutually exclusive with ATLAS_GDN_PREFILL_GATECACHE=1"
        );
        ensure!(
            !ctx.graph_capture,
            "ATLAS_GDN_C143_PREFILL=1 is eager-only and cannot run during graph capture"
        );

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let runtime = gdn_c143_runtime()?;
            let shape = ops::gdn_c143_sm121::GdnC143Shape::qwen38(rows)?;
            let workspace = ctx.buffers.ssm_conv_out_f32();
            let workspace_bytes = ctx.buffers.sizes().ssm_conv_out_f32;
            let buffers = ops::gdn_c143_sm121::GdnC143Buffers {
                state_fp32,
                qkv_bf16,
                gate_beta_fp32,
                output_bf16,
                workspace,
                workspace_bytes,
            };
            let prepared =
                runtime
                    .library
                    .prepare_borrowed(ctx.gpu, shape, stream, workspace_bytes)?;
            prepared.validate_buffers(buffers, stream)?;
            runtime.bind_execution(GdnC143ExecutionIdentity {
                stream,
                workspace,
                workspace_bytes,
            })?;
            Ok(Some(GdnC143PrefillPlan {
                runtime,
                shape,
                buffers,
                stream,
            }))
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (state_fp32, qkv_bf16, gate_beta_fp32, output_bf16, stream);
            anyhow::bail!("ATLAS_GDN_C143_PREFILL=1 requires CUDA on Linux")
        }
    }
}
