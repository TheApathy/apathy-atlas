// SPDX-License-Identifier: AGPL-3.0-only

//! I/O adapter for exact, position-free verifier FFN graph groups.
use super::dispatch::Glm53Dispatcher;
use super::ffn_graph::{Binding, GraphCache, GraphIo, Key, Setting};
use crate::layers::ops::{
    glm53_exact_verify_active, glm53_exact_wide_prefill_active, glm53_layer_major_prefill_active,
};
use crate::layers::{Glm53TargetEvent, Glm53TargetFfnKind};
use anyhow::{Result, ensure};
use spark_runtime::gpu::{GpuBackend, GraphHandle};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::Mutex;

fn environment() -> BTreeMap<OsString, OsString> {
    std::env::vars_os()
        .filter(|(k, _)| k.as_encoded_bytes().starts_with(b"ATLAS_GLM53_"))
        .collect()
}

pub(super) struct FfnGraphs {
    setting: Setting,
    scalar: bool,
    environment: BTreeMap<OsString, OsString>,
    cache: Mutex<GraphCache>,
}

/// `ATLAS_GLM53_FFN_GRAPHS_SCALAR`: extend the position-free MoE graph groups
/// to the one-row decode walk. Strict: absent or `0` off, exactly `1` on, and
/// it requires `ATLAS_GLM53_FFN_GRAPHS=1`.
pub(super) fn parse_scalar_setting(value: Option<&std::ffi::OsStr>, base: bool) -> Result<bool> {
    let on = match value {
        None => false,
        Some(v) if v == "0" => false,
        Some(v) if v == "1" => true,
        _ => anyhow::bail!("ATLAS_GLM53_FFN_GRAPHS_SCALAR must be absent or exactly 0 or 1"),
    };
    ensure!(
        !on || base,
        "ATLAS_GLM53_FFN_GRAPHS_SCALAR=1 requires ATLAS_GLM53_FFN_GRAPHS=1"
    );
    Ok(on)
}

impl FfnGraphs {
    pub(super) fn from_env() -> Result<Self> {
        let setting = Setting::parse(std::env::var_os("ATLAS_GLM53_FFN_GRAPHS").as_deref())
            .map_err(anyhow::Error::msg)?;
        let scalar = parse_scalar_setting(
            std::env::var_os("ATLAS_GLM53_FFN_GRAPHS_SCALAR").as_deref(),
            setting.enabled(),
        )?;
        let environment = if setting.enabled() {
            environment()
        } else {
            BTreeMap::new()
        };
        if setting.enabled() {
            for key in [
                "ATLAS_GLM53_EXACT_VERIFY",
                "ATLAS_GLM53_EXACT_WIDE_ROWEXACT",
                "ATLAS_GLM53_EXL3_ROUTE_PRIVATE",
            ] {
                ensure!(
                    environment
                        .get(std::ffi::OsStr::new(key))
                        .is_some_and(|v| v == "1"),
                    "FFN graphs require {key}=1"
                );
            }
            ensure!(
                environment
                    .get(std::ffi::OsStr::new("ATLAS_GLM53_EXL3_MOE"))
                    .is_none_or(|v| v == "fused"),
                "FFN graphs require fused MoE"
            );
            for key in ["ATLAS_GLM53_WIDE_TIMING", "ATLAS_GLM53_DUMP_DIR"] {
                ensure!(
                    !environment.contains_key(std::ffi::OsStr::new(key)),
                    "FFN graphs exclude {key}"
                );
            }
            tracing::info!(
                "GLM FFN graphs armed: exact position-free MoE groups, lazy warm/capture/replay, max_entries=336 scalar_walk={scalar}"
            );
        }
        Ok(Self {
            setting,
            scalar,
            environment,
            cache: Mutex::new(GraphCache::new()),
        })
    }

    /// Whether the one-row decode walk may run its MoE groups as graphs.
    pub(super) fn eligible_scalar(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<bool> {
        if !self.setting.enabled() || !self.scalar {
            return Ok(false);
        }
        ensure!(
            self.environment == environment(),
            "GLM FFN graph configuration changed after construction"
        );
        if glm53_layer_major_prefill_active() || glm53_exact_wide_prefill_active() {
            return Ok(false);
        }
        // The target-only decode walk runs on the legacy default stream, which
        // cannot host a graph. That is a reason to decline the graph and launch
        // eagerly, not a reason to fail the step: the scalar walk is a valid
        // configuration that simply gets no graph. (The wide-verify `eligible`
        // below keeps its hard check, because its caller has already committed
        // to a private stream and a null one there would be a real defect.)
        if stream == 0 || gpu.stream_is_capturing(stream) {
            return Ok(false);
        }
        Ok(true)
    }

    pub(super) fn execute_scalar(
        &self,
        dispatcher: &Glm53Dispatcher<'_>,
        gpu: &dyn GpuBackend,
        layer: u32,
        binding: Binding,
    ) -> Result<()> {
        let key = Key::new_scalar(layer).map_err(anyhow::Error::msg)?;
        let mut io = Adapter {
            gpu,
            body: Some((dispatcher, layer)),
            stream: binding.stream,
        };
        self.cache
            .lock()
            .map_err(|_| anyhow::anyhow!("FFN graph owner mutex poisoned"))?
            .execute(key, binding, &mut io)
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn eligible(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        stream: u64,
        has_capture: bool,
    ) -> Result<bool> {
        if !self.setting.enabled() {
            return Ok(false);
        }
        ensure!(
            self.environment == environment(),
            "GLM FFN graph configuration changed after construction"
        );
        if has_capture || glm53_layer_major_prefill_active() || glm53_exact_wide_prefill_active() {
            return Ok(false);
        }
        ensure!(
            (2..=8).contains(&rows) && glm53_exact_verify_active(),
            "FFN graphs require exact verification scope"
        );
        ensure!(
            stream != 0 && !gpu.stream_is_capturing(stream),
            "FFN graphs require eager nonzero stream"
        );
        Ok(true)
    }

    pub(super) fn execute(
        &self,
        dispatcher: &Glm53Dispatcher<'_>,
        gpu: &dyn GpuBackend,
        rows: u32,
        layer: u32,
        binding: Binding,
    ) -> Result<()> {
        let key = Key::new(rows, layer).map_err(anyhow::Error::msg)?;
        let mut io = Adapter {
            gpu,
            body: Some((dispatcher, layer)),
            stream: binding.stream,
        };
        self.cache
            .lock()
            .map_err(|_| anyhow::anyhow!("FFN graph owner mutex poisoned"))?
            .execute(key, binding, &mut io)
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn drain(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if self.setting.enabled() {
            eprintln!(
                "GLM_FFN_GRAPH_CLOSE counts={:?} rows_2_through_8={:?}",
                cache.counts(),
                cache.row_counts()
            );
        }
        let mut io = Adapter {
            gpu,
            body: None,
            stream: 0,
        };
        cache.drain(&mut io).map_err(anyhow::Error::msg)
    }
}

struct Adapter<'a, 'w> {
    gpu: &'a dyn GpuBackend,
    body: Option<(&'a Glm53Dispatcher<'w>, u32)>,
    stream: u64,
}
impl GraphIo for Adapter<'_, '_> {
    fn dispatch(&mut self) -> std::result::Result<(), String> {
        let (dispatcher, layer) = self.body.ok_or("FFN graph cleanup cannot dispatch")?;
        for event in [
            Glm53TargetEvent::PostAttention { layer },
            Glm53TargetEvent::Ffn {
                layer,
                kind: Glm53TargetFfnKind::Moe,
            },
            Glm53TargetEvent::PostFfn { layer },
        ] {
            dispatcher
                .dispatch(self.gpu, &event, self.stream)
                .map_err(|e| format!("{e:#}"))?;
        }
        Ok(())
    }
    fn begin(&mut self, stream: u64) -> std::result::Result<(), String> {
        self.gpu.begin_capture(stream).map_err(|e| format!("{e:#}"))
    }
    fn end(&mut self, stream: u64) -> std::result::Result<u64, String> {
        self.gpu
            .end_capture(stream)
            .map(|g| g.0)
            .map_err(|e| format!("{e:#}"))
    }
    fn launch(&mut self, graph: u64, stream: u64) -> std::result::Result<(), String> {
        self.gpu
            .launch_graph(GraphHandle(graph), stream)
            .map_err(|e| format!("{e:#}"))
    }
    fn synchronize(&mut self, stream: u64) -> std::result::Result<(), String> {
        self.gpu.synchronize(stream).map_err(|e| format!("{e:#}"))
    }
    fn destroy(&mut self, graph: u64) -> std::result::Result<(), String> {
        self.gpu
            .destroy_graph(GraphHandle(graph))
            .map_err(|e| format!("{e:#}"))
    }
}

#[cfg(test)]
mod scalar_setting_tests {
    use super::parse_scalar_setting;
    use std::ffi::OsStr;

    #[test]
    fn scalar_graphs_are_strict_default_off_and_require_the_base_flag() {
        assert!(!parse_scalar_setting(None, false).unwrap());
        assert!(!parse_scalar_setting(None, true).unwrap());
        assert!(!parse_scalar_setting(Some(OsStr::new("0")), true).unwrap());
        assert!(parse_scalar_setting(Some(OsStr::new("1")), true).unwrap());
        assert!(parse_scalar_setting(Some(OsStr::new("1")), false).is_err());
        for bad in ["", "true", "2", " 1"] {
            assert!(parse_scalar_setting(Some(OsStr::new(bad)), true).is_err());
        }
    }
}
