// SPDX-License-Identifier: AGPL-3.0-only

//! Sparse host dispatch recorder. Synthetic addresses, no CUDA or tensor emulation.
use anyhow::{Result, bail};
use spark_model::layers::ops::{
    GLM53_EXL3_MOE_LOCK_BYTES, Glm53Exl3Buffer, Glm53Exl3MoeBuffers, Glm53Exl3MoePlan,
    Glm53Exl3MoePointerTables,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use std::{collections::BTreeMap, ffi::c_void, sync::Mutex};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    Ptr(u64),
    Scalar(Vec<u8>),
}
impl Arg {
    pub fn ptr(value: Glm53Exl3Buffer) -> Self {
        Self::Ptr(value.ptr.0)
    }
    pub fn word(value: u32) -> Self {
        Self::Scalar(value.to_le_bytes().to_vec())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub module: String,
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared: u32,
    pub stream: u64,
    pub args: Vec<Arg>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    Launch(Launch),
    Memset(u64, u8, usize, u64),
    Other(&'static str),
}
pub struct TraceGpu {
    symbols: Mutex<BTreeMap<u64, (String, String)>>,
    effects: Mutex<Vec<Effect>>,
    address: Mutex<u64>,
    fail: Mutex<Option<String>>,
}
impl TraceGpu {
    pub fn new() -> Self {
        Self {
            symbols: Mutex::new(BTreeMap::new()),
            effects: Mutex::new(Vec::new()),
            address: Mutex::new(0x1000_0000),
            fail: Mutex::new(None),
        }
    }
    pub fn effects(&self) -> Vec<Effect> {
        self.effects.lock().unwrap().clone()
    }
    pub fn launches(&self) -> Vec<Launch> {
        self.effects()
            .into_iter()
            .filter_map(|e| {
                if let Effect::Launch(l) = e {
                    Some(l)
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn memsets(&self) -> Vec<(u64, u8, usize, u64)> {
        self.effects()
            .into_iter()
            .filter_map(|e| {
                if let Effect::Memset(p, v, b, s) = e {
                    Some((p, v, b, s))
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn fail_on(&self, name: &str) {
        *self.fail.lock().unwrap() = Some(name.to_owned());
    }
    pub fn buffer(&self, bytes: usize) -> Glm53Exl3Buffer {
        let mut address = self.address.lock().unwrap();
        let ptr = DevicePtr(*address);
        *address = address
            .checked_add(u64::try_from(bytes).unwrap())
            .unwrap()
            .checked_add(4095)
            .unwrap()
            & !255;
        Glm53Exl3Buffer { ptr, bytes }
    }
    fn reject(&self, operation: &'static str) -> Result<()> {
        self.effects.lock().unwrap().push(Effect::Other(operation));
        bail!("unexpected dispatch-side I/O: {operation}")
    }
}
impl GpuBackend for TraceGpu {
    fn alloc(&self, _: usize) -> Result<DevicePtr> {
        self.reject("alloc")?;
        unreachable!()
    }
    fn alloc_managed(&self, _: usize) -> Result<DevicePtr> {
        self.reject("alloc_managed")?;
        unreachable!()
    }
    fn free(&self, _: DevicePtr) -> Result<()> {
        self.reject("free")
    }
    fn copy_h2d(&self, _: &[u8], _: DevicePtr) -> Result<()> {
        self.reject("copy_h2d")
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        self.reject("copy_d2h")
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        self.reject("copy_d2d")
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        self.reject("memset_sync")
    }
    fn memset_async(&self, p: DevicePtr, v: u8, b: usize, s: u64) -> Result<()> {
        self.effects
            .lock()
            .unwrap()
            .push(Effect::Memset(p.0, v, b, s));
        Ok(())
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut c_void],
    ) -> Result<()> {
        self.reject("untyped_launch")
    }
    fn launch_typed(
        &self,
        handle: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let (module, name) = self
            .symbols
            .lock()
            .unwrap()
            .get(&handle.0)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown/null test kernel handle"))?;
        let args = args
            .iter()
            .map(|arg| match arg {
                KernelArg::Buffer(p) => Arg::Ptr(p.0),
                KernelArg::Bytes(bytes) => Arg::Scalar(bytes.to_vec()),
            })
            .collect();
        self.effects.lock().unwrap().push(Effect::Launch(Launch {
            module,
            name: name.clone(),
            grid,
            block,
            shared,
            stream,
            args,
        }));
        if self.fail.lock().unwrap().as_deref() == Some(name.as_str()) {
            bail!("injected launch failure: {name}");
        }
        Ok(())
    }
    fn synchronize(&self, _: u64) -> Result<()> {
        self.reject("synchronize")
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        let mut symbols = self.symbols.lock().unwrap();
        if let Some((&handle, _)) = symbols.iter().find(|(_, v)| v.0 == module && v.1 == name) {
            return Ok(KernelHandle(handle));
        }
        let handle = symbols.len() as u64 + 1;
        symbols.insert(handle, (module.to_owned(), name.to_owned()));
        Ok(KernelHandle(handle))
    }
    fn set_kernel_max_dynamic_shared_memory(&self, handle: KernelHandle, _: u32) -> Result<()> {
        if !self.symbols.lock().unwrap().contains_key(&handle.0) {
            bail!("attribute for unknown handle");
        }
        Ok(())
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024)
    }
}

pub struct Fixture {
    pub buffers: Glm53Exl3MoeBuffers,
    pub shared: Glm53Exl3Buffer,
    pub combined: Glm53Exl3Buffer,
}
impl Fixture {
    pub fn new(gpu: &TraceGpu, p: Glm53Exl3MoePlan) -> Self {
        let table = || gpu.buffer(p.pointer_table_u64_bytes);
        let pointers = Glm53Exl3MoePointerTables {
            gate_trellis: table(),
            gate_suh: table(),
            gate_svh: table(),
            up_trellis: table(),
            up_suh: table(),
            up_svh: table(),
            down_trellis: table(),
            down_suh: table(),
            down_svh: table(),
        };
        Self {
            buffers: Glm53Exl3MoeBuffers {
                input_f16: gpu.buffer(p.input_f16_bytes),
                output_f32: gpu.buffer(p.output_f32_bytes),
                route_private_f32: gpu.buffer(p.route_private_f32_bytes),
                route_ids_u32: gpu.buffer(p.route_ids_u32_bytes),
                route_weights_f32: gpu.buffer(p.route_weights_f32_bytes),
                expert_count_i64: gpu.buffer(p.expert_count_i64_bytes),
                token_sorted_i64: gpu.buffer(p.token_sorted_i64_bytes),
                weight_sorted_f16: gpu.buffer(p.weight_sorted_f16_bytes),
                temp_state_g_f16: gpu.buffer(p.temp_state_f16_bytes),
                temp_state_u_f16: gpu.buffer(p.temp_state_f16_bytes),
                temp_intermediate_g_f16: gpu.buffer(p.temp_intermediate_f16_bytes),
                temp_intermediate_u_f16: gpu.buffer(p.temp_intermediate_f16_bytes),
                route_status_u32: gpu.buffer(4),
                locks_i32: gpu.buffer(GLM53_EXL3_MOE_LOCK_BYTES),
                pair_expert_u32: gpu.buffer(p.pair_expert_u32_bytes),
                chunk_expert_u32: gpu.buffer(p.chunk_descriptor_u32_bytes),
                chunk_start_u32: gpu.buffer(p.chunk_descriptor_u32_bytes),
                chunk_rows_u32: gpu.buffer(p.chunk_descriptor_u32_bytes),
                chunk_count_u32: gpu.buffer(p.chunk_count_u32_bytes),
                pointers,
            },
            shared: gpu.buffer(p.input_f16_bytes),
            combined: gpu.buffer(p.input_f16_bytes),
        }
    }
}
