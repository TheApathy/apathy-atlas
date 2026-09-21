// SPDX-License-Identifier: AGPL-3.0-only

use super::authority::{HeldFile, preflight_authority, preflight_cubins};
use super::digest::hex_sha256;
use super::ffi::{self, ContextLease, Function, Module};
use super::manifest::{KERNELS, KernelSpec};
use anyhow::{Result, ensure};
use std::collections::HashSet;

pub struct PreparedArtifacts {
    authority: Vec<HeldFile>,
    cubins: Vec<HeldFile>,
}

impl PreparedArtifacts {
    pub fn preflight() -> Result<Self> {
        let authority = preflight_authority()?;
        let mut roles = HashSet::new();
        let mut functions = HashSet::new();
        for spec in &KERNELS {
            ensure!(roles.insert(spec.role), "duplicate kernel role");
            ensure!(functions.insert(spec.function), "duplicate kernel function");
        }
        let cubins = preflight_cubins()?;
        ensure!(cubins.len() == 5, "exact five-cubin family required");
        Ok(Self { authority, cubins })
    }

    fn recheck_all(&self) -> Result<()> {
        self.authority
            .iter()
            .chain(self.cubins.iter())
            .try_for_each(HeldFile::recheck)
    }

    pub unsafe fn load_on_current_context(self) -> Result<LoadedFamily> {
        self.recheck_all()?;
        let context = unsafe { ContextLease::retain_current_gb10() }?;
        unsafe { context.ensure_current() }?;
        let mut kernels = Vec::with_capacity(5);
        for (spec, held) in KERNELS.iter().zip(self.cubins) {
            kernels.push(unsafe { LoadedKernel::load(spec, held) }?);
        }
        let mut handles = HashSet::new();
        ensure!(
            kernels.iter().all(|kernel| handles.insert(kernel.function)),
            "duplicate CUDA function handle"
        );
        self.authority.iter().try_for_each(HeldFile::recheck)?;
        Ok(LoadedFamily { kernels, context })
    }
}

struct ModuleGuard(Option<Module>);

impl Drop for ModuleGuard {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            unsafe { ffi::unload(module) };
        }
    }
}

pub(super) struct LoadedKernel {
    pub spec: &'static KernelSpec,
    module: Module,
    pub function: Function,
    _held: HeldFile,
}

impl LoadedKernel {
    unsafe fn load(spec: &'static KernelSpec, held: HeldFile) -> Result<Self> {
        ensure!(
            hex_sha256(&held.bytes) == spec.cubin_sha256,
            "held cubin drift"
        );
        let module = unsafe { ffi::load_data(&held.bytes) }?;
        let mut guard = ModuleGuard(Some(module));
        ensure!(
            hex_sha256(&held.bytes) == spec.cubin_sha256,
            "cubin changed in load"
        );
        held.recheck()?;
        let function = unsafe { ffi::function(module, spec.function) }?;
        let expected = [
            (ffi::FUNC_STATIC_SHARED, spec.static_shared),
            (ffi::FUNC_LOCAL_BYTES, spec.local_bytes),
            (ffi::FUNC_REGISTERS, spec.registers),
        ];
        for (attribute, wanted) in expected {
            ensure!(
                unsafe { ffi::func_attribute(function, attribute) }? == wanted,
                "CUDA resource drift"
            );
        }
        ensure!(
            unsafe { ffi::func_attribute(function, ffi::FUNC_MAX_THREADS) }? >= spec.block_x as i32,
            "CUDA max-thread resource drift"
        );
        unsafe { ffi::set_dynamic_shared(function, spec.dynamic_shared) }?;
        guard.0.take();
        Ok(Self {
            spec,
            module,
            function,
            _held: held,
        })
    }
}

impl Drop for LoadedKernel {
    fn drop(&mut self) {
        unsafe { ffi::unload(self.module) };
    }
}

pub struct LoadedFamily {
    kernels: Vec<LoadedKernel>,
    context: ContextLease,
}

impl LoadedFamily {
    pub(super) fn kernel(&self, role: &str) -> Result<&LoadedKernel> {
        self.kernels
            .iter()
            .find(|kernel| kernel.spec.role == role)
            .ok_or_else(|| anyhow::anyhow!("missing loaded kernel role {role}"))
    }

    pub(super) unsafe fn ensure_context(&self) -> Result<()> {
        unsafe { self.context.ensure_current() }
    }
}
