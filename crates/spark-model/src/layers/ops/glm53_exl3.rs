// SPDX-License-Identifier: AGPL-3.0-only

//! Native host binding for the pinned ExLlamaV3 mul1 trellis GEMM.

use std::ffi::c_void;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const SHARED_MEMORY_BYTES: u32 = 90 * 1024;
// CUDA reports 48 multiprocessors on the pinned GB10 target. This value also
// determines the cooperative-grid cap, so understating it leaves most of the
// device idle while remaining deceptively correct.
const GB10_MULTIPROCESSORS: u32 = 48;
// Pinned sm_121a resource census: narrow uses 63-64 registers at 512 threads
// (two resident blocks/SM); wide uses 77-80 at 256 threads (three/SM).
const GEMV_NARROW_RESIDENT_BLOCKS: u32 = 2;
const GEMV_WIDE_RESIDENT_BLOCKS: u32 = 3;
const LOCK_BYTES: usize = 1024 * 1024 * size_of::<i32>();
const SYMBOL_SUFFIX: &str = "EvPK6__halfPKtPviiiPiS2_PS0_S2_";
const KDA_QKV_MODULE: &str = "glm53_exl3_kda_qkv_staged";
const KDA_QKV_ROWS_PER_BLOCK: u32 = 16;
const KDA_QKV_HADAMARD_RESIDENT_BLOCKS: u32 = 4;
const KDA_QKV_THREADS: u32 = 256;
const KDA_QKV_WARPS_PER_BLOCK: u32 = KDA_QKV_THREADS / 32;
const KDA_QKV_SHARED_MEMORY_BYTES: u32 = 24_064;
const KDA_QKV_SYMBOL: &str = "atlas_glm53_exl3_kda_qkv_n256";
const KDA_QKV_PAIR2_SYMBOL: &str = "atlas_glm53_exl3_kda_qkv_n256_pair2";
const KDA_QKV_ROW_X_MODULE: &str = "glm53_exl3_kda_qkv_row_x";
const KDA_QKV_ROW_X_SYMBOL: &str = "atlas_glm53_exl3_kda_qkv_n256_row_x";
const K32_N128_MODULE: &str = "glm53_exl3_k32_n128_staged";
const K32_N128_ROWS_PER_BLOCK: u32 = 16;
const K32_N128_THREADS: u32 = 512;
const K32_N128_SHARED_MEMORY_BYTES: u32 = 20_480;
const K32_N128_SYMBOL: &str = "atlas_glm53_exl3_k32_n128_sh4_f1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53Exl3Output {
    F16,
    F32,
}

impl Glm53Exl3Output {
    fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    fn mangled_bool(self) -> u8 {
        match self {
            Self::F16 => 0,
            Self::F32 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Exl3Shape {
    index: u8,
    tile_k: u32,
    tile_n: u32,
    block: u32,
    mangled_tail: &'static str,
}

const SHAPE_1: Exl3Shape = Exl3Shape {
    index: 1,
    tile_k: 16,
    tile_n: 128,
    block: 256,
    mangled_tail: "ELi16ELi16ELi128ELi6ELi5E",
};
const SHAPE_2: Exl3Shape = Exl3Shape {
    index: 2,
    tile_k: 32,
    tile_n: 128,
    block: 512,
    mangled_tail: "ELi16ELi32ELi128ELi4ELi3E",
};
const SHAPE_3: Exl3Shape = Exl3Shape {
    index: 3,
    tile_k: 32,
    tile_n: 256,
    block: 512,
    mangled_tail: "ELi16ELi32ELi256ELi4ELi3E",
};
const SHAPE_4: Exl3Shape = Exl3Shape {
    index: 4,
    tile_k: 16,
    tile_n: 512,
    block: 256,
    mangled_tail: "ELi16ELi16ELi512ELi4ELi3E",
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53Exl3GemmPlan {
    pub rows: u32,
    pub size_k: u32,
    pub size_n: u32,
    pub bits: u8,
    pub output: Glm53Exl3Output,
    pub shape_index: u8,
    pub grid: u32,
    pub block: u32,
    pub gemv_config: Option<u8>,
    pub input_bytes: usize,
    pub trellis_bytes: usize,
    pub output_bytes: usize,
    pub scale_in_bytes: usize,
    pub scale_out_bytes: usize,
}

impl Glm53Exl3GemmPlan {
    pub fn new(
        rows: u32,
        size_k: u32,
        size_n: u32,
        bits: u8,
        output: Glm53Exl3Output,
    ) -> Result<Self> {
        Self::build(rows, size_k, size_n, bits, output, true)
    }

    /// Build the regular cooperative GEMM plan even for small-M inputs.
    /// This is used only by the exact wide-prefill experiment, where the
    /// small-M GEMV changes the T=1 arithmetic trajectory.
    pub fn new_regular_gemm(
        rows: u32,
        size_k: u32,
        size_n: u32,
        bits: u8,
        output: Glm53Exl3Output,
    ) -> Result<Self> {
        Self::build(rows, size_k, size_n, bits, output, false)
    }

    fn build(
        rows: u32,
        size_k: u32,
        size_n: u32,
        bits: u8,
        output: Glm53Exl3Output,
        allow_small_m_gemv: bool,
    ) -> Result<Self> {
        if rows == 0 || size_k == 0 || size_n == 0 {
            bail!("EXL3 GEMM dimensions must be nonzero");
        }
        if !matches!(bits, 2..=5) {
            bail!("GLM-5.3 EXL3 admits only text K2/K3/K4/K5 tensors");
        }
        if size_k % 128 != 0 || size_n % 128 != 0 {
            bail!("EXL3 fused Hadamard dimensions must be divisible by 128");
        }
        let shape = select_blackwell_shape(size_k, size_n, bits);
        if size_k % shape.tile_k != 0 || size_n % shape.tile_n != 0 {
            bail!("selected EXL3 kernel shape is incompatible with matrix dimensions");
        }
        let tiles = u64::from(size_k / shape.tile_k)
            .checked_mul(u64::from(size_n / shape.tile_n))
            .context("EXL3 tile count overflow")?;
        let grid = u32::try_from(tiles.max(1).min(u64::from(GB10_MULTIPROCESSORS)))?;
        let gemv_config = if allow_small_m_gemv
            && (2..=8).contains(&rows)
            && matches!(bits, 2..=4)
            && output == Glm53Exl3Output::F16
        {
            Some(if size_n <= 8192 { 0 } else { 1 })
        } else {
            None
        };
        let matrix = usize::try_from(u64::from(size_k) * u64::from(size_n))?;
        let activations = usize::try_from(u64::from(rows) * u64::from(size_k))?;
        let outputs = usize::try_from(u64::from(rows) * u64::from(size_n))?;
        Ok(Self {
            rows,
            size_k,
            size_n,
            bits,
            output,
            shape_index: shape.index,
            grid,
            block: shape.block,
            gemv_config,
            input_bytes: activations
                .checked_mul(2)
                .context("EXL3 input byte overflow")?,
            trellis_bytes: matrix
                .checked_mul(usize::from(bits))
                .context("EXL3 trellis bit overflow")?
                / 8,
            output_bytes: outputs
                .checked_mul(output.bytes())
                .context("EXL3 output byte overflow")?,
            scale_in_bytes: usize::try_from(size_k)?.checked_mul(2).unwrap(),
            scale_out_bytes: usize::try_from(size_n)?.checked_mul(2).unwrap(),
        })
    }

    pub fn module(&self) -> String {
        format!("glm53_exl3_k{}_cb2", self.bits)
    }

    pub fn symbol(&self) -> String {
        let shape = shape_by_index(self.shape_index).expect("validated EXL3 shape");
        format!(
            "_Z16exl3_gemm_kernelILi{}ELb{}ELi2{}{}",
            self.bits,
            self.output.mangled_bool(),
            shape.mangled_tail,
            SYMBOL_SUFFIX
        )
    }
}

fn select_blackwell_shape(size_k: u32, size_n: u32, bits: u8) -> Exl3Shape {
    if matches!(bits, 2 | 4) && size_k <= 2048 {
        return SHAPE_1;
    }
    if size_n % 256 == 0 && size_n <= 4096 {
        return if size_k > 8192 && bits >= 3 {
            SHAPE_3
        } else {
            SHAPE_2
        };
    }
    if size_n % 512 == 0 && size_n > 16384 {
        return SHAPE_4;
    }
    if size_n % 256 == 0 {
        return SHAPE_3;
    }
    SHAPE_2
}

fn shape_by_index(index: u8) -> Option<Exl3Shape> {
    [SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4]
        .into_iter()
        .find(|shape| shape.index == index)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Exl3Buffer {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3GemmBuffers {
    pub input_f16: Glm53Exl3Buffer,
    pub trellis_i16: Glm53Exl3Buffer,
    pub output: Glm53Exl3Buffer,
    pub locks_i32: Glm53Exl3Buffer,
    pub scale_in_f16: Glm53Exl3Buffer,
    pub input_hadamard_f16: Glm53Exl3Buffer,
    pub scale_out_f16: Glm53Exl3Buffer,
}

#[derive(Debug)]
pub struct Glm53Exl3GemmKernel {
    handle: KernelHandle,
    gemv_config: Option<u8>,
    staged_qkv: Option<Glm53Exl3StagedQkvKernel>,
    staged_k32_n128: Option<Glm53Exl3StagedK32N128Kernel>,
}

#[derive(Debug)]
struct Glm53Exl3StagedQkvKernel {
    hadamard: KernelHandle,
    row_tile: Glm53Exl3StagedQkvRowTile,
    grid_order: Glm53Exl3StagedQkvGridOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Glm53Exl3StagedQkvRowTile {
    Rows16,
    Rows32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Glm53Exl3StagedQkvGridOrder {
    WeightX,
    RowX,
}

impl Glm53Exl3StagedQkvRowTile {
    fn rows(self) -> u32 {
        match self {
            Self::Rows16 => 16,
            Self::Rows32 => 32,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Rows16 => KDA_QKV_SYMBOL,
            Self::Rows32 => KDA_QKV_PAIR2_SYMBOL,
        }
    }
}

#[derive(Debug)]
struct Glm53Exl3StagedK32N128Kernel {
    hadamard: KernelHandle,
}

impl Glm53Exl3GemmKernel {
    pub fn load(gpu: &dyn GpuBackend, plan: &Glm53Exl3GemmPlan) -> Result<Self> {
        let staged_qkv_enabled = parse_staged_qkv_enabled(
            std::env::var("ATLAS_GLM53_EXL3_KDA_QKV_STAGED")
                .ok()
                .as_deref(),
        )?;
        let staged_qkv_row_tile = parse_staged_qkv_row_tile(
            std::env::var("ATLAS_GLM53_EXL3_KDA_QKV_ROW_TILE")
                .ok()
                .as_deref(),
        )?;
        let staged_qkv_grid_order = parse_staged_qkv_grid_order(
            std::env::var("ATLAS_GLM53_EXL3_KDA_QKV_GRID_ORDER")
                .ok()
                .as_deref(),
        )?;
        if staged_qkv_enabled && staged_qkv_plan_is_eligible(plan) {
            if staged_qkv_grid_order == Glm53Exl3StagedQkvGridOrder::RowX
                && staged_qkv_row_tile != Glm53Exl3StagedQkvRowTile::Rows16
            {
                bail!("row-X KDA QKV requires the exact 16-row tile");
            }
            let (module, symbol) = match staged_qkv_grid_order {
                Glm53Exl3StagedQkvGridOrder::WeightX => {
                    (KDA_QKV_MODULE, staged_qkv_row_tile.symbol())
                }
                Glm53Exl3StagedQkvGridOrder::RowX => (KDA_QKV_ROW_X_MODULE, KDA_QKV_ROW_X_SYMBOL),
            };
            let handle = gpu.kernel(module, symbol)?;
            gpu.set_kernel_max_dynamic_shared_memory(handle, KDA_QKV_SHARED_MEMORY_BYTES)?;
            return Ok(Self {
                handle,
                gemv_config: None,
                staged_qkv: Some(Glm53Exl3StagedQkvKernel {
                    hadamard: gpu.kernel(KDA_QKV_MODULE, "atlas_glm53_exl3_kda_qkv_hadamard")?,
                    row_tile: staged_qkv_row_tile,
                    grid_order: staged_qkv_grid_order,
                }),
                staged_k32_n128: None,
            });
        }
        let staged_k32_n128 = parse_staged_k32_n128_enabled(
            std::env::var("ATLAS_GLM53_EXL3_K32_N128_STAGED")
                .ok()
                .as_deref(),
        )?;
        if staged_k32_n128 && staged_k32_n128_plan_is_eligible(plan) {
            let handle = gpu.kernel(K32_N128_MODULE, K32_N128_SYMBOL)?;
            gpu.set_kernel_max_dynamic_shared_memory(handle, K32_N128_SHARED_MEMORY_BYTES)?;
            return Ok(Self {
                handle,
                gemv_config: None,
                staged_qkv: None,
                staged_k32_n128: Some(Glm53Exl3StagedK32N128Kernel {
                    hadamard: gpu.kernel(KDA_QKV_MODULE, "atlas_glm53_exl3_kda_qkv_hadamard")?,
                }),
            });
        }
        if let Some(config) = plan.gemv_config {
            let symbol = format!(
                "_Z16exl3_gemv_kernelILi{}ELb0ELi2ELi1ELi{}ELb0EEvPK6__halfPKtPviiiPiS2_PS0_S2_",
                plan.bits, config
            );
            return Ok(Self {
                handle: gpu.kernel("glm53_exl3_gemv_cb2", &symbol)?,
                gemv_config: Some(config),
                staged_qkv: None,
                staged_k32_n128: None,
            });
        }
        let handle = gpu.kernel(&plan.module(), &plan.symbol())?;
        gpu.set_kernel_max_dynamic_shared_memory(handle, SHARED_MEMORY_BYTES)?;
        Ok(Self {
            handle,
            gemv_config: None,
            staged_qkv: None,
            staged_k32_n128: None,
        })
    }

    pub fn load_row_exact(gpu: &dyn GpuBackend, plan: &Glm53Exl3GemmPlan) -> Result<Self> {
        ensure_row_exact_plan(plan)?;
        let family = if row_batch_enabled() {
            "rowbatch"
        } else {
            "rowexact"
        };
        let symbol = format!("glm53_exl3_{family}_k{}_s{}", plan.bits, plan.shape_index);
        let handle = gpu.kernel(&plan.module(), &symbol)?;
        gpu.set_kernel_max_dynamic_shared_memory(handle, SHARED_MEMORY_BYTES)?;
        Ok(Self {
            handle,
            gemv_config: None,
            staged_qkv: None,
            staged_k32_n128: None,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: &Glm53Exl3GemmPlan,
        buffers: Glm53Exl3GemmBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        if let Some(staged_qkv) = &self.staged_qkv {
            return staged_qkv.launch(gpu, self.handle, plan, buffers, stream);
        }
        if let Some(staged_k32_n128) = &self.staged_k32_n128 {
            return staged_k32_n128.launch(gpu, self.handle, plan, buffers, stream);
        }
        let mut input = buffers.input_f16.ptr.0;
        let mut trellis = buffers.trellis_i16.ptr.0;
        let mut output = buffers.output.ptr.0;
        let mut rows = i32::try_from(plan.rows)?;
        let mut size_k = i32::try_from(plan.size_k)?;
        let mut size_n = i32::try_from(plan.size_n)?;
        let mut locks = buffers.locks_i32.ptr.0;
        let mut scale_in = buffers.scale_in_f16.ptr.0;
        let mut input_hadamard = buffers.input_hadamard_f16.ptr.0;
        let mut scale_out = buffers.scale_out_f16.ptr.0;
        let mut params = [
            param(&mut input),
            param(&mut trellis),
            param(&mut output),
            param(&mut rows),
            param(&mut size_k),
            param(&mut size_n),
            param(&mut locks),
            param(&mut scale_in),
            param(&mut input_hadamard),
            param(&mut scale_out),
        ];
        let (grid, block, shared_memory) = match self.gemv_config {
            Some(0) => (
                (plan.size_n / 32).min(GB10_MULTIPROCESSORS * GEMV_NARROW_RESIDENT_BLOCKS),
                512,
                0,
            ),
            Some(1) => (
                (plan.size_n / 64).min(GB10_MULTIPROCESSORS * GEMV_WIDE_RESIDENT_BLOCKS),
                256,
                0,
            ),
            Some(_) => unreachable!("validated EXL3 GEMV config"),
            None => (plan.grid, plan.block, SHARED_MEMORY_BYTES),
        };
        gpu.launch_cooperative(
            self.handle,
            [grid, 1, 1],
            [block, 1, 1],
            shared_memory,
            stream,
            &mut params,
        )
    }
}

impl Glm53Exl3StagedK32N128Kernel {
    fn launch(
        &self,
        gpu: &dyn GpuBackend,
        core: KernelHandle,
        plan: &Glm53Exl3GemmPlan,
        buffers: Glm53Exl3GemmBuffers,
        stream: u64,
    ) -> Result<()> {
        ensure_staged_k32_n128_plan(plan)?;
        let row_tiles = plan.rows.div_ceil(K32_N128_ROWS_PER_BLOCK);
        let lock_bytes = usize::try_from(row_tiles)?
            .checked_mul(usize::try_from(plan.size_n / 16)?)
            .and_then(|locks| locks.checked_mul(size_of::<i32>()))
            .context("staged K32/N128 lock extent overflow")?;
        if lock_bytes > LOCK_BYTES {
            bail!("staged K32/N128 lock slab exceeds the exact workspace");
        }

        let total_hadamard_warps = u64::from(plan.rows)
            .checked_mul(u64::from(plan.size_k / 128))
            .context("staged K32/N128 Hadamard grid overflow")?;
        let hadamard_blocks = u32::try_from(
            total_hadamard_warps
                .div_ceil(u64::from(KDA_QKV_WARPS_PER_BLOCK))
                .min(u64::from(
                    GB10_MULTIPROCESSORS * KDA_QKV_HADAMARD_RESIDENT_BLOCKS,
                )),
        )?;
        let mut input = buffers.input_f16.ptr.0;
        let mut input_hadamard = buffers.input_hadamard_f16.ptr.0;
        let mut scale_in = buffers.scale_in_f16.ptr.0;
        let mut rows = i32::try_from(plan.rows)?;
        let mut size_k = i32::try_from(plan.size_k)?;
        let mut hadamard_params = [
            param(&mut input),
            param(&mut input_hadamard),
            param(&mut scale_in),
            param(&mut rows),
            param(&mut size_k),
        ];
        gpu.launch(
            self.hadamard,
            [hadamard_blocks, 1, 1],
            [KDA_QKV_THREADS, 1, 1],
            0,
            stream,
            &mut hadamard_params,
        )?;

        let mut trellis = buffers.trellis_i16.ptr.0;
        let mut output = buffers.output.ptr.0;
        let mut size_n = i32::try_from(plan.size_n)?;
        let mut locks = buffers.locks_i32.ptr.0;
        let mut scale_out = buffers.scale_out_f16.ptr.0;
        let mut core_params = [
            param(&mut input_hadamard),
            param(&mut trellis),
            param(&mut output),
            param(&mut rows),
            param(&mut size_k),
            param(&mut size_n),
            param(&mut locks),
            param(&mut scale_out),
        ];
        gpu.launch(
            core,
            [GB10_MULTIPROCESSORS, 1, row_tiles],
            [K32_N128_THREADS, 1, 1],
            K32_N128_SHARED_MEMORY_BYTES,
            stream,
            &mut core_params,
        )
    }
}

impl Glm53Exl3StagedQkvKernel {
    fn launch(
        &self,
        gpu: &dyn GpuBackend,
        core: KernelHandle,
        plan: &Glm53Exl3GemmPlan,
        buffers: Glm53Exl3GemmBuffers,
        stream: u64,
    ) -> Result<()> {
        ensure_staged_qkv_plan(plan)?;
        let lock_tiles = plan.rows.div_ceil(KDA_QKV_ROWS_PER_BLOCK);
        let row_tiles = plan.rows.div_ceil(self.row_tile.rows());
        let lock_bytes = usize::try_from(lock_tiles)?
            .checked_mul(usize::try_from(plan.size_n / 16)?)
            .and_then(|locks| locks.checked_mul(size_of::<i32>()))
            .context("staged KDA QKV lock extent overflow")?;
        if lock_bytes > LOCK_BYTES {
            bail!("staged KDA QKV lock slab exceeds the exact workspace");
        }

        let total_hadamard_warps = u64::from(plan.rows)
            .checked_mul(u64::from(plan.size_k / 128))
            .context("staged KDA QKV Hadamard grid overflow")?;
        let hadamard_blocks = u32::try_from(
            total_hadamard_warps
                .div_ceil(u64::from(KDA_QKV_WARPS_PER_BLOCK))
                .min(u64::from(
                    GB10_MULTIPROCESSORS * KDA_QKV_HADAMARD_RESIDENT_BLOCKS,
                )),
        )?;
        let mut input = buffers.input_f16.ptr.0;
        let mut input_hadamard = buffers.input_hadamard_f16.ptr.0;
        let mut scale_in = buffers.scale_in_f16.ptr.0;
        let mut rows = i32::try_from(plan.rows)?;
        let mut size_k = i32::try_from(plan.size_k)?;
        let mut hadamard_params = [
            param(&mut input),
            param(&mut input_hadamard),
            param(&mut scale_in),
            param(&mut rows),
            param(&mut size_k),
        ];
        gpu.launch(
            self.hadamard,
            [hadamard_blocks, 1, 1],
            [KDA_QKV_THREADS, 1, 1],
            0,
            stream,
            &mut hadamard_params,
        )?;

        let mut trellis = buffers.trellis_i16.ptr.0;
        let mut output = buffers.output.ptr.0;
        let mut size_n = i32::try_from(plan.size_n)?;
        let mut locks = buffers.locks_i32.ptr.0;
        let mut scale_out = buffers.scale_out_f16.ptr.0;
        let mut core_params = [
            param(&mut input_hadamard),
            param(&mut trellis),
            param(&mut output),
            param(&mut rows),
            param(&mut size_k),
            param(&mut size_n),
            param(&mut locks),
            param(&mut scale_out),
        ];
        let grid = match self.grid_order {
            Glm53Exl3StagedQkvGridOrder::WeightX => [GB10_MULTIPROCESSORS, 1, row_tiles],
            Glm53Exl3StagedQkvGridOrder::RowX => [row_tiles, 1, GB10_MULTIPROCESSORS],
        };
        gpu.launch(
            core,
            grid,
            [KDA_QKV_THREADS, 1, 1],
            KDA_QKV_SHARED_MEMORY_BYTES,
            stream,
            &mut core_params,
        )
    }
}

fn parse_staged_qkv_enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(_) => bail!("ATLAS_GLM53_EXL3_KDA_QKV_STAGED must be exactly 0 or 1"),
    }
}

fn parse_staged_qkv_row_tile(value: Option<&str>) -> Result<Glm53Exl3StagedQkvRowTile> {
    match value {
        None | Some("16") => Ok(Glm53Exl3StagedQkvRowTile::Rows16),
        Some("32") => Ok(Glm53Exl3StagedQkvRowTile::Rows32),
        Some(_) => bail!("ATLAS_GLM53_EXL3_KDA_QKV_ROW_TILE must be exactly 16 or 32"),
    }
}

fn parse_staged_qkv_grid_order(value: Option<&str>) -> Result<Glm53Exl3StagedQkvGridOrder> {
    match value {
        None | Some("weight-x") => Ok(Glm53Exl3StagedQkvGridOrder::WeightX),
        Some("row-x") => Ok(Glm53Exl3StagedQkvGridOrder::RowX),
        Some(_) => bail!("ATLAS_GLM53_EXL3_KDA_QKV_GRID_ORDER must be exactly weight-x or row-x"),
    }
}

fn staged_qkv_plan_is_eligible(plan: &Glm53Exl3GemmPlan) -> bool {
    plan.rows >= KDA_QKV_ROWS_PER_BLOCK
        && plan.size_k == 4_096
        && plan.size_n == 24_576
        && plan.bits == 4
        && plan.output == Glm53Exl3Output::F16
        && plan.gemv_config.is_none()
}

fn ensure_staged_qkv_plan(plan: &Glm53Exl3GemmPlan) -> Result<()> {
    if !staged_qkv_plan_is_eligible(plan) {
        bail!("staged KDA QKV kernel received an ineligible plan");
    }
    Ok(())
}

fn parse_staged_k32_n128_enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(_) => bail!("ATLAS_GLM53_EXL3_K32_N128_STAGED must be exactly 0 or 1"),
    }
}

fn staged_k32_n128_plan_is_eligible(plan: &Glm53Exl3GemmPlan) -> bool {
    plan.rows >= K32_N128_ROWS_PER_BLOCK
        && plan.bits == 4
        && plan.output == Glm53Exl3Output::F16
        && plan.shape_index == SHAPE_2.index
        && plan.gemv_config.is_none()
}

fn ensure_staged_k32_n128_plan(plan: &Glm53Exl3GemmPlan) -> Result<()> {
    if !staged_k32_n128_plan_is_eligible(plan) {
        bail!("staged K32/N128 kernel received an ineligible plan");
    }
    Ok(())
}

/// `ATLAS_GLM53_EXL3_ROWBATCH=1`: the row-exact projection runs every row in
/// one pass of the pinned M=1 inner kernel (weights read once) instead of one
/// pass per row. Same per-row arithmetic; see `glm53_exl3_rowbatch_body`.
fn row_batch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GLM53_EXL3_ROWBATCH").as_deref() == Ok("1"))
}

fn ensure_row_exact_plan(plan: &Glm53Exl3GemmPlan) -> Result<()> {
    if !(2..=8).contains(&plan.rows)
        || plan.output != Glm53Exl3Output::F16
        || plan.gemv_config.is_some()
    {
        bail!("EXL3 row-exact kernel requires regular F16 GEMM with 2..=8 rows");
    }
    Ok(())
}

fn param<T>(value: &mut T) -> *mut c_void {
    std::ptr::from_mut(value).cast()
}

fn validate_buffers(plan: &Glm53Exl3GemmPlan, buffers: Glm53Exl3GemmBuffers) -> Result<()> {
    let expected = [
        ("input", buffers.input_f16, plan.input_bytes),
        ("trellis", buffers.trellis_i16, plan.trellis_bytes),
        ("output", buffers.output, plan.output_bytes),
        ("locks", buffers.locks_i32, LOCK_BYTES),
        ("input scale", buffers.scale_in_f16, plan.scale_in_bytes),
        (
            "Hadamard scratch",
            buffers.input_hadamard_f16,
            plan.input_bytes,
        ),
        ("output scale", buffers.scale_out_f16, plan.scale_out_bytes),
    ];
    for (name, buffer, bytes) in expected {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != bytes {
            bail!("EXL3 {name} buffer is null or has the wrong exact extent");
        }
        buffer
            .ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .with_context(|| format!("EXL3 {name} device range overflows the address space"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    const STAGED_QKV_CUDA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_kda_qkv_staged.cu"
    ));
    const STAGED_QKV_ROW_X_CUDA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_kda_qkv_row_x.cu"
    ));
    const STAGED_K32_N128_CUDA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k32_n128_staged.cu"
    ));

    #[test]
    fn blackwell_shape_and_mangled_symbol_match_pinned_upstream() {
        let small = Glm53Exl3GemmPlan::new(1, 2048, 4096, 2, Glm53Exl3Output::F16).unwrap();
        assert_eq!((small.shape_index, small.grid, small.block), (1, 48, 256));
        assert_eq!(small.gemv_config, None);
        assert_eq!(small.trellis_bytes, 2048 * 4096 * 2 / 8);
        assert_eq!(
            small.symbol(),
            "_Z16exl3_gemm_kernelILi2ELb0ELi2ELi16ELi16ELi128ELi6ELi5EEvPK6__halfPKtPviiiPiS2_PS0_S2_"
        );

        let wide = Glm53Exl3GemmPlan::new(17, 16384, 32768, 4, Glm53Exl3Output::F32).unwrap();
        assert_eq!((wide.shape_index, wide.block), (4, 256));
        assert!(wide.symbol().contains("ILi4ELb1ELi2"));
        let tall = Glm53Exl3GemmPlan::new(1, 16384, 4096, 3, Glm53Exl3Output::F16).unwrap();
        assert_eq!((tall.shape_index, tall.block), (3, 512));
        let verify = Glm53Exl3GemmPlan::new(7, 4096, 2048, 2, Glm53Exl3Output::F16).unwrap();
        assert_eq!(verify.gemv_config, Some(0));
        let wide_verify = Glm53Exl3GemmPlan::new(7, 4096, 24_576, 4, Glm53Exl3Output::F16).unwrap();
        assert_eq!(wide_verify.gemv_config, Some(1));
        let exact_wide =
            Glm53Exl3GemmPlan::new_regular_gemm(7, 4096, 24_576, 4, Glm53Exl3Output::F16).unwrap();
        assert_eq!(exact_wide.gemv_config, None);
        assert_eq!(exact_wide.symbol(), wide_verify.symbol());
        assert!(ensure_row_exact_plan(&exact_wide).is_ok());
        assert!(ensure_row_exact_plan(&wide_verify).is_err());
    }

    #[test]
    fn unsupported_geometry_and_bits_fail_closed() {
        assert!(Glm53Exl3GemmPlan::new(0, 4096, 4096, 2, Glm53Exl3Output::F16).is_err());
        assert!(Glm53Exl3GemmPlan::new(1, 4097, 4096, 2, Glm53Exl3Output::F16).is_err());
        assert!(Glm53Exl3GemmPlan::new(1, 4096, 4096, 6, Glm53Exl3Output::F16).is_err());
    }

    #[test]
    fn unsupported_backend_rejects_function_attribute_during_load() {
        let gpu = MockGpuBackend::new();
        let plan = Glm53Exl3GemmPlan::new(1, 2048, 4096, 2, Glm53Exl3Output::F16).unwrap();
        let error = Glm53Exl3GemmKernel::load(&gpu, &plan).unwrap_err();
        assert!(error.to_string().contains("unsupported"));
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn staged_qkv_selector_is_default_on_with_exact_rollback() {
        let plan =
            Glm53Exl3GemmPlan::new_regular_gemm(1_875, 4_096, 24_576, 4, Glm53Exl3Output::F16)
                .unwrap();
        assert!(parse_staged_qkv_enabled(None).unwrap());
        assert!(parse_staged_qkv_enabled(Some("1")).unwrap());
        assert!(!parse_staged_qkv_enabled(Some("0")).unwrap());
        assert!(parse_staged_qkv_enabled(Some("n256")).is_err());
        assert_eq!(
            parse_staged_qkv_row_tile(None).unwrap(),
            Glm53Exl3StagedQkvRowTile::Rows16
        );
        assert_eq!(
            parse_staged_qkv_row_tile(Some("16")).unwrap(),
            Glm53Exl3StagedQkvRowTile::Rows16
        );
        assert_eq!(
            parse_staged_qkv_row_tile(Some("32")).unwrap(),
            Glm53Exl3StagedQkvRowTile::Rows32
        );
        assert!(parse_staged_qkv_row_tile(Some("64")).is_err());
        assert!(staged_qkv_plan_is_eligible(&plan));

        let decode =
            Glm53Exl3GemmPlan::new_regular_gemm(8, 4_096, 24_576, 4, Glm53Exl3Output::F16).unwrap();
        assert!(!staged_qkv_plan_is_eligible(&decode));
        let other =
            Glm53Exl3GemmPlan::new_regular_gemm(1_875, 4_096, 4_096, 4, Glm53Exl3Output::F16)
                .unwrap();
        assert!(!staged_qkv_plan_is_eligible(&other));
    }

    #[test]
    fn staged_qkv_geometry_and_resource_contract_are_pinned() {
        assert_eq!(KDA_QKV_SHARED_MEMORY_BYTES, 24_064);
        assert_eq!(KDA_QKV_SYMBOL, "atlas_glm53_exl3_kda_qkv_n256");
        assert_eq!(KDA_QKV_PAIR2_SYMBOL, "atlas_glm53_exl3_kda_qkv_n256_pair2");
        assert_eq!(Glm53Exl3StagedQkvRowTile::Rows16.rows(), 16);
        assert_eq!(Glm53Exl3StagedQkvRowTile::Rows32.rows(), 32);
        assert_eq!(Glm53Exl3StagedQkvRowTile::Rows16.symbol(), KDA_QKV_SYMBOL);
        assert_eq!(
            Glm53Exl3StagedQkvRowTile::Rows32.symbol(),
            KDA_QKV_PAIR2_SYMBOL
        );
        let max_row_tiles = 2_048u32.div_ceil(KDA_QKV_ROWS_PER_BLOCK);
        let lock_bytes = max_row_tiles as usize * (24_576 / 16) * size_of::<i32>();
        assert!(lock_bytes <= LOCK_BYTES);
        assert!(STAGED_QKV_CUDA.contains("There is no split-K producer"));
        assert!(STAGED_QKV_CUDA.contains("#define barrier_acquire(lock, stage)"));
        assert!(STAGED_QKV_CUDA.contains("const int row = 16 * blockIdx.z"));
        assert!(
            STAGED_QKV_CUDA.contains("QKV_KERNEL(atlas_glm53_exl3_kda_qkv_n256, 256, 3, 1, 3)")
        );
        assert!(STAGED_QKV_CUDA.contains("const int row = 32 * blockIdx.z"));
        assert!(STAGED_QKV_CUDA.contains("qkv_row_tile_at<256, 3, 1>"));
        assert!(STAGED_QKV_CUDA.contains("row + 16"));
        assert!(!STAGED_QKV_CUDA.contains("kda_qkv_n512"));
        assert!(!STAGED_QKV_CUDA.contains("kda_qkv_n128"));
    }

    #[test]
    fn staged_qkv_row_x_grid_order_is_explicit_and_exact() {
        assert_eq!(
            parse_staged_qkv_grid_order(None).unwrap(),
            Glm53Exl3StagedQkvGridOrder::WeightX
        );
        assert_eq!(
            parse_staged_qkv_grid_order(Some("weight-x")).unwrap(),
            Glm53Exl3StagedQkvGridOrder::WeightX
        );
        assert_eq!(
            parse_staged_qkv_grid_order(Some("row-x")).unwrap(),
            Glm53Exl3StagedQkvGridOrder::RowX
        );
        assert!(parse_staged_qkv_grid_order(Some("xz")).is_err());
        assert_eq!(KDA_QKV_ROW_X_MODULE, "glm53_exl3_kda_qkv_row_x");
        assert_eq!(KDA_QKV_ROW_X_SYMBOL, "atlas_glm53_exl3_kda_qkv_n256_row_x");
        assert!(STAGED_QKV_ROW_X_CUDA.contains("#define blockIdx atlas_swapped_block_idx()"));
        assert!(STAGED_QKV_ROW_X_CUDA.contains("#define gridDim atlas_swapped_grid_dim()"));
        assert!(STAGED_QKV_ROW_X_CUDA.contains("const int row = 16 * blockIdx.x"));
        assert!(STAGED_QKV_ROW_X_CUDA.contains("locks + blockIdx.x * lock_stride"));
    }

    #[test]
    fn staged_k32_n128_selector_is_default_on_with_rollback_and_shape_scope() {
        assert!(parse_staged_k32_n128_enabled(None).unwrap());
        assert!(parse_staged_k32_n128_enabled(Some("1")).unwrap());
        assert!(!parse_staged_k32_n128_enabled(Some("0")).unwrap());
        assert!(parse_staged_k32_n128_enabled(Some("sh4f1")).is_err());

        for (size_k, size_n) in [(4_096, 2_048), (8_192, 4_096), (4_096, 1_536), (4_096, 512)] {
            let plan =
                Glm53Exl3GemmPlan::new_regular_gemm(1_875, size_k, size_n, 4, Glm53Exl3Output::F16)
                    .unwrap();
            assert_eq!(plan.shape_index, SHAPE_2.index);
            assert!(staged_k32_n128_plan_is_eligible(&plan));
        }

        let short =
            Glm53Exl3GemmPlan::new_regular_gemm(8, 4_096, 2_048, 4, Glm53Exl3Output::F16).unwrap();
        assert!(!staged_k32_n128_plan_is_eligible(&short));
        let qkv =
            Glm53Exl3GemmPlan::new_regular_gemm(1_875, 4_096, 24_576, 4, Glm53Exl3Output::F16)
                .unwrap();
        assert!(!staged_k32_n128_plan_is_eligible(&qkv));
        let k3 = Glm53Exl3GemmPlan::new_regular_gemm(1_875, 4_096, 2_048, 3, Glm53Exl3Output::F16)
            .unwrap();
        assert!(!staged_k32_n128_plan_is_eligible(&k3));
    }

    #[test]
    fn staged_k32_n128_geometry_and_resource_contract_are_pinned() {
        assert_eq!(K32_N128_THREADS, 512);
        assert_eq!(GB10_MULTIPROCESSORS, 48);
        assert_eq!(K32_N128_SHARED_MEMORY_BYTES, 20_480);
        assert_eq!(K32_N128_SYMBOL, "atlas_glm53_exl3_k32_n128_sh4_f1");
        let max_row_tiles = 2_048u32.div_ceil(K32_N128_ROWS_PER_BLOCK);
        let lock_bytes = max_row_tiles as usize * (4_096 / 16) * size_of::<i32>();
        assert!(lock_bytes <= LOCK_BYTES);
        assert!(STAGED_K32_N128_CUDA.contains("exact 48-way split-K partial-sum"));
        assert!(STAGED_K32_N128_CUDA.contains("#include <quant/exl3_gemm_inner.cuh>"));
        assert!(!STAGED_K32_N128_CUDA.contains("#define barrier_acquire"));
        assert!(STAGED_K32_N128_CUDA.contains("const int row = 16 * blockIdx.z"));
        assert!(STAGED_K32_N128_CUDA.contains("<4, false, 2, 16, 32, 128"));
        assert!(STAGED_K32_N128_CUDA.contains("__launch_bounds__(K32_N128_THREADS, 2)"));
        assert!(STAGED_K32_N128_CUDA.contains(K32_N128_SYMBOL));
        assert!(!STAGED_K32_N128_CUDA.contains("k32_n128_sh3"));
        assert!(!STAGED_K32_N128_CUDA.contains("k32_n128_sh2"));
    }
}
