// SPDX-License-Identifier: AGPL-3.0-only

pub(super) const ROWS: [u32; 2] = [2_079, 8_192];
pub(super) const COLS: [u32; 2] = [5_120, 6_144];
pub(super) const WEIGHT_SCALE_2: f32 = 0.75;
pub(super) const REDZONE: usize = 4 * 1_024;
pub(super) const INVALID_CHILD_ENV: &str = "ATLAS_DYNAMIC_SCALE_INVALID_CHILD";

pub(super) const ATTR_MULTIPROCESSOR_COUNT: u32 = 16;
pub(super) const ATTR_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
pub(super) const ATTR_COMPUTE_CAPABILITY_MINOR: u32 = 76;
pub(super) const FUNC_ATTR_MAX_THREADS_PER_BLOCK: u32 = 0;
pub(super) const FUNC_ATTR_SHARED_SIZE_BYTES: u32 = 1;
pub(super) const FUNC_ATTR_LOCAL_SIZE_BYTES: u32 = 3;
pub(super) const FUNC_ATTR_NUM_REGS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fixture {
    Production,
    Cancellation,
    Halfway,
    SignedZero,
    AllZero,
    Subnormal,
    MaxFinite,
}

impl Fixture {
    pub(super) const ALL: [Self; 7] = [
        Self::Production,
        Self::Cancellation,
        Self::Halfway,
        Self::SignedZero,
        Self::AllZero,
        Self::Subnormal,
        Self::MaxFinite,
    ];
}

#[derive(Debug)]
pub(super) struct FunctionResources {
    pub(super) max_threads: i32,
    pub(super) shared_bytes: i32,
    pub(super) local_bytes: i32,
    pub(super) registers: i32,
}
