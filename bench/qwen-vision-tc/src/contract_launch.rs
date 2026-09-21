// SPDX-License-Identifier: AGPL-3.0-only

//! Buffer and launch types; only the parent admission module can bind receipts.

#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub(super) address: u64,
    bytes: usize,
    end: u64,
}

impl Region {
    pub fn new(address: u64, bytes: usize) -> Result<Self, String> {
        if address == 0 || bytes == 0 {
            return Err("empty or null region".into());
        }
        let length = u64::try_from(bytes).map_err(|_| "region length overflow")?;
        let end = address.checked_add(length).ok_or("region end overflow")?;
        Ok(Self {
            address,
            bytes,
            end,
        })
    }

    pub(super) fn require(&self, bytes: usize, alignment: u64) -> Result<(), String> {
        if self.bytes < bytes {
            return Err("region is smaller than the required extent".into());
        }
        if self.address % alignment != 0 {
            return Err("region alignment mismatch".into());
        }
        Ok(())
    }

    pub(super) fn overlaps(&self, other: Self) -> bool {
        self.address < other.end && other.address < self.end
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub a: Region,
    pub b: Region,
    pub bias: Region,
    pub c: Region,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Handles {
    pub scalar: u64,
    pub pipelined: u64,
    pub add_bias: u64,
    pub fused_bias: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    Ptr(u64),
    U32(u32),
}

#[derive(Debug)]
pub struct Launch {
    pub(super) kernel: u64,
    pub(super) grid: [u32; 3],
    pub(super) block: [u32; 3],
    pub(super) args: Vec<Arg>,
}

impl Launch {
    pub fn kernel(&self) -> u64 {
        self.kernel
    }
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }
    pub fn block(&self) -> [u32; 3] {
        self.block
    }
    pub fn args(&self) -> &[Arg] {
        &self.args
    }
}

#[derive(Debug)]
pub struct BoundPlan {
    pub(super) launches: Vec<Launch>,
    pub(super) output_elements: usize,
}

impl BoundPlan {
    pub fn launches(&self) -> &[Launch] {
        &self.launches
    }
    pub fn output_elements(&self) -> usize {
        self.output_elements
    }
}
