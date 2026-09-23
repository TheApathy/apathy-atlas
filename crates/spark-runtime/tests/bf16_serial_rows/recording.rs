// SPDX-License-Identifier: AGPL-3.0-only
//! Fake external descriptor I/O only: admission/lifecycle live in production.

use super::diagnostic_contract::HeuristicResult;
use super::serial_rows_contract::{
    ByteSpan, MatrixLayout, Orientation, RowCall, SerialRowsRequest,
};
use super::serial_rows_driver::{DescriptorSet, ResourceKind, SerialRowsIo};

pub const WORKSPACE_BYTES: usize = 64 * 1024 * 1024;
pub fn workspace() -> ByteSpan {
    ByteSpan {
        address: 0x8000_0000,
        bytes: WORKSPACE_BYTES,
    }
}
pub fn request(rows: u32, orientation: Orientation) -> SerialRowsRequest {
    SerialRowsRequest {
        rows,
        n: 64,
        k: 128,
        orientation,
        stream: 17,
        act: ByteSpan {
            address: 0x10_0000,
            bytes: rows as usize * 256,
        },
        weight: ByteSpan {
            address: 0x20_0000,
            bytes: 64 * 128 * 2,
        },
        out: ByteSpan {
            address: 0x30_0000,
            bytes: rows as usize * 128,
        },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    CreateDesc(i32, i32),
    SetDesc(u64, u32, i32),
    CreateLayout(MatrixLayout),
    CreatePref,
    SetPref(u64, u32, usize),
    Heuristic(DescriptorSet<u64>, i32),
    Matmul(DescriptorSet<u64>, RowCall, [u64; 8]),
    PrepareBatch(DescriptorSet<u64>, SerialRowsRequest, RowCall, [u64; 8]),
    Destroy(ResourceKind, u64),
}

#[derive(Clone, Copy, Debug)]
pub struct HeuristicOverride {
    pub query: usize,
    pub returned: i32,
    pub state: i32,
    pub bytes: usize,
    pub waves: f32,
}

#[derive(Default)]
pub struct Recording {
    pub events: Vec<Event>,
    pub created: Vec<(ResourceKind, u64)>,
    pub live: Vec<(ResourceKind, u64)>,
    pub fail_at: Option<usize>,
    pub panic_at: Option<usize>,
    pub destroy_errors: Vec<u64>,
    pub heuristic_override: Option<HeuristicOverride>,
    pub queries: usize,
    pub next: u64,
    pub batch_caps: Option<[u32; 5]>,
}
impl Recording {
    fn event(&mut self, event: Event) -> Result<(), String> {
        let index = self.events.len();
        self.events.push(event);
        if self.panic_at == Some(index) {
            panic!("injected row-driver panic");
        }
        if self.fail_at == Some(index) {
            return Err(format!("io-{index}"));
        }
        Ok(())
    }
    fn acquired(&mut self, kind: ResourceKind) -> u64 {
        self.next += 1;
        self.created.push((kind, self.next));
        self.live.push((kind, self.next));
        self.next
    }
    pub fn destroyed(&self) -> Vec<(ResourceKind, u64)> {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::Destroy(kind, handle) => Some((*kind, *handle)),
                _ => None,
            })
            .collect()
    }
    pub fn matmuls(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, Event::Matmul(..)))
            .count()
    }
    pub fn queried(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, Event::Heuristic(..)))
            .count()
    }
}
impl SerialRowsIo for Recording {
    type Handle = u64;
    type Error = String;
    fn create_matmul(&mut self, compute: i32, scale: i32) -> Result<u64, String> {
        self.event(Event::CreateDesc(compute, scale))?;
        Ok(self.acquired(ResourceKind::Descriptor))
    }
    fn set_matmul_i32(&mut self, desc: u64, attr: u32, value: i32) -> Result<(), String> {
        self.event(Event::SetDesc(desc, attr, value))
    }
    fn create_layout(&mut self, layout: MatrixLayout) -> Result<u64, String> {
        self.event(Event::CreateLayout(layout))?;
        Ok(self.acquired(ResourceKind::Layout))
    }
    fn create_preference(&mut self) -> Result<u64, String> {
        self.event(Event::CreatePref)?;
        Ok(self.acquired(ResourceKind::Preference))
    }
    fn set_preference_usize(&mut self, pref: u64, attr: u32, value: usize) -> Result<(), String> {
        self.event(Event::SetPref(pref, attr, value))
    }
    fn heuristic(
        &mut self,
        set: DescriptorSet<u64>,
        requested: i32,
    ) -> Result<(HeuristicResult, i32), String> {
        self.event(Event::Heuristic(set, requested))?;
        let query = self.queries;
        self.queries += 1;
        let mut result = HeuristicResult::default();
        result.algo = [0x100 + query as u64; 8];
        result.workspace_bytes = 4096;
        result.waves = 1.0;
        let mut returned = 1;
        if let Some(fault) = self.heuristic_override.filter(|f| f.query == query) {
            returned = fault.returned;
            result.state = fault.state;
            result.workspace_bytes = fault.bytes;
            result.waves = fault.waves;
        }
        Ok((result, returned))
    }
    fn matmul(
        &mut self,
        set: DescriptorSet<u64>,
        call: RowCall,
        result: &HeuristicResult,
    ) -> Result<(), String> {
        self.event(Event::Matmul(set, call, result.algo))
    }
    fn destroy(&mut self, kind: ResourceKind, handle: u64) -> Result<(), String> {
        self.event(Event::Destroy(kind, handle))?;
        if self.destroy_errors.contains(&handle) {
            return Err(format!("destroy-{handle}"));
        }
        let index = self
            .live
            .iter()
            .position(|r| *r == (kind, handle))
            .expect("destroy must target a live acquired resource exactly once");
        self.live.remove(index);
        Ok(())
    }
    fn prepare_strided_m1(
        &mut self,
        set: DescriptorSet<u64>,
        request: SerialRowsRequest,
        call: RowCall,
        result: &HeuristicResult,
    ) -> Result<bool, String> {
        self.event(Event::PrepareBatch(set, request, call, result.algo))?;
        let Some(caps) = self.batch_caps else {
            return Ok(false);
        };
        let strides = super::strided_rows_contract::admit_strided_m1(request, call, caps)
            .map_err(str::to_owned)?;
        assert_eq!(strides, [0, i64::from(request.k), i64::from(request.n)]);
        Ok(true)
    }
}
