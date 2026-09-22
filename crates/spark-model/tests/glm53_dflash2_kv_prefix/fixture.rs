// SPDX-License-Identifier: AGPL-3.0-only
//! Deferred real byte/tag storage, not a floating-point or attention oracle.
use crate::cache::{KvPrefixIo, KvTail};
use anyhow::{Result, bail};

#[derive(Clone, Copy)]
pub enum Fault {
    None,
    Enqueue(usize),
    Panic(usize),
    Fence,
}

pub fn tag(layer: usize, row: usize) -> u16 {
    u16::try_from(1 + layer * 4096 + row).unwrap()
}

pub struct DeferredIo {
    pub k: Vec<Vec<u16>>,
    pub v: Vec<Vec<u16>>,
    pub calls: Vec<(usize, KvTail, u64)>,
    pub fences: Vec<u64>,
    pub fault: Fault,
    queued: Vec<(usize, KvTail, u64)>,
}

impl DeferredIo {
    pub fn new() -> Self {
        Self {
            k: vec![vec![0; 2047 + 8]; 5],
            v: vec![vec![0; 2047 + 8]; 5],
            calls: vec![],
            fences: vec![],
            fault: Fault::None,
            queued: vec![],
        }
    }

    pub fn assert_committed(&self, rows: u32) {
        for layer in 0..5 {
            for row in 0..rows as usize {
                assert_eq!(
                    self.k[layer][row],
                    tag(layer, row),
                    "K layer={layer} row={row}"
                );
                assert_eq!(
                    self.v[layer][row],
                    tag(layer, row) ^ 0x4000,
                    "V layer={layer} row={row}"
                );
            }
        }
    }
}

impl KvPrefixIo for DeferredIo {
    fn enqueue_layer(&mut self, layer: usize, plan: KvTail, stream: u64) -> Result<()> {
        assert!(layer < 5);
        assert!(plan.new_rows() > 0);
        assert_eq!(plan.source_row(), plan.retained_rows());
        assert_eq!(plan.source_row() + plan.new_rows(), plan.committed_end());
        self.calls.push((layer, plan, stream));
        self.queued.push((layer, plan, stream));
        match self.fault {
            Fault::Enqueue(at) if at == layer => bail!("injected after enqueue"),
            Fault::Panic(at) if at == layer => panic!("injected after enqueue"),
            _ => Ok(()),
        }
    }

    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.fences.push(stream);
        if matches!(self.fault, Fault::Fence) {
            bail!("injected fence failure");
        }
        for (layer, plan, owner_stream) in self.queued.drain(..) {
            assert_eq!(stream, owner_stream);
            for row in plan.source_row() as usize..plan.committed_end() as usize {
                self.k[layer][row] = tag(layer, row);
                self.v[layer][row] = tag(layer, row) ^ 0x4000;
            }
            // Provisional draft slots exist physically but grant no committed cursor.
            for row in plan.committed_end() as usize..plan.committed_end() as usize + 8 {
                self.k[layer][row] = 0xc000 | row as u16;
                self.v[layer][row] = 0xe000 | row as u16;
            }
        }
        Ok(())
    }
}
