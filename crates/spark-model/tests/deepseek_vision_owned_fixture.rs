// SPDX-License-Identifier: AGPL-3.0-only
//! CPU-only deferred transfers and real owned bytes for the ownership tests.
#![allow(dead_code)]
use std::cell::RefCell;
use std::rc::Rc;

pub type Events = Rc<RefCell<Vec<String>>>;

pub struct Tracked {
    pub bytes: Vec<u8>,
    name: &'static str,
    events: Events,
}
impl Tracked {
    pub fn new(name: &'static str, events: &Events) -> Self {
        Self {
            bytes: vec![0x3f, 0x80, 0x40, 0x00],
            name,
            events: events.clone(),
        }
    }
}
impl Drop for Tracked {
    fn drop(&mut self) {
        self.events.borrow_mut().push(format!("drop:{}", self.name));
    }
}

/// The five entries represent handle/library, workspace, encoder arena,
/// selected-weight ledger, and backend/context lifetime, in release order.
/// No payload clone or CUDA call is used by this deferred-copy fixture.
pub struct Resources {
    pub owners: [Option<Tracked>; 5],
    pub upload: Option<Tracked>,
    pub readback: Option<Tracked>,
    pub events: Events,
    pub fail_fence: bool,
    pub fail_free: Option<usize>,
}
impl Resources {
    pub fn new(events: &Events) -> Self {
        let names = ["handle", "workspace", "arena", "weights", "backend"];
        Self {
            owners: names.map(|name| Some(Tracked::new(name, events))),
            upload: None,
            readback: None,
            events: events.clone(),
            fail_fence: false,
            fail_free: None,
        }
    }
    pub fn enqueue_upload(&mut self, upload: Tracked, expected_ptr: usize) {
        self.upload = Some(upload);
        assert_eq!(
            self.upload.as_ref().unwrap().bytes.as_ptr() as usize,
            expected_ptr
        );
        self.events.borrow_mut().push("enqueue:upload".into());
    }
    pub fn enqueue_readback(&mut self) {
        let mut readback = Tracked::new("readback", &self.events);
        readback.bytes.fill(0xff);
        self.readback = Some(readback);
        self.events.borrow_mut().push("enqueue:readback".into());
    }
    pub fn fence(&mut self) -> Result<(), String> {
        self.events.borrow_mut().push("fence".into());
        if self.fail_fence {
            return Err("injected completion failure".into());
        }
        if let Some(upload) = self.upload.take() {
            self.owners[2]
                .as_mut()
                .unwrap()
                .bytes
                .copy_from_slice(&upload.bytes);
        }
        if let Some(mut readback) = self.readback.take() {
            readback
                .bytes
                .copy_from_slice(&self.owners[2].as_ref().unwrap().bytes);
            assert_eq!(readback.bytes, [0x3f, 0x80, 0x40, 0x00]);
            self.events.borrow_mut().push("readback:complete".into());
        }
        Ok(())
    }
    pub fn free(&mut self) -> Result<(), String> {
        assert!(self.upload.is_none() && self.readback.is_none());
        for (i, owner) in self.owners.iter_mut().enumerate() {
            if owner.is_none() {
                continue;
            }
            self.events.borrow_mut().push(format!("free:{i}"));
            if self.fail_free == Some(i) {
                return Err(format!("injected free failure {i}"));
            }
            // Remove only after the simulated free succeeds, never drain(..).
            drop(owner.take());
        }
        Ok(())
    }
}

pub fn events() -> Events {
    Rc::new(RefCell::new(Vec::new()))
}
pub fn saw(events: &Events, value: &str) -> bool {
    events.borrow().iter().any(|event| event == value)
}
pub fn assert_no_drop(events: &Events) {
    assert!(
        !events
            .borrow()
            .iter()
            .any(|event| event.starts_with("drop:")),
        "an uncertain transfer lost a real Rust owner: {:?}",
        events.borrow()
    );
}
