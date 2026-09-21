// SPDX-License-Identifier: AGPL-3.0-only

//! Exact text-token inputs shared by the DSpark embedding and visual MoE paths.
//! This plan is validated before any ring update or token-table lookup.

pub(super) struct DraftInputPlan {
    tokens: Vec<u32>,
    bytes: Vec<u8>,
}

impl DraftInputPlan {
    pub(super) fn new(
        block: usize,
        committed: u32,
        noise: u32,
        vocab: u32,
    ) -> Result<Self, &'static str> {
        if !(1..=8).contains(&block) {
            return Err("DSpark block width must be 1..8");
        }
        if vocab == 0 || committed >= vocab || noise >= vocab {
            return Err("DSpark committed and noise IDs must be inside the text vocabulary");
        }
        let mut tokens = vec![noise; block];
        tokens[0] = committed;
        let bytes = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        Ok(Self { tokens, bytes })
    }

    pub(super) fn token_ids(&self) -> &[u32] {
        &self.tokens
    }

    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
