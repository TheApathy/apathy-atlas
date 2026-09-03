// SPDX-License-Identifier: AGPL-3.0-only

const CANONICAL: &str = "e85d6a9cba63289590f55a96b2ea18e722381ce11087b38548667c03af75d1ab";
const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub(super) fn digest(input: &[u8]) -> [u8; 32] {
    let bit_len = u64::try_from(input.len())
        .expect("test source length fits u64")
        .checked_mul(8)
        .expect("test source bit length fits u64");
    let padded_len = input
        .len()
        .checked_add(9)
        .and_then(|length| length.checked_add((64 - length % 64) % 64))
        .expect("test source padded length fits usize");
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(input);
    padded.push(0x80);
    padded.resize(padded_len - 8, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());
    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        compress(&mut state, chunk.try_into().expect("exact SHA-256 block"));
    }
    let mut output = [0; 32];
    for (chunk, value) in output.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    output
}

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut words = [0u32; 64];
    for (index, chunk) in block.chunks_exact(4).enumerate() {
        words[index] = u32::from_be_bytes(chunk.try_into().expect("four bytes"));
    }
    for index in 16..64 {
        let x = words[index - 15];
        let y = words[index - 2];
        let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
        let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
        words[index] = words[index - 16]
            .wrapping_add(s0)
            .wrapping_add(words[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ ((!e) & g);
        let first = h
            .wrapping_add(s1)
            .wrapping_add(choose)
            .wrapping_add(K[index])
            .wrapping_add(words[index]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let second = s0.wrapping_add((a & b) ^ (a & c) ^ (b & c));
        (h, g, f, e, d, c, b, a) = (
            g,
            f,
            e,
            d.wrapping_add(first),
            c,
            b,
            a,
            first.wrapping_add(second),
        );
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn matches(source: &str) -> bool {
    hex(digest(source.as_bytes())) == CANONICAL
}

const SOURCE: &str = include_str!("b1t1_bootstrap.rs");
const NONCE: &str = "        let transaction_nonce = self.next_nonce;\n        self.next_nonce = transaction_nonce\n            .checked_add(1)\n            .context(\"GLM B1/T1 bootstrap transaction nonce exhausted\")?;";
const GATHER: &str = "                destination_bf16: buffers.hidden_bf16,\n            },\n            stream,\n        )?;";
const EXPAND: &str = "            buffers.streams_bf16,\n            stream,\n        )?;";
const DEBUG_END: &str =
    "            .finish()\n    }\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]";
const LEAK: &str = "\nimpl fmt::Debug for Glm53B1T1BootstrapBuffers {\n    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n        f.debug_struct(\"Glm53B1T1BootstrapBuffers\").field(\"source_q5_k_bytes\", &(self.source_q5_k.ptr.0 ^ 0x55)).finish()\n    }\n}\n";

fn move_nonce_after(anchor: &str) -> String {
    assert_eq!(SOURCE.matches(NONCE).count(), 1);
    assert_eq!(SOURCE.matches(anchor).count(), 1);
    SOURCE
        .replacen(NONCE, "", 1)
        .replacen(anchor, &format!("{anchor}\n{NONCE}"), 1)
}

#[test]
fn known_vectors_and_canonical_source() {
    assert_eq!(
        hex(digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex(digest(&[b'a'; 55])),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
    );
    assert_eq!(
        hex(digest(&[b'a'; 56])),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
    );
    assert!(matches(SOURCE));
}

#[test]
fn full_source_pin_rejects_live_pointer_debug_and_late_nonce() {
    let disabled = SOURCE.replacen(
        "impl fmt::Debug for Glm53B1T1BootstrapBuffers {",
        "#[cfg(any())]\nimpl fmt::Debug for Glm53B1T1BootstrapBuffers {",
        1,
    );
    let commented = SOURCE
        .replacen(
            "impl fmt::Debug for Glm53B1T1BootstrapBuffers {",
            "/*\nimpl fmt::Debug for Glm53B1T1BootstrapBuffers {",
            1,
        )
        .replacen(
            DEBUG_END,
            "            .finish()\n    }\n}\n*/\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
            1,
        );
    assert!(!matches(&(disabled + LEAK)));
    assert!(!matches(&(commented + LEAK)));
    assert!(!matches(&move_nonce_after(GATHER)));
    assert!(!matches(&move_nonce_after(EXPAND)));
}
