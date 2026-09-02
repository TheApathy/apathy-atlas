// SPDX-License-Identifier: AGPL-3.0-only

use std::io::{self, Read, Seek, SeekFrom};

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

pub(super) struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: INITIAL,
            buffer: [0; 64],
            buffered: 0,
            bytes: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
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
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.bytes = self
            .bytes
            .checked_add(input.len() as u64)
            .expect("SHA-256 input too large");
        if self.buffered != 0 {
            let take = (64 - self.buffered).min(input.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&input[..take]);
            self.buffered += take;
            input = &input[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        for chunk in input.chunks_exact(64) {
            self.compress(chunk.try_into().expect("64 bytes"));
        }
        let remainder = input.len() % 64;
        if remainder != 0 {
            let start = input.len() - remainder;
            self.buffer[..remainder].copy_from_slice(&input[start..]);
            self.buffered = remainder;
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bit_len = self.bytes.checked_mul(8).expect("SHA-256 input too large");
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut out = [0; 32];
        for (chunk, value) in out.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&value.to_be_bytes());
        }
        out
    }
}

/// Forward-only reader whose parsed bytes and final digest are one byte stream.
pub(super) struct HashingReader<R> {
    inner: R,
    hash: Sha256,
    position: u64,
}

impl<R: Read> HashingReader<R> {
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            hash: Sha256::new(),
            position: 0,
        }
    }

    pub(super) fn finish(mut self) -> io::Result<([u8; 32], u64)> {
        io::copy(&mut self, &mut io::sink())?;
        Ok((self.hash.finish(), self.position))
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(out)?;
        self.hash.update(&out[..read]);
        self.position = self
            .position
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::other("GGUF position overflow"))?;
        Ok(read)
    }
}

impl<R: Read> Seek for HashingReader<R> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let target = match from {
            SeekFrom::Start(value) => Some(value),
            SeekFrom::Current(value) if value >= 0 => self.position.checked_add(value as u64),
            SeekFrom::Current(_) | SeekFrom::End(_) => None,
        }
        .ok_or_else(|| io::Error::other("GGUF hashing reader only supports forward seeks"))?;
        if target < self.position {
            return Err(io::Error::other(
                "GGUF hashing reader cannot seek backwards",
            ));
        }
        let remaining = target - self.position;
        io::copy(&mut self.by_ref().take(remaining), &mut io::sink())?;
        if self.position != target {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated GGUF file",
            ));
        }
        Ok(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::{HashingReader, Sha256};
    use std::io::{Read, Seek, SeekFrom};

    fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn known_vectors_and_streaming_match() {
        let mut empty = Sha256::new();
        empty.update(b"");
        assert_eq!(
            hex(empty.finish()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut abc = Sha256::new();
        for byte in b"abc" {
            abc.update(std::slice::from_ref(byte));
        }
        assert_eq!(
            hex(abc.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let (digest, len) = HashingReader::new(&b"abc"[..]).finish().unwrap();
        assert_eq!(len, 3);
        assert_eq!(
            hex(digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn forward_seek_hashes_skipped_bytes() {
        let mut reader = HashingReader::new(&b"abcdef"[..]);
        let mut first = [0];
        reader.read_exact(&mut first).unwrap();
        assert_eq!(reader.seek(SeekFrom::Start(5)).unwrap(), 5);
        let (digest, len) = reader.finish().unwrap();
        assert_eq!(len, 6);
        assert_eq!(
            hex(digest),
            "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
        );
    }
}
