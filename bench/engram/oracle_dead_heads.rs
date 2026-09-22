// Validates the REAL `engram_dead_heads` (spliced in from dead_heads.rs) against
// an oracle capture's `engram_dead` taps.
//
// Build (the crate source has zero external deps, so this compiles standalone):
//
//   cat crates/spark-model/src/layers/deepseek_v41_engram/dead_heads.rs \
//       bench/engram/oracle_dead_heads.rs > /tmp/oracle_dead_heads_combined.rs
//   rustc -O -o /tmp/oracle_dead_heads /tmp/oracle_dead_heads_combined.rs
//
// Usage, from a directory already holding ids_NNN.bin (from dump_chunk_ids.py):
//   oracle_dead_heads good        # candidate = the real function
//   oracle_dead_heads neg-shift   # candidate = deliberately off-by-one, must FAIL

use std::fs;

fn read_u32(path: &str) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "good".to_string());
    let mut occ = 0usize;
    loop {
        let ids_path = format!("ids_{occ:03}.bin");
        if !std::path::Path::new(&ids_path).exists() {
            break;
        }
        let ids = read_u32(&ids_path);
        let real = engram_dead_heads(&ids);

        let out_bytes: Vec<u8> = if mode == "good" {
            real.iter().map(|&b| u8::from(b)).collect()
        } else if mode == "neg-shift" {
            // Deliberately wrong: OR the mask with itself shifted forward one
            // position. Must NOT reproduce the reference wherever the image
            // span does not sit at the very start of the chunk.
            let cols = N_HEAD_COLS;
            let t = ids.len();
            let mut wrong = real.clone();
            for p in (1..t).rev() {
                for c in 0..cols {
                    wrong[p * cols + c] = real[p * cols + c] || real[(p - 1) * cols + c];
                }
            }
            wrong.iter().map(|&b| u8::from(b)).collect()
        } else {
            panic!("unknown mode {mode}");
        };

        let out = format!("cand_dead_{mode}_{occ:03}.bin");
        fs::write(&out, &out_bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
        println!(
            "occ={occ:03} tokens={} True={} -> {out}",
            ids.len(),
            real.iter().filter(|&&d| d).count()
        );
        occ += 1;
    }
    assert!(occ > 0, "no ids_NNN.bin files found in the working directory");
}
