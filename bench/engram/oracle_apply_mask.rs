// Validates `apply_dead_mask` (spliced in from dead_heads.rs): applying the
// REAL dead-head mask to the captured PRE-mask rows must reproduce the
// captured POST-mask rows exactly. Zero arithmetic on this path (masked_fill
// just zeroes slices), so this is a bit-exactness question, not a tolerance one.
//
// Build:
//   cat crates/spark-model/src/layers/deepseek_v41_engram/dead_heads.rs \
//       bench/engram/oracle_apply_mask.rs > /tmp/oracle_apply_mask_combined.rs
//   rustc -O -o /tmp/oracle_apply_mask /tmp/oracle_apply_mask_combined.rs
//
// Usage, from a directory holding dead.bin (bool, [T,24]) and premask.bin
// (f32, [T,24,256]):
//   oracle_apply_mask <T> good        # candidate = real apply_dead_mask
//   oracle_apply_mask <T> neg-noop    # candidate = premask left untouched

use std::fs;

fn read_bool(path: &str) -> Vec<bool> {
    fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}")).into_iter().map(|b| b != 0).collect()
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(path: &str, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, &bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).expect("usage: oracle_apply_mask <T> <mode>").parse().unwrap();
    let mode = args.get(2).map(String::as_str).unwrap_or("good");
    const ROW_DIM: usize = 256;

    let dead = read_bool("dead.bin");
    assert_eq!(dead.len(), t * N_HEAD_COLS, "dead.bin is not [T={t}, {N_HEAD_COLS}]");
    let mut rows = read_f32("premask.bin");
    assert_eq!(rows.len(), t * N_HEAD_COLS * ROW_DIM, "premask.bin is not [T={t}, {N_HEAD_COLS}, {ROW_DIM}]");

    match mode {
        "good" => apply_dead_mask(&mut rows, &dead, t, ROW_DIM),
        "neg-noop" => { /* deliberately skip masking */ }
        other => panic!("unknown mode {other}"),
    }

    write_f32("cand_rows.bin", &rows);
    let zeroed = dead.iter().filter(|&&d| d).count();
    println!("mode={mode} T={t} dead_cols={zeroed} -> cand_rows.bin");
}
