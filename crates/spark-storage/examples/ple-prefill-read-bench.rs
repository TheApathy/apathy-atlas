// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-sized PLE read: windowed vs streamed, same selections, byte-compared.
//! A 2048-token prefill selects 16 rows per token = 32,768 rows.

use anyhow::{Context, Result, ensure};
use rand::{Rng, SeedableRng, rngs::StdRng};
use spark_storage::ple_offload::PleOffloadReader;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 2, "usage: ple-prefill-read-bench MANIFEST [rows=32768] [reps=4] [queue_depth=256]");
    let rows: usize = args.get(2).map_or(Ok(32_768), |value| value.parse())?;
    let reps: usize = args.get(3).map_or(Ok(4), |value| value.parse())?;
    let depth: usize = args.get(4).map_or(Ok(256), |value| value.parse())?;
    let mut reader =
        PleOffloadReader::open(Path::new(&args[1]), depth, 0).context("open PLE offload reader")?;
    for rep in 0..reps {
        // Fresh random rows per rep and per arm so neither arm reads pages the
        // other just pulled (O_DIRECT, but the device has its own cache).
        let mut rng = StdRng::seed_from_u64(0x9E37_79B9 + rep as u64);
        let a: Vec<(usize, usize)> =
            (0..rows).map(|_| (rng.gen_range(0..128), rng.gen_range(0..2_500_012))).collect();
        let b: Vec<(usize, usize)> =
            (0..rows).map(|_| (rng.gen_range(0..128), rng.gen_range(0..2_500_012))).collect();
        let (first, second) = if rep % 2 == 0 { (&a, &b) } else { (&b, &a) };
        let t = Instant::now();
        let windowed = reader.read_rows_windowed(first)?;
        let windowed_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let streamed = reader.read_rows_streamed(second)?;
        let streamed_ms = t.elapsed().as_secs_f64() * 1e3;
        // Byte check on identical selections (not timed).
        let check_w = reader.read_rows_windowed(&a[..4096])?;
        let check_s = reader.read_rows_streamed(&a[..4096])?;
        let identical = check_w
            .iter()
            .zip(&check_s)
            .all(|(x, y)| x.record == y.record && x.scale2.to_bits() == y.scale2.to_bits());
        ensure!(windowed.len() == rows && streamed.len() == rows, "row count mismatch");
        println!(
            "rep {rep}: rows={rows} windowed={windowed_ms:.1} ms streamed={streamed_ms:.1} ms identical={identical}"
        );
        ensure!(identical, "streamed read returned different bytes");
    }
    Ok(())
}
