// SPDX-License-Identifier: AGPL-3.0-only

//! Cold/hybrid sparse PLE I/O benchmark.

use anyhow::{Context, Result, ensure};
use rand::{Rng, SeedableRng, rngs::StdRng};
use spark_storage::ple_offload::PleOffloadReader;
use std::path::Path;
use std::time::Instant;

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() >= 2,
        "usage: ple-offload-bench MANIFEST [iterations=1000] [cache_mb=0]"
    );
    let iterations = args.get(2).map_or(Ok(1000), |value| value.parse())?;
    let cache_mb: usize = args.get(3).map_or(Ok(0), |value| value.parse())?;
    let mut reader = PleOffloadReader::open(
        Path::new(&args[1]),
        16,
        cache_mb.saturating_mul(1024 * 1024),
    )
    .context("open PLE offload reader")?;
    let mut rng = StdRng::seed_from_u64(0x38F1_A5A5);
    let mut samples = Vec::with_capacity(iterations);
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let selections: Vec<(usize, usize)> = (0..16)
            .map(|_| (rng.gen_range(0..128), rng.gen_range(0..2_500_012)))
            .collect();
        let start = Instant::now();
        let rows = reader.read_rows(&selections)?;
        samples.push(start.elapsed().as_micros());
        for row in rows {
            checksum = checksum.wrapping_add(row.record[0] as u64);
        }
    }
    samples.sort_unstable();
    let mean = samples.iter().sum::<u128>() as f64 / samples.len() as f64;
    println!(
        "PLE sparse batch: iterations={iterations} rows/batch=16 cache={cache_mb}MiB \
         mean={mean:.1}us p50={}us p95={}us p99={}us ceiling_p50={:.0} tok/s checksum={checksum}",
        percentile(&samples, 50),
        percentile(&samples, 95),
        percentile(&samples, 99),
        1_000_000.0 / percentile(&samples, 50).max(1) as f64,
    );
    Ok(())
}
