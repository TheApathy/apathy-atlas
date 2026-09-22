// engram I/O floor probe: random 264-byte row gathers from the two 95 GB engram shards.
// Pure NVMe/host. No GPU. Mirrors what one decode step / one prefill chunk actually needs.
use std::os::unix::io::{AsRawFd, RawFd};
use std::fs::File;
use std::time::Instant;
use std::sync::Arc;

extern "C" {
    fn pread(fd: i32, buf: *mut u8, n: usize, off: i64) -> isize;
    fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
}
const FADV_RANDOM: i32 = 1;

// splitmix64: deterministic, and we advance the seed across iterations so rows stay cold.
fn sm64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

struct Shard { fd: RawFd, w_off: i64, s_off: i64, n_rows: u64 }

fn gather(shards: &[Arc<Shard>], ids: &[Vec<u64>], nthreads: usize, cached_scale: bool) -> f64 {
    // ids[s][i] = row id for shard s. Work is split over threads by a flat index over (shard, row).
    let total: usize = ids.iter().map(|v| v.len()).sum();
    let t0 = Instant::now();
    let per = (total + nthreads - 1) / nthreads;
    std::thread::scope(|sc| {
        for t in 0..nthreads {
            let lo = per * t;
            let hi = ((per * (t + 1)).min(total)).max(lo);
            let shards = &shards; let ids = &ids;
            sc.spawn(move || {
                let mut buf = [0u8; 264];
                for k in lo..hi {
                    // map flat k -> (shard, index)
                    let mut k2 = k; let mut s = 0usize;
                    while k2 >= ids[s].len() { k2 -= ids[s].len(); s += 1; }
                    let r = ids[s][k2] as i64;
                    let sh = &shards[s];
                    unsafe {
                        let a = pread(sh.fd, buf.as_mut_ptr(), 256, sh.w_off + r * 256);
                        if a != 256 { panic!("short weight read"); }
                        if !cached_scale {
                            let b = pread(sh.fd, buf.as_mut_ptr().add(256), 8, sh.s_off + r * 8);
                            if b != 8 { panic!("short scale read"); }
                        }
                    }
                    std::hint::black_box(&buf);
                }
            });
        }
    });
    t0.elapsed().as_secs_f64()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
    let files = [
        (format!("{dir}/model-00047-of-00048.safetensors"), 664i64, 98305579672i64, 384006168u64),
        (format!("{dir}/model-00048-of-00048.safetensors"), 672i64, 98308271264i64, 384016682u64),
    ];
    let mut shards = Vec::new();
    let mut _keep = Vec::new();
    for (p, w, s, n) in files.iter() {
        let f = File::open(p).unwrap();
        let fd = f.as_raw_fd();
        unsafe { posix_fadvise(fd, 0, 0, FADV_RANDOM); }
        shards.push(Arc::new(Shard { fd, w_off: *w, s_off: *s, n_rows: *n }));
        _keep.push(f);
    }

    let tokens: usize = a.get(1).map(|x| x.parse().unwrap()).unwrap_or(1);
    let iters: usize  = a.get(2).map(|x| x.parse().unwrap()).unwrap_or(20);
    let cached_scale = a.get(3).map(|x| x == "cached").unwrap_or(false);
    let thread_list: Vec<usize> = a.get(4).map(|x| x.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or(vec![1, 8, 32, 64, 128, 256]);

    // 24 rows per token per layer; one shard per engram layer.
    let rows_per_shard = tokens * 24;
    let mut seed: u64 = a.get(5).map(|x| x.parse().unwrap()).unwrap_or(0xDEADBEEF);

    println!("# tokens={tokens} rows/step={} ({} per shard x2) iters={iters} scale={}",
        rows_per_shard * 2, rows_per_shard, if cached_scale {"CACHED (1 pread/row)"} else {"on-disk (2 preads/row)"});
    println!("{:>7} {:>11} {:>11} {:>12} {:>12}", "thr", "ms/step", "us/row", "rows/s", "tok/s_cap");
    for &nt in &thread_list {
        let nt = nt.min(rows_per_shard * 2).max(1);
        let mut times = Vec::new();
        for _ in 0..iters {
            // fresh ids every iteration: keeps the pages cold
            let ids: Vec<Vec<u64>> = shards.iter()
                .map(|sh| (0..rows_per_shard).map(|_| sm64(&mut seed) % sh.n_rows).collect())
                .collect();
            times.push(gather(&shards, &ids, nt, cached_scale));
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = times[times.len() / 2];
        let rows = (rows_per_shard * 2) as f64;
        println!("{:>7} {:>11.3} {:>11.2} {:>12.0} {:>12.1}",
            nt, med * 1e3, med * 1e6 / rows, rows / med, tokens as f64 / med);
    }
}
