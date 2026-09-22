// SPDX-License-Identifier: AGPL-3.0-only

//! Device regression only: descriptors and pre-dereference rejection, not GEMM parity.
//! Root-only native invocation: glm53_exl3_private_device --run

use anyhow::{Context, Result, anyhow, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::panic::{AssertUnwindSafe, catch_unwind};

const MODULE: &str = "glm53_exl3_moe_staged_private";
const EXPERTS: usize = 288;
const MAX_PAIRS: usize = 2048 * 8;
const MAX_CHUNKS: usize = MAX_PAIRS.div_ceil(16) + EXPERTS;
const GUARD: usize = 256;
const DEVICE_CAP: usize = 256 * 1024;
const BAD_PAIRS: u32 = 9 * 8;
const POISON: u32 = 0xcdcdcdcd;
const NULL: DevicePtr = DevicePtr::NULL;

#[path = "glm53_exl3_private_device/session.rs"]
mod session;
use session::{Buffer, Session};

#[derive(Clone, Copy)]
struct Buffers {
    counts: Buffer,
    pair: Buffer,
    experts: Buffer,
    starts: Buffer,
    rows: Buffer,
    chunks: Buffer,
    status: Buffer,
    source: Buffer,
}
impl Buffers {
    fn new(s: &mut Session) -> Result<Self> {
        Ok(Self {
            counts: s.alloc((EXPERTS + 1) * 8)?,
            pair: s.alloc(MAX_PAIRS * 4)?,
            experts: s.alloc(MAX_CHUNKS * 4)?,
            starts: s.alloc(MAX_CHUNKS * 4)?,
            rows: s.alloc(MAX_CHUNKS * 4)?,
            chunks: s.alloc(4)?,
            status: s.alloc(4)?,
            source: s.alloc(BAD_PAIRS as usize * 8)?,
        })
    }
}
#[derive(Clone, Copy, Debug)]
enum Consumer {
    Gather,
    Activate,
    Scatter,
}
impl Consumer {
    fn symbol(self) -> &'static str {
        match self {
            Self::Gather => "atlas_glm53_exl3_staged_gather_private",
            Self::Activate => "atlas_glm53_exl3_staged_activate_private",
            Self::Scatter => "atlas_glm53_exl3_staged_scatter_private",
        }
    }
    fn launch(
        self,
        s: &Session,
        k: KernelHandle,
        b: Buffers,
        pair: DevicePtr,
        source: DevicePtr,
    ) -> Result<()> {
        let launch = KernelLaunch::new(&*s.gpu, k)
            .grid([BAD_PAIRS, 1, 1])
            .block([256, 1, 1]);
        match self {
            Self::Gather => launch
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(source)
                .arg_ptr(pair)
                .arg_u32(BAD_PAIRS)
                .arg_ptr(b.status.ptr)
                .launch(s.stream),
            Self::Activate => launch
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(pair)
                .arg_u32(BAD_PAIRS)
                .arg_ptr(b.status.ptr)
                .launch(s.stream),
            Self::Scatter => launch
                .shared_mem(4096)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(NULL)
                .arg_ptr(source)
                .arg_ptr(NULL)
                .arg_ptr(pair)
                .arg_u32(BAD_PAIRS)
                .arg_ptr(b.status.ptr)
                .launch(s.stream),
        }
    }
}
fn builder(
    s: &Session,
    k: KernelHandle,
    b: Buffers,
    counts: DevicePtr,
    pairs: u32,
    cap: u32,
    descriptors: bool,
) -> Result<()> {
    let use_ptr = |buffer: Buffer| if descriptors { buffer.ptr } else { NULL };
    KernelLaunch::new(&*s.gpu, k)
        .grid([1, 1, 1])
        .block([320, 1, 1])
        .arg_ptr(counts)
        .arg_ptr(use_ptr(b.pair))
        .arg_ptr(use_ptr(b.experts))
        .arg_ptr(use_ptr(b.starts))
        .arg_ptr(use_ptr(b.rows))
        .arg_ptr(b.chunks.ptr)
        .arg_ptr(b.status.ptr)
        .arg_u32(EXPERTS as u32)
        .arg_u32(cap)
        .arg_u32(pairs)
        .launch(s.stream)
}
fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn words(raw: &[u8]) -> Result<Vec<u32>> {
    ensure!(raw.len().is_multiple_of(4), "u32 payload width");
    Ok(raw
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}
fn histogram(rows: u32) -> Vec<i64> {
    let active = [0, 3, 17, 42, 91, 128, 200, 250, 287];
    let mut counts = vec![0; EXPERTS + 1];
    for token in 0..rows as usize {
        for (slot, expert) in active.iter().enumerate() {
            if slot != token % active.len() {
                counts[*expert] += 1;
            }
        }
    }
    counts
}
struct Expected {
    experts: Vec<u32>,
    starts: Vec<u32>,
    rows: Vec<u32>,
    pairs: Vec<u32>,
}
fn expected(counts: &[i64], pairs: u32, cap: usize) -> Result<Expected> {
    ensure!(
        counts.len() == EXPERTS + 1 && pairs as usize <= MAX_PAIRS,
        "fixture geometry"
    );
    let mut out = Expected {
        experts: vec![],
        starts: vec![],
        rows: vec![],
        pairs: vec![],
    };
    let mut start = 0u32;
    for (expert, count) in counts[..EXPERTS].iter().enumerate() {
        let count = u32::try_from(*count).context("fixture count range")?;
        let end = start.checked_add(count).context("fixture sum overflow")?;
        ensure!(end <= pairs, "fixture sum exceeds pairs");
        for offset in (0..count).step_by(16) {
            ensure!(out.rows.len() < cap, "fixture chunk capacity");
            out.experts.push(expert as u32);
            out.starts.push(start + offset);
            out.rows.push((count - offset).min(16));
        }
        out.pairs
            .extend(std::iter::repeat_n(expert as u32, count as usize));
        start = end;
    }
    ensure!(start == pairs, "fixture sum differs from pairs");
    Ok(out)
}
fn check_prefix(s: &mut Session, b: Buffer, expected: &[u32], label: &str) -> Result<()> {
    let got = words(&s.read(b)?)?;
    ensure!(expected.len() <= got.len(), "{label} extent");
    for (index, want) in expected.iter().enumerate() {
        ensure!(
            got[index] == *want,
            "{label}[{index}] got={} expected={want}",
            got[index]
        );
    }
    ensure!(
        got[expected.len()..].iter().all(|x| *x == POISON),
        "{label} wrote outside published extent"
    );
    Ok(())
}
fn reset_builder(s: &mut Session, b: Buffers, counts: &[i64], status: u32) -> Result<()> {
    s.write(b.counts, i64_bytes(counts))?;
    s.write(b.status, status.to_le_bytes().to_vec())?;
    for dst in [b.pair, b.experts, b.starts, b.rows, b.chunks] {
        s.poison(dst)?;
    }
    Ok(())
}
fn report(case: &str, extra: serde_json::Value) {
    println!(
        "{}",
        serde_json::json!({"schema":"atlas.glm53.private_device.v1",
        "case":case,"pass":true,"details":extra,"arithmetic_parity_qualified":false})
    );
}
fn valid_builders(s: &mut Session, k: KernelHandle, b: Buffers) -> Result<()> {
    for rows in [9, 106, 1024, 2038, 2048] {
        eprintln!("PRIVATE_DEVICE_BEGIN valid_builder rows={rows}");
        let counts = histogram(rows);
        let want = expected(&counts, rows * 8, MAX_CHUNKS)?;
        reset_builder(s, b, &counts, 0)?;
        builder(s, k, b, b.counts.ptr, rows * 8, MAX_CHUNKS as u32, true)?;
        s.fence()
            .with_context(|| format!("valid builder rows={rows}"))?;
        ensure!(s.word(b.status)? == 0, "valid builder status rows={rows}");
        ensure!(
            s.word(b.chunks)? as usize == want.rows.len(),
            "chunk count rows={rows}"
        );
        check_prefix(s, b.experts, &want.experts, "chunk_expert")?;
        check_prefix(s, b.starts, &want.starts, "chunk_start")?;
        check_prefix(s, b.rows, &want.rows, "chunk_rows")?;
        check_prefix(s, b.pair, &want.pairs, "pair_expert")?;
        ensure!(
            s.read(b.counts)? == i64_bytes(&counts),
            "builder modified counts"
        );
        s.guards()?;
        report(
            "valid_builder",
            serde_json::json!({"rows":rows,"pairs":rows*8,"chunks":want.rows.len()}),
        );
    }
    Ok(())
}
fn rejected_builders(s: &mut Session, k: KernelHandle, b: Buffers) -> Result<()> {
    eprintln!("PRIVATE_DEVICE_BEGIN incoming_status_null_builder");
    s.write(b.status, 1u32.to_le_bytes().to_vec())?;
    s.poison(b.chunks)?;
    builder(s, k, b, NULL, BAD_PAIRS, MAX_CHUNKS as u32, false)?;
    s.fence()
        .context("incoming status1 with NULL builder descriptors/counts")?;
    ensure!(
        s.word(b.chunks)? == 0 && s.word(b.status)? == 1,
        "incoming error not preserved/zero-count"
    );
    report(
        "incoming_status_null_builder",
        serde_json::json!({"status":1,"chunks":0}),
    );
    for kind in [
        "negative",
        "huge",
        "short_sum",
        "long_sum",
        "zero_cap",
        "overflow_cap",
    ] {
        eprintln!("PRIVATE_DEVICE_BEGIN invalid_builder kind={kind}");
        let mut counts = histogram(9);
        match kind {
            "negative" => counts[0] = -1,
            "huge" => counts[0] = i64::MAX,
            "short_sum" => counts[0] -= 1,
            "long_sum" => counts[0] += 1,
            _ => {}
        }
        let cap = match kind {
            "zero_cap" => 0,
            "overflow_cap" => 1,
            _ => MAX_CHUNKS as u32,
        };
        reset_builder(s, b, &counts, 0)?;
        builder(s, k, b, b.counts.ptr, BAD_PAIRS, cap, true)?;
        s.fence()
            .with_context(|| format!("invalid builder {kind}"))?;
        let status = s.word(b.status)?;
        ensure!(
            status != 0 && s.word(b.chunks)? == 0,
            "invalid builder accepted {kind}"
        );
        s.guards()?;
        report(
            "invalid_builder",
            serde_json::json!({"kind":kind,"status":status,"chunks":0}),
        );
    }
    Ok(())
}
fn rejected_consumers(
    s: &mut Session,
    kernels: &[(Consumer, KernelHandle)],
    b: Buffers,
) -> Result<()> {
    for &(kind, kernel) in kernels {
        for fault in [
            "incoming_status",
            "expert",
            "source_high",
            "source_negative",
        ] {
            if matches!(kind, Consumer::Activate) && fault.starts_with("source_") {
                continue;
            }
            eprintln!("PRIVATE_DEVICE_BEGIN consumer={kind:?} fault={fault}");
            let mut experts = vec![0u32; MAX_PAIRS];
            if fault == "expert" {
                experts.fill(EXPERTS as u32);
            }
            let sources: Vec<i64> = (0..BAD_PAIRS)
                .map(|pair| match fault {
                    "source_high" => i64::from(BAD_PAIRS),
                    "source_negative" => -1,
                    _ => i64::from(pair),
                })
                .collect();
            s.write(b.pair, u32_bytes(&experts))?;
            s.write(b.source, i64_bytes(&sources))?;
            // Every CTA independently has an invalid descriptor. No valid CTA
            // is allowed to race an error and enter arithmetic with NULL data.
            for repeat in 0..8 {
                let incoming = if fault == "incoming_status" { 1u32 } else { 0 };
                s.write(b.status, incoming.to_le_bytes().to_vec())?;
                let (pair, source) = if incoming == 1 {
                    (NULL, NULL)
                } else {
                    (b.pair.ptr, b.source.ptr)
                };
                kind.launch(s, kernel, b, pair, source)?;
                s.fence()
                    .with_context(|| format!("consumer {kind:?}/{fault}/repeat{repeat}"))?;
                let status = s.word(b.status)?;
                ensure!(status != 0, "consumer did not reject {kind:?}/{fault}");
                if incoming == 1 {
                    ensure!(status == 1, "incoming status changed");
                }
            }
            ensure!(
                s.read(b.pair)? == u32_bytes(&experts),
                "consumer modified expert descriptors"
            );
            ensure!(
                s.read(b.source)? == i64_bytes(&sources),
                "consumer modified source descriptors"
            );
            s.guards()?;
            report(
                "invalid_consumer",
                serde_json::json!({"consumer":format!("{kind:?}"),"fault":fault,"repeats":8}),
            );
        }
    }
    Ok(())
}
fn run(s: &mut Session) -> Result<()> {
    let build = s
        .gpu
        .kernel(MODULE, "atlas_glm53_exl3_build_chunks_private")?;
    let mut consumers = Vec::with_capacity(3);
    for kind in [Consumer::Gather, Consumer::Activate, Consumer::Scatter] {
        consumers.push((kind, s.gpu.kernel(MODULE, kind.symbol())?));
    }
    let b = Buffers::new(s)?;
    valid_builders(s, build, b)?;
    rejected_builders(s, build, b)?;
    rejected_consumers(s, &consumers, b)?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args == [std::ffi::OsString::from("--run")],
        "usage: glm53_exl3_private_device --run"
    );
    let mut session = Session::new()?;
    let result = catch_unwind(AssertUnwindSafe(|| run(&mut session)))
        .unwrap_or_else(|_| Err(anyhow!("private device probe panicked")));
    let cleanup = session.close();
    match (result, cleanup) {
        (Ok(()), Ok(())) => {
            report(
                "COMPLETE",
                serde_json::json!({"valid_builders":5,"invalid_builders":7,
                "invalid_consumers":10,"device_cap_bytes":DEVICE_CAP,"model_loaded":false}),
            );
            Ok(())
        }
        (primary, cleanup) => Err(anyhow!(
            "private device probe failed: primary={primary:?}; cleanup={cleanup:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_histograms_are_real_top8_counts_with_complete_pair_coverage() {
        for rows in [9, 106, 1024, 2038, 2048] {
            let counts = histogram(rows);
            assert_eq!(counts.iter().sum::<i64>(), i64::from(rows * 8));
            assert!(counts.iter().all(|x| (0..=i64::from(rows)).contains(x)));
            let e = expected(&counts, rows * 8, MAX_CHUNKS).unwrap();
            assert_eq!(e.pairs.len(), rows as usize * 8);
            assert_eq!(e.rows.iter().sum::<u32>(), rows * 8);
            assert!(e.rows.iter().all(|x| (1..=16).contains(x)));
        }
    }
    #[test]
    fn m9_expected_descriptors_have_explicit_known_boundaries() {
        let e = expected(&histogram(9), 72, MAX_CHUNKS).unwrap();
        assert_eq!(e.experts, [0, 3, 17, 42, 91, 128, 200, 250, 287]);
        assert_eq!(e.starts, [0, 8, 16, 24, 32, 40, 48, 56, 64]);
        assert_eq!(e.rows, [8; 9]);
        assert_eq!(&e.pairs[64..], &[287; 8]);
    }
    #[test]
    fn expected_fixture_rejects_negative_sum_drift_and_insufficient_capacity() {
        let mut counts = histogram(9);
        assert!(expected(&counts, 72, 1).is_err());
        assert!(expected(&counts, 71, MAX_CHUNKS).is_err());
        assert!(expected(&counts, 73, MAX_CHUNKS).is_err());
        counts[0] = -1;
        assert!(expected(&counts, 72, MAX_CHUNKS).is_err());
    }
}
