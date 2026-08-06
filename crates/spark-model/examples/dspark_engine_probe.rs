// SPDX-License-Identifier: AGPL-3.0-only

//! Engine-vs-reference diff for the DSpark propose forward.
//!
//! Replays the same ATLAS_DSPARK_DUMP captures the Python oracle consumed
//! (`bench/deepseek-v4/dspark_probe/probe.py`) through the native
//! `DsparkDraftHead`, and scores both (a) agreement with the reference
//! drafts (`dspark_ref_drafts.bin`, written by probe.py) and (b) acceptance
//! against the actually-generated continuation. (a) validates the port;
//! (b) is the number that matters (reference: 3.81 tok/step ungated).
//!
//! Usage:
//!   cargo run --release -p spark-model --example dspark_engine_probe -- \
//!     [dump.bin] [ref_drafts.bin]

use anyhow::{Context, Result, bail};
use spark_model::layers::dspark_head::DsparkDraftHead;
use spark_model::weight_loader::deepseek_v4::dspark;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightLoader;
use std::io::{Read, Seek, SeekFrom};

const DRAFTER_DIR: &str = "/home/flocka/models/DeepSeek-V4-Flash-0731-drafter";
const TARGET_DIR: &str = "/home/flocka/models/DeepSeek-V4-Flash-162B";

struct Rec {
    kind: u32,
    start: u32,
    n: u32,
    token: u32,
    /// [nl, n, h] BF16 raw bytes.
    data: Vec<u8>,
}

fn read_dump(path: &str) -> Result<(Vec<Rec>, usize, usize)> {
    let mut f = std::fs::File::open(path)?;
    let mut recs = Vec::new();
    let (mut h_sz, mut nl_sz) = (0usize, 0usize);
    loop {
        let mut hdr = [0u8; 28];
        match f.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(_) => break,
        }
        let w = |i: usize| u32::from_le_bytes(hdr[i * 4..i * 4 + 4].try_into().unwrap());
        anyhow::ensure!(w(0) == 0x4453504B, "bad magic");
        let (kind, start, n, h, nl, token) = (w(1), w(2), w(3), w(4), w(5), w(6));
        h_sz = h as usize;
        nl_sz = nl as usize;
        let mut data = vec![0u8; nl_sz * n as usize * h_sz * 2];
        f.read_exact(&mut data)?;
        recs.push(Rec {
            kind,
            start,
            n,
            token,
            data,
        });
    }
    Ok((recs, h_sz, nl_sz))
}

/// Selective safetensors read: one tensor by name from an indexed multi-shard
/// checkpoint, uploaded to the GPU verbatim (caller knows the dtype).
fn load_tensor(dir: &str, name: &str, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{dir}/model.safetensors.index.json"
    ))?)?;
    let shard = idx["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} not in index"))?;
    let mut f = std::fs::File::open(format!("{dir}/{shard}"))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let hlen = u64::from_le_bytes(len8);
    let mut hjson = vec![0u8; hlen as usize];
    f.read_exact(&mut hjson)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hjson)?;
    let ent = &hdr[name];
    let offs = ent["data_offsets"]
        .as_array()
        .with_context(|| format!("{name}: no data_offsets"))?;
    let (b0, b1) = (offs[0].as_u64().unwrap(), offs[1].as_u64().unwrap());
    let dtype = ent["dtype"].as_str().unwrap_or("?");
    if dtype != "BF16" {
        bail!("{name}: expected BF16, got {dtype}");
    }
    f.seek(SeekFrom::Start(8 + hlen + b0))?;
    let mut bytes = vec![0u8; (b1 - b0) as usize];
    f.read_exact(&mut bytes)?;
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    // The drafter loader transposes its MoE to the unified `_t` layout (the
    // production V4 serve config — serve_single.sh exports this =1). The
    // MoeLayer's dispatch flag reads the env at construction; without it the
    // batchN path launches the NON-transposed kernels over freed originals.
    // SAFETY: single-threaded example, set before any reader.
    unsafe { std::env::set_var("ATLAS_UNIFIED_MOE_LAYOUT", "1") };
    let mut args = std::env::args().skip(1);
    let dump = args
        .next()
        .unwrap_or_else(|| "/home/flocka/deepseek-flash/dspark_dump.bin".into());
    let ref_path = args
        .next()
        .unwrap_or_else(|| "/home/flocka/deepseek-flash/dspark_ref_drafts.bin".into());

    let config_json =
        std::fs::read_to_string(std::path::Path::new(TARGET_DIR).join("config.json"))?;
    let target_config = atlas_core::config::parse_config(&config_json)?;

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();

    // ── drafter + shared embed/head ──
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    let store = loader.load(std::path::Path::new(DRAFTER_DIR), gpu, 0)?;
    let module = dspark::load_dspark_drafter(
        &store,
        &target_config,
        dspark::DsparkParams::V4_FLASH_0731(),
        gpu,
    )?;
    let block = module.params.block_size;
    let embed = spark_model::weight_map::DenseWeight {
        weight: load_tensor(TARGET_DIR, "embed.weight", gpu)?,
    };
    let head_w = spark_model::weight_map::DenseWeight {
        weight: load_tensor(TARGET_DIR, "head.weight", gpu)?,
    };
    println!("drafter + shared embed/head loaded");

    let mut drafter_config = target_config.clone();
    drafter_config.num_experts = 256;
    let head = DsparkDraftHead::new(
        module,
        &target_config,
        embed,
        Some(head_w),
        None,
        256,
        4096,
        gpu,
    )?;

    let buffers = spark_runtime::buffers::BufferArena::new(&drafter_config, 64, 4096, 16, gpu)?;
    let ctx = spark_model::layer::ForwardContext {
        buffers: &buffers,
        gpu,
        config: &drafter_config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
    };

    // ── reference drafts: (seq, pos) -> [block] tokens ──
    let ref_bytes = std::fs::read(&ref_path)?;
    let rec_sz = 8 + block * 4 + block * 4;
    let mut refs = std::collections::HashMap::new();
    for c in ref_bytes.chunks_exact(rec_sz) {
        let seq = u32::from_le_bytes(c[0..4].try_into()?);
        let pos = u32::from_le_bytes(c[4..8].try_into()?);
        let toks: Vec<u32> = (0..block)
            .map(|j| u32::from_le_bytes(c[8 + j * 4..12 + j * 4].try_into().unwrap()))
            .collect();
        refs.insert((seq, pos), toks);
    }
    println!("reference drafts: {} propose points", refs.len());

    // ── replay ──
    let (recs, h, nl) = read_dump(&dump)?;
    anyhow::ensure!(nl == 3 && h == target_config.hidden_size, "dump geometry");
    let cap_dev = gpu.alloc(3 * 4096 * h * 2)?; // one record's captures
    let mut seqs: Vec<Vec<&Rec>> = Vec::new();
    for r in &recs {
        if r.kind == 0 && r.start == 0 {
            seqs.push(Vec::new());
        }
        if let Some(s) = seqs.last_mut() {
            s.push(r);
        }
    }

    let mut agree = vec![0usize; block];
    let mut agree_total = vec![0usize; block];
    let mut pos_match = vec![0usize; block];
    let mut pos_total = vec![0usize; block];
    let mut chain_hist = vec![0usize; block + 1];
    let mut n_props = 0usize;
    let t0 = std::time::Instant::now();
    let mut propose_ms = 0.0f64;

    for (si, seq) in seqs.iter().enumerate() {
        let dec: Vec<&&Rec> = seq.iter().filter(|r| r.kind == 1).collect();
        if dec.is_empty() {
            continue;
        }
        let tok_at: std::collections::HashMap<u32, u32> =
            dec.iter().map(|r| (r.start, r.token)).collect();

        // Seed the ring from every prefill position, in order.
        for r in seq.iter().filter(|r| r.kind == 0) {
            gpu.copy_h2d(&r.data, cap_dev)?;
            let n = r.n as usize;
            for j in 0..n {
                let caps = [
                    cap_dev.offset(j * h * 2),
                    cap_dev.offset((n + j) * h * 2),
                    cap_dev.offset((2 * n + j) * h * 2),
                ];
                head.seed_position(gpu, caps, r.start as usize + j, stream)?;
            }
        }
        gpu.synchronize(stream)?;

        for r in &dec {
            let p = r.start;
            let Some(&committed) = tok_at.get(&(p + 1)) else {
                continue;
            };
            gpu.copy_h2d(&r.data, cap_dev)?;
            let caps = [
                cap_dev.offset(0),
                cap_dev.offset(h * 2),
                cap_dev.offset(2 * h * 2),
            ];
            let tp = std::time::Instant::now();
            let (drafts, _confs, _top2) =
                head.propose_block(gpu, &ctx, caps, committed, p as usize, stream)?;
            propose_ms += tp.elapsed().as_secs_f64() * 1e3;
            n_props += 1;
            if std::env::var("ATLAS_DSPARK_DRAFT_LOG").as_deref() == Ok("1") {
                println!("ENGDRAFT pos={p} committed={committed} drafts={drafts:?}");
            }

            if let Some(rd) = refs.get(&(si as u32, p)) {
                for j in 0..block {
                    agree_total[j] += 1;
                    if drafts[j] == rd[j] {
                        agree[j] += 1;
                    }
                }
            }
            let mut chain = 0usize;
            for j in 0..block {
                let Some(&actual) = tok_at.get(&(p + 2 + j as u32)) else {
                    break;
                };
                pos_total[j] += 1;
                if drafts[j] == actual {
                    pos_match[j] += 1;
                    if chain == j {
                        chain = j + 1;
                    }
                }
            }
            chain_hist[chain] += 1;
        }
    }

    println!(
        "\n== DSpark ENGINE probe ({n_props} propose points, {:.1}s total, {:.1} ms/propose) ==",
        t0.elapsed().as_secs_f64(),
        propose_ms / n_props.max(1) as f64
    );
    println!("-- engine vs Python reference (port validation) --");
    for j in 0..block {
        if agree_total[j] > 0 {
            println!(
                "  draft[{j}] agree: {}/{} = {:.1}%",
                agree[j],
                agree_total[j],
                agree[j] as f64 / agree_total[j] as f64 * 100.0
            );
        }
    }
    println!("-- engine vs actual continuation (acceptance) --");
    for j in 0..block {
        if pos_total[j] > 0 {
            println!(
                "  draft[{j}] match: {}/{} = {:.1}%",
                pos_match[j],
                pos_total[j],
                pos_match[j] as f64 / pos_total[j] as f64 * 100.0
            );
        }
    }
    let mean: f64 = chain_hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum::<f64>()
        / n_props.max(1) as f64;
    println!("  chain hist: {chain_hist:?}");
    println!(
        "  mean accepted chain = {mean:.2}  -> tok/step = {:.2}",
        mean + 1.0
    );
    Ok(())
}
