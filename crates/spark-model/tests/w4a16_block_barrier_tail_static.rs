// SPDX-License-Identifier: AGPL-3.0-only

use std::process::Command;
const KERNEL: &str = include_str!("../../../kernels/gb10/common/w4a16_gemv.cu");
const WRAPPERS: &str = include_str!("../src/layers/ops/quant_dispatch.rs");
const KERNEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/common/w4a16_gemv.cu"
);
const RELEASED_SHA256: &str = "a3d5edb03501711d9dc98fdb62143902e259c6b3187c5d9eeacea59d09696374";
const MACRO_FNV1A: u64 = 0xdf190fc0bb6d0de5;
type TailKernel = (&'static str, usize, usize, &'static [usize]);
const SAFE: &[TailKernel] = &[
    ("w4a16_gemv_sw", 8, 410, &[]),
    ("w4a16_gemv_v2", 1, 1531, &[1540, 1555]),
];
const UNSAFE: &[TailKernel] = &[
    ("w4a16_gemv_batch2", 4, 546, &[559, 620]),
    ("w4a16_gemv_qg", 4, 659, &[668, 705]),
    ("w4a16_gemv_qkvz", 4, 758, &[767, 804]),
    ("w4a16_gemv_qg_batch2", 4, 859, &[871, 921]),
    ("w4a16_gemv_dual_batch2", 4, 978, &[990, 1038]),
    ("w4a16_gemv_batch3", 4, 1076, &[1090, 1151]),
    ("w4a16_gemv_qg_batch3", 4, 1190, &[1204, 1264]),
    ("w4a16_gemv_dual_batch3", 4, 1322, &[1336, 1393]),
    ("w4a16_gemv_v1", 2, 1474, &[1483, 1500]),
    ("w4a16_gemv_v3", 8, 1586, &[1594]),
    ("w4a16_gemv_batch3_logits", 8, 1654, &[1667]),
    ("w4a16_gemv_v4", 2, 1755, &[1764, 1779]),
    ("w4a16_gemv_qg_batch3_strided", 4, 1820, &[1834, 1894]),
    ("w4a16_gemv_dual_batch3_strided", 4, 1943, &[1957, 2014]),
    ("w4a16_gemv_dual_batch3_tuned", 4, 2107, &[2122, 2192]),
];
const WRAPPER_GROUPS: &[(&str, usize)] = &[
    ("w4a16_gemv_batch2", 4),
    ("w4a16_gemv_batch3", 4),
    ("w4a16_gemv_batch3_logits", 8),
    ("w4a16_gemv_qg", 4),
    ("w4a16_gemv_qkvz", 4),
    ("w4a16_gemv_qg_batch2", 4),
    ("w4a16_gemv_qg_batch3", 4),
    ("w4a16_gemv_dual_batch2", 4),
    ("w4a16_gemv_dual_batch3", 4),
    ("w4a16_gemv_dual_batch3_tuned", 4),
    ("w4a16_gemv_qg_batch3_strided", 4),
    ("w4a16_gemv_dual_batch3_strided", 4),
];

struct Body<'a> {
    text: &'a str,
    offset: usize,
}
fn body_after<'a>(source: &'a str, needle: &str) -> Body<'a> {
    let start = source.find(needle).expect("missing braced body");
    let open = start + source[start..].find('{').expect("missing opening brace");
    let mut depth = 0usize;
    for (i, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Body {
                        text: &source[open + 1..open + i],
                        offset: open + 1,
                    };
                }
            }
            _ => {}
        }
    }
    panic!("missing closing brace after {needle}")
}

fn kernel_body(name: &str) -> Body<'static> {
    body_after(KERNEL, &format!("void {name}("))
}
fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}
fn line(offset: usize) -> usize {
    KERNEL.as_bytes()[..offset]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}
fn dominates_barrier(body: &str) -> bool {
    let body = compact(body);
    body.find("if(n>=N)return;").is_some_and(|tail| {
        body.find("__syncthreads();")
            .is_some_and(|barrier| tail < barrier)
    })
}
fn sha256() -> String {
    let output = Command::new("sha256sum").arg(KERNEL_PATH).output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}
fn macro_definition() -> &'static str {
    let start = KERNEL.find("#define W4A16_INNER_FMA").unwrap();
    let len = KERNEL[start..].find("\n\n// ── Variant 1").unwrap();
    &KERNEL[start..start + len]
}
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}
fn assert_released_anchors(k: TailKernel) {
    let (name, _, return_line, barriers) = k;
    let body = kernel_body(name);
    let ret = body.text.find("if (n >= N) return;").unwrap();
    assert_eq!(line(body.offset + ret), return_line, "{name} return");
    let actual: Vec<_> = body
        .text
        .match_indices("__syncthreads();")
        .map(|(i, _)| line(body.offset + i))
        .collect();
    assert_eq!(actual, barriers, "{name} barriers");
}
fn future_errors(k: TailKernel) -> Vec<&'static str> {
    let (name, group, _, barriers) = k;
    let body = kernel_body(name);
    let compact_body = compact(body.text);
    let mut errors = Vec::new();
    if dominates_barrier(body.text) {
        errors.push("barrier-dominating tail return");
    }
    if !compact_body.contains("constboolvalid=n<N;") {
        errors.push("missing explicit valid predicate");
        return errors;
    }
    if !compact_body.contains("=0.0f;") || !compact_body.contains("if(valid){") {
        errors.push("missing zero participation");
    }
    if compact_body.matches("__syncthreads();").count() != barriers.len() {
        errors.push("barrier count changed");
    }
    let valid = body_after(body.text, "if (valid) {");
    if matches!(name, "w4a16_gemv_v1" | "w4a16_gemv_v3" | "w4a16_gemv_v4") {
        let call = "W4A16_INNER_FMA(acc, lane, TPO)";
        if body.text.matches(call).count() != 1 || valid.text.matches(call).count() != 1 {
            errors.push("unpredicated macro FMA");
        }
    } else {
        for load in ["B_packed +", "B_scale["] {
            if !valid.text.contains(load)
                || body.text.matches(load).count() != valid.text.matches(load).count()
            {
                errors.push("unpredicated n-dependent load");
            }
        }
    }
    if valid.text.contains("__syncthreads();") {
        errors.push("conditional block barrier");
    }
    if valid.text.contains("smem[") || valid.text.contains("s_lut[threadIdx.x] =") {
        errors.push("conditional shared write");
    }
    let store = "if (valid && lane == 0) {";
    let store_ok = body.text.contains(store)
        && body.text.matches("__float2bfloat16").count()
            == body_after(body.text, store)
                .text
                .matches("__float2bfloat16")
                .count();
    if !store_ok {
        errors.push("unpredicated final store");
    }
    if group < 8 {
        let producer = "if (warp_lane == 0) {";
        if !body.text.contains(producer) || !body_after(body.text, producer).text.contains("smem[")
        {
            errors.push("conditional shared producer");
        }
    }
    errors
}

#[test]
fn released_hash_and_classification_are_exact_until_the_fix_is_complete() {
    assert_eq!(fnv1a(macro_definition()), MACRO_FNV1A, "FMA macro drift");
    let offenders = UNSAFE
        .iter()
        .filter(|k| dominates_barrier(kernel_body(k.0).text))
        .count();
    if offenders == 0 {
        return;
    }
    assert_eq!(sha256(), RELEASED_SHA256, "partial fix or source drift");
    assert_eq!(offenders, UNSAFE.len());
    assert_eq!(KERNEL.matches("if (n >= N) return;").count(), 17);
    SAFE.iter()
        .chain(UNSAFE)
        .copied()
        .for_each(assert_released_anchors);
}

#[test]
fn only_sw_and_v2_are_safe_literal_returns() {
    let sw = compact(kernel_body(SAFE[0].0).text);
    assert!(sw.contains("n=blockIdx.x*N_PER_BLOCK_SW+local_out;"));
    assert!(sw.contains("if(n>=N)return;") && !sw.contains("__syncthreads();"));
    let v2 = compact(kernel_body(SAFE[1].0).text);
    assert!(v2.contains("n=blockIdx.x;") && v2.contains("if(n>=N)return;"));
    assert_eq!(v2.matches("__syncthreads();").count(), 2);
    let classified = SAFE
        .iter()
        .chain(UNSAFE)
        .filter(|k| compact(kernel_body(k.0).text).contains("if(n>=N)return;"))
        .count();
    assert_eq!(KERNEL.matches("if (n >= N) return;").count(), classified);
}

#[test]
fn every_unsafe_kernel_has_the_future_green_contract() {
    let failures: Vec<_> = UNSAFE
        .iter()
        .copied()
        .filter_map(|k| {
            let errors = future_errors(k);
            (!errors.is_empty()).then(|| format!("{}: {}", k.0, errors.join(", ")))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "barrier-tail contract failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn wrappers_are_ceil_div_without_modulo_guards() {
    for (name, group) in WRAPPER_GROUPS {
        let body = compact(body_after(WRAPPERS, &format!("pub fn {name}(")).text);
        let grid = format!(".grid([div_ceil(n,{group}),");
        assert!(body.contains(&grid), "{name}");
        for guard in ["n%", "is_multiple_of", "rem_euclid"] {
            assert!(!body.contains(guard), "{name} gained modulo guard {guard}");
        }
    }
}
