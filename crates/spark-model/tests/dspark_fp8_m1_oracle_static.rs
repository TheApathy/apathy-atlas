// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source-contract guard for the shared FP8 M=1 GEMV tail.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/dense_gemv_fp8w.cu");

const OUTPUTS_PER_BLOCK: usize = 4;
const THREADS_PER_OUTPUT: usize = 64;
const BLOCK_THREADS: usize = OUTPUTS_PER_BLOCK * THREADS_PER_OUTPUT;

fn dense_kernel() -> &'static str {
    let marker = "extern \"C\" __global__ void dense_gemv_fp8w(";
    let start = KERNEL
        .find(marker)
        .expect("dense_gemv_fp8w kernel must remain present");
    &KERNEL[start..]
}

fn braced_body_after(source: &str, needle: &str) -> (usize, std::ops::Range<usize>) {
    let needle_index = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing source contract: {needle}"));
    let open_index = source[needle_index..]
        .find('{')
        .map(|offset| needle_index + offset)
        .unwrap_or_else(|| panic!("missing opening brace after: {needle}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open_index..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return (needle_index, open_index + 1..open_index + offset);
                }
            }
            _ => {}
        }
    }
    panic!("missing closing brace after: {needle}");
}

fn last_block_topology(n: usize) -> (usize, [usize; OUTPUTS_PER_BLOCK], usize) {
    assert!(n > 0);
    let grid = n.div_ceil(OUTPUTS_PER_BLOCK);
    let base = (grid - 1) * OUTPUTS_PER_BLOCK;
    let rows = std::array::from_fn(|local_out| base + local_out);
    let valid_groups = rows.iter().filter(|&&row| row < n).count();
    (grid, rows, valid_groups)
}

#[test]
fn tail_topology_covers_every_n_mod_4_class() {
    for remainder in 0..OUTPUTS_PER_BLOCK {
        let n = 8 + remainder;
        let (grid, rows, valid_groups) = last_block_topology(n);
        let expected_groups = if remainder == 0 {
            OUTPUTS_PER_BLOCK
        } else {
            remainder
        };
        assert_eq!(grid, n.div_ceil(OUTPUTS_PER_BLOCK));
        assert_eq!(valid_groups, expected_groups, "N={n}, rows={rows:?}");
        assert_eq!(
            valid_groups * THREADS_PER_OUTPUT
                + (OUTPUTS_PER_BLOCK - valid_groups) * THREADS_PER_OUTPUT,
            BLOCK_THREADS,
        );
    }
}

#[test]
fn public_logical_tail_and_physical_crop_are_exact() {
    let (logical_grid, logical_rows, logical_groups) = last_block_topology(248_077);
    assert_eq!(logical_grid, 62_020);
    assert_eq!(logical_rows, [248_076, 248_077, 248_078, 248_079]);
    assert_eq!(logical_groups, 1);
    assert_eq!(logical_groups * THREADS_PER_OUTPUT, 64);
    assert_eq!(BLOCK_THREADS - logical_groups * THREADS_PER_OUTPUT, 192);

    let (physical_grid, physical_rows, physical_groups) = last_block_topology(248_320);
    assert_eq!(physical_grid, 62_080);
    assert_eq!(physical_rows, [248_316, 248_317, 248_318, 248_319]);
    assert_eq!(physical_groups, OUTPUTS_PER_BLOCK);

    assert_eq!(248_320 - 248_077, 243);
    assert_eq!((248_320 - 248_077) * size_of::<u16>(), 486);
}

#[test]
fn partial_output_groups_do_not_return_before_the_block_barrier() {
    let source = dense_kernel();
    let n_index = source
        .find("const unsigned int n =")
        .expect("kernel must compute its output row");
    let barrier_index = source[n_index..]
        .find("__syncthreads();")
        .map(|offset| n_index + offset)
        .expect("kernel must retain its cross-warp block barrier");
    let before_barrier = &source[n_index..barrier_index];

    assert!(
        !before_barrier.contains("return;"),
        "a partial output group must not exit before a block-wide barrier"
    );
    assert!(before_barrier.contains("const bool valid = n < N;"));
}

#[test]
fn invalid_groups_cannot_form_row_addresses_or_store_output() {
    let source = dense_kernel();
    let (valid_index, valid_body) = braced_body_after(source, "if (valid)");
    let reduction_index = source
        .find("// Warp shuffle reduction")
        .expect("kernel must retain its reduction boundary");
    assert!(valid_body.end < reduction_index);
    let guarded = &source[valid_body];
    let before_guard = &source[..valid_index];

    assert!(!before_guard.contains("row_scale[n]"));
    assert!(!before_guard.contains("B + (unsigned long long)n * K"));
    for required in [
        "row_scale[n]",
        "B + (unsigned long long)n * K",
        "uint4 b_data = B_vec[kv]",
        "uint4 a_data0 = ((const uint4*)A)[kv * 2]",
        "uint4 a_data1 = ((const uint4*)A)[kv * 2 + 1]",
    ] {
        assert!(
            guarded.contains(required),
            "unguarded/missing load: {required}"
        );
    }
    let (_, store_body) = braced_body_after(source, "if (valid && lane == 0)");
    assert!(source[store_body].contains("C[n] = __float2bfloat16(result);"));
}

#[test]
fn row_scale_precedes_the_existing_reduction_order() {
    let source = dense_kernel();
    let scale_index = source
        .find("acc *= scale;")
        .expect("valid rows must apply their direct row multiplier");
    let shuffle_index = source
        .find("__shfl_down_sync")
        .expect("kernel must retain its warp reduction");
    let shared_add_index = source
        .find("smem[local_out * 2] + smem[local_out * 2 + 1]")
        .expect("kernel must retain its two-warp add order");

    assert!(scale_index < shuffle_index);
    assert!(shuffle_index < shared_add_index);
}
