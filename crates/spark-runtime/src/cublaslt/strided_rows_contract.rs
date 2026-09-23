// SPDX-License-Identifier: AGPL-3.0-only
//! Per-row alignment admission for an already validated scoped M1 request.
use super::serial_rows_contract::{RowCall, SerialRowsRequest};

pub fn admit_strided_m1(
    request: SerialRowsRequest,
    call: RowCall,
    capabilities: [u32; 5],
) -> Result<[i64; 3], &'static str> {
    if !(2..=8).contains(&request.rows) || request.n == 0 || request.k == 0 {
        return Err("strided M1 geometry is outside the native row contract");
    }
    if capabilities[0] == 0 {
        return Err("selected scalar algorithm does not support strided batches");
    }
    let operands = [
        (call.weight, 0),
        (call.act, u64::from(request.k) * 2),
        (call.out, u64::from(request.n) * 2),
        (call.out, u64::from(request.n) * 2),
    ];
    for ((base, stride), alignment) in operands.into_iter().zip(capabilities[1..].iter().copied()) {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err("invalid selected algorithm alignment capability");
        }
        for row in 0..u64::from(request.rows) {
            let address = row
                .checked_mul(stride)
                .and_then(|offset| base.checked_add(offset))
                .ok_or("strided M1 operand address overflow")?;
            if address == 0 || address % u64::from(alignment) != 0 {
                return Err("strided M1 operand violates the scalar algorithm alignment");
            }
        }
    }
    // CUDA matrix-layout strides are ELEMENT counts, not bytes.
    Ok([0, i64::from(request.k), i64::from(request.n)])
}
