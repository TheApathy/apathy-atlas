// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed fc1 schema and the pinned teacher's FP64 metric equations. No gates
//! or tolerance knobs here: candidate acceptance remains exact payload SHA.
use anyhow::{Result, ensure};
use deepseek_vision_p1::contract::check_bf16;
use serde_json::{Value, json};

const ROWS: usize = 20;
const COLS: usize = 5632;
#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let next = self.value + value;
        self.correction += if self.value.abs() >= value.abs() {
            (self.value - next) + value
        } else {
            (value - next) + self.value
        };
        self.value = next;
    }
    fn total(&self) -> f64 {
        self.value + self.correction
    }
}
fn decode(raw: &[u8], index: usize) -> f64 {
    let bits = u16::from_le_bytes([raw[2 * index], raw[2 * index + 1]]);
    f32::from_bits(u32::from(bits) << 16) as f64
}
pub fn compare(actual: &[u8], reference: &[u8]) -> Result<Value> {
    ensure!(
        actual.len() == ROWS * COLS * 2 && reference.len() == actual.len(),
        "exact 20x5632 BF16 extent required"
    );
    check_bf16(actual)?;
    check_bf16(reference)?;
    let (mut aa, mut bb, mut dot, mut error2) = (
        Sum::default(),
        Sum::default(),
        Sum::default(),
        Sum::default(),
    );
    let mut worst = f64::INFINITY;
    let mut max_error = 0.0f64;
    let mut equal = 0usize;
    for row in 0..ROWS {
        let (mut ra, mut rb, mut rd) = (Sum::default(), Sum::default(), Sum::default());
        for col in 0..COLS {
            let index = row * COLS + col;
            let (a, b) = (decode(actual, index), decode(reference, index));
            ra.add(a * a);
            rb.add(b * b);
            rd.add(a * b);
            let error = a - b;
            error2.add(error * error);
            max_error = max_error.max(error.abs());
            if a == b {
                equal += 1;
            } // Torch numeric equality includes +0/-0.
        }
        let (a, b, d) = (ra.total(), rb.total(), rd.total());
        ensure!(a > 0.0 && b > 0.0, "zero norm comparison row {row}");
        worst = worst.min(d / (a.sqrt() * b.sqrt()));
        aa.add(a);
        bb.add(b);
        dot.add(d);
    }
    let reference_norm = bb.total().sqrt();
    let cosine = dot.total() / (aa.total().sqrt() * reference_norm);
    let relative_l2 = error2.total().sqrt() / reference_norm;
    let exact = (equal as f32 / (ROWS * COLS) as f32) as f64;
    ensure!(
        [cosine, worst, relative_l2, max_error, exact]
            .iter()
            .all(|v| v.is_finite()),
        "nonfinite FP64 metric"
    );
    Ok(
        json!({"cosine":cosine,"worst_row_cosine":worst,"relative_l2":relative_l2,
        "max_abs_error":max_error,"f32_exact_fraction":exact,
        "shape":[ROWS,COLS],"dtype":"BF16","reduction":"compensated serial FP64; Torch parallel FP64 can differ in final bits"}),
    )
}
