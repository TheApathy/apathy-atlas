// SPDX-License-Identifier: AGPL-3.0-only

//! Admission for the original-layout Qwen4 grouped-FFN experiment.
//! Config/arena admission is separate; this checks the actual loaded layer.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{ExpertPtrTable, ExpertWeight, MoeLayer, QuantizedWeight};

fn weight_valid(weight: &QuantizedWeight) -> bool {
    weight.weight != DevicePtr::NULL
        && weight.weight_scale != DevicePtr::NULL
        && weight.weight_scale_2.is_finite()
        && weight.weight_scale_2 > 0.0
}

fn expert_valid(expert: &ExpertWeight) -> bool {
    [&expert.gate_proj, &expert.up_proj, &expert.down_proj]
        .into_iter()
        .all(weight_valid)
}

fn table_valid(table: &ExpertPtrTable) -> bool {
    table.packed_ptrs != DevicePtr::NULL
        && table.scale_ptrs != DevicePtr::NULL
        && table.scale2_vals != DevicePtr::NULL
}

fn projection_bundle_valid(
    predequant: [Option<u64>; 4],
    fp8_gemm: u64,
    nvfp4_valid: bool,
    nvfp4_gemm: u64,
) -> bool {
    if predequant.iter().all(Option::is_none) {
        nvfp4_valid && nvfp4_gemm != 0
    } else {
        predequant.iter().all(|p| p.is_some_and(|p| p != 0)) && fp8_gemm != 0
    }
}

impl MoeLayer {
    /// No device effects or fallback: require the grouped NVFP4 expert route.
    pub(crate) fn validate_qwen4_prefill_moe(&self) -> Result<()> {
        super::qwen4_prefill_compact::validate_handles(self.qwen4_compact)?;
        let transposed = self.qwen4_compact.is_some_and(|kernels| kernels.transposed);
        let streamed = self.qwen4_compact.is_some_and(|kernels| kernels.streamed);
        ensure!(
            self.fp8_gate_weight_ptrs.is_none()
                && self.fp8_up_weight_ptrs.is_none()
                && self.fp8_down_weight_ptrs.is_none()
                && self.fp8_shared_expert.is_none(),
            "Qwen4 MoE-only prefill rejects native FP8 expert fallback"
        );
        if transposed {
            let tables = self.gate_ptrs_t.as_ref().is_some_and(table_valid)
                && self.up_ptrs_t.as_ref().is_some_and(table_valid)
                && self.down_ptrs_t.as_ref().is_some_and(table_valid);
            if streamed {
                ensure!(
                    !self.unified_layout
                        && !self.hybrid_layout
                        && tables
                        && self.stream_t_scratch.is_some()
                        && self.down_t_scratch_packed.is_none()
                        && self.down_t_scratch_scale.is_none()
                        && !self.nvfp4_moe_worklist,
                    "Qwen4 stream compact prefill requires complete scratch and originals"
                );
            } else {
                ensure!(
                    self.unified_layout
                        && !self.hybrid_layout
                        && tables
                        && self.stream_t_scratch.is_none()
                        && self.down_t_scratch_packed.is_none()
                        && self.down_t_scratch_scale.is_none()
                        && !self.nvfp4_moe_worklist,
                    "Qwen4 transposed compact prefill requires a complete unified replacement"
                );
            }
        } else {
            ensure!(
                self.gate_ptrs_t.is_none()
                    && self.up_ptrs_t.is_none()
                    && self.down_ptrs_t.is_none()
                    && self.shared_gate_t.is_none()
                    && self.shared_up_t.is_none()
                    && self.shared_down_t.is_none()
                    && self.stream_t_scratch.is_none()
                    && self.down_t_scratch_packed.is_none()
                    && self.down_t_scratch_scale.is_none()
                    && !self.nvfp4_moe_worklist,
                "Qwen4 MoE-only prefill requires original-layout grouped experts"
            );
        }
        ensure!(
            self.weights.router_pre_norm.is_none()
                && self.pre_expert_norm.is_none()
                && self.correction_bias_dev.is_none()
                && self.weights.correction_bias.is_none()
                && !self.gelu_activation,
            "Qwen4 MoE-only prefill rejects alternate routing/activation"
        );
        ensure!(
            self.weights.experts.len() == 512,
            "Qwen4 MoE-only prefill requires 512 experts"
        );
        if !transposed || streamed {
            ensure!(
                self.weights.experts.iter().all(expert_valid)
                    && table_valid(&self.gate_ptrs)
                    && table_valid(&self.up_ptrs)
                    && table_valid(&self.down_ptrs),
                "Qwen4 MoE-only prefill requires resident original NVFP4 experts"
            );
        }
        ensure!(
            ((!streamed && transposed) || expert_valid(&self.weights.shared_expert))
                && self.weights.shared_expert_gate.weight != DevicePtr::NULL,
            "Qwen4 MoE-only prefill requires the shared expert gate"
        );
        ensure!(
            [
                self.moe_topk_batched.0,
                self.moe_sort_by_expert.0,
                self.moe_grouped_gemm.0,
                self.moe_act_mul.0,
                self.moe_unpermute_reduce.0,
                self.moe_batched_blend.0,
            ]
            .into_iter()
            .all(|k| k != 0),
            "Qwen4 MoE-only prefill requires all selected grouped kernels"
        );
        // The existing loader normally predequants router/shared weights to
        // FP8 for prefill. Admit that COMPLETE bundle explicitly; it is a
        // numerical delta from decode, not evidence of GEMV/GEMM parity.
        // With no predequant bundle, admit only the complete NVFP4 path.
        ensure!(
            projection_bundle_valid(
                [
                    self.gate_fp8.map(|p| p.0),
                    self.shared_gate_fp8.map(|p| p.0),
                    self.shared_up_fp8.map(|p| p.0),
                    self.shared_down_fp8.map(|p| p.0),
                ],
                self.fp8_gemm_k.0,
                self.gate_nvfp4.as_ref().is_some_and(weight_valid),
                self.w4a16_gemm.0,
            ),
            "Qwen4 MoE-only prefill requires a complete router/shared projection bundle"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::projection_bundle_valid;

    #[test]
    fn complete_fp8_does_not_require_unused_nvfp4_kernel() {
        assert!(projection_bundle_valid([Some(1); 4], 1, false, 0));
        assert!(!projection_bundle_valid([Some(1); 4], 0, true, 1));
    }

    #[test]
    fn absent_predequant_requires_nvfp4_weight_and_kernel() {
        assert!(projection_bundle_valid([None; 4], 0, true, 1));
        assert!(!projection_bundle_valid([None; 4], 1, false, 1));
        assert!(!projection_bundle_valid([None; 4], 1, true, 0));
    }

    #[test]
    fn every_partial_or_null_predequant_bundle_rejects() {
        for mask in 1..15 {
            let values = std::array::from_fn(|i| (mask & (1 << i) != 0).then_some(1));
            assert!(!projection_bundle_valid(values, 1, true, 1));
        }
        for null_index in 0..4 {
            let mut values = [Some(1); 4];
            values[null_index] = Some(0);
            assert!(!projection_bundle_valid(values, 1, true, 1));
        }
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;

    #[test]
    fn null_or_nonfinite_expert_metadata_rejects() {
        let valid = QuantizedWeight {
            weight: DevicePtr(1),
            weight_scale: DevicePtr(2),
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL, // unused by selected W4A16 kernels
        };
        assert!(weight_valid(&valid));
        assert!(!weight_valid(&QuantizedWeight::null()));
        assert!(!weight_valid(&QuantizedWeight {
            weight_scale: DevicePtr::NULL,
            ..valid
        }));
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!weight_valid(&QuantizedWeight {
                weight_scale_2: bad,
                ..valid
            }));
        }
        let mut expert = ExpertWeight {
            gate_proj: valid,
            up_proj: valid,
            down_proj: valid,
        };
        assert!(expert_valid(&expert));
        expert.up_proj = QuantizedWeight::null();
        assert!(!expert_valid(&expert));
    }

    #[test]
    fn every_original_pointer_table_component_is_required() {
        for missing in 0..3 {
            let mut pointers = [DevicePtr(1); 3];
            pointers[missing] = DevicePtr::NULL;
            assert!(!table_valid(&ExpertPtrTable {
                packed_ptrs: pointers[0],
                scale_ptrs: pointers[1],
                scale2_vals: pointers[2],
            }));
        }
        assert!(table_valid(&ExpertPtrTable {
            packed_ptrs: DevicePtr(1),
            scale_ptrs: DevicePtr(2),
            scale2_vals: DevicePtr(3),
        }));
    }
}
