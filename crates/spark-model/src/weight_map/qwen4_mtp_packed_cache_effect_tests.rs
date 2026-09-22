// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::{DevicePtr, KernelHandle, mock::MockGpuBackend};

use super::*;

const EXPERTS: usize = 512;
const INTER: usize = 640;
const HIDDEN: usize = 2_560;

#[test]
fn public_cached_helper_rejects_invalid_device_spans_before_gpu_effects() {
    let gpu = MockGpuBackend::new();
    for source in [
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape: vec![EXPERTS, 2 * INTER, HIDDEN],
            dtype: WeightDtype::BF16,
        },
        WeightTensor {
            ptr: DevicePtr(u64::MAX - 1),
            shape: vec![EXPERTS, 2 * INTER, HIDDEN],
            dtype: WeightDtype::BF16,
        },
    ] {
        let result = quantize_packed_mtp_slice(
            true,
            GATE_UP_NAME,
            &source,
            0,
            PackedMtpProjection::Gate,
            0,
            INTER,
            HIDDEN,
            &gpu,
            KernelHandle(1),
            KernelHandle(2),
            0,
        );
        assert!(result.is_err());
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
    }
}
