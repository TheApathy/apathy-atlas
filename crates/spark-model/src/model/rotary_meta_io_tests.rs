// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::model::vision_embeddings::test_gpu::MemoryGpu;
use crate::traits::ImageSpan;

fn image_map() -> RotaryPositions {
    RotaryPositions::from_image_spans(
        6,
        32,
        &[ImageSpan {
            start: 1,
            height: 2,
            width: 2,
        }],
    )
    .unwrap()
}

#[test]
fn scalar_axis_upload_preserves_physical_metadata_and_pointer_topology() {
    let gpu = MemoryGpu::new();
    let base = gpu.alloc(512).unwrap();
    for physical in [2, 6, 4, 7] {
        gpu.copy_h2d(&[0xdd; 512], base).unwrap();
        let map = image_map();
        let axes = SingleRotary::new(&map, physical).unwrap();
        let pointers = axes.upload_axes(&gpu, base, 0).unwrap();
        assert_eq!(pointers, (base.offset(20), base.offset(24)));
        let mut expected = vec![0xdd; 512];
        let [_, h, w] = map.position(physical).unwrap();
        expected[20..24].copy_from_slice(&h.to_le_bytes());
        expected[24..28].copy_from_slice(&w.to_le_bytes());
        assert_eq!(gpu.bytes(base), expected);
    }
    gpu.free(base).unwrap();
}

#[test]
fn text_alias_upload_is_byte_exact_noop_and_image_overflow_is_preflighted() {
    let gpu = MemoryGpu::new();
    let base = gpu.alloc(512).unwrap();
    let text = SingleRotary::new(&RotaryPositions::identity(), 7).unwrap();
    assert_eq!(text.upload_axes(&gpu, base, 0).unwrap(), (base, base));
    assert_eq!(gpu.bytes(base), vec![0xdd; 512]);
    let image = SingleRotary::new(&image_map(), 2).unwrap();
    assert!(
        image
            .upload_axes(&gpu, DevicePtr(u64::MAX - 24), 0)
            .is_err()
    );
    assert_eq!(gpu.bytes(base), vec![0xdd; 512]);
    gpu.free(base).unwrap();
}

#[test]
fn batch_axes_and_padding_never_touch_physical_slot_region() {
    let gpu = MemoryGpu::new();
    let base = gpu.alloc(1024).unwrap();
    let map = image_map();
    let batch = BatchRotary::new(&[(&map, 2), (&map, 6)], 4).unwrap();
    assert_eq!(
        batch.upload_axes(&gpu, base, 0).unwrap(),
        (base.offset(16), base.offset(32))
    );
    let mut expected = vec![0xdd; 1024];
    expected[16..32].fill(0);
    expected[32..48].fill(0);
    expected[16..20].copy_from_slice(&1u32.to_le_bytes());
    expected[20..24].copy_from_slice(&4u32.to_le_bytes());
    expected[32..36].copy_from_slice(&2u32.to_le_bytes());
    expected[36..40].copy_from_slice(&4u32.to_le_bytes());
    assert_eq!(gpu.bytes(base), expected);
    assert!(
        batch
            .upload_axes(&gpu, DevicePtr(u64::MAX - 40), 0)
            .is_err()
    );
    assert_eq!(gpu.bytes(base), expected);
    gpu.free(base).unwrap();
}
