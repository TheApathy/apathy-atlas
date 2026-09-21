// SPDX-License-Identifier: AGPL-3.0-only

//! Vision topology policy, checked before costly startup work.

/// Server passes all requested speculation flags and declared topology; factory
/// repeats the same policy with effective configuration and installed inputs.
/// `supported_execution` is target-only or the separately validated explicit
/// native-DSpark qualification candidate. No mode is silently downgraded.
pub fn validate_deepseek_vision_execution(
    is_vision: bool,
    supported_execution: bool,
    single_gpu: bool,
    max_batch_size: usize,
    prefix_caching: bool,
) -> Result<(), &'static str> {
    if !is_vision {
        return Ok(());
    }
    if !supported_execution {
        return Err("DeepSeek Vision speculative decoding is not yet qualified; use target-only");
    }
    if !single_gpu || max_batch_size != 1 {
        return Err("DeepSeek Vision currently requires single-GPU C1");
    }
    if prefix_caching {
        return Err("DeepSeek Vision currently requires prefix caching disabled");
    }
    Ok(())
}
