// SPDX-License-Identifier: AGPL-3.0-only

//! Arithmetic-only GLM-5.3-Flash target-cache residency planning.
//!
//! This module allocates nothing and enables no cache kernel. In particular,
//! selecting FP8 here only describes a requested E4M3 representation; runtime
//! kernel support and accuracy still require separate qualification. Callers
//! must supply bytes left after weights, workspaces, and optional speculative
//! checkpoint/intermediate buffers. DSA region geometry mirrors the dedicated
//! runtime planner; this module does not claim that its kernels are implemented.

use anyhow::{Context, Result, bail};

pub const GLM53_MAX_CONTEXT_TOKENS: u32 = 1_048_576;
pub const GLM53_KDA_LAYERS: u32 = 34;
pub const GLM53_DSA_LAYERS: u32 = 11;
pub const GLM53_DENSE_KV_CACHE_STREAMS: u32 = 0;

const ALIGNMENT_BYTES: u64 = 256;
const KDA_HEADS: u64 = 64;
const KDA_KEY_DIM: u64 = 128;
const KDA_VALUE_DIM: u64 = 128;
const KDA_CONV_STREAMS: u64 = 3;
const KDA_CONV_CHANNELS: u64 = 8_192;
const KDA_CONV_KERNEL: u64 = 4;
const FP32_BYTES: u64 = 4;
const DSA_LATENT_DIM: u64 = 512;
const DSA_INDEX_DIM: u64 = 128;
const DSA_INDEX_KPOOL: u64 = 4;
const DSA_INDEX_TAIL_CAPACITY: u64 = DSA_INDEX_KPOOL - 1;
const DSA_VALIDITY_BYTES: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53ContextDtype {
    Bf16,
    Fp8E4M3,
}

impl Glm53ContextDtype {
    const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4M3 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaStorage {
    pub latent: Glm53ContextDtype,
    pub index: Glm53ContextDtype,
}

impl Glm53DsaStorage {
    pub const BF16: Self = Self {
        latent: Glm53ContextDtype::Bf16,
        index: Glm53ContextDtype::Bf16,
    };
    pub const FP8: Self = Self {
        latent: Glm53ContextDtype::Fp8E4M3,
        index: Glm53ContextDtype::Fp8E4M3,
    };

    fn validate(self) -> Result<()> {
        if self.latent != self.index {
            bail!("GLM-5.3 DSA runtime requires one storage width for latent and index");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53ContextRegion {
    pub offset_bytes: u64,
    pub payload_bytes: u64,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53ContextPlan {
    pub batch: u32,
    pub positions: u32,
    /// Active sequences plus the fixed padding slot used by `SsmStatePool`.
    pub kda_pool_slots: u64,
    pub index_pool_entries_per_sequence: u32,
    /// Persistent raw tail capacity after complete kpool4 groups are compressed.
    pub index_tail_capacity: u32,
    pub index_tail_valid_tokens: u32,
    pub storage: Glm53DsaStorage,
    pub kda_recurrent_f32: Glm53ContextRegion,
    pub kda_conv_f32: Glm53ContextRegion,
    pub dsa_latent: Glm53ContextRegion,
    pub dsa_pooled_index: Glm53ContextRegion,
    /// One byte for every completed pooled index key.
    pub dsa_pool_validity: Glm53ContextRegion,
    /// Three raw key128 entries per DSA layer and sequence.
    pub dsa_tail_keys: Glm53ContextRegion,
    /// Three raw gate128 entries per DSA layer and sequence.
    pub dsa_tail_gates: Glm53ContextRegion,
    /// One byte for each raw tail entry.
    pub dsa_tail_validity: Glm53ContextRegion,
    /// Load-bearing proof that expanded 64-head K/V is not part of this plan.
    pub dense_kv_bytes: u64,
    pub total_bytes: u64,
}

impl Glm53ContextPlan {
    pub fn new(batch: u32, positions: u32, storage: Glm53DsaStorage) -> Result<Self> {
        if batch == 0 {
            bail!("GLM-5.3 context planning requires a nonzero batch");
        }
        if positions == 0 || positions > GLM53_MAX_CONTEXT_TOKENS {
            bail!("GLM-5.3 context positions must be in 1..={GLM53_MAX_CONTEXT_TOKENS}");
        }
        storage.validate()?;

        let batch = u64::from(batch);
        let positions = u64::from(positions);
        let kda_pool_slots = batch
            .checked_add(1)
            .context("GLM-5.3 KDA padding-slot count overflow")?;
        let pool_entries = positions / DSA_INDEX_KPOOL;
        let tail_valid = positions % DSA_INDEX_KPOOL;
        let mut cursor = 0;

        let kda_recurrent_f32 = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 KDA recurrent state",
                &[
                    kda_pool_slots,
                    u64::from(GLM53_KDA_LAYERS),
                    KDA_HEADS,
                    KDA_KEY_DIM,
                    KDA_VALUE_DIM,
                    FP32_BYTES,
                ],
            )?,
        )?;
        let kda_conv_f32 = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 KDA convolution state",
                &[
                    kda_pool_slots,
                    u64::from(GLM53_KDA_LAYERS),
                    KDA_CONV_STREAMS,
                    KDA_CONV_CHANNELS,
                    KDA_CONV_KERNEL,
                    FP32_BYTES,
                ],
            )?,
        )?;
        let dsa_latent = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 DSA latent cache",
                &[
                    batch,
                    u64::from(GLM53_DSA_LAYERS),
                    positions,
                    DSA_LATENT_DIM,
                    storage.latent.bytes_per_element(),
                ],
            )?,
        )?;
        let dsa_pooled_index = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 DSA pooled index cache",
                &[
                    batch,
                    u64::from(GLM53_DSA_LAYERS),
                    pool_entries,
                    DSA_INDEX_DIM,
                    storage.index.bytes_per_element(),
                ],
            )?,
        )?;
        let dsa_pool_validity = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 DSA completed-pool validity",
                &[
                    batch,
                    u64::from(GLM53_DSA_LAYERS),
                    pool_entries,
                    DSA_VALIDITY_BYTES,
                ],
            )?,
        )?;
        let dsa_tail_keys = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 DSA tail keys",
                &[
                    batch,
                    u64::from(GLM53_DSA_LAYERS),
                    DSA_INDEX_TAIL_CAPACITY,
                    DSA_INDEX_DIM,
                    storage.index.bytes_per_element(),
                ],
            )?,
        )?;
        let dsa_tail_gates = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 DSA tail gates",
                &[
                    batch,
                    u64::from(GLM53_DSA_LAYERS),
                    DSA_INDEX_TAIL_CAPACITY,
                    DSA_INDEX_DIM,
                    storage.index.bytes_per_element(),
                ],
            )?,
        )?;
        let dsa_tail_validity = append_region(
            &mut cursor,
            checked_product(
                "GLM-5.3 DSA tail validity",
                &[
                    batch,
                    u64::from(GLM53_DSA_LAYERS),
                    DSA_INDEX_TAIL_CAPACITY,
                    DSA_VALIDITY_BYTES,
                ],
            )?,
        )?;

        Ok(Self {
            batch: u32::try_from(batch)?,
            positions: u32::try_from(positions)?,
            kda_pool_slots,
            index_pool_entries_per_sequence: u32::try_from(pool_entries)?,
            index_tail_capacity: u32::try_from(DSA_INDEX_TAIL_CAPACITY)?,
            index_tail_valid_tokens: u32::try_from(tail_valid)?,
            storage,
            kda_recurrent_f32,
            kda_conv_f32,
            dsa_latent,
            dsa_pooled_index,
            dsa_pool_validity,
            dsa_tail_keys,
            dsa_tail_gates,
            dsa_tail_validity,
            dense_kv_bytes: u64::from(GLM53_DENSE_KV_CACHE_STREAMS),
            total_bytes: cursor,
        })
    }

    pub fn kda_bytes(self) -> u64 {
        self.kda_recurrent_f32.allocation_bytes + self.kda_conv_f32.allocation_bytes
    }

    pub fn dsa_bytes(self) -> u64 {
        self.dsa_latent.allocation_bytes
            + self.dsa_pooled_index.allocation_bytes
            + self.dsa_pool_validity.allocation_bytes
            + self.dsa_tail_keys.allocation_bytes
            + self.dsa_tail_gates.allocation_bytes
            + self.dsa_tail_validity.allocation_bytes
    }

    pub fn admit_residency(self, usable_bytes: u64) -> Glm53ContextAdmission {
        let fits = self.total_bytes <= usable_bytes;
        Glm53ContextAdmission {
            required_bytes: self.total_bytes,
            usable_bytes,
            fits,
            headroom_bytes: usable_bytes.saturating_sub(self.total_bytes),
            shortfall_bytes: self.total_bytes.saturating_sub(usable_bytes),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53ContextAdmission {
    pub required_bytes: u64,
    pub usable_bytes: u64,
    pub fits: bool,
    pub headroom_bytes: u64,
    pub shortfall_bytes: u64,
}

fn checked_product(label: &str, factors: &[u64]) -> Result<u64> {
    let mut product = 1u64;
    for factor in factors {
        product = product
            .checked_mul(*factor)
            .with_context(|| format!("{label} byte count overflow"))?;
    }
    Ok(product)
}

fn align_up(value: u64) -> Result<u64> {
    value
        .checked_add(ALIGNMENT_BYTES - 1)
        .map(|rounded| rounded / ALIGNMENT_BYTES * ALIGNMENT_BYTES)
        .context("GLM-5.3 context alignment overflow")
}

fn append_region(cursor: &mut u64, payload_bytes: u64) -> Result<Glm53ContextRegion> {
    let offset_bytes = align_up(*cursor)?;
    let allocation_bytes = align_up(payload_bytes)?;
    *cursor = offset_bytes
        .checked_add(allocation_bytes)
        .context("GLM-5.3 context arena extent overflow")?;
    Ok(Glm53ContextRegion {
        offset_bytes,
        payload_bytes,
        allocation_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_one_million_bf16_report_is_pinned() {
        let plan =
            Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::BF16).unwrap();
        assert_eq!(plan.kda_pool_slots, 2);
        assert_eq!(plan.kda_recurrent_f32.payload_bytes, 285_212_672);
        assert_eq!(plan.kda_conv_f32.payload_bytes, 26_738_688);
        assert_eq!(plan.dsa_latent.payload_bytes, 11_811_160_064);
        assert_eq!(plan.dsa_pooled_index.payload_bytes, 738_197_504);
        assert_eq!(plan.dsa_pool_validity.payload_bytes, 2_883_584);
        assert_eq!(plan.dsa_tail_keys.payload_bytes, 8_448);
        assert_eq!(plan.dsa_tail_gates.payload_bytes, 8_448);
        assert_eq!(plan.dsa_tail_validity.payload_bytes, 33);
        assert_eq!(plan.dsa_tail_validity.allocation_bytes, 256);
        assert_eq!(plan.index_pool_entries_per_sequence, 262_144);
        assert_eq!(plan.index_tail_capacity, 3);
        assert_eq!(plan.index_tail_valid_tokens, 0);
        assert_eq!(plan.kda_bytes(), 311_951_360);
        assert_eq!(plan.dsa_bytes(), 12_552_258_304);
        assert_eq!(plan.total_bytes, 12_864_209_664);
        for region in [
            plan.kda_recurrent_f32,
            plan.kda_conv_f32,
            plan.dsa_latent,
            plan.dsa_pooled_index,
            plan.dsa_pool_validity,
            plan.dsa_tail_keys,
            plan.dsa_tail_gates,
            plan.dsa_tail_validity,
        ] {
            assert_eq!(region.offset_bytes % ALIGNMENT_BYTES, 0);
            assert_eq!(region.allocation_bytes % ALIGNMENT_BYTES, 0);
        }
    }

    #[test]
    fn fp8_sizes_only_dsa_payloads_and_never_dense_kv() {
        let plan =
            Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::FP8).unwrap();
        assert_eq!(plan.kda_recurrent_f32.payload_bytes, 285_212_672);
        assert_eq!(plan.kda_conv_f32.payload_bytes, 26_738_688);
        assert_eq!(plan.dsa_latent.payload_bytes, 5_905_580_032);
        assert_eq!(plan.dsa_pooled_index.payload_bytes, 369_098_752);
        assert_eq!(plan.dsa_pool_validity.payload_bytes, 2_883_584);
        assert_eq!(plan.dsa_tail_keys.payload_bytes, 4_224);
        assert_eq!(plan.dsa_tail_keys.allocation_bytes, 4_352);
        assert_eq!(plan.dsa_tail_gates.payload_bytes, 4_224);
        assert_eq!(plan.dsa_tail_gates.allocation_bytes, 4_352);
        assert_eq!(plan.dsa_tail_validity.payload_bytes, 33);
        assert_eq!(plan.dense_kv_bytes, 0);
        assert_eq!(plan.dsa_bytes(), 6_277_571_328);
        assert_eq!(plan.total_bytes, 6_589_522_688);

        let forbidden_dense = 11u64 * 1_048_576 * 2 * 64 * 256;
        assert!(plan.dsa_bytes() < forbidden_dense);
        assert_eq!(GLM53_DENSE_KV_CACHE_STREAMS, 0);

        assert!(
            Glm53ContextPlan::new(
                1,
                GLM53_MAX_CONTEXT_TOKENS,
                Glm53DsaStorage {
                    latent: Glm53ContextDtype::Bf16,
                    index: Glm53ContextDtype::Fp8E4M3,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn kpool_tail_and_context_boundary_have_no_off_by_one() {
        let short = Glm53ContextPlan::new(1, 3, Glm53DsaStorage::BF16).unwrap();
        assert_eq!(short.index_pool_entries_per_sequence, 0);
        assert_eq!(short.index_tail_capacity, 3);
        assert_eq!(short.index_tail_valid_tokens, 3);
        assert_eq!(short.dsa_pool_validity.payload_bytes, 0);

        let pooled = Glm53ContextPlan::new(1, 4, Glm53DsaStorage::BF16).unwrap();
        assert_eq!(pooled.index_pool_entries_per_sequence, 1);
        assert_eq!(pooled.index_tail_valid_tokens, 0);
        assert_eq!(pooled.dsa_pool_validity.payload_bytes, 11);

        let below =
            Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS - 1, Glm53DsaStorage::BF16).unwrap();
        assert_eq!(below.index_pool_entries_per_sequence, 262_143);
        assert_eq!(below.index_tail_valid_tokens, 3);

        let exact =
            Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::BF16).unwrap();
        assert_eq!(exact.index_pool_entries_per_sequence, 262_144);
        assert_eq!(exact.index_tail_valid_tokens, 0);
        assert!(
            Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS + 1, Glm53DsaStorage::BF16,).is_err()
        );
        assert!(Glm53ContextPlan::new(1, 0, Glm53DsaStorage::BF16).is_err());
        assert!(Glm53ContextPlan::new(0, 1, Glm53DsaStorage::BF16).is_err());
    }

    #[test]
    fn arithmetic_overflow_and_residency_edges_fail_closed() {
        assert!(
            Glm53ContextPlan::new(u32::MAX, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::BF16,)
                .is_err()
        );
        let plan =
            Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::FP8).unwrap();
        let exact = plan.admit_residency(plan.total_bytes);
        assert!(exact.fits);
        assert_eq!(exact.headroom_bytes, 0);
        assert_eq!(exact.shortfall_bytes, 0);
        let short = plan.admit_residency(plan.total_bytes - 1);
        assert!(!short.fits);
        assert_eq!(short.shortfall_bytes, 1);
        assert_eq!(short.headroom_bytes, 0);
    }
}
