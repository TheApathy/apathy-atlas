// SPDX-License-Identifier: AGPL-3.0-only

pub(crate) const MODULE: &str = "w4a16_gemv";
pub(crate) const K: usize = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Abi {
    Single,
    Qg,
    Qkvz,
    Dual,
    QgStrided,
    DualStrided,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Case {
    pub(crate) symbol: &'static str,
    pub(crate) group: usize,
    pub(crate) block: u32,
    pub(crate) rows: usize,
    pub(crate) abi: Abi,
    pub(crate) grid_z: u32,
}

impl Case {
    pub(crate) fn projections(self) -> usize {
        match self.abi {
            Abi::Dual | Abi::DualStrided => 2,
            _ => 1,
        }
    }

    pub(crate) fn stride(self) -> usize {
        self.group + 3
    }

    pub(crate) fn mapped_col(self, n: usize) -> usize {
        match self.abi {
            // num_heads=2, head_dim=1 maps [Q0,G0,Q1,G1] to [Q0,Q1,G0,G1].
            Abi::Qg | Abi::QgStrided => {
                let head = n / 2;
                if n.is_multiple_of(2) { head } else { 2 + head }
            }
            // (groups,kdim,vheads,vhdim)=(1,1,1,1) maps [Q,K,V,Z] explicitly.
            Abi::Qkvz | Abi::Single | Abi::Dual | Abi::DualStrided => n,
        }
    }

    pub(crate) fn output_index(self, launch_n: usize, row: usize, logical_n: usize) -> usize {
        let row_base = match self.abi {
            Abi::QgStrided | Abi::DualStrided => row * self.stride(),
            _ => row * launch_n,
        };
        row_base + self.mapped_col(logical_n)
    }

    pub(crate) fn output_words(self) -> usize {
        let occupied = match self.abi {
            Abi::QgStrided | Abi::DualStrided => (self.rows - 1) * self.stride() + self.group,
            _ => self.rows * self.group,
        };
        occupied + self.group // interior canary pad after the mapped output
    }
}

pub(crate) const CASES: &[Case] = &[
    Case {
        symbol: "w4a16_gemv_batch2",
        group: 4,
        block: 256,
        rows: 2,
        abi: Abi::Single,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_batch3",
        group: 4,
        block: 256,
        rows: 3,
        abi: Abi::Single,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_batch3_logits",
        group: 8,
        block: 256,
        rows: 3,
        abi: Abi::Single,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_qg",
        group: 4,
        block: 256,
        rows: 1,
        abi: Abi::Qg,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_qkvz",
        group: 4,
        block: 256,
        rows: 1,
        abi: Abi::Qkvz,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_qg_batch2",
        group: 4,
        block: 256,
        rows: 2,
        abi: Abi::Qg,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_qg_batch3",
        group: 4,
        block: 256,
        rows: 3,
        abi: Abi::Qg,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_dual_batch2",
        group: 4,
        block: 256,
        rows: 2,
        abi: Abi::Dual,
        grid_z: 2,
    },
    Case {
        symbol: "w4a16_gemv_dual_batch3",
        group: 4,
        block: 256,
        rows: 3,
        abi: Abi::Dual,
        grid_z: 2,
    },
    Case {
        symbol: "w4a16_gemv_dual_batch3_tuned",
        group: 4,
        block: 512,
        rows: 3,
        abi: Abi::Dual,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_qg_batch3_strided",
        group: 4,
        block: 256,
        rows: 3,
        abi: Abi::QgStrided,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_dual_batch3_strided",
        group: 4,
        block: 256,
        rows: 3,
        abi: Abi::DualStrided,
        grid_z: 2,
    },
    Case {
        symbol: "w4a16_gemv_v1",
        group: 2,
        block: 256,
        rows: 1,
        abi: Abi::Single,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_v3",
        group: 8,
        block: 256,
        rows: 1,
        abi: Abi::Single,
        grid_z: 1,
    },
    Case {
        symbol: "w4a16_gemv_v4",
        group: 2,
        block: 128,
        rows: 1,
        abi: Abi::Single,
        grid_z: 1,
    },
];
