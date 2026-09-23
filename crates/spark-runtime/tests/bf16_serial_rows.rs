// SPDX-License-Identifier: AGPL-3.0-only
//! CPU tests import the real, initially missing production helpers.
//! No CUDA/context construction or numerical emulation occurs here.

#[allow(dead_code)]
#[path = "../src/cublaslt/diagnostic_contract.rs"]
mod diagnostic_contract;
#[path = "../src/cublaslt/serial_rows_contract.rs"]
mod serial_rows_contract;
#[allow(dead_code)] // This test binary exercises the serial entrypoint only.
#[path = "../src/cublaslt/serial_rows_driver.rs"]
mod serial_rows_driver;
#[path = "../src/cublaslt/strided_rows_contract.rs"]
mod strided_rows_contract;

#[path = "bf16_serial_rows/contract.rs"]
mod contract_tests;
#[path = "bf16_serial_rows/driver.rs"]
mod driver_tests;
#[path = "bf16_serial_rows/recording.rs"]
mod recording;
