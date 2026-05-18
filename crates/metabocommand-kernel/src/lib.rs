//! # metabocommand-kernel
//!
//! Pure computation kernels for metabolic commerce platform.
//! Handles escalation management, velocity scoring, and CSV generation.
//! Designed for WASM target — no I/O, no async, pure functions only.

pub mod types;
pub mod escalation;
pub mod velocity;
pub mod csv_builder;

pub use types::*;
pub use escalation::*;
pub use velocity::*;
pub use csv_builder::CsvBuilder;
