//! # minescope-kernel
//!
//! Pure computation kernels for critical mineral supply chain intelligence.
//! Designed for WASM target — no I/O, no async, pure functions only.

pub mod types;
pub mod prospectivity;
pub mod risk;
pub mod pricing;

pub use types::*;
pub use prospectivity::*;
pub use risk::*;
pub use pricing::*;
