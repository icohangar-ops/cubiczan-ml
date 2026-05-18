//! SwarmFi Perps — Rust Port
//!
//! AI agent swarm intelligence platform for analyzing perpetual futures markets.
//! Nine specialized agents independently evaluate market conditions through
//! stigmergic coordination and adversarial weighted consensus to produce
//! LONG/SHORT/NEUTRAL trading signals.
//!
//! Ported from the TypeScript SwarmFi Perps web application.

pub mod types;
pub mod math;
pub mod agents;
pub mod consensus;
pub mod pipeline;
pub mod dydx;

pub use types::*;
pub use agents::*;
pub use consensus::*;
pub use pipeline::*;
