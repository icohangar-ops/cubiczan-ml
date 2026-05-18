//! CritMin Oracle — Off-Chain Risk Pipeline (Rust Port)
//!
//! AI-powered critical minerals supply chain risk scoring pipeline.
//!
//! This crate implements:
//!   1. Commodity price fetching (Alpha Vantage API)
//!   2. Macro data fetching (FRED API)
//!   3. Risk score computation (sentiment, regulatory, price forecast)
//!   4. On-chain scaling and keccak256 hashing
//!   5. Full pipeline orchestration (demo + live modes)

mod config;
mod scaling;
mod sentiment;
mod forecast;
mod prices;
mod macro_data;
mod pipeline;

pub use config::*;
pub use scaling::*;
pub use sentiment::*;
pub use forecast::*;
pub use prices::*;
pub use macro_data::*;
pub use pipeline::*;
