//! SwarmFi Perps — Rust Port
//!
//! AI agent swarm intelligence platform for analyzing perpetual futures markets.
//! Nine specialized agents independently evaluate market conditions through
//! stigmergic coordination and adversarial weighted consensus to produce
//! LONG/SHORT/NEUTRAL trading signals.
//!
//! # Modules
//!
//! - **types**: Canonical data types (Signal, MarketDataBundle, etc.)
//! - **math**: Mathematical utilities for price/stat computation
//! - **agents**: Nine specialized market analysis agents
//! - **consensus**: Adversarial weighted consensus engine
//! - **pipeline**: End-to-end analysis pipeline and mock data generators
//! - **dydx**: dYdX v4 Indexer API client
//! - **websocket**: Real-time WebSocket integration (dYdX v4 WS)
//! - **arbitrage**: Cross-exchange arbitrage (dYdX, GMX, Synthetix)
//! - **backtest**: Historical signal performance backtesting engine
//! - **alerts**: Telegram/Discord webhook alert dispatch system
//! - **vault**: MegaVault PnL tracking and yield analytics
//! - **compliance**: Regulatory risk scoring and pre-trade screening
//! - **mobile**: REST API response types for React Native companion
//!
//! Ported from the TypeScript SwarmFi Perps web application.

pub mod types;
pub mod math;
pub mod agents;
pub mod consensus;
pub mod pipeline;
pub mod dydx;
pub mod websocket;
pub mod arbitrage;
pub mod backtest;
pub mod alerts;
pub mod vault;
pub mod compliance;
pub mod mobile;

pub use types::*;
pub use agents::*;
pub use consensus::*;
pub use pipeline::*;
