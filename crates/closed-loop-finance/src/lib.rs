//! # Closed-Loop Finance
//!
//! Autonomous closed-loop finance system implementing the Observe-Decide-Execute-Learn
//! cycle with PID control, risk management, and adaptive learning.
//!
//! ## Modules
//! - [`types`] — Core types: phases, regimes, actions, signals, portfolio state
//! - [`observer`] — Market observation: regime detection, volatility classification, trend estimation
//! - [`decider`] — Decision engine: multi-factor scoring, position sizing, risk assessment
//! - [`executor`] — Execution engine: order simulation with slippage, fees, partial fills
//! - [`learner`] — Learning engine: outcome tracking, pattern analysis, parameter adaptation
//! - [`controller`] — Control loop: PID feedback control for risk, leverage, exposure, drawdown
//! - [`risk_manager`] — Integrated risk: VaR/CVaR, drawdown control, circuit breaker, risk budgeting
//! - [`pipeline`] — Pipeline orchestration: full O-D-E-L cycle coordination and simulation

pub mod types;
pub mod observer;
pub mod decider;
pub mod executor;
pub mod learner;
pub mod controller;
pub mod risk_manager;
pub mod pipeline;

// Re-export all core types for convenience
pub use types::*;
