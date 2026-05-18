//! # courtvision-ai — Sports Analytics Engine
//!
//! A comprehensive Rust crate for sports analytics, game prediction,
//! player analysis, and betting value detection across multiple sports.

pub mod types;
pub mod stats;
pub mod predictions;
pub mod player;
pub mod betting;
pub mod pipeline;

// Re-exports for convenience
pub use types::{
    Sport, PlayerStats, TeamStats, GameEvent, GamePrediction,
    PlayerProjection, BettingLine, Matchup,
};
pub use stats::StatsEngine;
pub use predictions::{PredictionEngine, LineComparison};
pub use player::{PlayerAnalyzer, PlayerComparison, UsageEfficiencyResult};
pub use betting::{
    BettingAnalyzer, BetRecord, BetType, BetResult,
    BettingPerformance, ValueAssessment, LineMovement,
};
pub use pipeline::{CourtvisionPipeline, DailyReport, TopPlay};
