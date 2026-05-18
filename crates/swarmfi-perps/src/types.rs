//! Canonical data types for the SwarmFi Perps swarm engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trading signal produced by an agent or consensus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Signal {
    #[serde(rename = "LONG")]
    Long,
    #[serde(rename = "SHORT")]
    Short,
    #[serde(rename = "NEUTRAL")]
    Neutral,
}

impl Default for Signal {
    fn default() -> Self {
        Signal::Neutral
    }
}

impl Signal {
    pub fn as_str(&self) -> &'static str {
        match self {
            Signal::Long => "LONG",
            Signal::Short => "SHORT",
            Signal::Neutral => "NEUTRAL",
        }
    }
}

/// A vote from a single agent in the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVote {
    pub agent_type: String,
    pub signal: Signal,
    /// 0–100 confidence level.
    pub confidence: f64,
    pub reasoning: String,
}

/// Result of the adversarial weighted consensus across all agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub market: String,
    pub signal: Signal,
    /// 0–100 clamped confidence.
    pub confidence: f64,
    pub agent_votes: Vec<AgentVote>,
    /// Unix timestamp (milliseconds).
    pub timestamp: i64,
    #[serde(flatten)]
    pub stigmergy_board: StigmergyBoard,
}

/// Orderbook level (bid or ask).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookLevel {
    pub price: f64,
    pub size: f64,
}

/// Orderbook with bids and asks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orderbook {
    pub bids: Vec<OrderbookLevel>,
    pub asks: Vec<OrderbookLevel>,
}

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub side: TradeSide,
    pub size: f64,
    pub price: f64,
    /// Unix timestamp (seconds).
    pub created_at: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    #[serde(rename = "BUY")]
    Buy,
    #[serde(rename = "SELL")]
    Sell,
}

/// OHLCV candle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub started_at: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub base_token_volume: f64,
    pub usd_volume: f64,
    pub trades: u32,
}

/// Historical funding rate entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingEntry {
    pub rate: String,
    pub effective_at: String,
    pub price: String,
}

/// Market metadata from dYdX.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketInfo {
    pub ticker: String,
    pub oracle_price: String,
    pub open_interest: String,
    pub volume_24h: String,
    pub next_funding_time: String,
}

/// Pre-computed market statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketStats {
    pub mid_price: f64,
    pub spread: f64,
    pub volume_24h: f64,
    pub open_interest: f64,
    /// Annualized percentage (1h rate × 24 × 365 × 100).
    pub funding_rate_1h: f64,
}

/// The complete market data bundle passed to every agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataBundle {
    pub orderbook: Option<Orderbook>,
    pub trades: Vec<Trade>,
    pub candles: Vec<Candle>,
    pub funding: Vec<FundingEntry>,
    pub market: Option<MarketInfo>,
    pub stats: MarketStats,
}

/// Agent weight configuration for consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentWeight {
    pub agent_type: String,
    pub weight: f64,
    pub description: String,
}

/// Volatility regime level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VolatilityRegime {
    #[serde(rename = "LOW")]
    Low,
    #[default]
    #[serde(rename = "NORMAL")]
    Normal,
    #[serde(rename = "HIGH")]
    High,
}

/// Liquidation risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RiskLevel {
    #[default]
    #[serde(rename = "LOW")]
    Low,
    #[serde(rename = "MEDIUM")]
    Medium,
    #[serde(rename = "HIGH")]
    High,
}

/// Signal count breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SignalCounts {
    #[serde(rename = "LONG")]
    pub long: u32,
    #[serde(rename = "SHORT")]
    pub short: u32,
    #[serde(rename = "NEUTRAL")]
    pub neutral: u32,
}

/// Shared stigmergy board — agents leave traces here between runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StigmergyBoard {
    #[serde(default)]
    pub last_signals: HashMap<String, String>,
    #[serde(default)]
    pub signal_counts: SignalCounts,
    pub average_confidence: f64,
    #[serde(default)]
    pub last_updated: i64,
    #[serde(default)]
    pub liquidation_risk_level: RiskLevel,
    #[serde(default)]
    pub volatility_regime: VolatilityRegime,
    /// Signals from the previous run (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_signals: Option<HashMap<String, String>>,
}

/// Database record for a consensus signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusSignalRecord {
    pub id: String,
    pub market: String,
    pub signal: String,
    pub confidence: f64,
    pub timestamp: String,
    pub agent_votes_json: String,
    pub market_data_json: String,
}

/// Database record for an agent state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateRecord {
    pub id: String,
    pub agent_type: String,
    pub market: String,
    pub last_vote: String,
    pub score: f64,
    pub timestamp: String,
}

/// Database record for a market snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotRecord {
    pub id: String,
    pub market: String,
    pub mid_price: f64,
    pub spread: f64,
    pub volume_24h: f64,
    pub open_interest: f64,
    pub funding_1h: f64,
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_serde_round_trip() {
        let signal = Signal::Long;
        let json = serde_json::to_string(&signal).unwrap();
        assert_eq!(json, "\"LONG\"");
        let restored: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, Signal::Long);
    }

    #[test]
    fn test_agent_vote_serde() {
        let vote = AgentVote {
            agent_type: "FundingAgent".into(),
            signal: Signal::Short,
            confidence: 75.0,
            reasoning: "High positive funding".into(),
        };
        let json = serde_json::to_string(&vote).unwrap();
        let restored: AgentVote = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.agent_type, "FundingAgent");
        assert_eq!(restored.confidence, 75.0);
    }

    #[test]
    fn test_market_data_bundle_default_stats() {
        let bundle = MarketDataBundle {
            orderbook: None,
            trades: vec![],
            candles: vec![],
            funding: vec![],
            market: None,
            stats: MarketStats::default(),
        };
        assert_eq!(bundle.stats.mid_price, 0.0);
    }

    #[test]
    fn test_stigmergy_board_default() {
        let board = StigmergyBoard::default();
        assert_eq!(board.volatility_regime, VolatilityRegime::Normal);
        assert_eq!(board.liquidation_risk_level, RiskLevel::Low);
    }
}
