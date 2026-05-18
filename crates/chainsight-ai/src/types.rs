//! Core type definitions for the chainsight-ai anomaly detection engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Chain
// ---------------------------------------------------------------------------

/// Supported blockchain networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Chain {
    Ethereum,
    Solana,
    Mantle,
    Arbitrum,
    Optimism,
    Polygon,
    Avalanche,
    Bsc,
    Base,
}

impl Chain {
    /// Returns the chain identifier string used in transaction hashes / routing.
    pub fn chain_id(&self) -> &'static str {
        match self {
            Chain::Ethereum => "1",
            Chain::Solana => "solana",
            Chain::Mantle => "5000",
            Chain::Arbitrum => "42161",
            Chain::Optimism => "10",
            Chain::Polygon => "137",
            Chain::Avalanche => "43114",
            Chain::Bsc => "56",
            Chain::Base => "8453",
        }
    }

    /// Parse a chain-id string (case-insensitive).
    pub fn from_chain_id(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "1" | "ethereum" => Some(Chain::Ethereum),
            "solana" => Some(Chain::Solana),
            "5000" | "mantle" => Some(Chain::Mantle),
            "42161" | "arbitrum" => Some(Chain::Arbitrum),
            "10" | "optimism" => Some(Chain::Optimism),
            "137" | "polygon" => Some(Chain::Polygon),
            "43114" | "avalanche" => Some(Chain::Avalanche),
            "56" | "bsc" => Some(Chain::Bsc),
            "8453" | "base" => Some(Chain::Base),
            _ => None,
        }
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Chain::Ethereum => write!(f, "Ethereum"),
            Chain::Solana => write!(f, "Solana"),
            Chain::Mantle => write!(f, "Mantle"),
            Chain::Arbitrum => write!(f, "Arbitrum"),
            Chain::Optimism => write!(f, "Optimism"),
            Chain::Polygon => write!(f, "Polygon"),
            Chain::Avalanche => write!(f, "Avalanche"),
            Chain::Bsc => write!(f, "BSC"),
            Chain::Base => write!(f, "Base"),
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction & Block
// ---------------------------------------------------------------------------

/// A normalised on-chain transaction record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub chain: Chain,
    pub block_number: u64,
    pub from_address: String,
    pub to_address: String,
    pub value: f64,
    pub gas_used: f64,
    pub gas_price: f64,
    pub timestamp: DateTime<Utc>,
    pub token_symbol: Option<String>,
    pub memo: Option<String>,
}

/// A block-level aggregation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub number: u64,
    pub chain: Chain,
    pub timestamp: DateTime<Utc>,
    pub tx_count: usize,
    pub total_value: f64,
    pub gas_used_total: f64,
}

// ---------------------------------------------------------------------------
// Anomaly types & scores
// ---------------------------------------------------------------------------

/// The kind of anomaly detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Transaction value is far outside normal range.
    ValueOutlier,
    /// Sudden spike in transaction count.
    VolumeSpike,
    /// Gas price deviates significantly from the moving average.
    GasPriceDeviation,
    /// Sender address behaves like a known attack pattern.
    PatternMatch,
    /// Statistical test (e.g. KS / χ²) flagged a distribution shift.
    DistributionShift,
}

/// A numeric anomaly score in [0, 1] where higher is more anomalous.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AnomalyScore(pub f64);

impl AnomalyScore {
    pub fn new(score: f64) -> Self {
        Self(score.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn is_critical(&self) -> bool {
        self.0 >= 0.9
    }

    pub fn is_high(&self) -> bool {
        self.0 >= 0.7
    }

    pub fn is_medium(&self) -> bool {
        self.0 >= 0.4
    }

    pub fn is_low(&self) -> bool {
        self.0 < 0.4
    }
}

// ---------------------------------------------------------------------------
// Alert
// ---------------------------------------------------------------------------

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl AlertLevel {
    pub fn from_score(score: AnomalyScore) -> Self {
        if score.is_critical() {
            AlertLevel::Critical
        } else if score.is_high() {
            AlertLevel::High
        } else if score.is_medium() {
            AlertLevel::Medium
        } else {
            AlertLevel::Low
        }
    }
}

impl fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertLevel::Low => write!(f, "LOW"),
            AlertLevel::Medium => write!(f, "MEDIUM"),
            AlertLevel::High => write!(f, "HIGH"),
            AlertLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// An alert dispatched by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub chain: Chain,
    pub anomaly_type: AnomalyType,
    pub score: AnomalyScore,
    pub level: AlertLevel,
    pub message: String,
    pub tx_hash: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub fingerprint: String,
}

// ---------------------------------------------------------------------------
// Detection config
// ---------------------------------------------------------------------------

/// Tunable knobs for the anomaly detection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Z-score threshold for flagging outliers (default 3.0).
    pub zscore_threshold: f64,
    /// Window size for moving-average computations (default 100).
    pub moving_avg_window: usize,
    /// Multiplier for volume-spike detection (default 5.0).
    pub volume_spike_multiplier: f64,
    /// Minimum data points required before running statistical tests.
    pub min_data_points: usize,
    /// Whether to enable pattern-matching detector.
    pub enable_pattern_match: bool,
    /// Sliding window length for the time-series store.
    pub sliding_window_size: usize,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            zscore_threshold: 3.0,
            moving_avg_window: 100,
            volume_spike_multiplier: 5.0,
            min_data_points: 30,
            enable_pattern_match: true,
            sliding_window_size: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Time-series point
// ---------------------------------------------------------------------------

/// A single data point in a time-series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub volume: f64,
}

// ---------------------------------------------------------------------------
// OHLCV candle
// ---------------------------------------------------------------------------

/// Open-High-Low-Close-Volume candle used for aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvCandle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp_start: DateTime<Utc>,
    pub timestamp_end: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_chain_id() {
        assert_eq!(Chain::Ethereum.chain_id(), "1");
        assert_eq!(Chain::Solana.chain_id(), "solana");
        assert_eq!(Chain::Mantle.chain_id(), "5000");
    }

    #[test]
    fn test_chain_from_chain_id() {
        assert_eq!(Chain::from_chain_id("1"), Some(Chain::Ethereum));
        assert_eq!(Chain::from_chain_id("ETHEREUM"), Some(Chain::Ethereum));
        assert_eq!(Chain::from_chain_id("solana"), Some(Chain::Solana));
        assert_eq!(Chain::from_chain_id("5000"), Some(Chain::Mantle));
        assert_eq!(Chain::from_chain_id("unknown"), None);
    }

    #[test]
    fn test_chain_display() {
        assert_eq!(format!("{}", Chain::Ethereum), "Ethereum");
        assert_eq!(format!("{}", Chain::Bsc), "BSC");
    }

    #[test]
    fn test_anomaly_score_clamp() {
        let s = AnomalyScore::new(1.5);
        assert!((s.value() - 1.0).abs() < f64::EPSILON);
        let s2 = AnomalyScore::new(-0.5);
        assert!((s2.value() - 0.0).abs() < f64::EPSILON);
        let s3 = AnomalyScore::new(0.75);
        assert!((s3.value() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_anomaly_score_classifications() {
        assert!(AnomalyScore::new(0.95).is_critical());
        assert!(AnomalyScore::new(0.95).is_high());
        assert!(!AnomalyScore::new(0.95).is_low());

        assert!(!AnomalyScore::new(0.5).is_critical());
        assert!(!AnomalyScore::new(0.5).is_high());
        assert!(AnomalyScore::new(0.5).is_medium());
        assert!(!AnomalyScore::new(0.5).is_low());

        assert!(AnomalyScore::new(0.2).is_low());
    }

    #[test]
    fn test_alert_level_from_score() {
        assert_eq!(AlertLevel::from_score(AnomalyScore::new(0.95)), AlertLevel::Critical);
        assert_eq!(AlertLevel::from_score(AnomalyScore::new(0.75)), AlertLevel::High);
        assert_eq!(AlertLevel::from_score(AnomalyScore::new(0.5)), AlertLevel::Medium);
        assert_eq!(AlertLevel::from_score(AnomalyScore::new(0.2)), AlertLevel::Low);
    }

    #[test]
    fn test_alert_level_ordering() {
        assert!(AlertLevel::Critical > AlertLevel::High);
        assert!(AlertLevel::High > AlertLevel::Medium);
        assert!(AlertLevel::Medium > AlertLevel::Low);
    }

    #[test]
    fn test_detection_config_default() {
        let cfg = DetectionConfig::default();
        assert!((cfg.zscore_threshold - 3.0).abs() < f64::EPSILON);
        assert_eq!(cfg.moving_avg_window, 100);
        assert_eq!(cfg.min_data_points, 30);
    }

    #[test]
    fn test_time_series_point_construction() {
        let pt = TimeSeriesPoint {
            timestamp: Utc::now(),
            value: 42.0,
            volume: 100.0,
        };
        assert!((pt.value - 42.0).abs() < f64::EPSILON);
    }
}
