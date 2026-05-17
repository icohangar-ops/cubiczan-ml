//! # Trading Signal Types
//!
//! Data structures and utilities for generating, combining, and evaluating
//! trading signals in the CubicZan ML ecosystem.
//!
//! ## Components
//!
//! - **Signal** — Core signal with direction, confidence, timestamp, and source
//! - **SignalStrength** — Enum classifying signal conviction (Weak → Very Strong)
//! - **SignalAggregator** — Combine multiple signals into a single consensus
//! - **Consensus methods** — Weighted average, median, Borda count
//! - **SignalHistory** — Track and evaluate historical signal performance
//! - **Conflict detection** — Identify and resolve opposing signals

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during signal operations.
#[derive(Debug, Error)]
pub enum SignalError {
    #[error("no signals provided for aggregation")]
    NoSignals,
    #[error("signal conflict: {reason}")]
    Conflict { reason: String },
    #[error("invalid confidence: must be in [0, 1], got {value}")]
    InvalidConfidence { value: f64 },
    #[error("insufficient history for evaluation")]
    InsufficientHistory,
}

// ---------------------------------------------------------------------------
// Signal Direction
// ---------------------------------------------------------------------------

/// Direction of a trading signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum SignalDirection {
    /// Buy / go long.
    Buy,
    /// Sell / go short.
    Sell,
    /// Hold / no position change.
    Hold,
}

impl SignalDirection {
    /// Convert to a numeric score: Buy = +1, Hold = 0, Sell = -1.
    pub fn to_score(self) -> f64 {
        match self {
            SignalDirection::Buy => 1.0,
            SignalDirection::Hold => 0.0,
            SignalDirection::Sell => -1.0,
        }
    }

    /// Create from a numeric score.
    pub fn from_score(score: f64) -> Self {
        if score > 0.1 {
            SignalDirection::Buy
        } else if score < -0.1 {
            SignalDirection::Sell
        } else {
            SignalDirection::Hold
        }
    }

    /// Whether this direction opposes another.
    pub fn opposes(self, other: SignalDirection) -> bool {
        matches!(
            (self, other),
            (SignalDirection::Buy, SignalDirection::Sell)
                | (SignalDirection::Sell, SignalDirection::Buy)
        )
    }
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDirection::Buy => write!(f, "BUY"),
            SignalDirection::Sell => write!(f, "SELL"),
            SignalDirection::Hold => write!(f, "HOLD"),
        }
    }
}

// ---------------------------------------------------------------------------
// Signal Strength
// ---------------------------------------------------------------------------

/// Classification of signal conviction / strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignalStrength {
    /// Very weak signal; noise-like, not actionable alone.
    VeryWeak = 1,
    /// Weak signal; marginal conviction.
    Weak = 2,
    /// Moderate signal; reasonable conviction.
    Moderate = 3,
    /// Strong signal; high conviction.
    Strong = 4,
    /// Very strong signal; very high conviction.
    VeryStrong = 5,
}

impl SignalStrength {
    /// Create from a confidence value in [0, 1].
    pub fn from_confidence(confidence: f64) -> Self {
        match confidence {
            c if c < 0.2 => SignalStrength::VeryWeak,
            c if c < 0.4 => SignalStrength::Weak,
            c if c < 0.6 => SignalStrength::Moderate,
            c if c <= 0.8 => SignalStrength::Strong,
            _ => SignalStrength::VeryStrong,
        }
    }

    /// Convert to a numeric weight for aggregation.
    pub fn to_weight(self) -> f64 {
        match self {
            SignalStrength::VeryWeak => 0.2,
            SignalStrength::Weak => 0.4,
            SignalStrength::Moderate => 0.6,
            SignalStrength::Strong => 0.8,
            SignalStrength::VeryStrong => 1.0,
        }
    }

    /// Minimum confidence threshold for this strength.
    pub fn min_confidence(self) -> f64 {
        match self {
            SignalStrength::VeryWeak => 0.0,
            SignalStrength::Weak => 0.2,
            SignalStrength::Moderate => 0.4,
            SignalStrength::Strong => 0.6,
            SignalStrength::VeryStrong => 0.8,
        }
    }
}

impl std::fmt::Display for SignalStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalStrength::VeryWeak => write!(f, "VeryWeak"),
            SignalStrength::Weak => write!(f, "Weak"),
            SignalStrength::Moderate => write!(f, "Moderate"),
            SignalStrength::Strong => write!(f, "Strong"),
            SignalStrength::VeryStrong => write!(f, "VeryStrong"),
        }
    }
}

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

/// A single trading signal from a specific source/strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Unique identifier for the signal.
    pub id: String,
    /// Trading direction (Buy, Sell, or Hold).
    pub direction: SignalDirection,
    /// Confidence level in [0.0, 1.0].
    pub confidence: f64,
    /// Signal strength classification.
    pub strength: SignalStrength,
    /// Timestamp when the signal was generated.
    pub timestamp: DateTime<Utc>,
    /// Source strategy or model that produced this signal.
    pub source: String,
    /// Optional target asset/symbol.
    pub symbol: Option<String>,
    /// Optional expected price target.
    pub price_target: Option<f64>,
    /// Optional stop-loss price.
    pub stop_loss: Option<f64>,
    /// Optional metadata (model features, explanation, etc.).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Signal {
    /// Create a new signal with auto-generated ID and derived strength.
    pub fn new(
        direction: SignalDirection,
        confidence: f64,
        timestamp: DateTime<Utc>,
        source: impl Into<String>,
    ) -> Result<Self, SignalError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(SignalError::InvalidConfidence { value: confidence });
        }

        let source_str = source.into();
        let id = format!("sig_{}_{}", source_str.replace(' ', "_"), timestamp.timestamp_millis());

        Ok(Signal {
            id,
            direction,
            confidence,
            strength: SignalStrength::from_confidence(confidence),
            timestamp,
            source: source_str,
            symbol: None,
            price_target: None,
            stop_loss: None,
            metadata: HashMap::new(),
        })
    }

    /// Create a signal with symbol.
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Create a signal with price target and stop-loss.
    pub fn with_targets(mut self, price_target: f64, stop_loss: f64) -> Self {
        self.price_target = Some(price_target);
        self.stop_loss = Some(stop_loss);
        self
    }

    /// Add metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Weighted score: direction × confidence.
    pub fn weighted_score(&self) -> f64 {
        self.direction.to_score() * self.confidence
    }

    /// Whether this signal is actionable (not Hold).
    pub fn is_actionable(&self) -> bool {
        self.direction != SignalDirection::Hold && self.confidence >= 0.3
    }

    /// Check if this signal conflicts with another.
    pub fn conflicts_with(&self, other: &Signal) -> bool {
        self.direction.opposes(other.direction)
            && self.confidence > 0.3
            && other.confidence > 0.3
    }
}

// ---------------------------------------------------------------------------
// Consensus Methods
// ---------------------------------------------------------------------------

/// Method for aggregating multiple signals into a consensus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusMethod {
    /// Weighted average of signal scores, weighted by confidence.
    WeightedAverage,
    /// Median of signal scores (robust to outliers).
    Median,
    /// Borda count: rank signals by confidence, sum direction-adjusted ranks.
    BordaCount,
    /// Majority vote: direction with most signals wins.
    MajorityVote,
    /// Best signal: use only the highest-confidence signal.
    BestSignal,
}

// ---------------------------------------------------------------------------
// Signal Aggregator
// ---------------------------------------------------------------------------

/// Aggregates multiple signals into a single consensus signal.
pub struct SignalAggregator {
    /// Method used for consensus.
    pub method: ConsensusMethod,
    /// Minimum number of signals required for aggregation.
    pub min_signals: usize,
    /// Minimum average confidence for the consensus to be actionable.
    pub confidence_threshold: f64,
    /// Conflict resolution strategy: when true, opposing signals cancel out.
    pub cancel_opposing: bool,
    /// Custom per-source weights (source_name → weight).
    pub source_weights: HashMap<String, f64>,
}

impl SignalAggregator {
    /// Create a new aggregator with default settings.
    pub fn new(method: ConsensusMethod) -> Self {
        SignalAggregator {
            method,
            min_signals: 1,
            confidence_threshold: 0.3,
            cancel_opposing: false,
            source_weights: HashMap::new(),
        }
    }

    /// Builder: set minimum signals required.
    pub fn with_min_signals(mut self, n: usize) -> Self {
        self.min_signals = n;
        self
    }

    /// Builder: set confidence threshold.
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Builder: enable/disable opposing signal cancellation.
    pub fn with_cancel_opposing(mut self, cancel: bool) -> Self {
        self.cancel_opposing = cancel;
        self
    }

    /// Builder: set source weights.
    pub fn with_source_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.source_weights = weights;
        self
    }

    /// Aggregate multiple signals into a single consensus signal.
    pub fn aggregate(&self, signals: &[Signal]) -> Result<Signal, SignalError> {
        if signals.is_empty() {
            return Err(SignalError::NoSignals);
        }

        if signals.len() < self.min_signals {
            return Err(SignalError::NoSignals);
        }

        let processed = if self.cancel_opposing {
            self.cancel_opposing_signals(signals)
        } else {
            signals.to_vec()
        };

        if processed.is_empty() {
            return Err(SignalError::Conflict {
                reason: "all signals cancelled out".into(),
            });
        }

        let (consensus_score, consensus_confidence) = match self.method {
            ConsensusMethod::WeightedAverage => self.weighted_average(&processed),
            ConsensusMethod::Median => self.median(&processed),
            ConsensusMethod::BordaCount => self.borda_count(&processed),
            ConsensusMethod::MajorityVote => self.majority_vote(&processed),
            ConsensusMethod::BestSignal => self.best_signal(&processed),
        };

        let direction = SignalDirection::from_score(consensus_score);
        let timestamp = processed
            .iter()
            .max_by_key(|s| s.timestamp)
            .map(|s| s.timestamp)
            .unwrap_or_else(Utc::now);

        Signal::new(direction, consensus_confidence, timestamp, "consensus")
    }

    /// Cancel opposing signals: for each Buy/Sell pair, remove the weaker one.
    fn cancel_opposing_signals(&self, signals: &[Signal]) -> Vec<Signal> {
        let mut buys: Vec<&Signal> = signals.iter().filter(|s| s.direction == SignalDirection::Buy).collect();
        let mut sells: Vec<&Signal> = signals.iter().filter(|s| s.direction == SignalDirection::Sell).collect();

        // Sort by confidence descending
        buys.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        sells.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Cancel weaker opposing signals
        let mut to_remove = std::collections::HashSet::new();
        let pairs = buys.len().min(sells.len());
        for i in 0..pairs {
            // Keep the stronger, remove the weaker
            if buys[i].confidence >= sells[i].confidence {
                to_remove.insert(sells[i].id.clone());
            } else {
                to_remove.insert(buys[i].id.clone());
            }
        }

        signals.iter().filter(|s| !to_remove.contains(&s.id)).cloned().collect()
    }

    /// Weighted average of signal scores.
    fn weighted_average(&self, signals: &[Signal]) -> (f64, f64) {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        let mut conf_weighted_sum = 0.0;

        for signal in signals {
            let weight = self
                .source_weights
                .get(&signal.source)
                .copied()
                .unwrap_or(signal.confidence);
            total_weight += weight;
            weighted_sum += signal.weighted_score() * weight;
            conf_weighted_sum += signal.confidence * weight;
        }

        if total_weight.abs() < 1e-15 {
            return (0.0, 0.0);
        }

        (weighted_sum / total_weight, conf_weighted_sum / total_weight)
    }

    /// Median of signal scores.
    fn median(&self, signals: &[Signal]) -> (f64, f64) {
        let mut scores: Vec<f64> = signals.iter().map(|s| s.weighted_score()).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_score = if scores.is_empty() {
            0.0
        } else if scores.len() % 2 == 1 {
            scores[scores.len() / 2]
        } else {
            (scores[scores.len() / 2 - 1] + scores[scores.len() / 2]) / 2.0
        };

        let median_conf = {
            let mut confs: Vec<f64> = signals.iter().map(|s| s.confidence).collect();
            confs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if confs.is_empty() {
                0.0
            } else if confs.len() % 2 == 1 {
                confs[confs.len() / 2]
            } else {
                (confs[confs.len() / 2 - 1] + confs[confs.len() / 2]) / 2.0
            }
        };

        (median_score, median_conf)
    }

    /// Borda count: assign ranks based on confidence, convert direction to score.
    fn borda_count(&self, signals: &[Signal]) -> (f64, f64) {
        let n = signals.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Rank signals by confidence (higher confidence = higher rank)
        let mut ranked: Vec<(usize, &Signal)> = signals.iter().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.confidence
                .partial_cmp(&a.1.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut total_score = 0.0;
        let mut total_conf = 0.0;

        for (rank, signal) in ranked.iter() {
            // Borda points: rank 0 gets n-1 points, rank n-1 gets 0 points
            let points = (n - 1 - *rank) as f64;
            total_score += signal.direction.to_score() * points;
            total_conf += signal.confidence * points;
        }

        let max_points = (n * (n - 1)) as f64 / 2.0;
        let normalized_score = if max_points > 0.0 { total_score / max_points } else { 0.0 };
        let normalized_conf = if max_points > 0.0 { total_conf / max_points } else { 0.0 };

        (normalized_score, normalized_conf.clamp(0.0, 1.0))
    }

    /// Majority vote: direction with most signals wins.
    fn majority_vote(&self, signals: &[Signal]) -> (f64, f64) {
        let mut buy_count = 0usize;
        let mut sell_count = 0usize;
        let mut hold_count = 0usize;
        let mut buy_conf = 0.0_f64;
        let mut sell_conf = 0.0_f64;
        let mut hold_conf = 0.0_f64;

        for signal in signals {
            match signal.direction {
                SignalDirection::Buy => {
                    buy_count += 1;
                    buy_conf += signal.confidence;
                }
                SignalDirection::Sell => {
                    sell_count += 1;
                    sell_conf += signal.confidence;
                }
                SignalDirection::Hold => {
                    hold_count += 1;
                    hold_conf += signal.confidence;
                }
            }
        }

        let (direction, confidence) = if buy_count > sell_count && buy_count > hold_count {
            (SignalDirection::Buy, buy_conf / buy_count.max(1) as f64)
        } else if sell_count > buy_count && sell_count > hold_count {
            (SignalDirection::Sell, sell_conf / sell_count.max(1) as f64)
        } else if hold_count > buy_count && hold_count > hold_count {
            (SignalDirection::Hold, hold_conf / hold_count.max(1) as f64)
        } else {
            // Tie: use confidence-weighted
            if buy_conf >= sell_conf && buy_conf >= hold_conf {
                (SignalDirection::Buy, buy_conf / buy_count.max(1) as f64)
            } else if sell_conf >= buy_conf {
                (SignalDirection::Sell, sell_conf / sell_count.max(1) as f64)
            } else {
                (SignalDirection::Hold, hold_conf / hold_count.max(1) as f64)
            }
        };

        (direction.to_score(), confidence)
    }

    /// Best signal: use only the highest-confidence signal.
    fn best_signal(&self, signals: &[Signal]) -> (f64, f64) {
        signals
            .iter()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| (s.weighted_score(), s.confidence))
            .unwrap_or((0.0, 0.0))
    }
}

// ---------------------------------------------------------------------------
// Signal History & Performance Tracking
// ---------------------------------------------------------------------------

/// Record of a signal and its outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRecord {
    /// The signal that was generated.
    pub signal: Signal,
    /// Price at the time of signal generation.
    pub entry_price: f64,
    /// Outcome price (exit price).
    pub exit_price: Option<f64>,
    /// Profit/loss percentage.
    pub pnl_pct: Option<f64>,
    /// Whether the signal was profitable.
    pub profitable: Option<bool>,
    /// Duration of the trade in seconds.
    pub duration_secs: Option<i64>,
    /// Whether the signal was correct in direction (regardless of magnitude).
    pub direction_correct: Option<bool>,
}

/// Tracks signal history and computes performance metrics.
pub struct SignalHistory {
    /// All signal records.
    pub records: Vec<SignalRecord>,
}

impl SignalHistory {
    /// Create an empty signal history.
    pub fn new() -> Self {
        SignalHistory {
            records: Vec::new(),
        }
    }

    /// Record a new signal.
    pub fn record_signal(&mut self, signal: Signal, entry_price: f64) {
        self.records.push(SignalRecord {
            signal,
            entry_price,
            exit_price: None,
            pnl_pct: None,
            profitable: None,
            duration_secs: None,
            direction_correct: None,
        });
    }

    /// Close a signal with the outcome.
    ///
    /// Finds the most recent open signal from the given source and closes it.
    pub fn close_signal(
        &mut self,
        source: &str,
        exit_price: f64,
        exit_time: DateTime<Utc>,
    ) -> Result<(), SignalError> {
        // Find the most recent open signal from the given source
        let idx = self
            .records
            .iter()
            .rposition(|r| r.signal.source == source && r.exit_price.is_none())
            .ok_or(SignalError::InsufficientHistory)?;

        let record = &mut self.records[idx];
        record.exit_price = Some(exit_price);

        if record.entry_price.abs() < 1e-15 {
            record.pnl_pct = Some(0.0);
            record.profitable = Some(false);
        } else {
            let pnl = (exit_price - record.entry_price) / record.entry_price;
            record.pnl_pct = Some(pnl);
            record.direction_correct = Some(match record.signal.direction {
                SignalDirection::Buy => exit_price > record.entry_price,
                SignalDirection::Sell => exit_price < record.entry_price,
                SignalDirection::Hold => true,
            });
            record.profitable = Some(record.direction_correct.unwrap_or(false));
        }

        record.duration_secs = Some(
            exit_time
                .timestamp()
                .saturating_sub(record.signal.timestamp.timestamp()),
        );

        Ok(())
    }

    /// Compute performance metrics over the history.
    pub fn performance(&self) -> SignalPerformance {
        let total = self.records.len();
        let closed: Vec<&SignalRecord> = self.records.iter().filter(|r| r.exit_price.is_some()).collect();
        let closed_count = closed.len();

        if closed_count == 0 {
            return SignalPerformance {
                total_signals: total,
                closed_signals: 0,
                win_rate: 0.0,
                avg_pnl: 0.0,
                total_pnl: 0.0,
                best_trade: 0.0,
                worst_trade: 0.0,
                avg_duration_secs: 0.0,
                direction_accuracy: 0.0,
                profit_factor: 0.0,
                avg_confidence: 0.0,
                confidence_calibration: 0.0,
                buy_signals: 0,
                sell_signals: 0,
                hold_signals: 0,
            };
        }

        let profitable_count = closed.iter().filter(|r| r.profitable == Some(true)).count();
        let win_rate = profitable_count as f64 / closed_count as f64;

        let pnls: Vec<f64> = closed
            .iter()
            .map(|r| r.pnl_pct.unwrap_or(0.0))
            .collect();

        let total_pnl: f64 = pnls.iter().sum();
        let avg_pnl = total_pnl / closed_count as f64;
        let best_trade = pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst_trade = pnls.iter().cloned().fold(f64::INFINITY, f64::min);

        let avg_duration: f64 = closed
            .iter()
            .map(|r| r.duration_secs.unwrap_or(0) as f64)
            .sum::<f64>()
            / closed_count as f64;

        let dir_correct = closed.iter().filter(|r| r.direction_correct == Some(true)).count();
        let direction_accuracy = dir_correct as f64 / closed_count as f64;

        // Profit factor: gross profits / gross losses
        let gross_profit: f64 = pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
        let profit_factor = if gross_loss.abs() < 1e-15 {
            f64::INFINITY
        } else {
            gross_profit / gross_loss
        };

        // Average confidence
        let avg_confidence: f64 = self.records.iter().map(|r| r.signal.confidence).sum::<f64>()
            / total.max(1) as f64;

        // Confidence calibration: correlation between confidence and profitability
        let confidence_calibration = self.compute_calibration();

        // Direction counts
        let buy_signals = self.records.iter().filter(|r| r.signal.direction == SignalDirection::Buy).count();
        let sell_signals = self.records.iter().filter(|r| r.signal.direction == SignalDirection::Sell).count();
        let hold_signals = self.records.iter().filter(|r| r.signal.direction == SignalDirection::Hold).count();

        SignalPerformance {
            total_signals: total,
            closed_signals: closed_count,
            win_rate,
            avg_pnl,
            total_pnl,
            best_trade,
            worst_trade,
            avg_duration_secs: avg_duration,
            direction_accuracy,
            profit_factor,
            avg_confidence,
            confidence_calibration,
            buy_signals,
            sell_signals,
            hold_signals,
        }
    }

    /// Compute confidence calibration: how well predicted confidence matches actual accuracy.
    ///
    /// Groups signals into confidence buckets and compares predicted vs actual win rates.
    /// Returns a value between 0 and 1, where 1.0 means perfect calibration.
    fn compute_calibration(&self) -> f64 {
        let closed: Vec<&SignalRecord> = self.records.iter().filter(|r| r.exit_price.is_some()).collect();
        if closed.len() < 10 {
            return 0.0;
        }

        // Group into 5 buckets: [0, 0.2), [0.2, 0.4), ..., [0.8, 1.0]
        let mut buckets: Vec<(f64, f64, usize, usize)> = vec![(0.0, 0.0, 0, 0); 5]; // (pred_rate, actual_rate, total, wins)

        for record in closed.iter() {
            let bucket_idx = (record.signal.confidence * 5.0).floor() as usize;
            let bucket_idx = bucket_idx.min(4);

            buckets[bucket_idx].0 = (bucket_idx as f64 + 0.5) / 5.0; // predicted rate
            buckets[bucket_idx].2 += 1;
            if record.profitable == Some(true) {
                buckets[bucket_idx].3 += 1;
            }
        }

        // Compute actual rates and mean absolute error
        let mut mae_sum = 0.0;
        let mut bucket_count = 0;
        for (pred, actual_rate, total, wins) in buckets.iter() {
            if *total > 0 {
                let actual = *wins as f64 / *total as f64;
                mae_sum += (pred - actual).abs();
                bucket_count += 1;
            }
        }

        if bucket_count == 0 {
            return 0.0;
        }

        // Convert MAE to calibration score: 1 - MAE (clamped to [0, 1])
        let mae = mae_sum / bucket_count as f64;
        (1.0 - mae).clamp(0.0, 1.0)
    }
}

impl Default for SignalHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Signal Performance Metrics
// ---------------------------------------------------------------------------

/// Aggregated performance metrics for a signal history.
#[derive(Debug, Clone)]
pub struct SignalPerformance {
    /// Total number of signals generated.
    pub total_signals: usize,
    /// Number of signals that have been closed (have exit price).
    pub closed_signals: usize,
    /// Win rate (percentage of profitable closed trades).
    pub win_rate: f64,
    /// Average P&L per closed trade.
    pub avg_pnl: f64,
    /// Total P&L across all closed trades.
    pub total_pnl: f64,
    /// Best single trade P&L.
    pub best_trade: f64,
    /// Worst single trade P&L.
    pub worst_trade: f64,
    /// Average trade duration in seconds.
    pub avg_duration_secs: f64,
    /// Percentage of signals where the direction was correct.
    pub direction_accuracy: f64,
    /// Profit factor (gross profit / gross loss).
    pub profit_factor: f64,
    /// Average confidence across all signals.
    pub avg_confidence: f64,
    /// Confidence calibration score (0 to 1).
    pub confidence_calibration: f64,
    /// Number of buy signals.
    pub buy_signals: usize,
    /// Number of sell signals.
    pub sell_signals: usize,
    /// Number of hold signals.
    pub hold_signals: usize,
}

// ---------------------------------------------------------------------------
// Conflict Detection
// ---------------------------------------------------------------------------

/// Result of signal conflict analysis.
#[derive(Debug, Clone)]
pub struct ConflictAnalysis {
    /// List of detected conflicts.
    pub conflicts: Vec<SignalConflict>,
    /// Total number of conflicts found.
    pub conflict_count: usize,
    /// Whether any unresolvable conflicts exist.
    pub has_unresolvable: bool,
    /// Recommended resolution strategy.
    pub resolution: ConflictResolution,
}

/// A single signal conflict between two opposing signals.
#[derive(Debug, Clone)]
pub struct SignalConflict {
    /// First signal in the conflict.
    pub signal_a: Signal,
    /// Second signal in the conflict.
    pub signal_b: Signal,
    /// Strength of the conflict (0.0 to 1.0), higher = more severe.
    pub severity: f64,
}

/// Recommended conflict resolution.
#[derive(Debug, Clone)]
pub enum ConflictResolution {
    /// Use the signal with higher confidence.
    UseHigherConfidence,
    /// Use weighted average of both signals.
    AverageSignals,
    /// Hold — don't trade when signals conflict.
    HoldPosition,
    /// No conflicts detected.
    NoConflict,
}

/// Detect and analyze conflicts between signals.
pub fn detect_conflicts(signals: &[Signal]) -> ConflictAnalysis {
    let mut conflicts = Vec::new();

    for i in 0..signals.len() {
        for j in (i + 1)..signals.len() {
            if signals[i].conflicts_with(&signals[j]) {
                // Severity based on both confidences
                let severity = signals[i].confidence * signals[j].confidence;
                conflicts.push(SignalConflict {
                    signal_a: signals[i].clone(),
                    signal_b: signals[j].clone(),
                    severity,
                });
            }
        }
    }

    let has_unresolvable = conflicts
        .iter()
        .any(|c| (c.signal_a.confidence - c.signal_b.confidence).abs() < 0.1);

    let resolution = if conflicts.is_empty() {
        ConflictResolution::NoConflict
    } else if has_unresolvable {
        ConflictResolution::HoldPosition
    } else if conflicts.iter().any(|c| c.severity > 0.5) {
        ConflictResolution::AverageSignals
    } else {
        ConflictResolution::UseHigherConfidence
    };

    ConflictAnalysis {
        conflict_count: conflicts.len(),
        has_unresolvable,
        resolution,
        conflicts,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(direction: SignalDirection, confidence: f64, source: &str) -> Signal {
        Signal::new(direction, confidence, Utc::now(), source).unwrap()
    }

    #[test]
    fn test_signal_direction() {
        assert_eq!(SignalDirection::Buy.to_score(), 1.0);
        assert_eq!(SignalDirection::Hold.to_score(), 0.0);
        assert_eq!(SignalDirection::Sell.to_score(), -1.0);
    }

    #[test]
    fn test_signal_direction_from_score() {
        assert_eq!(SignalDirection::from_score(0.5), SignalDirection::Buy);
        assert_eq!(SignalDirection::from_score(-0.5), SignalDirection::Sell);
        assert_eq!(SignalDirection::from_score(0.0), SignalDirection::Hold);
    }

    #[test]
    fn test_signal_strength_from_confidence() {
        assert_eq!(SignalStrength::from_confidence(0.1), SignalStrength::VeryWeak);
        assert_eq!(SignalStrength::from_confidence(0.3), SignalStrength::Weak);
        assert_eq!(SignalStrength::from_confidence(0.5), SignalStrength::Moderate);
        assert_eq!(SignalStrength::from_confidence(0.7), SignalStrength::Strong);
        assert_eq!(SignalStrength::from_confidence(0.9), SignalStrength::VeryStrong);
    }

    #[test]
    fn test_signal_creation() {
        let sig = make_signal(SignalDirection::Buy, 0.8, "momentum");
        assert_eq!(sig.direction, SignalDirection::Buy);
        assert_eq!(sig.confidence, 0.8);
        assert_eq!(sig.strength, SignalStrength::Strong);
        assert!(sig.is_actionable());
        assert!(sig.id.starts_with("sig_momentum_"));
    }

    #[test]
    fn test_signal_invalid_confidence() {
        let result = Signal::new(SignalDirection::Buy, 1.5, Utc::now(), "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_signal_with_symbol() {
        let sig = make_signal(SignalDirection::Buy, 0.8, "momentum")
            .with_symbol("BTC/USDT");
        assert_eq!(sig.symbol.as_deref(), Some("BTC/USDT"));
    }

    #[test]
    fn test_signal_conflicts() {
        let buy = make_signal(SignalDirection::Buy, 0.8, "strategy_a");
        let sell = make_signal(SignalDirection::Sell, 0.7, "strategy_b");
        assert!(buy.conflicts_with(&sell));
        assert!(!buy.conflicts_with(&buy));
    }

    #[test]
    fn test_weighted_average_aggregation() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.8, "s1"),
            make_signal(SignalDirection::Buy, 0.6, "s2"),
            make_signal(SignalDirection::Hold, 0.3, "s3"),
        ];

        let agg = SignalAggregator::new(ConsensusMethod::WeightedAverage);
        let consensus = agg.aggregate(&signals).unwrap();
        assert_eq!(consensus.direction, SignalDirection::Buy);
        assert!(consensus.confidence > 0.5);
    }

    #[test]
    fn test_median_aggregation() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.9, "s1"),
            make_signal(SignalDirection::Hold, 0.5, "s2"),
            make_signal(SignalDirection::Sell, 0.3, "s3"),
        ];

        let agg = SignalAggregator::new(ConsensusMethod::Median);
        let consensus = agg.aggregate(&signals).unwrap();
        // Scores: 0.9, 0, -0.3 → median is 0.0 → Hold
        assert_eq!(consensus.direction, SignalDirection::Hold);
    }

    #[test]
    fn test_majority_vote_aggregation() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.6, "s1"),
            make_signal(SignalDirection::Buy, 0.5, "s2"),
            make_signal(SignalDirection::Sell, 0.8, "s3"),
        ];

        let agg = SignalAggregator::new(ConsensusMethod::MajorityVote);
        let consensus = agg.aggregate(&signals).unwrap();
        assert_eq!(consensus.direction, SignalDirection::Buy);
    }

    #[test]
    fn test_best_signal_aggregation() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.6, "s1"),
            make_signal(SignalDirection::Sell, 0.9, "s2"),
            make_signal(SignalDirection::Hold, 0.5, "s3"),
        ];

        let agg = SignalAggregator::new(ConsensusMethod::BestSignal);
        let consensus = agg.aggregate(&signals).unwrap();
        assert_eq!(consensus.direction, SignalDirection::Sell);
        assert_eq!(consensus.confidence, 0.9);
    }

    #[test]
    fn test_borda_count_aggregation() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.9, "s1"),  // rank 0: 2 points
            make_signal(SignalDirection::Buy, 0.7, "s2"),  // rank 1: 1 point
            make_signal(SignalDirection::Sell, 0.3, "s3"), // rank 2: 0 points
        ];

        let agg = SignalAggregator::new(ConsensusMethod::BordaCount);
        let consensus = agg.aggregate(&signals).unwrap();
        assert_eq!(consensus.direction, SignalDirection::Buy);
    }

    #[test]
    fn test_aggregation_no_signals() {
        let agg = SignalAggregator::new(ConsensusMethod::WeightedAverage);
        let result = agg.aggregate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cancel_opposing() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.9, "s1"),
            make_signal(SignalDirection::Sell, 0.3, "s2"),
        ];

        let agg = SignalAggregator::new(ConsensusMethod::WeightedAverage).with_cancel_opposing(true);
        let consensus = agg.aggregate(&signals).unwrap();
        // Buy signal should win, sell cancelled
        assert_eq!(consensus.direction, SignalDirection::Buy);
    }

    #[test]
    fn test_signal_history_basic() {
        let mut history = SignalHistory::new();
        let sig = make_signal(SignalDirection::Buy, 0.8, "momentum");
        history.record_signal(sig, 100.0);

        assert_eq!(history.records.len(), 1);

        // Close the signal
        history.close_signal("momentum", 110.0, Utc::now()).unwrap();
        let record = &history.records[0];
        assert_eq!(record.exit_price, Some(110.0));
        assert_eq!(record.profitable, Some(true));
        assert_eq!(record.pnl_pct, Some(0.1));
    }

    #[test]
    fn test_signal_history_losing_trade() {
        let mut history = SignalHistory::new();
        let sig = make_signal(SignalDirection::Sell, 0.7, "mean_reversion");
        history.record_signal(sig, 100.0);
        history.close_signal("mean_reversion", 110.0, Utc::now()).unwrap();

        let record = &history.records[0];
        assert_eq!(record.profitable, Some(false));
        assert_eq!(record.direction_correct, Some(false)); // Sell was wrong direction
    }

    #[test]
    fn test_signal_performance() {
        let mut history = SignalHistory::new();

        // Add several trades
        for i in 0..5 {
            let direction = if i < 3 { SignalDirection::Buy } else { SignalDirection::Sell };
            let sig = make_signal(direction, 0.5 + i as f64 * 0.1, "test");
            history.record_signal(sig, 100.0);

            let exit = if i < 3 {
                100.0 + (i + 1) as f64 // profit
            } else {
                100.0 - (i - 2) as f64 // loss
            };
            history.close_signal("test", exit, Utc::now()).unwrap();
        }

        let perf = history.performance();
        assert_eq!(perf.total_signals, 5);
        assert_eq!(perf.closed_signals, 5);
        assert!(perf.win_rate > 0.0);
        assert!(perf.total_pnl > 0.0); // 3 wins, 2 losses
    }

    #[test]
    fn test_conflict_detection() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.8, "strategy_a"),
            make_signal(SignalDirection::Sell, 0.7, "strategy_b"),
            make_signal(SignalDirection::Hold, 0.3, "strategy_c"),
        ];

        let analysis = detect_conflicts(&signals);
        assert_eq!(analysis.conflict_count, 1);
        assert!(!analysis.has_unresolvable); // 0.8 vs 0.7 difference > 0.1
    }

    #[test]
    fn test_conflict_detection_unresolvable() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.75, "strategy_a"),
            make_signal(SignalDirection::Sell, 0.75, "strategy_b"),
        ];

        let analysis = detect_conflicts(&signals);
        assert_eq!(analysis.conflict_count, 1);
        assert!(analysis.has_unresolvable);
    }

    #[test]
    fn test_no_conflicts() {
        let signals = vec![
            make_signal(SignalDirection::Buy, 0.8, "strategy_a"),
            make_signal(SignalDirection::Buy, 0.6, "strategy_b"),
        ];

        let analysis = detect_conflicts(&signals);
        assert_eq!(analysis.conflict_count, 0);
    }

    #[test]
    fn test_source_weights() {
        let mut weights = HashMap::new();
        weights.insert("expert".to_string(), 2.0);
        weights.insert("novice".to_string(), 0.5);

        let signals = vec![
            make_signal(SignalDirection::Buy, 0.5, "expert"),
            make_signal(SignalDirection::Sell, 0.9, "novice"),
        ];

        let agg = SignalAggregator::new(ConsensusMethod::WeightedAverage)
            .with_source_weights(weights);
        let consensus = agg.aggregate(&signals).unwrap();
        // Expert should dominate: 2.0 * 0.5 vs 0.5 * (-0.9) = 1.0 - 0.45 = 0.55
        assert_eq!(consensus.direction, SignalDirection::Buy);
    }

    #[test]
    fn test_signal_serialization() {
        let sig = make_signal(SignalDirection::Buy, 0.8, "momentum")
            .with_symbol("BTC/USDT")
            .with_targets(120.0, 90.0);
        let json = serde_json::to_string(&sig).unwrap();
        let deserialized: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.direction, SignalDirection::Buy);
        assert_eq!(deserialized.symbol.as_deref(), Some("BTC/USDT"));
        assert_eq!(deserialized.price_target, Some(120.0));
        assert_eq!(deserialized.stop_loss, Some(90.0));
    }
}
