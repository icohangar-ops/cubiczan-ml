//! # Learning Engine
//!
//! The learning and adaptation layer. Tracks trade outcomes, analyzes patterns,
//! and adapts strategy parameters based on performance.
//!
//! ## Features
//! - Outcome tracking and storage
//! - Win/loss pattern analysis
//! - Strategy parameter adaptation
//! - Regime-specific learning
//! - Exponential decay weighting
//! - Adaptive learning rate

use crate::types::*;
use std::collections::HashMap;

/// A strategy parameter that can be adapted by the learner.
#[derive(Debug, Clone)]
pub struct StrategyParameter {
    pub name: String,
    pub value: f64,
    pub default_value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

impl StrategyParameter {
    pub fn new(name: impl Into<String>, value: f64, min: f64, max: f64, step: f64) -> Self {
        StrategyParameter {
            name: name.into(),
            value: value.clamp(min, max),
            default_value: value.clamp(min, max),
            min,
            max,
            step,
        }
    }

    pub fn adjust(&mut self, delta: f64) {
        self.value = (self.value + delta).clamp(self.min, self.max);
    }

    pub fn reset(&mut self) {
        self.value = self.default_value;
    }
}

/// Regime-specific performance statistics.
#[derive(Debug, Clone, Default)]
pub struct RegimeStats {
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub total_pnl: f64,
    pub avg_return: f64,
    pub win_rate: f64,
}

/// The learning engine that adapts strategy based on outcomes.
#[derive(Debug, Clone)]
pub struct LearningEngine {
    /// All tracked outcomes.
    pub outcomes: Vec<Outcome>,
    /// Strategy parameters that can be adapted.
    pub parameters: HashMap<String, StrategyParameter>,
    /// Per-regime statistics.
    pub regime_stats: HashMap<MarketRegime, RegimeStats>,
    /// Decay factor for exponential weighting (0 < lambda < 1).
    pub decay_factor: f64,
    /// Base learning rate for parameter adaptation.
    pub learning_rate: f64,
    /// Minimum outcomes needed before adapting.
    pub min_outcomes: usize,
    /// Performance window for adaptation (most recent N outcomes).
    pub adaptation_window: usize,
}

impl Default for LearningEngine {
    fn default() -> Self {
        let mut params = HashMap::new();
        params.insert("momentum_weight".into(), StrategyParameter::new("momentum_weight", 0.35, 0.0, 1.0, 0.05));
        params.insert("mean_reversion_weight".into(), StrategyParameter::new("mean_reversion_weight", 0.25, 0.0, 1.0, 0.05));
        params.insert("risk_weight".into(), StrategyParameter::new("risk_weight", 0.20, 0.0, 1.0, 0.05));
        params.insert("regime_weight".into(), StrategyParameter::new("regime_weight", 0.20, 0.0, 1.0, 0.05));
        params.insert("kelly_fraction".into(), StrategyParameter::new("kelly_fraction", 0.5, 0.1, 1.0, 0.05));
        params.insert("min_confidence".into(), StrategyParameter::new("min_confidence", 0.3, 0.1, 0.9, 0.05));

        LearningEngine {
            outcomes: Vec::new(),
            parameters: params,
            regime_stats: HashMap::new(),
            decay_factor: 0.95,
            learning_rate: 0.1,
            min_outcomes: 10,
            adaptation_window: 50,
        }
    }
}

impl LearningEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new outcome.
    pub fn record_outcome(&mut self, outcome: Outcome) {
        // Update regime stats
        let stats = self.regime_stats.entry(outcome.regime).or_default();
        stats.total_trades += 1;
        if outcome.profitable {
            stats.wins += 1;
        } else {
            stats.losses += 1;
        }
        stats.total_pnl += outcome.pnl;
        stats.win_rate = if stats.total_trades > 0 {
            stats.wins as f64 / stats.total_trades as f64
        } else {
            0.0
        };
        stats.avg_return = if stats.total_trades > 0 {
            stats.total_pnl / stats.total_trades as f64
        } else {
            0.0
        };

        self.outcomes.push(outcome);
    }

    /// Analyze win/loss patterns in recent outcomes.
    pub fn analyze_patterns(&self) -> PatternAnalysis {
        if self.outcomes.is_empty() {
            return PatternAnalysis::default();
        }

        let window = self.adaptation_window.min(self.outcomes.len());
        let recent: &[Outcome] = &self.outcomes[self.outcomes.len() - window..];

        let wins = recent.iter().filter(|o| o.profitable).count();
        let losses = recent.len() - wins;
        let win_rate = wins as f64 / recent.len().max(1) as f64;

        let avg_win: f64 = recent.iter().filter(|o| o.profitable)
            .map(|o| o.pnl).sum::<f64>() / wins.max(1) as f64;
        let avg_loss: f64 = recent.iter().filter(|o| !o.profitable)
            .map(|o| o.pnl.abs()).sum::<f64>() / losses.max(1) as f64;

        let profit_factor = if avg_loss.abs() < 1e-15 {
            f64::INFINITY
        } else {
            avg_win / avg_loss
        };

        // Consecutive win/loss streaks
        let mut max_win_streak = 0usize;
        let mut max_loss_streak = 0usize;
        let mut current_streak = 0usize;
        let mut current_winning = false;

        for o in recent {
            if o.profitable {
                if current_winning {
                    current_streak += 1;
                } else {
                    current_streak = 1;
                    current_winning = true;
                }
                max_win_streak = max_win_streak.max(current_streak);
            } else {
                if !current_winning {
                    current_streak += 1;
                } else {
                    current_streak = 1;
                    current_winning = false;
                }
                max_loss_streak = max_loss_streak.max(current_streak);
            }
        }

        // Average return with exponential decay
        let total_pnl: f64 = recent.iter().map(|o| o.pnl).sum();
        let avg_return = total_pnl / recent.len() as f64;

        PatternAnalysis {
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            max_win_streak,
            max_loss_streak,
            total_trades: recent.len(),
            avg_return,
            total_pnl,
            recent_window: window,
        }
    }

    /// Adapt strategy parameters based on recent performance.
    pub fn adapt_parameters(&mut self) -> Vec<(String, f64, f64)> {
        if self.outcomes.len() < self.min_outcomes {
            return Vec::new();
        }

        let analysis = self.analyze_patterns();
        let mut adjustments = Vec::new();

        // Adaptive learning rate: slow down when doing well, speed up when struggling
        let effective_lr = if analysis.win_rate > 0.6 {
            self.learning_rate * 0.5 // Slow down
        } else if analysis.win_rate < 0.4 {
            self.learning_rate * 2.0 // Speed up
        } else {
            self.learning_rate
        };

        // Adapt momentum weight: increase if trending regime is profitable
        if let Some(stats) = self.regime_stats.get(&MarketRegime::Trending) {
            if stats.total_trades > 5 {
                let delta = if stats.win_rate > 0.55 {
                    effective_lr * 0.05
                } else {
                    -effective_lr * 0.05
                };
                if let Some(param) = self.parameters.get_mut("momentum_weight") {
                    let old = param.value;
                    param.adjust(delta);
                    adjustments.push(("momentum_weight".into(), old, param.value));
                }
            }
        }

        // Adapt mean-reversion weight
        if let Some(stats) = self.regime_stats.get(&MarketRegime::MeanReverting) {
            if stats.total_trades > 5 {
                let delta = if stats.win_rate > 0.55 {
                    effective_lr * 0.05
                } else {
                    -effective_lr * 0.05
                };
                if let Some(param) = self.parameters.get_mut("mean_reversion_weight") {
                    let old = param.value;
                    param.adjust(delta);
                    adjustments.push(("mean_reversion_weight".into(), old, param.value));
                }
            }
        }

        // Adapt kelly fraction based on overall win rate
        if let Some(param) = self.parameters.get_mut("kelly_fraction") {
            let old = param.value;
            let delta = if analysis.win_rate > 0.55 {
                effective_lr * 0.03
            } else {
                -effective_lr * 0.03
            };
            param.adjust(delta);
            adjustments.push(("kelly_fraction".into(), old, param.value));
        }

        // Adapt min_confidence: increase if win rate is low (be more selective)
        if let Some(param) = self.parameters.get_mut("min_confidence") {
            let old = param.value;
            let delta = if analysis.win_rate < 0.45 {
                effective_lr * 0.05
            } else {
                -effective_lr * 0.02
            };
            param.adjust(delta);
            adjustments.push(("min_confidence".into(), old, param.value));
        }

        adjustments
    }

    /// Get decay-weighted performance score.
    pub fn decay_weighted_score(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }

        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        let mut weight = 1.0_f64;

        for outcome in self.outcomes.iter().rev() {
            weighted_sum += outcome.pnl * weight;
            weight_sum += weight;
            weight *= self.decay_factor;
        }

        if weight_sum < 1e-15 {
            0.0
        } else {
            weighted_sum / weight_sum
        }
    }

    /// Get regime-specific win rate.
    pub fn regime_win_rate(&self, regime: MarketRegime) -> f64 {
        self.regime_stats.get(&regime)
            .map(|s| s.win_rate)
            .unwrap_or(0.0)
    }

    /// Get learning rate adjusted for current performance.
    pub fn effective_learning_rate(&self) -> f64 {
        let analysis = self.analyze_patterns();
        if analysis.win_rate > 0.6 {
            self.learning_rate * 0.5
        } else if analysis.win_rate < 0.4 {
            self.learning_rate * 2.0
        } else {
            self.learning_rate
        }
    }

    /// Reset all parameters to defaults.
    pub fn reset_parameters(&mut self) {
        for param in self.parameters.values_mut() {
            param.reset();
        }
    }
}

/// Result of pattern analysis.
#[derive(Debug, Clone, Default)]
pub struct PatternAnalysis {
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub max_win_streak: usize,
    pub max_loss_streak: usize,
    pub total_trades: usize,
    pub avg_return: f64,
    pub total_pnl: f64,
    pub recent_window: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_outcome(profitable: bool, pnl: f64, regime: MarketRegime) -> Outcome {
        Outcome {
            decision_id: "test".into(),
            pnl,
            return_pct: pnl / 1000.0,
            risk_adjusted_return: pnl / 500.0,
            drawdown_impact: if profitable { 0.0 } else { pnl.abs() / 10000.0 },
            lesson: if profitable { "Good" } else { "Bad" }.into(),
            timestamp: chrono::Utc::now(),
            profitable,
            duration_secs: 3600.0,
            regime,
        }
    }

    #[test]
    fn test_record_outcome() {
        let mut engine = LearningEngine::new();
        engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Trending));
        assert_eq!(engine.outcomes.len(), 1);
        assert_eq!(engine.regime_stats[&MarketRegime::Trending].wins, 1);
    }

    #[test]
    fn test_regime_stats_tracking() {
        let mut engine = LearningEngine::new();
        engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Trending));
        engine.record_outcome(sample_outcome(false, -50.0, MarketRegime::Trending));
        engine.record_outcome(sample_outcome(true, 75.0, MarketRegime::Trending));
        let stats = &engine.regime_stats[&MarketRegime::Trending];
        assert_eq!(stats.total_trades, 3);
        assert!((stats.win_rate - 2.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_analysis_basic() {
        let mut engine = LearningEngine::new();
        for i in 0..20 {
            engine.record_outcome(sample_outcome(i < 12, if i < 12 { 100.0 } else { -80.0 }, MarketRegime::Trending));
        }
        let analysis = engine.analyze_patterns();
        assert!((analysis.win_rate - 0.6).abs() < 1e-10);
        assert_eq!(analysis.total_trades, 20);
    }

    #[test]
    fn test_pattern_analysis_streaks() {
        let mut engine = LearningEngine::new();
        for _ in 0..5 {
            engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Trending));
        }
        for _ in 0..3 {
            engine.record_outcome(sample_outcome(false, -50.0, MarketRegime::Trending));
        }
        let analysis = engine.analyze_patterns();
        assert_eq!(analysis.max_win_streak, 5);
        assert_eq!(analysis.max_loss_streak, 3);
    }

    #[test]
    fn test_adapt_parameters_insufficient_data() {
        let mut engine = LearningEngine::new();
        let adj = engine.adapt_parameters();
        assert!(adj.is_empty());
    }

    #[test]
    fn test_adapt_parameters_with_data() {
        let mut engine = LearningEngine::new();
        // Record enough outcomes with poor performance to trigger adaptation
        for i in 0..15 {
            let profitable = i < 5; // Only 5 wins out of 15
            engine.record_outcome(sample_outcome(profitable, if profitable { 50.0 } else { -100.0 }, MarketRegime::Trending));
        }
        let adj = engine.adapt_parameters();
        assert!(!adj.is_empty());
    }

    #[test]
    fn test_decay_weighted_score() {
        let mut engine = LearningEngine::new();
        engine.decay_factor = 0.9;
        engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Trending));
        engine.record_outcome(sample_outcome(true, 200.0, MarketRegime::Trending));
        let score = engine.decay_weighted_score();
        // More recent outcome should have higher weight
        assert!(score > 100.0 && score < 200.0);
    }

    #[test]
    fn test_decay_weighted_empty() {
        let engine = LearningEngine::new();
        assert_eq!(engine.decay_weighted_score(), 0.0);
    }

    #[test]
    fn test_regime_win_rate() {
        let mut engine = LearningEngine::new();
        engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Crisis));
        engine.record_outcome(sample_outcome(false, -50.0, MarketRegime::Crisis));
        assert!((engine.regime_win_rate(MarketRegime::Crisis) - 0.5).abs() < 1e-10);
        assert_eq!(engine.regime_win_rate(MarketRegime::Quiet), 0.0); // No data
    }

    #[test]
    fn test_effective_learning_rate_high_wr() {
        let mut engine = LearningEngine::new();
        for _ in 0..15 {
            engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Trending));
        }
        let lr = engine.effective_learning_rate();
        assert!(lr < engine.learning_rate); // Should slow down
    }

    #[test]
    fn test_effective_learning_rate_low_wr() {
        let mut engine = LearningEngine::new();
        for _ in 0..15 {
            engine.record_outcome(sample_outcome(false, -50.0, MarketRegime::Trending));
        }
        let lr = engine.effective_learning_rate();
        assert!(lr > engine.learning_rate); // Should speed up
    }

    #[test]
    fn test_strategy_parameter_adjust() {
        let mut param = StrategyParameter::new("test", 0.5, 0.0, 1.0, 0.05);
        param.adjust(0.2);
        assert_eq!(param.value, 0.7);
        param.adjust(0.5); // Would go to 1.2 but clamped
        assert_eq!(param.value, 1.0);
    }

    #[test]
    fn test_strategy_parameter_reset() {
        let mut param = StrategyParameter::new("test", 0.5, 0.0, 1.0, 0.05);
        param.adjust(0.3);
        param.reset();
        assert_eq!(param.value, 0.5);
    }

    #[test]
    fn test_reset_all_parameters() {
        let mut engine = LearningEngine::new();
        engine.parameters.get_mut("kelly_fraction").unwrap().adjust(0.3);
        engine.reset_parameters();
        assert!((engine.parameters["kelly_fraction"].value - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_profit_factor_calculation() {
        let mut engine = LearningEngine::new();
        for i in 0..10 {
            let profitable = i < 6;
            engine.record_outcome(sample_outcome(profitable, if profitable { 200.0 } else { -100.0 }, MarketRegime::Trending));
        }
        let analysis = engine.analyze_patterns();
        assert_eq!(analysis.profit_factor, 2.0); // avg_win=200, avg_loss=100
    }

    #[test]
    fn test_profit_factor_no_losses() {
        let mut engine = LearningEngine::new();
        for _ in 0..5 {
            engine.record_outcome(sample_outcome(true, 100.0, MarketRegime::Trending));
        }
        let analysis = engine.analyze_patterns();
        assert!(analysis.profit_factor.is_infinite());
    }

    #[test]
    fn test_adaptation_window_respected() {
        let mut engine = LearningEngine::new();
        engine.adaptation_window = 10;
        for i in 0..20 {
            engine.record_outcome(sample_outcome(i < 15, if i < 15 { 100.0 } else { -100.0 }, MarketRegime::Trending));
        }
        let analysis = engine.analyze_patterns();
        assert_eq!(analysis.recent_window, 10);
        // Recent 10 should have 5 wins
        assert!((analysis.win_rate - 0.5).abs() < 1e-10);
    }
}
