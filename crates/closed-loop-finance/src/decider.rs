//! # Decision Engine
//!
//! The decision-making layer of the closed-loop system. Takes market observations and
//! portfolio state as input, applies multi-factor scoring, and produces [`DecisionSignal`]s.
//!
//! ## Factor Scoring
//! - **Momentum**: Trend direction × strength
//! - **Mean-reversion**: Distance from moving averages
//! - **Risk**: Current portfolio risk level
//! - **Regime**: Market regime alignment
//!
//! ## Position Sizing
//! Uses a Kelly-like fraction combined with risk budgeting to determine position size.

use crate::types::*;
use std::collections::HashMap;

/// Weight configuration for factor scoring.
#[derive(Debug, Clone)]
pub struct FactorWeights {
    pub momentum: f64,
    pub mean_reversion: f64,
    pub risk: f64,
    pub regime: f64,
}

impl Default for FactorWeights {
    fn default() -> Self {
        FactorWeights {
            momentum: 0.35,
            mean_reversion: 0.25,
            risk: 0.20,
            regime: 0.20,
        }
    }
}

/// The decision engine that produces trading signals from observations.
#[derive(Debug, Clone)]
pub struct DecisionEngine {
    /// Factor weights for scoring.
    pub weights: FactorWeights,
    /// Minimum confidence to produce an actionable signal.
    pub min_confidence: f64,
    /// Kelly fraction multiplier (e.g., 0.5 = half-Kelly).
    pub kelly_fraction: f64,
    /// Maximum position size as fraction of capital.
    pub max_position_size: f64,
    /// Lookback for moving average computations.
    pub ma_lookback: usize,
    /// Decision ID counter.
    decision_counter: u64,
}

impl Default for DecisionEngine {
    fn default() -> Self {
        DecisionEngine {
            weights: FactorWeights::default(),
            min_confidence: 0.3,
            kelly_fraction: 0.5,
            max_position_size: 0.25,
            ma_lookback: 20,
            decision_counter: 0,
        }
    }
}

impl DecisionEngine {
    pub fn new(weights: FactorWeights, min_confidence: f64) -> Self {
        DecisionEngine {
            weights,
            min_confidence,
            ..Default::default()
        }
    }

    /// Produce a decision signal from market observation and portfolio state.
    pub fn decide(
        &mut self,
        observation: &MarketObservation,
        portfolio: &PortfolioState,
    ) -> DecisionSignal {
        // Compute factor scores
        let momentum_score = self.momentum_score(observation);
        let mr_score = self.mean_reversion_score(observation);
        let risk_score = self.risk_score(observation, portfolio);
        let regime_score = self.regime_score(observation);

        let mut factor_scores = HashMap::new();
        factor_scores.insert("momentum".into(), momentum_score);
        factor_scores.insert("mean_reversion".into(), mr_score);
        factor_scores.insert("risk".into(), risk_score);
        factor_scores.insert("regime".into(), regime_score);

        // Weighted aggregation
        let total_weight = self.weights.momentum + self.weights.mean_reversion
            + self.weights.risk + self.weights.regime;

        let weighted_score = (momentum_score * self.weights.momentum
            + mr_score * self.weights.mean_reversion
            + risk_score * self.weights.risk
            + regime_score * self.weights.regime)
            / total_weight.max(1e-15);

        // Determine action and confidence from score
        let (action, confidence, reasoning) = self.resolve_action(weighted_score, observation);

        // Position sizing
        let position_size = self.compute_position_size(
            confidence,
            observation,
            portfolio,
        );

        // Risk assessment
        let risk_assessment = self.compute_risk_assessment(observation, portfolio);

        self.decision_counter += 1;

        let mut signal = DecisionSignal {
            id: format!("dec_{}_{}", self.decision_counter, observation.timestamp.timestamp_millis()),
            action,
            confidence,
            reasoning,
            risk_assessment,
            timestamp: chrono::Utc::now(),
            symbol: None,
            position_size,
            factor_scores,
        };

        // Check confidence threshold
        if confidence < self.min_confidence {
            signal.action = Action::Hold;
            signal.confidence = confidence;
            signal.reasoning = "Below minimum confidence threshold".into();
        }

        signal
    }

    /// Compute momentum score from observation.
    ///
    /// Combines trend strength and direction.
    pub fn momentum_score(&self, obs: &MarketObservation) -> f64 {
        obs.trend_direction * obs.trend_strength
    }

    /// Compute mean-reversion score from observation.
    ///
    /// Higher when price has deviated far from recent average and trend is weak.
    pub fn mean_reversion_score(&self, obs: &MarketObservation) -> f64 {
        if obs.recent_returns.is_empty() {
            return 0.0;
        }

        // If returns are mostly positive, expect reversion down (negative score)
        // If returns are mostly negative, expect reversion up (positive score)
        let mean_ret: f64 = obs.recent_returns.iter().sum::<f64>() / obs.recent_returns.len() as f64;

        // Stronger reversion signal when trend is weak
        let trend_inverse = 1.0 - obs.trend_strength;
        let reversion_signal = -mean_ret * 10.0 * trend_inverse; // Scale up

        reversion_signal.clamp(-1.0, 1.0)
    }

    /// Compute risk score.
    ///
    /// Higher portfolio risk = lower risk score (more defensive).
    pub fn risk_score(&self, obs: &MarketObservation, portfolio: &PortfolioState) -> f64 {
        let vol_factor = match obs.volatility_regime {
            VolatilityRegime::Low => 0.8,
            VolatilityRegime::Medium => 0.5,
            VolatilityRegime::High => 0.2,
            VolatilityRegime::Extreme => -0.5,
        };

        let dd_penalty = if portfolio.max_drawdown > 0.10 {
            -(portfolio.max_drawdown - 0.10) * 5.0
        } else {
            0.0
        };

        let leverage_penalty = if portfolio.leverage > 1.5 {
            -(portfolio.leverage - 1.5) * 2.0
        } else {
            0.0
        };

        (vol_factor + dd_penalty + leverage_penalty).clamp(-1.0, 1.0)
    }

    /// Compute regime alignment score.
    ///
    /// Higher when regime favors active trading.
    pub fn regime_score(&self, obs: &MarketObservation) -> f64 {
        match obs.regime {
            MarketRegime::Trending => obs.trend_direction * 0.6,
            MarketRegime::MeanReverting => -obs.trend_direction * 0.5,
            MarketRegime::Quiet => 0.0,
            MarketRegime::Recovery => 0.3,
            MarketRegime::Volatile => -0.3,
            MarketRegime::Crisis => -0.8,
        }
    }

    /// Resolve the final action, confidence, and reasoning from the weighted score.
    fn resolve_action(&self, score: f64, obs: &MarketObservation) -> (Action, f64, String) {
        let abs_score = score.abs();
        let confidence = abs_score;

        let action = if obs.risk_event_detected {
            Action::Hedge
        } else if score > 0.3 {
            Action::Buy
        } else if score > 0.1 {
            Action::LeverageUp
        } else if score < -0.3 {
            Action::Sell
        } else if score < -0.1 {
            Action::Delever
        } else {
            Action::Hold
        };

        let reasoning = format!(
            "Weighted score: {:.3} (momentum={:.2}, mr={:.2}, risk={:.2}, regime={:.2}) | Regime: {}",
            score,
            self.momentum_score(obs),
            self.mean_reversion_score(obs),
            self.risk_score(obs, &PortfolioState::default()),
            self.regime_score(obs),
            obs.regime,
        );

        (action, confidence.clamp(0.0, 1.0), reasoning)
    }

    /// Compute position size using Kelly-like fraction with risk budgeting.
    fn compute_position_size(
        &self,
        confidence: f64,
        obs: &MarketObservation,
        _portfolio: &PortfolioState,
    ) -> f64 {
        if confidence < self.min_confidence {
            return 0.0;
        }

        // Base Kelly fraction scaled by confidence and regime risk
        let regime_adj = 1.0 / obs.regime.risk_multiplier();
        let vol_adj = match obs.volatility_regime {
            VolatilityRegime::Low => 1.0,
            VolatilityRegime::Medium => 0.8,
            VolatilityRegime::High => 0.5,
            VolatilityRegime::Extreme => 0.2,
        };

        let size = self.kelly_fraction * confidence * regime_adj * vol_adj;

        size.clamp(0.0, self.max_position_size)
    }

    /// Compute overall risk assessment score.
    fn compute_risk_assessment(&self, obs: &MarketObservation, portfolio: &PortfolioState) -> f64 {
        let vol_risk = obs.volatility / self.high_vol_default();
        let dd_risk = portfolio.max_drawdown / 0.20; // 20% max
        let lev_risk = portfolio.leverage / 3.0;

        let combined = vol_risk * 0.4 + dd_risk * 0.35 + lev_risk * 0.25;
        combined.clamp(0.0, 1.0)
    }

    fn high_vol_default(&self) -> f64 {
        0.30
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(regime: MarketRegime, trend_dir: f64, trend_str: f64, vol: f64) -> MarketObservation {
        MarketObservation {
            regime,
            price: 100.0,
            volatility_regime: if vol < 0.15 { VolatilityRegime::Low } else if vol < 0.30 { VolatilityRegime::Medium } else { VolatilityRegime::High },
            volatility: vol,
            trend_strength: trend_str,
            trend_direction: trend_dir,
            liquidity_score: 0.7,
            avg_correlation: 0.3,
            risk_event_detected: false,
            risk_event_description: None,
            recent_returns: vec![0.01, 0.02, -0.005, 0.015, 0.01],
            timestamp: chrono::Utc::now(),
        }
    }

    fn make_portfolio() -> PortfolioState {
        PortfolioState::default()
    }

    #[test]
    fn test_momentum_score_bullish() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, 0.8, 0.7, 0.15);
        let score = engine.momentum_score(&obs);
        assert!(score > 0.3, "Expected positive momentum, got {}", score);
    }

    #[test]
    fn test_momentum_score_bearish() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, -0.8, 0.7, 0.15);
        let score = engine.momentum_score(&obs);
        assert!(score < -0.3);
    }

    #[test]
    fn test_mean_reversion_score_positive_returns() {
        let engine = DecisionEngine::default();
        let mut obs = make_obs(MarketRegime::Quiet, 0.0, 0.1, 0.10);
        obs.recent_returns = vec![0.02, 0.03, 0.02, 0.01, 0.03];
        let score = engine.mean_reversion_score(&obs);
        assert!(score < 0.0, "Positive returns should give negative MR score, got {}", score);
    }

    #[test]
    fn test_mean_reversion_score_negative_returns() {
        let engine = DecisionEngine::default();
        let mut obs = make_obs(MarketRegime::Quiet, 0.0, 0.1, 0.10);
        obs.recent_returns = vec![-0.02, -0.03, -0.02, -0.01, -0.03];
        let score = engine.mean_reversion_score(&obs);
        assert!(score > 0.0);
    }

    #[test]
    fn test_risk_score_low_vol() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, 0.5, 0.6, 0.05);
        let score = engine.risk_score(&obs, &make_portfolio());
        assert!(score > 0.0, "Low vol should give positive risk score");
    }

    #[test]
    fn test_risk_score_high_drawdown() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Volatile, 0.0, 0.3, 0.35);
        let mut portfolio = make_portfolio();
        portfolio.max_drawdown = 0.15;
        let score = engine.risk_score(&obs, &portfolio);
        assert!(score < 0.0, "High DD should give negative risk score");
    }

    #[test]
    fn test_regime_score_trending() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, 0.7, 0.6, 0.15);
        let score = engine.regime_score(&obs);
        assert!(score > 0.0);
    }

    #[test]
    fn test_regime_score_crisis() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Crisis, 0.0, 0.0, 0.50);
        let score = engine.regime_score(&obs);
        assert!(score < -0.5);
    }

    #[test]
    fn test_decide_buy_signal() {
        let mut engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, 0.8, 0.8, 0.10);
        let signal = engine.decide(&obs, &make_portfolio());
        assert_eq!(signal.action, Action::Buy);
        assert!(signal.confidence > 0.3);
    }

    #[test]
    fn test_decide_sell_signal() {
        let mut engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, -1.0, 1.0, 0.05);
        let signal = engine.decide(&obs, &make_portfolio());
        assert_eq!(signal.action, Action::Sell);
    }

    #[test]
    fn test_decide_hold_low_confidence() {
        let mut engine = DecisionEngine::new(FactorWeights::default(), 0.9);
        let obs = make_obs(MarketRegime::Quiet, 0.1, 0.05, 0.08);
        let signal = engine.decide(&obs, &make_portfolio());
        assert_eq!(signal.action, Action::Hold);
    }

    #[test]
    fn test_decide_risk_event_hedge() {
        let mut engine = DecisionEngine::default();
        let mut obs = make_obs(MarketRegime::Trending, 0.8, 0.8, 0.10);
        obs.risk_event_detected = true;
        let signal = engine.decide(&obs, &make_portfolio());
        assert_eq!(signal.action, Action::Hedge);
    }

    #[test]
    fn test_position_size_nonzero() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, 0.7, 0.7, 0.15);
        let size = engine.compute_position_size(0.7, &obs, &make_portfolio());
        assert!(size > 0.0);
        assert!(size <= engine.max_position_size);
    }

    #[test]
    fn test_position_size_zero_low_confidence() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Quiet, 0.1, 0.05, 0.10);
        let size = engine.compute_position_size(0.1, &obs, &make_portfolio());
        assert_eq!(size, 0.0);
    }

    #[test]
    fn test_position_size_clamped() {
        let mut engine = DecisionEngine::new(FactorWeights::default(), 0.01);
        engine.max_position_size = 0.10;
        let obs = make_obs(MarketRegime::Trending, 0.9, 0.9, 0.05);
        // High confidence in low vol trending should approach max
        let size = engine.compute_position_size(0.95, &obs, &make_portfolio());
        assert!(size <= 0.10);
    }

    #[test]
    fn test_risk_assessment_calculation() {
        let engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Volatile, 0.0, 0.3, 0.40);
        let mut portfolio = make_portfolio();
        portfolio.max_drawdown = 0.10;
        portfolio.leverage = 2.0;
        let risk = engine.compute_risk_assessment(&obs, &portfolio);
        assert!(risk > 0.0 && risk <= 1.0);
    }

    #[test]
    fn test_factor_weights_sum() {
        let w = FactorWeights::default();
        let total = w.momentum + w.mean_reversion + w.risk + w.regime;
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_below_threshold_becomes_hold() {
        let mut engine = DecisionEngine::new(FactorWeights::default(), 0.9);
        let obs = make_obs(MarketRegime::Quiet, 0.1, 0.05, 0.08);
        let signal = engine.decide(&obs, &make_portfolio());
        assert_eq!(signal.action, Action::Hold);
        assert!(signal.reasoning.contains("Below minimum confidence"));
    }

    #[test]
    fn test_decision_id_increments() {
        let mut engine = DecisionEngine::default();
        let obs = make_obs(MarketRegime::Trending, 0.8, 0.8, 0.10);
        let s1 = engine.decide(&obs, &make_portfolio());
        let s2 = engine.decide(&obs, &make_portfolio());
        assert_ne!(s1.id, s2.id);
    }

    #[test]
    fn test_volatility_regime_adjustment() {
        let engine = DecisionEngine::default();
        let obs_low = make_obs(MarketRegime::Trending, 0.8, 0.7, 0.05);
        let obs_high = make_obs(MarketRegime::Volatile, 0.8, 0.7, 0.45);
        let size_low = engine.compute_position_size(0.8, &obs_low, &make_portfolio());
        let size_high = engine.compute_position_size(0.8, &obs_high, &make_portfolio());
        assert!(size_low > size_high, "Low vol should allow larger positions");
    }
}
