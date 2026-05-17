//! # Trading Policies
//!
//! Pre-built trading policies for RL agents: Kelly criterion, risk parity,
//! momentum, mean reversion, and adaptive policy composition.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

/// Core trading policy trait.
pub trait TradingPolicy: Debug + Send + Sync {
    /// Compute position size given portfolio context.
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal;

    /// Policy name.
    fn name(&self) -> &str;
}

/// Context for policy decision-making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyContext {
    /// Current portfolio value.
    pub portfolio_value: f64,
    /// Available cash.
    pub cash: f64,
    /// Current position size (number of units).
    pub current_position: f64,
    /// Current price.
    pub price: f64,
    /// Recent returns (oldest to newest).
    pub recent_returns: Vec<f64>,
    /// Volatility (annualized).
    pub volatility: f64,
    /// Signal strength from model (-1.0 to 1.0).
    pub signal_strength: f64,
    /// Maximum position size as fraction of portfolio.
    pub max_position_frac: f64,
    /// Risk-free rate.
    pub risk_free_rate: f64,
}

impl PolicyContext {
    pub fn new(price: f64, portfolio_value: f64) -> Self {
        Self {
            portfolio_value,
            cash: portfolio_value,
            current_position: 0.0,
            price,
            recent_returns: vec![],
            volatility: 0.2,
            signal_strength: 0.0,
            max_position_frac: 1.0,
            risk_free_rate: 0.05,
        }
    }

    /// Current position as a fraction of portfolio value.
    pub fn position_fraction(&self) -> f64 {
        if self.portfolio_value == 0.0 {
            return 0.0;
        }
        (self.current_position * self.price).abs() / self.portfolio_value
    }

    /// Sharpe ratio estimate from recent returns.
    pub fn estimate_sharpe(&self) -> f64 {
        if self.recent_returns.is_empty() || self.volatility == 0.0 {
            return 0.0;
        }
        let mean: f64 = self.recent_returns.iter().sum::<f64>() / self.recent_returns.len() as f64;
        (mean - self.risk_free_rate) / self.volatility
    }
}

/// Signal from a policy: direction and confidence-weighted size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSignal {
    /// Target number of units (positive = long, negative = short).
    pub target_units: f64,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Reasoning string.
    pub reason: String,
}

impl PositionSignal {
    pub fn hold(reason: &str) -> Self {
        Self { target_units: 0.0, confidence: 0.0, reason: reason.to_string() }
    }

    pub fn buy(units: f64, confidence: f64, reason: &str) -> Self {
        Self { target_units: units, confidence: confidence.clamp(0.0, 1.0), reason: reason.to_string() }
    }

    pub fn sell(units: f64, confidence: f64, reason: &str) -> Self {
        Self { target_units: -units, confidence: confidence.clamp(0.0, 1.0), reason: reason.to_string() }
    }
}

/// Kelly Criterion position sizing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyPolicy {
    /// Fraction of Kelly to use (0.0 to 1.0). Full Kelly can be volatile.
    pub kelly_fraction: f64,
    /// Maximum position as fraction of portfolio.
    pub max_position: f64,
    /// Minimum edge required to take a position.
    pub min_edge: f64,
}

impl KellyPolicy {
    pub fn new(kelly_fraction: f64) -> Self {
        Self {
            kelly_fraction: kelly_fraction.clamp(0.0, 1.0),
            max_position: 0.95,
            min_edge: 0.01,
        }
    }

    pub fn half_kelly() -> Self {
        Self::new(0.5)
    }

    pub fn quarter_kelly() -> Self {
        Self::new(0.25)
    }

    /// Calculate Kelly fraction from win probability and payoff ratio.
    pub fn kelly_fraction_from_prob(win_prob: f64, win_loss_ratio: f64) -> f64 {
        let loss_prob = 1.0 - win_prob;
        let kelly = win_prob - (loss_prob / win_loss_ratio);
        kelly.max(0.0)
    }
}

impl TradingPolicy for KellyPolicy {
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal {
        // Need return data to estimate payoff ratio
        if ctx.recent_returns.is_empty() {
            return PositionSignal::hold("No return data for Kelly sizing");
        }

        // Estimate win probability and payoff from signal and recent returns
        let win_prob = (ctx.signal_strength + 1.0) / 2.0; // map [-1,1] to [0,1]
        let avg_win = ctx.recent_returns.iter().filter(|&&r| r > 0.0).sum::<f64>()
            / ctx.recent_returns.iter().filter(|&&r| r > 0.0).count().max(1) as f64;
        let avg_loss = ctx.recent_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum::<f64>()
            / ctx.recent_returns.iter().filter(|&&r| r < 0.0).count().max(1) as f64;
        let payoff_ratio = if avg_loss > 0.0 { avg_win / avg_loss } else { 2.0 };

        let kelly = Self::kelly_fraction_from_prob(win_prob, payoff_ratio);
        let adjusted = kelly * self.kelly_fraction;

        if adjusted < self.min_edge {
            return PositionSignal::hold("Insufficient edge for Kelly position");
        }

        let capped = adjusted.min(self.max_position);
        let target_value = ctx.portfolio_value * capped;
        let target_units = if ctx.price > 0.0 { target_value / ctx.price } else { 0.0 };

        let direction = if ctx.signal_strength >= 0.0 { "Long" } else { "Short" };
        PositionSignal {
            target_units: if ctx.signal_strength >= 0.0 { target_units } else { -target_units },
            confidence: capped,
            reason: format!(
                "{} Kelly: adj={:.3}, win_prob={:.2}, payoff={:.2}",
                direction, adjusted, win_prob, payoff_ratio
            ),
        }
    }

    fn name(&self) -> &str { "Kelly Criterion" }
}

/// Risk parity position sizing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParityPolicy {
    /// Target volatility per position.
    pub target_vol: f64,
    /// Volatility lookback window (number of returns).
    pub vol_lookback: usize,
}

impl RiskParityPolicy {
    pub fn new(target_vol: f64) -> Self {
        Self { target_vol, vol_lookback: 20 }
    }

    /// Calculate risk parity weight from current volatility.
    pub fn weight_from_vol(&self, current_vol: f64) -> f64 {
        if current_vol < 1e-8 { return 1.0; }
        (self.target_vol / current_vol).min(2.0)
    }
}

impl TradingPolicy for RiskParityPolicy {
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal {
        let weight = self.weight_from_vol(ctx.volatility);
        let target_value = ctx.portfolio_value * weight * 0.5; // scale down for single asset

        if ctx.signal_strength.abs() < 0.1 {
            return PositionSignal::hold("Signal too weak for risk parity position");
        }

        let target_units = if ctx.price > 0.0 { target_value / ctx.price } else { 0.0 };
        PositionSignal {
            target_units: if ctx.signal_strength >= 0.0 { target_units } else { -target_units },
            confidence: weight.min(1.0),
            reason: format!(
                "RiskParity: vol={:.2}%, weight={:.2}, signal={:.2}",
                ctx.volatility, weight, ctx.signal_strength
            ),
        }
    }

    fn name(&self) -> &str { "Risk Parity" }
}

/// Momentum-based policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumPolicy {
    /// Lookback period for momentum calculation.
    pub lookback: usize,
    /// Momentum threshold to trigger a trade.
    pub threshold: f64,
}

impl MomentumPolicy {
    pub fn new(lookback: usize, threshold: f64) -> Self {
        Self { lookback, threshold }
    }

    /// Calculate momentum from returns.
    pub fn momentum_score(&self, returns: &[f64]) -> f64 {
        let window = returns.len().min(self.lookback);
        if window == 0 { return 0.0; }
        returns[returns.len() - window..].iter().sum()
    }
}

impl TradingPolicy for MomentumPolicy {
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal {
        let momentum = self.momentum_score(&ctx.recent_returns);

        if momentum.abs() < self.threshold {
            return PositionSignal::hold(&format!(
                "Momentum {:.4} below threshold {:.4}", momentum, self.threshold
            ));
        }

        let size_frac = (momentum.abs() * 2.0).min(ctx.max_position_frac);
        let target_value = ctx.portfolio_value * size_frac;
        let target_units = if ctx.price > 0.0 { target_value / ctx.price } else { 0.0 };

        let direction = if momentum > 0.0 { "Long" } else { "Short" };
        PositionSignal {
            target_units: if momentum > 0.0 { target_units } else { -target_units },
            confidence: size_frac,
            reason: format!("{} Momentum: score={:.4}, size={:.2}%", direction, momentum, size_frac * 100.0),
        }
    }

    fn name(&self) -> &str { "Momentum" }
}

/// Mean reversion policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionPolicy {
    /// Lookback for mean calculation.
    pub lookback: usize,
    /// Number of standard deviations for entry.
    pub entry_sigma: f64,
    /// Number of standard deviations for exit.
    pub exit_sigma: f64,
}

impl MeanReversionPolicy {
    pub fn new(lookback: usize, entry_sigma: f64, exit_sigma: f64) -> Self {
        Self { lookback, entry_sigma, exit_sigma }
    }

    /// Calculate z-score of the latest return relative to history.
    pub fn z_score(&self, returns: &[f64]) -> f64 {
        let window = returns.len().min(self.lookback);
        if window < 2 { return 0.0; }
        let slice = &returns[returns.len() - window..];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
        let std = variance.sqrt();
        if std < 1e-8 { return 0.0; }
        let latest = *slice.last().unwrap_or(&0.0);
        (latest - mean) / std
    }
}

impl TradingPolicy for MeanReversionPolicy {
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal {
        let z = self.z_score(&ctx.recent_returns);

        if z.abs() < self.entry_sigma {
            return PositionSignal::hold(&format!("Z-score {:.2} within entry band", z));
        }

        let size_frac = ((z.abs() - self.entry_sigma) * 0.2).min(ctx.max_position_frac);
        let target_value = ctx.portfolio_value * size_frac;
        let target_units = if ctx.price > 0.0 { target_value / ctx.price } else { 0.0 };

        // Mean reversion: if z > entry_sigma (overvalued), sell; if z < -entry_sigma, buy
        let direction = if z > 0.0 { "Short (overvalued)" } else { "Long (undervalued)" };
        PositionSignal {
            target_units: if z > 0.0 { -target_units } else { target_units },
            confidence: size_frac,
            reason: format!("{} MR: z={:.2}, size={:.2}%", direction, z, size_frac * 100.0),
        }
    }

    fn name(&self) -> &str { "Mean Reversion" }
}

/// Adaptive policy that switches strategy based on market regime.
#[derive(Debug)]
pub struct AdaptivePolicy {
    policies: Vec<Box<dyn TradingPolicy>>,
    /// Performance history per policy (rolling scores).
    policy_scores: Vec<f64>,
    /// Score decay factor.
    decay: f64,
    /// Minimum score difference to switch.
    switch_threshold: f64,
}

impl AdaptivePolicy {
    pub fn new(policies: Vec<Box<dyn TradingPolicy>>) -> Self {
        let n = policies.len();
        Self {
            policies,
            policy_scores: vec![0.0; n],
            decay: 0.95,
            switch_threshold: 0.05,
        }
    }

    /// Update policy scores based on realized returns.
    pub fn update_scores(&mut self, policy_idx: usize, realized_return: f64) {
        if policy_idx < self.policy_scores.len() {
            self.policy_scores[policy_idx] =
                self.policy_scores[policy_idx] * self.decay + realized_return;
        }
    }

    /// Get the index of the currently best-performing policy.
    pub fn best_policy_idx(&self) -> usize {
        self.policy_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl TradingPolicy for AdaptivePolicy {
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal {
        let best_idx = self.best_policy_idx();
        self.policies
            .get(best_idx)
            .map(|p| p.compute_position(ctx))
            .unwrap_or(PositionSignal::hold("No policies available"))
    }

    fn name(&self) -> &str { "Adaptive" }
}

/// Policy chain: compose multiple policies sequentially.
#[derive(Debug)]
pub struct PolicyChain {
    policies: Vec<Box<dyn TradingPolicy>>,
    /// How to combine: "first" uses first non-hold, "vote" takes majority, "min_size" takes smallest.
    combination: ChainCombination,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChainCombination {
    FirstNonHold,
    SmallestPosition,
    AverageSignal,
}

impl PolicyChain {
    pub fn new(policies: Vec<Box<dyn TradingPolicy>>, combination: ChainCombination) -> Self {
        Self { policies, combination }
    }
}

impl TradingPolicy for PolicyChain {
    fn compute_position(&self, ctx: &PolicyContext) -> PositionSignal {
        let signals: Vec<PositionSignal> = self.policies.iter().map(|p| p.compute_position(ctx)).collect();

        match &self.combination {
            ChainCombination::FirstNonHold => {
                signals.into_iter().find(|s| s.target_units.abs() > 1e-8)
                    .unwrap_or(PositionSignal::hold("All policies say hold"))
            }
            ChainCombination::SmallestPosition => {
                signals.into_iter()
                    .min_by(|a, b| a.target_units.abs().partial_cmp(&b.target_units.abs()).unwrap())
                    .unwrap_or(PositionSignal::hold("No signals"))
            }
            ChainCombination::AverageSignal => {
                let avg_units: f64 = signals.iter().map(|s| s.target_units).sum::<f64>() / signals.len().max(1) as f64;
                let avg_conf: f64 = signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len().max(1) as f64;
                PositionSignal {
                    target_units: avg_units,
                    confidence: avg_conf,
                    reason: format!("Chain avg: units={:.2}, conf={:.2}", avg_units, avg_conf),
                }
            }
        }
    }

    fn name(&self) -> &str { "Policy Chain" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_context(signal: f64) -> PolicyContext {
        let mut ctx = PolicyContext::new(100.0, 100_000.0);
        ctx.signal_strength = signal;
        ctx.recent_returns = vec![0.01, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, 0.05, -0.01, 0.03];
        ctx.volatility = 0.15;
        ctx
    }

    #[test]
    fn test_kelly_policy() {
        let kelly = KellyPolicy::half_kelly();
        let signal = kelly.compute_position(&sample_context(0.5));
        assert!(signal.target_units > 0.0);
        assert!(signal.confidence > 0.0);
    }

    #[test]
    fn test_kelly_hold() {
        let kelly = KellyPolicy::new(0.5);
        let ctx = PolicyContext::new(100.0, 100_000.0); // no signal, no returns
        let signal = kelly.compute_position(&ctx);
        // With no recent returns, there's insufficient data for Kelly sizing → hold
        assert_eq!(signal.target_units, 0.0);
    }

    #[test]
    fn test_momentum() {
        let mom = MomentumPolicy::new(5, 0.01);
        let signal = mom.compute_position(&sample_context(0.3));
        // With positive returns sum, should be long
        assert!(signal.target_units >= 0.0);
    }

    #[test]
    fn test_mean_reversion() {
        let mr = MeanReversionPolicy::new(10, 1.5, 0.5);
        let signal = mr.compute_position(&sample_context(0.0));
        // z-score of sample returns
        println!("z-score: {}, signal: {:?}", mr.z_score(&sample_context(0.0).recent_returns), signal);
    }

    #[test]
    fn test_risk_parity() {
        let rp = RiskParityPolicy::new(0.10);
        let signal = rp.compute_position(&sample_context(0.5));
        assert!(signal.reason.contains("RiskParity"));
    }

    #[test]
    fn test_adaptive_policy() {
        let policies: Vec<Box<dyn TradingPolicy>> = vec![
            Box::new(MomentumPolicy::new(5, 0.01)),
            Box::new(MeanReversionPolicy::new(10, 1.5, 0.5)),
        ];
        let mut adaptive = AdaptivePolicy::new(policies);
        let signal = adaptive.compute_position(&sample_context(0.3));
        assert!(signal.reason.len() > 0);
    }

    #[test]
    fn test_kelly_fraction_calc() {
        let f = KellyPolicy::kelly_fraction_from_prob(0.6, 2.0);
        assert!((f - 0.4).abs() < 0.01); // f = p - (1-p)/b = 0.6 - 0.4/2 = 0.4
    }
}
