//! # Integrated Risk Management
//!
//! Portfolio-level and position-level risk monitoring with circuit breaker logic,
//! VaR/CVaR computation, drawdown control, leverage control, and risk budgeting.

use crate::types::*;
use std::collections::HashMap;

/// Risk budgeting strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskBudgetingMode {
    /// Allocate capital equally across positions.
    EqualWeight,
    /// Each position contributes equally to portfolio risk.
    EqualRiskContribution,
}

/// Result of a position-level risk check.
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    /// Whether the position passes all risk limits.
    pub passed: bool,
    /// Descriptions of any violations.
    pub violations: Vec<String>,
    /// Recommended adjusted position size (fraction of portfolio).
    pub adjusted_size: f64,
}

/// Full risk assessment output.
#[derive(Debug, Clone, Default)]
pub struct RiskAssessment {
    /// Value at Risk (as a negative return fraction).
    pub var: f64,
    /// Conditional VaR / Expected Shortfall (as a negative return fraction).
    pub cvar: f64,
    /// Correlation-weighted portfolio risk estimate.
    pub portfolio_risk: f64,
    /// Current drawdown fraction.
    pub drawdown: f64,
    /// Leverage adjustment factor (0..1, multiply by target leverage).
    pub leverage_adjustment: f64,
    /// Drawdown-based exposure adjustment factor (0..1).
    pub drawdown_adjustment: f64,
    /// Whether the circuit breaker is currently active.
    pub circuit_breaker: bool,
    /// Risk budget weights per symbol.
    pub risk_budget_weights: HashMap<String, f64>,
    /// Feedback signals for the control loop.
    pub feedback_signals: Vec<FeedbackSignal>,
}

/// Integrated risk manager for the closed-loop system.
///
/// Provides portfolio-level risk monitoring (VaR, CVaR, drawdown),
/// position-level risk limits, correlation-based risk estimation,
/// drawdown control, leverage control, risk budgeting, and circuit breaker logic.
#[derive(Debug, Clone)]
pub struct IntegratedRiskManager {
    /// Confidence level for VaR/CVaR (e.g. 0.95 for 95th percentile).
    pub var_confidence: f64,
    /// Lookback window for historical simulation.
    pub lookback: usize,
    /// Maximum position size as fraction of portfolio.
    pub max_position_size: f64,
    /// Default stop-loss threshold (fraction of entry price).
    pub default_stop_loss: f64,
    /// Default take-profit threshold (fraction of entry price).
    pub default_take_profit: f64,
    /// Drawdown threshold at which to begin scaling down exposure.
    pub drawdown_threshold: f64,
    /// Maximum allowed drawdown before full halt.
    pub max_drawdown: f64,
    /// Circuit breaker: halt trading if daily loss exceeds this fraction of portfolio.
    pub circuit_breaker_threshold: f64,
    /// Maximum allowed leverage.
    pub max_leverage: f64,
    /// Risk budgeting mode.
    pub risk_budgeting_mode: RiskBudgetingMode,
    /// Whether the circuit breaker is currently tripped.
    pub circuit_breaker_active: bool,
    /// Accumulated daily PnL (for circuit breaker).
    pub daily_pnl: f64,
    /// Historical portfolio returns for VaR/CVaR.
    pub historical_returns: Vec<f64>,
    /// Peak portfolio value (for drawdown tracking).
    pub peak_value: f64,
    /// Per-position historical returns (symbol -> returns).
    pub position_returns: HashMap<String, Vec<f64>>,
}

impl Default for IntegratedRiskManager {
    fn default() -> Self {
        IntegratedRiskManager {
            var_confidence: 0.95,
            lookback: 252,
            max_position_size: 0.25,
            default_stop_loss: 0.05,
            default_take_profit: 0.15,
            drawdown_threshold: 0.10,
            max_drawdown: 0.20,
            circuit_breaker_threshold: 0.05,
            max_leverage: 2.0,
            risk_budgeting_mode: RiskBudgetingMode::EqualWeight,
            circuit_breaker_active: false,
            daily_pnl: 0.0,
            historical_returns: Vec::new(),
            peak_value: 100_000.0,
            position_returns: HashMap::new(),
        }
    }
}

impl IntegratedRiskManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute Value at Risk using historical simulation.
    ///
    /// Returns VaR as a **negative** return fraction (e.g. -0.03 means 3% loss).
    pub fn compute_var(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let window = self.lookback.min(returns.len());
        let recent: Vec<f64> = returns[returns.len() - window..].to_vec();
        let mut sorted = recent.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((1.0 - self.var_confidence) * (sorted.len() as f64 - 1.0)).ceil() as usize;
        let index = index.min(sorted.len() - 1);

        // VaR is the loss at the percentile (negative value = loss)
        sorted[index].min(0.0)
    }

    /// Compute Conditional VaR (Expected Shortfall).
    ///
    /// Returns CVaR as a **negative** return fraction. This is the average of all
    /// returns at or below the VaR level.
    pub fn compute_cvar(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let window = self.lookback.min(returns.len());
        let recent: Vec<f64> = returns[returns.len() - window..].to_vec();
        let mut sorted = recent.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let cutoff_index = ((1.0 - self.var_confidence) * (sorted.len() as f64 - 1.0)).ceil() as usize;
        let cutoff_index = cutoff_index.min(sorted.len() - 1);

        let tail: &[f64] = &sorted[..=cutoff_index];
        if tail.is_empty() {
            return 0.0;
        }
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        mean.min(0.0)
    }

    /// Update peak value and return current drawdown.
    ///
    /// Drawdown = (peak - current) / peak.
    pub fn update_drawdown(&mut self, current_value: f64) -> f64 {
        if current_value > self.peak_value {
            self.peak_value = current_value;
        }
        if self.peak_value < 1e-15 {
            return 0.0;
        }
        let dd = (self.peak_value - current_value) / self.peak_value;
        dd.max(0.0)
    }

    /// Check whether a position passes all risk limits.
    ///
    /// Checks: max position size, stop-loss, take-profit.
    pub fn check_position_limit(&self, position: &Position) -> RiskCheckResult {
        let mut violations = Vec::new();
        let mut adjusted = position.weight;

        // Check max position size
        if position.weight > self.max_position_size {
            violations.push(format!(
                "Position {} weight {:.2} exceeds max {:.2}",
                position.symbol, position.weight, self.max_position_size
            ));
            adjusted = self.max_position_size;
        }

        // Check stop-loss
        let pnl_pct = position.unrealized_pnl_pct();
        if pnl_pct < -self.default_stop_loss {
            violations.push(format!(
                "Position {} PnL {:.2}% exceeds stop-loss -{:.2}%",
                position.symbol,
                pnl_pct * 100.0,
                self.default_stop_loss * 100.0
            ));
        }

        // Check take-profit
        if pnl_pct > self.default_take_profit {
            violations.push(format!(
                "Position {} PnL {:.2}% exceeds take-profit +{:.2}%",
                position.symbol,
                pnl_pct * 100.0,
                self.default_take_profit * 100.0
            ));
        }

        RiskCheckResult {
            passed: violations.is_empty(),
            violations,
            adjusted_size: adjusted,
        }
    }

    /// Compute correlation-weighted portfolio risk estimate.
    ///
    /// Uses average pairwise correlation and individual position volatilities
    /// to estimate overall portfolio risk.
    pub fn portfolio_risk(&self, portfolio: &PortfolioState) -> f64 {
        let n = portfolio.positions.len();
        if n == 0 {
            return 0.0;
        }

        if n == 1 {
            // Single position: risk = weight * volatility estimate
            return portfolio.positions[0].weight;
        }

        // Compute average pairwise correlation from stored returns
        let avg_corr = self.average_position_correlation();

        // Portfolio risk: sqrt(w^T * Sigma * w) simplified
        // Using the approximation: sigma_p^2 ≈ sum(w_i^2 * sigma_i^2) + 2*sum(w_i*w_j*rho*sigma_i*sigma_j)
        // Simplified: use weight as proxy for volatility, correlation from stored data
        let mut risk = 0.0_f64;
        for i in 0..n {
            let wi = portfolio.positions[i].weight;
            risk += wi * wi;
            for j in (i + 1)..n {
                let wj = portfolio.positions[j].weight;
                risk += 2.0 * wi * wj * avg_corr;
            }
        }

        risk.sqrt()
    }

    /// Compute average pairwise correlation across tracked positions.
    fn average_position_correlation(&self) -> f64 {
        let symbols: Vec<&String> = self.position_returns.keys().collect();
        if symbols.len() < 2 {
            return 0.5; // Default moderate correlation
        }

        let mut total_corr = 0.0_f64;
        let mut pairs = 0usize;

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let a = &self.position_returns[symbols[i]];
                let b = &self.position_returns[symbols[j]];
                let c = pearson_correlation(a, b);
                if c.is_finite() {
                    total_corr += c.abs();
                    pairs += 1;
                }
            }
        }

        if pairs == 0 {
            return 0.5;
        }
        total_corr / pairs as f64
    }

    /// Drawdown control: linearly scale down exposure when drawdown exceeds threshold.
    ///
    /// Returns a factor in [0, 1]:
    /// - 1.0 when drawdown ≤ threshold
    /// - Linearly decreasing to 0.0 when drawdown = max_drawdown
    pub fn drawdown_adjustment(&self, current_drawdown: f64) -> f64 {
        if current_drawdown <= self.drawdown_threshold {
            return 1.0;
        }
        if current_drawdown >= self.max_drawdown {
            return 0.0;
        }
        // Linear interpolation between 1.0 and 0.0
        let range = self.max_drawdown - self.drawdown_threshold;
        if range < 1e-15 {
            return 0.0;
        }
        let fraction = (current_drawdown - self.drawdown_threshold) / range;
        1.0 - fraction
    }

    /// Leverage control: reduce allowed leverage in volatile regimes.
    ///
    /// Returns a factor in [0, 1] to multiply the target leverage by.
    pub fn leverage_adjustment(&self, vol_regime: VolatilityRegime) -> f64 {
        match vol_regime {
            VolatilityRegime::Low => 1.0,
            VolatilityRegime::Medium => 0.85,
            VolatilityRegime::High => 0.6,
            VolatilityRegime::Extreme => 0.3,
        }
    }

    /// Compute risk budget weights across positions.
    ///
    /// Returns a HashMap from symbol to target weight fraction.
    pub fn compute_risk_budget(&self, portfolio: &PortfolioState) -> HashMap<String, f64> {
        let n = portfolio.positions.len();
        if n == 0 {
            return HashMap::new();
        }

        match self.risk_budgeting_mode {
            RiskBudgetingMode::EqualWeight => {
                let weight = 1.0 / n as f64;
                portfolio.positions.iter()
                    .map(|p| (p.symbol.clone(), weight))
                    .collect()
            }
            RiskBudgetingMode::EqualRiskContribution => {
                self.equal_risk_contribution(portfolio)
            }
        }
    }

    /// Compute equal risk contribution weights.
    ///
    /// Each position is weighted inversely proportional to its estimated volatility,
    /// so that each position contributes roughly the same amount of risk.
    fn equal_risk_contribution(&self, portfolio: &PortfolioState) -> HashMap<String, f64> {
        let mut volatilities: Vec<f64> = Vec::new();

        for pos in &portfolio.positions {
            let vol = self.position_returns
                .get(&pos.symbol)
                .map(|rets| self.estimate_volatility(rets))
                .unwrap_or(0.15); // Default 15% volatility

            volatilities.push(vol.max(1e-6));
        }

        // Inverse volatility weighting
        let inv_vol_sum: f64 = volatilities.iter().map(|v| 1.0 / v).sum();
        if inv_vol_sum < 1e-15 {
            let weight = 1.0 / portfolio.positions.len() as f64;
            return portfolio.positions.iter()
                .map(|p| (p.symbol.clone(), weight))
                .collect();
        }

        portfolio.positions.iter().zip(volatilities.iter())
            .map(|(pos, vol)| (pos.symbol.clone(), (1.0 / vol) / inv_vol_sum))
            .collect()
    }

    /// Estimate volatility from a returns series.
    fn estimate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        variance.sqrt() * 252.0_f64.sqrt() // Annualized
    }

    /// Check and update circuit breaker status.
    ///
    /// Returns `true` if the circuit breaker is now active (trading should halt).
    pub fn check_circuit_breaker(&mut self, daily_pnl: f64) -> bool {
        self.daily_pnl += daily_pnl;
        if self.daily_pnl < -(self.circuit_breaker_threshold) {
            self.circuit_breaker_active = true;
        }
        self.circuit_breaker_active
    }

    /// Reset daily PnL (call at start of each day).
    pub fn reset_daily_pnl(&mut self) {
        self.daily_pnl = 0.0;
        self.circuit_breaker_active = false;
    }

    /// Reset circuit breaker (manual override).
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker_active = false;
        self.daily_pnl = 0.0;
    }

    /// Record a portfolio return for VaR/CVaR computation.
    pub fn record_return(&mut self, ret: f64) {
        self.historical_returns.push(ret);
        // Keep bounded
        let max_len = self.lookback * 2;
        if self.historical_returns.len() > max_len {
            self.historical_returns.drain(..self.historical_returns.len() - max_len);
        }
    }

    /// Record a position return for correlation tracking.
    pub fn record_position_return(&mut self, symbol: &str, ret: f64) {
        let returns = self.position_returns.entry(symbol.to_string()).or_insert_with(Vec::new);
        returns.push(ret);
        let max_len = self.lookback * 2;
        if returns.len() > max_len {
            returns.drain(..returns.len() - max_len);
        }
    }

    /// Perform a full risk assessment given the current portfolio state and observation.
    ///
    /// Returns a [`RiskAssessment`] with all risk metrics and feedback signals.
    pub fn assess(
        &mut self,
        portfolio: &PortfolioState,
        observation: &MarketObservation,
    ) -> RiskAssessment {
        let var = self.compute_var(&self.historical_returns);
        let cvar = self.compute_cvar(&self.historical_returns);
        let pf_risk = self.portfolio_risk(portfolio);
        let drawdown = portfolio.max_drawdown;
        let lev_adj = self.leverage_adjustment(observation.volatility_regime);
        let dd_adj = self.drawdown_adjustment(drawdown);

        let risk_budget = self.compute_risk_budget(portfolio);

        let mut feedback_signals = Vec::new();

        // VaR feedback
        feedback_signals.push(FeedbackSignal::new(
            "var",
            var.abs(),
            0.0,
            -var.abs() * 0.5,
        ));

        // Drawdown feedback
        feedback_signals.push(FeedbackSignal::new(
            "drawdown",
            drawdown,
            self.drawdown_threshold,
            dd_adj - 1.0,
        ));

        // Leverage feedback
        feedback_signals.push(FeedbackSignal::new(
            "leverage_adjustment",
            lev_adj,
            1.0,
            lev_adj - 1.0,
        ));

        // Portfolio risk feedback
        feedback_signals.push(FeedbackSignal::new(
            "portfolio_risk",
            pf_risk,
            0.5,
            (0.5 - pf_risk) * 0.3,
        ));

        RiskAssessment {
            var,
            cvar,
            portfolio_risk: pf_risk,
            drawdown,
            leverage_adjustment: lev_adj,
            drawdown_adjustment: dd_adj,
            circuit_breaker: self.circuit_breaker_active,
            risk_budget_weights: risk_budget,
            feedback_signals,
        }
    }

    /// Compute stop-loss price for a position.
    pub fn stop_loss_price(&self, position: &Position) -> f64 {
        match position.quantity.signum() {
            s if s > 0.0 => position.avg_entry * (1.0 - self.default_stop_loss),
            s if s < 0.0 => position.avg_entry * (1.0 + self.default_stop_loss),
            _ => position.avg_entry,
        }
    }

    /// Compute take-profit price for a position.
    pub fn take_profit_price(&self, position: &Position) -> f64 {
        match position.quantity.signum() {
            s if s > 0.0 => position.avg_entry * (1.0 + self.default_take_profit),
            s if s < 0.0 => position.avg_entry * (1.0 - self.default_take_profit),
            _ => position.avg_entry,
        }
    }
}

/// Pearson correlation between two slices of equal or different lengths.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mb: f64 = b[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn sample_returns() -> Vec<f64> {
        // 50 daily returns, mostly small with a few tail events
        vec![
            0.01, -0.005, 0.015, 0.008, -0.012, 0.003, -0.008, 0.011, -0.002, 0.006,
            0.004, -0.007, 0.009, -0.003, 0.012, -0.006, 0.001, -0.009, 0.007, -0.004,
            0.005, -0.011, 0.008, -0.001, 0.010, -0.005, 0.006, -0.008, 0.003, -0.007,
            0.002, -0.004, 0.009, -0.006, 0.011, -0.003, 0.007, -0.010, 0.004, -0.002,
            0.013, -0.009, 0.005, -0.006, 0.008, -0.004, 0.001, -0.003, 0.006, -0.05,
        ]
    }

    fn make_position(symbol: &str, weight: f64, pnl_pct: f64) -> Position {
        let avg_entry = 100.0;
        let current = avg_entry * (1.0 + pnl_pct);
        Position {
            symbol: symbol.to_string(),
            quantity: weight * 100_000.0 / current,
            avg_entry,
            current_price: current,
            unrealized_pnl: weight * 100_000.0 * pnl_pct,
            weight,
        }
    }

    fn make_portfolio(positions: Vec<Position>) -> PortfolioState {
        let positions_value: f64 = positions.iter().map(|p| p.market_value()).sum();
        let cash = 100_000.0 - positions_value;
        PortfolioState {
            total_value: 100_000.0,
            cash: cash.max(0.0),
            positions,
            leverage: 1.0,
            exposure: positions_value / 100_000.0,
            max_drawdown: 0.05,
            sharpe: 1.0,
            timestamp: Utc::now(),
        }
    }

    fn make_observation(vol_regime: VolatilityRegime) -> MarketObservation {
        MarketObservation {
            regime: MarketRegime::Trending,
            price: 100.0,
            volatility_regime: vol_regime,
            volatility: 0.15,
            trend_strength: 0.5,
            trend_direction: 0.3,
            liquidity_score: 0.7,
            avg_correlation: 0.3,
            risk_event_detected: false,
            risk_event_description: None,
            recent_returns: sample_returns(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_var_with_returns() {
        let rm = IntegratedRiskManager::new();
        let var = rm.compute_var(&sample_returns());
        assert!(var <= 0.0, "VaR should be negative (loss)");
        assert!(var >= -0.1, "VaR should not exceed 10% loss for this data");
    }

    #[test]
    fn test_var_empty_returns() {
        let rm = IntegratedRiskManager::new();
        let var = rm.compute_var(&[]);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn test_cvar_with_returns() {
        let rm = IntegratedRiskManager::new();
        let cvar = rm.compute_cvar(&sample_returns());
        assert!(cvar <= 0.0, "CVaR should be negative");
    }

    #[test]
    fn test_cvar_more_extreme_than_var() {
        let rm = IntegratedRiskManager::new();
        let var = rm.compute_var(&sample_returns());
        let cvar = rm.compute_cvar(&sample_returns());
        assert!(cvar <= var, "CVaR should be at least as extreme as VaR");
    }

    #[test]
    fn test_cvar_empty_returns() {
        let rm = IntegratedRiskManager::new();
        let cvar = rm.compute_cvar(&[]);
        assert_eq!(cvar, 0.0);
    }

    #[test]
    fn test_drawdown_tracking_increasing() {
        let mut rm = IntegratedRiskManager::new();
        rm.peak_value = 100_000.0;
        let dd = rm.update_drawdown(110_000.0);
        assert_eq!(dd, 0.0);
        assert_eq!(rm.peak_value, 110_000.0);
    }

    #[test]
    fn test_drawdown_tracking_decreasing() {
        let mut rm = IntegratedRiskManager::new();
        rm.peak_value = 100_000.0;
        let dd = rm.update_drawdown(90_000.0);
        assert!(close(dd, 0.1, 1e-10));
    }

    #[test]
    fn test_drawdown_adjustment_below_threshold() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.drawdown_adjustment(0.05);
        assert!(close(adj, 1.0, 1e-10));
    }

    #[test]
    fn test_drawdown_adjustment_at_threshold() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.drawdown_adjustment(rm.drawdown_threshold);
        assert!(close(adj, 1.0, 1e-10));
    }

    #[test]
    fn test_drawdown_adjustment_above_threshold() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.drawdown_adjustment(0.15);
        assert!(adj < 1.0 && adj > 0.0, "Expected partial reduction, got {}", adj);
    }

    #[test]
    fn test_drawdown_adjustment_at_max() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.drawdown_adjustment(rm.max_drawdown);
        assert!(close(adj, 0.0, 1e-10));
    }

    #[test]
    fn test_drawdown_adjustment_exceeds_max() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.drawdown_adjustment(0.30);
        assert!(close(adj, 0.0, 1e-10));
    }

    #[test]
    fn test_position_check_within_limits() {
        let rm = IntegratedRiskManager::new();
        let pos = make_position("BTC", 0.10, 0.02);
        let result = rm.check_position_limit(&pos);
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_position_check_exceeds_max_size() {
        let rm = IntegratedRiskManager::new();
        let pos = make_position("BTC", 0.50, 0.02);
        let result = rm.check_position_limit(&pos);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("exceeds max")));
        assert!(close(result.adjusted_size, rm.max_position_size, 1e-10));
    }

    #[test]
    fn test_position_check_stop_loss() {
        let rm = IntegratedRiskManager::new();
        let pos = make_position("BTC", 0.10, -0.08);
        let result = rm.check_position_limit(&pos);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("stop-loss")));
    }

    #[test]
    fn test_position_check_take_profit() {
        let rm = IntegratedRiskManager::new();
        let pos = make_position("BTC", 0.10, 0.20);
        let result = rm.check_position_limit(&pos);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("take-profit")));
    }

    #[test]
    fn test_leverage_adjustment_low_vol() {
        let rm = IntegratedRiskManager::new();
        assert!(close(rm.leverage_adjustment(VolatilityRegime::Low), 1.0, 1e-10));
    }

    #[test]
    fn test_leverage_adjustment_high_vol() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.leverage_adjustment(VolatilityRegime::High);
        assert!(adj < 1.0 && adj > 0.0);
    }

    #[test]
    fn test_leverage_adjustment_extreme_vol() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.leverage_adjustment(VolatilityRegime::Extreme);
        assert!(close(adj, 0.3, 1e-10));
    }

    #[test]
    fn test_leverage_adjustment_medium_vol() {
        let rm = IntegratedRiskManager::new();
        let adj = rm.leverage_adjustment(VolatilityRegime::Medium);
        assert!(close(adj, 0.85, 1e-10));
    }

    #[test]
    fn test_circuit_breaker_not_triggered() {
        let mut rm = IntegratedRiskManager::new();
        assert!(!rm.check_circuit_breaker(-0.02));
        assert!(!rm.circuit_breaker_active);
    }

    #[test]
    fn test_circuit_breaker_triggered() {
        let mut rm = IntegratedRiskManager::new();
        assert!(rm.check_circuit_breaker(-0.06));
        assert!(rm.circuit_breaker_active);
    }

    #[test]
    fn test_circuit_breaker_accumulated() {
        let mut rm = IntegratedRiskManager::new();
        rm.check_circuit_breaker(-0.03); // Not enough
        assert!(!rm.circuit_breaker_active);
        rm.check_circuit_breaker(-0.03); // Total -0.06 > threshold
        assert!(rm.circuit_breaker_active);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut rm = IntegratedRiskManager::new();
        rm.check_circuit_breaker(-0.10);
        assert!(rm.circuit_breaker_active);
        rm.reset_circuit_breaker();
        assert!(!rm.circuit_breaker_active);
        assert!(close(rm.daily_pnl, 0.0, 1e-10));
    }

    #[test]
    fn test_daily_pnl_reset() {
        let mut rm = IntegratedRiskManager::new();
        rm.daily_pnl = -0.03;
        rm.reset_daily_pnl();
        assert!(close(rm.daily_pnl, 0.0, 1e-10));
        assert!(!rm.circuit_breaker_active);
    }

    #[test]
    fn test_portfolio_risk_empty() {
        let rm = IntegratedRiskManager::new();
        let portfolio = PortfolioState::default();
        assert!(close(rm.portfolio_risk(&portfolio), 0.0, 1e-10));
    }

    #[test]
    fn test_portfolio_risk_single_position() {
        let rm = IntegratedRiskManager::new();
        let positions = vec![make_position("BTC", 0.5, 0.02)];
        let portfolio = make_portfolio(positions);
        let risk = rm.portfolio_risk(&portfolio);
        assert!(risk > 0.0);
    }

    #[test]
    fn test_portfolio_risk_multiple_positions() {
        let rm = IntegratedRiskManager::new();
        let positions = vec![
            make_position("BTC", 0.3, 0.02),
            make_position("ETH", 0.3, -0.01),
            make_position("SOL", 0.2, 0.03),
        ];
        let portfolio = make_portfolio(positions);
        let risk = rm.portfolio_risk(&portfolio);
        assert!(risk > 0.0);
        assert!(risk <= 1.0);
    }

    #[test]
    fn test_risk_budget_equal_weight() {
        let rm = IntegratedRiskManager::new();
        let positions = vec![
            make_position("BTC", 0.3, 0.02),
            make_position("ETH", 0.3, -0.01),
        ];
        let portfolio = make_portfolio(positions);
        let budget = rm.compute_risk_budget(&portfolio);
        assert_eq!(budget.len(), 2);
        assert!(close(budget["BTC"], 0.5, 1e-10));
        assert!(close(budget["ETH"], 0.5, 1e-10));
    }

    #[test]
    fn test_risk_budget_equal_risk_contribution() {
        let mut rm = IntegratedRiskManager::new();
        rm.risk_budgeting_mode = RiskBudgetingMode::EqualRiskContribution;
        // Record different volatilities for positions: BTC is more volatile
        for i in 0..20 {
            rm.record_position_return("BTC", 0.02 * (i as f64 % 5.0 - 2.0).sin());
            rm.record_position_return("ETH", 0.005 * (i as f64 % 5.0 - 2.0).sin());
        }
        let positions = vec![
            make_position("BTC", 0.3, 0.02),
            make_position("ETH", 0.3, -0.01),
        ];
        let portfolio = make_portfolio(positions);
        let budget = rm.compute_risk_budget(&portfolio);
        assert_eq!(budget.len(), 2);
        // BTC has higher volatility, should get lower weight
        assert!(budget["BTC"] < budget["ETH"]);
        // Weights should sum to ~1.0
        let total: f64 = budget.values().sum();
        assert!(close(total, 1.0, 1e-10));
    }

    #[test]
    fn test_risk_budget_empty_portfolio() {
        let rm = IntegratedRiskManager::new();
        let portfolio = PortfolioState::default();
        let budget = rm.compute_risk_budget(&portfolio);
        assert!(budget.is_empty());
    }

    #[test]
    fn test_full_risk_assessment() {
        let mut rm = IntegratedRiskManager::new();
        for r in &sample_returns() {
            rm.record_return(*r);
        }
        let positions = vec![make_position("BTC", 0.2, 0.02)];
        let portfolio = make_portfolio(positions);
        let observation = make_observation(VolatilityRegime::Medium);
        let assessment = rm.assess(&portfolio, &observation);
        assert!(assessment.var <= 0.0);
        assert!(assessment.cvar <= assessment.var);
        assert!(assessment.portfolio_risk > 0.0);
        assert!(!assessment.circuit_breaker);
        assert!(!assessment.feedback_signals.is_empty());
    }

    #[test]
    fn test_full_assessment_circuit_breaker() {
        let mut rm = IntegratedRiskManager::new();
        rm.check_circuit_breaker(-0.10);
        let portfolio = make_portfolio(vec![make_position("BTC", 0.2, 0.02)]);
        let observation = make_observation(VolatilityRegime::Low);
        let assessment = rm.assess(&portfolio, &observation);
        assert!(assessment.circuit_breaker);
    }

    #[test]
    fn test_record_return_bounded() {
        let mut rm = IntegratedRiskManager::new();
        rm.lookback = 10;
        for i in 0..100 {
            rm.record_return(i as f64 * 0.001);
        }
        assert!(rm.historical_returns.len() <= 20); // 2 * lookback
    }

    #[test]
    fn test_record_position_return() {
        let mut rm = IntegratedRiskManager::new();
        rm.record_position_return("BTC", 0.01);
        rm.record_position_return("BTC", -0.005);
        assert_eq!(rm.position_returns["BTC"].len(), 2);
    }

    #[test]
    fn test_stop_loss_price_long() {
        let rm = IntegratedRiskManager::new();
        let pos = make_position("BTC", 0.2, 0.0);
        let sl = rm.stop_loss_price(&pos);
        assert!(close(sl, pos.avg_entry * (1.0 - rm.default_stop_loss), 1e-10));
    }

    #[test]
    fn test_take_profit_price_long() {
        let rm = IntegratedRiskManager::new();
        let pos = make_position("BTC", 0.2, 0.0);
        let tp = rm.take_profit_price(&pos);
        assert!(close(tp, pos.avg_entry * (1.0 + rm.default_take_profit), 1e-10));
    }

    #[test]
    fn test_stop_loss_price_short() {
        let rm = IntegratedRiskManager::new();
        let mut pos = make_position("BTC", 0.2, 0.0);
        pos.quantity = -pos.quantity;
        let sl = rm.stop_loss_price(&pos);
        assert!(close(sl, pos.avg_entry * (1.0 + rm.default_stop_loss), 1e-10));
    }

    #[test]
    fn test_average_correlation_no_positions() {
        let rm = IntegratedRiskManager::new();
        assert!(close(rm.average_position_correlation(), 0.5, 1e-10));
    }

    #[test]
    fn test_average_correlation_with_data() {
        let mut rm = IntegratedRiskManager::new();
        for _ in 0..20 {
            rm.record_position_return("BTC", 0.01);
            rm.record_position_return("ETH", 0.008);
        }
        let corr = rm.average_position_correlation();
        assert!(corr >= 0.0 && corr <= 1.0);
    }

    #[test]
    fn test_default_creation() {
        let rm = IntegratedRiskManager::default();
        assert!(close(rm.var_confidence, 0.95, 1e-10));
        assert!(!rm.circuit_breaker_active);
        assert_eq!(rm.risk_budgeting_mode, RiskBudgetingMode::EqualWeight);
    }
}
