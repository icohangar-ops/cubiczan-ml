//! # Commodity Risk Metrics
//!
//! Computes risk metrics for commodity investments including Value at Risk
//! (historical simulation), Conditional VaR / Expected Shortfall, max drawdown,
//! correlation, and volatility regime detection.

use crate::types::{CommodityType, PricePoint, RiskMetricsSummary, VolatilityRegime};

// ---------------------------------------------------------------------------
// Commodity Risk Analyzer
// ---------------------------------------------------------------------------

/// Analyzer for computing risk metrics on commodity price series.
pub struct CommodityRiskAnalyzer {
    /// Annualization factor (default 252 trading days).
    pub trading_days_per_year: usize,
    /// Risk-free rate for Sharpe ratio computation.
    pub risk_free_rate: f64,
    /// Confidence level for VaR/CVaR (default 0.95).
    pub var_confidence: f64,
    /// Rolling window for volatility calculation.
    pub volatility_window: usize,
}

impl Default for CommodityRiskAnalyzer {
    fn default() -> Self {
        CommodityRiskAnalyzer {
            trading_days_per_year: 252,
            risk_free_rate: 0.04,
            var_confidence: 0.95,
            volatility_window: 20,
        }
    }
}

impl CommodityRiskAnalyzer {
    /// Create a new risk analyzer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Return computation helpers
    // -----------------------------------------------------------------------

    /// Compute simple returns from close prices.
    fn compute_returns(prices: &[PricePoint]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }
        (1..prices.len())
            .map(|i| {
                if prices[i - 1].close.abs() < 1e-15 { 0.0 }
                else { (prices[i].close - prices[i - 1].close) / prices[i - 1].close }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Value at Risk (Historical Simulation)
    // -----------------------------------------------------------------------

    /// Compute Value at Risk using historical simulation.
    /// Sorts returns and takes the percentile.
    /// Returns VaR as a positive number (loss amount).
    pub fn value_at_risk(&self, prices: &[PricePoint]) -> f64 {
        let returns = Self::compute_returns(prices);
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - self.var_confidence) * (sorted.len() as f64 - 1.0)).round() as usize;
        let var_value = sorted[index.min(sorted.len() - 1)];

        // VaR is a loss, so we return the absolute value of the negative
        -var_value.max(0.0) // return positive loss amount
    }

    // -----------------------------------------------------------------------
    // CVaR / Expected Shortfall
    // -----------------------------------------------------------------------

    /// Compute Conditional VaR (Expected Shortfall).
    /// Average of all returns below the VaR threshold.
    /// Returns CVaR as a positive number.
    pub fn conditional_var(&self, prices: &[PricePoint]) -> f64 {
        let returns = Self::compute_returns(prices);
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let cutoff_index = ((1.0 - self.var_confidence) * (sorted.len() as f64)).ceil() as usize;
        let cutoff_index = cutoff_index.min(sorted.len());

        if cutoff_index == 0 {
            return 0.0;
        }

        let tail: &[f64] = &sorted[..cutoff_index];
        let avg_tail = tail.iter().sum::<f64>() / tail.len() as f64;

        -avg_tail.max(0.0) // return positive loss
    }

    // -----------------------------------------------------------------------
    // Max Drawdown
    // -----------------------------------------------------------------------

    /// Compute maximum drawdown from peak to trough.
    /// Returns drawdown as a positive fraction (e.g., 0.15 = 15% drawdown).
    pub fn max_drawdown(&self, prices: &[PricePoint]) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }

        let mut peak = f64::NEG_INFINITY;
        let mut max_dd = 0.0;

        for p in prices {
            if p.close > peak {
                peak = p.close;
            }
            if peak.abs() > 1e-15 {
                let dd = (peak - p.close) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
        }

        max_dd
    }

    // -----------------------------------------------------------------------
    // Correlation
    // -----------------------------------------------------------------------

    /// Compute simple Pearson correlation between two price series.
    /// Both series must have the same length.
    pub fn correlation(&self, prices_a: &[PricePoint], prices_b: &[PricePoint]) -> f64 {
        let returns_a = Self::compute_returns(prices_a);
        let returns_b = Self::compute_returns(prices_b);

        if returns_a.len() != returns_b.len() || returns_a.is_empty() {
            return 0.0;
        }

        let n = returns_a.len() as f64;
        let mean_a = returns_a.iter().sum::<f64>() / n;
        let mean_b = returns_b.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for (a, b) in returns_a.iter().zip(returns_b.iter()) {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom.abs() < 1e-15 {
            return 0.0;
        }

        (cov / denom).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // Volatility Regime Detection
    // -----------------------------------------------------------------------

    /// Detect the current volatility regime based on rolling std dev percentiles.
    /// Computes rolling std devs over the full history and classifies the current
    /// volatility as low/medium/high/extreme.
    pub fn volatility_regime(&self, prices: &[PricePoint]) -> VolatilityRegime {
        let returns = Self::compute_returns(prices);
        if returns.len() < self.volatility_window + 1 {
            return VolatilityRegime::Medium;
        }

        // Compute rolling std devs
        let window = self.volatility_window;
        let mut rolling_stds: Vec<f64> = Vec::new();
        for i in window..=returns.len() {
            let slice = &returns[i - window..i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
            rolling_stds.push(var.sqrt());
        }

        if rolling_stds.is_empty() {
            return VolatilityRegime::Medium;
        }

        // Current volatility (annualized)
        let _current_std = *rolling_stds.last().unwrap() * (self.trading_days_per_year as f64).sqrt();

        // Classify based on the distribution of historical rolling std devs
        let mut sorted = rolling_stds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p25 = sorted[((sorted.len() as f64 * 0.25) as usize).min(sorted.len() - 1)];
        let p75 = sorted[((sorted.len() as f64 * 0.75) as usize).min(sorted.len() - 1)];
        let current_annualized = *rolling_stds.last().unwrap();

        if current_annualized <= p25 {
            VolatilityRegime::Low
        } else if current_annualized <= p75 {
            VolatilityRegime::Medium
        } else {
            // Check if it's extreme (above p95)
            let p95 = sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1)];
            if current_annualized > p95 {
                VolatilityRegime::Extreme
            } else {
                VolatilityRegime::High
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full Risk Summary
    // -----------------------------------------------------------------------

    /// Compute a full `RiskMetricsSummary` for a commodity price series.
    pub fn full_risk_summary(&self, _commodity: CommodityType, prices: &[PricePoint]) -> RiskMetricsSummary {
        let returns = Self::compute_returns(prices);

        // Annualized volatility
        let daily_vol = if returns.len() > 1 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };
        let annualized_vol = daily_vol * (self.trading_days_per_year as f64).sqrt();

        // VaR and CVaR
        let var_95 = -self.value_at_risk(prices); // negative because it's a loss
        let cvar_95 = -self.conditional_var(prices);

        // Max drawdown (negative to indicate loss)
        let max_dd = -self.max_drawdown(prices);

        // Volatility regime
        let regime = self.volatility_regime(prices);

        // Sharpe ratio
        let annualized_return = if returns.is_empty() {
            0.0
        } else {
            let mean_daily = returns.iter().sum::<f64>() / returns.len() as f64;
            mean_daily * self.trading_days_per_year as f64
        };
        let excess_return = annualized_return - self.risk_free_rate;
        let sharpe = if annualized_vol.abs() < 1e-15 {
            0.0
        } else {
            excess_return / annualized_vol
        };

        RiskMetricsSummary {
            volatility: annualized_vol,
            var_95,
            cvar_95,
            max_drawdown: max_dd,
            volatility_regime: regime.to_string(),
            sharpe_ratio: sharpe,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn make_rising_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = 100.0 + i as f64 * 0.5;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 0.3, price + 0.5, price - 0.5, price, 1000.0,
                )
            })
            .collect()
    }

    fn make_volatile_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = 100.0 + 15.0 * (i as f64 * 0.3).sin();
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 2.0, price + 3.0, price - 3.0, price, 1000.0,
                )
            })
            .collect()
    }

    fn make_flat_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    100.0, 100.2, 99.8, 100.0, 1000.0,
                )
            })
            .collect()
    }

    fn make_prices_with_drawdown(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = if i < n / 2 {
                    100.0 + i as f64 * 2.0 // rise
                } else {
                    100.0 + (n / 2) as f64 * 2.0 - (i - n / 2) as f64 * 3.0 // sharp fall
                };
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 0.5, price + 1.0, price - 1.0, price, 1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_var_positive() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_volatile_prices(100);
        let var = analyzer.value_at_risk(&prices);
        assert!(var >= 0.0);
    }

    #[test]
    fn test_var_empty() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = vec![PricePoint::new(Utc::now(), 100.0, 101.0, 99.0, 100.0, 1000.0)];
        let var = analyzer.value_at_risk(&prices);
        assert!((var - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cvar_greater_than_var() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_volatile_prices(100);
        let var = analyzer.value_at_risk(&prices);
        let cvar = analyzer.conditional_var(&prices);
        assert!(cvar >= var);
    }

    #[test]
    fn test_cvar_empty() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = vec![PricePoint::new(Utc::now(), 100.0, 101.0, 99.0, 100.0, 1000.0)];
        let cvar = analyzer.conditional_var(&prices);
        assert!((cvar - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown_positive() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_prices_with_drawdown(50);
        let dd = analyzer.max_drawdown(&prices);
        assert!(dd > 0.0);
        // Should be significant since we have a sharp fall
        assert!(dd > 0.1);
    }

    #[test]
    fn test_max_drawdown_flat() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_flat_prices(50);
        let dd = analyzer.max_drawdown(&prices);
        // Flat prices → near-zero drawdown
        assert!(dd < 0.01);
    }

    #[test]
    fn test_max_drawdown_empty() {
        let analyzer = CommodityRiskAnalyzer::new();
        let dd = analyzer.max_drawdown(&[]);
        assert!((dd - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown_no_decline() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_rising_prices(50);
        let dd = analyzer.max_drawdown(&prices);
        assert!(dd < 0.01);
    }

    #[test]
    fn test_correlation_same_series() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_volatile_prices(50);
        let corr = analyzer.correlation(&prices, &prices);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_range() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices_a = make_volatile_prices(50);
        let prices_b = make_rising_prices(50);
        let corr = analyzer.correlation(&prices_a, &prices_b);
        assert!(corr >= -1.0 && corr <= 1.0);
    }

    #[test]
    fn test_correlation_different_lengths() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices_a = make_volatile_prices(50);
        let prices_b = make_volatile_prices(30);
        let corr = analyzer.correlation(&prices_a, &prices_b);
        assert!((corr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_empty() {
        let analyzer = CommodityRiskAnalyzer::new();
        let corr = analyzer.correlation(&[], &[]);
        assert!((corr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_regime_volatile() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_volatile_prices(100);
        let regime = analyzer.volatility_regime(&prices);
        // Volatile data should not be "Low"
        assert_ne!(regime, VolatilityRegime::Low);
    }

    #[test]
    fn test_volatility_regime_flat() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_flat_prices(100);
        let regime = analyzer.volatility_regime(&prices);
        // Flat data should be "Low"
        assert_eq!(regime, VolatilityRegime::Low);
    }

    #[test]
    fn test_volatility_regime_insufficient_data() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_flat_prices(10);
        let regime = analyzer.volatility_regime(&prices);
        assert_eq!(regime, VolatilityRegime::Medium);
    }

    #[test]
    fn test_full_risk_summary() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_volatile_prices(100);
        let summary = analyzer.full_risk_summary(CommodityType::Gold, &prices);
        assert!(summary.volatility >= 0.0);
        assert!(summary.var_95 <= 0.0); // negative loss
        assert!(summary.cvar_95 <= 0.0); // negative loss
        assert!(summary.max_drawdown <= 0.0); // negative loss
        assert!(!summary.volatility_regime.is_empty());
    }

    #[test]
    fn test_full_risk_summary_sharpe() {
        let analyzer = CommodityRiskAnalyzer::new();
        let prices = make_rising_prices(100);
        let summary = analyzer.full_risk_summary(CommodityType::Gold, &prices);
        // Steadily rising prices → positive Sharpe
        assert!(summary.sharpe_ratio > 0.0);
    }
}
