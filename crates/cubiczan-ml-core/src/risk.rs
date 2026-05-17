//! # Risk Management
//!
//! Comprehensive risk management tools for portfolio construction, position sizing,
//! and drawdown control in the CubicZan finance/DeFi ecosystem.
//!
//! ## Components
//!
//! - **Position Sizing** — Kelly criterion, fixed fractional, volatility-adjusted methods
//! - **Risk Metrics** — Tracking error, information ratio, beta, alpha, Jensen's alpha
//! - **Drawdown Tracking** — Real-time drawdown monitoring with configurable thresholds
//! - **Portfolio Construction** — Equal weight, inverse volatility, risk parity helpers
//! - **Stop-Loss / Take-Profit** — Multiple calculation methods (fixed, ATR, trailing)
//! - **Exposure Limits** — Per-asset, per-direction, and portfolio-level exposure caps

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during risk management operations.
#[derive(Debug, Error)]
pub enum RiskError {
    #[error("insufficient data: need at least {required} data points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("invalid parameter: {reason}")]
    InvalidParam { reason: String },
    #[error("portfolio weights must sum to 1.0, got {sum}")]
    WeightsSumError { sum: f64 },
    #[error("drawdown limit breached: {current_drawdown:.4} > {limit:.4}")]
    DrawdownBreached { current_drawdown: f64, limit: f64 },
    #[error("exposure limit breached: {current:.4} > {limit:.4}")]
    ExposureBreached { current: f64, limit: f64 },
    #[error("negative value where positive required: {field}")]
    NegativeValue { field: String },
}

// ---------------------------------------------------------------------------
// Position Sizing
// ---------------------------------------------------------------------------

/// Result of a position sizing calculation.
#[derive(Debug, Clone)]
pub struct PositionSize {
    /// Position size in base currency (e.g., USD).
    pub size: f64,
    /// Number of units/shares to purchase.
    pub units: f64,
    /// Size as a fraction of total capital.
    pub fraction: f64,
    /// Risk per trade in base currency.
    pub risk_amount: f64,
    /// Sizing method used.
    pub method: String,
}

/// Kelly Criterion position sizing.
///
/// The Kelly Criterion maximizes the long-run growth rate of capital:
/// f* = (p·b - q) / b
/// where p = win probability, b = win/loss ratio, q = 1 - p.
///
/// In practice, a fractional Kelly (e.g., half-Kelly) is recommended to reduce
/// variance while preserving most of the growth advantage.
pub struct KellyCriterion;

impl KellyCriterion {
    /// Compute the optimal Kelly fraction.
    ///
    /// # Arguments
    /// * `win_rate` — Probability of a winning trade (p), in [0, 1].
    /// * `avg_win` — Average dollar amount won per winning trade.
    /// * `avg_loss` — Average dollar amount lost per losing trade (positive value).
    /// * `fraction` — Fraction of Kelly to use (1.0 = full Kelly, 0.5 = half Kelly).
    ///
    /// # Returns
    /// The recommended fraction of capital to risk per trade.
    pub fn compute_fraction(
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        fraction: f64,
    ) -> Result<f64, RiskError> {
        if !(0.0..=1.0).contains(&win_rate) {
            return Err(RiskError::InvalidParam {
                reason: format!("win_rate must be in [0, 1], got {}", win_rate),
            });
        }
        if avg_win < 0.0 {
            return Err(RiskError::NegativeValue {
                field: "avg_win".into(),
            });
        }
        if avg_loss <= 0.0 {
            return Err(RiskError::InvalidParam {
                reason: "avg_loss must be > 0".into(),
            });
        }
        if fraction <= 0.0 || fraction > 1.0 {
            return Err(RiskError::InvalidParam {
                reason: format!("fraction must be in (0, 1], got {}", fraction),
            });
        }

        // Kelly fraction: f* = p - (1-p) / b  where b = avg_win / avg_loss
        let b = avg_win / avg_loss;
        let kelly = win_rate - (1.0 - win_rate) / b;

        // Apply fractional Kelly and clamp to [0, 1]
        Ok((kelly * fraction).clamp(0.0, 1.0))
    }

    /// Compute position size based on Kelly criterion.
    ///
    /// # Arguments
    /// * `capital` — Total available capital.
    /// * `win_rate` — Win probability.
    /// * `avg_win` — Average win amount.
    /// * `avg_loss` — Average loss amount.
    /// * `fraction` — Fractional Kelly multiplier.
    /// * `price` — Current price of the asset.
    pub fn compute_position(
        capital: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        fraction: f64,
        price: f64,
    ) -> Result<PositionSize, RiskError> {
        let kelly_frac = Self::compute_fraction(win_rate, avg_win, avg_loss, fraction)?;
        let size = capital * kelly_frac;
        let units = if price > 1e-15 { size / price } else { 0.0 };
        let risk_amount = size * (1.0 - win_rate);

        Ok(PositionSize {
            size,
            units,
            fraction: kelly_frac,
            risk_amount,
            method: "kelly_criterion".into(),
        })
    }
}

/// Fixed fractional position sizing.
///
/// Risks a fixed percentage of capital per trade.
pub struct FixedFractional;

impl FixedFractional {
    /// Compute position size for fixed fractional sizing.
    ///
    /// # Arguments
    /// * `capital` — Total available capital.
    /// * `risk_pct` — Percentage of capital to risk per trade (e.g., 0.02 = 2%).
    /// * `entry_price` — Entry price.
    /// * `stop_loss` — Stop-loss price.
    pub fn compute(
        capital: f64,
        risk_pct: f64,
        entry_price: f64,
        stop_loss: f64,
    ) -> Result<PositionSize, RiskError> {
        if risk_pct <= 0.0 || risk_pct > 1.0 {
            return Err(RiskError::InvalidParam {
                reason: format!("risk_pct must be in (0, 1], got {}", risk_pct),
            });
        }
        if entry_price <= 0.0 || stop_loss <= 0.0 {
            return Err(RiskError::InvalidParam {
                reason: "entry_price and stop_loss must be > 0".into(),
            });
        }

        let risk_amount = capital * risk_pct;
        let risk_per_unit = (entry_price - stop_loss).abs();

        let units = if risk_per_unit > 1e-15 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        let size = units * entry_price;
        let fraction = size / capital;

        Ok(PositionSize {
            size,
            units,
            fraction,
            risk_amount,
            method: "fixed_fractional".into(),
        })
    }
}

/// Volatility-adjusted position sizing.
///
/// Scales position size inversely with the asset's volatility so that
/// each position contributes roughly equal risk to the portfolio.
pub struct VolatilityAdjusted;

impl VolatilityAdjusted {
    /// Compute volatility-adjusted position size.
    ///
    /// # Arguments
    /// * `capital` — Total available capital.
    /// * `target_vol_pct` — Target portfolio volatility as a fraction (e.g., 0.01 = 1%).
    /// * `asset_vol` — Asset's annualized volatility (e.g., 0.25 = 25%).
    /// * `periods_per_year` — Number of trading periods per year.
    /// * `price` — Current asset price.
    pub fn compute(
        capital: f64,
        target_vol_pct: f64,
        asset_vol: f64,
        periods_per_year: f64,
        price: f64,
    ) -> Result<PositionSize, RiskError> {
        if target_vol_pct <= 0.0 {
            return Err(RiskError::InvalidParam {
                reason: "target_vol_pct must be > 0".into(),
            });
        }
        if asset_vol <= 0.0 {
            return Err(RiskError::InvalidParam {
                reason: "asset_vol must be > 0".into(),
            });
        }

        // Scale annual volatility to per-period
        let per_period_vol = asset_vol / periods_per_year.sqrt();

        // Position size = target_vol / asset_vol * capital
        let fraction = (target_vol_pct / per_period_vol).min(1.0);
        let size = capital * fraction;
        let units = if price > 1e-15 { size / price } else { 0.0 };
        let risk_amount = size * per_period_vol;

        Ok(PositionSize {
            size,
            units,
            fraction,
            risk_amount,
            method: "volatility_adjusted".into(),
        })
    }
}

/// Unified position sizer that dispatches to the appropriate method.
pub struct PositionSizer;

impl PositionSizer {
    /// Compute position size using the specified method.
    pub fn compute(params: &PositionSizingParams) -> Result<PositionSize, RiskError> {
        match params.method.as_str() {
            "kelly" => KellyCriterion::compute_position(
                params.capital,
                params.win_rate,
                params.avg_win,
                params.avg_loss,
                params.kelly_fraction.unwrap_or(0.5),
                params.price,
            ),
            "fixed_fractional" => {
                let stop = params.stop_loss.ok_or_else(|| RiskError::InvalidParam {
                    reason: "stop_loss required for fixed_fractional sizing".into(),
                })?;
                FixedFractional::compute(params.capital, params.risk_pct, params.price, stop)
            }
            "volatility_adjusted" => VolatilityAdjusted::compute(
                params.capital,
                params.target_vol_pct.unwrap_or(0.01),
                params.asset_vol.ok_or_else(|| RiskError::InvalidParam {
                    reason: "asset_vol required for volatility_adjusted sizing".into(),
                })?,
                params.periods_per_year.unwrap_or(252.0),
                params.price,
            ),
            other => Err(RiskError::InvalidParam {
                reason: format!("unknown sizing method: {}", other),
            }),
        }
    }
}

/// Parameters for position sizing.
#[derive(Debug, Clone)]
pub struct PositionSizingParams {
    /// Sizing method: "kelly", "fixed_fractional", "volatility_adjusted".
    pub method: String,
    /// Total capital available.
    pub capital: f64,
    /// Current asset price.
    pub price: f64,
    /// Win rate (for Kelly).
    pub win_rate: f64,
    /// Average win amount (for Kelly).
    pub avg_win: f64,
    /// Average loss amount (for Kelly).
    pub avg_loss: f64,
    /// Kelly fraction multiplier (for Kelly).
    pub kelly_fraction: Option<f64>,
    /// Risk percentage per trade (for fixed fractional).
    pub risk_pct: f64,
    /// Stop-loss price (for fixed fractional).
    pub stop_loss: Option<f64>,
    /// Target volatility per period (for volatility-adjusted).
    pub target_vol_pct: Option<f64>,
    /// Asset's annualized volatility (for volatility-adjusted).
    pub asset_vol: Option<f64>,
    /// Trading periods per year (default 252).
    pub periods_per_year: Option<f64>,
}

impl PositionSizingParams {
    /// Create Kelly criterion params.
    pub fn kelly(capital: f64, price: f64, win_rate: f64, avg_win: f64, avg_loss: f64, fraction: f64) -> Self {
        PositionSizingParams {
            method: "kelly".into(),
            capital,
            price,
            win_rate,
            avg_win,
            avg_loss,
            kelly_fraction: Some(fraction),
            risk_pct: 0.02,
            stop_loss: None,
            target_vol_pct: None,
            asset_vol: None,
            periods_per_year: None,
        }
    }

    /// Create fixed fractional params.
    pub fn fixed_fractional(capital: f64, price: f64, risk_pct: f64, stop_loss: f64) -> Self {
        PositionSizingParams {
            method: "fixed_fractional".into(),
            capital,
            price,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            kelly_fraction: None,
            risk_pct,
            stop_loss: Some(stop_loss),
            target_vol_pct: None,
            asset_vol: None,
            periods_per_year: None,
        }
    }

    /// Create volatility-adjusted params.
    pub fn volatility_adjusted(
        capital: f64,
        price: f64,
        target_vol: f64,
        asset_vol: f64,
        periods_per_year: f64,
    ) -> Self {
        PositionSizingParams {
            method: "volatility_adjusted".into(),
            capital,
            price,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            kelly_fraction: None,
            risk_pct: 0.02,
            stop_loss: None,
            target_vol_pct: Some(target_vol),
            asset_vol: Some(asset_vol),
            periods_per_year: Some(periods_per_year),
        }
    }
}

// ---------------------------------------------------------------------------
// Risk Metrics
// ---------------------------------------------------------------------------

/// Comprehensive risk metrics for portfolio evaluation.
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Tracking error relative to benchmark.
    pub tracking_error: f64,
    /// Information ratio (excess return / tracking error).
    pub information_ratio: f64,
    /// Portfolio beta relative to benchmark.
    pub beta: f64,
    /// Jensen's alpha (risk-adjusted excess return).
    pub alpha: f64,
    /// Total portfolio volatility (annualized).
    pub portfolio_volatility: f64,
    /// Correlation with benchmark.
    pub benchmark_correlation: f64,
    /// Specific risk (idiosyncratic volatility).
    pub specific_risk: f64,
}

impl RiskMetrics {
    /// Compute all risk metrics relative to a benchmark.
    ///
    /// # Arguments
    /// * `portfolio_returns` — Periodic portfolio returns.
    /// * `benchmark_returns` — Periodic benchmark returns (same length).
    /// * `risk_free_rate` — Annualized risk-free rate.
    /// * `periods_per_year` — Number of periods per year.
    pub fn compute(
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
        risk_free_rate: f64,
        periods_per_year: f64,
    ) -> Result<RiskMetrics, RiskError> {
        if portfolio_returns.len() != benchmark_returns.len() {
            return Err(RiskError::InvalidParam {
                reason: format!(
                    "portfolio returns ({}) and benchmark returns ({}) must have same length",
                    portfolio_returns.len(),
                    benchmark_returns.len()
                ),
            });
        }
        if portfolio_returns.len() < 2 {
            return Err(RiskError::InsufficientData {
                required: 2,
                actual: portfolio_returns.len(),
            });
        }

        let n = portfolio_returns.len();

        // Mean returns (annualized)
        let p_mean = portfolio_returns.iter().sum::<f64>() / n as f64 * periods_per_year;
        let b_mean = benchmark_returns.iter().sum::<f64>() / n as f64 * periods_per_year;

        // Volatilities (annualized)
        let p_var: f64 = {
            let m = portfolio_returns.iter().sum::<f64>() / n as f64;
            portfolio_returns.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (n - 1) as f64
        };
        let b_var: f64 = {
            let m = benchmark_returns.iter().sum::<f64>() / n as f64;
            benchmark_returns.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (n - 1) as f64
        };
        let p_vol = p_var.sqrt() * periods_per_year.sqrt();
        let b_vol = b_var.sqrt() * periods_per_year.sqrt();

        // Covariance
        let p_mean_daily = portfolio_returns.iter().sum::<f64>() / n as f64;
        let b_mean_daily = benchmark_returns.iter().sum::<f64>() / n as f64;
        let cov: f64 = (0..n)
            .map(|i| (portfolio_returns[i] - p_mean_daily) * (benchmark_returns[i] - b_mean_daily))
            .sum::<f64>()
            / (n - 1) as f64;

        // Beta
        let beta = if b_var.abs() < 1e-15 {
            1.0
        } else {
            cov / b_var
        };

        // Correlation
        let denom = (p_var * b_var).sqrt();
        let benchmark_correlation = if denom.abs() < 1e-15 {
            0.0
        } else {
            cov / denom
        };

        // Tracking error (annualized)
        let excess_returns: Vec<f64> = (0..n)
            .map(|i| portfolio_returns[i] - benchmark_returns[i])
            .collect();
        let te_var: f64 = {
            let m = excess_returns.iter().sum::<f64>() / n as f64;
            excess_returns.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (n - 1) as f64
        };
        let tracking_error = te_var.sqrt() * periods_per_year.sqrt();

        // Information ratio
        let excess_annual = p_mean - b_mean;
        let information_ratio = if tracking_error.abs() < 1e-15 {
            0.0
        } else {
            excess_annual / tracking_error
        };

        // Jensen's alpha
        let alpha = p_mean - (risk_free_rate + beta * (b_mean - risk_free_rate));

        // Specific risk (idiosyncratic volatility)
        let systematic_var = beta * beta * b_var;
        let specific_var = (p_var - systematic_var).max(0.0);
        let specific_risk = specific_var.sqrt() * periods_per_year.sqrt();

        Ok(RiskMetrics {
            tracking_error,
            information_ratio,
            beta,
            alpha,
            portfolio_volatility: p_vol,
            benchmark_correlation,
            specific_risk,
        })
    }
}

// ---------------------------------------------------------------------------
// Drawdown Tracking
// ---------------------------------------------------------------------------

/// Real-time drawdown tracker.
#[derive(Debug, Clone)]
pub struct DrawdownTracker {
    /// Current cumulative high-water mark.
    pub high_water_mark: f64,
    /// Current drawdown as a positive fraction (e.g., 0.15 = 15% drawdown).
    pub current_drawdown: f64,
    /// Maximum observed drawdown.
    pub max_drawdown: f64,
    /// Drawdown limit (e.g., 0.2 = 20% max allowed drawdown).
    pub drawdown_limit: f64,
    /// Whether the drawdown limit has been breached.
    pub limit_breached: bool,
    /// Number of consecutive periods in drawdown.
    pub drawdown_periods: usize,
    /// Total number of periods tracked.
    pub total_periods: usize,
}

impl DrawdownTracker {
    /// Create a new tracker with an initial value and drawdown limit.
    pub fn new(initial_value: f64, drawdown_limit: f64) -> Self {
        DrawdownTracker {
            high_water_mark: initial_value,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            drawdown_limit,
            limit_breached: false,
            drawdown_periods: 0,
            total_periods: 0,
        }
    }

    /// Update the tracker with a new portfolio value.
    ///
    /// Returns an error if the drawdown limit is breached.
    pub fn update(&mut self, value: f64) -> Result<(), RiskError> {
        self.total_periods += 1;

        if value > self.high_water_mark {
            self.high_water_mark = value;
            self.current_drawdown = 0.0;
            self.drawdown_periods = 0;
            self.limit_breached = false;
        } else {
            let dd = (self.high_water_mark - value) / self.high_water_mark;
            self.current_drawdown = dd;
            self.drawdown_periods += 1;

            if dd > self.max_drawdown {
                self.max_drawdown = dd;
            }

            if dd > self.drawdown_limit && !self.limit_breached {
                self.limit_breached = true;
                return Err(RiskError::DrawdownBreached {
                    current_drawdown: dd,
                    limit: self.drawdown_limit,
                });
            }
        }

        Ok(())
    }

    /// Reset the tracker to a new value.
    pub fn reset(&mut self, value: f64) {
        self.high_water_mark = value;
        self.current_drawdown = 0.0;
        self.max_drawdown = 0.0;
        self.limit_breached = false;
        self.drawdown_periods = 0;
    }

    /// The recovery percentage needed to get back to the high-water mark.
    pub fn recovery_needed(&self) -> f64 {
        if self.high_water_mark.abs() < 1e-15 || self.current_drawdown < 1e-15 {
            0.0
        } else {
            let current = self.high_water_mark * (1.0 - self.current_drawdown);
            (self.high_water_mark - current) / current
        }
    }

    /// Average drawdown duration (drawdown_periods / total_periods).
    pub fn avg_drawdown_ratio(&self) -> f64 {
        if self.total_periods == 0 {
            0.0
        } else {
            self.drawdown_periods as f64 / self.total_periods as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Portfolio Construction
// ---------------------------------------------------------------------------

/// Helper methods for portfolio construction.
pub struct PortfolioConstructor;

impl PortfolioConstructor {
    /// Equal-weight portfolio allocation.
    ///
    /// Returns weights as a vector of length `n_assets`, each being 1/n.
    pub fn equal_weight(n_assets: usize) -> Result<Vec<f64>, RiskError> {
        if n_assets == 0 {
            return Err(RiskError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        Ok(vec![1.0 / n_assets as f64; n_assets])
    }

    /// Inverse-volatility portfolio allocation.
    ///
    /// Weights are proportional to 1/σ_i, then normalized to sum to 1.
    /// This ensures each asset contributes equal risk to the portfolio.
    pub fn inverse_volatility(volatilities: &[f64]) -> Result<Vec<f64>, RiskError> {
        if volatilities.is_empty() {
            return Err(RiskError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        let inv_vols: Vec<f64> = volatilities
            .iter()
            .map(|&v| {
                if v.abs() < 1e-15 {
                    0.0
                } else {
                    1.0 / v
                }
            })
            .collect();

        let sum: f64 = inv_vols.iter().sum();
        if sum.abs() < 1e-15 {
            return Err(RiskError::InvalidParam {
                reason: "all volatilities are zero".into(),
            });
        }

        Ok(inv_vols.iter().map(|&v| v / sum).collect())
    }

    /// Inverse-variance portfolio allocation.
    ///
    /// Weights are proportional to 1/σ²_i, then normalized to sum to 1.
    pub fn inverse_variance(volatilities: &[f64]) -> Result<Vec<f64>, RiskError> {
        if volatilities.is_empty() {
            return Err(RiskError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        let inv_vars: Vec<f64> = volatilities
            .iter()
            .map(|&v| {
                let v2 = v * v;
                if v2.abs() < 1e-15 { 0.0 } else { 1.0 / v2 }
            })
            .collect();

        let sum: f64 = inv_vars.iter().sum();
        if sum.abs() < 1e-15 {
            return Err(RiskError::InvalidParam {
                reason: "all volatilities are zero".into(),
            });
        }

        Ok(inv_vars.iter().map(|&v| v / sum).collect())
    }

    /// Risk parity allocation (simplified).
    ///
    /// Attempts to equalize the risk contribution of each asset.
    /// Uses the inverse-volatility approach as a first approximation,
    /// then iteratively adjusts weights.
    pub fn risk_parity(
        volatilities: &[f64],
        correlation_matrix: Option<&Array2<f64>>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>, RiskError> {
        // Start with inverse-volatility weights
        let mut weights = Self::inverse_volatility(volatilities)?;

        if let Some(corr) = correlation_matrix {
            let n = weights.len();

            // Iteratively adjust weights toward equal risk contribution
            for _ in 0..max_iterations {
                // Compute marginal risk contributions
                let portfolio_vol = Self::portfolio_vol_from_weights(&weights, volatilities, corr)?;

                let mut risk_contributions = Vec::with_capacity(n);
                for i in 0..n {
                    let mut marginal = 0.0;
                    for j in 0..n {
                        marginal += weights[j] * volatilities[i] * volatilities[j] * corr[[i, j]];
                    }
                    let rc = weights[i] * marginal / portfolio_vol;
                    risk_contributions.push(rc);
                }

                // Target equal risk contribution
                let total_rc: f64 = risk_contributions.iter().sum();
                let target_rc = total_rc / n as f64;

                // Adjust weights proportionally
                let mut converged = true;
                for i in 0..n {
                    if risk_contributions[i].abs() > 1e-15 {
                        let adjustment = target_rc / risk_contributions[i];
                        let new_weight = weights[i] * adjustment;

                        // Smooth the adjustment (half-step)
                        weights[i] = weights[i] * 0.5 + new_weight * 0.5;

                        if (weights[i] - (target_rc / risk_contributions[i]) * weights[i]).abs() > tolerance {
                            converged = false;
                        }
                    }
                }

                // Normalize weights
                let sum: f64 = weights.iter().sum();
                if sum.abs() > 1e-15 {
                    for w in weights.iter_mut() {
                        *w /= sum;
                    }
                }

                if converged {
                    break;
                }
            }
        }

        Ok(weights)
    }

    /// Compute portfolio volatility given weights, asset volatilities, and correlation matrix.
    fn portfolio_vol_from_weights(
        weights: &[f64],
        vols: &[f64],
        corr: &Array2<f64>,
    ) -> Result<f64, RiskError> {
        let n = weights.len();
        if n != vols.len() {
            return Err(RiskError::InvalidParam {
                reason: "weights and volatilities must have same length".into(),
            });
        }

        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                var += weights[i] * weights[j] * vols[i] * vols[j] * corr[[i, j]];
            }
        }

        Ok(var.sqrt())
    }

    /// Cap portfolio weights to a maximum value, redistributing excess proportionally.
    pub fn cap_weights(weights: &mut [f64], max_weight: f64) -> Result<(), RiskError> {
        if !(0.0..1.0).contains(&max_weight) {
            return Err(RiskError::InvalidParam {
                reason: format!("max_weight must be in (0, 1), got {}", max_weight),
            });
        }

        loop {
            let mut excess = 0.0;
            let mut capped_count = 0;

            for w in weights.iter_mut() {
                if *w > max_weight {
                    excess += *w - max_weight;
                    *w = max_weight;
                    capped_count += 1;
                }
            }

            if capped_count == 0 || excess < 1e-15 {
                break;
            }

            // Redistribute excess to uncapped weights
            let uncapped_sum: f64 = weights.iter().filter(|&&w| w < max_weight - 1e-15).sum();
            if uncapped_sum < 1e-15 {
                break;
            }

            for w in weights.iter_mut() {
                if *w < max_weight - 1e-15 {
                    *w += excess * (*w / uncapped_sum);
                }
            }
        }

        // Final normalization
        let sum: f64 = weights.iter().sum();
        if sum.abs() > 1e-15 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Stop-Loss / Take-Profit
// ---------------------------------------------------------------------------

/// Stop-loss and take-profit calculator.
pub struct StopLossCalculator;

impl StopLossCalculator {
    /// Fixed percentage stop-loss.
    ///
    /// # Arguments
    /// * `entry_price` — Entry price.
    /// * `stop_pct` — Stop-loss as a percentage (e.g., 0.05 = 5%).
    /// * `is_long` — Whether the position is long (true) or short (false).
    pub fn fixed_stop(entry_price: f64, stop_pct: f64, is_long: bool) -> (f64, f64) {
        let stop = if is_long {
            entry_price * (1.0 - stop_pct)
        } else {
            entry_price * (1.0 + stop_pct)
        };
        (stop, stop)
    }

    /// Fixed percentage take-profit.
    pub fn fixed_take_profit(entry_price: f64, tp_pct: f64, is_long: bool) -> f64 {
        if is_long {
            entry_price * (1.0 + tp_pct)
        } else {
            entry_price * (1.0 - tp_pct)
        }
    }

    /// ATR-based stop-loss.
    ///
    /// Places the stop at a multiple of ATR from the entry price.
    pub fn atr_stop(entry_price: f64, atr: f64, atr_multiplier: f64, is_long: bool) -> f64 {
        if is_long {
            entry_price - atr * atr_multiplier
        } else {
            entry_price + atr * atr_multiplier
        }
    }

    /// ATR-based take-profit.
    pub fn atr_take_profit(entry_price: f64, atr: f64, atr_multiplier: f64, is_long: bool) -> f64 {
        if is_long {
            entry_price + atr * atr_multiplier
        } else {
            entry_price - atr * atr_multiplier
        }
    }

    /// Trailing stop calculation.
    ///
    /// The stop level adjusts upward (for long) or downward (for short) as the
    /// price moves favorably, but never moves backward.
    ///
    /// # Arguments
    /// * `current_price` — Current market price.
    /// * `current_stop` — Current stop level (use entry-based stop for the first call).
    /// * `trail_pct` — Trailing distance as a percentage.
    /// * `is_long` — Whether the position is long.
    ///
    /// # Returns
    /// The new (potentially updated) stop level.
    pub fn trailing_stop(
        current_price: f64,
        current_stop: f64,
        trail_pct: f64,
        is_long: bool,
    ) -> f64 {
        if is_long {
            let new_stop = current_price * (1.0 - trail_pct);
            new_stop.max(current_stop) // only move up
        } else {
            let new_stop = current_price * (1.0 + trail_pct);
            new_stop.min(current_stop) // only move down
        }
    }

    /// Compute a risk/reward ratio.
    pub fn risk_reward_ratio(entry_price: f64, stop_loss: f64, take_profit: f64, is_long: bool) -> f64 {
        let risk = (entry_price - stop_loss).abs();
        let reward = (take_profit - entry_price).abs();
        if risk < 1e-15 {
            f64::INFINITY
        } else {
            reward / risk
        }
    }
}

// ---------------------------------------------------------------------------
// Exposure Limits
// ---------------------------------------------------------------------------

/// Manages exposure limits for a portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureLimit {
    /// Maximum total portfolio exposure (1.0 = fully invested).
    pub max_total_exposure: f64,
    /// Maximum exposure to a single asset (as a fraction of total capital).
    pub max_single_asset: f64,
    /// Maximum long exposure.
    pub max_long_exposure: f64,
    /// Maximum short exposure.
    pub max_short_exposure: f64,
    /// Maximum net exposure (long - short).
    pub max_net_exposure: f64,
    /// Maximum gross exposure (long + short).
    pub max_gross_exposure: f64,
    /// Per-asset exposure limits (asset_name → max_fraction).
    pub per_asset_limits: std::collections::HashMap<String, f64>,
}

impl ExposureLimit {
    /// Create a default exposure limit configuration.
    pub fn new() -> Self {
        ExposureLimit {
            max_total_exposure: 1.0,
            max_single_asset: 0.25,
            max_long_exposure: 1.0,
            max_short_exposure: 0.5,
            max_net_exposure: 1.0,
            max_gross_exposure: 1.5,
            per_asset_limits: std::collections::HashMap::new(),
        }
    }

    /// Validate that a set of positions complies with all exposure limits.
    pub fn validate(
        &self,
        positions: &[(String, f64)], // (asset, dollar_exposure; positive=long, negative=short)
        total_capital: f64,
    ) -> Result<ExposureReport, RiskError> {
        let mut total_long = 0.0_f64;
        let mut total_short = 0.0_f64;
        let mut gross = 0.0_f64;
        let mut violations: Vec<String> = Vec::new();

        for (asset, exposure) in positions {
            let fraction = exposure.abs() / total_capital;

            if *exposure > 0.0 {
                total_long += exposure;
            } else {
                total_short += exposure.abs();
            }
            gross += exposure.abs();

            // Check per-asset limit
            let limit = self
                .per_asset_limits
                .get(asset)
                .copied()
                .unwrap_or(self.max_single_asset);

            if fraction > limit {
                violations.push(format!(
                    "Asset {} exposure {:.4} exceeds limit {:.4}",
                    asset, fraction, limit
                ));
            }
        }

        let net = total_long - total_short;
        let gross_exposure = gross / total_capital;
        let net_exposure = net / total_capital;
        let long_exposure = total_long / total_capital;
        let short_exposure = total_short / total_capital;

        if long_exposure > self.max_long_exposure {
            violations.push(format!(
                "Long exposure {:.4} exceeds limit {:.4}",
                long_exposure, self.max_long_exposure
            ));
        }
        if short_exposure > self.max_short_exposure {
            violations.push(format!(
                "Short exposure {:.4} exceeds limit {:.4}",
                short_exposure, self.max_short_exposure
            ));
        }
        if net_exposure.abs() > self.max_net_exposure {
            violations.push(format!(
                "Net exposure {:.4} exceeds limit {:.4}",
                net_exposure, self.max_net_exposure
            ));
        }
        if gross_exposure > self.max_gross_exposure {
            violations.push(format!(
                "Gross exposure {:.4} exceeds limit {:.4}",
                gross_exposure, self.max_gross_exposure
            ));
        }

        Ok(ExposureReport {
            total_long,
            total_short,
            net,
            gross,
            long_exposure,
            short_exposure,
            net_exposure,
            gross_exposure,
            violations: violations.clone(),
            is_valid: violations.is_empty(),
        })
    }
}

impl Default for ExposureLimit {
    fn default() -> Self {
        Self::new()
    }
}

/// Report from exposure validation.
#[derive(Debug, Clone)]
pub struct ExposureReport {
    pub total_long: f64,
    pub total_short: f64,
    pub net: f64,
    pub gross: f64,
    pub long_exposure: f64,
    pub short_exposure: f64,
    pub net_exposure: f64,
    pub gross_exposure: f64,
    pub violations: Vec<String>,
    pub is_valid: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper macro for approximate float equality.
    // Usage: assert_relative_eq!(a, b) or assert_relative_eq!(a, b, epsilon = 1e-10)
    macro_rules! assert_relative_eq {
        ($a:expr, $b:expr) => {
            assert_relative_eq!($a, $b, epsilon = 1e-6)
        };
        ($a:expr, $b:expr, epsilon = $eps:expr) => {{
            let a_val = $a;
            let b_val = $b;
            let eps_val = $eps;
            assert!(
                (a_val - b_val).abs() < eps_val,
                "assertion failed: |{} - {}| = {} >= {}",
                a_val, b_val, (a_val - b_val).abs(), eps_val
            );
        }};
    }

    #[test]
    fn test_kelly_fraction() {
        // p=0.6, b=2 (avg_win=200, avg_loss=100): f* = 0.6 - 0.4/2 = 0.4
        let frac = KellyCriterion::compute_fraction(0.6, 200.0, 100.0, 0.5).unwrap();
        // Half Kelly of 0.4 = 0.2
        assert_relative_eq!(frac, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_kelly_position() {
        let pos = KellyCriterion::compute_position(
            100_000.0, 0.6, 200.0, 100.0, 0.5, 100.0,
        )
        .unwrap();
        assert_relative_eq!(pos.size, 20_000.0, epsilon = 1.0);
        assert_eq!(pos.method, "kelly_criterion");
    }

    #[test]
    fn test_kelly_invalid_win_rate() {
        let result = KellyCriterion::compute_fraction(1.5, 200.0, 100.0, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_fractional() {
        // Capital 100k, risk 2%, entry 100, stop 95 → risk per unit = 5
        // Units = 2000 / 5 = 400, size = 400 * 100 = 40,000
        let pos = FixedFractional::compute(100_000.0, 0.02, 100.0, 95.0).unwrap();
        assert_relative_eq!(pos.units, 400.0, epsilon = 1e-10);
        assert_relative_eq!(pos.size, 40_000.0, epsilon = 1e-10);
        assert_relative_eq!(pos.risk_amount, 2_000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_volatility_adjusted() {
        let pos = VolatilityAdjusted::compute(
            100_000.0, // capital
            0.01,      // target 1% daily vol
            0.30,      // asset vol 30% annual
            252.0,     // daily
            100.0,     // price
        )
        .unwrap();
        assert!(pos.size > 0.0);
        assert!(pos.units > 0.0);
    }

    #[test]
    fn test_risk_metrics() {
        let portfolio = vec![0.01, 0.02, -0.01, 0.03, 0.00, -0.02, 0.01, 0.02, -0.01, 0.01];
        let benchmark = vec![0.005, 0.01, -0.005, 0.02, 0.005, -0.01, 0.005, 0.015, -0.005, 0.005];

        let metrics = RiskMetrics::compute(&portfolio, &benchmark, 0.02, 252.0).unwrap();
        assert!(metrics.portfolio_volatility > 0.0);
        assert!(metrics.benchmark_correlation.abs() <= 1.0);
    }

    #[test]
    fn test_drawdown_tracker() {
        let mut tracker = DrawdownTracker::new(100_000.0, 0.20);

        tracker.update(105_000.0).unwrap(); // new high
        assert_relative_eq!(tracker.high_water_mark, 105_000.0);
        assert_relative_eq!(tracker.current_drawdown, 0.0);

        tracker.update(100_000.0).unwrap(); // 4.76% drawdown
        assert!(tracker.current_drawdown > 0.04);
        assert!(tracker.current_drawdown < 0.05);

        // Breach limit
        tracker.update(70_000.0).unwrap_err(); // ~33% drawdown > 20% limit
        assert!(tracker.limit_breached);
    }

    #[test]
    fn test_drawdown_recovery_needed() {
        let mut tracker = DrawdownTracker::new(100.0, 0.5);
        tracker.update(80.0).unwrap(); // 20% drawdown
        let recovery = tracker.recovery_needed();
        // Need 25% gain to recover: (100-80)/80 = 0.25
        assert_relative_eq!(recovery, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_equal_weight() {
        let weights = PortfolioConstructor::equal_weight(4).unwrap();
        assert_eq!(weights.len(), 4);
        for w in &weights {
            assert_relative_eq!(*w, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_volatility() {
        let vols = vec![0.1, 0.2, 0.3];
        let weights = PortfolioConstructor::inverse_volatility(&vols).unwrap();
        assert_relative_eq!(weights.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        // Lower vol should get higher weight
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }

    #[test]
    fn test_inverse_variance() {
        let vols = vec![0.1, 0.2, 0.3];
        let weights = PortfolioConstructor::inverse_variance(&vols).unwrap();
        assert_relative_eq!(weights.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cap_weights() {
        let mut weights = vec![0.5, 0.3, 0.2];
        PortfolioConstructor::cap_weights(&mut weights, 0.4).unwrap();
        for w in &weights {
            assert!(*w <= 0.4 + 1e-10);
        }
        assert_relative_eq!(weights.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_stop_long() {
        let (sl, _) = StopLossCalculator::fixed_stop(100.0, 0.05, true);
        assert_relative_eq!(sl, 95.0);
    }

    #[test]
    fn test_fixed_stop_short() {
        let (sl, _) = StopLossCalculator::fixed_stop(100.0, 0.05, false);
        assert_relative_eq!(sl, 105.0);
    }

    #[test]
    fn test_fixed_take_profit() {
        let tp = StopLossCalculator::fixed_take_profit(100.0, 0.10, true);
        assert_relative_eq!(tp, 110.0);
    }

    #[test]
    fn test_atr_stop() {
        let sl = StopLossCalculator::atr_stop(100.0, 2.0, 2.0, true);
        assert_relative_eq!(sl, 96.0);
    }

    #[test]
    fn test_trailing_stop_long() {
        // Start with stop at 95, price moves to 110, trail 5%
        let new_stop = StopLossCalculator::trailing_stop(110.0, 95.0, 0.05, true);
        // New stop = 110 * 0.95 = 104.5, which is > 95, so update
        assert_relative_eq!(new_stop, 104.5);
    }

    #[test]
    fn test_trailing_stop_long_no_decrease() {
        // Price drops, stop should NOT decrease
        let new_stop = StopLossCalculator::trailing_stop(95.0, 104.5, 0.05, true);
        assert_relative_eq!(new_stop, 104.5); // stays at previous level
    }

    #[test]
    fn test_risk_reward_ratio() {
        let rr = StopLossCalculator::risk_reward_ratio(100.0, 95.0, 110.0, true);
        assert_relative_eq!(rr, 2.0); // reward=10, risk=5
    }

    #[test]
    fn test_exposure_validation() {
        let limits = ExposureLimit::new();
        let positions = vec![
            ("BTC".to_string(), 25_000.0),
            ("ETH".to_string(), 15_000.0),
            ("SOL".to_string(), 10_000.0),
        ];
        let report = limits.validate(&positions, 100_000.0).unwrap();
        assert!(report.is_valid);
        assert_relative_eq!(report.long_exposure, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_exposure_violation() {
        let limits = ExposureLimit::new();
        let positions = vec![
            ("BTC".to_string(), 50_000.0),
            ("ETH".to_string(), 50_000.0),
            ("SOL".to_string(), 50_000.0),
        ];
        let report = limits.validate(&positions, 100_000.0).unwrap();
        assert!(!report.is_valid);
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn test_exposure_per_asset_limit() {
        let mut limits = ExposureLimit::new();
        limits.per_asset_limits.insert("BTC".to_string(), 0.1);

        let positions = vec![("BTC".to_string(), 25_000.0)];
        let report = limits.validate(&positions, 100_000.0).unwrap();
        assert!(!report.is_valid);
    }

}
