//! # Financial Math Primitives
//!
//! Core mathematical operations for financial analysis and ML preprocessing.
//! All array operations use `ndarray` for performance and composability.
//!
//! ## Key components
//!
//! - **Moving averages**: SMA, EMA, WMA with configurable windows
//! - **Volatility estimators**: Realized, Parkinson, Garman-Klass
//! - **Portfolio statistics**: Sharpe ratio, Sortino ratio, max drawdown, VaR, CVaR
//! - **Correlation/covariance**: Pearson method
//! - **Quantile functions**: Linear interpolation
//! - **Statistical tests**: Z-score test, t-tests

use ndarray::{s, Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use thiserror::Error;

/// Errors that can occur during financial math operations.
#[derive(Debug, Error)]
pub enum MathError {
    #[error("insufficient data: need at least {required} elements, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("zero variance encountered")]
    ZeroVariance,
    #[error("invalid parameter: {reason}")]
    InvalidParam { reason: String },
    #[error("matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Type of moving average to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovingAverageType {
    SMA,
    EMA,
    WMA,
}

/// Computes a moving average over a 1-D array.
pub struct MovingAverage;

impl MovingAverage {
    /// Compute a moving average of the given type over `data` with window size `window`.
    pub fn compute(
        data: &Array1<f64>,
        window: usize,
        ma_type: MovingAverageType,
    ) -> Result<Array1<f64>, MathError> {
        if window == 0 {
            return Err(MathError::InvalidParam { reason: "window must be > 0".into() });
        }
        if data.is_empty() {
            return Ok(Array1::zeros(0));
        }
        match ma_type {
            MovingAverageType::SMA => Self::sma(data, window),
            MovingAverageType::EMA => Self::ema(data, window),
            MovingAverageType::WMA => Self::wma(data, window),
        }
    }

    /// Simple Moving Average.
    pub fn sma(data: &Array1<f64>, window: usize) -> Result<Array1<f64>, MathError> {
        if window == 0 {
            return Err(MathError::InvalidParam { reason: "window must be > 0".into() });
        }
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        if n < window { return Ok(result); }

        let mut window_sum = data.slice(s![..window]).sum();
        result[window - 1] = window_sum / window as f64;
        for i in window..n {
            window_sum += data[i] - data[i - window];
            result[i] = window_sum / window as f64;
        }
        Ok(result)
    }

    /// Exponential Moving Average.
    pub fn ema(data: &Array1<f64>, window: usize) -> Result<Array1<f64>, MathError> {
        if window == 0 {
            return Err(MathError::InvalidParam { reason: "window must be > 0".into() });
        }
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        if n < window { return Ok(result); }

        let alpha = 2.0 / (window as f64 + 1.0);
        let mut ema_val = data.slice(s![..window]).sum() / window as f64;
        result[window - 1] = ema_val;
        for i in window..n {
            ema_val = alpha * data[i] + (1.0 - alpha) * ema_val;
            result[i] = ema_val;
        }
        Ok(result)
    }

    /// Weighted Moving Average.
    pub fn wma(data: &Array1<f64>, window: usize) -> Result<Array1<f64>, MathError> {
        if window == 0 {
            return Err(MathError::InvalidParam { reason: "window must be > 0".into() });
        }
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        if n < window { return Ok(result); }

        let weight_sum = (window * (window + 1)) as f64 / 2.0;
        for i in (window - 1)..n {
            let start = i + 1 - window;
            let mut weighted_sum = 0.0;
            for j in 0..window {
                weighted_sum += data[start + j] * (j as f64 + 1.0);
            }
            result[i] = weighted_sum / weight_sum;
        }
        Ok(result)
    }
}

/// Method for computing volatility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityMethod {
    Realized,
    Parkinson,
    GarmanKlass,
}

/// Computes various volatility estimators for financial time series.
pub struct Volatility;

impl Volatility {
    /// Annualized realized volatility from close prices.
    pub fn realized(closes: &Array1<f64>, periods_per_year: f64) -> Result<f64, MathError> {
        if closes.len() < 2 {
            return Err(MathError::InsufficientData { required: 2, actual: closes.len() });
        }
        let log_returns: Vec<f64> = (1..closes.len()).map(|i| (closes[i] / closes[i - 1]).ln()).collect();
        let arr = Array1::from(log_returns);
        let std = crate::math::stats::std_dev(&arr)?;
        Ok(std * periods_per_year.sqrt())
    }

    /// Parkinson volatility estimator.
    pub fn parkinson(highs: &Array1<f64>, lows: &Array1<f64>, periods_per_year: f64) -> Result<f64, MathError> {
        if highs.len() < 1 || highs.len() != lows.len() {
            return Err(MathError::InsufficientData { required: 1, actual: highs.len() });
        }
        let n = highs.len() as f64;
        let factor = 1.0 / (4.0 * n * 2.0_f64.ln());
        let sum_sq: f64 = (0..highs.len())
            .map(|i| { let hl = (highs[i] / lows[i]).ln(); hl * hl })
            .sum();
        Ok((factor * sum_sq).sqrt() * periods_per_year.sqrt())
    }

    /// Garman-Klass volatility estimator.
    pub fn garman_klass(
        opens: &Array1<f64>, highs: &Array1<f64>, lows: &Array1<f64>, closes: &Array1<f64>,
        periods_per_year: f64,
    ) -> Result<f64, MathError> {
        let n = closes.len();
        if n < 1 { return Err(MathError::InsufficientData { required: 1, actual: 0 }); }
        if opens.len() != n || highs.len() != n || lows.len() != n {
            return Err(MathError::DimensionMismatch {
                expected: n, actual: opens.len().max(highs.len()).max(lows.len()),
            });
        }
        let coeff = 2.0_f64.ln() - 1.0;
        let sum: f64 = (0..n).map(|i| {
            let hl = (highs[i] / lows[i]).ln();
            let co = (closes[i] / opens[i]).ln();
            0.5 * hl * hl - coeff * co * co
        }).sum();
        Ok((sum / n as f64).sqrt() * periods_per_year.sqrt())
    }
}

/// Method for computing correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    Pearson,
}

/// Compute a correlation matrix from a 2-D array.
pub fn correlation_matrix(data: &Array2<f64>, method: CorrelationMethod) -> Result<Array2<f64>, MathError> {
    match method {
        CorrelationMethod::Pearson => pearson_correlation_matrix(data),
    }
}

fn pearson_correlation_matrix(data: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let (n, p) = data.dim();
    if n < 2 { return Err(MathError::InsufficientData { required: 2, actual: n }); }
    let cov = covariance_matrix(data)?;
    let mut corr = Array2::<f64>::eye(p);
    for i in 0..p {
        for j in (i + 1)..p {
            let denom = (cov[[i, i]] * cov[[j, j]]).sqrt();
            let r = if denom.abs() < 1e-15 { 0.0 } else { cov[[i, j]] / denom };
            corr[[i, j]] = r;
            corr[[j, i]] = r;
        }
    }
    Ok(corr)
}

/// Covariance matrix (sample, with Bessel's correction).
pub fn covariance_matrix(data: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let (n, p) = data.dim();
    if n < 2 { return Err(MathError::InsufficientData { required: 2, actual: n }); }
    let means = data.mean_axis(ndarray::Axis(0)).unwrap();
    let centered = data - &means;
    Ok(centered.t().dot(&centered) / (n - 1) as f64)
}

/// Pearson correlation coefficient between two 1-D arrays.
pub fn pearson_corr(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, MathError> {
    if x.len() != y.len() { return Err(MathError::DimensionMismatch { expected: x.len(), actual: y.len() }); }
    if x.len() < 2 { return Err(MathError::InsufficientData { required: 2, actual: x.len() }); }
    let mx = x.mean().unwrap_or(0.0);
    let my = y.mean().unwrap_or(0.0);
    let n = x.len() as f64;
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov_xy += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    Ok(if denom < 1e-15 { 0.0 } else { cov_xy / denom })
}

/// Quantile computation utilities.
pub struct Quantile;

impl Quantile {
    /// Compute a quantile using linear interpolation (R-7 method).
    pub fn compute(data: &Array1<f64>, q: f64) -> Result<f64, MathError> {
        if data.is_empty() { return Err(MathError::InsufficientData { required: 1, actual: 0 }); }
        if !(0.0..=1.0).contains(&q) {
            return Err(MathError::InvalidParam { reason: format!("q must be in [0,1], got {}", q) });
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 1 { return Ok(sorted[0]); }
        let idx = q * (n as f64 - 1.0);
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        Ok(if lo == hi { sorted[lo] } else {
            let frac = idx - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        })
    }

    pub fn compute_many(data: &Array1<f64>, quantiles: &[f64]) -> Result<Vec<f64>, MathError> {
        quantiles.iter().map(|&q| Self::compute(data, q)).collect()
    }

    pub fn median(data: &Array1<f64>) -> Result<f64, MathError> { Self::compute(data, 0.5) }

    pub fn iqr(data: &Array1<f64>) -> Result<f64, MathError> {
        Ok(Self::compute(data, 0.75)? - Self::compute(data, 0.25)?)
    }
}

/// Common statistical tests.
pub struct StatisticalTest;

impl StatisticalTest {
    /// One-sample z-test: H0: μ = μ0.
    pub fn z_test_one_sample(data: &Array1<f64>, mu0: f64, alpha: f64) -> Result<ZTestResult, MathError> {
        if data.len() < 2 { return Err(MathError::InsufficientData { required: 2, actual: data.len() }); }
        let mean = data.mean().unwrap();
        let std = stats::std_dev(data)?;
        let n = data.len() as f64;
        let z = (mean - mu0) / (std / n.sqrt());
        let normal = Normal::new(0.0, 1.0).map_err(|e| MathError::InvalidParam { reason: e.to_string() })?;
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));
        Ok(ZTestResult { z_statistic: z, p_value, reject_null: p_value < alpha })
    }

    /// One-sample t-test: H0: μ = μ0.
    pub fn t_test_one_sample(data: &Array1<f64>, mu0: f64, alpha: f64) -> Result<TTestResult, MathError> {
        if data.len() < 2 { return Err(MathError::InsufficientData { required: 2, actual: data.len() }); }
        let mean = data.mean().unwrap();
        let std = stats::std_dev(data)?;
        let n = data.len() as f64;
        let df = n - 1.0;
        let t = (mean - mu0) / (std / n.sqrt());
        let normal = Normal::new(0.0, 1.0).map_err(|e| MathError::InvalidParam { reason: e.to_string() })?;
        let p_value = 2.0 * (1.0 - normal.cdf(t.abs()));
        Ok(TTestResult { t_statistic: t, df, p_value, reject_null: p_value < alpha })
    }

    /// Two-sample t-test (Welch's): H0: μ1 = μ2.
    pub fn t_test_two_sample(a: &Array1<f64>, b: &Array1<f64>, alpha: f64) -> Result<TTestResult, MathError> {
        if a.len() < 2 || b.len() < 2 {
            return Err(MathError::InsufficientData { required: 2, actual: a.len().min(b.len()) });
        }
        let mean_a = a.mean().unwrap();
        let mean_b = b.mean().unwrap();
        let var_a = stats::variance(a)?;
        let var_b = stats::variance(b)?;
        let n_a = a.len() as f64;
        let n_b = b.len() as f64;
        let se = (var_a / n_a + var_b / n_b).sqrt();
        if se < 1e-15 { return Err(MathError::ZeroVariance); }
        let t = (mean_a - mean_b) / se;
        let num = (var_a / n_a + var_b / n_b).powi(2);
        let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
        let df = if denom.abs() < 1e-15 { n_a + n_b - 2.0 } else { num / denom };
        let normal = Normal::new(0.0, 1.0).map_err(|e| MathError::InvalidParam { reason: e.to_string() })?;
        let p_value = 2.0 * (1.0 - normal.cdf(t.abs()));
        Ok(TTestResult { t_statistic: t, df, p_value, reject_null: p_value < alpha })
    }
}

#[derive(Debug, Clone)]
pub struct ZTestResult {
    pub z_statistic: f64,
    pub p_value: f64,
    pub reject_null: bool,
}

#[derive(Debug, Clone)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub df: f64,
    pub p_value: f64,
    pub reject_null: bool,
}

/// Aggregated portfolio performance and risk metrics.
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub var: f64,
    pub cvar: f64,
    pub calmar_ratio: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl PortfolioMetrics {
    /// Compute all portfolio metrics from a series of returns.
    pub fn compute(
        returns: &Array1<f64>, periods_per_year: f64, risk_free_rate: f64, var_confidence: f64,
    ) -> Result<PortfolioMetrics, MathError> {
        if returns.len() < 2 { return Err(MathError::InsufficientData { required: 2, actual: returns.len() }); }
        let mean_ret = returns.mean().unwrap();
        let std_ret = stats::std_dev(returns)?;
        let annualized_return = mean_ret * periods_per_year;
        let annualized_volatility = std_ret * periods_per_year.sqrt();
        let excess = annualized_return - risk_free_rate;
        let sharpe_ratio = if annualized_volatility.abs() < 1e-15 { 0.0 } else { excess / annualized_volatility };
        let downside: f64 = returns.iter().filter(|&&r| r < 0.0).map(|&r| r * r).sum::<f64>() / returns.len() as f64;
        let downside_dev = downside.sqrt() * periods_per_year.sqrt();
        let sortino_ratio = if downside_dev.abs() < 1e-15 { 0.0 } else { excess / downside_dev };
        let max_drawdown = Self::max_drawdown(returns);
        let var = Self::value_at_risk(returns, var_confidence)?;
        let cvar = Self::conditional_var(returns, var_confidence)?;
        let calmar_ratio = if max_drawdown.abs() < 1e-15 { 0.0 } else { annualized_return / max_drawdown };
        let skewness = stats::skewness(returns)?;
        let kurtosis = stats::excess_kurtosis(returns)?;
        Ok(PortfolioMetrics { annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio, max_drawdown, var, cvar, calmar_ratio, skewness, kurtosis })
    }

    /// Maximum drawdown from a series of returns (positive value).
    pub fn max_drawdown(returns: &Array1<f64>) -> f64 {
        if returns.is_empty() { return 0.0; }
        let mut cumulative = 1.0;
        let mut peak = f64::NEG_INFINITY;
        let mut max_dd = 0.0_f64;
        for &r in returns.iter() {
            cumulative *= 1.0 + r;
            if cumulative > peak { peak = cumulative; }
            let dd = (peak - cumulative) / peak;
            if dd > max_dd { max_dd = dd; }
        }
        max_dd
    }

    /// Historical Value at Risk (positive number).
    pub fn value_at_risk(returns: &Array1<f64>, confidence: f64) -> Result<f64, MathError> {
        if returns.is_empty() { return Err(MathError::InsufficientData { required: 1, actual: 0 }); }
        Ok(-Quantile::compute(returns, 1.0 - confidence)?)
    }

    /// Conditional Value at Risk (Expected Shortfall).
    pub fn conditional_var(returns: &Array1<f64>, confidence: f64) -> Result<f64, MathError> {
        if returns.is_empty() { return Err(MathError::InsufficientData { required: 1, actual: 0 }); }
        let var_val = Self::value_at_risk(returns, confidence)?;
        let threshold = -var_val;
        let tail: Vec<f64> = returns.iter().filter(|&&r| r <= threshold).cloned().collect();
        Ok(if tail.is_empty() { var_val } else { -tail.iter().sum::<f64>() / tail.len() as f64 })
    }
}

/// Internal descriptive statistics.
pub(crate) mod stats {
    use super::MathError;
    use ndarray::Array1;

    pub fn std_dev(data: &Array1<f64>) -> Result<f64, MathError> { variance(data).map(|v| v.sqrt()) }

    pub fn variance(data: &Array1<f64>) -> Result<f64, MathError> {
        if data.len() < 2 { return Err(MathError::InsufficientData { required: 2, actual: data.len() }); }
        let mean = data.mean().unwrap();
        Ok(data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64)
    }

    pub fn skewness(data: &Array1<f64>) -> Result<f64, MathError> {
        if data.len() < 3 { return Err(MathError::InsufficientData { required: 3, actual: data.len() }); }
        let n = data.len() as f64;
        let mean = data.mean().unwrap();
        let m3 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
        let m2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        Ok(m3 / m2.max(1e-15).powf(1.5))
    }

    pub fn excess_kurtosis(data: &Array1<f64>) -> Result<f64, MathError> {
        if data.len() < 4 { return Err(MathError::InsufficientData { required: 4, actual: data.len() }); }
        let n = data.len() as f64;
        let mean = data.mean().unwrap();
        let m4 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
        let m2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        Ok(m4 / m2.max(1e-15).powi(2) - 3.0)
    }
}

/// Element-wise operations on 1-D arrays.
pub struct VecOps;

impl VecOps {
    pub fn dot(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, MathError> {
        if a.len() != b.len() { return Err(MathError::DimensionMismatch { expected: a.len(), actual: b.len() }); }
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }
    pub fn add(a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>, MathError> {
        if a.len() != b.len() { return Err(MathError::DimensionMismatch { expected: a.len(), actual: b.len() }); }
        Ok(a + b)
    }
    pub fn sub(a: &Array1<f64>, b: &Array1<f64>) -> Result<Array1<f64>, MathError> {
        if a.len() != b.len() { return Err(MathError::DimensionMismatch { expected: a.len(), actual: b.len() }); }
        Ok(a - b)
    }
    pub fn scale(a: &Array1<f64>, scalar: f64) -> Array1<f64> { a * scalar }
    pub fn norm_l2(a: &Array1<f64>) -> f64 { a.iter().map(|&x| x * x).sum::<f64>().sqrt() }
    pub fn norm_l1(a: &Array1<f64>) -> f64 { a.iter().map(|&x| x.abs()).sum() }
    pub fn mean(a: &Array1<f64>) -> f64 { if a.is_empty() { 0.0 } else { a.mean().unwrap() } }
    pub fn cumsum(a: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(a.len());
        if a.is_empty() { return out; }
        out[0] = a[0];
        for i in 1..a.len() { out[i] = out[i - 1] + a[i]; }
        out
    }
    pub fn clip(a: &Array1<f64>, min: f64, max: f64) -> Array1<f64> { a.mapv(|x| x.clamp(min, max)) }
}

/// Matrix utilities using ndarray.
pub struct MatOps;

impl MatOps {
    pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let (_, k1) = a.dim();
        let (k2, _) = b.dim();
        if k1 != k2 { return Err(MathError::DimensionMismatch { expected: k1, actual: k2 }); }
        Ok(a.dot(b))
    }
    pub fn transpose(a: &Array2<f64>) -> Array2<f64> { a.t().to_owned() }
    pub fn row_means(a: &Array2<f64>) -> Array1<f64> { a.mean_axis(ndarray::Axis(0)).unwrap_or_else(|| Array1::zeros(0)) }
    pub fn col_means(a: &Array2<f64>) -> Array1<f64> { a.mean_axis(ndarray::Axis(1)).unwrap_or_else(|| Array1::zeros(0)) }
    pub fn identity(n: usize) -> Array2<f64> { Array2::eye(n) }
    pub fn column(a: &Array2<f64>, col: usize) -> Result<Array1<f64>, MathError> {
        let (_, ncols) = a.dim();
        if col >= ncols { return Err(MathError::DimensionMismatch { expected: ncols, actual: col + 1 }); }
        Ok(a.column(col).to_owned())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn price_data() -> Array1<f64> {
        Array1::from_vec(vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0])
    }

    fn assert_close(a: f64, b: f64, eps: f64) { assert!((a - b).abs() < eps, "|{} - {}| = {} >= {}", a, b, (a - b).abs(), eps); }

    #[test]
    fn test_sma() {
        let data = price_data();
        let result = MovingAverage::sma(&data, 3).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_close(result[2], (100.0 + 102.0 + 101.0) / 3.0, 1e-10);
        assert_close(result[3], (102.0 + 101.0 + 103.0) / 3.0, 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = price_data();
        let result = MovingAverage::ema(&data, 3).unwrap();
        assert!(result[0].is_nan());
        assert_close(result[2], (100.0 + 102.0 + 101.0) / 3.0, 1e-10);
    }

    #[test]
    fn test_wma() {
        let data = price_data();
        let result = MovingAverage::wma(&data, 3).unwrap();
        let expected = (100.0 * 1.0 + 102.0 * 2.0 + 101.0 * 3.0) / 6.0;
        assert_close(result[2], expected, 1e-10);
    }

    #[test]
    fn test_realized_volatility() {
        let data = price_data();
        let vol = Volatility::realized(&data, 252.0).unwrap();
        assert!(vol > 0.0);
    }

    #[test]
    fn test_parkinson_volatility() {
        let highs = Array1::from_vec(vec![102.0, 104.0, 105.0, 108.0, 112.0]);
        let lows = Array1::from_vec(vec![99.0, 100.0, 103.0, 106.0, 108.0]);
        let vol = Volatility::parkinson(&highs, &lows, 252.0).unwrap();
        assert!(vol > 0.0);
    }

    #[test]
    fn test_garman_klass_volatility() {
        let opens = Array1::from_vec(vec![100.0, 102.0, 103.0, 106.0, 108.0]);
        let highs = Array1::from_vec(vec![102.0, 104.0, 105.0, 108.0, 112.0]);
        let lows = Array1::from_vec(vec![99.0, 100.0, 103.0, 106.0, 108.0]);
        let closes = Array1::from_vec(vec![102.0, 103.0, 106.0, 108.0, 111.0]);
        let vol = Volatility::garman_klass(&opens, &highs, &lows, &closes, 252.0).unwrap();
        assert!(vol > 0.0);
    }

    #[test]
    fn test_correlation_matrix() {
        let data = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]).unwrap();
        let corr = pearson_correlation_matrix(&data).unwrap();
        assert_close(corr[[0, 1]], 1.0, 1e-6);
    }

    #[test]
    fn test_covariance_matrix() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]).unwrap();
        let cov = covariance_matrix(&data).unwrap();
        assert_close(cov[[0, 0]], 5.0 / 3.0, 1e-6);
        assert_close(cov[[1, 1]], 5.0 / 3.0, 1e-6);
    }

    #[test]
    fn test_quantile() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_close(Quantile::compute(&data, 0.5).unwrap(), 5.5, 1e-6);
        assert_close(Quantile::compute(&data, 0.0).unwrap(), 1.0, 1e-6);
        assert_close(Quantile::compute(&data, 1.0).unwrap(), 10.0, 1e-6);
    }

    #[test]
    fn test_median_and_iqr() {
        let data = Array1::from_vec(vec![3.0, 1.0, 2.0]);
        assert_close(Quantile::median(&data).unwrap(), 2.0, 1e-6);
        let data2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_close(Quantile::iqr(&data2).unwrap(), 4.5, 1e-6);
    }

    #[test]
    fn test_z_test() {
        let data = Array1::from_vec(vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.9, 5.1, 5.0]);
        let result = StatisticalTest::z_test_one_sample(&data, 5.0, 0.05).unwrap();
        assert!(!result.reject_null);
    }

    #[test]
    fn test_t_tests() {
        let data = Array1::from_vec(vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.9, 5.1, 5.0]);
        assert!(!StatisticalTest::t_test_one_sample(&data, 5.0, 0.05).unwrap().reject_null);
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = StatisticalTest::t_test_two_sample(&a, &b, 0.05).unwrap();
        assert!(result.t_statistic.abs() > 0.0);
    }

    #[test]
    fn test_portfolio_metrics() {
        let returns = Array1::from_vec(vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.005, 0.015, 0.008, -0.01, 0.025, -0.015]);
        let metrics = PortfolioMetrics::compute(&returns, 252.0, 0.0, 0.95).unwrap();
        assert!(metrics.annualized_volatility > 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.var >= 0.0);
        assert!(metrics.cvar >= metrics.var);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = Array1::from_vec(vec![0.1, -0.05, -0.05, 0.1]);
        let dd = PortfolioMetrics::max_drawdown(&returns);
        assert!(dd > 0.09 && dd < 0.10);
    }

    #[test]
    fn test_vec_ops() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        assert_close(VecOps::dot(&a, &b).unwrap(), 32.0, 1e-10);
        assert_close(VecOps::norm_l2(&a), 14.0_f64.sqrt(), 1e-10);
        assert_close(VecOps::norm_l1(&a), 6.0, 1e-10);
        assert_close(VecOps::mean(&a), 2.0, 1e-10);
    }

    #[test]
    fn test_mat_ops() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = MatOps::matmul(&a, &MatOps::identity(2)).unwrap();
        assert_close(result[[0, 0]], 1.0, 1e-10);
        assert_close(result[[1, 1]], 4.0, 1e-10);
    }

    #[test]
    fn test_insufficient_data_and_errors() {
        let data = Array1::from_vec(vec![1.0]);
        assert!(MovingAverage::sma(&data, 5).unwrap().iter().all(|v| v.is_nan()));
        assert!(MovingAverage::sma(&price_data(), 0).is_err());
    }
}
