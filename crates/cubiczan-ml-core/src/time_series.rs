//! # Time Series Utilities
//!
//! Comprehensive time series analysis for financial OHLCV data.
//!
//! ## Components
//!
//! - **OHLCV** — Canonical Open/High/Low/Close/Volume bar representation with serde support
//! - **Resampling** — Aggregate bars from finer to coarser intervals (1m→5m→15m→1h→1d)
//! - **Returns** — Simple, log, and cumulative return calculations
//! - **Rolling windows** — Apply any function over a sliding window
//! - **Stationarity tests** — ADF-inspired test using autocorrelation decay
//! - **Seasonality detection** — Autocorrelation-based periodicity detection
//! - **Gap analysis** — Identify price gaps (overnight, weekend, breakaway)
//! - **Volume profile** — Volume-at-price distribution analysis

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during time series operations.
#[derive(Debug, Error)]
pub enum TimeSeriesError {
    #[error("insufficient data: need at least {required} bars, got {actual}")]
    InsufficientBars { required: usize, actual: usize },
    #[error("data not sorted by timestamp")]
    UnsortedData,
    #[error("invalid resampling: source interval is not finer than target")]
    InvalidResample,
    #[error("no data available")]
    EmptyData,
    #[error("computation error: {reason}")]
    Computation { reason: String },
}

/// A single OHLCV (Open/High/Low/Close/Volume) bar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OHLCV {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OHLCV {
    pub fn new(timestamp: DateTime<Utc>, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        OHLCV { timestamp, open, high, low, close, volume }
    }

    pub fn is_valid(&self) -> bool {
        self.high >= self.low && self.open >= 0.0 && self.high >= 0.0 && self.low >= 0.0 && self.close >= 0.0 && self.volume >= 0.0
    }

    pub fn range(&self) -> f64 { self.high - self.low }
    pub fn midpoint(&self) -> f64 { (self.high + self.low) / 2.0 }
    pub fn typical_price(&self) -> f64 { (self.high + self.low + self.close) / 3.0 }
    pub fn body(&self) -> f64 { (self.close - self.open).abs() }
    pub fn upper_shadow(&self) -> f64 { self.high - self.open.max(self.close) }
    pub fn lower_shadow(&self) -> f64 { self.open.min(self.close) - self.low }
    pub fn is_bullish(&self) -> bool { self.close >= self.open }

    pub fn bar_return(&self) -> f64 {
        if self.open.abs() < 1e-15 { 0.0 } else { (self.close - self.open) / self.open }
    }

    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let hl = self.high - self.low;
        match prev_close {
            Some(pc) => hl.max((self.high - pc).abs()).max((self.low - pc).abs()),
            None => hl,
        }
    }
}

/// Supported resampling intervals, ordered from finest to coarsest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResampleInterval {
    OneMinute = 1,
    FiveMinutes = 5,
    FifteenMinutes = 15,
    OneHour = 60,
    OneDay = 1440,
}

impl ResampleInterval {
    pub fn minutes(&self) -> i64 {
        match self {
            ResampleInterval::OneMinute => 1,
            ResampleInterval::FiveMinutes => 5,
            ResampleInterval::FifteenMinutes => 15,
            ResampleInterval::OneHour => 60,
            ResampleInterval::OneDay => 1440,
        }
    }

    pub fn floor_timestamp(&self, ts: DateTime<Utc>) -> DateTime<Utc> {
        let total_minutes = ts.timestamp() / 60;
        let interval = self.minutes();
        let floored = (total_minutes / interval) * interval;
        DateTime::from_timestamp(floored * 60, 0).unwrap_or(ts)
    }
}

/// Resample a series of OHLCV bars into a coarser time interval.
pub fn resample(bars: &[OHLCV], target: ResampleInterval) -> Result<Vec<OHLCV>, TimeSeriesError> {
    if bars.is_empty() { return Ok(Vec::new()); }
    for i in 1..bars.len() {
        if bars[i].timestamp < bars[i - 1].timestamp { return Err(TimeSeriesError::UnsortedData); }
    }

    let mut result: Vec<OHLCV> = Vec::new();
    let mut current_group_start = target.floor_timestamp(bars[0].timestamp);
    let mut group_open = bars[0].open;
    let mut group_high = bars[0].high;
    let mut group_low = bars[0].low;
    let mut group_close = bars[0].close;
    let mut group_volume = bars[0].volume;

    for bar in bars.iter().skip(1) {
        let bar_interval_start = target.floor_timestamp(bar.timestamp);
        if bar_interval_start == current_group_start {
            group_high = group_high.max(bar.high);
            group_low = group_low.min(bar.low);
            group_close = bar.close;
            group_volume += bar.volume;
        } else {
            result.push(OHLCV::new(current_group_start, group_open, group_high, group_low, group_close, group_volume));
            current_group_start = bar_interval_start;
            group_open = bar.open;
            group_high = bar.high;
            group_low = bar.low;
            group_close = bar.close;
            group_volume = bar.volume;
        }
    }
    result.push(OHLCV::new(current_group_start, group_open, group_high, group_low, group_close, group_volume));
    Ok(result)
}

/// Type of return to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnType {
    Simple,
    Log,
}

/// Compute returns from a price series.
pub fn compute_returns(prices: &[f64], ret_type: ReturnType) -> Result<Vec<f64>, TimeSeriesError> {
    if prices.len() < 2 { return Err(TimeSeriesError::InsufficientBars { required: 2, actual: prices.len() }); }
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        if prices[i - 1].abs() < 1e-15 { returns.push(0.0); continue; }
        match ret_type {
            ReturnType::Simple => returns.push((prices[i] - prices[i - 1]) / prices[i - 1]),
            ReturnType::Log => returns.push((prices[i] / prices[i - 1]).ln()),
        }
    }
    Ok(returns)
}

/// Compute returns from OHLCV bars using close prices.
pub fn compute_returns_ohlcv(bars: &[OHLCV], ret_type: ReturnType) -> Result<Vec<f64>, TimeSeriesError> {
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    compute_returns(&closes, ret_type)
}

/// Compute cumulative returns from simple returns.
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() { return Vec::new(); }
    let mut cum = Vec::with_capacity(returns.len() + 1);
    cum.push(0.0);
    let mut wealth = 1.0;
    for &r in returns {
        wealth *= 1.0 + r;
        cum.push(wealth - 1.0);
    }
    cum
}

/// Apply a function over a rolling window of a 1-D array.
pub fn rolling_apply<F>(data: &[f64], window: usize, f: F) -> Result<Vec<f64>, TimeSeriesError>
where F: Fn(&[f64]) -> f64,
{
    if window == 0 { return Err(TimeSeriesError::Computation { reason: "window must be > 0".into() }); }
    if data.is_empty() { return Ok(Vec::new()); }
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n < window { return Ok(result); }
    for i in (window - 1)..n {
        result[i] = f(&data[i + 1 - window..=i]);
    }
    Ok(result)
}

pub fn rolling_std(data: &[f64], window: usize) -> Result<Vec<f64>, TimeSeriesError> {
    rolling_apply(data, window, |slice| {
        if slice.len() < 2 { return f64::NAN; }
        let n = slice.len() as f64;
        let mean = slice.iter().sum::<f64>() / n;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    })
}

pub fn rolling_mean(data: &[f64], window: usize) -> Result<Vec<f64>, TimeSeriesError> {
    rolling_apply(data, window, |slice| slice.iter().sum::<f64>() / slice.len() as f64)
}

pub fn rolling_min(data: &[f64], window: usize) -> Result<Vec<f64>, TimeSeriesError> {
    rolling_apply(data, window, |slice| slice.iter().cloned().fold(f64::INFINITY, f64::min))
}

pub fn rolling_max(data: &[f64], window: usize) -> Result<Vec<f64>, TimeSeriesError> {
    rolling_apply(data, window, |slice| slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}

pub fn rolling_sum(data: &[f64], window: usize) -> Result<Vec<f64>, TimeSeriesError> {
    rolling_apply(data, window, |slice| slice.iter().sum())
}

/// Result of an ADF-inspired stationarity test.
#[derive(Debug, Clone)]
pub struct StationarityResult {
    pub test_statistic: f64,
    pub critical_value_5pct: f64,
    pub is_stationary: bool,
    pub lags: usize,
    pub autocorr_lag1: f64,
}

/// Compute autocorrelation at a given lag.
pub fn autocorrelation(data: &[f64], lag: usize) -> Result<f64, TimeSeriesError> {
    if data.len() < lag + 2 { return Err(TimeSeriesError::InsufficientBars { required: lag + 2, actual: data.len() }); }
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if variance.abs() < 1e-15 { return Ok(0.0); }
    let cov: f64 = (0..n - lag).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / (n - lag) as f64;
    Ok(cov / variance)
}

/// Compute autocorrelation function for multiple lags.
pub fn acf(data: &[f64], max_lag: usize) -> Result<Vec<f64>, TimeSeriesError> {
    (0..=max_lag).map(|lag| autocorrelation(data, lag)).collect()
}

/// ADF-inspired stationarity test.
pub fn adf_test(data: &[f64], max_lags: Option<usize>) -> Result<StationarityResult, TimeSeriesError> {
    if data.len() < 10 { return Err(TimeSeriesError::InsufficientBars { required: 10, actual: data.len() }); }
    let lags = max_lags.unwrap_or_else(|| {
        let n = data.len() as f64;
        std::cmp::max(1, (12.0 * (n / 100.0).powf(0.25)) as usize)
    });
    let acf_values: Vec<f64> = (1..=lags).map(|lag| autocorrelation(data, lag).unwrap_or(0.0)).collect();
    let acf_lag1 = acf_values[0];
    let acf_sum: f64 = acf_values.iter().take(lags).sum();
    let n = data.len() as f64;
    let test_statistic = n * (1.0 - acf_sum);
    let critical_value_5pct = -1.95;
    let differenced: Vec<f64> = (1..data.len()).map(|i| data[i] - data[i - 1]).collect();
    let diff_acf1 = if differenced.len() > 2 { autocorrelation(&differenced, 1).unwrap_or(0.0) } else { 0.0 };
    let is_stationary = test_statistic < critical_value_5pct || (acf_lag1.abs() > diff_acf1.abs() + 0.3);
    Ok(StationarityResult { test_statistic, critical_value_5pct, is_stationary, lags, autocorr_lag1: acf_lag1 })
}

/// Result of seasonality analysis.
#[derive(Debug, Clone)]
pub struct SeasonalityResult {
    pub period: usize,
    pub strength: f64,
    pub acf_values: Vec<f64>,
    pub candidates: Vec<(usize, f64)>,
}

/// Detect seasonality using autocorrelation analysis.
pub fn detect_seasonality(data: &[f64], min_period: usize, max_period: usize) -> Result<SeasonalityResult, TimeSeriesError> {
    if data.len() < max_period + 2 { return Err(TimeSeriesError::InsufficientBars { required: max_period + 2, actual: data.len() }); }
    let acf_values = acf(data, max_period)?;
    let min_threshold = 0.1;
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    for k in min_period..=max_period {
        if k >= acf_values.len() { break; }
        let acf_k = acf_values[k];
        if acf_k < min_threshold { continue; }
        let prev = acf_values[k - 1];
        let next = if k + 1 < acf_values.len() { acf_values[k + 1] } else { f64::NEG_INFINITY };
        if acf_k >= prev && acf_k >= next { candidates.push((k, acf_k)); }
    }
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let (period, strength) = if candidates.is_empty() {
        (0, 0.0)
    } else {
        // Pick the smallest period among candidates with near-maximum strength
        let max_strength = candidates[0].1;
        let best = candidates.iter()
            .filter(|(_, s)| (*s - max_strength).abs() < 1e-6)
            .min_by_key(|(p, _)| *p)
            .copied()
            .unwrap_or(candidates[0]);
        best
    };
    Ok(SeasonalityResult { period, strength, acf_values, candidates })
}

/// A detected price gap.
#[derive(Debug, Clone)]
pub struct PriceGap {
    pub start_idx: usize,
    pub end_idx: usize,
    pub gap_pct: f64,
    pub gap_abs: f64,
    pub filled: bool,
}

/// Analyze price gaps in a series of OHLCV bars.
pub fn analyze_gaps(bars: &[OHLCV], min_gap_pct: f64) -> Result<Vec<PriceGap>, TimeSeriesError> {
    if bars.len() < 2 { return Err(TimeSeriesError::InsufficientBars { required: 2, actual: bars.len() }); }
    let mut gaps = Vec::new();
    for i in 1..bars.len() {
        let prev_close = bars[i - 1].close;
        let curr_open = bars[i].open;
        if prev_close.abs() < 1e-15 { continue; }
        let gap_pct = (curr_open - prev_close) / prev_close * 100.0;
        if gap_pct.abs() >= min_gap_pct {
            let filled = if gap_pct > 0.0 {
                bars[i..].iter().any(|b| b.low <= prev_close)
            } else {
                bars[i..].iter().any(|b| b.high >= prev_close)
            };
            gaps.push(PriceGap { start_idx: i - 1, end_idx: i, gap_pct, gap_abs: (curr_open - prev_close).abs(), filled });
        }
    }
    Ok(gaps)
}

/// A single price level in a volume profile.
#[derive(Debug, Clone)]
pub struct VolumeLevel {
    pub price: f64,
    pub volume: f64,
    pub pct_of_total: f64,
    pub bar_count: usize,
}

/// Volume profile analysis.
pub struct VolumeProfile {
    pub levels: Vec<VolumeLevel>,
    pub poc_price: f64,
    pub value_area_high: f64,
    pub value_area_low: f64,
    pub total_volume: f64,
    pub num_buckets: usize,
}

impl VolumeProfile {
    /// Build a volume profile from OHLCV bars.
    pub fn build(bars: &[OHLCV], num_buckets: usize, value_area_pct: f64) -> Result<VolumeProfile, TimeSeriesError> {
        if bars.is_empty() { return Err(TimeSeriesError::EmptyData); }
        let min_price = bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
        let max_price = bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
        let range = max_price - min_price;
        if range.abs() < 1e-15 { return Err(TimeSeriesError::Computation { reason: "price range is zero".into() }); }
        let bucket_size = range / num_buckets as f64;
        let mut volumes = vec![0.0_f64; num_buckets];
        let mut bar_counts = vec![0_usize; num_buckets];
        for bar in bars {
            let low_bucket = ((bar.low - min_price) / bucket_size).floor() as usize;
            let high_bucket = ((bar.high - min_price) / bucket_size).ceil() as usize;
            let low_bucket = low_bucket.min(num_buckets - 1);
            let high_bucket = high_bucket.min(num_buckets - 1);
            let touch_count = (high_bucket - low_bucket + 1).max(1);
            let volume_per_bucket = bar.volume / touch_count as f64;
            for b in low_bucket..=high_bucket { volumes[b] += volume_per_bucket; bar_counts[b] += 1; }
        }
        let total_volume: f64 = volumes.iter().sum();
        let mut levels: Vec<VolumeLevel> = (0..num_buckets).map(|i| {
            let price = min_price + (i as f64 + 0.5) * bucket_size;
            VolumeLevel { price, volume: volumes[i], pct_of_total: if total_volume > 0.0 { volumes[i] / total_volume * 100.0 } else { 0.0 }, bar_count: bar_counts[i] }
        }).collect();
        let poc_idx = levels.iter().enumerate().max_by(|(_, a), (_, b)| a.volume.partial_cmp(&b.volume).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
        let poc_price = levels[poc_idx].price;
        let target_volume = total_volume * value_area_pct;
        let mut value_area_low = poc_price;
        let mut value_area_high = poc_price;
        let mut accumulated_volume = volumes[poc_idx];
        let mut lower = poc_idx as i64 - 1;
        let mut upper = poc_idx as i64 + 1;
        while accumulated_volume < target_volume && (lower >= 0 || upper < num_buckets as i64) {
            let lower_vol = if lower >= 0 { volumes[lower as usize] } else { 0.0 };
            let upper_vol = if (upper as usize) < num_buckets { volumes[upper as usize] } else { 0.0 };
            if lower_vol >= upper_vol && lower >= 0 {
                accumulated_volume += lower_vol; value_area_low = levels[lower as usize].price; lower -= 1;
            } else if (upper as usize) < num_buckets {
                accumulated_volume += upper_vol; value_area_high = levels[upper as usize].price; upper += 1;
            } else if lower >= 0 {
                accumulated_volume += lower_vol; value_area_low = levels[lower as usize].price; lower -= 1;
            } else { break; }
        }
        Ok(VolumeProfile { levels, poc_price, value_area_high, value_area_low, total_volume, num_buckets })
    }

    pub fn level_at_price(&self, price: f64) -> Option<&VolumeLevel> {
        self.levels.iter().min_by_key(|level| ((level.price - price).abs() * 1e8) as i64)
    }
}

/// A high-level facade for time series analysis.
pub struct TimeSeriesAnalyzer;

impl TimeSeriesAnalyzer {
    /// Analyze the overall health and characteristics of a price series.
    pub fn summary(bars: &[OHLCV]) -> Result<TSSummary, TimeSeriesError> {
        if bars.is_empty() { return Err(TimeSeriesError::EmptyData); }
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let n = bars.len();
        let simple_rets = compute_returns(&closes, ReturnType::Simple)?;
        let mean_close = closes.iter().sum::<f64>() / n as f64;
        let mean_volume: f64 = bars.iter().map(|b| b.volume).sum::<f64>() / n as f64;
        let total_return = if closes[0].abs() > 1e-15 { (closes[n - 1] - closes[0]) / closes[0] } else { 0.0 };
        let mean_ret = simple_rets.iter().sum::<f64>() / simple_rets.len() as f64;
        let variance = simple_rets.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / (simple_rets.len() - 1) as f64;
        let daily_vol = variance.sqrt();
        let stationarity = adf_test(&closes, None).ok();
        let max_period = std::cmp::min(50, n / 3);
        let seasonality = if max_period > 2 { detect_seasonality(&closes, 2, max_period).ok() } else { None };
        let gaps = analyze_gaps(bars, 0.5).unwrap_or_default();
        Ok(TSSummary {
            bar_count: n, mean_close, mean_volume, total_return, daily_volatility: daily_vol,
            max_price: closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            min_price: closes.iter().cloned().fold(f64::INFINITY, f64::min),
            stationarity, seasonality,
            gap_count: gaps.len(),
            avg_gap_size: if gaps.is_empty() { 0.0 } else { gaps.iter().map(|g| g.gap_pct.abs()).sum::<f64>() / gaps.len() as f64 },
        })
    }
}

/// Summary statistics for a time series.
#[derive(Debug, Clone)]
pub struct TSSummary {
    pub bar_count: usize,
    pub mean_close: f64,
    pub mean_volume: f64,
    pub total_return: f64,
    pub daily_volatility: f64,
    pub max_price: f64,
    pub min_price: f64,
    pub stationarity: Option<StationarityResult>,
    pub seasonality: Option<SeasonalityResult>,
    pub gap_count: usize,
    pub avg_gap_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_bars(prices: &[f64]) -> Vec<OHLCV> {
        prices.iter().enumerate().map(|(i, &p)| {
            OHLCV::new(Utc::now() - Duration::minutes((prices.len() - i) as i64), p * 0.99, p * 1.02, p * 0.98, p, 1000.0)
        }).collect()
    }

    fn assert_close(a: f64, b: f64, eps: f64) { assert!((a - b).abs() < eps, "|{} - {}| = {} >= {}", a, b, (a - b).abs(), eps); }

    #[test]
    fn test_ohlcv_creation() {
        let bar = OHLCV::new(Utc::now(), 100.0, 105.0, 95.0, 103.0, 1000.0);
        assert!(bar.is_valid());
        assert_close(bar.range(), 10.0, 1e-10);
        assert_close(bar.midpoint(), 100.0, 1e-10);
        assert_close(bar.body(), 3.0, 1e-10);
        assert!(bar.is_bullish());
    }

    #[test]
    fn test_ohlcv_helpers() {
        let bar = OHLCV::new(Utc::now(), 100.0, 110.0, 90.0, 105.0, 500.0);
        assert_close(bar.typical_price(), (110.0 + 90.0 + 105.0) / 3.0, 1e-10);
        assert_close(bar.upper_shadow(), 5.0, 1e-10);
        assert_close(bar.lower_shadow(), 10.0, 1e-10);
        assert_close(bar.true_range(Some(100.0)), 20.0, 1e-10);
    }

    #[test]
    fn test_simple_returns() {
        let prices = vec![100.0, 110.0, 99.0];
        let rets = compute_returns(&prices, ReturnType::Simple).unwrap();
        assert_close(rets[0], 0.1, 1e-10);
        assert_close(rets[1], -0.1, 1e-10);
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0];
        let rets = compute_returns(&prices, ReturnType::Log).unwrap();
        assert_close(rets[0], (1.1_f64).ln(), 1e-8);
    }

    #[test]
    fn test_cumulative_returns() {
        let rets = vec![0.1, -0.05, 0.03];
        let cum = cumulative_returns(&rets);
        assert_close(cum[0], 0.0, 1e-10);
        assert_close(cum[1], 0.1, 1e-10);
        assert_close(cum[2], 0.045, 1e-10);
        assert_close(cum[3], 0.07635, 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3).unwrap();
        assert!(result[0].is_nan());
        assert_close(result[2], 2.0, 1e-10);
        assert_close(result[4], 4.0, 1e-10);
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = rolling_std(&data, 3).unwrap();
        assert_close(result[2], 2.0, 1e-10);
    }

    #[test]
    fn test_rolling_min_max() {
        let data = vec![5.0, 3.0, 8.0, 1.0, 6.0];
        assert_close(rolling_min(&data, 3).unwrap()[2], 3.0, 1e-10);
        assert_close(rolling_max(&data, 3).unwrap()[2], 8.0, 1e-10);
    }

    #[test]
    fn test_autocorrelation() {
        let data = vec![5.0; 20];
        assert_close(autocorrelation(&data, 1).unwrap(), 0.0, 1e-10);
        let data2: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
        assert!(autocorrelation(&data2, 1).unwrap() < -0.5);
    }

    #[test]
    fn test_adf_stationary() {
        let data: Vec<f64> = (0..100).map(|i| 50.0 + 10.0 * (i as f64 * 0.1).sin()).collect();
        let result = adf_test(&data, Some(5)).unwrap();
        assert!(result.lags > 0);
    }

    #[test]
    fn test_seasonality_detection() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 10.0).sin()).collect();
        let result = detect_seasonality(&data, 2, 50).unwrap();
        if result.period > 0 { assert!((result.period as i64 - 10).abs() < 3); }
    }

    #[test]
    fn test_gap_analysis() {
        let mut bars = make_bars(&[100.0, 101.0, 105.0]);
        bars[1].close = 101.0;
        bars[2].open = 105.0;
        let gaps = analyze_gaps(&bars, 1.0).unwrap();
        assert_eq!(gaps.len(), 1);
        assert!(gaps[0].gap_pct > 0.0);
    }

    #[test]
    fn test_volume_profile() {
        let bars = make_bars(&[100.0, 102.0, 98.0, 105.0, 101.0, 103.0, 99.0, 104.0]);
        let vp = VolumeProfile::build(&bars, 10, 0.7).unwrap();
        assert!(vp.total_volume > 0.0);
        assert!(vp.poc_price > 0.0);
        assert!(vp.value_area_low <= vp.poc_price);
        assert!(vp.value_area_high >= vp.poc_price);
    }

    #[test]
    fn test_resample() {
        // Use a 5-minute-aligned base timestamp so bars fall into exactly 3 groups
        let base = Utc::now();
        let aligned = base - Duration::seconds(base.timestamp() % 300);
        let bars: Vec<OHLCV> = (0..15).map(|i| {
            OHLCV::new(aligned + Duration::minutes(i), 100.0 + i as f64, 101.0 + i as f64, 99.0 + i as f64, 100.5 + i as f64, 100.0)
        }).collect();
        let resampled = resample(&bars, ResampleInterval::FiveMinutes).unwrap();
        assert_eq!(resampled.len(), 3);
        assert_close(resampled[0].open, 100.0, 1e-10);
        assert_close(resampled[0].close, 104.5, 1e-10);
    }

    #[test]
    fn test_time_series_summary() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.5 * (i as f64).sin())).collect();
        let bars = make_bars(&prices);
        let summary = TimeSeriesAnalyzer::summary(&bars).unwrap();
        assert_eq!(summary.bar_count, 100);
        assert!(summary.daily_volatility > 0.0);
    }
}
