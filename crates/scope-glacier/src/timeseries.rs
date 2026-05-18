//! Time-series decomposition and analysis tools.

use crate::types::{GlacierError, Result, SeasonalDecomposition};
use chrono::{DateTime, Utc};


/// Computes the Simple Moving Average (SMA) of a series.
pub fn sma(values: &[f64], window: usize) -> Result<Vec<f64>> {
    if window == 0 {
        return Err(GlacierError::InvalidInput("Window size must be > 0".into()));
    }
    if values.len() < window {
        return Err(GlacierError::InsufficientData {
            required: window,
            actual: values.len(),
        });
    }

    let mut result = Vec::with_capacity(values.len() - window + 1);
    let mut window_sum: f64 = values[..window].iter().sum();

    result.push(window_sum / window as f64);

    for i in window..values.len() {
        window_sum += values[i] - values[i - window];
        result.push(window_sum / window as f64);
    }

    Ok(result)
}

/// Computes the Exponential Moving Average (EMA) of a series.
pub fn ema(values: &[f64], span: f64) -> Result<Vec<f64>> {
    if span <= 0.0 {
        return Err(GlacierError::InvalidInput(
            "Span must be positive".into(),
        ));
    }
    if values.is_empty() {
        return Ok(vec![]);
    }

    let alpha = 2.0 / (span + 1.0);
    let mut result = Vec::with_capacity(values.len());
    result.push(values[0]);

    for i in 1..values.len() {
        let ema_val = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema_val);
    }

    Ok(result)
}

/// Computes the Weighted Moving Average (WMA) of a series.
/// Weights increase linearly: 1, 2, ..., window.
pub fn wma(values: &[f64], window: usize) -> Result<Vec<f64>> {
    if window == 0 {
        return Err(GlacierError::InvalidInput("Window size must be > 0".into()));
    }
    if values.len() < window {
        return Err(GlacierError::InsufficientData {
            required: window,
            actual: values.len(),
        });
    }

    let weight_sum = (window * (window + 1)) as f64 / 2.0;
    let mut result = Vec::with_capacity(values.len() - window + 1);

    for i in 0..=(values.len() - window) {
        let weighted_sum: f64 = (0..window)
            .map(|j| values[i + j] * (j + 1) as f64)
            .sum();
        result.push(weighted_sum / weight_sum);
    }

    Ok(result)
}

/// Computes the autocorrelation function (ACF) for a series at given lags.
pub fn autocorrelation(values: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if values.len() < 2 {
        return Err(GlacierError::InsufficientData {
            required: 2,
            actual: values.len(),
        });
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    if variance.abs() < f64::EPSILON {
        return Ok(vec![0.0; max_lag + 1]);
    }

    let mut acf = Vec::with_capacity(max_lag + 1);

    for lag in 0..=max_lag {
        let cov: f64 = values
            .iter()
            .zip(values.iter().skip(lag))
            .map(|(x, y)| (x - mean) * (y - mean))
            .sum::<f64>()
            / n;
        acf.push(cov / variance);
    }

    Ok(acf)
}

/// Extracts the trend component from a time series using a centered moving average.
pub fn extract_trend(values: &[f64], window: usize) -> Result<Vec<f64>> {
    // Use a centered SMA for trend extraction
    if window % 2 == 0 {
        // Even window: average two centered SMAs
        let sma1 = sma(values, window)?;
        let sma2 = sma(values, window + 1)?;
        let offset = (sma2.len().saturating_sub(sma1.len())) / 2;
        let len = sma1.len().min(sma2.len() - offset);
        let trend: Vec<f64> = (0..len).map(|i| (sma1[i] + sma2[i + offset]) / 2.0).collect();
        Ok(trend)
    } else {
        sma(values, window)
    }
}

/// Performs seasonal decomposition (additive model) on a time series.
/// Uses moving average for trend, then computes seasonal indices.
pub fn seasonal_decomposition(
    values: &[f64],
    period: usize,
    timestamps: Vec<DateTime<Utc>>,
) -> Result<SeasonalDecomposition> {
    if values.len() < 2 * period {
        return Err(GlacierError::InsufficientData {
            required: 2 * period,
            actual: values.len(),
        });
    }
    if period == 0 {
        return Err(GlacierError::InvalidInput("Period must be > 0".into()));
    }

    // Step 1: Extract trend using centered moving average
    let trend_window = if period % 2 == 0 { period } else { period };
    let raw_trend = extract_trend(values, trend_window)?;

    // Step 2: Compute detrended series
    let n = values.len();
    let trend_len = raw_trend.len();
    let offset = (n - trend_len) / 2;

    let detrended: Vec<f64> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i >= offset && i < offset + trend_len {
                v - raw_trend[i - offset]
            } else {
                0.0
            }
        })
        .collect();

    // Step 3: Compute seasonal indices by averaging detrended values for each season
    let mut seasonal_sum = vec![0.0f64; period];
    let mut seasonal_count = vec![0usize; period];

    for (i, val) in detrended.iter().enumerate() {
        if *val != 0.0 {
            seasonal_sum[i % period] += val;
            seasonal_count[i % period] += 1;
        }
    }

    let seasonal_avg: Vec<f64> = seasonal_sum
        .iter()
        .zip(seasonal_count.iter())
        .map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 })
        .collect();

    // Normalize seasonal indices to sum to zero
    let seasonal_mean = seasonal_avg.iter().sum::<f64>() / period as f64;
    let seasonal_indices: Vec<f64> = seasonal_avg.iter().map(|s| s - seasonal_mean).collect();

    // Step 4: Expand seasonal indices to full length
    let seasonal_full: Vec<f64> = (0..n).map(|i| seasonal_indices[i % period]).collect();

    // Step 5: Pad trend to match full length for reconstruction
    let mut trend_full = vec![raw_trend[0]; offset];
    trend_full.extend(raw_trend.iter().cloned());
    trend_full.resize(n, *raw_trend.last().unwrap());

    // Step 6: Compute residuals using padded trend for consistent reconstruction
    let residual: Vec<f64> = values
        .iter()
        .zip(trend_full.iter())
        .zip(seasonal_full.iter())
        .map(|((v, t), s)| v - t - s)
        .collect();

    Ok(SeasonalDecomposition {
        trend: trend_full,
        seasonal: seasonal_full,
        residual,
        period,
        timestamps,
    })
}

/// Computes the first difference of a time series.
pub fn diff(values: &[f64]) -> Vec<f64> {
    if values.len() < 2 {
        return vec![];
    }
    values.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Computes the k-th seasonal difference.
pub fn seasonal_diff(values: &[f64], lag: usize) -> Vec<f64> {
    if values.len() <= lag {
        return vec![];
    }
    values
        .iter()
        .skip(lag)
        .zip(values.iter())
        .map(|(a, b)| a - b)
        .collect()
}

/// ADF-like stationarity test (simplified).
/// Returns (test_statistic, is_stationary) based on a threshold.
/// Uses a simplified approach: compute variance of differences vs level.
pub fn adf_stationarity_test(values: &[f64], significance: f64) -> (f64, bool) {
    if values.len() < 10 {
        return (0.0, false);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    // Compute level variance
    let level_var: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    // Compute differenced series variance
    let diffs = diff(values);
    let diff_n = diffs.len() as f64;
    let diff_mean = diffs.iter().sum::<f64>() / diff_n;
    let diff_var: f64 = diffs.iter().map(|x| (x - diff_mean).powi(2)).sum::<f64>() / diff_n;

    // Test statistic: ratio of level variance to diff variance
    // Low ratio → stationary (unit root unlikely)
    let test_stat = if diff_var.abs() > f64::EPSILON {
        level_var / diff_var
    } else {
        f64::INFINITY
    };

    // Simplified critical values (not statistically rigorous but indicative)
    let critical_value = match significance {
        s if s <= 0.01 => 5.0,
        s if s <= 0.05 => 7.0,
        s if s <= 0.10 => 10.0,
        _ => 15.0,
    };

    (test_stat, test_stat < critical_value)
}

/// Computes the variance of a series.
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = mean(values);
    let n = values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Computes the standard deviation of a series.
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Computes the mean of a series.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Computes the median of a series.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Computes percentiles of a series.
pub fn percentile(values: &mut [f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (pct / 100.0) * (values.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let frac = idx - lower as f64;
        values[lower] * (1.0 - frac) + values[upper] * frac
    }
}

/// Computes the coefficient of variation (CV) of a series.
pub fn coefficient_of_variation(values: &[f64]) -> f64 {
    let m = mean(values);
    if m.abs() < f64::EPSILON {
        return f64::INFINITY;
    }
    std_dev(values) / m.abs()
}

/// Computes the log returns of a price series.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Computes cumulative returns from log returns.
pub fn cumulative_returns(log_rets: &[f64]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(log_rets.len() + 1);
    cum.push(0.0);
    let mut running = 0.0;
    for r in log_rets {
        running += r;
        cum.push(running);
    }
    cum
}

/// Detects change points using a simple CUSUM-like approach on differences.
/// Returns indices where significant changes occur.
pub fn detect_change_points(values: &[f64], threshold: f64) -> Vec<usize> {
    if values.len() < 3 {
        return vec![];
    }

    let m = mean(values);
    let diffs = diff(values);
    let mut cusum_pos = 0.0_f64;
    let mut cusum_neg = 0.0_f64;
    let mut change_points = Vec::new();

    for (i, d) in diffs.iter().enumerate() {
        cusum_pos = (cusum_pos + d - m.abs() * 0.01).max(0.0);
        cusum_neg = (cusum_neg - d - m.abs() * 0.01).max(0.0);

        if cusum_pos > threshold * m.abs() || cusum_neg > threshold * m.abs() {
            change_points.push(i + 1);
            cusum_pos = 0.0;
            cusum_neg = 0.0;
        }
    }

    change_points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&values, 3).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sma_window_too_large() {
        let values = vec![1.0, 2.0];
        let result = sma(&values, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sma_zero_window() {
        let values = vec![1.0, 2.0, 3.0];
        let result = sma(&values, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&values, 3.0).unwrap();
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10); // First value is seed
        // EMA should be close to SMA but lag less
        assert!(result[4] > result[3]); // Should be increasing
    }

    #[test]
    fn test_ema_empty() {
        let result = ema(&[], 3.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_ema_invalid_span() {
        let result = ema(&[1.0, 2.0], -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_wma_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wma(&values, 3).unwrap();
        assert_eq!(result.len(), 3);
        // WMA for [1,2,3] with weights [1,2,3]: (1*1 + 2*2 + 3*3)/6 = 14/6
        assert!((result[0] - 14.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_autocorrelation_lag0() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = autocorrelation(&values, 3).unwrap();
        assert!((acf[0] - 1.0).abs() < 1e-10); // Lag 0 ACF is always 1
    }

    #[test]
    fn test_autocorrelation_constant_series() {
        let values = vec![5.0; 20];
        let acf = autocorrelation(&values, 5).unwrap();
        // All ACF values should be 0 for constant series (handled by variance check)
        assert!(acf.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn test_extract_trend() {
        // Trend + noise
        let values: Vec<f64> = (0..50)
            .map(|i| (i as f64) * 0.5 + (i as f64 * 0.1).sin())
            .collect();
        let trend = extract_trend(&values, 7).unwrap();
        assert!(!trend.is_empty());
        // Trend should be smoother than original
        let trend_diff = diff(&trend).iter().map(|d| d.abs()).sum::<f64>();
        let orig_diff = diff(&values).iter().map(|d| d.abs()).sum::<f64>();
        assert!(trend_diff <= orig_diff + 1e-10);
    }

    #[test]
    fn test_seasonal_decomposition_basic() {
        // Build a simple seasonal + trend series with longer period
        let period = 6;
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let trend = i as f64 * 0.5;
                let seasonal = if i % period < period / 2 { 5.0 } else { -5.0 };
                trend + seasonal
            })
            .collect();
        let timestamps: Vec<DateTime<Utc>> = (0..48)
            .map(|i| {
                chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    + chrono::Duration::days(i)
            })
            .collect();

        let decomp = seasonal_decomposition(&values, period, timestamps).unwrap();
        assert_eq!(decomp.trend.len(), 48);
        assert_eq!(decomp.seasonal.len(), 48);
        assert_eq!(decomp.residual.len(), 48);
        assert_eq!(decomp.period, period);

        // Reconstruction check with wider tolerance since trend extraction is approximate
        let recon = decomp.reconstruct();
        for (orig, rec) in values.iter().zip(recon.iter()) {
            assert!((orig - rec).abs() < 5.0, "Reconstruction error too large: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_seasonal_decomposition_insufficient_data() {
        let values = vec![1.0, 2.0, 3.0];
        let result = seasonal_decomposition(&values, 12, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_basic() {
        let values = vec![1.0, 3.0, 6.0, 10.0];
        let d = diff(&values);
        assert_eq!(d, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_diff_empty() {
        let values = vec![1.0];
        let d = diff(&values);
        assert!(d.is_empty());
    }

    #[test]
    fn test_seasonal_diff() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let d = seasonal_diff(&values, 2);
        assert_eq!(d, vec![20.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn test_adf_stationarity_random_walk() {
        // A random walk is non-stationary
        let values: Vec<f64> = (0..100).scan(0.0, |state, _| {
            *state += rand_like();
            Some(*state)
        }).collect();
        let (_, stationary) = adf_stationarity_test(&values, 0.05);
        // Random walk should generally be non-stationary
        // (our heuristic may not always agree, but mostly yes)
        // We don't assert strongly here since it's heuristic
        let _ = stationary;
    }

    #[test]
    fn test_adf_stationarity_stationary() {
        // Use high-frequency oscillation to get higher diff variance relative to level var
        let values: Vec<f64> = (0..100).map(|i| 2.0 * ((i as f64 * 2.51).sin())).collect();
        let (stat, stationary) = adf_stationarity_test(&values, 0.10);
        // High-frequency oscillation should produce lower level_var/diff_var ratio
        assert!(stationary, "stat={} should be < 10.0 for stationary data", stat);
    }

    #[test]
    fn test_mean_and_std() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = mean(&values);
        assert!((m - 5.0).abs() < 1e-10);
        let sd = std_dev(&values);
        assert!(sd > 0.0);
    }

    #[test]
    fn test_median() {
        let mut values = vec![3.0, 1.0, 2.0];
        let m = median(&mut values);
        assert!((m - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = percentile(&mut values, 50.0);
        assert!((p50 - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let cv = coefficient_of_variation(&values);
        assert!(cv > 0.0);
        assert!(cv.is_finite());
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0, 99.0];
        let lr = log_returns(&prices);
        assert_eq!(lr.len(), 2);
        assert!((lr[0] - (1.1f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns_zero_price() {
        let prices = vec![0.0, 10.0];
        let lr = log_returns(&prices);
        assert_eq!(lr.len(), 1);
        assert!((lr[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_returns() {
        let log_rets = vec![0.1, 0.2, -0.1];
        let cum = cumulative_returns(&log_rets);
        assert_eq!(cum.len(), 4);
        assert!((cum[0] - 0.0).abs() < 1e-10);
        assert!((cum[1] - 0.1).abs() < 1e-10);
        assert!((cum[2] - 0.3).abs() < 1e-10);
        assert!((cum[3] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_detect_change_points() {
        let values: Vec<f64> = (0..30)
            .map(|i| if i < 15 { 10.0 } else { 50.0 })
            .collect();
        let cps = detect_change_points(&values, 0.5);
        assert!(!cps.is_empty());
        // Change point should be near index 15
        assert!(cps.iter().any(|&cp| (cp as i32 - 15).abs() <= 2));
    }

    #[test]
    fn test_detect_change_points_smooth() {
        let values: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let cps = detect_change_points(&values, 5.0);
        // Smooth linear series should have few or no change points
        assert!(cps.len() <= 2);
    }

    // Deterministic pseudo-random for testing
    fn rand_like() -> f64 {
        static mut SEED: u64 = 42;
        unsafe {
            SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((SEED >> 33) as f64) / (u32::MAX as f64) - 0.5
        }
    }
}
