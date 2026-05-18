//! Anomaly detection engine – Z-score, moving average deviation, volume spikes,
//! pattern matching, and statistical distribution tests.

use crate::types::{
    AnomalyScore, AnomalyType, Chain, DetectionConfig, TimeSeriesPoint, Transaction,
};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DetectionResult
// ---------------------------------------------------------------------------

/// The outcome of running anomaly detection on a single data point.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub anomaly_type: AnomalyType,
    pub score: AnomalyScore,
    pub description: String,
    pub tx_hash: Option<String>,
}

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

/// Compute the mean of a slice.
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the population standard deviation.
fn std_dev(data: &[f64]) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Compute a simple moving average of the last `window` values.
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let slice = &data[start..=i];
        result.push(mean(slice));
    }
    result
}

/// Compute the Z-score of a value against a reference distribution.
pub fn zscore(value: f64, data: &[f64]) -> f64 {
    let m = mean(data);
    let s = std_dev(data);
    if s < f64::EPSILON {
        return 0.0;
    }
    (value - m) / s
}

// ---------------------------------------------------------------------------
// KS test (two-sample, simplified)
// ---------------------------------------------------------------------------

/// Simplified Kolmogorov–Smirnov two-sample test.
/// Returns the KS statistic D (in [0, 1]).
pub fn ks_test(sample_a: &[f64], sample_b: &[f64]) -> f64 {
    if sample_a.is_empty() || sample_b.is_empty() {
        return 0.0;
    }
    let mut a_sorted = sample_a.to_vec();
    let mut b_sorted = sample_b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let n_a = a_sorted.len() as f64;
    let n_b = b_sorted.len() as f64;

    let ecdf = |sorted: &[f64], x: f64, n: f64| -> f64 {
        let count = sorted.iter().filter(|&&v| v <= x).count();
        count as f64 / n
    };

    // Evaluate at all unique values from both samples
    let mut all_vals: Vec<f64> = a_sorted.iter().chain(b_sorted.iter()).copied().collect();
    all_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    all_vals.dedup();

    let mut max_d = 0.0_f64;
    for &v in &all_vals {
        let d = (ecdf(&a_sorted, v, n_a) - ecdf(&b_sorted, v, n_b)).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

// ---------------------------------------------------------------------------
// Chi-squared goodness-of-fit (binned)
// ---------------------------------------------------------------------------

/// Simplified χ² goodness-of-fit test.
/// Returns (chi2_statistic, p_value_approximation).
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> (f64, f64) {
    if observed.len() != expected.len() || observed.is_empty() {
        return (0.0, 1.0);
    }
    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(o, e)| {
            if *e < f64::EPSILON {
                0.0
            } else {
                (o - e).powi(2) / e
            }
        })
        .sum();
    let k = (observed.len() - 1).max(1) as f64;
    // Simple normal approximation of χ² for large k
    let approx = Normal::new(k, (2.0 * k).sqrt()).unwrap();
    let p_value = 1.0 - approx.cdf(chi2);
    (chi2, p_value)
}

// ---------------------------------------------------------------------------
// AnomalyDetector
// ---------------------------------------------------------------------------

/// The main anomaly detection engine.
pub struct AnomalyDetector {
    config: DetectionConfig,
    // Per-chain time-series of transaction values (sliding window).
    value_history: HashMap<Chain, Vec<f64>>,
    // Per-chain tx count per block (sliding window).
    volume_history: HashMap<Chain, Vec<f64>>,
    // Gas-price history per chain.
    gas_history: HashMap<Chain, Vec<f64>>,
}

impl AnomalyDetector {
    pub fn new(config: DetectionConfig) -> Self {
        Self {
            config,
            value_history: HashMap::new(),
            volume_history: HashMap::new(),
            gas_history: HashMap::new(),
        }
    }

    /// Ingest a time-series point (used for bulk historical warm-up).
    pub fn feed(&mut self, chain: Chain, point: TimeSeriesPoint) {
        self.push_value(chain, point.value);
        self.push_volume(chain, point.volume);
    }

    /// Feed a transaction into the historical buffers.
    pub fn feed_transaction(&mut self, tx: &Transaction) {
        self.push_value(tx.chain, tx.value);
        self.push_gas(tx.chain, tx.gas_price);
    }

    /// Feed a block-level tx count.
    pub fn feed_block_tx_count(&mut self, chain: Chain, count: f64) {
        self.push_volume(chain, count);
    }

    /// Run full anomaly detection on a single transaction.
    pub fn detect(&self, tx: &Transaction) -> Vec<DetectionResult> {
        let mut results = Vec::new();

        // 1) Z-score value outlier
        if let Some(result) = self.detect_value_outlier(tx) {
            results.push(result);
        }

        // 2) Gas-price deviation
        if let Some(result) = self.detect_gas_deviation(tx) {
            results.push(result);
        }

        // 3) Pattern matching (placeholder: checks for extreme value)
        if self.config.enable_pattern_match {
            if let Some(result) = self.detect_pattern(tx) {
                results.push(result);
            }
        }

        results
    }

    /// Detect volume spike for a given chain given a new tx count.
    pub fn detect_volume_spike(&self, chain: Chain, current_count: f64) -> Option<DetectionResult> {
        let history = self.volume_history.get(&chain)?;
        if history.len() < self.config.min_data_points {
            return None;
        }
        let avg = mean(history);
        if avg < f64::EPSILON {
            return None;
        }
        let ratio = current_count / avg;
        if ratio > self.config.volume_spike_multiplier {
            let score = AnomalyScore::new((ratio / self.config.volume_spike_multiplier).min(1.0));
            return Some(DetectionResult {
                anomaly_type: AnomalyType::VolumeSpike,
                score,
                description: format!(
                    "Volume spike: {} txs vs avg {:.1} ({:.1}x)",
                    current_count, avg, ratio
                ),
                tx_hash: None,
            });
        }
        None
    }

    /// KS distribution shift test between two chains.
    pub fn detect_distribution_shift(&self, chain_a: Chain, chain_b: Chain) -> Option<DetectionResult> {
        let a = self.value_history.get(&chain_a)?;
        let b = self.value_history.get(&chain_b)?;
        if a.len() < self.config.min_data_points || b.len() < self.config.min_data_points {
            return None;
        }
        let d = ks_test(a, b);
        let score = AnomalyScore::new((d / 0.3).min(1.0));
        if score.is_medium() {
            return Some(DetectionResult {
                anomaly_type: AnomalyType::DistributionShift,
                score,
                description: format!(
                    "Distribution shift between {} and {}: D={:.4}",
                    chain_a, chain_b, d
                ),
                tx_hash: None,
            });
        }
        None
    }

    // -- private helpers ---------------------------------------------------

    fn detect_value_outlier(&self, tx: &Transaction) -> Option<DetectionResult> {
        let history = self.value_history.get(&tx.chain)?;
        if history.len() < self.config.min_data_points {
            return None;
        }
        let z = zscore(tx.value, history).abs();
        if z > self.config.zscore_threshold {
            // Map z to a score using normal CDF tail probability approximation
            let norm = Normal::new(0.0, 1.0).ok()?;
            let tail = 1.0 - norm.cdf(z);
            let score = AnomalyScore::new((1.0 - tail).min(1.0));
            return Some(DetectionResult {
                anomaly_type: AnomalyType::ValueOutlier,
                score,
                description: format!("Value outlier: z={:.2}", z),
                tx_hash: Some(tx.hash.clone()),
            });
        }
        None
    }

    fn detect_gas_deviation(&self, tx: &Transaction) -> Option<DetectionResult> {
        let history = self.gas_history.get(&tx.chain)?;
        if history.len() < self.config.min_data_points {
            return None;
        }
        let ma = moving_average(history, self.config.moving_avg_window.min(history.len()));
        if let Some(&last_ma) = ma.last() {
            if last_ma < f64::EPSILON {
                return None;
            }
            let deviation = ((tx.gas_price - last_ma) / last_ma).abs();
            if deviation > 0.5 {
                // > 50 % deviation
                let score = AnomalyScore::new((deviation / 2.0).min(1.0));
                return Some(DetectionResult {
                    anomaly_type: AnomalyType::GasPriceDeviation,
                    score,
                    description: format!("Gas deviation: {:.1}%", deviation * 100.0),
                    tx_hash: Some(tx.hash.clone()),
                });
            }
        }
        None
    }

    fn detect_pattern(&self, tx: &Transaction) -> Option<DetectionResult> {
        // Simple heuristic: extremely high value relative to historical max
        let history = self.value_history.get(&tx.chain)?;
        if history.is_empty() {
            return None;
        }
        let max_val = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val < f64::EPSILON {
            return None;
        }
        let ratio = tx.value / max_val;
        if ratio > 10.0 {
            let score = AnomalyScore::new(0.95);
            return Some(DetectionResult {
                anomaly_type: AnomalyType::PatternMatch,
                score,
                description: format!(
                    "Pattern match: tx value {:.2} is {:.1}x historical max {:.2}",
                    tx.value, ratio, max_val
                ),
                tx_hash: Some(tx.hash.clone()),
            });
        }
        None
    }

    // -- buffer helpers ----------------------------------------------------

    fn push_value(&mut self, chain: Chain, val: f64) {
        let window = self.config.sliding_window_size;
        let buf = self.value_history.entry(chain).or_default();
        buf.push(val);
        if buf.len() > window {
            let drain = buf.len() - window;
            buf.drain(0..drain);
        }
    }

    fn push_volume(&mut self, chain: Chain, val: f64) {
        let window = self.config.sliding_window_size;
        let buf = self.volume_history.entry(chain).or_default();
        buf.push(val);
        if buf.len() > window {
            let drain = buf.len() - window;
            buf.drain(0..drain);
        }
    }

    fn push_gas(&mut self, chain: Chain, val: f64) {
        let window = self.config.sliding_window_size;
        let buf = self.gas_history.entry(chain).or_default();
        buf.push(val);
        if buf.len() > window {
            let drain = buf.len() - window;
            buf.drain(0..drain);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeSeriesPoint;
    use chrono::Utc;

    fn normal_data() -> Vec<f64> {
        (0..200).map(|i| 50.0 + (i as f64) * 0.1).collect()
    }

    fn make_config() -> DetectionConfig {
        let mut cfg = DetectionConfig::default();
        cfg.min_data_points = 10;
        cfg.sliding_window_size = 500;
        cfg.moving_avg_window = 10;
        cfg.zscore_threshold = 2.0;
        cfg.volume_spike_multiplier = 3.0;
        cfg
    }

    fn make_tx(chain: Chain, value: f64, gas: f64) -> Transaction {
        Transaction {
            hash: format!("0x{:08x}", rand::random::<u32>()),
            chain,
            block_number: 1,
            from_address: "0xA".into(),
            to_address: "0xB".into(),
            value,
            gas_used: 21000.0,
            gas_price: gas,
            timestamp: Utc::now(),
            token_symbol: None,
            memo: None,
        }
    }

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < f64::EPSILON);
        assert!((mean(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_std_dev() {
        let sd = std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((sd - 2.0).abs() < 0.01);
        assert!((std_dev(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&data, 3);
        assert_eq!(ma.len(), 5);
        assert!((ma[0] - 1.0).abs() < f64::EPSILON);
        assert!((ma[1] - 1.5).abs() < f64::EPSILON);
        assert!((ma[2] - 2.0).abs() < f64::EPSILON);
        assert!((ma[3] - 3.0).abs() < f64::EPSILON);
        assert!((ma[4] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_moving_average_empty() {
        assert!(moving_average(&[], 5).is_empty());
        assert!(moving_average(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn test_zscore_normal() {
        let data = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let z = zscore(12.0, &data);
        assert!(z.abs() < 1.0); // 12 is near the mean
    }

    #[test]
    fn test_zscore_outlier() {
        let data = vec![10.0, 11.0, 9.0, 10.5, 9.5];
        let z = zscore(100.0, &data).abs();
        assert!(z > 2.0);
    }

    #[test]
    fn test_ks_test_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = ks_test(&data, &data);
        assert!(d < f64::EPSILON);
    }

    #[test]
    fn test_ks_test_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let d = ks_test(&a, &b);
        assert!(d > 0.5);
    }

    #[test]
    fn test_chi_squared_similar() {
        let obs = vec![10.0, 10.0, 10.0, 10.0];
        let exp = vec![10.0, 10.0, 10.0, 10.0];
        let (stat, _p) = chi_squared_test(&obs, &exp);
        assert!(stat < f64::EPSILON);
    }

    #[test]
    fn test_chi_squared_different() {
        let obs = vec![50.0, 0.0, 0.0, 0.0];
        let exp = vec![12.5, 12.5, 12.5, 12.5];
        let (stat, _p) = chi_squared_test(&obs, &exp);
        assert!(stat > 100.0);
    }

    #[test]
    fn test_anomaly_detector_value_outlier() {
        let cfg = make_config();
        let mut det = AnomalyDetector::new(cfg);
        // Warm up with normal data
        for v in &normal_data() {
            det.push_value(Chain::Ethereum, *v);
        }
        let tx = make_tx(Chain::Ethereum, 1000.0, 30.0);
        let results = det.detect(&tx);
        let has_outlier = results.iter().any(|r| r.anomaly_type == AnomalyType::ValueOutlier);
        assert!(has_outlier, "Expected ValueOutlier for extreme tx value");
    }

    #[test]
    fn test_anomaly_detector_no_false_positive() {
        let cfg = make_config();
        let mut det = AnomalyDetector::new(cfg);
        for v in &normal_data() {
            det.push_value(Chain::Ethereum, *v);
            det.push_gas(Chain::Ethereum, 30.0);
        }
        let tx = make_tx(Chain::Ethereum, 55.0, 30.0); // within normal range
        let results = det.detect(&tx);
        assert!(results.is_empty(), "Normal tx should not trigger anomalies");
    }

    #[test]
    fn test_volume_spike_detection() {
        let cfg = make_config();
        let mut det = AnomalyDetector::new(cfg);
        for _ in 0..50 {
            det.push_volume(Chain::Ethereum, 10.0);
        }
        let result = det.detect_volume_spike(Chain::Ethereum, 100.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().anomaly_type, AnomalyType::VolumeSpike);
    }

    #[test]
    fn test_distribution_shift() {
        let cfg = make_config();
        let mut det = AnomalyDetector::new(cfg);
        for _ in 0..50 {
            det.push_value(Chain::Ethereum, 10.0);
            det.push_value(Chain::Solana, 1000.0);
        }
        let result = det.detect_distribution_shift(Chain::Ethereum, Chain::Solana);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap().anomaly_type,
            AnomalyType::DistributionShift
        );
    }

    #[test]
    fn test_feed_time_series_point() {
        let cfg = make_config();
        let mut det = AnomalyDetector::new(cfg);
        let pt = TimeSeriesPoint {
            timestamp: Utc::now(),
            value: 42.0,
            volume: 10.0,
        };
        det.feed(Chain::Ethereum, pt);
        let hist = det.value_history.get(&Chain::Ethereum).unwrap();
        assert_eq!(hist.len(), 1);
        assert!((hist[0] - 42.0).abs() < f64::EPSILON);
    }
}
