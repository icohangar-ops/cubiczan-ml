//! # Anomaly Detection Engine
//!
//! Multi-method anomaly detection for mining sensor data:
//! - Z-score based single-point detection
//! - Moving average deviation detection
//! - IQR-based outlier detection
//! - Rate-of-change anomaly detection
//! - Persistent anomaly tracking
//! - Multi-sensor correlation anomalies
//! - Severity classification

use crate::sensors::SensorDatabase;
use crate::types::{AnomalyEvent, AnomalyType, SensorType, Severity};
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// Configuration for the anomaly detector.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Z-score threshold for spike detection (default: 3.0).
    pub zscore_threshold: f64,
    /// Window size for moving average (default: 20).
    pub ma_window: usize,
    /// Moving average deviation threshold (default: 2.5 std devs).
    pub ma_deviation_threshold: f64,
    /// IQR multiplier for outlier detection (default: 1.5).
    pub iqr_multiplier: f64,
    /// Rate-of-change threshold (default: 3.0 std devs of first differences).
    pub roc_threshold: f64,
    /// Correlation coefficient drop threshold for multi-sensor anomalies (default: 0.3).
    pub correlation_drop_threshold: f64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        AnomalyConfig {
            zscore_threshold: 3.0,
            ma_window: 20,
            ma_deviation_threshold: 2.5,
            iqr_multiplier: 1.5,
            roc_threshold: 3.0,
            correlation_drop_threshold: 0.3,
        }
    }
}

/// The anomaly detection engine.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    config: AnomalyConfig,
    /// Tracked active anomalies by sensor_id.
    active_anomalies: HashMap<String, Vec<AnomalyEvent>>,
    /// History of all resolved anomalies.
    anomaly_history: Vec<AnomalyEvent>,
    /// Baseline statistics per sensor: (mean, std_dev).
    baselines: HashMap<String, (f64, f64)>,
    /// Baseline correlation between sensor pairs.
    baseline_correlations: HashMap<(String, String), f64>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector with default config.
    pub fn new() -> Self {
        AnomalyDetector {
            config: AnomalyConfig::default(),
            active_anomalies: HashMap::new(),
            anomaly_history: Vec::new(),
            baselines: HashMap::new(),
            baseline_correlations: HashMap::new(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: AnomalyConfig) -> Self {
        AnomalyDetector {
            config,
            active_anomalies: HashMap::new(),
            anomaly_history: Vec::new(),
            baselines: HashMap::new(),
            baseline_correlations: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Baseline
    // -----------------------------------------------------------------------

    /// Compute and store baseline statistics for a sensor.
    pub fn compute_baseline(&mut self, sensor_id: &str, values: &[f64]) {
        if values.len() < 3 {
            return;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        self.baselines.insert(sensor_id.to_string(), (mean, std_dev));
    }

    /// Compute baseline correlation between two sensors.
    pub fn compute_baseline_correlation(&mut self, sensor_a: &str, sensor_b: &str, corr: f64) {
        let key = (sensor_a.to_string(), sensor_b.to_string());
        self.baseline_correlations.insert(key, corr);
    }

    /// Get the baseline (mean, std_dev) for a sensor.
    pub fn get_baseline(&self, sensor_id: &str) -> Option<(f64, f64)> {
        self.baselines.get(sensor_id).copied()
    }

    // -----------------------------------------------------------------------
    // Z-score detection
    // -----------------------------------------------------------------------

    /// Detect anomalies using Z-score method against baseline or computed stats.
    /// Returns indices of anomalous readings.
    pub fn detect_zscore(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < 3 {
            return Vec::new();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        if std_dev.abs() < 1e-15 {
            return Vec::new();
        }

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| ((v - mean).abs() / std_dev) > self.config.zscore_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Detect Z-score anomalies for a sensor in the database.
    pub fn detect_zscore_sensor(&self, db: &SensorDatabase, sensor_id: &str) -> Vec<AnomalyEvent> {
        let values = db.values(sensor_id);
        let readings = db.get_readings(sensor_id).unwrap_or(&[]);
        let anomaly_indices = self.detect_zscore(&values);

        anomaly_indices
            .into_iter()
            .map(|idx| {
                let reading = &readings[idx];
                let magnitude = if let Some((mean, std)) = self.get_baseline(sensor_id) {
                    ((reading.value - mean).abs() / std.max(1e-10)).min(10.0)
                } else {
                    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
                    let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / (values.len().max(1) as f64 - 1.0))
                    .sqrt();
                    ((reading.value - mean).abs() / std.max(1e-10)).min(10.0)
                };
                AnomalyEvent::new(
                    AnomalyType::Spike,
                    sensor_id,
                    reading.timestamp,
                    self.classify_severity(magnitude),
                    magnitude,
                    format!("Z-score anomaly: {:.2} at value {:.2}", magnitude, reading.value),
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Moving average deviation
    // -----------------------------------------------------------------------

    /// Detect anomalies using moving average deviation.
    /// Compares each value to a rolling mean and flags if deviation exceeds threshold.
    pub fn detect_ma_deviation(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < self.config.ma_window + 1 {
            return Vec::new();
        }
        let window = self.config.ma_window;
        let mut anomalies = Vec::new();

        for i in window..values.len() {
            let window_slice = &values[i - window..i];
            let w_mean = window_slice.iter().sum::<f64>() / window as f64;
            let w_std = (window_slice.iter().map(|v| (v - w_mean).powi(2)).sum::<f64>() / (window - 1) as f64).sqrt();

            if w_std.abs() > 1e-15 {
                let deviation = (values[i] - w_mean).abs() / w_std;
                if deviation > self.config.ma_deviation_threshold {
                    anomalies.push(i);
                }
            }
        }

        anomalies
    }

    /// Detect moving average deviation anomalies for a sensor.
    pub fn detect_ma_deviation_sensor(&self, db: &SensorDatabase, sensor_id: &str) -> Vec<AnomalyEvent> {
        let values = db.values(sensor_id);
        let readings = db.get_readings(sensor_id).unwrap_or(&[]);
        let anomaly_indices = self.detect_ma_deviation(&values);

        anomaly_indices
            .into_iter()
            .map(|idx| {
                let reading = &readings[idx];
                AnomalyEvent::new(
                    AnomalyType::Drift,
                    sensor_id,
                    reading.timestamp,
                    Severity::Warning,
                    (reading.value - values[..idx].iter().sum::<f64>() / idx.max(1) as f64).abs(),
                    format!("MA deviation at value {:.2}", reading.value),
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // IQR outlier detection
    // -----------------------------------------------------------------------

    /// Detect outliers using Interquartile Range method.
    pub fn detect_iqr(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < 4 {
            return Vec::new();
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = sorted.len() / 4;
        let q3_idx = (3 * sorted.len()) / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        if iqr.abs() < 1e-15 {
            return Vec::new();
        }

        let lower = q1 - self.config.iqr_multiplier * iqr;
        let upper = q3 + self.config.iqr_multiplier * iqr;

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < lower || v > upper)
            .map(|(i, _)| i)
            .collect()
    }

    /// Detect IQR anomalies for a sensor in the database.
    pub fn detect_iqr_sensor(&self, db: &SensorDatabase, sensor_id: &str) -> Vec<AnomalyEvent> {
        let values = db.values(sensor_id);
        let readings = db.get_readings(sensor_id).unwrap_or(&[]);
        let anomaly_indices = self.detect_iqr(&values);

        anomaly_indices
            .into_iter()
            .map(|idx| {
                let reading = &readings[idx];
                AnomalyEvent::new(
                    AnomalyType::OutOfRange,
                    sensor_id,
                    reading.timestamp,
                    Severity::Info,
                    reading.value.abs(),
                    format!("IQR outlier: value {:.2}", reading.value),
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Rate-of-change detection
    // -----------------------------------------------------------------------

    /// Detect sudden changes in rate of change (acceleration/deceleration).
    pub fn detect_rate_of_change(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < 4 {
            return Vec::new();
        }

        // Compute first differences
        let diffs: Vec<f64> = (1..values.len()).map(|i| values[i] - values[i - 1]).collect();

        if diffs.len() < 3 {
            return Vec::new();
        }

        let n = diffs.len() as f64;
        let mean_diff = diffs.iter().sum::<f64>() / n;
        let std_diff = (diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        if std_diff.abs() < 1e-15 {
            return Vec::new();
        }

        diffs
            .iter()
            .enumerate()
            .filter(|(_, &d)| ((d - mean_diff).abs() / std_diff) > self.config.roc_threshold)
            .map(|(i, _)| i + 1) // +1 to map back to original values index
            .collect()
    }

    /// Detect rate-of-change anomalies for a sensor.
    pub fn detect_roc_sensor(&self, db: &SensorDatabase, sensor_id: &str) -> Vec<AnomalyEvent> {
        let values = db.values(sensor_id);
        let readings = db.get_readings(sensor_id).unwrap_or(&[]);
        let anomaly_indices = self.detect_rate_of_change(&values);

        anomaly_indices
            .into_iter()
            .map(|idx| {
                let reading = &readings[idx];
                let prev_val = if idx > 0 { values[idx - 1] } else { reading.value };
                let change = (reading.value - prev_val).abs();
                AnomalyEvent::new(
                    AnomalyType::StepChange,
                    sensor_id,
                    reading.timestamp,
                    Severity::Warning,
                    change,
                    format!("Rate-of-change anomaly: delta = {:.2}", change),
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Persistent anomaly tracking
    // -----------------------------------------------------------------------

    /// Add an anomaly to tracking.
    pub fn track_anomaly(&mut self, anomaly: AnomalyEvent) {
        let sensor_id = anomaly.sensor_id.clone();
        self.active_anomalies
            .entry(sensor_id)
            .or_default()
            .push(anomaly);
    }

    /// Resolve the most recent active anomaly for a sensor.
    pub fn resolve_anomaly(&mut self, sensor_id: &str, end_time: DateTime<Utc>) -> bool {
        if let Some(anomalies) = self.active_anomalies.get_mut(sensor_id) {
            if let Some(last) = anomalies.last_mut() {
                if last.is_ongoing() {
                    last.resolve(end_time);
                    let resolved = last.clone();
                    self.anomaly_history.push(resolved);
                    return true;
                }
            }
        }
        false
    }

    /// Check if a sensor has active (ongoing) anomalies.
    pub fn has_active_anomaly(&self, sensor_id: &str) -> bool {
        self.active_anomalies
            .get(sensor_id)
            .map(|anomalies| anomalies.iter().any(|a| a.is_ongoing()))
            .unwrap_or(false)
    }

    /// Get all active (ongoing) anomalies.
    pub fn active_anomalies(&self) -> Vec<&AnomalyEvent> {
        self.active_anomalies
            .values()
            .flatten()
            .filter(|a| a.is_ongoing())
            .collect()
    }

    /// Get count of active anomalies by sensor.
    pub fn active_count(&self) -> usize {
        self.active_anomalies
            .values()
            .flatten()
            .filter(|a| a.is_ongoing())
            .count()
    }

    /// Get anomaly history.
    pub fn history(&self) -> &[AnomalyEvent] {
        &self.anomaly_history
    }

    // -----------------------------------------------------------------------
    // Multi-sensor correlation anomalies
    // -----------------------------------------------------------------------

    /// Detect correlation breakdown between sensor pairs.
    /// When two sensors that normally correlate start diverging, it can indicate
    /// equipment issues (e.g., vibration rises but temperature doesn't follow).
    pub fn detect_correlation_anomaly(
        &self,
        db: &SensorDatabase,
        sensor_a: &str,
        sensor_b: &str,
        window_size: usize,
    ) -> Option<AnomalyEvent> {
        let readings_a = db.get_readings(sensor_a)?;
        let readings_b = db.get_readings(sensor_b)?;
        if readings_a.len() < window_size * 2 || readings_b.len() < window_size * 2 {
            return None;
        }

        let baseline = self
            .baseline_correlations
            .get(&(sensor_a.to_string(), sensor_b.to_string()))
            .copied()
            .unwrap_or_else(|| {
                // Compute from first half
                let n = (readings_a.len().min(readings_b.len())) / 2;
                let vals_a: Vec<f64> = readings_a[..n].iter().map(|r| r.value).collect();
                let vals_b: Vec<f64> = readings_b[..n].iter().map(|r| r.value).collect();
                Self::pearson(&vals_a, &vals_b).unwrap_or(0.0)
            });

        // Compute current correlation from last window
        let n = readings_a.len().min(readings_b.len());
        let start = n.saturating_sub(window_size);
        let current_a: Vec<f64> = readings_a[start..n].iter().map(|r| r.value).collect();
        let current_b: Vec<f64> = readings_b[start..n].iter().map(|r| r.value).collect();
        let current_corr = Self::pearson(&current_a, &current_b).unwrap_or(0.0);

        let drop = (baseline - current_corr).abs();
        if drop > self.config.correlation_drop_threshold {
            let now = readings_a.last().unwrap().timestamp;
            Some(AnomalyEvent::new(
                AnomalyType::TrendShift,
                format!("{}_{}", sensor_a, sensor_b),
                now,
                self.classify_severity(drop * 3.0),
                drop,
                format!(
                    "Correlation breakdown between {} and {}: baseline={:.2}, current={:.2}",
                    sensor_a, sensor_b, baseline, current_corr
                ),
            ))
        } else {
            None
        }
    }

    /// Detect correlation anomalies for all sensor pairs.
    pub fn detect_all_correlation_anomalies(
        &self,
        db: &SensorDatabase,
        window_size: usize,
    ) -> Vec<AnomalyEvent> {
        let sensor_ids = db.sensor_ids();
        let mut anomalies = Vec::new();

        for i in 0..sensor_ids.len() {
            for j in (i + 1)..sensor_ids.len() {
                if let Some(anomaly) =
                    self.detect_correlation_anomaly(db, &sensor_ids[i], &sensor_ids[j], window_size)
                {
                    anomalies.push(anomaly);
                }
            }
        }

        anomalies
    }

    // -----------------------------------------------------------------------
    // Comprehensive detection
    // -----------------------------------------------------------------------

    /// Run all detection methods on a sensor and return all anomalies found.
    pub fn detect_all(&self, db: &SensorDatabase, sensor_id: &str) -> Vec<AnomalyEvent> {
        let mut all = Vec::new();
        all.extend(self.detect_zscore_sensor(db, sensor_id));
        all.extend(self.detect_ma_deviation_sensor(db, sensor_id));
        all.extend(self.detect_iqr_sensor(db, sensor_id));
        all.extend(self.detect_roc_sensor(db, sensor_id));
        // Deduplicate by timestamp (keep highest severity)
        all.sort_by(|a, b| {
            a.start_time
                .cmp(&b.start_time)
                .then_with(|| b.severity.cmp(&a.severity))
        });
        all.dedup_by(|a, b| a.start_time == b.start_time && a.sensor_id == b.sensor_id);
        all
    }

    /// Run detection on all sensors in the database.
    pub fn detect_all_sensors(&self, db: &SensorDatabase) -> Vec<AnomalyEvent> {
        let mut all = Vec::new();
        for sensor_id in db.sensor_ids() {
            all.extend(self.detect_all(db, &sensor_id));
        }
        all
    }

    // -----------------------------------------------------------------------
    // Severity
    // -----------------------------------------------------------------------

    /// Classify anomaly severity based on magnitude.
    pub fn classify_severity(&self, magnitude: f64) -> Severity {
        match magnitude {
            m if m > 8.0 => Severity::Emergency,
            m if m > 5.0 => Severity::Critical,
            m if m > 3.0 => Severity::Warning,
            _ => Severity::Info,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() < 3 || x.len() != y.len() {
            return None;
        }
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0_f64;
        let mut vx = 0.0_f64;
        let mut vy = 0.0_f64;
        for i in 0..x.len() {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        let denom = (vx * vy).sqrt();
        if denom.abs() < 1e-15 {
            None
        } else {
            Some(cov / denom)
        }
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SensorReading;
    use chrono::Duration;

    fn make_db_with_spike() -> (SensorDatabase, String) {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        let sid = "VIB-001".to_string();
        for i in 0..100 {
            let val = if i == 50 { 100.0 } else { 10.0 + rand::random::<f64>() };
            db.add_reading(SensorReading::new(&sid, SensorType::Vibration, base + Duration::seconds(i), val));
        }
        (db, sid)
    }

    fn make_db_with_drift() -> (SensorDatabase, String) {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        let sid = "TMP-001".to_string();
        for i in 0..100 {
            let val = 60.0 + i as f64 * 1.5 + rand::random::<f64>() * 0.5;
            db.add_reading(SensorReading::new(&sid, SensorType::Temperature, base + Duration::seconds(i), val));
        }
        (db, sid)
    }

    fn make_db_with_step() -> (SensorDatabase, String) {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        let sid = "PRS-001".to_string();
        for i in 0..100 {
            let val = if i >= 50 { 250.0 } else { 200.0 };
            db.add_reading(SensorReading::new(&sid, SensorType::Pressure, base + Duration::seconds(i), val));
        }
        (db, sid)
    }

    #[test]
    fn test_new_detector() {
        let d = AnomalyDetector::new();
        assert_eq!(d.active_count(), 0);
        assert!(d.active_anomalies().is_empty());
    }

    #[test]
    fn test_compute_baseline() {
        let mut d = AnomalyDetector::new();
        d.compute_baseline("VIB", &[10.0, 11.0, 12.0, 10.0, 11.0]);
        let (mean, std) = d.get_baseline("VIB").unwrap();
        assert!((mean - 10.8).abs() < 0.01);
        assert!(std > 0.0);
    }

    #[test]
    fn test_detect_zscore_spike() {
        let d = AnomalyDetector::new();
        let values: Vec<f64> = (0..50).map(|i| if i == 25 { 100.0 } else { 10.0 }).collect();
        let anomalies = d.detect_zscore(&values);
        assert!(!anomalies.is_empty());
        assert!(anomalies.contains(&25));
    }

    #[test]
    fn test_detect_zscore_no_anomaly() {
        let d = AnomalyDetector::new();
        let values: Vec<f64> = (0..100).map(|_| 10.0 + rand::random::<f64>() * 0.5).collect();
        let anomalies = d.detect_zscore(&values);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_detect_zscore_short() {
        let d = AnomalyDetector::new();
        assert!(d.detect_zscore(&[1.0, 2.0]).is_empty());
    }

    #[test]
    fn test_detect_zscore_constant() {
        let d = AnomalyDetector::new();
        let values = vec![5.0; 20];
        assert!(d.detect_zscore(&values).is_empty());
    }

    #[test]
    fn test_detect_zscore_sensor() {
        let d = AnomalyDetector::new();
        let (db, sid) = make_db_with_spike();
        let events = d.detect_zscore_sensor(&db, &sid);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_detect_ma_deviation_drift() {
        let config = AnomalyConfig {
            ma_deviation_threshold: 1.5,
            ..AnomalyConfig::default()
        };
        let d = AnomalyDetector::with_config(config);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 1.0).collect();
        let anomalies = d.detect_ma_deviation(&values);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_detect_ma_deviation_stable() {
        let d = AnomalyDetector::new();
        let values: Vec<f64> = (0..50).map(|_| 10.0 + rand::random::<f64>() * 0.5).collect();
        let anomalies = d.detect_ma_deviation(&values);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_detect_ma_deviation_short() {
        let d = AnomalyDetector::new();
        let values = vec![1.0, 2.0, 3.0];
        assert!(d.detect_ma_deviation(&values).is_empty());
    }

    #[test]
    fn test_detect_ma_deviation_sensor() {
        let config = AnomalyConfig {
            ma_deviation_threshold: 1.5,
            ..AnomalyConfig::default()
        };
        let d = AnomalyDetector::with_config(config);
        let (db, sid) = make_db_with_drift();
        let events = d.detect_ma_deviation_sensor(&db, &sid);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_detect_iqr_outlier() {
        let d = AnomalyDetector::new();
        let mut values: Vec<f64> = (0..50).map(|_| 10.0 + rand::random::<f64>() * 2.0).collect();
        values.push(1000.0);
        let anomalies = d.detect_iqr(&values);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_detect_iqr_no_outlier() {
        let d = AnomalyDetector::new();
        let values: Vec<f64> = (0..50).map(|_| 10.0 + rand::random::<f64>() * 2.0).collect();
        let anomalies = d.detect_iqr(&values);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_detect_iqr_short() {
        let d = AnomalyDetector::new();
        assert!(d.detect_iqr(&[1.0, 2.0, 3.0]).is_empty());
    }

    #[test]
    fn test_detect_iqr_constant() {
        let d = AnomalyDetector::new();
        assert!(d.detect_iqr(&[5.0; 10]).is_empty());
    }

    #[test]
    fn test_detect_iqr_sensor() {
        let d = AnomalyDetector::new();
        let (db, sid) = make_db_with_spike();
        let events = d.detect_iqr_sensor(&db, &sid);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_detect_roc_step() {
        let d = AnomalyDetector::new();
        let values: Vec<f64> = (0..100).map(|i| if i >= 50 { 250.0 } else { 200.0 }).collect();
        let anomalies = d.detect_rate_of_change(&values);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_detect_roc_smooth() {
        let d = AnomalyDetector::new();
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.05).sin()).collect();
        let anomalies = d.detect_rate_of_change(&values);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_detect_roc_short() {
        let d = AnomalyDetector::new();
        assert!(d.detect_rate_of_change(&[1.0, 2.0, 3.0]).is_empty());
    }

    #[test]
    fn test_detect_roc_sensor() {
        let d = AnomalyDetector::new();
        let (db, sid) = make_db_with_step();
        let events = d.detect_roc_sensor(&db, &sid);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_track_anomaly() {
        let mut d = AnomalyDetector::new();
        let evt = AnomalyEvent::new(AnomalyType::Spike, "VIB-001", Utc::now(), Severity::Warning, 3.5, "Test");
        d.track_anomaly(evt);
        assert_eq!(d.active_count(), 1);
    }

    #[test]
    fn test_resolve_anomaly() {
        let mut d = AnomalyDetector::new();
        d.track_anomaly(AnomalyEvent::new(AnomalyType::Drift, "TMP-001", Utc::now(), Severity::Critical, 2.0, "Drift"));
        assert!(d.has_active_anomaly("TMP-001"));
        let resolved = d.resolve_anomaly("TMP-001", Utc::now() + Duration::minutes(5));
        assert!(resolved);
        assert!(!d.has_active_anomaly("TMP-001"));
        assert_eq!(d.history().len(), 1);
    }

    #[test]
    fn test_resolve_no_anomaly() {
        let mut d = AnomalyDetector::new();
        assert!(!d.resolve_anomaly("NONE", Utc::now()));
    }

    #[test]
    fn test_has_active_anomaly_false() {
        let d = AnomalyDetector::new();
        assert!(!d.has_active_anomaly("VIB-001"));
    }

    #[test]
    fn test_active_anomalies_list() {
        let mut d = AnomalyDetector::new();
        d.track_anomaly(AnomalyEvent::new(AnomalyType::Spike, "A", Utc::now(), Severity::Warning, 1.0, ""));
        d.track_anomaly(AnomalyEvent::new(AnomalyType::Drift, "B", Utc::now(), Severity::Critical, 2.0, ""));
        assert_eq!(d.active_anomalies().len(), 2);
    }

    #[test]
    fn test_detect_correlation_anomaly() {
        let mut d = AnomalyDetector::new();
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        // First half: correlated
        for i in 0..50 {
            let v = 10.0 * (i as f64 * 0.1).sin();
            db.add_reading(SensorReading::new("A", SensorType::Vibration, base + Duration::seconds(i), v));
            db.add_reading(SensorReading::new("B", SensorType::Acoustic, base + Duration::seconds(i), v * 2.0));
        }
        // Second half: decorrelated
        for i in 50..100 {
            db.add_reading(SensorReading::new("A", SensorType::Vibration, base + Duration::seconds(i), rand::random::<f64>() * 10.0));
            db.add_reading(SensorReading::new("B", SensorType::Acoustic, base + Duration::seconds(i), rand::random::<f64>() * 20.0));
        }
        d.compute_baseline_correlation("A", "B", 0.95);
        let anomaly = d.detect_correlation_anomaly(&db, "A", "B", 20);
        assert!(anomaly.is_some());
    }

    #[test]
    fn test_detect_all_correlation_anomalies_empty() {
        let d = AnomalyDetector::new();
        let db = SensorDatabase::new();
        assert!(d.detect_all_correlation_anomalies(&db, 10).is_empty());
    }

    #[test]
    fn test_detect_all_methods() {
        let d = AnomalyDetector::new();
        let (db, sid) = make_db_with_spike();
        let all = d.detect_all(&db, &sid);
        assert!(!all.is_empty());
    }

    #[test]
    fn test_detect_all_sensors() {
        let d = AnomalyDetector::new();
        let (db, _) = make_db_with_spike();
        let all = d.detect_all_sensors(&db);
        assert!(!all.is_empty());
    }

    #[test]
    fn test_classify_severity() {
        let d = AnomalyDetector::new();
        assert_eq!(d.classify_severity(1.0), Severity::Info);
        assert_eq!(d.classify_severity(4.0), Severity::Warning);
        assert_eq!(d.classify_severity(6.0), Severity::Critical);
        assert_eq!(d.classify_severity(9.0), Severity::Emergency);
    }

    #[test]
    fn test_with_config() {
        let config = AnomalyConfig {
            zscore_threshold: 2.0,
            ma_window: 10,
            ma_deviation_threshold: 2.0,
            iqr_multiplier: 1.0,
            roc_threshold: 2.0,
            correlation_drop_threshold: 0.2,
        };
        let d = AnomalyDetector::with_config(config);
        let values: Vec<f64> = (0..50).map(|i| if i == 25 { 20.0 } else { 10.0 }).collect();
        let anomalies = d.detect_zscore(&values);
        assert!(!anomalies.is_empty());
    }
}
