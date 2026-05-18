//! # Sensor Data Management
//!
//! Storage, quality validation, fusion, resampling, and mock data
//! generation for mining sensor data.

use crate::types::{QualityFlag, SensorReading, SensorType};
use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use std::collections::HashMap;

/// In-memory sensor database for mining operations.
#[derive(Debug, Clone)]
pub struct SensorDatabase {
    /// sensor_id -> list of readings (sorted by timestamp).
    readings: HashMap<String, Vec<SensorReading>>,
    /// sensor_id -> sensor type.
    sensor_types: HashMap<String, SensorType>,
    /// Maximum age (seconds) before data is considered stale.
    staleness_threshold_secs: i64,
    /// Noise threshold for spike detection (std deviations).
    noise_std_threshold: f64,
}

impl SensorDatabase {
    /// Create an empty sensor database.
    pub fn new() -> Self {
        SensorDatabase {
            readings: HashMap::new(),
            sensor_types: HashMap::new(),
            staleness_threshold_secs: 300, // 5 minutes
            noise_std_threshold: 3.0,
        }
    }

    /// Configure staleness threshold.
    pub fn with_staleness_threshold(mut self, secs: i64) -> Self {
        self.staleness_threshold_secs = secs;
        self
    }

    /// Configure noise threshold (in standard deviations).
    pub fn with_noise_threshold(mut self, threshold: f64) -> Self {
        self.noise_std_threshold = threshold;
        self
    }

    // -----------------------------------------------------------------------
    // CRUD
    // -----------------------------------------------------------------------

    /// Add a reading to the database.
    pub fn add_reading(&mut self, reading: SensorReading) {
        let sensor_id = reading.sensor_id.clone();
        self.sensor_types.insert(sensor_id.clone(), reading.sensor_type);
        self.readings
            .entry(sensor_id.clone())
            .or_default()
            .push(reading);
        // Keep sorted
        let entries = self.readings.get_mut(&sensor_id).unwrap();
        entries.sort_by_key(|r| r.timestamp);
    }

    /// Add multiple readings at once.
    pub fn add_readings(&mut self, readings: Vec<SensorReading>) {
        for r in readings {
            self.add_reading(r);
        }
    }

    /// Get all readings for a sensor.
    pub fn get_readings(&self, sensor_id: &str) -> Option<&[SensorReading]> {
        self.readings.get(sensor_id).map(|v| v.as_slice())
    }

    /// Get readings within a time range.
    pub fn get_range(
        &self,
        sensor_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<SensorReading> {
        self.readings
            .get(sensor_id)
            .map(|readings| {
                readings
                    .iter()
                    .filter(|r| r.timestamp >= start && r.timestamp <= end)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the most recent reading for each sensor.
    pub fn latest_readings(&self) -> HashMap<String, SensorReading> {
        let mut result = HashMap::new();
        for (sensor_id, readings) in &self.readings {
            if let Some(latest) = readings.last() {
                result.insert(sensor_id.clone(), latest.clone());
            }
        }
        result
    }

    /// Get the latest reading for a specific sensor.
    pub fn latest_reading(&self, sensor_id: &str) -> Option<&SensorReading> {
        self.readings.get(sensor_id).and_then(|r| r.last())
    }

    /// Get values for a sensor as a time series.
    pub fn values(&self, sensor_id: &str) -> Vec<f64> {
        self.readings
            .get(sensor_id)
            .map(|r| r.iter().map(|rd| rd.value).collect())
            .unwrap_or_default()
    }

    /// Get timestamps for a sensor as a time series.
    pub fn timestamps(&self, sensor_id: &str) -> Vec<DateTime<Utc>> {
        self.readings
            .get(sensor_id)
            .map(|r| r.iter().map(|rd| rd.timestamp).collect())
            .unwrap_or_default()
    }

    /// Get all sensor IDs in the database.
    pub fn sensor_ids(&self) -> Vec<String> {
        self.readings.keys().cloned().collect()
    }

    /// Get the sensor type for a given sensor ID.
    pub fn sensor_type(&self, sensor_id: &str) -> Option<SensorType> {
        self.sensor_types.get(sensor_id).copied()
    }

    /// Total number of readings across all sensors.
    pub fn total_readings(&self) -> usize {
        self.readings.values().map(|v| v.len()).sum()
    }

    /// Number of sensors in the database.
    pub fn sensor_count(&self) -> usize {
        self.readings.len()
    }

    // -----------------------------------------------------------------------
    // Data quality
    // -----------------------------------------------------------------------

    /// Check range validity of a reading against its sensor type.
    pub fn validate_range(&self, reading: &SensorReading) -> bool {
        let (min, max) = reading.sensor_type.typical_range();
        reading.value >= min && reading.value <= max
    }

    /// Find stale sensors (no readings within the staleness threshold).
    pub fn find_stale_sensors(&self, now: DateTime<Utc>) -> Vec<String> {
        let threshold = Duration::seconds(self.staleness_threshold_secs);
        self.readings
            .iter()
            .filter_map(|(id, readings)| {
                readings
                    .last()
                    .filter(|r| now - r.timestamp > threshold)
                    .map(|_| id.clone())
            })
            .collect()
    }

    /// Filter out readings marked as Bad or Missing quality.
    pub fn filter_good_quality(&self, sensor_id: &str) -> Vec<SensorReading> {
        self.readings
            .get(sensor_id)
            .map(|r| {
                r.iter()
                    .filter(|rd| rd.quality_flag == QualityFlag::Good)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Detect noise spikes using a simple standard deviation filter.
    /// Returns indices of readings that exceed the noise threshold.
    pub fn detect_noise(&self, sensor_id: &str) -> Vec<usize> {
        let readings = match self.readings.get(sensor_id) {
            Some(r) => r,
            None => return Vec::new(),
        };
        if readings.len() < 3 {
            return Vec::new();
        }

        let values: Vec<f64> = readings.iter().map(|r| r.value).collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        if std_dev.abs() < 1e-15 {
            return Vec::new();
        }

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| ((v - mean).abs() / std_dev) > self.noise_std_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Sensor fusion
    // -----------------------------------------------------------------------

    /// Combine multiple sensor types into a composite health score.
    /// Weights: vibration=0.3, temperature=0.25, pressure=0.15, acoustic=0.15, power=0.15
    pub fn fuse_sensor_scores(&self, sensor_ids: &[&str]) -> Option<f64> {
        if sensor_ids.is_empty() {
            return None;
        }
        let weights: HashMap<SensorType, f64> = [
            (SensorType::Vibration, 0.30),
            (SensorType::Temperature, 0.25),
            (SensorType::Pressure, 0.15),
            (SensorType::Acoustic, 0.15),
            (SensorType::Power, 0.15),
        ]
        .into_iter()
        .collect();

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for id in sensor_ids {
            let latest = match self.latest_reading(id) {
                Some(r) => r,
                None => continue,
            };
            let w = weights.get(&latest.sensor_type).copied().unwrap_or(0.05);
            let (min, max) = latest.sensor_type.typical_range();
            let range = max - min;
            if range.abs() < 1e-15 {
                continue;
            }
            // Normalize to [0, 1]: lower value = better
            let normalized = 1.0 - ((latest.value - min) / range).clamp(0.0, 1.0);
            weighted_sum += normalized * w;
            total_weight += w;
        }

        if total_weight.abs() < 1e-15 {
            None
        } else {
            Some(weighted_sum / total_weight * 100.0)
        }
    }

    /// Cross-correlate two sensors' value series using Pearson correlation.
    pub fn correlate_sensors(&self, sensor_a: &str, sensor_b: &str) -> Option<f64> {
        let vals_a = self.values(sensor_a);
        let vals_b = self.values(sensor_b);
        if vals_a.len() < 3 || vals_b.len() < 3 {
            return None;
        }

        let len = vals_a.len().min(vals_b.len());
        let a = &vals_a[..len];
        let b = &vals_b[..len];

        let mean_a = a.iter().sum::<f64>() / len as f64;
        let mean_b = b.iter().sum::<f64>() / len as f64;

        let mut cov = 0.0_f64;
        let mut var_a = 0.0_f64;
        let mut var_b = 0.0_f64;
        for i in 0..len {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom.abs() < 1e-15 {
            Some(0.0)
        } else {
            Some(cov / denom)
        }
    }

    // -----------------------------------------------------------------------
    // Resampling & interpolation
    // -----------------------------------------------------------------------

    /// Resample readings to a uniform time grid using linear interpolation.
    /// `interval_secs` is the desired interval between samples.
    pub fn resample(
        &self,
        sensor_id: &str,
        interval_secs: i64,
    ) -> Option<Vec<SensorReading>> {
        let readings = self.readings.get(sensor_id)?;
        if readings.len() < 2 {
            return None;
        }

        let start = readings.first().unwrap().timestamp;
        let end = readings.last().unwrap().timestamp;
        let sensor_type = readings[0].sensor_type;

        let mut result = Vec::new();
        let mut t = start;
        let mut idx = 0usize;

        while t <= end {
            // Advance idx to the reading just before or at t
            while idx < readings.len() - 1 && readings[idx + 1].timestamp <= t {
                idx += 1;
            }

            let interpolated = if idx >= readings.len() - 1 {
                readings.last().unwrap().value
            } else if idx == 0 && readings[0].timestamp > t {
                readings[0].value
            } else {
                // Linear interpolation
                let r0 = &readings[idx];
                let r1 = &readings[idx + 1];
                let dt = (r1.timestamp - r0.timestamp).num_milliseconds() as f64;
                let elapsed = (t - r0.timestamp).num_milliseconds() as f64;
                if dt.abs() < 1e-10 {
                    r0.value
                } else {
                    let frac = elapsed / dt;
                    r0.value + frac * (r1.value - r0.value)
                }
            };

            result.push(SensorReading::new(
                format!("{}_resampled", sensor_id),
                sensor_type,
                t,
                interpolated,
            ));

            t = t + Duration::seconds(interval_secs);
        }

        Some(result)
    }

    /// Fill missing values using linear interpolation on the values array.
    pub fn interpolate_values(values: &[Option<f64>]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        // Find first and last non-None
        let first_idx = values.iter().position(|v| v.is_some()).unwrap_or(0);
        let last_idx = values
            .iter()
            .rposition(|v| v.is_some())
            .unwrap_or(values.len() - 1);

        let mut result = Vec::with_capacity(values.len());
        for (i, val) in values.iter().enumerate() {
            match val {
                Some(v) => result.push(*v),
                None => {
                    if i <= first_idx {
                        result.push(values[first_idx].unwrap());
                    } else if i >= last_idx {
                        result.push(values[last_idx].unwrap());
                    } else {
                        // Find surrounding known values
                        let prev_idx = (0..i).rev().find(|&j| values[j].is_some()).unwrap();
                        let next_idx = (i + 1..values.len()).find(|&j| values[j].is_some()).unwrap();
                        let prev_val = values[prev_idx].unwrap();
                        let next_val = values[next_idx].unwrap();
                        let gap = (next_idx - prev_idx) as f64;
                        let pos = (i - prev_idx) as f64;
                        result.push(prev_val + pos / gap * (next_val - prev_val));
                    }
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Mock data generation
    // -----------------------------------------------------------------------

    /// Generate mock sensor readings for a given sensor type.
    /// `count` readings at `interval_secs` spacing, with realistic patterns.
    pub fn generate_mock_readings(
        sensor_id: &str,
        sensor_type: SensorType,
        count: usize,
        interval_secs: i64,
    ) -> Vec<SensorReading> {
        let (base_val, amplitude) = match sensor_type {
            SensorType::Vibration => (5.0, 2.0),
            SensorType::Temperature => (65.0, 15.0),
            SensorType::Pressure => (200.0, 50.0),
            SensorType::Acoustic => (85.0, 10.0),
            SensorType::Electromagnetic => (50.0, 20.0),
            SensorType::Chemical => (30.0, 10.0),
            SensorType::FlowRate => (300.0, 50.0),
            SensorType::Power => (500.0, 100.0),
        };

        let base_time = Utc::now() - Duration::seconds(count as i64 * interval_secs);
        let mut rng = rand::thread_rng();

        (0..count)
            .map(|i| {
                let t = base_time + Duration::seconds(i as i64 * interval_secs);
                // Add periodic pattern + random noise
                let periodic = amplitude * 0.3 * (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin();
                let noise = rng.gen_range(-amplitude * 0.2..amplitude * 0.2);
                let value = base_val + periodic + noise;
                SensorReading::new(sensor_id, sensor_type, t, value)
            })
            .collect()
    }

    /// Generate mock readings with an injected anomaly (spike) at a given index.
    pub fn generate_mock_with_spike(
        sensor_id: &str,
        sensor_type: SensorType,
        count: usize,
        interval_secs: i64,
        spike_index: usize,
        spike_magnitude: f64,
    ) -> Vec<SensorReading> {
        let mut readings = Self::generate_mock_readings(sensor_id, sensor_type, count, interval_secs);
        if spike_index < readings.len() {
            readings[spike_index].value += spike_magnitude;
        }
        readings
    }

    /// Generate mock readings with a drift trend.
    pub fn generate_mock_with_drift(
        sensor_id: &str,
        sensor_type: SensorType,
        count: usize,
        interval_secs: i64,
        drift_rate: f64,
    ) -> Vec<SensorReading> {
        let base_time = Utc::now() - Duration::seconds(count as i64 * interval_secs);
        let (base_val, amplitude) = match sensor_type {
            SensorType::Temperature => (65.0, 15.0),
            _ => (50.0, 10.0),
        };
        let mut rng = rand::thread_rng();

        (0..count)
            .map(|i| {
                let t = base_time + Duration::seconds(i as i64 * interval_secs);
                let drift = drift_rate * i as f64;
                let noise = rng.gen_range(-amplitude * 0.15..amplitude * 0.15);
                let value = base_val + drift + noise;
                SensorReading::new(sensor_id, sensor_type, t, value)
            })
            .collect()
    }

    /// Generate mock readings with step change.
    pub fn generate_mock_with_step(
        sensor_id: &str,
        sensor_type: SensorType,
        count: usize,
        interval_secs: i64,
        step_index: usize,
        step_size: f64,
    ) -> Vec<SensorReading> {
        let mut readings = Self::generate_mock_readings(sensor_id, sensor_type, count, interval_secs);
        for r in readings.iter_mut().skip(step_index) {
            r.value += step_size;
        }
        readings
    }

    /// Generate mock data for all sensor types.
    pub fn generate_all_sensor_mocks(
        count: usize,
        interval_secs: i64,
    ) -> HashMap<String, Vec<SensorReading>> {
        let mut result = HashMap::new();
        for st in SensorType::all() {
            let _id = format!("{:?}", st).to_uppercase();
            let prefix = match st {
                SensorType::Vibration => "VIB",
                SensorType::Temperature => "TMP",
                SensorType::Pressure => "PRS",
                SensorType::Acoustic => "ACO",
                SensorType::Electromagnetic => "EMF",
                SensorType::Chemical => "CHM",
                SensorType::FlowRate => "FLW",
                SensorType::Power => "PWR",
            };
            let sensor_id = format!("{}-001", prefix);
            result.insert(sensor_id.clone(), Self::generate_mock_readings(&sensor_id, *st, count, interval_secs));
        }
        result
    }
}

impl Default for SensorDatabase {
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

    #[test]
    fn test_new_database() {
        let db = SensorDatabase::new();
        assert_eq!(db.total_readings(), 0);
        assert_eq!(db.sensor_count(), 0);
    }

    #[test]
    fn test_add_and_get_readings() {
        let mut db = SensorDatabase::new();
        let now = Utc::now();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now, 5.0));
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now + Duration::seconds(1), 6.0));

        let readings = db.get_readings("VIB-001").unwrap();
        assert_eq!(readings.len(), 2);
    }

    #[test]
    fn test_add_multiple_readings() {
        let mut db = SensorDatabase::new();
        let now = Utc::now();
        let readings: Vec<SensorReading> = (0..5)
            .map(|i| SensorReading::new("TMP-001", SensorType::Temperature, now + Duration::seconds(i), 60.0 + i as f64))
            .collect();
        db.add_readings(readings);
        assert_eq!(db.total_readings(), 5);
    }

    #[test]
    fn test_get_range() {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        for i in 0..10 {
            db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, base + Duration::seconds(i * 10), i as f64));
        }
        let range = db.get_range("VIB-001", base + Duration::seconds(20), base + Duration::seconds(50));
        assert_eq!(range.len(), 4);
    }

    #[test]
    fn test_get_range_empty() {
        let db = SensorDatabase::new();
        assert!(db.get_range("NONE", Utc::now(), Utc::now()).is_empty());
    }

    #[test]
    fn test_latest_readings() {
        let mut db = SensorDatabase::new();
        let now = Utc::now();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now, 5.0));
        db.add_reading(SensorReading::new("TMP-001", SensorType::Temperature, now + Duration::seconds(1), 70.0));
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now + Duration::seconds(2), 6.0));

        let latest = db.latest_readings();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest["VIB-001"].value, 6.0);
    }

    #[test]
    fn test_latest_reading() {
        let mut db = SensorDatabase::new();
        let now = Utc::now();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now, 1.0));
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now + Duration::seconds(1), 2.0));
        assert_eq!(db.latest_reading("VIB-001").unwrap().value, 2.0);
    }

    #[test]
    fn test_values_and_timestamps() {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, base, 1.0));
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, base + Duration::seconds(1), 2.0));
        assert_eq!(db.values("VIB-001"), vec![1.0, 2.0]);
        assert_eq!(db.timestamps("VIB-001").len(), 2);
    }

    #[test]
    fn test_sensor_ids() {
        let mut db = SensorDatabase::new();
        db.add_reading(SensorReading::new("A", SensorType::Vibration, Utc::now(), 1.0));
        db.add_reading(SensorReading::new("B", SensorType::Temperature, Utc::now(), 2.0));
        let ids = db.sensor_ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_sensor_type_lookup() {
        let mut db = SensorDatabase::new();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 1.0));
        assert_eq!(db.sensor_type("VIB-001"), Some(SensorType::Vibration));
    }

    #[test]
    fn test_validate_range() {
        let db = SensorDatabase::new();
        let good = SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 5.0);
        assert!(db.validate_range(&good));
        let bad = SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 999.0);
        assert!(!db.validate_range(&bad));
    }

    #[test]
    fn test_find_stale_sensors() {
        let mut db = SensorDatabase::new().with_staleness_threshold(10);
        let old = Utc::now() - Duration::seconds(60);
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, old, 5.0));
        let stale = db.find_stale_sensors(Utc::now());
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], "VIB-001");
    }

    #[test]
    fn test_filter_good_quality() {
        let mut db = SensorDatabase::new();
        let now = Utc::now();
        db.add_reading(SensorReading::with_quality("VIB-001", SensorType::Vibration, now, 1.0, "mm/s", QualityFlag::Good));
        db.add_reading(SensorReading::with_quality("VIB-001", SensorType::Vibration, now + Duration::seconds(1), 2.0, "mm/s", QualityFlag::Bad));
        db.add_reading(SensorReading::with_quality("VIB-001", SensorType::Vibration, now + Duration::seconds(2), 3.0, "mm/s", QualityFlag::Suspect));
        let good = db.filter_good_quality("VIB-001");
        assert_eq!(good.len(), 1);
    }

    #[test]
    fn test_detect_noise() {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        for i in 0..50 {
            let val = 10.0 + (i as f64 * 0.01).sin();
            db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, base + Duration::seconds(i), val));
        }
        // Inject a spike at index 25
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, base + Duration::seconds(25), 100.0));
        let noise_indices = db.detect_noise("VIB-001");
        assert!(!noise_indices.is_empty());
    }

    #[test]
    fn test_fuse_sensor_scores() {
        let mut db = SensorDatabase::new();
        let now = Utc::now();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, now, 5.0));
        db.add_reading(SensorReading::new("TMP-001", SensorType::Temperature, now, 65.0));
        let score = db.fuse_sensor_scores(&["VIB-001", "TMP-001"]);
        assert!(score.is_some());
        assert!(score.unwrap() > 0.0 && score.unwrap() <= 100.0);
    }

    #[test]
    fn test_fuse_empty() {
        let db = SensorDatabase::new();
        assert!(db.fuse_sensor_scores(&[]).is_none());
    }

    #[test]
    fn test_correlate_sensors() {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        for i in 0..50 {
            let val = 10.0 * (i as f64 * 0.2).sin();
            db.add_reading(SensorReading::new("A", SensorType::Vibration, base + Duration::seconds(i), val));
            db.add_reading(SensorReading::new("B", SensorType::Acoustic, base + Duration::seconds(i), val * 2.0));
        }
        let corr = db.correlate_sensors("A", "B");
        assert!(corr.is_some());
        assert!(corr.unwrap().abs() > 0.9);
    }

    #[test]
    fn test_correlate_insufficient() {
        let db = SensorDatabase::new();
        assert!(db.correlate_sensors("A", "B").is_none());
    }

    #[test]
    fn test_resample() {
        let mut db = SensorDatabase::new();
        let base = Utc::now();
        for i in 0..20 {
            db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, base + Duration::seconds(i), i as f64));
        }
        let resampled = db.resample("VIB-001", 5).unwrap();
        assert!(!resampled.is_empty());
        // Each sample should be 5 seconds apart
        for i in 1..resampled.len() {
            let dt = resampled[i].timestamp - resampled[i - 1].timestamp;
            assert_eq!(dt.num_seconds(), 5);
        }
    }

    #[test]
    fn test_resample_insufficient() {
        let mut db = SensorDatabase::new();
        db.add_reading(SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 1.0));
        assert!(db.resample("VIB-001", 1).is_none());
    }

    #[test]
    fn test_interpolate_values() {
        let values = vec![Some(1.0), None, None, Some(4.0), None, Some(6.0)];
        let interpolated = SensorDatabase::interpolate_values(&values);
        assert_eq!(interpolated.len(), 6);
        assert_eq!(interpolated[0], 1.0);
        assert!((interpolated[1] - 2.0).abs() < 1e-10);
        assert_eq!(interpolated[3], 4.0);
        assert_eq!(interpolated[5], 6.0);
    }

    #[test]
    fn test_interpolate_empty() {
        assert!(SensorDatabase::interpolate_values(&[]).is_empty());
    }

    #[test]
    fn test_generate_mock_readings() {
        let readings = SensorDatabase::generate_mock_readings("VIB-001", SensorType::Vibration, 100, 10);
        assert_eq!(readings.len(), 100);
        assert_eq!(readings[0].sensor_id, "VIB-001");
        assert_eq!(readings[0].sensor_type, SensorType::Vibration);
        // Check that timestamps are evenly spaced
        for i in 1..readings.len() {
            let dt = readings[i].timestamp - readings[i - 1].timestamp;
            assert_eq!(dt.num_seconds(), 10);
        }
    }

    #[test]
    fn test_generate_mock_with_spike() {
        let readings =
            SensorDatabase::generate_mock_with_spike("VIB-001", SensorType::Vibration, 50, 1, 25, 50.0);
        assert!(readings[25].value > readings[24].value + 40.0);
    }

    #[test]
    fn test_generate_mock_with_drift() {
        let readings = SensorDatabase::generate_mock_with_drift("TMP-001", SensorType::Temperature, 100, 1, 0.5);
        let first_avg = readings[..10].iter().map(|r| r.value).sum::<f64>() / 10.0;
        let last_avg = readings[90..].iter().map(|r| r.value).sum::<f64>() / 10.0;
        assert!(last_avg > first_avg + 30.0);
    }

    #[test]
    fn test_generate_mock_with_step() {
        let readings =
            SensorDatabase::generate_mock_with_step("VIB-001", SensorType::Vibration, 50, 1, 25, 20.0);
        let pre_avg = readings[..25].iter().map(|r| r.value).sum::<f64>() / 25.0;
        let post_avg = readings[25..].iter().map(|r| r.value).sum::<f64>() / 25.0;
        assert!((post_avg - pre_avg - 20.0).abs() < 5.0);
    }

    #[test]
    fn test_generate_all_sensor_mocks() {
        let all = SensorDatabase::generate_all_sensor_mocks(50, 10);
        assert_eq!(all.len(), 8);
        for readings in all.values() {
            assert_eq!(readings.len(), 50);
        }
    }
}
