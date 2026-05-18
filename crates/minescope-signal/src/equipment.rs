//! # Equipment Health Monitoring
//!
//! Vibration baseline computation, temperature trending, operating hour
//! tracking, maintenance scheduling, composite health scoring, and
//! predictive maintenance indicators for mining equipment.

use crate::sensors::SensorDatabase;
use crate::types::{EquipmentHealth, SensorReading, SensorType};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Equipment state
// ---------------------------------------------------------------------------

/// Baseline statistics for a sensor on a piece of equipment.
#[derive(Debug, Clone, Copy)]
pub struct SensorBaseline {
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Number of samples used to compute the baseline.
    pub sample_count: usize,
}

impl SensorBaseline {
    /// Create a new baseline.
    pub fn new(mean: f64, std_dev: f64, sample_count: usize) -> Self {
        SensorBaseline {
            mean,
            std_dev,
            sample_count,
        }
    }
}

/// Temperature trend result.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureTrend {
    /// Slope of the linear trend (°C per reading).
    pub slope: f64,
    /// Direction label.
    pub direction: TrendDirection,
}

/// Direction of a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Rising,
    Stable,
    Falling,
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrendDirection::Rising => write!(f, "Rising"),
            TrendDirection::Stable => write!(f, "Stable"),
            TrendDirection::Falling => write!(f, "Falling"),
        }
    }
}

/// Predictive maintenance indicator.
#[derive(Debug, Clone)]
pub struct MaintenanceIndicator {
    /// Equipment ID.
    pub equipment_id: String,
    /// Whether maintenance is recommended.
    pub maintenance_recommended: bool,
    /// Health score (0-100).
    pub health_score: f64,
    /// Estimated failure probability (0.0-1.0).
    pub failure_probability: f64,
    /// Recommended action.
    pub recommended_action: String,
}

// ---------------------------------------------------------------------------
// EquipmentMonitor
// ---------------------------------------------------------------------------

/// Equipment health monitoring engine.
#[derive(Debug, Clone)]
pub struct EquipmentMonitor {
    /// Vibration baselines per equipment: equipment_id -> (mean, std_dev).
    vibration_baselines: HashMap<String, SensorBaseline>,
    /// Temperature history per equipment.
    temperature_history: HashMap<String, Vec<f64>>,
    /// Operating hours per equipment.
    operating_hours: HashMap<String, f64>,
    /// Health score history per equipment.
    health_history: HashMap<String, Vec<f64>>,
    /// Maintenance threshold (health score below this triggers maintenance).
    maintenance_threshold: f64,
    /// Maintenance interval in operating hours.
    maintenance_interval_hours: f64,
}

impl EquipmentMonitor {
    /// Create a new equipment monitor with defaults.
    pub fn new() -> Self {
        EquipmentMonitor {
            vibration_baselines: HashMap::new(),
            temperature_history: HashMap::new(),
            operating_hours: HashMap::new(),
            health_history: HashMap::new(),
            maintenance_threshold: 40.0,
            maintenance_interval_hours: 8000.0,
        }
    }

    /// Configure maintenance threshold.
    pub fn with_maintenance_threshold(mut self, threshold: f64) -> Self {
        self.maintenance_threshold = threshold;
        self
    }

    /// Configure maintenance interval.
    pub fn with_maintenance_interval(mut self, hours: f64) -> Self {
        self.maintenance_interval_hours = hours;
        self
    }

    // -----------------------------------------------------------------------
    // Vibration baseline
    // -----------------------------------------------------------------------

    /// Compute vibration baseline (mean + std) from historical readings.
    pub fn compute_vibration_baseline(&mut self, equipment_id: &str, values: &[f64]) {
        if values.len() < 3 {
            return;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        self.vibration_baselines.insert(
            equipment_id.to_string(),
            SensorBaseline::new(mean, std_dev, values.len()),
        );
    }

    /// Compute vibration baseline directly from sensor readings.
    pub fn compute_vibration_baseline_from_readings(
        &mut self,
        equipment_id: &str,
        readings: &[SensorReading],
    ) {
        let values: Vec<f64> = readings.iter().map(|r| r.value).collect();
        self.compute_vibration_baseline(equipment_id, &values);
    }

    /// Get the vibration baseline for an equipment.
    pub fn vibration_baseline(&self, equipment_id: &str) -> Option<SensorBaseline> {
        self.vibration_baselines.get(equipment_id).copied()
    }

    /// Check if current vibration is abnormal relative to baseline.
    /// Returns the number of standard deviations from the mean.
    pub fn vibration_zscore(&self, equipment_id: &str, current_value: f64) -> Option<f64> {
        let baseline = self.vibration_baselines.get(equipment_id)?;
        if baseline.std_dev.abs() < 1e-15 {
            // If baseline has zero variance, any deviation is anomalous
            let diff = (current_value - baseline.mean).abs();
            return Some(if diff < 1e-10 { 0.0 } else { 10.0 });
        }
        Some((current_value - baseline.mean) / baseline.std_dev)
    }

    // -----------------------------------------------------------------------
    // Temperature trending
    // -----------------------------------------------------------------------

    /// Record a temperature reading for an equipment.
    pub fn record_temperature(&mut self, equipment_id: &str, temperature: f64) {
        self.temperature_history
            .entry(equipment_id.to_string())
            .or_default()
            .push(temperature);
    }

    /// Compute temperature trend using simple linear regression.
    pub fn temperature_trend(&self, equipment_id: &str) -> Option<TemperatureTrend> {
        let history = self.temperature_history.get(equipment_id)?;
        if history.len() < 4 {
            return None;
        }

        let n = history.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = history.iter().sum::<f64>() / n;

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        for (i, &y) in history.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator.abs() < 1e-15 {
            0.0
        } else {
            numerator / denominator
        };

        let direction = if slope.abs() < 0.01 {
            TrendDirection::Stable
        } else if slope > 0.0 {
            TrendDirection::Rising
        } else {
            TrendDirection::Falling
        };

        Some(TemperatureTrend { slope, direction })
    }

    /// Get the latest temperature for an equipment.
    pub fn latest_temperature(&self, equipment_id: &str) -> Option<f64> {
        self.temperature_history
            .get(equipment_id)?
            .last()
            .copied()
    }

    // -----------------------------------------------------------------------
    // Operating hours
    // -----------------------------------------------------------------------

    /// Set operating hours for an equipment.
    pub fn set_operating_hours(&mut self, equipment_id: &str, hours: f64) {
        self.operating_hours
            .insert(equipment_id.to_string(), hours);
    }

    /// Add operating hours to an equipment.
    pub fn add_operating_hours(&mut self, equipment_id: &str, additional_hours: f64) {
        let current = self
            .operating_hours
            .get(equipment_id)
            .copied()
            .unwrap_or(0.0);
        self.operating_hours
            .insert(equipment_id.to_string(), current + additional_hours);
    }

    /// Get operating hours for an equipment.
    pub fn operating_hours(&self, equipment_id: &str) -> f64 {
        self.operating_hours
            .get(equipment_id)
            .copied()
            .unwrap_or(0.0)
    }

    /// Check if maintenance is due based on operating hours.
    pub fn maintenance_due_hours(&self, equipment_id: &str) -> bool {
        let hours = self.operating_hours(equipment_id);
        hours >= self.maintenance_interval_hours
    }

    // -----------------------------------------------------------------------
    // Health score computation
    // -----------------------------------------------------------------------

    /// Compute a composite health score (0-100) for an equipment.
    ///
    /// Factors:
    /// - Vibration penalty: how far current vibration is from baseline (0-30 pts)
    /// - Temperature penalty: how far current temp is above baseline (0-30 pts)
    /// - Age penalty: based on operating hours (0-20 pts)
    /// - Base score: 100 minus penalties
    pub fn compute_health_score(
        &self,
        equipment_id: &str,
        current_vibration: f64,
        current_temperature: f64,
    ) -> f64 {
        // Vibration penalty (0-30 pts max)
        let vibration_penalty = match self.vibration_zscore(equipment_id, current_vibration) {
            Some(z) => {
                if z <= 1.0 {
                    0.0
                } else {
                    ((z - 1.0) * 10.0).min(30.0)
                }
            }
            None => 5.0, // unknown baseline, mild penalty
        };

        // Temperature penalty (0-30 pts max)
        let baseline_temp: f64 = self
            .temperature_history
            .get(equipment_id)
            .map(|h| h.iter().sum::<f64>() / h.len().max(1) as f64)
            .unwrap_or(60.0);
        let temp_diff = (current_temperature - baseline_temp).max(0.0);
        let temp_penalty = (temp_diff * 0.5).min(30.0);

        // Age penalty (0-20 pts max)
        let hours = self.operating_hours(equipment_id);
        let age_penalty = (hours / self.maintenance_interval_hours * 20.0).min(20.0);

        let score = 100.0 - vibration_penalty - temp_penalty - age_penalty;
        score.clamp(0.0, 100.0)
    }

    /// Build an `EquipmentHealth` struct for an equipment.
    pub fn build_equipment_health(
        &self,
        equipment_id: &str,
        current_vibration: f64,
        current_temperature: f64,
    ) -> EquipmentHealth {
        let health_score = self.compute_health_score(equipment_id, current_vibration, current_temperature);
        EquipmentHealth::new(
            equipment_id,
            health_score,
            current_vibration,
            current_temperature,
            self.operating_hours(equipment_id),
        )
    }

    // -----------------------------------------------------------------------
    // Predictive maintenance
    // -----------------------------------------------------------------------

    /// Generate a predictive maintenance indicator.
    pub fn maintenance_indicator(
        &self,
        equipment_id: &str,
        current_vibration: f64,
        current_temperature: f64,
    ) -> MaintenanceIndicator {
        let health_score = self.compute_health_score(equipment_id, current_vibration, current_temperature);
        let failure_probability = self.failure_probability(health_score);
        let maintenance_recommended =
            health_score < self.maintenance_threshold || self.maintenance_due_hours(equipment_id);

        let recommended_action = if failure_probability > 0.7 {
            "Immediate shutdown and inspection required"
        } else if maintenance_recommended {
            "Schedule maintenance within next available window"
        } else {
            "Continue normal operation"
        };

        MaintenanceIndicator {
            equipment_id: equipment_id.to_string(),
            maintenance_recommended,
            health_score,
            failure_probability,
            recommended_action: recommended_action.to_string(),
        }
    }

    /// Estimate failure probability based on health score.
    /// Simple sigmoid model: P = 1 / (1 + exp(k * (score - midpoint)))
    pub fn failure_probability(&self, health_score: f64) -> f64 {
        // Sigmoid centered at score=30, steepness=0.1
        let exponent = 0.1 * (health_score - 30.0);
        1.0 / (1.0 + exponent.exp())
    }

    // -----------------------------------------------------------------------
    // Health history
    // -----------------------------------------------------------------------

    /// Record a health score to history.
    pub fn record_health(&mut self, equipment_id: &str, score: f64) {
        self.health_history
            .entry(equipment_id.to_string())
            .or_default()
            .push(score);
    }

    /// Get health score history for an equipment.
    pub fn health_history(&self, equipment_id: &str) -> &[f64] {
        self.health_history
            .get(equipment_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

impl Default for EquipmentMonitor {
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
    use chrono::{Duration, Utc};

    #[test]
    fn test_new_monitor() {
        let monitor = EquipmentMonitor::new();
        assert_eq!(monitor.maintenance_threshold, 40.0);
    }

    #[test]
    fn test_with_maintenance_threshold() {
        let monitor = EquipmentMonitor::new().with_maintenance_threshold(50.0);
        assert_eq!(monitor.maintenance_threshold, 50.0);
    }

    #[test]
    fn test_compute_vibration_baseline() {
        let mut monitor = EquipmentMonitor::new();
        let values = vec![5.0, 6.0, 5.5, 5.2, 5.8, 6.1, 5.4];
        monitor.compute_vibration_baseline("MILL-01", &values);
        let bl = monitor.vibration_baseline("MILL-01").unwrap();
        assert!((bl.mean - 5.571).abs() < 0.1);
        assert!(bl.std_dev > 0.0);
        assert_eq!(bl.sample_count, 7);
    }

    #[test]
    fn test_compute_vibration_baseline_insufficient() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("X", &[1.0, 2.0]);
        assert!(monitor.vibration_baseline("X").is_none());
    }

    #[test]
    fn test_vibration_zscore() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("MILL-01", &[5.0, 6.0, 5.0, 6.0, 5.0, 6.0]);
        let z = monitor.vibration_zscore("MILL-01", 10.0).unwrap();
        assert!(z > 1.0);
    }

    #[test]
    fn test_vibration_zscore_no_baseline() {
        let monitor = EquipmentMonitor::new();
        assert!(monitor.vibration_zscore("NONE", 5.0).is_none());
    }

    #[test]
    fn test_record_temperature() {
        let mut monitor = EquipmentMonitor::new();
        monitor.record_temperature("MILL-01", 60.0);
        monitor.record_temperature("MILL-01", 62.0);
        assert_eq!(monitor.latest_temperature("MILL-01"), Some(62.0));
    }

    #[test]
    fn test_temperature_trend_rising() {
        let mut monitor = EquipmentMonitor::new();
        for i in 0..10 {
            monitor.record_temperature("MILL-01", 50.0 + i as f64 * 2.0);
        }
        let trend = monitor.temperature_trend("MILL-01").unwrap();
        assert_eq!(trend.direction, TrendDirection::Rising);
        assert!(trend.slope > 0.0);
    }

    #[test]
    fn test_temperature_trend_falling() {
        let mut monitor = EquipmentMonitor::new();
        for i in 0..10 {
            monitor.record_temperature("MILL-01", 100.0 - i as f64 * 2.0);
        }
        let trend = monitor.temperature_trend("MILL-01").unwrap();
        assert_eq!(trend.direction, TrendDirection::Falling);
    }

    #[test]
    fn test_temperature_trend_stable() {
        let mut monitor = EquipmentMonitor::new();
        // Use constant temperature to guarantee stable trend
        for _i in 0..10 {
            monitor.record_temperature("MILL-01", 60.0);
        }
        let trend = monitor.temperature_trend("MILL-01").unwrap();
        assert_eq!(trend.direction, TrendDirection::Stable);
    }

    #[test]
    fn test_temperature_trend_insufficient() {
        let monitor = EquipmentMonitor::new();
        assert!(monitor.temperature_trend("NONE").is_none());
    }

    #[test]
    fn test_operating_hours() {
        let mut monitor = EquipmentMonitor::new();
        monitor.set_operating_hours("MILL-01", 5000.0);
        assert_eq!(monitor.operating_hours("MILL-01"), 5000.0);
        monitor.add_operating_hours("MILL-01", 1000.0);
        assert_eq!(monitor.operating_hours("MILL-01"), 6000.0);
    }

    #[test]
    fn test_operating_hours_default_zero() {
        let monitor = EquipmentMonitor::new();
        assert_eq!(monitor.operating_hours("NONE"), 0.0);
    }

    #[test]
    fn test_maintenance_due_hours() {
        let mut monitor = EquipmentMonitor::new();
        monitor.set_operating_hours("MILL-01", 9000.0);
        assert!(monitor.maintenance_due_hours("MILL-01"));
        assert!(!monitor.maintenance_due_hours("MILL-02"));
    }

    #[test]
    fn test_compute_health_score_good() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("MILL-01", &[5.0, 5.0, 5.0, 5.0, 5.0]);
        for _ in 0..10 {
            monitor.record_temperature("MILL-01", 60.0);
        }
        monitor.set_operating_hours("MILL-01", 1000.0);
        let score = monitor.compute_health_score("MILL-01", 5.0, 60.0);
        assert!(score > 80.0);
    }

    #[test]
    fn test_compute_health_score_poor() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("MILL-01", &[3.0, 3.0, 3.0, 3.0, 3.0]);
        for _ in 0..10 {
            monitor.record_temperature("MILL-01", 60.0);
        }
        monitor.set_operating_hours("MILL-01", 15000.0);
        let score = monitor.compute_health_score("MILL-01", 20.0, 120.0);
        assert!(score < 50.0);
    }

    #[test]
    fn test_build_equipment_health() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("MILL-01", &[3.0, 3.0, 3.0, 3.0]);
        for _ in 0..10 {
            monitor.record_temperature("MILL-01", 55.0);
        }
        monitor.set_operating_hours("MILL-01", 2000.0);
        let health = monitor.build_equipment_health("MILL-01", 3.0, 55.0);
        assert_eq!(health.equipment_id, "MILL-01");
        assert!(health.health_score > 70.0);
    }

    #[test]
    fn test_maintenance_indicator_healthy() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("MILL-01", &[3.0, 3.0, 3.0]);
        for _ in 0..10 {
            monitor.record_temperature("MILL-01", 55.0);
        }
        monitor.set_operating_hours("MILL-01", 1000.0);
        let ind = monitor.maintenance_indicator("MILL-01", 3.0, 55.0);
        assert!(!ind.maintenance_recommended);
        assert!(ind.failure_probability < 0.5);
    }

    #[test]
    fn test_maintenance_indicator_critical() {
        let mut monitor = EquipmentMonitor::new();
        monitor.compute_vibration_baseline("MILL-01", &[3.0, 3.0, 3.0]);
        for _ in 0..10 {
            monitor.record_temperature("MILL-01", 60.0);
        }
        monitor.set_operating_hours("MILL-01", 15000.0);
        let ind = monitor.maintenance_indicator("MILL-01", 20.0, 100.0);
        assert!(ind.maintenance_recommended);
        // Health score should be very low due to constant baseline deviation
        assert!(ind.failure_probability > 0.3);
    }

    #[test]
    fn test_failure_probability_high_health() {
        let monitor = EquipmentMonitor::new();
        let p = monitor.failure_probability(90.0);
        assert!(p < 0.1);
    }

    #[test]
    fn test_failure_probability_low_health() {
        let monitor = EquipmentMonitor::new();
        let p = monitor.failure_probability(10.0);
        assert!(p > 0.5);
    }

    #[test]
    fn test_record_health_history() {
        let mut monitor = EquipmentMonitor::new();
        monitor.record_health("MILL-01", 85.0);
        monitor.record_health("MILL-01", 80.0);
        let history = monitor.health_history("MILL-01");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_health_history_empty() {
        let monitor = EquipmentMonitor::new();
        assert!(monitor.health_history("NONE").is_empty());
    }

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(format!("{}", TrendDirection::Rising), "Rising");
        assert_eq!(format!("{}", TrendDirection::Stable), "Stable");
        assert_eq!(format!("{}", TrendDirection::Falling), "Falling");
    }

    #[test]
    fn test_sensor_baseline_new() {
        let bl = SensorBaseline::new(5.0, 1.0, 100);
        assert!((bl.mean - 5.0).abs() < 1e-10);
        assert!((bl.std_dev - 1.0).abs() < 1e-10);
        assert_eq!(bl.sample_count, 100);
    }
}
