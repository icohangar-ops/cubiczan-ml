//! # Full Analysis Pipeline Orchestration
//!
//! End-to-end processing pipeline for mining sensor data:
//! - Sensor ingestion → anomaly detection → FFT → grading → equipment health
//! - Alert generation with configurable thresholds
//! - Multi-site monitoring aggregation
//! - Health summary report generation

use crate::anomaly::AnomalyDetector;
use crate::equipment::{EquipmentMonitor, MaintenanceIndicator};
use crate::fft::SignalProcessor;
use crate::grading::GradeEstimator;
use crate::processing::ProcessAnalyzer;
use crate::sensors::SensorDatabase;
use crate::types::{
    AlertLevel, AnomalyEvent, EquipmentHealth, GradeEstimate, MiningAlert,
    MineralType, ProcessingStage, SensorReading, SensorType,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Configuration for alert generation thresholds.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Minimum anomaly severity to generate an alert.
    pub min_anomaly_severity: crate::types::Severity,
    /// Minimum health score to consider healthy (no alert).
    pub healthy_score_threshold: f64,
    /// Enable grade-related alerts.
    pub grade_alerts_enabled: bool,
    /// Cut-off grade threshold for grade alerts.
    pub grade_cutoff: f64,
    /// Enable equipment alerts.
    pub equipment_alerts_enabled: bool,
}

impl Default for AlertConfig {
    fn default() -> Self {
        AlertConfig {
            min_anomaly_severity: crate::types::Severity::Warning,
            healthy_score_threshold: 40.0,
            grade_alerts_enabled: true,
            grade_cutoff: 2.0,
            equipment_alerts_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline result
// ---------------------------------------------------------------------------

/// Result of running the full pipeline on a single site.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Site identifier.
    pub site_id: String,
    /// Detected anomalies.
    pub anomalies: Vec<AnomalyEvent>,
    /// Grade estimates.
    pub grade_estimates: Vec<GradeEstimate>,
    /// Equipment health statuses.
    pub equipment_health: Vec<EquipmentHealth>,
    /// Generated alerts.
    pub alerts: Vec<MiningAlert>,
    /// Dominant frequencies per sensor.
    pub dominant_frequencies: HashMap<String, Option<(f64, f64)>>,
    /// Maintenance indicators.
    pub maintenance_indicators: Vec<MaintenanceIndicator>,
}

// ---------------------------------------------------------------------------
// Site configuration
// ---------------------------------------------------------------------------

/// Configuration for a monitored site.
#[derive(Debug, Clone)]
pub struct SiteConfig {
    /// Site identifier.
    pub site_id: String,
    /// Sensors belonging to this site.
    pub sensor_ids: Vec<String>,
    /// Equipment IDs at this site.
    pub equipment_ids: Vec<String>,
    /// Mineral type for grade estimation.
    pub mineral: MineralType,
    /// Mapping from sensor IDs to equipment IDs (vibration sensors to equipment).
    pub sensor_to_equipment: HashMap<String, String>,
}

impl SiteConfig {
    /// Create a new site configuration.
    pub fn new(site_id: impl Into<String>, mineral: MineralType) -> Self {
        SiteConfig {
            site_id: site_id.into(),
            sensor_ids: Vec::new(),
            equipment_ids: Vec::new(),
            mineral,
            sensor_to_equipment: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// MiningPipeline
// ---------------------------------------------------------------------------

/// Full mining analysis pipeline orchestrator.
pub struct MiningPipeline {
    /// Sensor database.
    sensor_db: SensorDatabase,
    /// Anomaly detector.
    anomaly_detector: AnomalyDetector,
    /// Signal processor (FFT).
    signal_processor: SignalProcessor,
    /// Grade estimator.
    grade_estimator: GradeEstimator,
    /// Equipment monitor.
    equipment_monitor: EquipmentMonitor,
    /// Process analyzer.
    process_analyzer: ProcessAnalyzer,
    /// Alert configuration.
    alert_config: AlertConfig,
    /// Site configurations.
    sites: HashMap<String, SiteConfig>,
}

impl MiningPipeline {
    /// Create a new mining pipeline with default settings.
    pub fn new() -> Self {
        MiningPipeline {
            sensor_db: SensorDatabase::new(),
            anomaly_detector: AnomalyDetector::new(),
            signal_processor: SignalProcessor::default(),
            grade_estimator: GradeEstimator::new(),
            equipment_monitor: EquipmentMonitor::new(),
            process_analyzer: ProcessAnalyzer::new(),
            alert_config: AlertConfig::default(),
            sites: HashMap::new(),
        }
    }

    /// Create with custom alert configuration.
    pub fn with_alert_config(mut self, config: AlertConfig) -> Self {
        self.alert_config = config;
        self
    }

    /// Get a reference to the sensor database.
    pub fn sensor_db(&self) -> &SensorDatabase {
        &self.sensor_db
    }

    /// Get a mutable reference to the sensor database.
    pub fn sensor_db_mut(&mut self) -> &mut SensorDatabase {
        &mut self.sensor_db
    }

    /// Register a site configuration.
    pub fn register_site(&mut self, site: SiteConfig) {
        self.sites.insert(site.site_id.clone(), site);
    }

    /// Ingest a batch of sensor readings.
    pub fn ingest_readings(&mut self, readings: Vec<SensorReading>) {
        for r in readings {
            self.sensor_db.add_reading(r);
        }
    }

    // -----------------------------------------------------------------------
    // Full pipeline run
    // -----------------------------------------------------------------------

    /// Run the full analysis pipeline for a specific site.
    pub fn run_pipeline(&mut self, site_id: &str) -> PipelineResult {
        let site = self.sites.get(site_id).cloned().unwrap_or_else(|| {
            let mut s = SiteConfig::new(site_id, MineralType::Copper);
            s.sensor_ids = self.sensor_db.sensor_ids();
            s
        });

        let mut result = PipelineResult {
            site_id: site_id.to_string(),
            anomalies: Vec::new(),
            grade_estimates: Vec::new(),
            equipment_health: Vec::new(),
            alerts: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };

        // Step 1: Anomaly detection
        for sensor_id in &site.sensor_ids {
            let anomalies = self.anomaly_detector.detect_all(&self.sensor_db, sensor_id);
            result.anomalies.extend(anomalies);
        }

        // Step 2: FFT analysis
        for sensor_id in &site.sensor_ids {
            let values = self.sensor_db.values(sensor_id);
            let dom_freq = self.signal_processor.dominant_frequency(&values);
            result.dominant_frequencies.insert(sensor_id.clone(), dom_freq);
        }

        // Step 3: Grade estimation
        let mut grade_readings: Vec<SensorReading> = Vec::new();
        for sensor_id in &site.sensor_ids {
            if let Some(latest) = self.sensor_db.latest_reading(sensor_id) {
                if matches!(
                    latest.sensor_type,
                    SensorType::Vibration
                        | SensorType::Acoustic
                        | SensorType::Electromagnetic
                        | SensorType::Chemical
                ) {
                    grade_readings.push(latest.clone());
                }
            }
        }
        if !grade_readings.is_empty() {
            let grade = self
                .grade_estimator
                .weighted_grade_estimate(&grade_readings, site.mineral);
            result.grade_estimates.push(grade);
        }

        // Step 4: Equipment health
        for eq_id in &site.equipment_ids {
            // Find vibration sensor for this equipment
            let vib_value = site
                .sensor_to_equipment
                .iter()
                .find(|(_, eq)| *eq == eq_id)
                .and_then(|(sid, _)| self.sensor_db.latest_reading(sid))
                .map(|r| r.value)
                .unwrap_or(5.0);

            let temp_value = self
                .sensor_db
                .latest_reading(
                    &site
                        .sensor_ids
                        .iter()
                        .find(|sid| {
                            self.sensor_db
                                .sensor_type(sid)
                                .map(|st| st == SensorType::Temperature)
                                .unwrap_or(false)
                        })
                        .cloned()
                        .unwrap_or_default(),
                )
                .map(|r| r.value)
                .unwrap_or(60.0);

            let health = self
                .equipment_monitor
                .build_equipment_health(eq_id, vib_value, temp_value);
            result.equipment_health.push(health);

            let mi = self
                .equipment_monitor
                .maintenance_indicator(eq_id, vib_value, temp_value);
            result.maintenance_indicators.push(mi);
        }

        // Step 5: Alert generation
        result.alerts = self.generate_alerts(&result);

        result
    }

    // -----------------------------------------------------------------------
    // Alert generation
    // -----------------------------------------------------------------------

    /// Generate alerts from pipeline results based on alert config.
    pub fn generate_alerts(&self, result: &PipelineResult) -> Vec<MiningAlert> {
        let mut alerts = Vec::new();

        // Anomaly-based alerts
        for anomaly in &result.anomalies {
            if anomaly.severity >= self.alert_config.min_anomaly_severity {
                let alert_level = match anomaly.severity {
                    crate::types::Severity::Emergency => AlertLevel::Critical,
                    crate::types::Severity::Critical => AlertLevel::Critical,
                    crate::types::Severity::Warning => AlertLevel::Warning,
                    crate::types::Severity::Info => AlertLevel::Info,
                };
                alerts.push(MiningAlert::new(
                    alert_level,
                    "anomaly",
                    &anomaly.description,
                    vec![anomaly.sensor_id.clone()],
                    anomaly.start_time,
                    "Investigate sensor anomaly",
                ));
            }
        }

        // Equipment alerts
        if self.alert_config.equipment_alerts_enabled {
            for eh in &result.equipment_health {
                if eh.health_score < self.alert_config.healthy_score_threshold {
                    let level = if eh.health_score < 20.0 {
                        AlertLevel::Critical
                    } else {
                        AlertLevel::Warning
                    };
                    alerts.push(MiningAlert::new(
                        level,
                        "equipment_health",
                        format!(
                            "Equipment {} health score: {:.1}% ({})",
                            eh.equipment_id, eh.health_score, eh.status()
                        ),
                        vec![eh.equipment_id.clone()],
                        chrono::Utc::now(),
                        "Schedule maintenance inspection",
                    ));
                }
            }
        }

        // Grade alerts
        if self.alert_config.grade_alerts_enabled {
            for ge in &result.grade_estimates {
                if ge.estimated_grade < self.alert_config.grade_cutoff {
                    alerts.push(MiningAlert::new(
                        AlertLevel::Advisory,
                        "grade",
                        format!(
                            "Low grade estimate: {:.2} {} for {}",
                            ge.estimated_grade,
                            ge.mineral.grade_unit(),
                            ge.mineral
                        ),
                        vec![],
                        chrono::Utc::now(),
                        "Review ore source quality",
                    ));
                }
            }
        }

        alerts
    }

    // -----------------------------------------------------------------------
    // Multi-site aggregation
    // -----------------------------------------------------------------------

    /// Run pipeline across all registered sites.
    pub fn run_all_sites(&mut self) -> HashMap<String, PipelineResult> {
        let site_ids: Vec<String> = self.sites.keys().cloned().collect();
        let mut results = HashMap::new();
        for site_id in site_ids {
            results.insert(site_id.clone(), self.run_pipeline(&site_id));
        }
        results
    }

    /// Aggregate alerts from multiple pipeline results.
    pub fn aggregate_alerts(results: &[PipelineResult]) -> Vec<MiningAlert> {
        let mut all_alerts: Vec<MiningAlert> = results
            .iter()
            .flat_map(|r| r.alerts.clone())
            .collect();
        all_alerts.sort_by(|a, b| b.alert_level.cmp(&a.alert_level));
        all_alerts
    }

    /// Aggregate health summary across multiple sites.
    pub fn aggregate_health_summary(results: &[PipelineResult]) -> String {
        let mut summary = String::from("=== Multi-Site Health Summary ===\n");
        let mut total_anomalies = 0usize;
        let mut total_alerts = 0usize;
        let mut total_equipment = 0usize;
        let mut critical_equipment = 0usize;

        for result in results {
            let avg_health = if !result.equipment_health.is_empty() {
                result
                    .equipment_health
                    .iter()
                    .map(|eh| eh.health_score)
                    .sum::<f64>()
                    / result.equipment_health.len() as f64
            } else {
                100.0
            };

            summary.push_str(&format!(
                "Site: {} | Anomalies: {} | Alerts: {} | Avg Health: {:.1}%\n",
                result.site_id,
                result.anomalies.len(),
                result.alerts.len(),
                avg_health
            ));

            total_anomalies += result.anomalies.len();
            total_alerts += result.alerts.len();
            total_equipment += result.equipment_health.len();
            critical_equipment += result.equipment_health.iter().filter(|eh| eh.health_score < 40.0).count();
        }

        summary.push_str(&format!(
            "\nTotals: {} anomalies, {} alerts, {}/{} equipment critical\n",
            total_anomalies, total_alerts, critical_equipment, total_equipment
        ));

        summary
    }

    // -----------------------------------------------------------------------
    // Report generation
    // -----------------------------------------------------------------------

    /// Generate a comprehensive health report for a pipeline result.
    pub fn generate_report(result: &PipelineResult) -> String {
        let mut report = String::new();
        report.push_str(&format!("=== Report for Site: {} ===\n", result.site_id));

        // Anomalies
        report.push_str(&format!("Anomalies Detected: {}\n", result.anomalies.len()));
        for a in &result.anomalies {
            report.push_str(&format!(
                "  - [{:?}] {} on {}: {}\n",
                a.severity, a.anomaly_type, a.sensor_id, a.description
            ));
        }

        // Grade estimates
        if !result.grade_estimates.is_empty() {
            for ge in &result.grade_estimates {
                report.push_str(&format!(
                    "Grade Estimate: {:.3} {} ({}, confidence: {:.0}%)\n",
                    ge.estimated_grade,
                    ge.mineral.grade_unit(),
                    ge.mineral,
                    ge.confidence
                ));
            }
        }

        // Equipment
        if !result.equipment_health.is_empty() {
            report.push_str("Equipment Health:\n");
            for eh in &result.equipment_health {
                report.push_str(&format!(
                    "  - {}: {:.1}% ({}) | Vib: {:.1} mm/s | Temp: {:.1}°C | Hours: {:.0}\n",
                    eh.equipment_id,
                    eh.health_score,
                    eh.status(),
                    eh.vibration_level,
                    eh.temperature,
                    eh.operating_hours
                ));
            }
        }

        // Alerts
        report.push_str(&format!("Alerts: {}\n", result.alerts.len()));
        for alert in &result.alerts {
            report.push_str(&format!(
                "  - [{}] {}: {}\n",
                alert.alert_level, alert.category, alert.message
            ));
        }

        report
    }
}

impl Default for MiningPipeline {
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
    use crate::types::QualityFlag;
    use chrono::{Duration, Utc};

    fn make_simple_pipeline() -> MiningPipeline {
        let mut pipeline = MiningPipeline::new();

        // Set up a simple site
        let mut site = SiteConfig::new("MINE-A", MineralType::Gold);
        site.sensor_ids = vec![
            "VIB-001".to_string(),
            "TMP-001".to_string(),
            "ACO-001".to_string(),
        ];
        site.equipment_ids = vec!["CRUSHER-01".to_string()];
        site.sensor_to_equipment
            .insert("VIB-001".to_string(), "CRUSHER-01".to_string());
        pipeline.register_site(site);

        // Add some readings
        let base = Utc::now();
        let readings: Vec<SensorReading> = (0..50)
            .flat_map(|i| {
                vec![
                    SensorReading::new("VIB-001", SensorType::Vibration, base + Duration::seconds(i), 5.0 + rand::random::<f64>() * 0.5),
                    SensorReading::new("TMP-001", SensorType::Temperature, base + Duration::seconds(i), 60.0 + rand::random::<f64>() * 2.0),
                    SensorReading::new("ACO-001", SensorType::Acoustic, base + Duration::seconds(i), 85.0 + rand::random::<f64>() * 2.0),
                ]
            })
            .collect();
        pipeline.ingest_readings(readings);

        // Set up equipment baselines
        let vib_values: Vec<f64> = (0..50)
            .map(|_| 5.0 + rand::random::<f64>() * 0.5)
            .collect();
        pipeline
            .equipment_monitor
            .compute_vibration_baseline("CRUSHER-01", &vib_values);
        for _ in 0..50 {
            pipeline
                .equipment_monitor
                .record_temperature("CRUSHER-01", 60.0);
        }
        pipeline
            .equipment_monitor
            .set_operating_hours("CRUSHER-01", 2000.0);

        pipeline
    }

    #[test]
    fn test_new_pipeline() {
        let pipeline = MiningPipeline::new();
        assert_eq!(pipeline.sensor_db.total_readings(), 0);
    }

    #[test]
    fn test_register_site() {
        let mut pipeline = MiningPipeline::new();
        let site = SiteConfig::new("MINE-A", MineralType::Gold);
        pipeline.register_site(site);
        assert!(pipeline.sites.contains_key("MINE-A"));
    }

    #[test]
    fn test_ingest_readings() {
        let mut pipeline = MiningPipeline::new();
        let reading = SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 5.0);
        pipeline.ingest_readings(vec![reading]);
        assert_eq!(pipeline.sensor_db.total_readings(), 1);
    }

    #[test]
    fn test_run_pipeline() {
        let mut pipeline = make_simple_pipeline();
        let result = pipeline.run_pipeline("MINE-A");
        assert_eq!(result.site_id, "MINE-A");
        // Should have at least one grade estimate (we have 3 relevant sensors)
        assert!(!result.grade_estimates.is_empty());
    }

    #[test]
    fn test_pipeline_equipment_health() {
        let mut pipeline = make_simple_pipeline();
        let result = pipeline.run_pipeline("MINE-A");
        assert!(!result.equipment_health.is_empty());
        let health = &result.equipment_health[0];
        assert_eq!(health.equipment_id, "CRUSHER-01");
    }

    #[test]
    fn test_pipeline_maintenance_indicators() {
        let mut pipeline = make_simple_pipeline();
        let result = pipeline.run_pipeline("MINE-A");
        assert!(!result.maintenance_indicators.is_empty());
    }

    #[test]
    fn test_pipeline_dominant_frequencies() {
        let mut pipeline = make_simple_pipeline();
        let result = pipeline.run_pipeline("MINE-A");
        assert!(result.dominant_frequencies.contains_key("VIB-001"));
    }

    #[test]
    fn test_generate_alerts_empty() {
        let pipeline = MiningPipeline::new();
        let result = PipelineResult {
            site_id: "TEST".to_string(),
            anomalies: Vec::new(),
            grade_estimates: Vec::new(),
            equipment_health: Vec::new(),
            alerts: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let alerts = pipeline.generate_alerts(&result);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_generate_alerts_equipment() {
        let mut pipeline = MiningPipeline::new();
        pipeline.alert_config.equipment_alerts_enabled = true;
        pipeline.alert_config.healthy_score_threshold = 40.0;

        let result = PipelineResult {
            site_id: "TEST".to_string(),
            anomalies: Vec::new(),
            grade_estimates: Vec::new(),
            equipment_health: vec![EquipmentHealth::new("X", 25.0, 10.0, 90.0, 5000.0)],
            alerts: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let alerts = pipeline.generate_alerts(&result);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_level, AlertLevel::Warning);
    }

    #[test]
    fn test_generate_alerts_grade() {
        let mut pipeline = MiningPipeline::new();
        pipeline.alert_config.grade_alerts_enabled = true;
        pipeline.alert_config.grade_cutoff = 5.0;

        let result = PipelineResult {
            site_id: "TEST".to_string(),
            anomalies: Vec::new(),
            grade_estimates: vec![GradeEstimate::new(MineralType::Gold, 2.0, 0.8, 1.0, "test")],
            equipment_health: Vec::new(),
            alerts: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let alerts = pipeline.generate_alerts(&result);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_level, AlertLevel::Advisory);
    }

    #[test]
    fn test_run_all_sites() {
        let mut pipeline = make_simple_pipeline();
        let site2 = SiteConfig::new("MINE-B", MineralType::Copper);
        pipeline.register_site(site2);

        let results = pipeline.run_all_sites();
        assert!(results.contains_key("MINE-A"));
        assert!(results.contains_key("MINE-B"));
    }

    #[test]
    fn test_aggregate_alerts() {
        let r1 = PipelineResult {
            site_id: "A".to_string(),
            alerts: vec![MiningAlert::new(
                AlertLevel::Warning,
                "test",
                "msg",
                vec![],
                Utc::now(),
                "act",
            )],
            anomalies: Vec::new(),
            grade_estimates: Vec::new(),
            equipment_health: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let r2 = PipelineResult {
            site_id: "B".to_string(),
            alerts: vec![MiningAlert::new(
                AlertLevel::Critical,
                "test",
                "msg",
                vec![],
                Utc::now(),
                "act",
            )],
            anomalies: Vec::new(),
            grade_estimates: Vec::new(),
            equipment_health: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let aggregated = MiningPipeline::aggregate_alerts(&[r1, r2]);
        assert_eq!(aggregated.len(), 2);
        // Critical should come first
        assert_eq!(aggregated[0].alert_level, AlertLevel::Critical);
    }

    #[test]
    fn test_aggregate_health_summary() {
        let r1 = PipelineResult {
            site_id: "A".to_string(),
            anomalies: vec![AnomalyEvent::new(
                crate::types::AnomalyType::Spike,
                "VIB",
                Utc::now(),
                crate::types::Severity::Warning,
                3.0,
                "spike",
            )],
            alerts: Vec::new(),
            grade_estimates: Vec::new(),
            equipment_health: vec![EquipmentHealth::new("X", 85.0, 3.0, 55.0, 2000.0)],
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let summary = MiningPipeline::aggregate_health_summary(&[r1]);
        assert!(summary.contains("Site: A"));
        assert!(summary.contains("Totals"));
    }

    #[test]
    fn test_generate_report() {
        let result = PipelineResult {
            site_id: "MINE-A".to_string(),
            anomalies: Vec::new(),
            grade_estimates: vec![GradeEstimate::new(MineralType::Gold, 5.2, 0.9, 1.0, "weighted_multi")],
            equipment_health: vec![EquipmentHealth::new("CRUSHER-01", 85.0, 3.0, 55.0, 2000.0)],
            alerts: Vec::new(),
            dominant_frequencies: HashMap::new(),
            maintenance_indicators: Vec::new(),
        };
        let report = MiningPipeline::generate_report(&result);
        assert!(report.contains("MINE-A"));
        assert!(report.contains("Gold"));
        assert!(report.contains("CRUSHER-01"));
    }

    #[test]
    fn test_with_alert_config() {
        let config = AlertConfig {
            min_anomaly_severity: crate::types::Severity::Critical,
            healthy_score_threshold: 50.0,
            grade_alerts_enabled: false,
            grade_cutoff: 1.0,
            equipment_alerts_enabled: false,
        };
        let pipeline = MiningPipeline::new().with_alert_config(config);
        assert_eq!(pipeline.alert_config.healthy_score_threshold, 50.0);
    }
}
