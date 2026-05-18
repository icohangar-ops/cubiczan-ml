//! # Core Types for Minescope Signal Processing
//!
//! Domain types for mining sensor data, anomaly events, process signals,
//! grade estimation, equipment health, and alerting.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Mineral types
// ---------------------------------------------------------------------------

/// Types of minerals tracked in mining operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MineralType {
    Gold,
    Silver,
    Copper,
    Lithium,
    Nickel,
    Cobalt,
    IronOre,
    Bauxite,
    Zinc,
    Manganese,
}

impl MineralType {
    /// Typical market symbol for the mineral.
    pub fn symbol(&self) -> &'static str {
        match self {
            MineralType::Gold => "AU",
            MineralType::Silver => "AG",
            MineralType::Copper => "CU",
            MineralType::Lithium => "LI",
            MineralType::Nickel => "NI",
            MineralType::Cobalt => "CO",
            MineralType::IronOre => "FE",
            MineralType::Bauxite => "AL",
            MineralType::Zinc => "ZN",
            MineralType::Manganese => "MN",
        }
    }

    /// Typical unit of measurement for ore grade.
    pub fn grade_unit(&self) -> &'static str {
        match self {
            MineralType::Gold => "g/t",
            MineralType::Silver => "g/t",
            MineralType::Copper => "%",
            MineralType::Lithium => "% Li₂O",
            MineralType::Nickel => "%",
            MineralType::Cobalt => "%",
            MineralType::IronOre => "% Fe",
            MineralType::Bauxite => "% Al₂O₃",
            MineralType::Zinc => "%",
            MineralType::Manganese => "% Mn",
        }
    }

    /// All mineral variants.
    pub fn all() -> &'static [MineralType] {
        &[
            MineralType::Gold,
            MineralType::Silver,
            MineralType::Copper,
            MineralType::Lithium,
            MineralType::Nickel,
            MineralType::Cobalt,
            MineralType::IronOre,
            MineralType::Bauxite,
            MineralType::Zinc,
            MineralType::Manganese,
        ]
    }
}

impl std::fmt::Display for MineralType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MineralType::Gold => write!(f, "Gold"),
            MineralType::Silver => write!(f, "Silver"),
            MineralType::Copper => write!(f, "Copper"),
            MineralType::Lithium => write!(f, "Lithium"),
            MineralType::Nickel => write!(f, "Nickel"),
            MineralType::Cobalt => write!(f, "Cobalt"),
            MineralType::IronOre => write!(f, "Iron Ore"),
            MineralType::Bauxite => write!(f, "Bauxite"),
            MineralType::Zinc => write!(f, "Zinc"),
            MineralType::Manganese => write!(f, "Manganese"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sensor types
// ---------------------------------------------------------------------------

/// Types of sensors deployed in mining environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorType {
    Vibration,
    Temperature,
    Pressure,
    Acoustic,
    Electromagnetic,
    Chemical,
    FlowRate,
    Power,
}

impl SensorType {
    /// Typical measurement unit for this sensor type.
    pub fn default_unit(&self) -> &'static str {
        match self {
            SensorType::Vibration => "mm/s",
            SensorType::Temperature => "°C",
            SensorType::Pressure => "kPa",
            SensorType::Acoustic => "dB",
            SensorType::Electromagnetic => "μT",
            SensorType::Chemical => "ppm",
            SensorType::FlowRate => "m³/h",
            SensorType::Power => "kW",
        }
    }

    /// Typical operational range (min, max).
    pub fn typical_range(&self) -> (f64, f64) {
        match self {
            SensorType::Vibration => (0.0, 50.0),
            SensorType::Temperature => (-20.0, 200.0),
            SensorType::Pressure => (0.0, 500.0),
            SensorType::Acoustic => (40.0, 140.0),
            SensorType::Electromagnetic => (0.0, 1000.0),
            SensorType::Chemical => (0.0, 500.0),
            SensorType::FlowRate => (0.0, 1000.0),
            SensorType::Power => (0.0, 5000.0),
        }
    }

    /// All sensor type variants.
    pub fn all() -> &'static [SensorType] {
        &[
            SensorType::Vibration,
            SensorType::Temperature,
            SensorType::Pressure,
            SensorType::Acoustic,
            SensorType::Electromagnetic,
            SensorType::Chemical,
            SensorType::FlowRate,
            SensorType::Power,
        ]
    }
}

impl std::fmt::Display for SensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensorType::Vibration => write!(f, "Vibration"),
            SensorType::Temperature => write!(f, "Temperature"),
            SensorType::Pressure => write!(f, "Pressure"),
            SensorType::Acoustic => write!(f, "Acoustic"),
            SensorType::Electromagnetic => write!(f, "Electromagnetic"),
            SensorType::Chemical => write!(f, "Chemical"),
            SensorType::FlowRate => write!(f, "FlowRate"),
            SensorType::Power => write!(f, "Power"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sensor reading
// ---------------------------------------------------------------------------

/// Quality flag for sensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityFlag {
    Good,
    Suspect,
    Bad,
    Missing,
}

/// A single sensor reading from the field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub unit: String,
    pub quality_flag: QualityFlag,
}

impl SensorReading {
    /// Create a new sensor reading with default unit from sensor type.
    pub fn new(
        sensor_id: impl Into<String>,
        sensor_type: SensorType,
        timestamp: DateTime<Utc>,
        value: f64,
    ) -> Self {
        SensorReading {
            sensor_id: sensor_id.into(),
            sensor_type,
            timestamp,
            value,
            unit: sensor_type.default_unit().to_string(),
            quality_flag: QualityFlag::Good,
        }
    }

    /// Create a reading with explicit unit and quality flag.
    pub fn with_quality(
        sensor_id: impl Into<String>,
        sensor_type: SensorType,
        timestamp: DateTime<Utc>,
        value: f64,
        unit: impl Into<String>,
        quality_flag: QualityFlag,
    ) -> Self {
        SensorReading {
            sensor_id: sensor_id.into(),
            sensor_type,
            timestamp,
            value,
            unit: unit.into(),
            quality_flag,
        }
    }

    /// Check if the reading value is within the typical range for its sensor type.
    pub fn is_in_range(&self) -> bool {
        let (min, max) = self.sensor_type.typical_range();
        self.value >= min && self.value <= max
    }
}

// ---------------------------------------------------------------------------
// Anomaly types
// ---------------------------------------------------------------------------

/// Classification of anomaly patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    Spike,
    Drift,
    StepChange,
    Noise,
    Intermittent,
    TrendShift,
    OutOfRange,
}

impl std::fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyType::Spike => write!(f, "Spike"),
            AnomalyType::Drift => write!(f, "Drift"),
            AnomalyType::StepChange => write!(f, "StepChange"),
            AnomalyType::Noise => write!(f, "Noise"),
            AnomalyType::Intermittent => write!(f, "Intermittent"),
            AnomalyType::TrendShift => write!(f, "TrendShift"),
            AnomalyType::OutOfRange => write!(f, "OutOfRange"),
        }
    }
}

/// Severity levels for anomaly events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "Info"),
            Severity::Warning => write!(f, "Warning"),
            Severity::Critical => write!(f, "Critical"),
            Severity::Emergency => write!(f, "Emergency"),
        }
    }
}

/// A detected anomaly event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub anomaly_type: AnomalyType,
    pub sensor_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub severity: Severity,
    pub magnitude: f64,
    pub description: String,
}

impl AnomalyEvent {
    /// Create a new anomaly event.
    pub fn new(
        anomaly_type: AnomalyType,
        sensor_id: impl Into<String>,
        start_time: DateTime<Utc>,
        severity: Severity,
        magnitude: f64,
        description: impl Into<String>,
    ) -> Self {
        AnomalyEvent {
            anomaly_type,
            sensor_id: sensor_id.into(),
            start_time,
            end_time: None,
            severity,
            magnitude,
            description: description.into(),
        }
    }

    /// Mark the anomaly as resolved.
    pub fn resolve(&mut self, end_time: DateTime<Utc>) {
        self.end_time = Some(end_time);
    }

    /// Check if this anomaly is still ongoing.
    pub fn is_ongoing(&self) -> bool {
        self.end_time.is_none()
    }

    /// Duration of the anomaly in seconds, or None if ongoing.
    pub fn duration_secs(&self) -> Option<i64> {
        self.end_time.map(|et| et.timestamp() - self.start_time.timestamp())
    }
}

// ---------------------------------------------------------------------------
// Processing stages
// ---------------------------------------------------------------------------

/// Stages in the mineral processing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingStage {
    Crushing,
    Grinding,
    Flotation,
    Leaching,
    Smelting,
    Refining,
    Conveying,
    Sorting,
}

impl ProcessingStage {
    /// All processing stages in order.
    pub fn all_ordered() -> &'static [ProcessingStage] {
        &[
            ProcessingStage::Crushing,
            ProcessingStage::Grinding,
            ProcessingStage::Flotation,
            ProcessingStage::Leaching,
            ProcessingStage::Smelting,
            ProcessingStage::Refining,
            ProcessingStage::Conveying,
            ProcessingStage::Sorting,
        ]
    }

    /// Position of this stage in the processing pipeline (0-indexed).
    pub fn index(&self) -> usize {
        match self {
            ProcessingStage::Crushing => 0,
            ProcessingStage::Grinding => 1,
            ProcessingStage::Flotation => 2,
            ProcessingStage::Leaching => 3,
            ProcessingStage::Smelting => 4,
            ProcessingStage::Refining => 5,
            ProcessingStage::Conveying => 6,
            ProcessingStage::Sorting => 7,
        }
    }
}

impl std::fmt::Display for ProcessingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingStage::Crushing => write!(f, "Crushing"),
            ProcessingStage::Grinding => write!(f, "Grinding"),
            ProcessingStage::Flotation => write!(f, "Flotation"),
            ProcessingStage::Leaching => write!(f, "Leaching"),
            ProcessingStage::Smelting => write!(f, "Smelting"),
            ProcessingStage::Refining => write!(f, "Refining"),
            ProcessingStage::Conveying => write!(f, "Conveying"),
            ProcessingStage::Sorting => write!(f, "Sorting"),
        }
    }
}

// ---------------------------------------------------------------------------
// Process signal
// ---------------------------------------------------------------------------

/// A signal from a processing stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessSignal {
    pub stage: ProcessingStage,
    pub signal_type: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ProcessSignal {
    /// Create a new process signal.
    pub fn new(
        stage: ProcessingStage,
        signal_type: impl Into<String>,
        value: f64,
        timestamp: DateTime<Utc>,
        confidence: f64,
    ) -> Self {
        ProcessSignal {
            stage,
            signal_type: signal_type.into(),
            value,
            timestamp,
            confidence: confidence.clamp(0.0, 1.0),
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Grade estimation
// ---------------------------------------------------------------------------

/// A mineral grade estimate from sensor data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeEstimate {
    pub mineral: MineralType,
    pub estimated_grade: f64,
    pub confidence: f64,
    pub sample_weight: f64,
    pub method: String,
}

impl GradeEstimate {
    /// Create a new grade estimate.
    pub fn new(
        mineral: MineralType,
        estimated_grade: f64,
        confidence: f64,
        sample_weight: f64,
        method: impl Into<String>,
    ) -> Self {
        GradeEstimate {
            mineral,
            estimated_grade,
            confidence: confidence.clamp(0.0, 1.0),
            sample_weight,
            method: method.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Equipment health
// ---------------------------------------------------------------------------

/// Equipment health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquipmentHealth {
    pub equipment_id: String,
    pub health_score: f64,      // 0-100
    pub vibration_level: f64,
    pub temperature: f64,
    pub operating_hours: f64,
    pub maintenance_due: bool,
}

impl EquipmentHealth {
    /// Create a new equipment health record.
    pub fn new(
        equipment_id: impl Into<String>,
        health_score: f64,
        vibration_level: f64,
        temperature: f64,
        operating_hours: f64,
    ) -> Self {
        EquipmentHealth {
            equipment_id: equipment_id.into(),
            health_score: health_score.clamp(0.0, 100.0),
            vibration_level,
            temperature,
            operating_hours,
            maintenance_due: health_score < 40.0,
        }
    }

    /// Health status category.
    pub fn status(&self) -> &'static str {
        if self.health_score >= 80.0 {
            "Healthy"
        } else if self.health_score >= 60.0 {
            "Fair"
        } else if self.health_score >= 40.0 {
            "Warning"
        } else {
            "Critical"
        }
    }
}

// ---------------------------------------------------------------------------
// Mining alert
// ---------------------------------------------------------------------------

/// Alert level for mining operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Advisory,
    Warning,
    Critical,
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertLevel::Info => write!(f, "Info"),
            AlertLevel::Advisory => write!(f, "Advisory"),
            AlertLevel::Warning => write!(f, "Warning"),
            AlertLevel::Critical => write!(f, "Critical"),
        }
    }
}

/// A mining alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningAlert {
    pub alert_level: AlertLevel,
    pub category: String,
    pub message: String,
    pub sensor_ids: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub recommended_action: String,
}

impl MiningAlert {
    /// Create a new mining alert.
    pub fn new(
        alert_level: AlertLevel,
        category: impl Into<String>,
        message: impl Into<String>,
        sensor_ids: Vec<String>,
        timestamp: DateTime<Utc>,
        recommended_action: impl Into<String>,
    ) -> Self {
        MiningAlert {
            alert_level,
            category: category.into(),
            message: message.into(),
            sensor_ids,
            timestamp,
            recommended_action: recommended_action.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mineral_type_symbols() {
        assert_eq!(MineralType::Gold.symbol(), "AU");
        assert_eq!(MineralType::Copper.symbol(), "CU");
        assert_eq!(MineralType::IronOre.symbol(), "FE");
    }

    #[test]
    fn test_mineral_type_grade_units() {
        assert_eq!(MineralType::Gold.grade_unit(), "g/t");
        assert_eq!(MineralType::Copper.grade_unit(), "%");
    }

    #[test]
    fn test_mineral_type_all() {
        assert_eq!(MineralType::all().len(), 10);
    }

    #[test]
    fn test_mineral_type_display() {
        assert_eq!(format!("{}", MineralType::Lithium), "Lithium");
        assert_eq!(format!("{}", MineralType::IronOre), "Iron Ore");
    }

    #[test]
    fn test_sensor_type_units() {
        assert_eq!(SensorType::Vibration.default_unit(), "mm/s");
        assert_eq!(SensorType::Temperature.default_unit(), "°C");
        assert_eq!(SensorType::Power.default_unit(), "kW");
    }

    #[test]
    fn test_sensor_type_ranges() {
        let (lo, hi) = SensorType::Vibration.typical_range();
        assert!(lo < hi);
        assert_eq!(lo, 0.0);
    }

    #[test]
    fn test_sensor_type_all() {
        assert_eq!(SensorType::all().len(), 8);
    }

    #[test]
    fn test_sensor_reading_new() {
        let r = SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 5.2);
        assert_eq!(r.sensor_id, "VIB-001");
        assert_eq!(r.unit, "mm/s");
        assert_eq!(r.quality_flag, QualityFlag::Good);
        assert!(r.is_in_range());
    }

    #[test]
    fn test_sensor_reading_out_of_range() {
        let r = SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 999.0);
        assert!(!r.is_in_range());
    }

    #[test]
    fn test_sensor_reading_with_quality() {
        let r = SensorReading::with_quality(
            "TMP-001",
            SensorType::Temperature,
            Utc::now(),
            75.0,
            "°C",
            QualityFlag::Suspect,
        );
        assert_eq!(r.quality_flag, QualityFlag::Suspect);
    }

    #[test]
    fn test_anomaly_event_new() {
        let evt = AnomalyEvent::new(
            AnomalyType::Spike,
            "VIB-001",
            Utc::now(),
            Severity::Warning,
            3.5,
            "Sudden vibration spike",
        );
        assert!(evt.is_ongoing());
        assert!(evt.duration_secs().is_none());
    }

    #[test]
    fn test_anomaly_event_resolve() {
        let start = Utc::now();
        let mut evt = AnomalyEvent::new(AnomalyType::Drift, "TMP-002", start, Severity::Critical, 1.0, "Drift");
        let end = start + chrono::Duration::minutes(5);
        evt.resolve(end);
        assert!(!evt.is_ongoing());
        assert_eq!(evt.duration_secs(), Some(300));
    }

    #[test]
    fn test_processing_stage_ordering() {
        let stages = ProcessingStage::all_ordered();
        assert_eq!(stages.len(), 8);
        assert_eq!(stages[0], ProcessingStage::Crushing);
        assert_eq!(stages[stages.len() - 1], ProcessingStage::Sorting);
    }

    #[test]
    fn test_processing_stage_index() {
        assert_eq!(ProcessingStage::Crushing.index(), 0);
        assert_eq!(ProcessingStage::Refining.index(), 5);
    }

    #[test]
    fn test_process_signal_new() {
        let ps = ProcessSignal::new(ProcessingStage::Grinding, "particle_size", 120.0, Utc::now(), 0.85);
        assert_eq!(ps.stage, ProcessingStage::Grinding);
        assert_eq!(ps.confidence, 0.85);
    }

    #[test]
    fn test_process_signal_confidence_clamped() {
        let ps = ProcessSignal::new(ProcessingStage::Crushing, "throughput", 100.0, Utc::now(), 1.5);
        assert_eq!(ps.confidence, 1.0);
    }

    #[test]
    fn test_grade_estimate_new() {
        let ge = GradeEstimate::new(MineralType::Gold, 5.2, 0.9, 1.0, "acoustic_proxy");
        assert_eq!(ge.mineral, MineralType::Gold);
        assert_eq!(ge.estimated_grade, 5.2);
    }

    #[test]
    fn test_equipment_health_new() {
        let eh = EquipmentHealth::new("MILL-01", 85.0, 3.2, 55.0, 12000.0);
        assert_eq!(eh.status(), "Healthy");
        assert!(!eh.maintenance_due);
    }

    #[test]
    fn test_equipment_health_critical() {
        let eh = EquipmentHealth::new("CRUSHER-02", 25.0, 15.0, 120.0, 45000.0);
        assert_eq!(eh.status(), "Critical");
        assert!(eh.maintenance_due);
    }

    #[test]
    fn test_equipment_health_score_clamped() {
        let eh = EquipmentHealth::new("X", 150.0, 0.0, 0.0, 0.0);
        assert_eq!(eh.health_score, 100.0);
    }

    #[test]
    fn test_mining_alert_new() {
        let alert = MiningAlert::new(
            AlertLevel::Critical,
            "overheat",
            "Crusher motor temperature critical",
            vec!["TMP-001".to_string()],
            Utc::now(),
            "Shut down crusher immediately",
        );
        assert_eq!(alert.alert_level, AlertLevel::Critical);
        assert_eq!(alert.sensor_ids.len(), 1);
    }

    #[test]
    fn test_serde_roundtrip_sensor_reading() {
        let r = SensorReading::new("VIB-001", SensorType::Vibration, Utc::now(), 5.2);
        let json = serde_json::to_string(&r).unwrap();
        let r2: SensorReading = serde_json::from_str(&json).unwrap();
        assert_eq!(r.sensor_id, r2.sensor_id);
        assert_eq!(r.sensor_type, r2.sensor_type);
    }

    #[test]
    fn test_serde_roundtrip_anomaly_event() {
        let evt = AnomalyEvent::new(AnomalyType::Spike, "VIB-001", Utc::now(), Severity::Warning, 3.5, "Spike");
        let json = serde_json::to_string(&evt).unwrap();
        let evt2: AnomalyEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(evt.anomaly_type, evt2.anomaly_type);
        assert_eq!(evt.sensor_id, evt2.sensor_id);
    }

    #[test]
    fn test_serde_roundtrip_equipment_health() {
        let eh = EquipmentHealth::new("MILL-01", 75.0, 4.0, 60.0, 8000.0);
        let json = serde_json::to_string(&eh).unwrap();
        let eh2: EquipmentHealth = serde_json::from_str(&json).unwrap();
        assert_eq!(eh.equipment_id, eh2.equipment_id);
    }
}
