//! # Minescope Signal Processing
//!
//! A comprehensive Rust crate for mining sensor signal analysis, anomaly
//! detection, frequency-domain processing, grade estimation, equipment
//! health monitoring, and end-to-end pipeline orchestration.
//!
//! ## Modules
//!
//! - **types**: Core domain types (sensors, anomalies, grades, equipment, alerts)
//! - **sensors**: Sensor data storage, quality validation, fusion, resampling
//! - **anomaly**: Multi-method anomaly detection (Z-score, MA, IQR, ROC)
//! - **fft**: Frequency-domain analysis (DFT, filtering, spectral features)
//! - **processing**: Mining process signal analysis and efficiency scoring
//! - **grading**: Mineral grade estimation from sensor proxies
//! - **equipment**: Equipment health monitoring and predictive maintenance
//! - **pipeline**: End-to-end analysis orchestration

pub mod anomaly;
pub mod equipment;
pub mod fft;
pub mod grading;
pub mod pipeline;
pub mod processing;
pub mod sensors;
pub mod types;

// ---------------------------------------------------------------------------
// Key type re-exports
// ---------------------------------------------------------------------------

pub use anomaly::{AnomalyConfig, AnomalyDetector};
pub use equipment::{EquipmentMonitor, MaintenanceIndicator, SensorBaseline, TemperatureTrend};
pub use fft::{FilterType, SignalProcessor};
pub use grading::{CutOffDecision, GradeEstimator, GradeProxyModel, GradeTrend, ReconciliationResult, WeightedGradeConfig};
pub use pipeline::{AlertConfig, MiningPipeline, PipelineResult, SiteConfig};
pub use processing::{EfficiencyReport, ProcessAnalyzer, StageBenchmark};
pub use sensors::SensorDatabase;
pub use types::{
    AlertLevel, AnomalyEvent, AnomalyType, EquipmentHealth, GradeEstimate, MiningAlert,
    MineralType, ProcessingStage, ProcessSignal, QualityFlag, SensorReading, SensorType,
    Severity,
};
