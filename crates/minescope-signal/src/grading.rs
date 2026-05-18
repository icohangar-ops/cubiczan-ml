//! # Mineral Grade Estimation
//!
//! Real-time grade estimation from sensor proxies, multi-signal weighted
//! grade models, trend analysis, cut-off grade optimization, and grade
//! reconciliation against laboratory assay results.

use crate::types::{GradeEstimate, MineralType, SensorReading, SensorType};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Grade estimation model types
// ---------------------------------------------------------------------------

/// A simple linear model mapping a sensor proxy value to a grade estimate.
#[derive(Debug, Clone)]
pub struct GradeProxyModel {
    /// Scale factor (grade = intercept + slope * value).
    pub slope: f64,
    /// Intercept (grade = intercept + slope * value).
    pub intercept: f64,
    /// Minimum plausible grade clamp.
    pub min_grade: f64,
    /// Maximum plausible grade clamp.
    pub max_grade: f64,
}

impl GradeProxyModel {
    /// Create a new linear proxy model.
    pub fn new(slope: f64, intercept: f64, min_grade: f64, max_grade: f64) -> Self {
        GradeProxyModel {
            slope,
            intercept,
            min_grade,
            max_grade,
        }
    }

    /// Estimate grade from a sensor value.
    pub fn estimate(&self, value: f64) -> f64 {
        (self.intercept + self.slope * value).clamp(self.min_grade, self.max_grade)
    }
}

impl Default for GradeProxyModel {
    fn default() -> Self {
        GradeProxyModel::new(1.0, 0.0, 0.0, 100.0)
    }
}

// ---------------------------------------------------------------------------
// Weighted grade model
// ---------------------------------------------------------------------------

/// Configuration for multi-signal weighted grade estimation.
#[derive(Debug, Clone)]
pub struct WeightedGradeConfig {
    /// Weight for vibration signals (default 0.25).
    pub vibration_weight: f64,
    /// Weight for acoustic signals (default 0.30).
    pub acoustic_weight: f64,
    /// Weight for electromagnetic signals (default 0.35).
    pub em_weight: f64,
    /// Weight for chemical signals (default 0.10).
    pub chemical_weight: f64,
}

impl Default for WeightedGradeConfig {
    fn default() -> Self {
        WeightedGradeConfig {
            vibration_weight: 0.25,
            acoustic_weight: 0.30,
            em_weight: 0.35,
            chemical_weight: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// Cut-off decision
// ---------------------------------------------------------------------------

/// Result of a cut-off grade comparison.
#[derive(Debug, Clone)]
pub struct CutOffDecision {
    /// Whether the material should be processed (true) or rejected (false).
    pub process: bool,
    /// Estimated grade.
    pub estimated_grade: f64,
    /// Cut-off grade threshold.
    pub cutoff_grade: f64,
    /// Margin above or below cutoff (positive = above).
    pub margin: f64,
}

// ---------------------------------------------------------------------------
// Grade reconciliation
// ---------------------------------------------------------------------------

/// Result of reconciling estimated grades against actual assay values.
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean bias (positive = overestimation, negative = underestimation).
    pub bias: f64,
    /// Number of samples compared.
    pub sample_count: usize,
}

// ---------------------------------------------------------------------------
// Grade trend
// ---------------------------------------------------------------------------

/// Direction of grade trend over recent readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradeTrend {
    Improving,
    Stable,
    Declining,
}

impl std::fmt::Display for GradeTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradeTrend::Improving => write!(f, "Improving"),
            GradeTrend::Stable => write!(f, "Stable"),
            GradeTrend::Declining => write!(f, "Declining"),
        }
    }
}

// ---------------------------------------------------------------------------
// GradeEstimator
// ---------------------------------------------------------------------------

/// The main grade estimator engine.
#[derive(Debug, Clone)]
pub struct GradeEstimator {
    /// Proxy models per sensor type.
    proxy_models: HashMap<SensorType, GradeProxyModel>,
    /// Weighted grade config.
    weighted_config: WeightedGradeConfig,
    /// Historical grade estimates.
    grade_history: Vec<GradeEstimate>,
}

impl GradeEstimator {
    /// Create a new grade estimator with default proxy models.
    pub fn new() -> Self {
        let mut proxy_models = HashMap::new();
        // Default proxy models for common sensor types
        proxy_models.insert(
            SensorType::Vibration,
            GradeProxyModel::new(0.15, 0.5, 0.0, 50.0),
        );
        proxy_models.insert(
            SensorType::Acoustic,
            GradeProxyModel::new(0.20, 0.3, 0.0, 50.0),
        );
        proxy_models.insert(
            SensorType::Electromagnetic,
            GradeProxyModel::new(0.10, 0.2, 0.0, 50.0),
        );
        proxy_models.insert(
            SensorType::Chemical,
            GradeProxyModel::new(0.05, 0.1, 0.0, 50.0),
        );

        GradeEstimator {
            proxy_models,
            weighted_config: WeightedGradeConfig::default(),
            grade_history: Vec::new(),
        }
    }

    /// Create with custom weighted config.
    pub fn with_weighted_config(mut self, config: WeightedGradeConfig) -> Self {
        self.weighted_config = config;
        self
    }

    /// Set or replace a proxy model for a sensor type.
    pub fn set_proxy_model(&mut self, sensor_type: SensorType, model: GradeProxyModel) {
        self.proxy_models.insert(sensor_type, model);
    }

    /// Get the proxy model for a sensor type.
    pub fn proxy_model(&self, sensor_type: SensorType) -> Option<&GradeProxyModel> {
        self.proxy_models.get(&sensor_type)
    }

    // -----------------------------------------------------------------------
    // Single-sensor grade estimation
    // -----------------------------------------------------------------------

    /// Estimate grade from a single sensor reading using the proxy model.
    pub fn estimate_from_reading(
        &self,
        reading: &SensorReading,
        mineral: MineralType,
    ) -> GradeEstimate {
        let model = self
            .proxy_models
            .get(&reading.sensor_type)
            .cloned()
            .unwrap_or_default();
        let grade = model.estimate(reading.value);
        GradeEstimate::new(mineral, grade, 0.7, 1.0, "single_proxy")
    }

    // -----------------------------------------------------------------------
    // Multi-signal weighted grade model
    // -----------------------------------------------------------------------

    /// Estimate grade from multiple sensor readings using weighted model.
    pub fn weighted_grade_estimate(
        &self,
        readings: &[SensorReading],
        mineral: MineralType,
    ) -> GradeEstimate {
        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for reading in readings {
            let model = self
                .proxy_models
                .get(&reading.sensor_type)
                .cloned()
                .unwrap_or_default();
            let grade = model.estimate(reading.value);
            let weight = match reading.sensor_type {
                SensorType::Vibration => self.weighted_config.vibration_weight,
                SensorType::Acoustic => self.weighted_config.acoustic_weight,
                SensorType::Electromagnetic => self.weighted_config.em_weight,
                SensorType::Chemical => self.weighted_config.chemical_weight,
                _ => 0.05, // small default weight for other sensor types
            };
            weighted_sum += grade * weight;
            total_weight += weight;
        }

        let grade = if total_weight.abs() < 1e-15 {
            0.0
        } else {
            weighted_sum / total_weight
        };

        // Confidence increases with more signal sources
        let confidence = (0.5 + readings.len() as f64 * 0.1).clamp(0.5, 0.95);

        GradeEstimate::new(mineral, grade, confidence, readings.len() as f64, "weighted_multi")
    }

    // -----------------------------------------------------------------------
    // Grade trend analysis
    // -----------------------------------------------------------------------

    /// Analyze grade trend over recent estimates.
    /// Requires at least 4 estimates in the history.
    pub fn grade_trend(&self) -> GradeTrend {
        if self.grade_history.len() < 4 {
            return GradeTrend::Stable;
        }
        let n = self.grade_history.len();
        let first_half_avg: f64 = self.grade_history[..n / 2]
            .iter()
            .map(|g| g.estimated_grade)
            .sum::<f64>()
            / (n / 2) as f64;
        let second_half_avg: f64 = self.grade_history[n / 2..]
            .iter()
            .map(|g| g.estimated_grade)
            .sum::<f64>()
            / (n - n / 2) as f64;

        let change = second_half_avg - first_half_avg;
        let threshold = first_half_avg * 0.05; // 5% threshold

        if change > threshold {
            GradeTrend::Improving
        } else if change < -threshold {
            GradeTrend::Declining
        } else {
            GradeTrend::Stable
        }
    }

    /// Analyze trend from an explicit list of grades.
    pub fn grade_trend_from_values(grades: &[f64]) -> GradeTrend {
        if grades.len() < 4 {
            return GradeTrend::Stable;
        }
        let n = grades.len();
        let first_half_avg: f64 = grades[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let second_half_avg: f64 = grades[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
        let change = second_half_avg - first_half_avg;
        let threshold = first_half_avg.abs() * 0.05;

        if change > threshold {
            GradeTrend::Improving
        } else if change < -threshold {
            GradeTrend::Declining
        } else {
            GradeTrend::Stable
        }
    }

    // -----------------------------------------------------------------------
    // Cut-off grade optimization
    // -----------------------------------------------------------------------

    /// Compare estimated grade against a cut-off threshold.
    pub fn cutoff_decision(&self, estimated_grade: f64, cutoff_grade: f64) -> CutOffDecision {
        let margin = estimated_grade - cutoff_grade;
        CutOffDecision {
            process: margin >= 0.0,
            estimated_grade,
            cutoff_grade,
            margin,
        }
    }

    /// Batch cut-off decisions for multiple grades.
    pub fn batch_cutoff_decisions(
        &self,
        estimates: &[GradeEstimate],
        cutoff_grade: f64,
    ) -> Vec<CutOffDecision> {
        estimates
            .iter()
            .map(|e| self.cutoff_decision(e.estimated_grade, cutoff_grade))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Grade reconciliation
    // -----------------------------------------------------------------------

    /// Reconcile estimated grades against actual assay values.
    /// `estimated` and `actual` must be the same length.
    pub fn reconcile(&self, estimated: &[f64], actual: &[f64]) -> Option<ReconciliationResult> {
        if estimated.is_empty() || estimated.len() != actual.len() {
            return None;
        }
        let n = estimated.len();
        let mae: f64 = estimated
            .iter()
            .zip(actual.iter())
            .map(|(e, a)| (e - a).abs())
            .sum::<f64>()
            / n as f64;
        let mse: f64 = estimated
            .iter()
            .zip(actual.iter())
            .map(|(e, a)| (e - a).powi(2))
            .sum::<f64>()
            / n as f64;
        let rmse = mse.sqrt();
        let bias: f64 = estimated
            .iter()
            .zip(actual.iter())
            .map(|(e, a)| e - a)
            .sum::<f64>()
            / n as f64;

        Some(ReconciliationResult {
            mae,
            rmse,
            bias,
            sample_count: n,
        })
    }

    // -----------------------------------------------------------------------
    // History management
    // -----------------------------------------------------------------------

    /// Record a grade estimate to the history.
    pub fn record_estimate(&mut self, estimate: GradeEstimate) {
        self.grade_history.push(estimate);
    }

    /// Get the grade history.
    pub fn history(&self) -> &[GradeEstimate] {
        &self.grade_history
    }

    /// Get the latest grade estimate.
    pub fn latest_estimate(&self) -> Option<&GradeEstimate> {
        self.grade_history.last()
    }

    /// Clear all history.
    pub fn clear_history(&mut self) {
        self.grade_history.clear();
    }

    /// Number of recorded estimates.
    pub fn history_len(&self) -> usize {
        self.grade_history.len()
    }
}

impl Default for GradeEstimator {
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

    fn make_vibration_reading(value: f64, offset_secs: i64) -> SensorReading {
        SensorReading::new(
            "VIB-001",
            SensorType::Vibration,
            Utc::now() + Duration::seconds(offset_secs),
            value,
        )
    }

    fn make_acoustic_reading(value: f64, offset_secs: i64) -> SensorReading {
        SensorReading::new(
            "ACO-001",
            SensorType::Acoustic,
            Utc::now() + Duration::seconds(offset_secs),
            value,
        )
    }

    fn make_em_reading(value: f64, offset_secs: i64) -> SensorReading {
        SensorReading::new(
            "EMF-001",
            SensorType::Electromagnetic,
            Utc::now() + Duration::seconds(offset_secs),
            value,
        )
    }

    #[test]
    fn test_proxy_model_estimate() {
        let model = GradeProxyModel::new(0.5, 1.0, 0.0, 100.0);
        assert!((model.estimate(10.0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_proxy_model_clamp() {
        let model = GradeProxyModel::new(100.0, 0.0, 0.0, 10.0);
        assert_eq!(model.estimate(50.0), 10.0);
        assert_eq!(model.estimate(-10.0), 0.0);
    }

    #[test]
    fn test_proxy_model_default() {
        let model = GradeProxyModel::default();
        assert!((model.estimate(5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_new_grade_estimator() {
        let ge = GradeEstimator::new();
        assert!(ge.proxy_model(SensorType::Vibration).is_some());
        assert!(ge.history().is_empty());
    }

    #[test]
    fn test_set_proxy_model() {
        let mut ge = GradeEstimator::new();
        let custom = GradeProxyModel::new(2.0, 5.0, 0.0, 100.0);
        ge.set_proxy_model(SensorType::Chemical, custom);
        let model = ge.proxy_model(SensorType::Chemical).unwrap();
        assert!((model.slope - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_from_reading() {
        let ge = GradeEstimator::new();
        let reading = make_vibration_reading(10.0, 0);
        let est = ge.estimate_from_reading(&reading, MineralType::Gold);
        assert_eq!(est.mineral, MineralType::Gold);
        assert!(est.estimated_grade > 0.0);
        assert_eq!(est.method, "single_proxy");
    }

    #[test]
    fn test_weighted_grade_estimate() {
        let ge = GradeEstimator::new();
        let readings = vec![
            make_vibration_reading(10.0, 0),
            make_acoustic_reading(15.0, 0),
            make_em_reading(20.0, 0),
        ];
        let est = ge.weighted_grade_estimate(&readings, MineralType::Copper);
        assert!(est.estimated_grade > 0.0);
        assert!(est.confidence >= 0.5);
        assert_eq!(est.method, "weighted_multi");
    }

    #[test]
    fn test_weighted_grade_estimate_single() {
        let ge = GradeEstimator::new();
        let readings = vec![make_vibration_reading(10.0, 0)];
        let est = ge.weighted_grade_estimate(&readings, MineralType::IronOre);
        assert!(est.estimated_grade > 0.0);
    }

    #[test]
    fn test_weighted_grade_estimate_empty() {
        let ge = GradeEstimator::new();
        let readings: Vec<SensorReading> = vec![];
        let est = ge.weighted_grade_estimate(&readings, MineralType::Gold);
        assert_eq!(est.estimated_grade, 0.0);
    }

    #[test]
    fn test_grade_trend_improving() {
        let grades = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(GradeEstimator::grade_trend_from_values(&grades), GradeTrend::Improving);
    }

    #[test]
    fn test_grade_trend_declining() {
        let grades = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(GradeEstimator::grade_trend_from_values(&grades), GradeTrend::Declining);
    }

    #[test]
    fn test_grade_trend_stable() {
        let grades = vec![5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1];
        assert_eq!(GradeEstimator::grade_trend_from_values(&grades), GradeTrend::Stable);
    }

    #[test]
    fn test_grade_trend_short() {
        let grades = vec![1.0, 2.0];
        assert_eq!(GradeEstimator::grade_trend_from_values(&grades), GradeTrend::Stable);
    }

    #[test]
    fn test_grade_trend_from_history() {
        let mut ge = GradeEstimator::new();
        for i in 0..10 {
            ge.record_estimate(GradeEstimate::new(
                MineralType::Gold,
                1.0 + i as f64,
                0.8,
                1.0,
                "test",
            ));
        }
        assert_eq!(ge.grade_trend(), GradeTrend::Improving);
    }

    #[test]
    fn test_cutoff_decision_process() {
        let ge = GradeEstimator::new();
        let decision = ge.cutoff_decision(5.0, 3.0);
        assert!(decision.process);
        assert!((decision.margin - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cutoff_decision_reject() {
        let ge = GradeEstimator::new();
        let decision = ge.cutoff_decision(2.0, 3.0);
        assert!(!decision.process);
        assert!((decision.margin - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cutoff_decision_exact() {
        let ge = GradeEstimator::new();
        let decision = ge.cutoff_decision(3.0, 3.0);
        assert!(decision.process);
    }

    #[test]
    fn test_batch_cutoff_decisions() {
        let ge = GradeEstimator::new();
        let estimates = vec![
            GradeEstimate::new(MineralType::Gold, 5.0, 0.8, 1.0, "test"),
            GradeEstimate::new(MineralType::Gold, 1.0, 0.8, 1.0, "test"),
            GradeEstimate::new(MineralType::Gold, 3.0, 0.8, 1.0, "test"),
        ];
        let decisions = ge.batch_cutoff_decisions(&estimates, 2.0);
        assert_eq!(decisions.len(), 3);
        assert!(decisions[0].process);
        assert!(!decisions[1].process);
    }

    #[test]
    fn test_reconcile() {
        let ge = GradeEstimator::new();
        let estimated = vec![5.0, 6.0, 4.0, 7.0];
        let actual = vec![5.5, 5.5, 4.5, 6.5];
        let result = ge.reconcile(&estimated, &actual).unwrap();
        assert!(result.mae < 1.0);
        assert!(result.rmse > 0.0);
        assert!(result.bias.abs() < 1.0);
        assert_eq!(result.sample_count, 4);
    }

    #[test]
    fn test_reconcile_empty() {
        let ge = GradeEstimator::new();
        assert!(ge.reconcile(&[], &[]).is_none());
    }

    #[test]
    fn test_reconcile_mismatched() {
        let ge = GradeEstimator::new();
        assert!(ge.reconcile(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn test_record_and_history() {
        let mut ge = GradeEstimator::new();
        let est = GradeEstimate::new(MineralType::Silver, 10.0, 0.9, 1.0, "test");
        ge.record_estimate(est);
        assert_eq!(ge.history_len(), 1);
        assert!(ge.latest_estimate().is_some());
        assert_eq!(ge.latest_estimate().unwrap().estimated_grade, 10.0);
    }

    #[test]
    fn test_clear_history() {
        let mut ge = GradeEstimator::new();
        ge.record_estimate(GradeEstimate::new(MineralType::Gold, 5.0, 0.8, 1.0, "test"));
        ge.clear_history();
        assert!(ge.history().is_empty());
    }

    #[test]
    fn test_grade_trend_display() {
        assert_eq!(format!("{}", GradeTrend::Improving), "Improving");
        assert_eq!(format!("{}", GradeTrend::Stable), "Stable");
        assert_eq!(format!("{}", GradeTrend::Declining), "Declining");
    }
}
