//! Core types for Fed statement analysis and portfolio risk briefing.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Type of Federal Reserve statement or release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatementType {
    FOMC,
    PressConference,
    Testimony,
    Speech,
    Minutes,
}

/// A parsed Federal Reserve statement or FOMC minute release.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedStatement {
    pub date: NaiveDate,
    pub text: String,
    pub statement_type: StatementType,
}

impl FedStatement {
    /// Create a new Fed statement.
    pub fn new(date: NaiveDate, text: impl Into<String>, statement_type: StatementType) -> Self {
        Self {
            date,
            text: text.into(),
            statement_type,
        }
    }
}

/// FOMC minute entry with additional granularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FOMCMinute {
    pub date: NaiveDate,
    pub text: String,
    pub participants_voting: Vec<String>,
    pub participants_dissenting: Vec<String>,
}

impl FOMCMinute {
    pub fn new(date: NaiveDate, text: impl Into<String>) -> Self {
        Self {
            date,
            text: text.into(),
            participants_voting: vec![],
            participants_dissenting: vec![],
        }
    }
}

/// Monetary policy stance classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonetaryPolicy {
    Hawkish,
    Neutral,
    Dovish,
}

impl fmt::Display for MonetaryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MonetaryPolicy::Hawkish => write!(f, "Hawkish"),
            MonetaryPolicy::Neutral => write!(f, "Neutral"),
            MonetaryPolicy::Dovish => write!(f, "Dovish"),
        }
    }
}

/// Multi-dimensional sentiment score output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    /// Overall composite score: -1.0 (dovish) to +1.0 (hawkish).
    pub overall: f64,
    /// Inflation sub-score.
    pub inflation: f64,
    /// Employment sub-score.
    pub employment: f64,
    /// Economic growth sub-score.
    pub growth: f64,
    /// Financial stability sub-score.
    pub financial_stability: f64,
    /// Determined monetary policy stance.
    pub monetary_policy: MonetaryPolicy,
    /// Confidence level of the classification (0.0 - 1.0).
    pub confidence: f64,
    /// Tone shift magnitude from prior statement (positive = more hawkish).
    pub tone_shift: Option<f64>,
    /// Raw keyword counts.
    pub hawkish_keyword_count: usize,
    pub dovish_keyword_count: usize,
}

impl Default for SentimentScore {
    fn default() -> Self {
        Self {
            overall: 0.0,
            inflation: 0.0,
            employment: 0.0,
            growth: 0.0,
            financial_stability: 0.0,
            monetary_policy: MonetaryPolicy::Neutral,
            confidence: 0.0,
            tone_shift: None,
            hawkish_keyword_count: 0,
            dovish_keyword_count: 0,
        }
    }
}

impl SentimentScore {
    /// Create a new neutral sentiment score.
    pub fn neutral() -> Self {
        Self {
            confidence: 1.0,
            ..Default::default()
        }
    }

    /// Classify the overall score into a monetary policy stance.
    pub fn classify_stance(score: f64) -> MonetaryPolicy {
        if score >= 0.2 {
            MonetaryPolicy::Hawkish
        } else if score <= -0.2 {
            MonetaryPolicy::Dovish
        } else {
            MonetaryPolicy::Neutral
        }
    }
}

/// Rate decision extracted from a statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateDecision {
    pub action: RateAction,
    pub basis_points: i32,
    pub new_target_low: f64,
    pub new_target_high: f64,
    pub prior_target_low: f64,
    pub prior_target_high: f64,
}

/// Rate action types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateAction {
    Hike,
    Cut,
    Hold,
    EmergencyHike,
    EmergencyCut,
}

impl fmt::Display for RateAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RateAction::Hike => write!(f, "Rate Hike"),
            RateAction::Cut => write!(f, "Rate Cut"),
            RateAction::Hold => write!(f, "Hold"),
            RateAction::EmergencyHike => write!(f, "Emergency Hike"),
            RateAction::EmergencyCut => write!(f, "Emergency Cut"),
        }
    }
}

/// Value-at-Risk metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRMetric {
    pub method: VaRMethod,
    pub confidence_level: f64,
    pub time_horizon_days: u32,
    pub var_value: f64,
    pub portfolio_value: f64,
    pub var_percentage: f64,
}

impl VaRMetric {
    pub fn new(method: VaRMethod, confidence_level: f64, time_horizon_days: u32,
               var_value: f64, portfolio_value: f64) -> Self {
        let var_percentage = if portfolio_value > 0.0 {
            (var_value / portfolio_value) * 100.0
        } else {
            0.0
        };
        Self {
            method,
            confidence_level,
            time_horizon_days,
            var_value,
            portfolio_value,
            var_percentage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaRMethod {
    Historical,
    Parametric,
}

/// Stress test scenario and results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    pub name: String,
    pub rate_shock_bps: i32,
    pub estimated_loss: f64,
    pub loss_percentage: f64,
    pub description: String,
}

/// A portfolio position for risk analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPosition {
    pub asset: String,
    pub duration: f64,
    pub convexity: f64,
    pub notional: f64,
    pub weight: f64,
}

impl PortfolioPosition {
    pub fn new(asset: impl Into<String>, duration: f64, convexity: f64,
               notional: f64, weight: f64) -> Self {
        Self {
            asset: asset.into(),
            duration,
            convexity,
            notional,
            weight,
        }
    }

    /// Calculate price change estimate for a given rate change (in bps).
    pub fn estimate_price_change(&self, rate_change_bps: f64) -> f64 {
        let delta_yield = rate_change_bps / 10000.0;
        let duration_effect = -self.duration * delta_yield;
        let convexity_effect = 0.5 * self.convexity * delta_yield * delta_yield;
        (duration_effect + convexity_effect) * self.notional
    }
}

/// Correlation matrix entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEntry {
    pub asset_a: String,
    pub asset_b: String,
    pub correlation: f64,
}

/// Risk metric result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub interpretation: String,
}

impl RiskMetric {
    pub fn new(name: impl Into<String>, value: f64, unit: impl Into<String>,
               interpretation: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            interpretation: interpretation.into(),
        }
    }
}

/// Verification gate for confidence-gated outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationGate {
    pub passed: bool,
    pub confidence: f64,
    pub threshold: f64,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl VerificationGate {
    /// Create a new verification gate with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            passed: false,
            confidence: 0.0,
            threshold,
            warnings: vec![],
            errors: vec![],
        }
    }

    /// Evaluate the gate against a confidence score.
    pub fn evaluate(&mut self, confidence: f64, warnings: Vec<String>, errors: Vec<String>) {
        self.confidence = confidence;
        self.warnings = warnings;
        self.errors = errors;
        self.passed = confidence >= self.threshold && self.errors.is_empty();
    }
}

/// Rebalancing suggestion from the risk engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceSuggestion {
    pub asset: String,
    pub action: String,
    pub current_weight: f64,
    pub suggested_weight: f64,
    pub rationale: String,
}

/// Complete risk briefing output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskBriefing {
    pub generated_at: NaiveDate,
    pub statement_date: NaiveDate,
    pub sentiment: SentimentScore,
    pub rate_decision: Option<RateDecision>,
    pub var_metrics: Vec<VaRMetric>,
    pub stress_scenarios: Vec<StressScenario>,
    pub risk_metrics: Vec<RiskMetric>,
    pub rebalance_suggestions: Vec<RebalanceSuggestion>,
    pub verification_gate: VerificationGate,
    pub summary: String,
    pub recommendations: Vec<String>,
}

impl RiskBriefing {
    pub fn new(statement_date: NaiveDate, sentiment: SentimentScore,
               verification_gate: VerificationGate) -> Self {
        Self {
            generated_at: chrono::Local::now().date_naive(),
            statement_date,
            sentiment,
            rate_decision: None,
            var_metrics: vec![],
            stress_scenarios: vec![],
            risk_metrics: vec![],
            rebalance_suggestions: vec![],
            verification_gate,
            summary: String::new(),
            recommendations: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monetary_policy_display() {
        assert_eq!(format!("{}", MonetaryPolicy::Hawkish), "Hawkish");
        assert_eq!(format!("{}", MonetaryPolicy::Neutral), "Neutral");
        assert_eq!(format!("{}", MonetaryPolicy::Dovish), "Dovish");
    }

    #[test]
    fn test_rate_action_display() {
        assert_eq!(format!("{}", RateAction::Hike), "Rate Hike");
        assert_eq!(format!("{}", RateAction::Cut), "Rate Cut");
        assert_eq!(format!("{}", RateAction::Hold), "Hold");
    }

    #[test]
    fn test_sentiment_score_classify_stance() {
        assert_eq!(SentimentScore::classify_stance(0.5), MonetaryPolicy::Hawkish);
        assert_eq!(SentimentScore::classify_stance(-0.5), MonetaryPolicy::Dovish);
        assert_eq!(SentimentScore::classify_stance(0.0), MonetaryPolicy::Neutral);
        assert_eq!(SentimentScore::classify_stance(0.21), MonetaryPolicy::Hawkish);
        assert_eq!(SentimentScore::classify_stance(-0.21), MonetaryPolicy::Dovish);
        assert_eq!(SentimentScore::classify_stance(0.2), MonetaryPolicy::Hawkish);
        assert_eq!(SentimentScore::classify_stance(-0.2), MonetaryPolicy::Dovish);
    }

    #[test]
    fn test_sentiment_score_default() {
        let score = SentimentScore::default();
        assert_eq!(score.overall, 0.0);
        assert_eq!(score.monetary_policy, MonetaryPolicy::Neutral);
        assert_eq!(score.confidence, 0.0);
        assert!(score.tone_shift.is_none());
    }

    #[test]
    fn test_sentiment_score_neutral() {
        let score = SentimentScore::neutral();
        assert_eq!(score.overall, 0.0);
        assert_eq!(score.confidence, 1.0);
    }

    #[test]
    fn test_fed_statement_new() {
        let stmt = FedStatement::new(
            NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            "Test statement text.",
            StatementType::FOMC,
        );
        assert_eq!(stmt.text, "Test statement text.");
        assert_eq!(stmt.statement_type, StatementType::FOMC);
    }

    #[test]
    fn test_fomc_minute_new() {
        let minute = FOMCMinute::new(
            NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            "Minutes text.",
        );
        assert_eq!(minute.text, "Minutes text.");
        assert!(minute.participants_voting.is_empty());
    }

    #[test]
    fn test_portfolio_position_price_change() {
        let pos = PortfolioPosition::new("Bond", 5.0, 50.0, 1_000_000.0, 1.0);
        // Rate increase of 100 bps
        let loss = pos.estimate_price_change(100.0);
        assert!(loss < 0.0, "Price should decrease on rate increase");
        // Rate decrease of 100 bps
        let gain = pos.estimate_price_change(-100.0);
        assert!(gain > 0.0, "Price should increase on rate decrease");
    }

    #[test]
    fn test_var_metric_new() {
        let var = VaRMetric::new(VaRMethod::Historical, 0.95, 1, 50_000.0, 1_000_000.0);
        assert_eq!(var.var_percentage, 5.0);
    }

    #[test]
    fn test_risk_metric_new() {
        let rm = RiskMetric::new("Duration", 5.0, "years", "Moderate interest rate risk");
        assert_eq!(rm.name, "Duration");
        assert_eq!(rm.value, 5.0);
    }

    #[test]
    fn test_verification_gate_new() {
        let gate = VerificationGate::new(0.8);
        assert!(!gate.passed);
        assert_eq!(gate.threshold, 0.8);
    }

    #[test]
    fn test_verification_gate_evaluate_passed() {
        let mut gate = VerificationGate::new(0.7);
        gate.evaluate(0.85, vec![], vec![]);
        assert!(gate.passed);
    }

    #[test]
    fn test_verification_gate_evaluate_failed_low_confidence() {
        let mut gate = VerificationGate::new(0.9);
        gate.evaluate(0.5, vec![], vec![]);
        assert!(!gate.passed);
    }

    #[test]
    fn test_verification_gate_evaluate_failed_errors() {
        let mut gate = VerificationGate::new(0.5);
        gate.evaluate(0.8, vec![], vec!["Insufficient data".to_string()]);
        assert!(!gate.passed);
    }

    #[test]
    fn test_risk_briefing_new() {
        let briefing = RiskBriefing::new(
            NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            SentimentScore::neutral(),
            VerificationGate::new(0.7),
        );
        assert!(briefing.var_metrics.is_empty());
        assert!(briefing.stress_scenarios.is_empty());
        assert!(briefing.recommendations.is_empty());
    }
}
