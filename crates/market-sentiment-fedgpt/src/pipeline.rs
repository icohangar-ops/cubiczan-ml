//! End-to-end analysis pipeline: parser → tone → risk → brief.

use crate::types::*;
use crate::tone::ToneAnalyzer;
use crate::risk::RiskCalculator;
use crate::brief::{BriefingConfig, BriefingGenerator};

/// End-to-end analysis pipeline that orchestrates all components.
pub struct AnalysisPipeline {
    tone_analyzer: ToneAnalyzer,
    risk_calculator: RiskCalculator,
    briefing_generator: BriefingGenerator,
    confidence_threshold: f64,
}

impl AnalysisPipeline {
    /// Create a new analysis pipeline with the given confidence threshold.
    pub fn new(confidence_threshold: f64) -> Self {
        let config = BriefingConfig::new(confidence_threshold);
        let risk_calculator = RiskCalculator::new(0.95, 1);
        Self {
            tone_analyzer: ToneAnalyzer::new(),
            risk_calculator: risk_calculator.clone(),
            briefing_generator: BriefingGenerator::new(config),
            confidence_threshold,
        }
    }

    /// Run the full analysis pipeline: parse statement → score tone →
    /// calculate risk → generate briefing.
    pub fn run(
        &self,
        statement: &FedStatement,
        prior_statement: Option<&FedStatement>,
        positions: &[PortfolioPosition],
    ) -> anyhow::Result<RiskBriefing> {
        // Step 1: Analyze tone
        let sentiment = if let Some(prior) = prior_statement {
            self.tone_analyzer.analyze_with_shift(&statement.text, &prior.text)
        } else {
            let mut score = self.tone_analyzer.analyze(&statement.text);
            score.confidence = self.tone_analyzer.compute_confidence(
                &self.tone_analyzer.parser().parse(&statement.text),
            );
            score
        };

        // Step 2: Calculate stress test scenarios
        let stress_scenarios = self.risk_calculator.stress_test(positions, &sentiment);

        // Step 3: Generate briefing
        let briefing = self.briefing_generator.generate(
            statement.date,
            sentiment,
            positions,
            stress_scenarios,
        );

        // Step 4: Final verification
        if !briefing.verification_gate.passed {
            // Still return the briefing, but the gate marks it as unverified
        }

        Ok(briefing)
    }

    /// Run tone analysis only.
    pub fn analyze_tone(&self, text: &str) -> SentimentScore {
        let mut score = self.tone_analyzer.analyze(text);
        score.confidence = self.tone_analyzer.compute_confidence(
            &self.tone_analyzer.parser().parse(text),
        );
        score
    }

    /// Run tone analysis with shift detection.
    pub fn analyze_tone_with_shift(&self, current: &str, prior: &str) -> SentimentScore {
        self.tone_analyzer.analyze_with_shift(current, prior)
    }

    /// Run risk calculations only.
    pub fn calculate_risk(&self, positions: &[PortfolioPosition], sentiment: &SentimentScore)
        -> (Vec<VaRMetric>, Vec<StressScenario>, Vec<RiskMetric>) {
        let total: f64 = positions.iter().map(|p| p.notional).sum();

        // Generate synthetic returns for VaR
        let returns = self.briefing_generator.generate_synthetic_returns(sentiment, positions);
        let hist_var = self.risk_calculator.historical_var(&returns, total);
        let param_var = self.risk_calculator.parametric_var(0.001, 0.02, total);
        let var_metrics = vec![hist_var, param_var];

        // Stress tests
        let stress = self.risk_calculator.stress_test(positions, sentiment);

        // Risk metrics
        let var_ref = var_metrics.first().cloned()
            .unwrap_or_else(|| VaRMetric::new(VaRMethod::Historical, 0.95, 1, 0.0, 0.0));
        let risk_metrics = self.risk_calculator.compute_risk_metrics(positions, &var_ref);

        (var_metrics, stress, risk_metrics)
    }

    /// Get the confidence threshold used by this pipeline.
    pub fn confidence_threshold(&self) -> f64 {
        self.confidence_threshold
    }

    /// Check if a given briefing passes the verification gate.
    pub fn is_verified(briefing: &RiskBriefing) -> bool {
        briefing.verification_gate.passed
    }
}

// Make internal methods accessible for testing
impl AnalysisPipeline {
    /// Get a reference to the internal tone analyzer.
    #[cfg(test)]
    pub fn tone_analyzer(&self) -> &ToneAnalyzer {
        &self.tone_analyzer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hawkish_statement() -> FedStatement {
        FedStatement::new(
            chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            "The Committee decided to maintain the target range for the federal \
             funds rate at 5.25 to 5.50 percent. Inflation remains elevated and \
             the labor market continues to be tight. The Committee is strongly \
             committed to returning inflation to its 2 percent objective.",
            StatementType::FOMC,
        )
    }

    fn dovish_statement() -> FedStatement {
        FedStatement::new(
            chrono::NaiveDate::from_ymd_opt(2024, 12, 18).unwrap(),
            "The Committee decided to lower the target range. Inflation has \
             moderated significantly. The labor market is softening. Gradual \
             easing is appropriate to support the economy.",
            StatementType::FOMC,
        )
    }

    fn sample_positions() -> Vec<PortfolioPosition> {
        vec![
            PortfolioPosition::new("10Y Treasury", 8.5, 75.0, 1_000_000.0, 0.5),
            PortfolioPosition::new("S&P 500 ETF", 0.0, 0.0, 1_000_000.0, 0.5),
        ]
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = AnalysisPipeline::new(0.7);
        assert_eq!(pipeline.confidence_threshold(), 0.7);
    }

    #[test]
    fn test_pipeline_analyze_tone() {
        let pipeline = AnalysisPipeline::new(0.5);
        let score = pipeline.analyze_tone(
            "Inflation remains elevated. Additional rate increases may be appropriate."
        );
        assert!(score.overall > 0.0);
    }

    #[test]
    fn test_pipeline_analyze_tone_with_shift() {
        let pipeline = AnalysisPipeline::new(0.5);
        let score = pipeline.analyze_tone_with_shift(
            "Inflation moderating. Consider rate cuts.",
            "Inflation elevated. Rate hikes needed.",
        );
        assert!(score.tone_shift.is_some());
        assert!(score.tone_shift.unwrap() < 0.0);
    }

    #[test]
    fn test_pipeline_calculate_risk() {
        let pipeline = AnalysisPipeline::new(0.5);
        let sentiment = SentimentScore {
            overall: 0.5,
            monetary_policy: MonetaryPolicy::Hawkish,
            confidence: 0.8,
            ..Default::default()
        };

        let (var, stress, metrics) = pipeline.calculate_risk(&sample_positions(), &sentiment);

        assert!(!var.is_empty());
        assert!(!stress.is_empty());
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_pipeline_run_hawkish() {
        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&hawkish_statement(), None, &sample_positions()).unwrap();

        assert_eq!(briefing.sentiment.monetary_policy, MonetaryPolicy::Hawkish);
        assert!(!briefing.summary.is_empty());
        assert!(!briefing.recommendations.is_empty());
    }

    #[test]
    fn test_pipeline_run_dovish() {
        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&dovish_statement(), None, &sample_positions()).unwrap();

        assert_eq!(briefing.sentiment.monetary_policy, MonetaryPolicy::Dovish);
    }

    #[test]
    fn test_pipeline_run_with_prior() {
        let pipeline = AnalysisPipeline::new(0.5);
        let prior = FedStatement::new(
            chrono::NaiveDate::from_ymd_opt(2024, 3, 20).unwrap(),
            "The Committee is strongly committed to tightening. Inflation remains elevated.",
            StatementType::FOMC,
        );
        let current = FedStatement::new(
            chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            "The Committee has decided to maintain rates. Inflation is moderating.",
            StatementType::FOMC,
        );

        let briefing = pipeline.run(&current, Some(&prior), &sample_positions()).unwrap();

        assert!(briefing.sentiment.tone_shift.is_some());
    }

    #[test]
    fn test_pipeline_verification_passed() {
        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&hawkish_statement(), None, &sample_positions()).unwrap();
        assert!(AnalysisPipeline::is_verified(&briefing));
    }

    #[test]
    fn test_pipeline_verification_failed_low_threshold_text() {
        let pipeline = AnalysisPipeline::new(0.95);
        let short_stmt = FedStatement::new(
            chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            "The Committee met today.",
            StatementType::FOMC,
        );
        let briefing = pipeline.run(&short_stmt, None, &sample_positions()).unwrap();
        assert!(!AnalysisPipeline::is_verified(&briefing));
    }

    #[test]
    fn test_pipeline_empty_positions() {
        let pipeline = AnalysisPipeline::new(0.3);
        let result = pipeline.run(&hawkish_statement(), None, &[]);
        assert!(result.is_ok());
        // Should have error about zero notional
        assert!(!result.unwrap().verification_gate.errors.is_empty());
    }

    #[test]
    fn test_pipeline_briefing_has_var_metrics() {
        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&hawkish_statement(), None, &sample_positions()).unwrap();
        assert!(!briefing.var_metrics.is_empty());
        assert!(briefing.var_metrics.iter().any(|v| v.method == VaRMethod::Historical));
        assert!(briefing.var_metrics.iter().any(|v| v.method == VaRMethod::Parametric));
    }

    #[test]
    fn test_pipeline_briefing_has_stress_scenarios() {
        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&hawkish_statement(), None, &sample_positions()).unwrap();
        assert!(!briefing.stress_scenarios.is_empty());
    }
}
