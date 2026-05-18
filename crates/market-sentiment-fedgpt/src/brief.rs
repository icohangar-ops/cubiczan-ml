//! Structured risk briefing generator.
//!
//! Generates formatted risk briefings with scenario analysis, recommendations,
//! and verification-gated outputs.

use crate::types::*;
use crate::risk::RiskCalculator;

/// Briefing template configuration.
#[derive(Debug, Clone)]
pub struct BriefingConfig {
    /// Confidence threshold for verification gate.
    pub confidence_threshold: f64,
    /// Include stress test results in briefing.
    pub include_stress_tests: bool,
    /// Include rebalancing suggestions in briefing.
    pub include_rebalancing: bool,
    /// Maximum number of stress scenarios to include.
    pub max_stress_scenarios: usize,
    /// Include correlation analysis.
    pub include_correlations: bool,
}

impl Default for BriefingConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            include_stress_tests: true,
            include_rebalancing: true,
            max_stress_scenarios: 5,
            include_correlations: true,
        }
    }
}

impl BriefingConfig {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
            ..Default::default()
        }
    }
}

/// Structured risk briefing generator.
pub struct BriefingGenerator {
    config: BriefingConfig,
    risk_calculator: RiskCalculator,
}

impl BriefingGenerator {
    pub fn new(config: BriefingConfig) -> Self {
        let risk_calc = RiskCalculator::new(0.95, 1);
        Self {
            config,
            risk_calculator: risk_calc,
        }
    }

    /// Generate a complete risk briefing from all analysis components.
    pub fn generate(
        &self,
        statement_date: chrono::NaiveDate,
        sentiment: SentimentScore,
        positions: &[PortfolioPosition],
        stress_scenarios: Vec<StressScenario>,
    ) -> RiskBriefing {
        let mut briefing = RiskBriefing::new(
            statement_date,
            sentiment.clone(),
            VerificationGate::new(self.config.confidence_threshold),
        );

        // Compute VaR
        let returns = self.generate_synthetic_returns(&sentiment, positions);
        let var = self.risk_calculator.historical_var(&returns, self.total_notional(positions));
        briefing.var_metrics.push(var);

        // Compute parametric VaR
        let (mean_ret, std_ret) = self.compute_return_stats(&returns);
        let parametric_var = self.risk_calculator.parametric_var(
            mean_ret, std_ret, self.total_notional(positions),
        );
        briefing.var_metrics.push(parametric_var);

        // Add stress scenarios
        if self.config.include_stress_tests {
            briefing.stress_scenarios = stress_scenarios
                .into_iter()
                .take(self.config.max_stress_scenarios)
                .collect();
        }

        // Compute risk metrics
        let var_ref = briefing.var_metrics.first()
            .cloned()
            .unwrap_or_else(|| VaRMetric::new(VaRMethod::Historical, 0.95, 1, 0.0, 0.0));
        briefing.risk_metrics = self.risk_calculator.compute_risk_metrics(positions, &var_ref);

        // Rebalancing suggestions
        if self.config.include_rebalancing {
            briefing.rebalance_suggestions = self.risk_calculator
                .suggest_rebalancing(positions, &sentiment, &briefing.stress_scenarios);
        }

        // Generate summary
        briefing.summary = self.build_summary(&briefing.sentiment, &briefing.risk_metrics,
                                               &briefing.stress_scenarios, positions);

        // Generate recommendations
        briefing.recommendations = self.build_recommendations(
            &briefing.sentiment, &briefing.stress_scenarios, &briefing.rebalance_suggestions,
        );

        // Evaluate verification gate
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        if self.total_notional(positions) == 0.0 {
            errors.push("Portfolio has zero total notional value.".to_string());
        }

        if positions.is_empty() {
            warnings.push("No positions provided — risk analysis limited.".to_string());
        }

        if sentiment.confidence < 0.5 {
            warnings.push(format!(
                "Low sentiment confidence ({:.1}%) — tone classification may be unreliable.",
                sentiment.confidence * 100.0
            ));
        }

        let (duration, _, _) = self.risk_calculator.portfolio_duration_convexity(positions);
        if duration > 10.0 {
            warnings.push(format!(
                "Very high portfolio duration ({:.2}) — significant rate sensitivity.",
                duration
            ));
        }

        if briefing.stress_scenarios.iter().any(|s| s.loss_percentage > 10.0) {
            warnings.push(
                "Stress scenarios indicate potential losses exceeding 10% of portfolio.".to_string()
            );
        }

        briefing.verification_gate.evaluate(sentiment.confidence, warnings, errors);

        briefing
    }

    /// Build a human-readable risk summary.
    pub fn build_summary(
        &self,
        sentiment: &SentimentScore,
        risk_metrics: &[RiskMetric],
        stress_scenarios: &[StressScenario],
        positions: &[PortfolioPosition],
    ) -> String {
        let mut summary = String::new();

        // Monetary policy assessment
        summary.push_str(&format!(
            "FED TONE ASSESSMENT: {}\n",
            sentiment.monetary_policy
        ));
        summary.push_str(&format!(
            "Overall Sentiment Score: {:.3} (Confidence: {:.1}%)\n",
            sentiment.overall,
            sentiment.confidence * 100.0
        ));

        // Tone shift
        if let Some(shift) = sentiment.tone_shift {
            let direction = if shift > 0.05 {
                "shifted HAWKISH"
            } else if shift < -0.05 {
                "shifted DOVISH"
            } else {
                "remained NEUTRAL"
            };
            summary.push_str(&format!(
                "Tone Shift: {} ({:+.3})\n", direction, shift
            ));
        }

        summary.push('\n');

        // Dimension scores
        summary.push_str("DIMENSION SCORES:\n");
        summary.push_str(&format!("  Inflation:        {:+.3}\n", sentiment.inflation));
        summary.push_str(&format!("  Employment:       {:+.3}\n", sentiment.employment));
        summary.push_str(&format!("  Growth:           {:+.3}\n", sentiment.growth));
        summary.push_str(&format!("  Financial Stab:   {:+.3}\n", sentiment.financial_stability));

        summary.push('\n');

        // Portfolio risk summary
        let (duration, convexity, total) = self.risk_calculator.portfolio_duration_convexity(positions);
        summary.push_str(&format!(
            "PORTFOLIO: ${:.0} | Duration: {:.2}y | Convexity: {:.1}\n",
            total, duration, convexity
        ));

        // VaR summary
        for metric in risk_metrics {
            if metric.name.contains("VaR") {
                summary.push_str(&format!(
                    "  {} | {} {:.0} ({:.2}%)\n",
                    metric.name, metric.unit, metric.value, metric.value / total * 100.0
                ));
            }
        }

        // Stress test summary
        if !stress_scenarios.is_empty() {
            summary.push_str("\nKEY STRESS SCENARIOS:\n");
            for scenario in stress_scenarios.iter().take(3) {
                summary.push_str(&format!(
                    "  {}: {:+}bps → {:.2}% loss\n",
                    scenario.name, scenario.rate_shock_bps, scenario.loss_percentage
                ));
            }
        }

        summary
    }

    /// Build actionable recommendations.
    pub fn build_recommendations(
        &self,
        sentiment: &SentimentScore,
        stress_scenarios: &[StressScenario],
        rebalance_suggestions: &[RebalanceSuggestion],
    ) -> Vec<String> {
        let mut recs = Vec::new();

        // Monetary policy based recommendations
        match sentiment.monetary_policy {
            MonetaryPolicy::Hawkish => {
                recs.push("MONETARY POLICY: Fed tone is hawkish. Consider reducing \
                           duration exposure and increasing cash reserves.".to_string());
                if sentiment.overall > 0.5 {
                    recs.push("RISK ALERT: Strong hawkish signal detected. Multiple rate \
                               hikes are likely — portfolio should be defensively positioned.".to_string());
                }
            }
            MonetaryPolicy::Dovish => {
                recs.push("MONETARY POLICY: Fed tone is dovish. Consider extending \
                           duration to capture bond price appreciation.".to_string());
            }
            MonetaryPolicy::Neutral => {
                recs.push("MONETARY POLICY: Fed tone is neutral. Maintain current \
                           positioning with modest hedging.".to_string());
            }
        }

        // Tone shift recommendation
        if let Some(shift) = sentiment.tone_shift {
            if shift > 0.2 {
                recs.push("TONE SHIFT: Significant hawkish shift detected. \
                           Review duration exposure immediately.".to_string());
            } else if shift < -0.2 {
                recs.push("TONE SHIFT: Significant dovish shift detected. \
                           Consider extending duration.".to_string());
            }
        }

        // Stress test warnings
        for scenario in stress_scenarios {
            if scenario.loss_percentage > 8.0 {
                recs.push(format!(
                    "STRESS WARNING: {} scenario produces {:.1}% portfolio loss. \
                     Hedging recommended.",
                    scenario.name, scenario.loss_percentage
                ));
            }
        }

        // Rebalancing suggestions
        for suggestion in rebalance_suggestions {
            recs.push(format!(
                "REBALANCE: {} — {}", suggestion.action, suggestion.rationale
            ));
        }

        if recs.is_empty() {
            recs.push("No immediate action required. Monitor Fed communications for \
                       tone changes.".to_string());
        }

        recs
    }

    /// Format a scenario analysis report as a string.
    pub fn format_scenario_report(&self, scenarios: &[StressScenario]) -> String {
        let mut report = String::new();
        report.push_str("STRESS TEST SCENARIO ANALYSIS\n");
        report.push_str(&"=".repeat(60));
        report.push('\n');

        for scenario in scenarios {
            report.push_str(&format!("\nScenario: {}\n", scenario.name));
            report.push_str(&format!("  Rate Shock:    {:+} bps\n", scenario.rate_shock_bps));
            report.push_str(&format!("  Est. Loss:     ${:.0}\n", scenario.estimated_loss));
            report.push_str(&format!("  Loss %:        {:.2}%\n", scenario.loss_percentage));
            report.push_str(&format!("  Description:   {}\n", scenario.description));
        }

        report
    }

    /// Format rebalancing suggestions as a string.
    pub fn format_rebalancing(&self, suggestions: &[RebalanceSuggestion]) -> String {
        let mut report = String::new();
        report.push_str("REBALANCING SUGGESTIONS\n");
        report.push_str(&"=".repeat(60));
        report.push('\n');

        for suggestion in suggestions {
            report.push_str(&format!("\nAsset: {}\n", suggestion.asset));
            report.push_str(&format!("  Action:       {}\n", suggestion.action));
            report.push_str(&format!("  Current:      {:.1}%\n", suggestion.current_weight * 100.0));
            report.push_str(&format!("  Suggested:    {:.1}%\n", suggestion.suggested_weight * 100.0));
            report.push_str(&format!("  Rationale:    {}\n", suggestion.rationale));
        }

        report
    }

    /// Format the verification gate status.
    pub fn format_verification_status(&self, gate: &VerificationGate) -> String {
        let mut status = String::new();
        status.push_str(&format!(
            "VERIFICATION GATE: {}\n",
            if gate.passed { "PASSED" } else { "FAILED" }
        ));
        status.push_str(&format!(
            "  Confidence:   {:.1}% (Threshold: {:.1}%)\n",
            gate.confidence * 100.0, gate.threshold * 100.0
        ));

        if !gate.warnings.is_empty() {
            status.push_str("  Warnings:\n");
            for w in &gate.warnings {
                status.push_str(&format!("    - {}\n", w));
            }
        }

        if !gate.errors.is_empty() {
            status.push_str("  Errors:\n");
            for e in &gate.errors {
                status.push_str(&format!("    - {}\n", e));
            }
        }

        status
    }

    // -- Helpers --

    fn total_notional(&self, positions: &[PortfolioPosition]) -> f64 {
        positions.iter().map(|p| p.notional).sum()
    }

    /// Generate synthetic returns for VaR estimation based on sentiment.
    pub fn generate_synthetic_returns(&self, sentiment: &SentimentScore, positions: &[PortfolioPosition])
        -> Vec<f64> {
        let (_, _, total) = self.risk_calculator.portfolio_duration_convexity(positions);
        if total == 0.0 {
            return vec![];
        }

        // Base volatility scales with duration and sentiment
        let base_vol = 0.01; // 1% daily
        let sentiment_factor = 1.0 + sentiment.overall.abs() * 0.5;

        // Generate deterministic returns based on sentiment
        // Use a seeded-like approach for reproducibility
        let mut returns = Vec::with_capacity(252);
        let drift = -sentiment.overall * 0.001; // hawkish = negative for bonds

        for i in 0..252 {
            // Pseudo-random but deterministic based on index and sentiment
            let cycle = (i as f64 * 0.1).sin() * 0.3
                + (i as f64 * 0.07).cos() * 0.2
                + (i as f64 * 0.03).sin() * 0.15;
            let vol_adjustment = base_vol * sentiment_factor;
            let ret = drift + cycle * vol_adjustment;
            returns.push(ret);
        }

        returns
    }

    fn compute_return_stats(&self, returns: &[f64]) -> (f64, f64) {
        if returns.is_empty() {
            return (0.0, 0.0);
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();
        (mean, std_dev)
    }
}

impl Default for BriefingGenerator {
    fn default() -> Self {
        Self::new(BriefingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions() -> Vec<PortfolioPosition> {
        vec![
            PortfolioPosition::new("10Y Treasury", 8.5, 75.0, 1_000_000.0, 0.5),
            PortfolioPosition::new("S&P 500 ETF", 0.0, 0.0, 1_000_000.0, 0.5),
        ]
    }

    fn hawkish_sentiment() -> SentimentScore {
        SentimentScore {
            overall: 0.6,
            inflation: 0.7,
            employment: 0.3,
            growth: 0.2,
            financial_stability: 0.1,
            monetary_policy: MonetaryPolicy::Hawkish,
            confidence: 0.85,
            tone_shift: Some(0.15),
            hawkish_keyword_count: 8,
            dovish_keyword_count: 2,
        }
    }

    fn dovish_sentiment() -> SentimentScore {
        SentimentScore {
            overall: -0.5,
            inflation: -0.4,
            employment: -0.3,
            growth: -0.4,
            financial_stability: 0.0,
            monetary_policy: MonetaryPolicy::Dovish,
            confidence: 0.8,
            tone_shift: Some(-0.3),
            hawkish_keyword_count: 2,
            dovish_keyword_count: 8,
        }
    }

    #[test]
    fn test_briefing_config_default() {
        let config = BriefingConfig::default();
        assert_eq!(config.confidence_threshold, 0.5);
        assert!(config.include_stress_tests);
        assert!(config.include_rebalancing);
    }

    #[test]
    fn test_briefing_config_new() {
        let config = BriefingConfig::new(0.8);
        assert_eq!(config.confidence_threshold, 0.8);
    }

    #[test]
    fn test_generate_briefing_hawkish() {
        let gen = BriefingGenerator::new(BriefingConfig::new(0.5));
        let date = chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap();
        let stress = vec![
            StressScenario {
                name: "Hawkish Hike".to_string(),
                rate_shock_bps: 100,
                estimated_loss: 85_000.0,
                loss_percentage: 4.25,
                description: "Test".to_string(),
            },
        ];

        let briefing = gen.generate(date, hawkish_sentiment(), &sample_positions(), stress);

        assert!(briefing.sentiment.monetary_policy == MonetaryPolicy::Hawkish);
        assert!(!briefing.summary.is_empty());
        assert!(!briefing.recommendations.is_empty());
        assert!(!briefing.var_metrics.is_empty());
        assert!(!briefing.risk_metrics.is_empty());
        assert!(briefing.verification_gate.passed);
    }

    #[test]
    fn test_generate_briefing_dovish() {
        let gen = BriefingGenerator::new(BriefingConfig::new(0.5));
        let date = chrono::NaiveDate::from_ymd_opt(2024, 12, 18).unwrap();

        let briefing = gen.generate(date, dovish_sentiment(), &sample_positions(), vec![]);

        assert!(briefing.sentiment.monetary_policy == MonetaryPolicy::Dovish);
        assert!(!briefing.summary.is_empty());
    }

    #[test]
    fn test_briefing_low_confidence_fails_gate() {
        let gen = BriefingGenerator::new(BriefingConfig::new(0.9));
        let date = chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap();
        let low_confidence = SentimentScore {
            overall: 0.3,
            confidence: 0.4,
            monetary_policy: MonetaryPolicy::Hawkish,
            ..Default::default()
        };

        let briefing = gen.generate(date, low_confidence, &sample_positions(), vec![]);

        assert!(!briefing.verification_gate.passed);
    }

    #[test]
    fn test_build_summary_contains_tone() {
        let gen = BriefingGenerator::default();
        let summary = gen.build_summary(
            &hawkish_sentiment(), &[], &[], &sample_positions(),
        );
        assert!(summary.contains("HAWKISH"));
        assert!(summary.contains("Sentiment Score"));
    }

    #[test]
    fn test_build_summary_contains_dimensions() {
        let gen = BriefingGenerator::default();
        let summary = gen.build_summary(
            &hawkish_sentiment(), &[], &[], &sample_positions(),
        );
        assert!(summary.contains("Inflation:"));
        assert!(summary.contains("Employment:"));
        assert!(summary.contains("Growth:"));
    }

    #[test]
    fn test_build_summary_tone_shift() {
        let gen = BriefingGenerator::default();
        let summary = gen.build_summary(
            &hawkish_sentiment(), &[], &[], &sample_positions(),
        );
        assert!(summary.contains("Tone Shift"));
    }

    #[test]
    fn test_build_summary_stress_scenarios() {
        let gen = BriefingGenerator::default();
        let stress = vec![
            StressScenario {
                name: "Test Scenario".to_string(),
                rate_shock_bps: 50,
                estimated_loss: 42_500.0,
                loss_percentage: 2.13,
                description: "Test".to_string(),
            },
        ];
        let summary = gen.build_summary(
            &hawkish_sentiment(), &[], &stress, &sample_positions(),
        );
        assert!(summary.contains("Test Scenario"));
    }

    #[test]
    fn test_build_recommendations_hawkish() {
        let gen = BriefingGenerator::default();
        let recs = gen.build_recommendations(&hawkish_sentiment(), &[], &[]);
        assert!(!recs.is_empty());
        assert!(recs.iter().any(|r| r.contains("hawkish") || r.contains("duration")));
    }

    #[test]
    fn test_build_recommendations_dovish() {
        let gen = BriefingGenerator::default();
        let recs = gen.build_recommendations(&dovish_sentiment(), &[], &[]);
        assert!(!recs.is_empty());
        assert!(recs.iter().any(|r| r.contains("dovish")));
    }

    #[test]
    fn test_build_recommendations_empty() {
        let gen = BriefingGenerator::default();
        let neutral = SentimentScore::neutral();
        let recs = gen.build_recommendations(&neutral, &[], &[]);
        assert!(!recs.is_empty()); // should have default recommendation
    }

    #[test]
    fn test_format_scenario_report() {
        let gen = BriefingGenerator::default();
        let scenarios = vec![
            StressScenario {
                name: "Crisis".to_string(),
                rate_shock_bps: 300,
                estimated_loss: 250_000.0,
                loss_percentage: 12.5,
                description: "Extreme scenario".to_string(),
            },
        ];
        let report = gen.format_scenario_report(&scenarios);
        assert!(report.contains("Crisis"));
        assert!(report.contains("300"));
        assert!(report.contains("12.50%"));
    }

    #[test]
    fn test_format_verification_passed() {
        let gen = BriefingGenerator::default();
        let mut gate = VerificationGate::new(0.7);
        gate.evaluate(0.85, vec!["Warning".to_string()], vec![]);
        let status = gen.format_verification_status(&gate);
        assert!(status.contains("PASSED"));
        assert!(status.contains("Warning"));
    }

    #[test]
    fn test_format_verification_failed() {
        let gen = BriefingGenerator::default();
        let mut gate = VerificationGate::new(0.9);
        gate.evaluate(0.5, vec![], vec!["Error".to_string()]);
        let status = gen.format_verification_status(&gate);
        assert!(status.contains("FAILED"));
        assert!(status.contains("Error"));
    }

    #[test]
    fn test_format_rebalancing() {
        let gen = BriefingGenerator::default();
        let suggestions = vec![
            RebalanceSuggestion {
                asset: "Fixed Income".to_string(),
                action: "Reduce Duration".to_string(),
                current_weight: 0.7,
                suggested_weight: 0.5,
                rationale: "Rate hike risk".to_string(),
            },
        ];
        let report = gen.format_rebalancing(&suggestions);
        assert!(report.contains("Reduce Duration"));
        assert!(report.contains("Rate hike risk"));
    }

    #[test]
    fn test_empty_positions_warning() {
        let gen = BriefingGenerator::new(BriefingConfig::new(0.3));
        let date = chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap();
        let briefing = gen.generate(date, SentimentScore::neutral(), &[], vec![]);
        // Should have an error about zero notional
        assert!(!briefing.verification_gate.errors.is_empty());
    }
}
