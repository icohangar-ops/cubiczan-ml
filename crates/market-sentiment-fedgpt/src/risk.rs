//! Portfolio risk calculator with VaR, stress testing, and duration analysis.

use crate::types::*;

/// Portfolio risk calculator.
#[derive(Clone)]
pub struct RiskCalculator {
    /// Confidence level for VaR calculations (e.g., 0.95).
    confidence_level: f64,
    /// Time horizon in days for VaR.
    time_horizon_days: u32,
}

impl RiskCalculator {
    pub fn new(confidence_level: f64, time_horizon_days: u32) -> Self {
        Self {
            confidence_level: confidence_level.clamp(0.5, 0.999),
            time_horizon_days,
        }
    }

    /// Calculate Historical VaR from a series of returns.
    pub fn historical_var(&self, returns: &[f64], portfolio_value: f64) -> VaRMetric {
        if returns.is_empty() {
            return VaRMetric::new(VaRMethod::Historical, self.confidence_level,
                                  self.time_horizon_days, 0.0, portfolio_value);
        }

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((1.0 - self.confidence_level) * sorted.len() as f64) as usize;
        let var_return = sorted.get(index.min(sorted.len() - 1)).copied().unwrap_or(0.0);
        let var_value = (var_return.abs() * portfolio_value).min(portfolio_value);

        // Scale for time horizon (sqrt(t))
        let scaled_var = var_value * (self.time_horizon_days as f64).sqrt();

        VaRMetric::new(VaRMethod::Historical, self.confidence_level,
                       self.time_horizon_days, scaled_var, portfolio_value)
    }

    /// Calculate Parametric VaR assuming normal distribution.
    pub fn parametric_var(&self, mean: f64, std_dev: f64, portfolio_value: f64) -> VaRMetric {
        let z_score = self.normal_z_score(self.confidence_level);
        let var_return = mean - z_score * std_dev;
        let var_value = (var_return.abs() * portfolio_value).min(portfolio_value);

        // Scale for time horizon
        let scaled_var = var_value * (self.time_horizon_days as f64).sqrt();

        VaRMetric::new(VaRMethod::Parametric, self.confidence_level,
                       self.time_horizon_days, scaled_var, portfolio_value)
    }

    /// Run stress test scenarios against portfolio positions.
    pub fn stress_test(&self, positions: &[PortfolioPosition], sentiment: &SentimentScore)
        -> Vec<StressScenario> {
        let mut scenarios = Vec::new();

        // Base scenarios defined by Fed tone
        if sentiment.monetary_policy == MonetaryPolicy::Hawkish {
            scenarios.push(self.create_rate_shock_scenario(
                positions, 100, "Hawkish Rate Hike (+100bps)",
                "Aggressive tightening scenario based on hawkish Fed tone.",
            ));
            scenarios.push(self.create_rate_shock_scenario(
                positions, 50, "Moderate Hike (+50bps)",
                "Expected rate hike consistent with hawkish guidance.",
            ));
            scenarios.push(self.create_rate_shock_scenario(
                positions, 200, "Aggressive Hike (+200bps)",
                "Tail risk scenario of double rate hike.",
            ));
        } else if sentiment.monetary_policy == MonetaryPolicy::Dovish {
            scenarios.push(self.create_rate_shock_scenario(
                positions, -50, "Dovish Rate Cut (-50bps)",
                "Rate cut scenario consistent with dovish guidance.",
            ));
            scenarios.push(self.create_rate_shock_scenario(
                positions, -25, "Modest Cut (-25bps)",
                "Standard 25bps rate cut scenario.",
            ));
            scenarios.push(self.create_rate_shock_scenario(
                positions, -100, "Aggressive Cut (-100bps)",
                "Tail risk scenario of large rate cut.",
            ));
        } else {
            scenarios.push(self.create_rate_shock_scenario(
                positions, 0, "Base Case (No Change)",
                "Rates on hold scenario consistent with neutral guidance.",
            ));
            scenarios.push(self.create_rate_shock_scenario(
                positions, 50, "Upside Surprise (+50bps)",
                "Hawkish surprise rate hike scenario.",
            ));
            scenarios.push(self.create_rate_shock_scenario(
                positions, -50, "Downside Surprise (-50bps)",
                "Dovish surprise rate cut scenario.",
            ));
        }

        // Always add extreme scenarios
        scenarios.push(self.create_rate_shock_scenario(
            positions, 300, "Crisis Hike (+300bps)",
            "Extreme crisis scenario with emergency rate hike.",
        ));
        scenarios.push(self.create_rate_shock_scenario(
            positions, -200, "Emergency Cut (-200bps)",
            "Emergency rate cut during financial crisis.",
        ));

        scenarios
    }

    /// Create a single rate shock stress scenario.
    fn create_rate_shock_scenario(
        &self,
        positions: &[PortfolioPosition],
        shock_bps: i32,
        name: &str,
        description: &str,
    ) -> StressScenario {
        let total_portfolio: f64 = positions.iter().map(|p| p.notional).sum();
        let total_loss: f64 = positions.iter()
            .map(|p| p.estimate_price_change(shock_bps as f64))
            .sum();

        let loss_percentage = if total_portfolio > 0.0 {
            (total_loss / total_portfolio) * 100.0
        } else {
            0.0
        };

        StressScenario {
            name: name.to_string(),
            rate_shock_bps: shock_bps,
            estimated_loss: total_loss.abs(),
            loss_percentage: loss_percentage.abs(),
            description: description.to_string(),
        }
    }

    /// Calculate portfolio-level duration and convexity.
    pub fn portfolio_duration_convexity(&self, positions: &[PortfolioPosition])
        -> (f64, f64, f64) {
        let total_notional: f64 = positions.iter().map(|p| p.notional).sum();

        if total_notional == 0.0 {
            return (0.0, 0.0, total_notional);
        }

        let weighted_duration: f64 = positions.iter()
            .map(|p| p.duration * (p.notional / total_notional))
            .sum();

        let weighted_convexity: f64 = positions.iter()
            .map(|p| p.convexity * (p.notional / total_notional))
            .sum();

        (weighted_duration, weighted_convexity, total_notional)
    }

    /// Estimate a simple correlation matrix from position data.
    /// Returns pairs of assets with estimated correlations based on type.
    pub fn estimate_correlations(&self, positions: &[PortfolioPosition])
        -> Vec<CorrelationEntry> {
        let mut entries = Vec::new();

        // Simple heuristic correlations based on asset types
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let corr = self.estimate_asset_correlation(
                    &positions[i].asset, &positions[j].asset,
                );
                entries.push(CorrelationEntry {
                    asset_a: positions[i].asset.clone(),
                    asset_b: positions[j].asset.clone(),
                    correlation: corr,
                });
            }
        }

        entries
    }

    /// Estimate correlation between two asset types using heuristics.
    fn estimate_asset_correlation(&self, asset_a: &str, asset_b: &str) -> f64 {
        let a_lower = asset_a.to_lowercase();
        let b_lower = asset_b.to_lowercase();

        // Both are bonds -> high correlation
        let a_is_bond = self.is_bond(&a_lower);
        let b_is_bond = self.is_bond(&b_lower);
        let a_is_equity = self.is_equity(&a_lower);
        let b_is_equity = self.is_equity(&b_lower);

        if a_is_bond && b_is_bond {
            0.85
        } else if a_is_equity && b_is_equity {
            0.75
        } else if (a_is_bond && b_is_equity) || (a_is_equity && b_is_bond) {
            -0.30
        } else {
            0.50 // default moderate correlation
        }
    }

    fn is_bond(&self, name: &str) -> bool {
        name.contains("bond") || name.contains("treasury") || name.contains("note")
            || name.contains("t-bill") || name.contains("agency")
    }

    fn is_equity(&self, name: &str) -> bool {
        name.contains("etf") || name.contains("equity") || name.contains("stock")
            || name.contains("s&p") || name.contains("index")
    }

    /// Generate portfolio rebalancing suggestions based on risk metrics and sentiment.
    pub fn suggest_rebalancing(
        &self,
        positions: &[PortfolioPosition],
        sentiment: &SentimentScore,
        stress_results: &[StressScenario],
    ) -> Vec<RebalanceSuggestion> {
        let mut suggestions = Vec::new();

        let (duration, _convexity, total) = self.portfolio_duration_convexity(positions);

        // Duration adjustment based on Fed tone
        if sentiment.monetary_policy == MonetaryPolicy::Hawkish && duration > 5.0 {
            let reduction = (duration - 5.0) / duration;
            suggestions.push(RebalanceSuggestion {
                asset: "Fixed Income".to_string(),
                action: "Reduce Duration".to_string(),
                current_weight: positions.iter().filter(|p| p.duration > 0.0)
                    .map(|p| p.weight).sum(),
                suggested_weight: positions.iter().filter(|p| p.duration > 0.0)
                    .map(|p| p.weight * (1.0 - reduction * 0.5)).sum(),
                rationale: format!(
                    "Hawkish Fed tone suggests reducing portfolio duration from {:.2} to \
                     mitigate rate hike risk. Current duration {:.2} exceeds 5.0 year target.",
                    duration, duration
                ),
            });
        } else if sentiment.monetary_policy == MonetaryPolicy::Dovish && duration < 3.0 && total > 0.0 {
            suggestions.push(RebalanceSuggestion {
                asset: "Fixed Income".to_string(),
                action: "Extend Duration".to_string(),
                current_weight: positions.iter().filter(|p| p.duration > 0.0)
                    .map(|p| p.weight).sum(),
                suggested_weight: positions.iter().filter(|p| p.duration > 0.0)
                    .map(|p| (p.weight * 1.1).min(1.0)).sum(),
                rationale: format!(
                    "Dovish Fed tone suggests extending portfolio duration from {:.2} to \
                     capture bond price appreciation. Current duration is below the 3.0 year threshold.",
                    duration
                ),
            });
        }

        // Stress test-based suggestions
        for scenario in stress_results.iter().take(2) {
            if scenario.loss_percentage > 5.0 {
                suggestions.push(RebalanceSuggestion {
                    asset: "Overall Portfolio".to_string(),
                    action: "Increase Hedging".to_string(),
                    current_weight: 1.0,
                    suggested_weight: 1.0,
                    rationale: format!(
                        "{} scenario results in {:.2}% loss ({:.0}). \
                         Consider increasing rate hedges.",
                        scenario.name, scenario.loss_percentage, scenario.estimated_loss
                    ),
                });
            }
        }

        suggestions
    }

    /// Compute basic risk metrics for the portfolio.
    pub fn compute_risk_metrics(
        &self,
        positions: &[PortfolioPosition],
        var: &VaRMetric,
    ) -> Vec<RiskMetric> {
        let mut metrics = Vec::new();

        let (duration, convexity, total) = self.portfolio_duration_convexity(positions);

        metrics.push(RiskMetric::new(
            "Portfolio Duration",
            duration,
            "years",
            if duration > 7.0 {
                "High interest rate sensitivity"
            } else if duration > 4.0 {
                "Moderate interest rate sensitivity"
            } else {
                "Low interest rate sensitivity"
            },
        ));

        metrics.push(RiskMetric::new(
            "Portfolio Convexity",
            convexity,
            "",
            if convexity > 80.0 {
                "High convexity provides protection against large rate moves"
            } else {
                "Standard convexity profile"
            },
        ));

        metrics.push(RiskMetric::new(
            "Portfolio Value",
            total,
            "USD",
            format!("Total portfolio notional: {:.0}", total).as_str(),
        ));

        metrics.push(RiskMetric::new(
            format!("Value at Risk ({:.0}%, {}d)", var.confidence_level * 100.0, var.time_horizon_days),
            var.var_value,
            "USD",
            format!("{:.2}% of portfolio value", var.var_percentage).as_str(),
        ));

        // Number of positions
        metrics.push(RiskMetric::new(
            "Position Count",
            positions.len() as f64,
            "",
            if positions.len() > 10 {
                "Well-diversified portfolio"
            } else if positions.len() > 5 {
                "Moderate diversification"
            } else {
                "Concentrated portfolio — consider diversifying"
            },
        ));

        metrics
    }

    /// Approximate z-score from confidence level using statrs.
    fn normal_z_score(&self, confidence: f64) -> f64 {
        use statrs::distribution::{Normal, ContinuousCDF};
        let _alpha: f64 = 1.0 - confidence;
        let normal = Normal::new(0.0, 1.0).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
        normal.inverse_cdf(confidence)
    }
}

impl Default for RiskCalculator {
    fn default() -> Self {
        Self::new(0.95, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions() -> Vec<PortfolioPosition> {
        vec![
            PortfolioPosition::new("10Y Treasury", 8.5, 75.0, 1_000_000.0, 0.5),
            PortfolioPosition::new("2Y Treasury", 2.0, 5.0, 500_000.0, 0.25),
            PortfolioPosition::new("S&P 500 ETF", 0.0, 0.0, 500_000.0, 0.25),
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
            tone_shift: Some(0.1),
            hawkish_keyword_count: 5,
            dovish_keyword_count: 1,
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
            tone_shift: Some(-0.2),
            hawkish_keyword_count: 1,
            dovish_keyword_count: 5,
        }
    }

    fn neutral_sentiment() -> SentimentScore {
        SentimentScore {
            overall: 0.1,
            inflation: 0.1,
            employment: 0.0,
            growth: 0.0,
            financial_stability: 0.0,
            monetary_policy: MonetaryPolicy::Neutral,
            confidence: 0.7,
            tone_shift: None,
            hawkish_keyword_count: 2,
            dovish_keyword_count: 2,
        }
    }

    #[test]
    fn test_historical_var_basic() {
        let calc = RiskCalculator::new(0.95, 1);
        let returns: Vec<f64> = vec![
            -0.02, -0.01, 0.005, 0.01, 0.03, -0.015, 0.008,
            -0.025, 0.012, -0.003, 0.007, -0.008, 0.015,
            -0.04, 0.02, -0.01, 0.003, -0.005, 0.006,
        ];
        let var = calc.historical_var(&returns, 1_000_000.0);

        assert!(var.var_value > 0.0);
        assert!(var.var_percentage > 0.0);
        assert_eq!(var.method, VaRMethod::Historical);
        assert_eq!(var.confidence_level, 0.95);
    }

    #[test]
    fn test_historical_var_empty() {
        let calc = RiskCalculator::new(0.95, 1);
        let var = calc.historical_var(&[], 1_000_000.0);
        assert_eq!(var.var_value, 0.0);
    }

    #[test]
    fn test_parametric_var() {
        let calc = RiskCalculator::new(0.95, 1);
        let var = calc.parametric_var(0.001, 0.02, 1_000_000.0);

        assert!(var.var_value > 0.0);
        assert_eq!(var.method, VaRMethod::Parametric);
    }

    #[test]
    fn test_parametric_var_high_volatility() {
        let calc = RiskCalculator::new(0.95, 1);
        let var_low = calc.parametric_var(0.001, 0.01, 1_000_000.0);
        let var_high = calc.parametric_var(0.001, 0.05, 1_000_000.0);
        assert!(var_high.var_value > var_low.var_value);
    }

    #[test]
    fn test_stress_test_hawkish() {
        let calc = RiskCalculator::new(0.95, 1);
        let scenarios = calc.stress_test(&sample_positions(), &hawkish_sentiment());

        assert!(!scenarios.is_empty());
        // Hawkish scenario should include rate hike scenarios
        assert!(scenarios.iter().any(|s| s.rate_shock_bps > 0));
    }

    #[test]
    fn test_stress_test_dovish() {
        let calc = RiskCalculator::new(0.95, 1);
        let scenarios = calc.stress_test(&sample_positions(), &dovish_sentiment());

        assert!(!scenarios.is_empty());
        // Dovish scenario should include rate cut scenarios
        assert!(scenarios.iter().any(|s| s.rate_shock_bps < 0));
    }

    #[test]
    fn test_stress_test_neutral() {
        let calc = RiskCalculator::new(0.95, 1);
        let scenarios = calc.stress_test(&sample_positions(), &neutral_sentiment());
        assert!(!scenarios.is_empty());
    }

    #[test]
    fn test_stress_scenario_results() {
        let calc = RiskCalculator::new(0.95, 1);
        let scenario = calc.create_rate_shock_scenario(
            &sample_positions(), 100, "Test", "Test scenario",
        );

        // 100bps increase should cause losses on long-duration bonds
        assert!(scenario.estimated_loss > 0.0);
        assert!(scenario.loss_percentage > 0.0);
        assert_eq!(scenario.rate_shock_bps, 100);
    }

    #[test]
    fn test_portfolio_duration_convexity() {
        let calc = RiskCalculator::new(0.95, 1);
        let (duration, convexity, total) = calc.portfolio_duration_convexity(&sample_positions());

        assert!(duration > 0.0);
        assert!(convexity > 0.0);
        assert_eq!(total, 2_000_000.0);
    }

    #[test]
    fn test_portfolio_duration_empty() {
        let calc = RiskCalculator::new(0.95, 1);
        let (duration, convexity, total) = calc.portfolio_duration_convexity(&[]);
        assert_eq!(duration, 0.0);
        assert_eq!(convexity, 0.0);
        assert_eq!(total, 0.0);
    }

    #[test]
    fn test_correlation_estimation() {
        let calc = RiskCalculator::new(0.95, 1);
        let positions = vec![
            PortfolioPosition::new("10Y Treasury", 8.0, 60.0, 100.0, 0.5),
            PortfolioPosition::new("Corporate Bond", 5.0, 40.0, 100.0, 0.5),
        ];
        let correlations = calc.estimate_correlations(&positions);
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0].correlation > 0.5); // bond-bond = 0.85
    }

    #[test]
    fn test_correlation_bond_equity() {
        let calc = RiskCalculator::new(0.95, 1);
        let positions = vec![
            PortfolioPosition::new("Treasury Bond", 5.0, 30.0, 100.0, 0.5),
            PortfolioPosition::new("S&P 500 ETF", 0.0, 0.0, 100.0, 0.5),
        ];
        let correlations = calc.estimate_correlations(&positions);
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0].correlation < 0.0); // negative bond-equity correlation
    }

    #[test]
    fn test_rebalance_suggestions_hawkish() {
        let calc = RiskCalculator::new(0.95, 1);
        let long_positions = vec![
            PortfolioPosition::new("30Y Treasury", 20.0, 300.0, 2_000_000.0, 0.8),
            PortfolioPosition::new("Cash", 0.0, 0.0, 500_000.0, 0.2),
        ];
        let suggestions = calc.suggest_rebalancing(
            &long_positions, &hawkish_sentiment(), &[],
        );
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.action.contains("Reduce Duration")));
    }

    #[test]
    fn test_rebalance_suggestions_stress() {
        let calc = RiskCalculator::new(0.95, 1);
        let stress = vec![StressScenario {
            name: "Crisis".to_string(),
            rate_shock_bps: 300,
            estimated_loss: 200_000.0,
            loss_percentage: 10.0,
            description: "Crisis scenario".to_string(),
        }];
        let suggestions = calc.suggest_rebalancing(
            &sample_positions(), &neutral_sentiment(), &stress,
        );
        assert!(suggestions.iter().any(|s| s.action.contains("Hedging")));
    }

    #[test]
    fn test_risk_metrics() {
        let calc = RiskCalculator::new(0.95, 1);
        let var = VaRMetric::new(VaRMethod::Historical, 0.95, 1, 50_000.0, 1_000_000.0);
        let metrics = calc.compute_risk_metrics(&sample_positions(), &var);
        assert!(!metrics.is_empty());
        assert!(metrics.iter().any(|m| m.name.contains("Duration")));
        assert!(metrics.iter().any(|m| m.name.contains("Value at Risk")));
    }

    #[test]
    fn test_time_horizon_scaling() {
        let calc_1d = RiskCalculator::new(0.95, 1);
        let calc_4d = RiskCalculator::new(0.95, 4);
        let returns = vec![-0.02, -0.01, 0.005, 0.01, 0.03, -0.015, 0.008, -0.025, 0.012, -0.003];
        let var_1d = calc_1d.historical_var(&returns, 1_000_000.0);
        let var_4d = calc_4d.historical_var(&returns, 1_000_000.0);
        assert!(var_4d.var_value > var_1d.var_value, "4d VaR should be > 1d VaR");
    }
}
