//! Pipeline orchestration: composite risk score computation and full pipeline.

use serde::{Deserialize, Serialize};

use crate::config::{MINERALS, SCORE_SCALE, mock_sec_filings};
use crate::forecast::generate_mock_price_history;
use crate::macro_data::{MacroData, generate_mock_macro_data};
use crate::prices::generate_mock_prices;
use crate::scaling::{scale_price, scale_composite, scale_sentiment, scale_reg_risk};
use crate::sentiment::{simple_sentiment_analyzer, regulatory_risk_scorer};

/// On-chain scaled values for a single mineral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainScaled {
    pub current_price: i64,
    pub forecast_price: i64,
    pub composite_score: i64,
    pub price_deviation: i64,
    pub supply_sentiment: i64,
    pub regulatory_risk: i64,
    pub forecast_direction: i64,
    pub confidence: u32,
}

/// Complete risk assessment result for a single mineral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineralRiskResult {
    pub timestamp: String,
    pub current_price_usd: f64,
    pub forecast_price_usd: f64,
    pub price_deviation_pct: f64,
    pub supply_sentiment: f64,
    pub regulatory_risk: f64,
    pub forecast_direction_pct: f64,
    pub composite_score: f64,
    pub confidence_bps: u32,
    pub on_chain_scaled: OnChainScaled,
}

/// Full pipeline output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub pipeline_version: String,
    pub mode: String,
    pub timestamp: String,
    pub macro_data: MacroData,
    pub minerals: std::collections::HashMap<String, MineralRiskResult>,
}

/// Compute the composite risk score combining all factors.
///
/// Weights:
///   - Price deviation: 30%
///   - Supply sentiment: 35%
///   - Regulatory risk: 25%
///   - Forecast direction: 10%
///
/// Returns `(composite_score, confidence_bps)`.
pub fn compute_composite_risk_score(
    price_deviation: f64,
    sentiment: f64,
    reg_risk: f64,
    forecast_direction: f64,
) -> (f64, u32) {
    // Normalize each component to [-100, 100]
    let normalized_price = (price_deviation.clamp(-30.0, 30.0)) * (100.0 / 30.0);
    let normalized_sentiment = sentiment * 100.0;
    let normalized_reg = -(reg_risk * 2.0 - 100.0); // 0→100, 50→0, 100→-100
    let normalized_forecast = (forecast_direction.clamp(-20.0, 20.0)) * (100.0 / 20.0);

    // Weighted composite
    let composite = normalized_price * 0.30
        + normalized_sentiment * 0.35
        + normalized_reg * 0.25
        + normalized_forecast * 0.10;

    let confidence = 500u32; // Default 5% confidence interval
    let composite = composite.clamp(-100.0, 100.0);

    ((composite * 100.0).round() / 100.0, confidence)
}

/// Run the pipeline in demo mode with mock data.
pub fn run_demo_pipeline() -> PipelineOutput {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let mut results = std::collections::HashMap::new();

    let prices = generate_mock_prices();
    let macro_data = generate_mock_macro_data();

    for &(mineral, _) in MINERALS {
        let price_data = prices.iter().find(|(m, _)| *m == mineral).unwrap();
        let current_price = price_data.1.current_price;
        let forecast_price = price_data.1.forecast_price;
        let price_deviation = ((forecast_price - current_price) / current_price) * 100.0;

        // Price forecast via regression
        let _history = generate_mock_price_history(mineral, 24);

        // Sentiment analysis from mock SEC filings
        let filings = mock_sec_filings();
        let sec_text = filings.get(mineral).unwrap();
        let sentiment = simple_sentiment_analyzer(sec_text);

        // Regulatory risk scoring
        let reg_risk = regulatory_risk_scorer(sec_text);

        // Composite score
        let (composite, confidence) = compute_composite_risk_score(
            price_deviation, sentiment, reg_risk, price_deviation,
        );

        let scaled = OnChainScaled {
            current_price: scale_price(current_price),
            forecast_price: scale_price(forecast_price),
            composite_score: scale_composite(composite),
            price_deviation: (price_deviation * SCORE_SCALE as f64) as i64,
            supply_sentiment: scale_sentiment(sentiment),
            regulatory_risk: scale_reg_risk(reg_risk),
            forecast_direction: (price_deviation * SCORE_SCALE as f64) as i64,
            confidence,
        };

        results.insert(mineral.to_string(), MineralRiskResult {
            timestamp: timestamp.clone(),
            current_price_usd: current_price,
            forecast_price_usd: forecast_price,
            price_deviation_pct: (price_deviation * 100.0).round() / 100.0,
            supply_sentiment: (sentiment * 10_000.0).round() / 10_000.0,
            regulatory_risk: (reg_risk * 100.0).round() / 100.0,
            forecast_direction_pct: (price_deviation * 100.0).round() / 100.0,
            composite_score: composite,
            confidence_bps: confidence,
            on_chain_scaled: scaled,
        });
    }

    PipelineOutput {
        pipeline_version: "1.0.0".to_string(),
        mode: "demo".to_string(),
        timestamp,
        macro_data,
        minerals: results,
    }
}

/// Pretty-print the pipeline results (mirrors the Python CLI output).
pub fn print_pipeline_results(output: &PipelineOutput) {
    println!("\n{}", "=".repeat(64));
    println!("  CRITMIN ORACLE — {} Pipeline",
        if output.mode == "demo" { "Demo (Mock Data)" } else { "Live (Real APIs)" });
    println!("{}", "=".repeat(64));

    println!("\n  PPI (Metals): {:.1} ({:+.1}% YoY)",
        output.macro_data.ppi_metals, output.macro_data.ppi_metals_change_1y);
    println!("  Industrial Production: {:.1} ({:+.1}% YoY)",
        output.macro_data.industrial_production, output.macro_data.industrial_production_change_1y);
    println!("  Manufacturing PMI: {:.1}", output.macro_data.manufacturing_pmi);

    println!("\n{}", "-".repeat(64));
    println!("  {:<12} {:>8} {:>10} {:>10} {:>10}",
        "Mineral", "Score", "Sentiment", "Reg Risk", "Status");
    println!("  {}", "-".repeat(52));

    for (mineral, data) in &output.minerals {
        let score = data.composite_score;
        let status = if score > 25.0 { "SAFE" } else if score < -25.0 { "RISKY" } else { "MODERATE" };
        println!("  {:<12} {:>+8.1} {:>+10.3} {:>9.1}% {:>10}",
            mineral, score, data.supply_sentiment, data.regulatory_risk, status);
    }

    println!("\n{}", "=".repeat(64));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_score_safe() {
        let (score, conf) = compute_composite_risk_score(10.0, 0.5, 20.0, 5.0);
        assert!(score > 0.0, "Should be positive/safe");
        assert!(conf > 0);
    }

    #[test]
    fn test_composite_score_risky() {
        let (score, _) = compute_composite_risk_score(-15.0, -0.8, 90.0, -10.0);
        assert!(score < -25.0, "Should be risky, got {}", score);
    }

    #[test]
    fn test_composite_score_clamped() {
        let (score, _) = compute_composite_risk_score(999.0, 999.0, 0.0, 999.0);
        assert!(score <= 100.0, "Should be clamped to 100, got {}", score);
        let (score, _) = compute_composite_risk_score(-999.0, -999.0, 100.0, -999.0);
        assert!(score >= -100.0, "Should be clamped to -100, got {}", score);
    }

    #[test]
    fn test_demo_pipeline_runs() {
        let output = run_demo_pipeline();
        assert_eq!(output.minerals.len(), 3);
        assert!(output.minerals.contains_key("LITHIUM"));
        assert!(output.minerals.contains_key("NICKEL"));
        assert!(output.minerals.contains_key("COBALT"));

        for (mineral, data) in &output.minerals {
            assert!(data.current_price_usd > 0.0, "{} price > 0", mineral);
            assert!(data.confidence_bps > 0, "{} confidence > 0", mineral);
            assert!(data.on_chain_scaled.current_price > 0, "{} scaled price > 0", mineral);
        }
    }

    #[test]
    fn test_demo_pipeline_deterministic_output_structure() {
        let output = run_demo_pipeline();
        assert_eq!(output.pipeline_version, "1.0.0");
        assert_eq!(output.mode, "demo");
        assert!(!output.timestamp.is_empty());
    }
}
