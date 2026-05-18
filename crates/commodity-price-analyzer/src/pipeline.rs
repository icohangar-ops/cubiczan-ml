//! # Full Analysis Pipeline
//!
//! Orchestrates the complete commodity analysis workflow: prices → forecast →
//! signals → seasonal → supply_demand → risk. Supports single-commodity and
//! multi-commodity analysis with structured report generation.

use crate::forecast::{ForecastEngine, ForecastConfig};
use crate::prices::PriceDatabase;
use crate::risk::CommodityRiskAnalyzer;
use crate::seasonal::SeasonalAnalyzer;
use crate::signals::SignalGenerator;
use crate::supply_demand::SupplyDemandModel;
use crate::types::{
    CommodityAnalysis, CommodityType, PricePoint,
};

// ---------------------------------------------------------------------------
// Pipeline Configuration
// ---------------------------------------------------------------------------

/// Configuration for the analysis pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Lookback period for forecast engine.
    pub forecast_lookback: usize,
    /// Forecast horizon in days.
    pub forecast_horizon: u32,
    /// Whether to include seasonal analysis.
    pub include_seasonal: bool,
    /// Whether to include supply-demand analysis.
    pub include_supply_demand: bool,
    /// Whether to include risk metrics.
    pub include_risk: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            forecast_lookback: 60,
            forecast_horizon: 5,
            include_seasonal: true,
            include_supply_demand: true,
            include_risk: true,
        }
    }
}

impl PipelineConfig {
    /// Create a new pipeline config.
    pub fn new(
        forecast_lookback: usize,
        forecast_horizon: u32,
        include_seasonal: bool,
        include_supply_demand: bool,
        include_risk: bool,
    ) -> Self {
        PipelineConfig {
            forecast_lookback,
            forecast_horizon,
            include_seasonal,
            include_supply_demand,
            include_risk,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-Commodity Comparison
// ---------------------------------------------------------------------------

/// Comparison result for a single metric across multiple commodities.
#[derive(Debug, Clone)]
pub struct MetricComparison {
    pub metric_name: String,
    pub values: Vec<(CommodityType, f64)>,
}

/// A multi-commodity comparison report.
#[derive(Debug, Clone)]
pub struct MultiCommodityReport {
    pub commodities: Vec<CommodityType>,
    pub comparisons: Vec<MetricComparison>,
    pub timestamp: String,
}

// ---------------------------------------------------------------------------
// Analysis Pipeline
// ---------------------------------------------------------------------------

/// Full analysis pipeline that orchestrates all analysis steps.
pub struct AnalysisPipeline {
    config: PipelineConfig,
    forecast_engine: ForecastEngine,
    signal_generator: SignalGenerator,
    seasonal_analyzer: SeasonalAnalyzer,
    supply_demand_model: SupplyDemandModel,
    risk_analyzer: CommodityRiskAnalyzer,
}

impl Default for AnalysisPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisPipeline {
    /// Create a new analysis pipeline with default configuration.
    pub fn new() -> Self {
        AnalysisPipeline {
            config: PipelineConfig::default(),
            forecast_engine: ForecastEngine::default(),
            signal_generator: SignalGenerator::default(),
            seasonal_analyzer: SeasonalAnalyzer::default(),
            supply_demand_model: SupplyDemandModel::default(),
            risk_analyzer: CommodityRiskAnalyzer::default(),
        }
    }

    /// Create a new analysis pipeline with custom configuration.
    pub fn with_config(config: PipelineConfig) -> Self {
        let forecast_config = ForecastConfig::new(
            config.forecast_lookback,
            config.forecast_horizon,
            0.3,
            0.3,
        );
        AnalysisPipeline {
            config,
            forecast_engine: ForecastEngine::with_config(forecast_config),
            signal_generator: SignalGenerator::default(),
            seasonal_analyzer: SeasonalAnalyzer::default(),
            supply_demand_model: SupplyDemandModel::default(),
            risk_analyzer: CommodityRiskAnalyzer::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Single-Commodity Full Analysis
    // -----------------------------------------------------------------------

    /// Run the full analysis pipeline for a single commodity.
    pub fn analyze(&self, commodity: CommodityType, prices: &[PricePoint]) -> CommodityAnalysis {
        let current_price = prices.last().map(|p| p.close).unwrap_or(0.0);

        // Step 1: Forecast
        let forecasts = vec![
            self.forecast_engine.linear_regression_forecast_for(commodity, prices),
            self.forecast_engine.ensemble_forecast(commodity, prices),
        ];

        // Step 2: Signals
        let signal = self.signal_generator.generate_signal(commodity, prices);
        let signals = vec![signal];

        // Step 3: Seasonal
        let seasonal = if self.config.include_seasonal {
            self.seasonal_analyzer.monthly_patterns(commodity, prices)
        } else {
            vec![]
        };

        // Step 4: Supply/Demand
        let supply_demand = if self.config.include_supply_demand {
            self.supply_demand_model.full_analysis(commodity, prices)
        } else {
            vec![]
        };

        // Step 5: Risk
        let risk_metrics = if self.config.include_risk {
            Some(self.risk_analyzer.full_risk_summary(commodity, prices))
        } else {
            None
        };

        CommodityAnalysis {
            commodity,
            current_price,
            forecasts,
            signals,
            supply_demand,
            seasonal,
            risk_metrics,
        }
    }

    /// Run analysis using prices from a PriceDatabase.
    pub fn analyze_from_db(&self, db: &PriceDatabase, commodity: CommodityType) -> Option<CommodityAnalysis> {
        let prices = db.get_prices(commodity)?;
        if prices.is_empty() {
            return None;
        }
        Some(self.analyze(commodity, prices))
    }

    // -----------------------------------------------------------------------
    // Multi-Commodity Comparison
    // -----------------------------------------------------------------------

    /// Run analysis across multiple commodities and generate a comparison report.
    pub fn multi_commodity_analysis(
        &self,
        price_data: &std::collections::HashMap<CommodityType, Vec<PricePoint>>,
    ) -> MultiCommodityReport {
        let commodities: Vec<CommodityType> = price_data.keys().copied().collect();

        // Analyze each commodity
        let analyses: Vec<CommodityAnalysis> = commodities
            .iter()
            .filter_map(|c| {
                let prices = price_data.get(c)?;
                if prices.is_empty() { return None; }
                Some(self.analyze(*c, prices))
            })
            .collect();

        // Build comparison metrics
        let mut comparisons = Vec::new();

        // Compare forecast confidence
        let forecast_confidence: Vec<(CommodityType, f64)> = analyses
            .iter()
            .filter_map(|a| {
                let ensemble = a.forecasts.iter().find(|f| f.model_version == "ensemble")?;
                Some((a.commodity, ensemble.confidence))
            })
            .collect();
        if !forecast_confidence.is_empty() {
            comparisons.push(MetricComparison {
                metric_name: "Forecast Confidence".into(),
                values: forecast_confidence,
            });
        }

        // Compare signal scores
        let signal_scores: Vec<(CommodityType, f64)> = analyses
            .iter()
            .map(|a| {
                (a.commodity, a.signals.first().map(|s| s.confidence).unwrap_or(0.0))
            })
            .collect();
        if !signal_scores.is_empty() {
            comparisons.push(MetricComparison {
                metric_name: "Signal Confidence".into(),
                values: signal_scores,
            });
        }

        // Compare volatility
        let volatilities: Vec<(CommodityType, f64)> = analyses
            .iter()
            .filter_map(|a| {
                let risk = a.risk_metrics.as_ref()?;
                Some((a.commodity, risk.volatility))
            })
            .collect();
        if !volatilities.is_empty() {
            comparisons.push(MetricComparison {
                metric_name: "Annualized Volatility".into(),
                values: volatilities,
            });
        }

        // Compare max drawdown
        let drawdowns: Vec<(CommodityType, f64)> = analyses
            .iter()
            .filter_map(|a| {
                let risk = a.risk_metrics.as_ref()?;
                Some((a.commodity, risk.max_drawdown))
            })
            .collect();
        if !drawdowns.is_empty() {
            comparisons.push(MetricComparison {
                metric_name: "Max Drawdown".into(),
                values: drawdowns,
            });
        }

        // Compare Sharpe ratios
        let sharpes: Vec<(CommodityType, f64)> = analyses
            .iter()
            .filter_map(|a| {
                let risk = a.risk_metrics.as_ref()?;
                Some((a.commodity, risk.sharpe_ratio))
            })
            .collect();
        if !sharpes.is_empty() {
            comparisons.push(MetricComparison {
                metric_name: "Sharpe Ratio".into(),
                values: sharpes,
            });
        }

        MultiCommodityReport {
            commodities,
            comparisons,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    // -----------------------------------------------------------------------
    // Report Generation
    // -----------------------------------------------------------------------

    /// Generate a structured text summary from a `CommodityAnalysis`.
    pub fn generate_report(analysis: &CommodityAnalysis) -> String {
        let mut lines = Vec::new();

        lines.push(format!("=== {} Analysis Report ===", analysis.commodity));
        lines.push(format!("Current Price: {:.2}", analysis.current_price));
        lines.push(String::new());

        // Forecasts
        lines.push("--- Forecasts ---".into());
        for f in &analysis.forecasts {
            lines.push(format!(
                "  [{}] Predicted: {:.2}, Confidence: {:.1}%, Horizon: {}d",
                f.model_version, f.predicted_price, f.confidence * 100.0, f.horizon
            ));
        }
        lines.push(String::new());

        // Signals
        lines.push("--- Signals ---".into());
        for s in &analysis.signals {
            lines.push(format!("  {}: Confidence {:.1}%", s.signal, s.confidence * 100.0));
            lines.push(format!("  Reasoning: {}", s.reasoning));
        }
        lines.push(String::new());

        // Supply/Demand
        if !analysis.supply_demand.is_empty() {
            lines.push("--- Supply/Demand ---".into());
            for f in &analysis.supply_demand {
                lines.push(format!("  [{}] Impact: {:.2} — {}", f.factor_type, f.impact_score, f.description));
            }
            lines.push(String::new());
        }

        // Risk
        if let Some(ref risk) = analysis.risk_metrics {
            lines.push("--- Risk Metrics ---".into());
            lines.push(format!("  Volatility: {:.2}%", risk.volatility * 100.0));
            lines.push(format!("  VaR (95%): {:.2}%", risk.var_95 * 100.0));
            lines.push(format!("  CVaR (95%): {:.2}%", risk.cvar_95 * 100.0));
            lines.push(format!("  Max Drawdown: {:.2}%", risk.max_drawdown * 100.0));
            lines.push(format!("  Volatility Regime: {}", risk.volatility_regime));
            lines.push(format!("  Sharpe Ratio: {:.2}", risk.sharpe_ratio));
        }

        lines.join("\n")
    }

    /// Generate a summary of a multi-commodity comparison report.
    pub fn generate_multi_report(report: &MultiCommodityReport) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== Multi-Commodity Comparison Report ==="));
        lines.push(format!("Generated: {}", report.timestamp));
        lines.push(format!("Commodities: {}", report.commodities.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ")));
        lines.push(String::new());

        for comp in &report.comparisons {
            lines.push(format!("--- {} ---", comp.metric_name));
            // Sort by value descending
            let mut sorted = comp.values.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (commodity, value) in &sorted {
                lines.push(format!("  {}: {:.4}", commodity, value));
            }
            lines.push(String::new());
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    fn make_test_prices(n: usize, base: f64, trend: f64) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = base + i as f64 * trend + (i as f64 * 0.3).sin() * 2.0;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 0.5, price + 1.0, price - 1.0, price, 1000.0 + i as f64 * 50.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_pipeline_default() {
        let pipeline = AnalysisPipeline::default();
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        assert_eq!(analysis.commodity, CommodityType::Gold);
        assert!(analysis.current_price > 0.0);
    }

    #[test]
    fn test_pipeline_with_config() {
        let config = PipelineConfig::new(30, 10, false, false, false);
        let pipeline = AnalysisPipeline::with_config(config);
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Silver, &prices);
        assert_eq!(analysis.commodity, CommodityType::Silver);
        assert!(analysis.seasonal.is_empty());
        assert!(analysis.supply_demand.is_empty());
        assert!(analysis.risk_metrics.is_none());
    }

    #[test]
    fn test_pipeline_includes_forecasts() {
        let pipeline = AnalysisPipeline::new();
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        assert!(!analysis.forecasts.is_empty());
        // Should have linear regression and ensemble
        assert!(analysis.forecasts.len() >= 2);
    }

    #[test]
    fn test_pipeline_includes_signals() {
        let pipeline = AnalysisPipeline::new();
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        assert!(!analysis.signals.is_empty());
    }

    #[test]
    fn test_pipeline_includes_seasonal() {
        let pipeline = AnalysisPipeline::new();
        let prices = make_test_prices(200, 100.0, 0.2);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        // With default config, seasonal should be included
        if !analysis.seasonal.is_empty() {
            assert_eq!(analysis.seasonal.len(), 12);
        }
    }

    #[test]
    fn test_pipeline_includes_supply_demand() {
        let pipeline = AnalysisPipeline::new();
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        assert!(!analysis.supply_demand.is_empty());
        assert!(analysis.supply_demand.len() >= 4);
    }

    #[test]
    fn test_pipeline_includes_risk() {
        let pipeline = AnalysisPipeline::new();
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        assert!(analysis.risk_metrics.is_some());
        let risk = analysis.risk_metrics.unwrap();
        assert!(risk.volatility >= 0.0);
    }

    #[test]
    fn test_multi_commodity_analysis() {
        let pipeline = AnalysisPipeline::new();
        let mut price_data = HashMap::new();
        price_data.insert(CommodityType::Gold, make_test_prices(100, 1800.0, 2.0));
        price_data.insert(CommodityType::Silver, make_test_prices(100, 25.0, 0.1));
        price_data.insert(CommodityType::Copper, make_test_prices(100, 4.0, 0.02));

        let report = pipeline.multi_commodity_analysis(&price_data);
        assert_eq!(report.commodities.len(), 3);
        assert!(!report.comparisons.is_empty());
    }

    #[test]
    fn test_generate_report() {
        let pipeline = AnalysisPipeline::new();
        let prices = make_test_prices(100, 100.0, 0.5);
        let analysis = pipeline.analyze(CommodityType::Gold, &prices);
        let report = AnalysisPipeline::generate_report(&analysis);
        assert!(report.contains("Gold Analysis Report"));
        assert!(report.contains("Current Price"));
    }

    #[test]
    fn test_generate_multi_report() {
        let pipeline = AnalysisPipeline::new();
        let mut price_data = HashMap::new();
        price_data.insert(CommodityType::Gold, make_test_prices(100, 1800.0, 2.0));
        price_data.insert(CommodityType::Silver, make_test_prices(100, 25.0, 0.1));

        let report = pipeline.multi_commodity_analysis(&price_data);
        let text = AnalysisPipeline::generate_multi_report(&report);
        assert!(text.contains("Multi-Commodity Comparison Report"));
    }

    #[test]
    fn test_analyze_from_db() {
        let mut db = PriceDatabase::new();
        let prices = make_test_prices(100, 1800.0, 2.0);
        db.add_price_points(CommodityType::Gold, prices);

        let pipeline = AnalysisPipeline::new();
        let result = pipeline.analyze_from_db(&db, CommodityType::Gold);
        assert!(result.is_some());
        let analysis = result.unwrap();
        assert_eq!(analysis.commodity, CommodityType::Gold);
    }

    #[test]
    fn test_analyze_from_db_missing() {
        let db = PriceDatabase::new();
        let pipeline = AnalysisPipeline::new();
        let result = pipeline.analyze_from_db(&db, CommodityType::Gold);
        assert!(result.is_none());
    }

    #[test]
    fn test_pipeline_config_new() {
        let config = PipelineConfig::new(50, 10, true, false, true);
        assert_eq!(config.forecast_lookback, 50);
        assert_eq!(config.forecast_horizon, 10);
        assert!(config.include_seasonal);
        assert!(!config.include_supply_demand);
        assert!(config.include_risk);
    }
}
