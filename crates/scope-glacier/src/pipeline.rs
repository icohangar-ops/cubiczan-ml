//! End-to-end integration pipeline: ingest → timeseries → analytics → forecast.

use crate::analytics::{
    analyze_spot_prices, compute_volatility_surface, generate_price_signals,
    score_imbalance, ImbalanceScore, SpotPriceAnalysis, VolatilityPoint,
};
use crate::forecast::{
    backtest_forecast,
};
use crate::ingest::{
    normalize_timeseries, parse_eia_csv, parse_eia_json, remove_outliers_iqr,
    validate_price_series, IngestFormat, IngestStats,
};
use crate::timeseries::{
    adf_stationarity_test, autocorrelation, seasonal_decomposition,
};
use crate::types::*;

use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// The result of a complete pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub config: PipelineConfig,
    pub ingest_stats: IngestStats,
    pub spot_analysis: SpotPriceAnalysis,
    pub volatility_surface: Vec<VolatilityPoint>,
    pub imbalance_score: Option<ImbalanceScore>,
    pub forecast: ForecastResult,
    pub signals: Vec<MarketSignal>,
    pub seasonal_strength: f64,
    pub is_stationary: bool,
    pub n_raw_points: usize,
    pub n_clean_points: usize,
    pub n_normalized: usize,
}

/// Runs the full end-to-end analytics pipeline on a JSON data feed.
pub fn run_pipeline_json(
    json_data: &str,
    sd_records: &[SupplyDemandRecord],
    config: PipelineConfig,
) -> Result<PipelineResult> {
    let price_points = parse_eia_json(json_data, config.commodity)?;

    // If no data from JSON, create synthetic for testing
    if price_points.is_empty() {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    run_pipeline_from_points(&price_points, sd_records, config)
}

/// Runs the full end-to-end analytics pipeline on a CSV data feed.
pub fn run_pipeline_csv(
    csv_data: &str,
    sd_records: &[SupplyDemandRecord],
    config: PipelineConfig,
) -> Result<PipelineResult> {
    let price_points = parse_eia_csv(csv_data, config.commodity)?;

    if price_points.is_empty() {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    run_pipeline_from_points(&price_points, sd_records, config)
}

/// Core pipeline logic operating on parsed PricePoints.
fn run_pipeline_from_points(
    price_points: &[PricePoint],
    sd_records: &[SupplyDemandRecord],
    config: PipelineConfig,
) -> Result<PipelineResult> {
    // Phase 1: Ingest and validate
    let n_raw_points = price_points.len();
    let (valid_points, ingest_stats) = validate_price_series(price_points);

    if valid_points.is_empty() {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let n_clean_points = valid_points.len();

    // Phase 2: Normalize and clean
    let mut normalized = normalize_timeseries(&valid_points)?;
    let _outliers_removed = remove_outliers_iqr(&mut normalized, 1.5);
    let n_normalized = normalized.len();

    if n_normalized < 10 {
        return Err(GlacierError::InsufficientData {
            required: 10,
            actual: n_normalized,
        });
    }

    let prices: Vec<f64> = normalized.iter().map(|r| r.value).collect();

    // Phase 3: Time-series analysis
    let (_, is_stationary) = adf_stationarity_test(&prices, 0.05);

    let period = estimate_seasonal_period(&prices);
    let timestamps: Vec<DateTime<Utc>> = normalized.iter().map(|r| r.timestamp).collect();

    let seasonal_strength = if period > 0 && prices.len() >= 2 * period {
        seasonal_decomposition(&prices, period, timestamps.clone())
            .map(|d| d.seasonal_strength())
            .unwrap_or(0.0)
    } else {
        0.0
    };

    // Phase 4: Analytics
    let spot_analysis = analyze_spot_prices(&prices, config.commodity)?;

    let vol_surface =
        compute_volatility_surface(&prices, &[5, 10, 20, 50, 100]).unwrap_or_default();

    let imbalance_score = if !sd_records.is_empty() {
        score_imbalance(sd_records).ok()
    } else {
        None
    };

    let signals = generate_price_signals(&spot_analysis);

    // Phase 5: Forecasting
    let forecast = backtest_forecast(
        &prices,
        0.8,
        config.model,
        config.forecast_horizon as usize,
        config.commodity,
    )?;

    Ok(PipelineResult {
        config,
        ingest_stats,
        spot_analysis,
        volatility_surface: vol_surface,
        imbalance_score,
        forecast,
        signals,
        seasonal_strength,
        is_stationary,
        n_raw_points,
        n_clean_points,
        n_normalized,
    })
}

/// Estimates the dominant seasonal period from autocorrelation.
fn estimate_seasonal_period(prices: &[f64]) -> usize {
    if prices.len() < 10 {
        return 0;
    }

    let max_lag = prices.len().min(50);
    let acf = match autocorrelation(prices, max_lag) {
        Ok(a) => a,
        Err(_) => return 0,
    };

    // Find first significant peak after lag 1
    let mut best_lag = 0;
    let mut best_acf = 0.0_f64;

    for lag in 2..acf.len() {
        // A peak is where ACF[i] > ACF[i-1] and ACF[i] > ACF[i+1]
        if lag + 1 < acf.len()
            && acf[lag] > acf[lag - 1]
            && acf[lag] > acf[lag + 1]
            && acf[lag] > best_acf
            && acf[lag] > 0.3
        {
            best_lag = lag;
            best_acf = acf[lag];
        }
    }

    best_lag
}

/// Generates a comprehensive summary report from pipeline results.
pub fn generate_report(result: &PipelineResult) -> String {
    let mut lines = Vec::new();

    lines.push(format!(
        "=== Scope-Glacier Pipeline Report ==="
    ));
    lines.push(format!("Commodity: {}", result.config.commodity));
    lines.push(format!("Model: {}", result.config.model));
    lines.push(String::new());
    lines.push(format!("--- Ingestion ---"));
    lines.push(format!(
        "Raw points: {}, Clean: {}, Normalized: {}",
        result.n_raw_points, result.n_clean_points, result.n_normalized
    ));
    lines.push(format!("{}", result.ingest_stats.summary()));
    lines.push(String::new());
    lines.push(format!("--- Spot Price Analysis ---"));
    lines.push(format!("Current: {:.2} {}", result.spot_analysis.current_price, result.config.commodity.unit()));
    lines.push(format!("Mean: {:.2}, Median: {:.2}", result.spot_analysis.mean_price, result.spot_analysis.median_price));
    lines.push(format!("Std Dev: {:.2}, CV: {:.2}%", result.spot_analysis.std_deviation, result.spot_analysis.cv * 100.0));
    lines.push(format!("Range: [{:.2}, {:.2}]", result.spot_analysis.min_price, result.spot_analysis.max_price));
    lines.push(format!("Z-Score: {:.2}, Percentile: {:.1}%", result.spot_analysis.z_score, result.spot_analysis.percentile_rank));
    lines.push(format!("Trend: {}", result.spot_analysis.trend_direction));
    lines.push(String::new());
    lines.push(format!("--- Forecast ---"));
    lines.push(format!(
        "Horizon: {} steps, MAE: {:.4}, RMSE: {:.4}, MAPE: {:.2}%",
        result.forecast.len(),
        result.forecast.mae,
        result.forecast.rmse,
        result.forecast.mape
    ));
    if !result.forecast.predictions.is_empty() {
        lines.push(format!(
            "Next period estimate: {:.2} [{:.2}, {:.2}]",
            result.forecast.predictions[0],
            result.forecast.lower_bound[0],
            result.forecast.upper_bound[0]
        ));
    }
    lines.push(String::new());
    lines.push(format!("--- Market Signals ---"));
    if result.signals.is_empty() {
        lines.push("No signals generated.".to_string());
    } else {
        for sig in &result.signals {
            lines.push(format!(
                "[{}] {} (strength: {:.2}): {}",
                sig.signal_type, sig.commodity, sig.strength, sig.description
            ));
        }
    }

    if let Some(ref imb) = result.imbalance_score {
        lines.push(String::new());
        lines.push(format!("--- Supply/Demand ---"));
        lines.push(format!("Imbalance score: {:.3} ({:?})", imb.current_imbalance, imb.severity));
        lines.push(format!("Recommendation: {}", imb.recommendation));
    }

    lines.push(String::new());
    lines.push(format!("--- Time Series Properties ---"));
    lines.push(format!("Stationary: {}", result.is_stationary));
    lines.push(format!("Seasonal strength: {:.3}", result.seasonal_strength));

    lines.join("\n")
}

/// Multi-commodity batch pipeline that processes several commodities in parallel (sequential here).
pub fn run_multi_commodity_pipeline(
    feeds: &HashMap<EnergyCommodity, String>,
    sd_records: &HashMap<EnergyCommodity, Vec<SupplyDemandRecord>>,
    format: IngestFormat,
    config: &PipelineConfig,
) -> HashMap<EnergyCommodity, Result<PipelineResult>> {
    let mut results = HashMap::new();

    for (commodity, data) in feeds {
        let cfg = PipelineConfig {
            commodity: *commodity,
            ..config.clone()
        };

        let sd = sd_records.get(commodity).map(|v| v.as_slice()).unwrap_or(&[]);

        let result = match format {
            IngestFormat::Json => run_pipeline_json(data, sd, cfg),
            IngestFormat::Csv => run_pipeline_csv(data, sd, cfg),
        };

        results.insert(*commodity, result);
    }

    results
}

/// Quick analysis shortcut: takes raw values and returns a basic analysis.
pub fn quick_analysis(
    prices: &[f64],
    commodity: EnergyCommodity,
) -> Result<(SpotPriceAnalysis, Vec<MarketSignal>)> {
    if prices.is_empty() {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let analysis = analyze_spot_prices(prices, commodity)?;
    let signals = generate_price_signals(&analysis);

    Ok((analysis, signals))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_json(n: usize) -> String {
        let data_points: Vec<String> = (0..n)
            .map(|i| {
                let day = 1 + i % 28;
                let month = 1 + (i / 28) % 12;
                let price = 70.0 + i as f64 * 0.3 + (i as f64 * 0.5).sin() * 3.0;
                format!(
                    r#"{{"period": "2023-{:02}-{:02}", "value": {:.2}}}"#,
                    month, day, price
                )
            })
            .collect();
        format!(
            r#"{{"response": {{"data": [{{"series_id": "TEST", "units": "$/barrel", "data": [{}]}}]}}}}"#,
            data_points.join(", ")
        )
    }

    fn make_test_csv(n: usize) -> String {
        let mut lines = vec!["period,price,volume".to_string()];
        for i in 0..n {
            let day = 1 + i % 28;
            let month = 1 + (i / 28) % 12;
            let price = 70.0 + i as f64 * 0.3 + (i as f64 * 0.5).sin() * 3.0;
            lines.push(format!("2023-{:02}-{:02},{:.2},{}", month, day, price, 1000 + i));
        }
        lines.join("\n")
    }

    fn make_sd_records(n: usize) -> Vec<SupplyDemandRecord> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let mut rec = SupplyDemandRecord::new(now - chrono::Duration::days(i as i64), EnergyCommodity::CrudeOil);
                rec.supply_mmbtu = Some(100.0 + (i as f64 * 0.5).sin() * 10.0);
                rec.demand_mmbtu = Some(95.0 + (i as f64 * 0.3).cos() * 8.0);
                rec.inventory_level = Some(500.0);
                rec.consumption_rate = Some(50.0);
                rec
            })
            .collect()
    }

    #[test]
    fn test_run_pipeline_json() {
        let json = make_test_json(60);
        let sd = make_sd_records(20);
        let config = PipelineConfig::new(EnergyCommodity::CrudeOil)
            .with_model(ForecastModel::Regression);

        let result = run_pipeline_json(&json, &sd, config).unwrap();
        assert_eq!(result.n_raw_points, 60);
        assert!(result.n_clean_points > 0);
        assert!(result.n_normalized > 0);
        assert!(result.forecast.len() > 0);
        assert!(!result.signals.is_empty());
    }

    #[test]
    fn test_run_pipeline_csv() {
        let csv = make_test_csv(60);
        let config = PipelineConfig::new(EnergyCommodity::CrudeOil)
            .with_model(ForecastModel::HoltWinters);

        let result = run_pipeline_csv(&csv, &[], config).unwrap();
        assert!(result.n_raw_points > 0);
        assert!(!result.forecast.predictions.is_empty());
    }

    #[test]
    fn test_run_pipeline_insufficient_data() {
        let json = make_test_json(3); // Too few points
        let config = PipelineConfig::new(EnergyCommodity::CrudeOil);
        let result = run_pipeline_json(&json, &[], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_pipeline_empty_data() {
        let json = r#"{"response": {"data": []}}"#;
        let config = PipelineConfig::new(EnergyCommodity::CrudeOil);
        let result = run_pipeline_json(json, &[], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_report() {
        let json = make_test_json(60);
        let sd = make_sd_records(20);
        let config = PipelineConfig::new(EnergyCommodity::CrudeOil);
        let result = run_pipeline_json(&json, &sd, config).unwrap();
        let report = generate_report(&result);

        assert!(report.contains("Scope-Glacier"));
        assert!(report.contains("CrudeOil") || report.contains("WTI"));
        assert!(report.contains("MAE"));
    }

    #[test]
    fn test_estimate_seasonal_period() {
        // Create data with period-7 seasonality
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + (i % 7) as f64 * 5.0 + (i as f64 * 0.1).sin())
            .collect();
        let period = estimate_seasonal_period(&values);
        // Should detect something near period 7 (may not be exact due to noise)
        assert!(period > 0);
    }

    #[test]
    fn test_estimate_seasonal_period_short() {
        let values = vec![1.0, 2.0, 3.0];
        let period = estimate_seasonal_period(&values);
        assert_eq!(period, 0);
    }

    #[test]
    fn test_quick_analysis() {
        let prices: Vec<f64> = (0..30).map(|i| 70.0 + i as f64 * 0.5).collect();
        let (analysis, signals) = quick_analysis(&prices, EnergyCommodity::NaturalGas).unwrap();
        assert_eq!(analysis.n_points, 30);
        assert!(!signals.is_empty());
    }

    #[test]
    fn test_quick_analysis_empty() {
        let result = quick_analysis(&[], EnergyCommodity::CrudeOil);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_commodity_pipeline() {
        let mut feeds = HashMap::new();
        feeds.insert(EnergyCommodity::CrudeOil, make_test_json(60));
        feeds.insert(EnergyCommodity::NaturalGas, make_test_json(60));

        let mut sd = HashMap::new();
        sd.insert(EnergyCommodity::CrudeOil, make_sd_records(20));

        let config = PipelineConfig::new(EnergyCommodity::CrudeOil);
        let results = run_multi_commodity_pipeline(&feeds, &sd, IngestFormat::Json, &config);

        assert!(results.contains_key(&EnergyCommodity::CrudeOil));
        assert!(results.contains_key(&EnergyCommodity::NaturalGas));
        assert!(results[&EnergyCommodity::CrudeOil].is_ok());
        assert!(results[&EnergyCommodity::NaturalGas].is_ok());
    }
}
