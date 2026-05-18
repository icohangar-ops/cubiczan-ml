//! # scope-glacier
//!
//! Energy markets intelligence platform for EIA data analytics and forecasting.
//!
//! This crate provides:
//! - **Ingestion**: Parse EIA JSON/CSV data feeds with validation and gap filling
//! - **Time-series**: Decomposition, moving averages, autocorrelation, stationarity tests
//! - **Analytics**: Spot price analysis, spread calculations, volatility surfaces, supply/demand scoring
//! - **Forecasting**: Holt-Winters, linear regression, ensemble methods with confidence intervals
//! - **Pipeline**: End-to-end integration from raw data to actionable market signals
//!
//! ## Example
//!
//! ```ignore
//! use scope_glacier::pipeline::run_pipeline_json;
//! use scope_glacier::types::PipelineConfig;
//!
//! let json_data = r#"..."#;  // EIA JSON feed
//! let config = PipelineConfig::default();
//! let result = run_pipeline_json(json_data, &[], config)?;
//! ```

pub mod analytics;
pub mod forecast;
pub mod ingest;
pub mod pipeline;
pub mod timeseries;
pub mod types;

// Re-exports for convenience
pub use types::{
    EnergyCommodity, ForecastModel, ForecastResult, GlacierError, MarketSignal, PipelineConfig,
    PricePoint, SeasonalDecomposition, SupplyDemandRecord,
};

#[cfg(test)]
mod integration_tests {
    use crate::analytics::*;
    use crate::forecast::*;
    use crate::ingest::*;
    use crate::pipeline::*;
    use crate::timeseries::*;
    use crate::types::*;
    use chrono::Utc;
    use std::collections::HashMap;

    /// Generates a synthetic energy price series with trend + seasonality + noise.
    fn generate_synthetic_prices(n: usize, base: f64, trend: f64, seasonal_amp: f64, period: usize, noise: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                base + i as f64 * trend
                    + seasonal_amp * ((2.0 * std::f64::consts::PI * i as f64) / period as f64).sin()
                    + noise * pseudo_random(i as u64)
            })
            .collect()
    }

    fn pseudo_random(seed: u64) -> f64 {
        let mut s = seed;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (u32::MAX as f64) - 0.5
    }

    fn make_eia_json(prices: &[f64], start_day: i64) -> String {
        let data: Vec<String> = prices
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let ts = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    + chrono::Duration::days(start_day + i as i64);
                format!(r#"{{"period": "{}", "value": {:.4}}}"#, ts.format("%Y-%m-%d"), p)
            })
            .collect();
        format!(
            r#"{{"response": {{"data": [{{"series_id": "PET.RWTC.D", "name": "WTI Spot", "units": "$/barrel", "data": [{}]}}]}}}}"#,
            data.join(", ")
        )
    }

    fn make_sd_records(prices: &[f64], commodity: EnergyCommodity) -> Vec<SupplyDemandRecord> {
        let now = Utc::now();
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let mut rec = SupplyDemandRecord::new(
                    now - chrono::Duration::days(prices.len() as i64 - i as i64),
                    commodity,
                );
                rec.supply_mmbtu = Some(p * 1.1);
                rec.demand_mmbtu = Some(p * 0.95);
                rec.inventory_level = Some(p * 50.0);
                rec.consumption_rate = Some(p * 0.03);
                rec
            })
            .collect()
    }

    #[test]
    fn test_full_pipeline_integration() {
        // Generate 120 days of synthetic natural gas data
        let prices = generate_synthetic_prices(120, 3.0, 0.005, 0.3, 30, 0.1);
        let json = make_eia_json(&prices, 0);
        let sd = make_sd_records(&prices, EnergyCommodity::NaturalGas);

        let config = PipelineConfig::new(EnergyCommodity::NaturalGas)
            .with_model(ForecastModel::HoltWinters)
            .with_horizon(15);

        let result = run_pipeline_json(&json, &sd, config).unwrap();

        // Verify all pipeline phases ran
        assert!(result.n_raw_points > 0);
        assert!(result.n_normalized > 0);
        assert!(!result.signals.is_empty());
        assert!(result.forecast.len() > 0);
        assert!(result.forecast.mae.is_finite());
        assert!(result.spot_analysis.n_points > 0);

        // Report generation
        let report = generate_report(&result);
        assert!(report.contains("Scope-Glacier"));
        assert!(report.len() > 100);
    }

    #[test]
    fn test_csv_pipeline_integration() {
        let prices = generate_synthetic_prices(80, 75.0, 0.2, 2.0, 7, 1.5);
        let mut csv = String::from("period,price,volume\n");
        for (i, &p) in prices.iter().enumerate() {
            let ts = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                + chrono::Duration::days(i as i64);
            csv.push_str(&format!("{},{:.2},{}\n", ts.format("%Y-%m-%d"), p, 500 + i));
        }

        let config = PipelineConfig::new(EnergyCommodity::CrudeOil)
            .with_model(ForecastModel::Regression);

        let result = run_pipeline_csv(&csv, &[], config).unwrap();
        assert_eq!(result.n_raw_points, 80);
        assert!(result.forecast.predictions.len() > 0);
    }

    #[test]
    fn test_multi_commodity_batch() {
        let crude_prices = generate_synthetic_prices(100, 75.0, 0.15, 2.0, 30, 1.0);
        let gas_prices = generate_synthetic_prices(100, 3.0, 0.005, 0.3, 30, 0.05);

        let mut feeds = HashMap::new();
        feeds.insert(EnergyCommodity::CrudeOil, make_eia_json(&crude_prices, 0));
        feeds.insert(EnergyCommodity::NaturalGas, make_eia_json(&gas_prices, 0));

        let mut sd = HashMap::new();
        sd.insert(EnergyCommodity::CrudeOil, make_sd_records(&crude_prices, EnergyCommodity::CrudeOil));
        sd.insert(EnergyCommodity::NaturalGas, make_sd_records(&gas_prices, EnergyCommodity::NaturalGas));

        let config = PipelineConfig::default();
        let results = run_multi_commodity_pipeline(&feeds, &sd, IngestFormat::Json, &config);

        for (commodity, result) in &results {
            assert!(result.is_ok(), "Pipeline failed for {:?}", commodity);
            let r = result.as_ref().unwrap();
            assert!(r.n_raw_points > 0);
            assert!(!r.signals.is_empty());
        }
    }

    #[test]
    fn test_ensemble_across_models() {
        let prices = generate_synthetic_prices(80, 50.0, 0.1, 1.0, 20, 0.5);
        let json = make_eia_json(&prices, 0);

        let models = [
            ForecastModel::HoltWinters,
            ForecastModel::Regression,
        ];

        let mut forecasts = Vec::new();
        for model in &models {
            let config = PipelineConfig::new(EnergyCommodity::CrudeOil)
                .with_model(*model);
            if let Ok(result) = run_pipeline_json(&json, &[], config) {
                forecasts.push(result.forecast);
            }
        }

        assert!(!forecasts.is_empty());
        let ensemble = ensemble_forecast(&forecasts);
        assert!(ensemble.is_some());
        let ens = ensemble.unwrap();
        assert!(!ens.predictions.is_empty());
    }

    #[test]
    fn test_data_quality_pipeline() {
        // Create data with gaps and outliers
        let mut prices: Vec<f64> = (0..40).map(|i| 70.0 + i as f64 * 0.3).collect();
        // Add outlier
        prices[20] = 500.0;
        // Add some noise
        prices[30] = -10.0;

        let json = make_eia_json(&prices, 0);
        let config = PipelineConfig::new(EnergyCommodity::CrudeOil);

        let result = run_pipeline_json(&json, &[], config);
        // Should either succeed with cleaned data or fail due to insufficient data after cleaning
        match result {
            Ok(r) => {
                // If successful, cleaned data should be less than raw
                assert!(r.n_normalized <= r.n_raw_points);
            }
            Err(GlacierError::InsufficientData { .. }) => {
                // Acceptable: too many data points removed
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    fn test_stationarity_in_pipeline() {
        // Stationary series (mean-reverting noise around a constant)
        let stationary_prices: Vec<f64> = (0..100)
            .map(|i| 50.0 + pseudo_random(i as u64) * 2.0)
            .collect();
        let (_, is_stat) = adf_stationarity_test(&stationary_prices, 0.10);
        // Stationary series should be detected as such with looser threshold
        assert!(is_stat);

        // Trending series (non-stationary)
        let trending_prices: Vec<f64> = (0..100).map(|i| 50.0 + i as f64 * 0.5).collect();
        let (_, is_stat_trend) = adf_stationarity_test(&trending_prices, 0.05);
        // Trending series should be non-stationary
        assert!(!is_stat_trend);
    }

    #[test]
    fn test_all_commodity_types() {
        // Verify all commodity types have proper metadata
        for commodity in EnergyCommodity::all() {
            assert!(!commodity.label().is_empty());
            assert!(!commodity.unit().is_empty());
            assert!(!commodity.eia_prefix().is_empty());
        }
    }

    #[test]
    fn test_walk_forward_in_pipeline() {
        let prices = generate_synthetic_prices(100, 100.0, 0.2, 3.0, 12, 2.0);
        let results = walk_forward_validation(&prices, 30, 10, 5);
        assert!(!results.is_empty());

        // All validation windows should produce finite metrics
        for (mae_v, rmse_v, mape_v) in &results {
            assert!(mae_v.is_finite());
            assert!(rmse_v.is_finite());
            assert!(mape_v.is_finite());
        }
    }

    #[test]
    fn test_imbalance_scoring_integration() {
        let now = Utc::now();

        // Deficit scenario
        let deficit_records: Vec<SupplyDemandRecord> = (0..20)
            .map(|i| {
                let mut rec = SupplyDemandRecord::new(now - chrono::Duration::days(i), EnergyCommodity::NaturalGas);
                rec.supply_mmbtu = Some(80.0);
                rec.demand_mmbtu = Some(120.0);
                rec
            })
            .collect();

        let score = score_imbalance(&deficit_records).unwrap();
        assert!(matches!(score.severity, ImbalanceSeverity::SevereDeficit | ImbalanceSeverity::ModerateDeficit));

        // Surplus scenario
        let surplus_records: Vec<SupplyDemandRecord> = (0..20)
            .map(|i| {
                let mut rec = SupplyDemandRecord::new(now - chrono::Duration::days(i), EnergyCommodity::CrudeOil);
                rec.supply_mmbtu = Some(150.0);
                rec.demand_mmbtu = Some(100.0);
                rec
            })
            .collect();

        let score = score_imbalance(&surplus_records).unwrap();
        assert!(matches!(score.severity, ImbalanceSeverity::ModerateSurplus | ImbalanceSeverity::SevereSurplus));
    }

    #[test]
    fn test_volatility_surface_integration() {
        let prices = generate_synthetic_prices(200, 3.0, 0.002, 0.5, 7, 0.15);
        let surface = compute_volatility_surface(&prices, &[5, 10, 20, 50, 100]).unwrap();

        assert!(!surface.is_empty());
        // Volatility at different tenors should vary
        let vols: Vec<f64> = surface.iter().map(|p| p.volatility).collect();
        let all_same = vols.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
        // With synthetic noisy data, volatilities should differ across windows
        assert!(!all_same);
    }
}
