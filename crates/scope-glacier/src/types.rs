//! Core domain types for the scope-glacier energy markets intelligence platform.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Energy commodity types tracked by the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergyCommodity {
    CrudeOil,
    NaturalGas,
    Coal,
    Uranium,
    Gasoline,
    HeatingOil,
    Electricity,
    Ethanol,
    RefinedProducts,
}

impl EnergyCommodity {
    /// Returns the standard EIA series prefix for this commodity.
    pub fn eia_prefix(&self) -> &'static str {
        match self {
            EnergyCommodity::CrudeOil => "PET.RWTC",
            EnergyCommodity::NaturalGas => "NG.RNGC",
            EnergyCommodity::Coal => "COAL.WMC",
            EnergyCommodity::Uranium => "NUC.UX",
            EnergyCommodity::Gasoline => "PET.EER_EPMRU_PF4_RGA_DPG",
            EnergyCommodity::HeatingOil => "PET.EER_EPJK_PF4_Y35_NUS_DPG",
            EnergyCommodity::Electricity => "ELEC.PRICE",
            EnergyCommodity::Ethanol => "PET.EMD_EPD2D_PTE_NUS_DPG",
            EnergyCommodity::RefinedProducts => "PET.RPPC",
        }
    }

    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            EnergyCommodity::CrudeOil => "WTI Crude Oil",
            EnergyCommodity::NaturalGas => "Henry Hub Natural Gas",
            EnergyCommodity::Coal => "Powder River Basin Coal",
            EnergyCommodity::Uranium => "Uranium U3O8",
            EnergyCommodity::Gasoline => "U.S. Regular Gasoline",
            EnergyCommodity::HeatingOil => "U.S. No. 2 Heating Oil",
            EnergyCommodity::Electricity => "U.S. Electricity",
            EnergyCommodity::Ethanol => "U.S. Ethanol",
            EnergyCommodity::RefinedProducts => "Refined Products Index",
        }
    }

    /// Returns the typical unit of measurement.
    pub fn unit(&self) -> &'static str {
        match self {
            EnergyCommodity::CrudeOil => "$/barrel",
            EnergyCommodity::NaturalGas => "$/MMBtu",
            EnergyCommodity::Coal => "$/short ton",
            EnergyCommodity::Uranium => "$/lb U3O8",
            EnergyCommodity::Gasoline => "$/gallon",
            EnergyCommodity::HeatingOil => "$/gallon",
            EnergyCommodity::Electricity => "$/MWh",
            EnergyCommodity::Ethanol => "$/gallon",
            EnergyCommodity::RefinedProducts => "$/barrel",
        }
    }

    /// Lists all supported commodities.
    pub fn all() -> Vec<EnergyCommodity> {
        vec![
            EnergyCommodity::CrudeOil,
            EnergyCommodity::NaturalGas,
            EnergyCommodity::Coal,
            EnergyCommodity::Uranium,
            EnergyCommodity::Gasoline,
            EnergyCommodity::HeatingOil,
            EnergyCommodity::Electricity,
            EnergyCommodity::Ethanol,
            EnergyCommodity::RefinedProducts,
        ]
    }
}

impl fmt::Display for EnergyCommodity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// A single price observation at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub commodity: EnergyCommodity,
    pub price: f64,
    pub volume: Option<f64>,
    pub source: String,
}

impl PricePoint {
    pub fn new(
        timestamp: DateTime<Utc>,
        commodity: EnergyCommodity,
        price: f64,
    ) -> Self {
        PricePoint {
            timestamp,
            commodity,
            price,
            volume: None,
            source: String::from("unknown"),
        }
    }

    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = Some(volume);
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Returns true if the price is valid (positive and finite).
    pub fn is_valid(&self) -> bool {
        self.price.is_finite() && self.price > 0.0
    }
}

/// A supply/demand record for a given commodity and time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyDemandRecord {
    pub timestamp: DateTime<Utc>,
    pub commodity: EnergyCommodity,
    pub supply_mmbtu: Option<f64>,
    pub demand_mmbtu: Option<f64>,
    pub inventory_level: Option<f64>,
    pub production_rate: Option<f64>,
    pub consumption_rate: Option<f64>,
    pub region: String,
}

impl SupplyDemandRecord {
    pub fn new(timestamp: DateTime<Utc>, commodity: EnergyCommodity) -> Self {
        SupplyDemandRecord {
            timestamp,
            commodity,
            supply_mmbtu: None,
            demand_mmbtu: None,
            inventory_level: None,
            production_rate: None,
            consumption_rate: None,
            region: String::from("US"),
        }
    }

    /// Computes the supply-demand imbalance ratio.
    /// Positive = oversupply, negative = deficit.
    pub fn imbalance(&self) -> Option<f64> {
        match (self.supply_mmbtu, self.demand_mmbtu) {
            (Some(s), Some(d)) if d.abs() > f64::EPSILON => Some((s - d) / d),
            _ => None,
        }
    }

    /// Returns true if the record has at least supply and demand fields populated.
    pub fn is_complete(&self) -> bool {
        self.supply_mmbtu.is_some() && self.demand_mmbtu.is_some()
    }
}

/// Forecast model types available in the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForecastModel {
    ARIMA,
    HoltWinters,
    Regression,
}

impl fmt::Display for ForecastModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForecastModel::ARIMA => write!(f, "ARIMA"),
            ForecastModel::HoltWinters => write!(f, "Holt-Winters"),
            ForecastModel::Regression => write!(f, "Linear Regression"),
        }
    }
}

/// A market signal generated by the analytics engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSignal {
    pub timestamp: DateTime<Utc>,
    pub commodity: EnergyCommodity,
    pub signal_type: SignalType,
    pub strength: f64, // -1.0 to 1.0
    pub description: String,
    pub horizon_days: u32,
}

impl MarketSignal {
    pub fn bullish(
        timestamp: DateTime<Utc>,
        commodity: EnergyCommodity,
        strength: f64,
        description: impl Into<String>,
    ) -> Self {
        MarketSignal {
            timestamp,
            commodity,
            signal_type: SignalType::Bullish,
            strength: strength.clamp(0.0, 1.0),
            description: description.into(),
            horizon_days: 30,
        }
    }

    pub fn bearish(
        timestamp: DateTime<Utc>,
        commodity: EnergyCommodity,
        strength: f64,
        description: impl Into<String>,
    ) -> Self {
        MarketSignal {
            timestamp,
            commodity,
            signal_type: SignalType::Bearish,
            strength: (-strength).clamp(-1.0, 0.0),
            description: description.into(),
            horizon_days: 30,
        }
    }

    pub fn neutral(
        timestamp: DateTime<Utc>,
        commodity: EnergyCommodity,
        description: impl Into<String>,
    ) -> Self {
        MarketSignal {
            timestamp,
            commodity,
            signal_type: SignalType::Neutral,
            strength: 0.0,
            description: description.into(),
            horizon_days: 30,
        }
    }
}

/// Signal direction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Bullish,
    Bearish,
    Neutral,
}

impl fmt::Display for SignalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalType::Bullish => write!(f, "BULLISH"),
            SignalType::Bearish => write!(f, "BEARISH"),
            SignalType::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// Result of seasonal decomposition of a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalDecomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub period: usize,
    pub timestamps: Vec<DateTime<Utc>>,
}

impl SeasonalDecomposition {
    /// Reconstructs the original series from components.
    pub fn reconstruct(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(self.seasonal.iter())
            .zip(self.residual.iter())
            .map(|((t, s), r)| t + s + r)
            .collect()
    }

    /// Returns the proportion of variance explained by the seasonal component.
    pub fn seasonal_strength(&self) -> f64 {
        let total_var: f64 = self.reconstruct().iter().map(|x| (x - self.mean()).powi(2)).sum();
        if total_var.abs() < f64::EPSILON {
            return 0.0;
        }
        let seasonal_var: f64 = self.seasonal.iter().map(|x| x.powi(2)).sum();
        (seasonal_var / total_var).clamp(0.0, 1.0)
    }

    fn mean(&self) -> f64 {
        let reconstructed = self.reconstruct();
        if reconstructed.is_empty() {
            return 0.0;
        }
        reconstructed.iter().sum::<f64>() / reconstructed.len() as f64
    }
}

/// Errors specific to the scope-glacier crate.
#[derive(Debug, thiserror::Error)]
pub enum GlacierError {
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Unsupported commodity: {0}")]
    UnsupportedCommodity(String),
}

pub type Result<T> = std::result::Result<T, GlacierError>;

/// Configuration for a processing pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub commodity: EnergyCommodity,
    pub lookback_days: u32,
    pub forecast_horizon: u32,
    pub model: ForecastModel,
    pub smoothing_alpha: f64,
    pub smoothing_beta: f64,
    pub smoothing_gamma: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            commodity: EnergyCommodity::NaturalGas,
            lookback_days: 365,
            forecast_horizon: 30,
            model: ForecastModel::HoltWinters,
            smoothing_alpha: 0.3,
            smoothing_beta: 0.1,
            smoothing_gamma: 0.1,
        }
    }
}

impl PipelineConfig {
    pub fn new(commodity: EnergyCommodity) -> Self {
        PipelineConfig {
            commodity,
            ..Default::default()
        }
    }

    pub fn with_model(mut self, model: ForecastModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_horizon(mut self, horizon: u32) -> Self {
        self.forecast_horizon = horizon;
        self
    }

    pub fn with_lookback(mut self, days: u32) -> Self {
        self.lookback_days = days;
        self
    }
}

/// A forecast result with point predictions and confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub commodity: EnergyCommodity,
    pub model: ForecastModel,
    pub predictions: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
}

impl ForecastResult {
    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_commodity_eia_prefix() {
        assert_eq!(EnergyCommodity::CrudeOil.eia_prefix(), "PET.RWTC");
        assert_eq!(EnergyCommodity::NaturalGas.eia_prefix(), "NG.RNGC");
    }

    #[test]
    fn test_energy_commodity_all_count() {
        assert_eq!(EnergyCommodity::all().len(), 9);
    }

    #[test]
    fn test_energy_commodity_display() {
        let c = EnergyCommodity::CrudeOil;
        assert_eq!(format!("{}", c), "WTI Crude Oil");
    }

    #[test]
    fn test_energy_commodity_units() {
        assert_eq!(EnergyCommodity::CrudeOil.unit(), "$/barrel");
        assert_eq!(EnergyCommodity::NaturalGas.unit(), "$/MMBtu");
        assert_eq!(EnergyCommodity::Coal.unit(), "$/short ton");
    }

    #[test]
    fn test_price_point_validity() {
        let valid = PricePoint::new(Utc::now(), EnergyCommodity::CrudeOil, 75.0);
        assert!(valid.is_valid());

        let invalid = PricePoint::new(Utc::now(), EnergyCommodity::CrudeOil, -5.0);
        assert!(!invalid.is_valid());

        let nan_price = PricePoint::new(Utc::now(), EnergyCommodity::CrudeOil, f64::NAN);
        assert!(!nan_price.is_valid());
    }

    #[test]
    fn test_price_point_builder() {
        let pp = PricePoint::new(Utc::now(), EnergyCommodity::NaturalGas, 3.5)
            .with_volume(1000.0)
            .with_source("EIA");
        assert_eq!(pp.volume, Some(1000.0));
        assert_eq!(pp.source, "EIA");
    }

    #[test]
    fn test_supply_demand_imbalance() {
        let mut rec = SupplyDemandRecord::new(Utc::now(), EnergyCommodity::NaturalGas);
        rec.supply_mmbtu = Some(110.0);
        rec.demand_mmbtu = Some(100.0);
        let imb = rec.imbalance().unwrap();
        assert!((imb - 0.1).abs() < 1e-10);

        assert!(rec.is_complete());
    }

    #[test]
    fn test_supply_demand_incomplete() {
        let rec = SupplyDemandRecord::new(Utc::now(), EnergyCommodity::CrudeOil);
        assert!(!rec.is_complete());
        assert!(rec.imbalance().is_none());
    }

    #[test]
    fn test_market_signal_creation() {
        let now = Utc::now();
        let bull = MarketSignal::bullish(now, EnergyCommodity::CrudeOil, 0.8, "Trend breakout");
        assert_eq!(bull.signal_type, SignalType::Bullish);
        assert!((bull.strength - 0.8).abs() < 1e-10);

        let bear = MarketSignal::bearish(now, EnergyCommodity::NaturalGas, 0.6, "Oversupply");
        assert_eq!(bear.signal_type, SignalType::Bearish);
        assert!((bear.strength - (-0.6)).abs() < 1e-10);

        let neutral = MarketSignal::neutral(now, EnergyCommodity::Coal, "No clear trend");
        assert_eq!(neutral.signal_type, SignalType::Neutral);
        assert!((neutral.strength).abs() < 1e-10);
    }

    #[test]
    fn test_signal_strength_clamping() {
        let now = Utc::now();
        let over = MarketSignal::bullish(now, EnergyCommodity::CrudeOil, 5.0, "Too strong");
        assert!((over.strength - 1.0).abs() < 1e-10);

        let under = MarketSignal::bearish(now, EnergyCommodity::CrudeOil, 10.0, "Too weak");
        assert!((under.strength - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_decomposition_reconstruct() {
        let decomp = SeasonalDecomposition {
            trend: vec![10.0, 11.0, 12.0],
            seasonal: vec![1.0, -1.0, 0.0],
            residual: vec![0.5, -0.5, 0.0],
            period: 3,
            timestamps: vec![],
        };
        let recon = decomp.reconstruct();
        assert_eq!(recon, vec![11.5, 9.5, 12.0]);
    }

    #[test]
    fn test_seasonal_strength() {
        let decomp = SeasonalDecomposition {
            trend: vec![10.0, 10.0, 10.0, 10.0],
            seasonal: vec![2.0, -2.0, 2.0, -2.0],
            residual: vec![0.0, 0.0, 0.0, 0.0],
            period: 2,
            timestamps: vec![],
        };
        let strength = decomp.seasonal_strength();
        assert!(strength > 0.0);
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.commodity, EnergyCommodity::NaturalGas);
        assert_eq!(cfg.lookback_days, 365);
        assert_eq!(cfg.forecast_horizon, 30);
    }

    #[test]
    fn test_pipeline_config_builder() {
        let cfg = PipelineConfig::new(EnergyCommodity::CrudeOil)
            .with_model(ForecastModel::ARIMA)
            .with_horizon(60);
        assert_eq!(cfg.commodity, EnergyCommodity::CrudeOil);
        assert_eq!(cfg.model, ForecastModel::ARIMA);
        assert_eq!(cfg.forecast_horizon, 60);
    }

    #[test]
    fn test_glacier_error_display() {
        let err = GlacierError::InsufficientData {
            required: 10,
            actual: 3,
        };
        assert!(err.to_string().contains("10") && err.to_string().contains("3"));
    }

    #[test]
    fn test_forecast_model_display() {
        assert_eq!(format!("{}", ForecastModel::ARIMA), "ARIMA");
        assert_eq!(format!("{}", ForecastModel::HoltWinters), "Holt-Winters");
        assert_eq!(format!("{}", ForecastModel::Regression), "Linear Regression");
    }

    #[test]
    fn test_supply_demand_zero_demand() {
        let mut rec = SupplyDemandRecord::new(Utc::now(), EnergyCommodity::NaturalGas);
        rec.supply_mmbtu = Some(100.0);
        rec.demand_mmbtu = Some(0.0);
        assert!(rec.imbalance().is_none());
    }

    #[test]
    fn test_forecast_result_empty() {
        let result = ForecastResult {
            commodity: EnergyCommodity::CrudeOil,
            model: ForecastModel::Regression,
            predictions: vec![],
            lower_bound: vec![],
            upper_bound: vec![],
            timestamps: vec![],
            mae: 0.0,
            rmse: 0.0,
            mape: 0.0,
        };
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }
}
