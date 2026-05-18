//! # Core Types for Commodity Price Analysis
//!
//! Defines all domain types used across the commodity-price-analyzer crate:
//! commodity identifiers, price points, forecasts, signals, seasonal patterns,
//! supply/demand factors, and composite analysis results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Commodity Type
// ---------------------------------------------------------------------------

/// Supported commodity types for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum CommodityType {
    Gold,
    Silver,
    Copper,
    Lithium,
    Nickel,
    Cobalt,
    Aluminum,
    Zinc,
    Platinum,
    Palladium,
    Uranium,
    RareEarths,
}

impl CommodityType {
    /// Returns the ticker symbol commonly associated with this commodity.
    pub fn ticker(&self) -> &'static str {
        match self {
            CommodityType::Gold => "XAU",
            CommodityType::Silver => "XAG",
            CommodityType::Copper => "XCU",
            CommodityType::Lithium => "LI",
            CommodityType::Nickel => "NI",
            CommodityType::Cobalt => "CO",
            CommodityType::Aluminum => "AL",
            CommodityType::Zinc => "ZN",
            CommodityType::Platinum => "XPT",
            CommodityType::Palladium => "XPD",
            CommodityType::Uranium => "UX",
            CommodityType::RareEarths => "RE",
        }
    }

    /// Returns a human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            CommodityType::Gold => "Gold",
            CommodityType::Silver => "Silver",
            CommodityType::Copper => "Copper",
            CommodityType::Lithium => "Lithium",
            CommodityType::Nickel => "Nickel",
            CommodityType::Cobalt => "Cobalt",
            CommodityType::Aluminum => "Aluminum",
            CommodityType::Zinc => "Zinc",
            CommodityType::Platinum => "Platinum",
            CommodityType::Palladium => "Palladium",
            CommodityType::Uranium => "Uranium",
            CommodityType::RareEarths => "Rare Earths",
        }
    }

    /// Returns a rough typical price range (low, high) in USD per troy ounce or metric ton.
    pub fn typical_price_range(&self) -> (f64, f64) {
        match self {
            CommodityType::Gold => (1500.0, 2800.0),
            CommodityType::Silver => (15.0, 40.0),
            CommodityType::Copper => (2.5, 5.5),
            CommodityType::Lithium => (10.0, 80.0),
            CommodityType::Nickel => (8.0, 25.0),
            CommodityType::Cobalt => (15.0, 50.0),
            CommodityType::Aluminum => (1500.0, 3000.0),
            CommodityType::Zinc => (1000.0, 3500.0),
            CommodityType::Platinum => (800.0, 1400.0),
            CommodityType::Palladium => (800.0, 3000.0),
            CommodityType::Uranium => (30.0, 120.0),
            CommodityType::RareEarths => (50.0, 300.0),
        }
    }

    /// Returns all commodity types as a vector.
    pub fn all() -> Vec<CommodityType> {
        vec![
            CommodityType::Gold,
            CommodityType::Silver,
            CommodityType::Copper,
            CommodityType::Lithium,
            CommodityType::Nickel,
            CommodityType::Cobalt,
            CommodityType::Aluminum,
            CommodityType::Zinc,
            CommodityType::Platinum,
            CommodityType::Palladium,
            CommodityType::Uranium,
            CommodityType::RareEarths,
        ]
    }
}

impl std::fmt::Display for CommodityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// Price Point
// ---------------------------------------------------------------------------

/// A single price data point with OHLCV information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PricePoint {
    /// Timestamp of the price data.
    pub timestamp: DateTime<Utc>,
    /// Opening price.
    pub open: f64,
    /// Highest price during the period.
    pub high: f64,
    /// Lowest price during the period.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Trading volume.
    pub volume: f64,
}

impl PricePoint {
    /// Create a new price point.
    pub fn new(timestamp: DateTime<Utc>, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        PricePoint {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Check if the OHLCV values are internally consistent.
    pub fn is_valid(&self) -> bool {
        self.high >= self.low
            && self.open >= 0.0
            && self.close >= 0.0
            && self.high >= 0.0
            && self.low >= 0.0
            && self.volume >= 0.0
    }

    /// Price range (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Typical price: (high + low + close) / 3.
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Simple return from open to close.
    pub fn intraday_return(&self) -> f64 {
        if self.open.abs() < 1e-15 {
            0.0
        } else {
            (self.close - self.open) / self.open
        }
    }
}

// ---------------------------------------------------------------------------
// Price Forecast
// ---------------------------------------------------------------------------

/// A price forecast produced by the forecasting engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceForecast {
    /// The commodity being forecast.
    pub commodity: CommodityType,
    /// Timestamp when the forecast was generated.
    pub timestamp: DateTime<Utc>,
    /// The predicted price.
    pub predicted_price: f64,
    /// Confidence level in [0, 1].
    pub confidence: f64,
    /// Forecast horizon in days.
    pub horizon: u32,
    /// Version identifier of the model used.
    pub model_version: String,
}

// ---------------------------------------------------------------------------
// Signal Strength (Commodity-Specific)
// ---------------------------------------------------------------------------

/// Signal strength classification for commodity trading signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

impl SignalStrength {
    /// Convert to a numeric score: StrongBuy=1.0, Buy=0.5, Neutral=0.0, Sell=-0.5, StrongSell=-1.0.
    pub fn to_score(&self) -> f64 {
        match self {
            SignalStrength::StrongBuy => 1.0,
            SignalStrength::Buy => 0.5,
            SignalStrength::Neutral => 0.0,
            SignalStrength::Sell => -0.5,
            SignalStrength::StrongSell => -1.0,
        }
    }

    /// Create from a numeric score in [-1, 1].
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 0.6 => SignalStrength::StrongBuy,
            s if s >= 0.1 => SignalStrength::Buy,
            s if s > -0.1 => SignalStrength::Neutral,
            s if s > -0.6 => SignalStrength::Sell,
            _ => SignalStrength::StrongSell,
        }
    }

    /// Whether this signal is actionable (not Neutral).
    pub fn is_actionable(&self) -> bool {
        *self != SignalStrength::Neutral
    }
}

impl std::fmt::Display for SignalStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalStrength::StrongBuy => write!(f, "STRONG_BUY"),
            SignalStrength::Buy => write!(f, "BUY"),
            SignalStrength::Neutral => write!(f, "NEUTRAL"),
            SignalStrength::Sell => write!(f, "SELL"),
            SignalStrength::StrongSell => write!(f, "STRONG_SELL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Commodity Signal
// ---------------------------------------------------------------------------

/// A trading signal for a specific commodity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommoditySignal {
    /// The commodity the signal is for.
    pub commodity: CommodityType,
    /// Signal classification.
    pub signal: SignalStrength,
    /// Confidence in [0, 1].
    pub confidence: f64,
    /// Human-readable reasoning for the signal.
    pub reasoning: String,
    /// Technical indicators that contributed to the signal.
    pub indicators: Vec<String>,
    /// Timestamp of signal generation.
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Seasonal Pattern
// ---------------------------------------------------------------------------

/// A detected seasonal pattern for a commodity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// The commodity.
    pub commodity: CommodityType,
    /// Month of the year (1-12).
    pub month: u32,
    /// Average historical return during this month.
    pub avg_return: f64,
    /// Historical volatility during this month.
    pub volatility: f64,
    /// Number of historical occurrences used to compute the average.
    pub historical_occurrences: u32,
}

// ---------------------------------------------------------------------------
// Supply Demand Factor
// ---------------------------------------------------------------------------

/// A factor influencing supply and demand for a commodity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyDemandFactor {
    /// Type of factor (e.g., "production", "inventory", "geopolitical").
    pub factor_type: String,
    /// Impact score in [-1, 1] where positive is bullish.
    pub impact_score: f64,
    /// Human-readable description.
    pub description: String,
    /// Source of the data/assessment.
    pub source: String,
}

// ---------------------------------------------------------------------------
// Commodity Analysis (Composite Result)
// ---------------------------------------------------------------------------

/// A comprehensive analysis result for a single commodity, combining
/// forecasts, signals, supply/demand data, seasonal patterns, and risk metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityAnalysis {
    /// The commodity analyzed.
    pub commodity: CommodityType,
    /// Current price.
    pub current_price: f64,
    /// Price forecasts from the forecasting engine.
    pub forecasts: Vec<PriceForecast>,
    /// Trading signals from the signal generator.
    pub signals: Vec<CommoditySignal>,
    /// Supply and demand factors.
    pub supply_demand: Vec<SupplyDemandFactor>,
    /// Detected seasonal patterns.
    pub seasonal: Vec<SeasonalPattern>,
    /// Risk metrics for this commodity.
    pub risk_metrics: Option<RiskMetricsSummary>,
}

// ---------------------------------------------------------------------------
// Risk Metrics Summary
// ---------------------------------------------------------------------------

/// Summarized risk metrics for a single commodity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetricsSummary {
    /// Annualized volatility.
    pub volatility: f64,
    /// Value at Risk at 95% confidence.
    pub var_95: f64,
    /// Conditional VaR (Expected Shortfall) at 95% confidence.
    pub cvar_95: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
    /// Volatility regime: "low", "medium", "high", or "extreme".
    pub volatility_regime: String,
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: f64,
}

// ---------------------------------------------------------------------------
// Resample Interval (Commodity-Specific)
// ---------------------------------------------------------------------------

/// Supported resampling intervals for price data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResamplePeriod {
    Daily,
    Weekly,
    Monthly,
}

// ---------------------------------------------------------------------------
// Volatility Regime
// ---------------------------------------------------------------------------

/// Volatility regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

impl VolatilityRegime {
    /// Classify based on annualized volatility percentile thresholds.
    pub fn from_volatility(vol: f64) -> Self {
        match vol {
            v if v < 0.15 => VolatilityRegime::Low,
            v if v < 0.30 => VolatilityRegime::Medium,
            v if v < 0.50 => VolatilityRegime::High,
            _ => VolatilityRegime::Extreme,
        }
    }

    /// Returns a multiplier for position sizing adjustments.
    pub fn position_multiplier(&self) -> f64 {
        match self {
            VolatilityRegime::Low => 1.5,
            VolatilityRegime::Medium => 1.0,
            VolatilityRegime::High => 0.6,
            VolatilityRegime::Extreme => 0.3,
        }
    }
}

impl std::fmt::Display for VolatilityRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolatilityRegime::Low => write!(f, "LOW"),
            VolatilityRegime::Medium => write!(f, "MEDIUM"),
            VolatilityRegime::High => write!(f, "HIGH"),
            VolatilityRegime::Extreme => write!(f, "EXTREME"),
        }
    }
}

// ---------------------------------------------------------------------------
// Timeframe for Signal Aggregation
// ---------------------------------------------------------------------------

/// Timeframe for trading signal analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Timeframe {
    Intraday,
    Daily,
    Weekly,
}

// ---------------------------------------------------------------------------
// Position Sizing Recommendation
// ---------------------------------------------------------------------------

/// A position sizing recommendation based on signal strength and risk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRecommendation {
    /// Recommended position size as a fraction of portfolio.
    pub size_fraction: f64,
    /// Stop-loss price.
    pub stop_loss: f64,
    /// Take-profit price.
    pub take_profit: f64,
    /// Risk/reward ratio.
    pub risk_reward_ratio: f64,
    /// Reasoning for the recommendation.
    pub reasoning: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commodity_type_ticker() {
        assert_eq!(CommodityType::Gold.ticker(), "XAU");
        assert_eq!(CommodityType::Silver.ticker(), "XAG");
        assert_eq!(CommodityType::Copper.ticker(), "XCU");
    }

    #[test]
    fn test_commodity_type_display() {
        assert_eq!(CommodityType::Gold.display_name(), "Gold");
        assert_eq!(CommodityType::RareEarths.display_name(), "Rare Earths");
    }

    #[test]
    fn test_commodity_type_all() {
        let all = CommodityType::all();
        assert_eq!(all.len(), 12);
        assert!(all.contains(&CommodityType::Gold));
        assert!(all.contains(&CommodityType::RareEarths));
    }

    #[test]
    fn test_commodity_type_typical_range() {
        let (lo, hi) = CommodityType::Gold.typical_price_range();
        assert!(lo < hi);
        assert!(lo > 0.0);
    }

    #[test]
    fn test_commodity_type_serde_roundtrip() {
        let ct = CommodityType::Platinum;
        let json = serde_json::to_string(&ct).unwrap();
        let back: CommodityType = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, back);
    }

    #[test]
    fn test_price_point_valid() {
        let pp = PricePoint::new(
            Utc::now(), 100.0, 105.0, 95.0, 103.0, 1000.0,
        );
        assert!(pp.is_valid());
        assert!((pp.range() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_point_invalid() {
        let pp = PricePoint::new(
            Utc::now(), 100.0, 95.0, 105.0, 100.0, 1000.0,
        );
        assert!(!pp.is_valid()); // high < low
    }

    #[test]
    fn test_price_point_typical_price() {
        let pp = PricePoint::new(Utc::now(), 100.0, 110.0, 90.0, 105.0, 500.0);
        assert!((pp.typical_price() - (110.0 + 90.0 + 105.0) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_point_intraday_return() {
        let pp = PricePoint::new(Utc::now(), 100.0, 110.0, 90.0, 110.0, 500.0);
        assert!((pp.intraday_return() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_price_point_serde_roundtrip() {
        let pp = PricePoint::new(Utc::now(), 100.0, 105.0, 95.0, 103.0, 1000.0);
        let json = serde_json::to_string(&pp).unwrap();
        let back: PricePoint = serde_json::from_str(&json).unwrap();
        assert_eq!(pp, back);
    }

    #[test]
    fn test_signal_strength_to_score() {
        assert!((SignalStrength::StrongBuy.to_score() - 1.0).abs() < 1e-10);
        assert!((SignalStrength::Neutral.to_score() - 0.0).abs() < 1e-10);
        assert!((SignalStrength::StrongSell.to_score() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_signal_strength_from_score() {
        assert_eq!(SignalStrength::from_score(0.8), SignalStrength::StrongBuy);
        assert_eq!(SignalStrength::from_score(0.3), SignalStrength::Buy);
        assert_eq!(SignalStrength::from_score(0.0), SignalStrength::Neutral);
        assert_eq!(SignalStrength::from_score(-0.3), SignalStrength::Sell);
        assert_eq!(SignalStrength::from_score(-0.8), SignalStrength::StrongSell);
    }

    #[test]
    fn test_signal_strength_actionable() {
        assert!(SignalStrength::StrongBuy.is_actionable());
        assert!(SignalStrength::Sell.is_actionable());
        assert!(!SignalStrength::Neutral.is_actionable());
    }

    #[test]
    fn test_forecast_serde_roundtrip() {
        let fc = PriceForecast {
            commodity: CommodityType::Gold,
            timestamp: Utc::now(),
            predicted_price: 2500.0,
            confidence: 0.85,
            horizon: 30,
            model_version: "v1.0".into(),
        };
        let json = serde_json::to_string(&fc).unwrap();
        let back: PriceForecast = serde_json::from_str(&json).unwrap();
        assert_eq!(fc.commodity, back.commodity);
        assert!((fc.predicted_price - back.predicted_price).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_pattern_creation() {
        let sp = SeasonalPattern {
            commodity: CommodityType::Gold,
            month: 1,
            avg_return: 0.02,
            volatility: 0.05,
            historical_occurrences: 20,
        };
        assert_eq!(sp.month, 1);
        assert!(sp.avg_return > 0.0);
    }

    #[test]
    fn test_supply_demand_factor_creation() {
        let sdf = SupplyDemandFactor {
            factor_type: "inventory_draw".into(),
            impact_score: 0.7,
            description: "Strong inventory drawdown".into(),
            source: "LME".into(),
        };
        assert!(sdf.impact_score > 0.0);
    }

    #[test]
    fn test_volatility_regime_from_vol() {
        assert_eq!(VolatilityRegime::from_volatility(0.05), VolatilityRegime::Low);
        assert_eq!(VolatilityRegime::from_volatility(0.20), VolatilityRegime::Medium);
        assert_eq!(VolatilityRegime::from_volatility(0.40), VolatilityRegime::High);
        assert_eq!(VolatilityRegime::from_volatility(0.60), VolatilityRegime::Extreme);
    }

    #[test]
    fn test_volatility_regime_multiplier() {
        assert!(VolatilityRegime::Low.position_multiplier() > VolatilityRegime::Extreme.position_multiplier());
    }

    #[test]
    fn test_commodity_analysis_creation() {
        let analysis = CommodityAnalysis {
            commodity: CommodityType::Copper,
            current_price: 4.50,
            forecasts: vec![],
            signals: vec![],
            supply_demand: vec![],
            seasonal: vec![],
            risk_metrics: None,
        };
        assert_eq!(analysis.commodity, CommodityType::Copper);
        assert!(analysis.forecasts.is_empty());
    }

    #[test]
    fn test_risk_metrics_summary() {
        let rms = RiskMetricsSummary {
            volatility: 0.20,
            var_95: -0.03,
            cvar_95: -0.05,
            max_drawdown: -0.15,
            volatility_regime: "Medium".into(),
            sharpe_ratio: 1.5,
        };
        assert!((rms.volatility - 0.20).abs() < 1e-10);
        assert_eq!(rms.volatility_regime, "Medium");
    }
}
