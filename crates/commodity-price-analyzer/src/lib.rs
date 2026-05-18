//! # Commodity Price Analyzer
//!
//! A comprehensive Rust crate for analyzing commodity prices. Provides:
//!
//! - **Price data management**: Storage, retrieval, resampling, and mock data generation
//! - **Forecasting**: Linear regression, moving averages, mean reversion, exponential smoothing, ensemble
//! - **Signal generation**: RSI, MACD, Bollinger Bands, ATR, Stochastic, composite scoring, position sizing
//! - **Seasonal analysis**: Monthly patterns, day-of-week effects, inventory cycle estimation
//! - **Supply/demand modeling**: Inventory tracking, production trends, consumption proxy, geopolitical risk
//! - **Risk metrics**: VaR, CVaR, max drawdown, correlation, volatility regime detection
//! - **Pipeline**: Full orchestrated analysis for single and multi-commodity workflows

pub mod types;
pub mod prices;
pub mod forecast;
pub mod signals;
pub mod seasonal;
pub mod supply_demand;
pub mod risk;
pub mod pipeline;

// ---------------------------------------------------------------------------
// Key Re-exports
// ---------------------------------------------------------------------------

pub use types::{
    CommodityAnalysis, CommoditySignal, CommodityType, PositionRecommendation,
    PriceForecast, PricePoint, ResamplePeriod, RiskMetricsSummary, SeasonalPattern,
    SignalStrength, SupplyDemandFactor, Timeframe, VolatilityRegime,
};

pub use prices::PriceDatabase;

pub use forecast::{ForecastConfig, ForecastEngine, EnsembleWeights};

pub use signals::{SignalGenerator, TechnicalIndicators};

pub use seasonal::SeasonalAnalyzer;

pub use supply_demand::SupplyDemandModel;

pub use risk::CommodityRiskAnalyzer;

pub use pipeline::{AnalysisPipeline, PipelineConfig, MultiCommodityReport, MetricComparison};
