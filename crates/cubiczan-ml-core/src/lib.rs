//! # CubicZan ML Core
//!
//! The foundational ML library for the CubicZan finance/DeFi AI ecosystem.
//! Provides financial math primitives, time series analysis, trading signals,
//! risk management, and data preprocessing utilities.
//!
//! ## Modules
//!
//! - [`math`] — Vector/matrix operations, moving averages, volatility, portfolio stats, statistical tests
//! - [`time_series`] — OHLCV handling, resampling, returns calculation, stationarity, seasonality
//! - [`signal`] — Trading signal types, aggregation, consensus, conflict detection
//! - [`risk`] — Position sizing, risk metrics, drawdown tracking, exposure limits
//! - [`preprocessing`] — Scalers, encoders, train/test splits, feature engineering
//! - [`utils`] — General-purpose ML helpers (softmax, sigmoid, one-hot, MSE, etc.)
//! - [`device`] — Compute device enumeration (CPU, CUDA)

pub mod math;
pub mod device;
pub mod error;
pub mod metrics;
pub mod normalization;
pub mod utils;
pub mod preprocessing;
pub mod risk;
pub mod signal;
pub mod time_series;

// Re-exports of commonly used types
pub use math::{
    CorrelationMethod, MathError, MovingAverage, MovingAverageType, PortfolioMetrics, Quantile,
    StatisticalTest, VecOps, Volatility, VolatilityMethod, MatOps,
};
pub use preprocessing::{LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler};
pub use risk::{
    DrawdownTracker, ExposureLimit, ExposureReport, KellyCriterion, PortfolioConstructor,
    PositionSizer, RiskMetrics, StopLossCalculator, FixedFractional, VolatilityAdjusted,
};
pub use signal::{
    ConsensusMethod, ConflictAnalysis, ConflictResolution, Signal, SignalAggregator,
    SignalDirection, SignalHistory, SignalPerformance, SignalStrength,
};
pub use time_series::{OHLCV, ResampleInterval, SeasonalityResult, TimeSeriesAnalyzer, VolumeProfile};
pub use error::{MlError, Result};
pub use device::DeviceType;
pub use metrics::Metrics;
pub use normalization::NormalizationStats;
