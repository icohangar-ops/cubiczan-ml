//! # Cubiczan ML Deep Learning
//!
//! Comprehensive deep learning module for the Cubiczan AI/DeFi ecosystem.
//! Provides neural network architectures, training configuration,
//! inference engines, and specialized components for time series forecasting,
//! anomaly detection, and on-chain analytics.
//!
//! ## Framework Support
//!
//! - **Candle**: Fast inference, transformer models, HuggingFace compatibility
//!
//! ## Modules
//!
//! - [`burn_backend`] — Training configuration, device management, checkpointing, scheduling
//! - [`models`] — Neural network architectures (LSTM, Transformer, Autoencoder, MLP, Conv1D)
//! - [`inference`] — Fast inference engines for Candle models
//! - [`time_series`] — Deep learning for financial time series forecasting
//! - [`on_chain`] — On-chain ML for blockchain transaction analysis

pub mod burn_backend;
pub mod models;
pub mod inference;
pub mod time_series;
pub mod on_chain;

// Re-exports for convenience
pub use burn_backend::{
    DeviceType, GpuInfo, DeviceManager,
    OptimizerType, LrSchedulerType,
    CheckpointConfig, EarlyStoppingConfig,
    TrainingState, TrainingConfig,
    DataLoaderConfig, BurnDataset, NumericDataset,
};
pub use models::{
    LstmConfig, LstmModel,
    TransformerEncoderConfig, TransformerModel,
    AutoencoderConfig, AutoencoderModel,
    MlpConfig, MlpModel, Conv1dConfig, Conv1dModel, Activation,
    FinanceTask, ModelZoo, ModelZooEntry,
};
pub use inference::{
    InferenceEngine, InferenceResult, CandleInferenceEngine,
    BatchInference, ModelLoader, InferenceBenchmark, ProfilingResult,
    Framework, ModelFormat,
};
pub use time_series::{
    TimeSeriesDataset, WindowConfig, SequencePredictor, MultiStepPredictor,
    AnomalyDetector, AnomalyResult, FeatureNormalizer, WalkForwardValidator,
    TimeSeriesPipeline, PipelineConfig, ForecastResult,
};
pub use on_chain::{
    TransactionEncoder, EncodedTransaction, OnChainAnomalyDetector, OnChainAnomalyResult,
    PatternRecognizer, MevPattern, FlowAnalyzer, TokenFlowReport,
    ContractRiskScorer, RiskAssessment,
};
