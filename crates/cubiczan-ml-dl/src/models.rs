//! # Neural Network Architectures
//!
//! Comprehensive collection of neural network model configurations for financial
//! and DeFi applications. Each architecture provides configurable hyperparameters.
//!
//! ## Architectures
//!
//! - [`LstmConfig`] — LSTM for time series forecasting
//! - [`TransformerEncoderConfig`] — Transformer encoder for sequence tasks
//! - [`AutoencoderConfig`] — Autoencoder for anomaly detection
//! - [`MlpConfig`] — MLP for tabular financial data
//! - [`Conv1dConfig`] — Conv1D for price pattern recognition
//! - [`ModelZoo`] — Pre-configured architectures for common finance tasks
//!
//! ## Example
//!
//! ```ignore
//! let config = LstmConfig::new(10, 64, 2)
//!     .num_layers(2)
//!     .dropout(0.2);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

// ─────────────────────────────────────────────────────────────────────────────
// LSTM Network — Time Series Forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the LSTM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub batch_first: bool,
}

impl LstmConfig {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            num_layers: 2,
            dropout: 0.2,
            batch_first: true,
        }
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    pub fn dropout(mut self, p: f64) -> Self {
        self.dropout = p;
        self
    }

    pub fn batch_first(mut self, enabled: bool) -> Self {
        self.batch_first = enabled;
        self
    }

    /// Estimate the number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let lstm_params = if self.num_layers == 1 {
            4 * (self.input_size + self.hidden_size + 1) * self.hidden_size
        } else {
            4 * (self.input_size + self.hidden_size + 1) * self.hidden_size
                + 4 * (self.hidden_size + self.hidden_size + 1) * self.hidden_size * (self.num_layers - 1)
        };
        let linear_params = (self.hidden_size + 1) * self.output_size;
        lstm_params + linear_params
    }
}

/// LSTM model holder (configuration only — backend-agnostic).
pub struct LstmModel {
    config: LstmConfig,
}

impl LstmModel {
    pub fn new(config: &LstmConfig) -> Result<Self> {
        info!(
            "LstmModel created: input={}, hidden={}, output={}, layers={}, params={}",
            config.input_size,
            config.hidden_size,
            config.output_size,
            config.num_layers,
            config.param_count(),
        );
        Ok(Self { config: config.clone() })
    }

    pub fn config(&self) -> &LstmConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transformer Encoder — Sequence Classification
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Transformer encoder model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerEncoderConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of encoder layers
    pub num_layers: usize,
    /// Dimension of the feedforward network
    pub dim_feedforward: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Feature input size
    pub input_size: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Whether to use GELU activation
    pub use_gelu: bool,
}

impl TransformerEncoderConfig {
    pub fn new(input_size: usize, d_model: usize, num_classes: usize) -> Self {
        Self {
            input_size,
            d_model,
            num_classes,
            num_heads: 8,
            num_layers: 4,
            dim_feedforward: d_model * 4,
            dropout: 0.1,
            max_seq_len: 512,
            use_gelu: true,
        }
    }

    pub fn num_heads(mut self, n: usize) -> Self {
        assert_eq!(self.d_model % n, 0, "d_model must be divisible by num_heads");
        self.num_heads = n;
        self
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    pub fn dim_feedforward(mut self, dim: usize) -> Self {
        self.dim_feedforward = dim;
        self
    }

    pub fn dropout(mut self, p: f64) -> Self {
        self.dropout = p;
        self
    }

    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Estimate parameter count.
    pub fn param_count(&self) -> usize {
        let attn_params = 4 * self.d_model * self.d_model;
        let ff_params = self.d_model * self.dim_feedforward
            + self.dim_feedforward
            + self.dim_feedforward * self.d_model
            + self.d_model;
        let ln_params = 2 * (2 * self.d_model + 2 * self.d_model);
        let encoder_params = (attn_params + ff_params + ln_params) * self.num_layers;
        let emb_params = self.input_size * self.d_model + self.d_model;
        let pos_params = self.max_seq_len * self.d_model;
        let cls_params = self.d_model * self.num_classes + self.num_classes;
        encoder_params + emb_params + pos_params + cls_params
    }
}

/// Transformer encoder model holder (configuration only).
pub struct TransformerModel {
    config: TransformerEncoderConfig,
}

impl TransformerModel {
    pub fn new(config: &TransformerEncoderConfig) -> Result<Self> {
        info!(
            "TransformerModel created: d_model={}, heads={}, layers={}, classes={}, params={}",
            config.d_model,
            config.num_heads,
            config.num_layers,
            config.num_classes,
            config.param_count(),
        );
        Ok(Self { config: config.clone() })
    }

    pub fn config(&self) -> &TransformerEncoderConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Autoencoder — Anomaly Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the autoencoder model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoencoderConfig {
    /// Input/output dimension
    pub input_dim: usize,
    /// Encoder hidden dimensions (bottleneck is the last element)
    pub encoder_dims: Vec<usize>,
    /// Decoder hidden dimensions
    pub decoder_dims: Vec<usize>,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
}

impl AutoencoderConfig {
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            encoder_dims: vec![input_dim / 2, input_dim / 4, input_dim / 8],
            decoder_dims: vec![input_dim / 4, input_dim / 2],
            dropout: 0.1,
            use_batch_norm: true,
        }
    }

    pub fn encoder_dims(mut self, dims: Vec<usize>) -> Self {
        self.encoder_dims = dims;
        self
    }

    pub fn dropout(mut self, p: f64) -> Self {
        self.dropout = p;
        self
    }

    /// Get the bottleneck (latent) dimension.
    pub fn latent_dim(&self) -> usize {
        *self.encoder_dims.last().unwrap_or(&self.input_dim)
    }

    /// Estimate parameter count.
    pub fn param_count(&self) -> usize {
        let mut total = 0;
        let mut prev = self.input_dim;
        for &dim in &self.encoder_dims {
            total += prev * dim + dim;
            if self.use_batch_norm { total += dim * 2; }
            prev = dim;
        }
        for &dim in &self.decoder_dims {
            total += prev * dim + dim;
            if self.use_batch_norm { total += dim * 2; }
            prev = dim;
        }
        total += prev * self.input_dim + self.input_dim;
        total
    }
}

/// Autoencoder model holder (configuration only).
pub struct AutoencoderModel {
    config: AutoencoderConfig,
}

impl AutoencoderModel {
    pub fn new(config: &AutoencoderConfig) -> Result<Self> {
        info!(
            "AutoencoderModel created: input={}, latent={}, encoder_layers={}, decoder_layers={}, params={}",
            config.input_dim,
            config.latent_dim(),
            config.encoder_dims.len(),
            config.decoder_dims.len(),
            config.param_count(),
        );
        Ok(Self { config: config.clone() })
    }

    pub fn config(&self) -> &AutoencoderConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP — Tabular Financial Data
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the MLP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer sizes
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: Activation,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
    /// Output activation (for binary classification)
    pub output_activation: Option<Activation>,
}

/// Supported activation functions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    LeakyReLU(f64),
    SELU,
    Mish,
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::ReLU => write!(f, "relu"),
            Activation::GELU => write!(f, "gelu"),
            Activation::Tanh => write!(f, "tanh"),
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::LeakyReLU(n) => write!(f, "leaky_relu({})", n),
            Activation::SELU => write!(f, "selu"),
            Activation::Mish => write!(f, "mish"),
        }
    }
}

impl MlpConfig {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims: vec![256, 128, 64],
            output_dim,
            activation: Activation::ReLU,
            dropout: 0.2,
            use_batch_norm: true,
            output_activation: None,
        }
    }

    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    pub fn activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn dropout(mut self, p: f64) -> Self {
        self.dropout = p;
        self
    }

    pub fn output_activation(mut self, act: Activation) -> Self {
        self.output_activation = Some(act);
        self
    }

    /// Estimate parameter count.
    pub fn param_count(&self) -> usize {
        let mut total = 0;
        let mut prev = self.input_dim;
        for &dim in &self.hidden_dims {
            total += prev * dim + dim;
            if self.use_batch_norm { total += dim * 2; }
            prev = dim;
        }
        total += prev * self.output_dim + self.output_dim;
        total
    }
}

/// MLP model holder (configuration only).
pub struct MlpModel {
    config: MlpConfig,
}

impl MlpModel {
    pub fn new(config: &MlpConfig) -> Result<Self> {
        info!(
            "MlpModel created: input={}, hidden={:?}, output={}, params={}",
            config.input_dim,
            config.hidden_dims,
            config.output_dim,
            config.param_count(),
        );
        Ok(Self { config: config.clone() })
    }

    pub fn config(&self) -> &MlpConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conv1D — Price Pattern Recognition
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Conv1D model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1dConfig {
    pub input_channels: usize,
    pub hidden_channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub output_dim: usize,
    pub dropout: f64,
}

impl Conv1dConfig {
    pub fn new(input_channels: usize, output_dim: usize) -> Self {
        Self {
            input_channels,
            hidden_channels: vec![32, 64],
            kernel_sizes: vec![3, 3],
            output_dim,
            dropout: 0.2,
        }
    }

    pub fn hidden_channels(mut self, channels: Vec<usize>) -> Self {
        self.hidden_channels = channels;
        self
    }

    pub fn kernel_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.kernel_sizes = sizes;
        self
    }

    pub fn dropout(mut self, p: f64) -> Self {
        self.dropout = p;
        self
    }

    /// Estimate parameter count.
    pub fn param_count(&self) -> usize {
        let mut total = 0;
        let mut prev_ch = self.input_channels;
        for (i, &ch) in self.hidden_channels.iter().enumerate() {
            let ks = self.kernel_sizes.get(i).copied().unwrap_or(3);
            // Conv1d params: out_ch * (in_ch * kernel_size + 1)
            total += ch * (prev_ch * ks + 1);
            prev_ch = ch;
        }
        // Final classification head
        total += prev_ch * self.output_dim + self.output_dim;
        total
    }
}

/// Conv1D model holder (configuration only).
pub struct Conv1dModel {
    config: Conv1dConfig,
}

impl Conv1dModel {
    pub fn new(config: &Conv1dConfig) -> Result<Self> {
        info!(
            "Conv1dModel created: input_ch={}, hidden_ch={:?}, output={}, params={}",
            config.input_channels,
            config.hidden_channels,
            config.output_dim,
            config.param_count(),
        );
        Ok(Self { config: config.clone() })
    }

    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelZoo — Pre-configured Architectures
// ─────────────────────────────────────────────────────────────────────────────

/// Supported finance tasks for model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinanceTask {
    PricePrediction,
    VolatilityForecast,
    AnomalyDetection,
    SentimentAnalysis,
    PortfolioOptimization,
    RiskScoring,
    Classification,
    Regression,
}

/// An entry in the model zoo mapping a task to a recommended architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelZooEntry {
    pub task: FinanceTask,
    pub name: String,
    pub description: String,
    pub default_config: String,
}

/// Pre-configured model architectures for common finance tasks.
pub struct ModelZoo;

impl ModelZoo {
    /// Get all available pre-configured models.
    pub fn all() -> Vec<ModelZooEntry> {
        vec![
            ModelZooEntry {
                task: FinanceTask::PricePrediction,
                name: "lstm_price".to_string(),
                description: "LSTM for price time series forecasting".to_string(),
                default_config: "LstmConfig(input_size=10, hidden_size=128, output_size=1, num_layers=3)".to_string(),
            },
            ModelZooEntry {
                task: FinanceTask::VolatilityForecast,
                name: "transformer_vol".to_string(),
                description: "Transformer encoder for volatility prediction".to_string(),
                default_config: "TransformerEncoderConfig(input_size=10, d_model=64, num_classes=1, num_heads=4, num_layers=2)".to_string(),
            },
            ModelZooEntry {
                task: FinanceTask::AnomalyDetection,
                name: "autoencoder_anomaly".to_string(),
                description: "Autoencoder for transaction anomaly detection".to_string(),
                default_config: "AutoencoderConfig(input_dim=20, encoder_dims=[16,8,4])".to_string(),
            },
            ModelZooEntry {
                task: FinanceTask::RiskScoring,
                name: "mlp_risk".to_string(),
                description: "MLP for contract risk scoring".to_string(),
                default_config: "MlpConfig(input_dim=50, output_dim=1, hidden_dims=[128,64,32])".to_string(),
            },
            ModelZooEntry {
                task: FinanceTask::SentimentAnalysis,
                name: "transformer_sentiment".to_string(),
                description: "Transformer encoder for text sentiment classification".to_string(),
                default_config: "TransformerEncoderConfig(input_size=768, d_model=256, num_classes=3, num_heads=8)".to_string(),
            },
            ModelZooEntry {
                task: FinanceTask::Classification,
                name: "mlp_classifier".to_string(),
                description: "MLP for general classification".to_string(),
                default_config: "MlpConfig(input_dim=32, output_dim=5, hidden_dims=[64,32])".to_string(),
            },
        ]
    }

    /// Get the recommended model for a specific finance task.
    pub fn recommend(task: FinanceTask) -> Option<ModelZooEntry> {
        Self::all().into_iter().find(|e| e.task == task)
    }
}
