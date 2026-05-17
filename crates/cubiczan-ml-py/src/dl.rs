//! # Deep Learning Configuration Bindings
//!
//! PyO3 Python bindings for deep learning model configurations and training
//! configuration from `cubiczan-ml-dl`.
//!
//! Exposes config types with builder-pattern methods and parameter estimation:
//! - [`PyLstmConfig`] — LSTM for time series forecasting
//! - [`PyTransformerConfig`] — Transformer encoder for sequence tasks
//! - [`PyAutoencoderConfig`] — Autoencoder for anomaly detection
//! - [`PyTrainingConfig`] — Complete training hyperparameter configuration

use pyo3::prelude::*;

use cubiczan_ml_dl::{
    AutoencoderConfig, LstmConfig, TrainingConfig, TransformerEncoderConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// PyLstmConfig
// ─────────────────────────────────────────────────────────────────────────────

/// LSTM model configuration with builder-pattern methods.
///
/// Wraps [`LstmConfig`] to expose it to Python.
///
/// ## Example
///
/// ```python
/// cfg = LstmConfig(10, 128, 1).num_layers(3).dropout(0.2)
/// print(cfg.param_count())
/// ```
#[pyclass(name = "LstmConfig")]
#[derive(Clone)]
pub struct PyLstmConfig {
    config: LstmConfig,
}

#[pymethods]
impl PyLstmConfig {
    /// Create a new LSTM configuration.
    ///
    /// Args:
    ///     input_size: Number of input features.
    ///     hidden_size: Hidden state dimension.
    ///     output_size: Output dimension.
    #[new]
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            config: LstmConfig::new(input_size, hidden_size, output_size),
        }
    }

    /// Set the number of LSTM layers (default: 2).
    ///
    /// Returns self for chaining.
    fn num_layers(mut slf: PyRefMut<'_, Self>, n: usize) -> PyRefMut<'_, Self> {
        slf.config = slf.config.clone().num_layers(n);
        slf
    }

    /// Set the dropout probability (default: 0.2).
    ///
    /// Returns self for chaining.
    fn dropout(mut slf: PyRefMut<'_, Self>, p: f64) -> PyRefMut<'_, Self> {
        slf.config = slf.config.clone().dropout(p);
        slf
    }

    /// Estimate the total number of trainable parameters.
    fn param_count(&self) -> usize {
        self.config.param_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "LstmConfig(input_size={}, hidden_size={}, output_size={}, num_layers={}, dropout={}, params={})",
            self.config.input_size,
            self.config.hidden_size,
            self.config.output_size,
            self.config.num_layers,
            self.config.dropout,
            self.config.param_count(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTransformerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Transformer encoder model configuration with builder-pattern methods.
///
/// Wraps [`TransformerEncoderConfig`] to expose it to Python.
///
/// ## Example
///
/// ```python
/// cfg = TransformerConfig(10, 64, 2).num_heads(4).num_layers(2)
/// print(cfg.param_count())
/// ```
#[pyclass(name = "TransformerConfig")]
#[derive(Clone)]
pub struct PyTransformerConfig {
    config: TransformerEncoderConfig,
}

#[pymethods]
impl PyTransformerConfig {
    /// Create a new Transformer encoder configuration.
    ///
    /// Args:
    ///     input_size: Feature input size.
    ///     d_model: Model dimension.
    ///     num_classes: Number of output classes.
    #[new]
    fn new(input_size: usize, d_model: usize, num_classes: usize) -> Self {
        Self {
            config: TransformerEncoderConfig::new(input_size, d_model, num_classes),
        }
    }

    /// Set the number of attention heads (default: 8).
    ///
    /// `d_model` must be divisible by `n`.
    ///
    /// Returns self for chaining.
    fn num_heads(mut slf: PyRefMut<'_, Self>, n: usize) -> PyResult<PyRefMut<'_, Self>> {
        if slf.config.d_model % n != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                slf.config.d_model, n
            )));
        }
        slf.config.num_heads = n;
        Ok(slf)
    }

    /// Set the number of encoder layers (default: 4).
    ///
    /// Returns self for chaining.
    fn num_layers(mut slf: PyRefMut<'_, Self>, n: usize) -> PyRefMut<'_, Self> {
        slf.config = slf.config.clone().num_layers(n);
        slf
    }

    /// Estimate the total number of trainable parameters.
    fn param_count(&self) -> usize {
        self.config.param_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "TransformerConfig(input_size={}, d_model={}, num_classes={}, num_heads={}, num_layers={}, params={})",
            self.config.input_size,
            self.config.d_model,
            self.config.num_classes,
            self.config.num_heads,
            self.config.num_layers,
            self.config.param_count(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyAutoencoderConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Autoencoder model configuration with builder-pattern methods.
///
/// Wraps [`AutoencoderConfig`] to expose it to Python.
///
/// ## Example
///
/// ```python
/// cfg = AutoencoderConfig(32).encoder_dims([16, 8, 4])
/// print(cfg.latent_dim())  # 4
/// print(cfg.param_count())
/// ```
#[pyclass(name = "AutoencoderConfig")]
#[derive(Clone)]
pub struct PyAutoencoderConfig {
    config: AutoencoderConfig,
}

#[pymethods]
impl PyAutoencoderConfig {
    /// Create a new Autoencoder configuration.
    ///
    /// Args:
    ///     input_dim: Input/output dimension.
    #[new]
    fn new(input_dim: usize) -> Self {
        Self {
            config: AutoencoderConfig::new(input_dim),
        }
    }

    /// Set the encoder hidden dimensions.
    ///
    /// The bottleneck (latent) dimension is the last element.
    ///
    /// Returns self for chaining.
    fn encoder_dims(mut slf: PyRefMut<'_, Self>, dims: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.config = slf.config.clone().encoder_dims(dims);
        slf
    }

    /// Get the bottleneck (latent) dimension.
    ///
    /// Returns the last element of `encoder_dims`, or `input_dim` if empty.
    fn latent_dim(&self) -> usize {
        self.config.latent_dim()
    }

    /// Estimate the total number of trainable parameters.
    fn param_count(&self) -> usize {
        self.config.param_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "AutoencoderConfig(input_dim={}, encoder_dims={:?}, latent_dim={}, params={})",
            self.config.input_dim,
            self.config.encoder_dims,
            self.config.latent_dim(),
            self.config.param_count(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTrainingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Training configuration with builder-pattern methods.
///
/// Wraps [`TrainingConfig`] to expose it to Python.
///
/// ## Example
///
/// ```python
/// cfg = TrainingConfig().learning_rate(0.001).batch_size(64).epochs(50)
/// ```
#[pyclass(name = "TrainingConfig")]
#[derive(Clone)]
pub struct PyTrainingConfig {
    config: TrainingConfig,
}

#[pymethods]
impl PyTrainingConfig {
    /// Create a new training configuration with sensible defaults.
    ///
    /// Defaults: learning_rate=0.001, batch_size=32, epochs=100, optimizer=Adam.
    #[new]
    fn new() -> Self {
        Self {
            config: TrainingConfig::default(),
        }
    }

    /// Set the learning rate (default: 0.001).
    ///
    /// Returns self for chaining.
    fn learning_rate(mut slf: PyRefMut<'_, Self>, lr: f64) -> PyRefMut<'_, Self> {
        slf.config.learning_rate = lr;
        slf
    }

    /// Set the batch size (default: 32).
    ///
    /// Returns self for chaining.
    fn batch_size(mut slf: PyRefMut<'_, Self>, bs: usize) -> PyRefMut<'_, Self> {
        slf.config.batch_size = bs;
        slf
    }

    /// Set the total number of training epochs (default: 100).
    ///
    /// Returns self for chaining.
    fn epochs(mut slf: PyRefMut<'_, Self>, e: usize) -> PyRefMut<'_, Self> {
        slf.config.epochs = e;
        slf
    }

    /// Compute the effective batch size (batch_size × gradient_accumulation_steps).
    fn effective_batch_size(&self) -> usize {
        self.config.effective_batch_size()
    }

    /// Get the current learning rate at a given training step (accounting for LR scheduling).
    fn current_learning_rate(&self, step: usize) -> f64 {
        self.config.current_learning_rate(step)
    }

    /// Validate the configuration and return a list of warnings.
    fn validate(&self) -> Vec<String> {
        self.config.validate()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingConfig(learning_rate={}, batch_size={}, epochs={}, optimizer={}, seed={}, mixed_precision={})",
            self.config.learning_rate,
            self.config.batch_size,
            self.config.epochs,
            self.config.optimizer,
            self.config.seed,
            self.config.mixed_precision,
        )
    }
}
