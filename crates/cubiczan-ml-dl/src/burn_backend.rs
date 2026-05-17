//! # Training Configuration & Infrastructure
//!
//! Provides training configuration, checkpointing, early stopping,
//! learning rate scheduling, and device management utilities.
//!
//! ## Modules
//!
//! - [`TrainingConfig`] — Complete hyperparameter configuration for training runs
//! - [`TrainingState`] — Mutable state tracked during training
//! - [`DeviceManager`] — Device auto-detection and selection
//! - [`OptimizerType`] / [`LrSchedulerType`] — Optimizer and scheduler enums
//! - [`CheckpointConfig`] / [`EarlyStoppingConfig`] — Checkpoint and early stopping config

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn, debug};

// ─────────────────────────────────────────────────────────────────────────────
// Device Type & Management
// ─────────────────────────────────────────────────────────────────────────────

/// Device type for computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// GPU device (with index)
    Gpu(u32),
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Gpu(idx) => write!(f, "gpu:{}", idx),
        }
    }
}

/// GPU device information for auto-detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// CUDA device index (0-based)
    pub index: u32,
    /// Device name as reported by the driver
    pub name: String,
    /// Total VRAM in bytes
    pub total_memory_bytes: u64,
    /// Whether the device supports CUDA
    pub cuda_supported: bool,
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU {}: {} ({} MB, cuda={})",
            self.index,
            self.name,
            self.total_memory_bytes / (1024 * 1024),
            self.cuda_supported
        )
    }
}

/// Manages device auto-detection and selection.
///
/// Provides a single point of device selection for all operations.
#[derive(Debug, Clone)]
pub struct DeviceManager {
    available_gpus: Vec<GpuInfo>,
    preferred_device: DeviceType,
}

impl DeviceManager {
    /// Create a new device manager with automatic GPU detection.
    pub fn new() -> Self {
        let mut manager = Self {
            available_gpus: Vec::new(),
            preferred_device: DeviceType::Cpu,
        };
        // GPU detection is deferred — Candle device detection
        // will be used at runtime when creating tensors.
        info!("DeviceManager initialized (GPU detection deferred to Candle)");
        manager.select_best_device();
        manager
    }

    /// Select the best available device (GPU preferred).
    fn select_best_device(&mut self) {
        if !self.available_gpus.is_empty() {
            let best = self.available_gpus
                .iter()
                .max_by_key(|gpu| gpu.total_memory_bytes)
                .unwrap();
            self.preferred_device = DeviceType::Gpu(best.index);
            info!("Selected GPU {} as preferred device", best.index);
        } else {
            self.preferred_device = DeviceType::Cpu;
            info!("No GPU available, selected CPU");
        }
    }

    /// Get the list of detected GPUs.
    pub fn available_gpus(&self) -> &[GpuInfo] {
        &self.available_gpus
    }

    /// Get the preferred device type.
    pub fn preferred_device(&self) -> &DeviceType {
        &self.preferred_device
    }

    /// Override the preferred device manually.
    pub fn set_preferred_device(&mut self, device: DeviceType) {
        match device {
            DeviceType::Gpu(idx) => {
                if self.available_gpus.iter().any(|g| g.index == idx) {
                    self.preferred_device = device;
                    info!("Manually set preferred device to GPU {}", idx);
                } else {
                    warn!("GPU {} not available, keeping current device", idx);
                }
            }
            DeviceType::Cpu => {
                self.preferred_device = DeviceType::Cpu;
                info!("Manually set preferred device to CPU");
            }
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer Types
// ─────────────────────────────────────────────────────────────────────────────

/// Supported optimizer types for training.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizerType {
    /// Adam optimizer (default for most tasks)
    Adam,
    /// AdamW with decoupled weight decay
    AdamW,
    /// Stochastic Gradient Descent with momentum
    Sgd,
    /// RMSprop optimizer
    RmsProp,
    /// AdaGrad optimizer
    AdaGrad,
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::Adam => write!(f, "Adam"),
            OptimizerType::AdamW => write!(f, "AdamW"),
            OptimizerType::Sgd => write!(f, "SGD"),
            OptimizerType::RmsProp => write!(f, "RMSprop"),
            OptimizerType::AdaGrad => write!(f, "AdaGrad"),
        }
    }
}

impl Default for OptimizerType {
    fn default() -> Self {
        OptimizerType::Adam
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LR Scheduler Types
// ─────────────────────────────────────────────────────────────────────────────

/// Supported learning rate scheduler types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LrSchedulerType {
    /// Constant learning rate (no scheduling)
    Constant,
    /// Step decay: reduce LR by `gamma` every `step_size` epochs
    StepDecay { step_size: usize, gamma: f64 },
    /// Exponential decay: LR = initial * gamma^epoch
    Exponential { gamma: f64 },
    /// Cosine annealing with optional warm restarts
    CosineAnnealing { t_max: usize, eta_min: f64 },
    /// Reduce LR on plateau (when validation loss stops improving)
    ReduceOnPlateau { factor: f64, patience: usize },
    /// Linear warmup followed by cosine decay
    WarmupCosine { warmup_steps: usize, total_steps: usize },
    /// One-cycle learning rate policy
    OneCycle { max_lr: f64, total_steps: usize },
}

impl LrSchedulerType {
    /// Compute the learning rate multiplier for a given step.
    pub fn lr_multiplier(&self, step: usize) -> f64 {
        match self {
            LrSchedulerType::Constant => 1.0,

            LrSchedulerType::StepDecay { step_size, gamma } => {
                let num_steps = step / step_size;
                gamma.powi(num_steps as i32)
            }

            LrSchedulerType::Exponential { gamma } => {
                gamma.powi(step as i32)
            }

            LrSchedulerType::CosineAnnealing { t_max, eta_min } => {
                let phase = step % t_max;
                let ratio = phase as f64 / *t_max as f64;
                *eta_min + (1.0 - *eta_min) * 0.5 * (1.0 + (std::f64::consts::PI * ratio).cos())
            }

            LrSchedulerType::ReduceOnPlateau { .. } => {
                // Requires history — return 1.0; actual logic in TrainingLoop
                1.0
            }

            LrSchedulerType::WarmupCosine { warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    step as f64 / *warmup_steps as f64
                } else {
                    let progress = (step - warmup_steps) as f64 / (*total_steps - warmup_steps) as f64;
                    let progress = progress.min(1.0);
                    0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
                }
            }

            LrSchedulerType::OneCycle { max_lr: _, total_steps } => {
                let half = *total_steps / 2;
                if step < half {
                    step as f64 / half as f64
                } else {
                    let remaining = *total_steps - step;
                    remaining as f64 / half as f64
                }
            }
        }
    }
}

impl Default for LrSchedulerType {
    fn default() -> Self {
        LrSchedulerType::CosineAnnealing {
            t_max: 100,
            eta_min: 1e-6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint & Early Stopping
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for model checkpoint saving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub save_dir: PathBuf,
    /// Save checkpoint every N epochs
    pub save_every_n_epochs: usize,
    /// Maximum number of checkpoints to keep (oldest deleted first)
    pub max_checkpoints: usize,
    /// Whether to save the best model (lowest validation loss)
    pub save_best: bool,
    /// File prefix for checkpoint files
    pub file_prefix: String,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            save_dir: PathBuf::from("./checkpoints"),
            save_every_n_epochs: 5,
            max_checkpoints: 3,
            save_best: true,
            file_prefix: "cubiczan-model".to_string(),
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint configuration with a custom save directory.
    pub fn new(save_dir: PathBuf) -> Self {
        Self {
            save_dir,
            ..Default::default()
        }
    }

    /// Set the checkpoint saving frequency.
    pub fn every_n_epochs(mut self, n: usize) -> Self {
        self.save_every_n_epochs = n;
        self
    }

    /// Set the maximum number of checkpoints to retain.
    pub fn max_checkpoints(mut self, n: usize) -> Self {
        self.max_checkpoints = n;
        self
    }

    /// Enable or disable saving the best model.
    pub fn save_best(mut self, enabled: bool) -> Self {
        self.save_best = enabled;
        self
    }

    /// Generate a checkpoint file path for a given epoch.
    pub fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.save_dir.join(format!(
            "{}_epoch_{}.safetensors",
            self.file_prefix, epoch
        ))
    }

    /// Generate the "best model" checkpoint path.
    pub fn best_model_path(&self) -> PathBuf {
        self.save_dir.join(format!("{}_best.safetensors", self.file_prefix))
    }

    /// Create the checkpoint directory if it doesn't exist.
    pub fn ensure_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.save_dir)
    }
}

/// Configuration for early stopping during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait for improvement before stopping
    pub patience: usize,
    /// Minimum change in monitored metric to qualify as improvement
    pub min_delta: f64,
    /// Whether higher metric values are better (e.g., accuracy)
    pub mode_higher_is_better: bool,
    /// Whether to restore model weights from the best epoch when stopping
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 1e-4,
            mode_higher_is_better: false,
            restore_best_weights: true,
        }
    }
}

impl EarlyStoppingConfig {
    /// Create early stopping with a given patience.
    pub fn patience(patience: usize) -> Self {
        Self {
            patience,
            ..Default::default()
        }
    }

    /// Set the minimum improvement threshold.
    pub fn min_delta(mut self, delta: f64) -> Self {
        self.min_delta = delta;
        self
    }

    /// Set the monitoring mode (higher is better, e.g., for accuracy).
    pub fn higher_is_better(mut self, enabled: bool) -> Self {
        self.mode_higher_is_better = enabled;
        self
    }

    /// Set whether to restore best weights when early stopping triggers.
    pub fn restore_best_weights(mut self, enabled: bool) -> Self {
        self.restore_best_weights = enabled;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Training State
// ─────────────────────────────────────────────────────────────────────────────

/// Mutable state tracked during a training run, used for checkpointing,
/// early stopping, and LR scheduling decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch number (0-based)
    pub current_epoch: usize,
    /// Total number of epochs
    pub total_epochs: usize,
    /// Best validation metric observed so far
    pub best_val_metric: f64,
    /// Epoch number at which the best metric was observed
    pub best_epoch: usize,
    /// Number of epochs without improvement (for early stopping)
    pub epochs_without_improvement: usize,
    /// Current learning rate
    pub current_lr: f64,
    /// Training loss history per epoch
    pub train_loss_history: Vec<f64>,
    /// Validation loss history per epoch
    pub val_loss_history: Vec<f64>,
    /// Whether training should stop
    pub should_stop: bool,
    /// Timestamp when training started
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Timestamp of last epoch completion
    pub last_epoch_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            current_epoch: 0,
            total_epochs: 0,
            best_val_metric: f64::INFINITY,
            best_epoch: 0,
            epochs_without_improvement: 0,
            current_lr: 0.001,
            train_loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            should_stop: false,
            started_at: chrono::Utc::now(),
            last_epoch_at: None,
        }
    }
}

impl TrainingState {
    /// Create a new training state with the given total epochs.
    pub fn new(total_epochs: usize, initial_lr: f64) -> Self {
        Self {
            total_epochs,
            current_lr: initial_lr,
            started_at: chrono::Utc::now(),
            ..Default::default()
        }
    }

    /// Check if the given metric is an improvement over the best so far.
    pub fn is_improvement(&self, metric: f64, higher_is_better: bool, min_delta: f64) -> bool {
        if higher_is_better {
            metric > self.best_val_metric + min_delta
        } else {
            metric < self.best_val_metric - min_delta
        }
    }

    /// Update state with new epoch results.
    pub fn update(
        &mut self,
        epoch: usize,
        train_loss: f64,
        val_loss: f64,
        early_stop: &EarlyStoppingConfig,
    ) {
        self.current_epoch = epoch;
        self.last_epoch_at = Some(chrono::Utc::now());
        self.train_loss_history.push(train_loss);
        self.val_loss_history.push(val_loss);

        let val_metric = if early_stop.mode_higher_is_better { val_loss } else { val_loss };

        if self.is_improvement(val_metric, early_stop.mode_higher_is_better, early_stop.min_delta) {
            self.best_val_metric = val_metric;
            self.best_epoch = epoch;
            self.epochs_without_improvement = 0;
            debug!(
                "Epoch {}: New best val loss {:.6} (previous {:.6})",
                epoch, val_metric, self.best_val_metric
            );
        } else {
            self.epochs_without_improvement += 1;
            debug!(
                "Epoch {}: No improvement for {} epochs (best: {:.6}, current: {:.6})",
                epoch, self.epochs_without_improvement, self.best_val_metric, val_metric
            );
        }

        // Early stopping check
        if self.epochs_without_improvement >= early_stop.patience {
            self.should_stop = true;
            warn!(
                "Early stopping triggered at epoch {} (patience={}, no improvement for {} epochs)",
                epoch, early_stop.patience, self.epochs_without_improvement
            );
        }
    }

    /// Get the elapsed training time.
    pub fn elapsed(&self) -> chrono::Duration {
        let now = self.last_epoch_at.unwrap_or_else(chrono::Utc::now);
        now.signed_duration_since(self.started_at)
    }

    /// Get average training loss.
    pub fn avg_train_loss(&self) -> f64 {
        if self.train_loss_history.is_empty() {
            return f64::INFINITY;
        }
        self.train_loss_history.iter().sum::<f64>() / self.train_loss_history.len() as f64
    }

    /// Get average validation loss.
    pub fn avg_val_loss(&self) -> f64 {
        if self.val_loss_history.is_empty() {
            return f64::INFINITY;
        }
        self.val_loss_history.iter().sum::<f64>() / self.val_loss_history.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Complete configuration for a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Total number of training epochs
    pub epochs: usize,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Learning rate weight decay (for AdamW, SGD)
    pub weight_decay: f64,
    /// Gradient clipping norm (0.0 = disabled)
    pub gradient_clipping: f32,
    /// Learning rate scheduler
    pub lr_scheduler: LrSchedulerType,
    /// Checkpoint configuration
    pub checkpoint: Option<CheckpointConfig>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Number of validation batches to log per epoch
    pub log_frequency: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of data loading worker threads
    pub num_workers: usize,
    /// Whether to use mixed precision training
    pub mixed_precision: bool,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            optimizer: OptimizerType::Adam,
            weight_decay: 0.01,
            gradient_clipping: 1.0,
            lr_scheduler: LrSchedulerType::default(),
            checkpoint: None,
            early_stopping: None,
            log_frequency: 10,
            seed: 42,
            num_workers: 4,
            mixed_precision: false,
            gradient_accumulation_steps: 1,
        }
    }
}

impl TrainingConfig {
    /// Create a new training config with the essential parameters.
    pub fn new(learning_rate: f64, batch_size: usize, epochs: usize) -> Self {
        Self {
            learning_rate,
            batch_size,
            epochs,
            ..Default::default()
        }
    }

    /// Set the optimizer type.
    pub fn with_optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set weight decay for regularization.
    pub fn with_weight_decay(mut self, decay: f64) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Set gradient clipping norm. Use 0.0 to disable.
    pub fn with_gradient_clipping(mut self, max_norm: f32) -> Self {
        self.gradient_clipping = max_norm;
        self
    }

    /// Set the learning rate scheduler.
    pub fn with_lr_scheduler(mut self, scheduler: LrSchedulerType) -> Self {
        self.lr_scheduler = scheduler;
        self
    }

    /// Set checkpoint configuration.
    pub fn with_checkpoint(mut self, config: CheckpointConfig) -> Self {
        self.checkpoint = Some(config);
        self
    }

    /// Set early stopping configuration.
    pub fn with_early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Set the logging frequency (in batches).
    pub fn with_log_frequency(mut self, freq: usize) -> Self {
        self.log_frequency = freq;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the number of data loading workers.
    pub fn with_num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Enable mixed precision training (FP16).
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Set gradient accumulation steps.
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = steps;
        self
    }

    /// Compute the effective batch size.
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size * self.gradient_accumulation_steps
    }

    /// Compute the current learning rate based on the scheduler and step.
    pub fn current_learning_rate(&self, step: usize) -> f64 {
        self.learning_rate * self.lr_scheduler.lr_multiplier(step)
    }

    /// Validate the configuration and return any issues.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            warnings.push(format!(
                "Unusual learning rate: {}. Typical range is [1e-6, 1e-1].",
                self.learning_rate
            ));
        }

        if self.batch_size == 0 {
            warnings.push("Batch size cannot be zero.".to_string());
        }

        if self.epochs == 0 {
            warnings.push("Number of epochs cannot be zero.".to_string());
        }

        if self.gradient_accumulation_steps == 0 {
            warnings.push("Gradient accumulation steps cannot be zero.".to_string());
        }

        if let Some(ref es) = self.early_stopping {
            if es.patience >= self.epochs {
                warnings.push(format!(
                    "Early stopping patience ({}) >= total epochs ({}). Consider reducing patience.",
                    es.patience, self.epochs
                ));
            }
        }

        warnings
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Loading Abstractions
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for data loaders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    /// Batch size
    pub batch_size: usize,
    /// Number of worker threads
    pub num_workers: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 4,
            shuffle: true,
            drop_last: false,
        }
    }
}

/// A trait for datasets that can produce labeled items.
pub trait BurnDataset: Send + Sync {
    /// Type of each training item
    type Item: Send + Sync + Clone;

    /// Get a single item by index
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Total number of items in the dataset
    fn len(&self) -> usize;

    /// Whether the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple in-memory dataset for numeric features and labels.
#[derive(Debug, Clone)]
pub struct NumericDataset {
    features: Vec<Vec<f32>>,
    labels: Vec<f32>,
}

impl NumericDataset {
    /// Create a new numeric dataset from features and labels.
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<f32>) -> Self {
        assert_eq!(features.len(), labels.len(), "Features and labels must have the same length");
        Self { features, labels }
    }

    /// Create from f64 data with conversion.
    pub fn from_f64(features: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {
        Self {
            features: features.into_iter().map(|v| v.into_iter().map(|x| x as f32).collect()).collect(),
            labels: labels.into_iter().map(|x| x as f32).collect(),
        }
    }

    /// Split into train and test datasets.
    pub fn train_test_split(&self, ratio: f64) -> (Self, Self) {
        assert!(ratio > 0.0 && ratio < 1.0);
        let split_idx = ((1.0 - ratio) * self.features.len() as f64) as usize;

        let (train_feat, test_feat) = self.features.split_at(split_idx);
        let (train_lab, test_lab) = self.labels.split_at(split_idx);

        (
            Self {
                features: train_feat.to_vec(),
                labels: train_lab.to_vec(),
            },
            Self {
                features: test_feat.to_vec(),
                labels: test_lab.to_vec(),
            },
        )
    }

    /// Get the feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.features.first().map(|f| f.len()).unwrap_or(0)
    }

    /// Get the features as a slice of slices.
    pub fn features(&self) -> &[Vec<f32>] {
        &self.features
    }

    /// Get the labels as a slice.
    pub fn labels(&self) -> &[f32] {
        &self.labels
    }
}
