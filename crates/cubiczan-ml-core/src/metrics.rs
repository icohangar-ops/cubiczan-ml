//! # ML Metrics
//!
//! Standard metrics tracked during model training and evaluation.

use serde::{Deserialize, Serialize};

/// Aggregated metrics produced during training or evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    /// Current loss value.
    pub loss: f64,
    /// Current accuracy (0.0–1.0), or `None` if not applicable.
    pub accuracy: Option<f64>,
    /// Current epoch number (0-indexed).
    pub epoch: usize,
    /// Current step / batch within the epoch.
    pub step: usize,
    /// Wall-clock elapsed time in seconds since training began.
    pub elapsed_secs: f64,
    /// Learning rate used at this point.
    pub learning_rate: Option<f64>,
    /// Additional named metric values (e.g. "precision", "recall", "f1").
    pub extra: std::collections::HashMap<String, f64>,
}

impl Metrics {
    /// Create a new empty `Metrics`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for loss.
    pub fn with_loss(mut self, loss: f64) -> Self {
        self.loss = loss;
        self
    }

    /// Builder-style setter for accuracy.
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    /// Builder-style setter for epoch.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }

    /// Builder-style setter for step.
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Insert an extra metric.
    pub fn set_extra(&mut self, key: impl Into<String>, value: f64) {
        self.extra.insert(key.into(), value);
    }

    /// Retrieve an extra metric by name.
    pub fn get_extra(&self, key: &str) -> Option<f64> {
        self.extra.get(key).copied()
    }
}
