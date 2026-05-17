//! # Normalization Statistics
//!
//! Stores per-feature statistics produced by fitting a normalizer.

use serde::{Deserialize, Serialize};

/// Per-feature statistics computed during normalization fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    /// Per-feature mean values.
    pub mean: Vec<f64>,
    /// Per-feature standard deviation values.
    pub std_dev: Vec<f64>,
    /// Number of samples used to compute the statistics.
    pub n_samples: usize,
}

impl NormalizationStats {
    /// Compute normalization statistics from column-major data.
    ///
    /// * `data` — A 2-D slice `&[Vec<f64>]` where each inner `Vec` is a column (feature).
    /// * Returns an error if `data` is empty or rows are inconsistent.
    pub fn compute(data: &[Vec<f64>]) -> crate::Result<Self> {
        if data.is_empty() {
            return Err(crate::MlError::InvalidInput(
                "no features provided".into(),
            ));
        }
        let n_samples = data[0].len();
        if n_samples == 0 {
            return Err(crate::MlError::InvalidInput(
                "no samples provided".into(),
            ));
        }
        for (i, col) in data.iter().enumerate() {
            if col.len() != n_samples {
                return Err(crate::MlError::ShapeMismatch {
                    expected: format!("{} samples", n_samples),
                    actual: format!("{} samples in column {}", col.len(), i),
                });
            }
        }

        let mut mean = Vec::with_capacity(data.len());
        let mut std_dev = Vec::with_capacity(data.len());

        for col in data {
            let sum: f64 = col.iter().sum();
            let m = sum / n_samples as f64;
            let variance =
                col.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n_samples - 1) as f64;
            let sd = variance.sqrt();
            mean.push(m);
            std_dev.push(if sd.abs() < 1e-15 { 1.0 } else { sd });
        }

        Ok(NormalizationStats {
            mean,
            std_dev,
            n_samples,
        })
    }

    /// Normalize a single sample in-place using the stored statistics.
    ///
    /// The sample length must match the number of features.
    pub fn normalize(&self, sample: &mut [f64]) -> crate::Result<()> {
        if sample.len() != self.mean.len() {
            return Err(crate::MlError::ShapeMismatch {
                expected: format!("{} features", self.mean.len()),
                actual: format!("{} values", sample.len()),
            });
        }
        for (i, v) in sample.iter_mut().enumerate() {
            *v = (*v - self.mean[i]) / self.std_dev[i];
        }
        Ok(())
    }

    /// Denormalize a single sample in-place (inverse of `normalize`).
    pub fn denormalize(&self, sample: &mut [f64]) -> crate::Result<()> {
        if sample.len() != self.mean.len() {
            return Err(crate::MlError::ShapeMismatch {
                expected: format!("{} features", self.mean.len()),
                actual: format!("{} values", sample.len()),
            });
        }
        for (i, v) in sample.iter_mut().enumerate() {
            *v = *v * self.std_dev[i] + self.mean[i];
        }
        Ok(())
    }
}

impl Default for NormalizationStats {
    fn default() -> Self {
        Self {
            mean: Vec::new(),
            std_dev: Vec::new(),
            n_samples: 0,
        }
    }
}
