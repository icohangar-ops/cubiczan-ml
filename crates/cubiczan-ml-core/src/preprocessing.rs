//! # Data Preprocessing
//!
//! Scalable data preprocessing tools for ML pipelines in the CubicZan ecosystem.
//! All scalers support fit/transform/inverse_transform workflows and are
//! serializable with `serde`.
//!
//! ## Components
//!
//! - **MinMaxScaler** — Scale features to a given range [min, max]
//! - **StandardScaler** — Z-score normalization (mean=0, std=1)
//! - **RobustScaler** — Median/IQR-based scaling, robust to outliers
//! - **LabelEncoder** — Encode categorical labels as integers
//! - **OneHotEncoder** — One-hot encode categorical features
//! - **Train/Test Split** — Random and stratified splitting
//! - **Feature Engineering** — Lag features, rolling statistics, cross-features

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during preprocessing.
#[derive(Debug, Error)]
pub enum PreprocessingError {
    #[error("not fitted: call fit() before transform()")]
    NotFitted,
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("empty data provided")]
    EmptyData,
    #[error("unknown category: '{category}' not seen during fit")]
    UnknownCategory { category: String },
    #[error("zero IQR encountered (constant column)")]
    ZeroIQR,
    #[error("invalid parameter: {reason}")]
    InvalidParam { reason: String },
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/// Scales features to a specified range, typically [0, 1].
///
/// For each feature (column), computes:
/// ```text
/// X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    /// Per-feature minimum values.
    pub min_: Vec<f64>,
    /// Per-feature maximum values.
    pub max_: Vec<f64>,
    /// Per-feature ranges (max - min).
    pub range_: Vec<f64>,
    /// Target range minimum.
    pub feature_range_min: f64,
    /// Target range maximum.
    pub feature_range_max: f64,
    /// Number of features seen during fit.
    pub n_features_: usize,
    /// Whether the scaler has been fitted.
    pub fitted: bool,
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler with range [0, 1].
    pub fn new() -> Self {
        MinMaxScaler {
            min_: Vec::new(),
            max_: Vec::new(),
            range_: Vec::new(),
            feature_range_min: 0.0,
            feature_range_max: 1.0,
            n_features_: 0,
            fitted: false,
        }
    }

    /// Create a new MinMaxScaler with a custom range.
    pub fn with_range(min: f64, max: f64) -> Result<Self, PreprocessingError> {
        if min >= max {
            return Err(PreprocessingError::InvalidParam {
                reason: format!("feature_range_min ({}) must be < feature_range_max ({})", min, max),
            });
        }
        Ok(MinMaxScaler {
            min_: Vec::new(),
            max_: Vec::new(),
            range_: Vec::new(),
            feature_range_min: min,
            feature_range_max: max,
            n_features_: 0,
            fitted: false,
        })
    }

    /// Fit the scaler to the data.
    ///
    /// Computes per-feature min, max, and range from the training data.
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), PreprocessingError> {
        let (_, n_cols) = data.dim();
        if n_cols == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        let n_rows = data.nrows();
        if n_rows == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        self.n_features_ = n_cols;
        self.min_ = Vec::with_capacity(n_cols);
        self.max_ = Vec::with_capacity(n_cols);
        self.range_ = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let column = data.column(col);
            let col_min = column.iter().cloned().fold(f64::INFINITY, f64::min);
            let col_max = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let col_range = col_max - col_min;

            self.min_.push(col_min);
            self.max_.push(col_max);
            self.range_.push(col_range);
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform the data using the fitted scaler.
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_features_,
                actual: n_cols,
            });
        }

        let mut result = data.clone();

        for col in 0..n_cols {
            for row in 0..n_rows {
                let val = data[[row, col]];
                let scaled = if self.range_[col].abs() < 1e-15 {
                    // Constant feature: scale to midpoint of target range
                    (self.feature_range_min + self.feature_range_max) / 2.0
                } else {
                    (val - self.min_[col]) / self.range_[col]
                        * (self.feature_range_max - self.feature_range_min)
                        + self.feature_range_min
                };
                result[[row, col]] = scaled;
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform: convert scaled data back to original scale.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_features_,
                actual: n_cols,
            });
        }

        let mut result = data.clone();

        for col in 0..n_cols {
            for row in 0..n_rows {
                let val = data[[row, col]];
                let original = if self.range_[col].abs() < 1e-15 {
                    self.min_[col]
                } else {
                    (val - self.feature_range_min)
                        / (self.feature_range_max - self.feature_range_min)
                        * self.range_[col]
                        + self.min_[col]
                };
                result[[row, col]] = original;
            }
        }

        Ok(result)
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/// Standardizes features by removing the mean and scaling to unit variance.
///
/// ```text
/// z = (X - μ) / σ
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    /// Per-feature mean.
    pub mean_: Vec<f64>,
    /// Per-feature standard deviation.
    pub std_: Vec<f64>,
    /// Whether to use the population std (divide by n) instead of sample std (divide by n-1).
    pub with_std: bool,
    /// Number of features seen during fit.
    pub n_features_: usize,
    /// Whether the scaler has been fitted.
    pub fitted: bool,
}

impl StandardScaler {
    /// Create a new StandardScaler.
    pub fn new() -> Self {
        StandardScaler {
            mean_: Vec::new(),
            std_: Vec::new(),
            with_std: true,
            n_features_: 0,
            fitted: false,
        }
    }

    /// Create with option to disable std scaling (only center).
    pub fn with_std(with_std: bool) -> Self {
        StandardScaler {
            with_std,
            ..Self::new()
        }
    }

    /// Fit the scaler.
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), PreprocessingError> {
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        self.n_features_ = n_cols;
        self.mean_ = Vec::with_capacity(n_cols);
        self.std_ = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let column = data.column(col);
            let sum: f64 = column.iter().sum();
            let mean = sum / n_rows as f64;

            let variance: f64 = if self.with_std {
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_rows - 1) as f64
            } else {
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_rows as f64
            };

            let std = variance.sqrt();

            self.mean_.push(mean);
            self.std_.push(if std.abs() < 1e-15 { 1.0 } else { std });
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform the data.
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_features_,
                actual: n_cols,
            });
        }

        let mut result = data.clone();

        for col in 0..n_cols {
            for row in 0..n_rows {
                result[[row, col]] = (data[[row, col]] - self.mean_[col]) / self.std_[col];
            }
        }

        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_features_,
                actual: n_cols,
            });
        }

        let mut result = data.clone();

        for col in 0..n_cols {
            for row in 0..n_rows {
                result[[row, col]] = data[[row, col]] * self.std_[col] + self.mean_[col];
            }
        }

        Ok(result)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

/// Scales features using statistics that are robust to outliers.
///
/// Uses the median and interquartile range (IQR):
/// ```text
/// X_scaled = (X - median) / IQR
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustScaler {
    /// Per-feature median.
    pub median_: Vec<f64>,
    /// Per-feature IQR (Q75 - Q25).
    pub iqr_: Vec<f64>,
    /// Quantile range to use (default [25.0, 75.0] for IQR).
    pub quantile_range: (f64, f64),
    /// Number of features seen during fit.
    pub n_features_: usize,
    /// Whether the scaler has been fitted.
    pub fitted: bool,
}

impl RobustScaler {
    /// Create a new RobustScaler with default IQR range.
    pub fn new() -> Self {
        RobustScaler {
            median_: Vec::new(),
            iqr_: Vec::new(),
            quantile_range: (25.0, 75.0),
            n_features_: 0,
            fitted: false,
        }
    }

    /// Create with custom quantile range.
    pub fn with_quantile_range(low: f64, high: f64) -> Self {
        RobustScaler {
            quantile_range: (low, high),
            ..Self::new()
        }
    }

    /// Compute a quantile from a 1-D array using linear interpolation.
    fn quantile(data: &[f64], q: f64) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted[0];
        }
        let idx = q / 100.0 * (n as f64 - 1.0);
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let frac = idx - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }

    /// Fit the scaler.
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), PreprocessingError> {
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        self.n_features_ = n_cols;
        self.median_ = Vec::with_capacity(n_cols);
        self.iqr_ = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let column: Vec<f64> = data.column(col).to_vec();
            let median = Self::quantile(&column, 50.0);
            let q_low = Self::quantile(&column, self.quantile_range.0);
            let q_high = Self::quantile(&column, self.quantile_range.1);
            let iqr = q_high - q_low;

            self.median_.push(median);
            self.iqr_.push(if iqr.abs() < 1e-15 { 1.0 } else { iqr });
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform the data.
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_features_,
                actual: n_cols,
            });
        }

        let mut result = data.clone();

        for col in 0..n_cols {
            for row in 0..n_rows {
                result[[row, col]] = (data[[row, col]] - self.median_[col]) / self.iqr_[col];
            }
        }

        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_features_,
                actual: n_cols,
            });
        }

        let mut result = data.clone();

        for col in 0..n_cols {
            for row in 0..n_rows {
                result[[row, col]] = data[[row, col]] * self.iqr_[col] + self.median_[col];
            }
        }

        Ok(result)
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LabelEncoder
// ---------------------------------------------------------------------------

/// Encodes categorical labels as integer values.
///
/// Maps each unique label to an integer starting from 0.
/// Labels are sorted alphabetically for deterministic encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEncoder {
    /// Map from label string to integer code.
    pub classes_: Vec<String>,
    /// Whether the encoder has been fitted.
    pub fitted: bool,
}

impl LabelEncoder {
    /// Create a new LabelEncoder.
    pub fn new() -> Self {
        LabelEncoder {
            classes_: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the encoder to a list of labels.
    pub fn fit(&mut self, labels: &[&str]) -> Result<(), PreprocessingError> {
        if labels.is_empty() {
            return Err(PreprocessingError::EmptyData);
        }

        let mut unique: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        unique.sort();
        unique.dedup();

        self.classes_ = unique;
        self.fitted = true;
        Ok(())
    }

    /// Transform labels to integer codes.
    pub fn transform(&self, labels: &[&str]) -> Result<Vec<usize>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }

        labels
            .iter()
            .map(|label| {
                self.classes_
                    .iter()
                    .position(|c| c == *label)
                    .ok_or_else(|| PreprocessingError::UnknownCategory {
                        category: (*label).to_string(),
                    })
            })
            .collect()
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, labels: &[&str]) -> Result<Vec<usize>, PreprocessingError> {
        self.fit(labels)?;
        self.transform(labels)
    }

    /// Inverse transform: convert integer codes back to label strings.
    pub fn inverse_transform(&self, codes: &[usize]) -> Result<Vec<String>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }

        codes
            .iter()
            .map(|&code| {
                self.classes_.get(code).cloned().ok_or_else(|| {
                    PreprocessingError::InvalidParam {
                        reason: format!("code {} out of range [0, {})", code, self.classes_.len()),
                    }
                })
            })
            .collect()
    }

    /// Number of unique classes.
    pub fn n_classes(&self) -> usize {
        self.classes_.len()
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

/// One-hot encodes categorical features.
///
/// Creates binary columns for each unique category value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneHotEncoder {
    /// Categories seen for each feature during fit.
    pub categories_: Vec<Vec<String>>,
    /// Number of input features.
    pub n_input_features_: usize,
    /// Whether the encoder has been fitted.
    pub fitted: bool,
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder.
    pub fn new() -> Self {
        OneHotEncoder {
            categories_: Vec::new(),
            n_input_features_: 0,
            fitted: false,
        }
    }

    /// Fit the encoder.
    ///
    /// `data` is a 2-D array where each row is a sample and each column is a
    /// categorical feature (represented as f64, but values are treated as string keys).
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), PreprocessingError> {
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        self.n_input_features_ = n_cols;
        self.categories_ = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let mut unique: Vec<String> = (0..n_rows)
                .map(|row| format!("{}", data[[row, col]]))
                .collect();
            unique.sort();
            unique.dedup();
            self.categories_.push(unique);
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform data to one-hot encoded representation.
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        if !self.fitted {
            return Err(PreprocessingError::NotFitted);
        }
        let (n_rows, n_cols) = data.dim();
        if n_cols != self.n_input_features_ {
            return Err(PreprocessingError::DimensionMismatch {
                expected: self.n_input_features_,
                actual: n_cols,
            });
        }

        // Total output columns
        let n_output_cols: usize = self.categories_.iter().map(|c| c.len()).sum();
        let mut result = Array2::zeros((n_rows, n_output_cols));

        let mut col_offset = 0;
        for input_col in 0..n_cols {
            let cats = &self.categories_[input_col];
            for row in 0..n_rows {
                let val = format!("{}", data[[row, input_col]]);
                if let Some(idx) = cats.iter().position(|c| c == &val) {
                    result[[row, col_offset + idx]] = 1.0;
                }
            }
            col_offset += cats.len();
        }

        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Get the total number of output columns after encoding.
    pub fn n_output_features(&self) -> usize {
        self.categories_.iter().map(|c| c.len()).sum()
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Train/Test Split
// ---------------------------------------------------------------------------

/// Result of a train/test split.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Training indices.
    pub train_indices: Vec<usize>,
    /// Test indices.
    pub test_indices: Vec<usize>,
}

/// Split data into training and test sets.
pub struct TrainTestSplit;

impl TrainTestSplit {
    /// Random split with a given test ratio.
    ///
    /// Uses a simple hash-based shuffle for deterministic results without needing
    /// an external RNG dependency.
    ///
    /// # Arguments
    /// * `n_samples` — Total number of samples.
    /// * `test_ratio` — Fraction of data to use for testing (e.g., 0.2 = 20%).
    /// * `seed` — Random seed for reproducibility.
    pub fn split(n_samples: usize, test_ratio: f64, seed: u64) -> Result<SplitResult, PreprocessingError> {
        if n_samples == 0 {
            return Err(PreprocessingError::EmptyData);
        }
        if !(0.0..1.0).contains(&test_ratio) {
            return Err(PreprocessingError::InvalidParam {
                reason: format!("test_ratio must be in (0, 1), got {}", test_ratio),
            });
        }

        // Generate shuffled indices using a simple LCG
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng_state = seed;
        for i in (1..n_samples).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let test_size = (n_samples as f64 * test_ratio).round() as usize;
        let train_size = n_samples - test_size;

        Ok(SplitResult {
            train_indices: indices[..train_size].to_vec(),
            test_indices: indices[train_size..].to_vec(),
        })
    }

    /// Time-ordered split (no shuffling) for time series data.
    ///
    /// Ensures that training data always comes before test data chronologically.
    pub fn time_split(
        n_samples: usize,
        test_ratio: f64,
    ) -> Result<SplitResult, PreprocessingError> {
        if n_samples == 0 {
            return Err(PreprocessingError::EmptyData);
        }
        if !(0.0..1.0).contains(&test_ratio) {
            return Err(PreprocessingError::InvalidParam {
                reason: format!("test_ratio must be in (0, 1), got {}", test_ratio),
            });
        }

        let test_size = (n_samples as f64 * test_ratio).round() as usize;
        let train_size = n_samples - test_size;

        Ok(SplitResult {
            train_indices: (0..train_size).collect(),
            test_indices: (train_size..n_samples).collect(),
        })
    }

    /// Split a 2-D array into train and test arrays.
    pub fn split_array(
        data: &Array2<f64>,
        test_ratio: f64,
        seed: u64,
    ) -> Result<(Array2<f64>, Array2<f64>), PreprocessingError> {
        let (n_rows, n_cols) = data.dim();
        let split = Self::split(n_rows, test_ratio, seed)?;

        let train_data = Array2::from_shape_vec(
            (split.train_indices.len(), n_cols),
            split
                .train_indices
                .iter()
                .flat_map(|&i| data.row(i).to_vec())
                .collect(),
        ).map_err(|e| PreprocessingError::InvalidParam { reason: format!("shape error: {}", e) })?;

        let test_data = Array2::from_shape_vec(
            (split.test_indices.len(), n_cols),
            split
                .test_indices
                .iter()
                .flat_map(|&i| data.row(i).to_vec())
                .collect(),
        ).map_err(|e| PreprocessingError::InvalidParam { reason: format!("shape error: {}", e) })?;

        Ok((train_data, test_data))
    }
}

// ---------------------------------------------------------------------------
// Feature Engineering
// ---------------------------------------------------------------------------

/// Feature engineering utilities for creating ML features from raw data.
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Create lag features from a 1-D array.
    ///
    /// For each lag in `lags`, appends a new column with the value shifted back
    /// by that many periods.
    ///
    /// # Returns
    /// A 2-D array where the first column is the original data and subsequent
    /// columns are lagged versions. Rows with insufficient history are filled with NaN.
    pub fn add_lags(data: &[f64], lags: &[usize]) -> Result<Array2<f64>, PreprocessingError> {
        if data.is_empty() {
            return Err(PreprocessingError::EmptyData);
        }

        let n = data.len();
        let n_cols = 1 + lags.len();
        let mut result = Array2::from_elem((n, n_cols), f64::NAN);

        // Original data
        for i in 0..n {
            result[[i, 0]] = data[i];
        }

        // Lag columns
        for (col_offset, &lag) in lags.iter().enumerate() {
            let col = col_offset + 1;
            for i in lag..n {
                result[[i, col]] = data[i - lag];
            }
        }

        Ok(result)
    }

    /// Create rolling statistic features.
    ///
    /// For each window in `windows`, computes the specified statistic and appends
    /// as a new column.
    pub fn add_rolling_stats(
        data: &[f64],
        windows: &[usize],
        stat_type: RollingStat,
    ) -> Result<Array2<f64>, PreprocessingError> {
        if data.is_empty() {
            return Err(PreprocessingError::EmptyData);
        }

        let n = data.len();
        let n_cols = 1 + windows.len();
        let mut result = Array2::from_elem((n, n_cols), f64::NAN);

        // Original data
        for i in 0..n {
            result[[i, 0]] = data[i];
        }

        // Rolling stat columns
        for (col_offset, &window) in windows.iter().enumerate() {
            let col = col_offset + 1;
            for i in (window - 1)..n {
                let slice = &data[i + 1 - window..=i];
                result[[i, col]] = match stat_type {
                    RollingStat::Mean => slice.iter().sum::<f64>() / window as f64,
                    RollingStat::Std => {
                        let m = slice.iter().sum::<f64>() / window as f64;
                        let v = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (window - 1) as f64;
                        v.sqrt()
                    }
                    RollingStat::Min => slice.iter().cloned().fold(f64::INFINITY, f64::min),
                    RollingStat::Max => slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    RollingStat::Median => {
                        let mut sorted = slice.to_vec();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        if sorted.len() % 2 == 0 {
                            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                        } else {
                            sorted[sorted.len() / 2]
                        }
                    }
                    RollingStat::Sum => slice.iter().sum(),
                };
            }
        }

        Ok(result)
    }

    /// Create cross features (interaction terms) between columns of a 2-D array.
    ///
    /// For each pair of columns (i, j) where i < j, creates a new column
    /// that is `col_i * col_j`.
    pub fn add_cross_features(data: &Array2<f64>) -> Result<Array2<f64>, PreprocessingError> {
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 || n_cols < 2 {
            return Err(PreprocessingError::EmptyData);
        }

        // Number of pairs = n_cols choose 2
        let n_pairs = n_cols * (n_cols - 1) / 2;
        let total_cols = n_cols + n_pairs;
        let mut result = Array2::zeros((n_rows, total_cols));

        // Copy original columns
        for col in 0..n_cols {
            for row in 0..n_rows {
                result[[row, col]] = data[[row, col]];
            }
        }

        // Add cross features
        let mut out_col = n_cols;
        for i in 0..n_cols {
            for j in (i + 1)..n_cols {
                for row in 0..n_rows {
                    result[[row, out_col]] = data[[row, i]] * data[[row, j]];
                }
                out_col += 1;
            }
        }

        Ok(result)
    }

    /// Create ratio features: col_i / col_j for specified pairs.
    ///
    /// Handles division by zero by returning NaN.
    pub fn add_ratio_features(
        data: &Array2<f64>,
        numerators: &[usize],
        denominators: &[usize],
    ) -> Result<Array2<f64>, PreprocessingError> {
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 {
            return Err(PreprocessingError::EmptyData);
        }

        let n_new = numerators.len().min(denominators.len());
        if n_new == 0 {
            return Err(PreprocessingError::InvalidParam {
                reason: "numerator and denominator lists must not be empty".into(),
            });
        }

        let total_cols = n_cols + n_new;
        let mut result = Array2::zeros((n_rows, total_cols));

        // Copy original
        for col in 0..n_cols {
            for row in 0..n_rows {
                result[[row, col]] = data[[row, col]];
            }
        }

        // Add ratios
        for k in 0..n_new {
            let num_col = numerators[k];
            let den_col = denominators[k];
            if num_col >= n_cols || den_col >= n_cols {
                return Err(PreprocessingError::DimensionMismatch {
                    expected: n_cols,
                    actual: num_col.max(den_col) + 1,
                });
            }
            for row in 0..n_rows {
                let den = data[[row, den_col]];
                result[[row, n_cols + k]] = if den.abs() < 1e-15 {
                    f64::NAN
                } else {
                    data[[row, num_col]] / den
                };
            }
        }

        Ok(result)
    }

    /// Replace NaN values in a 2-D array with a specified fill value.
    pub fn fill_nan(data: &Array2<f64>, fill_value: f64) -> Array2<f64> {
        data.mapv(|v| if v.is_nan() { fill_value } else { v })
    }

    /// Remove rows containing NaN values.
    pub fn drop_nan_rows(data: &Array2<f64>) -> Array2<f64> {
        let (n_rows, n_cols) = data.dim();
        let clean_rows: Vec<usize> = (0..n_rows)
            .filter(|&row| {
                (0..n_cols).all(|col| !data[[row, col]].is_nan())
            })
            .collect();

        if clean_rows.is_empty() {
            return Array2::zeros((0, n_cols));
        }

        Array2::from_shape_vec(
            (clean_rows.len(), n_cols),
            clean_rows
                .iter()
                .flat_map(|&row| data.row(row).to_vec())
                .collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((0, n_cols)))
    }
}

/// Type of rolling statistic to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingStat {
    Mean,
    Std,
    Min,
    Max,
    Median,
    Sum,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper macro for approximate float equality.
    // Usage: assert_relative_eq!(a, b) or assert_relative_eq!(a, b, epsilon = 1e-10)
    macro_rules! assert_relative_eq {
        ($a:expr, $b:expr) => {
            assert_relative_eq!($a, $b, epsilon = 1e-6)
        };
        ($a:expr, $b:expr, epsilon = $eps:expr) => {{
            let a_val = $a;
            let b_val = $b;
            let eps_val = $eps;
            assert!(
                (a_val - b_val).abs() < eps_val,
                "assertion failed: |{} - {}| = {} >= {}",
                a_val, b_val, (a_val - b_val).abs(), eps_val
            );
        }};
    }

    #[test]
    fn test_minmax_scaler() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        )
        .unwrap();

        let mut scaler = MinMaxScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // First column: [1,2,3,4] → [0, 0.333, 0.667, 1]
        assert_relative_eq!(scaled[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[[3, 0]], 1.0, epsilon = 1e-10);

        // Second column: [10,20,30,40] → [0, 0.333, 0.667, 1]
        assert_relative_eq!(scaled[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[[3, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minmax_scaler_inverse() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 5.0, 2.0, 10.0, 3.0, 15.0])
            .unwrap();

        let mut scaler = MinMaxScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        let restored = scaler.inverse_transform(&scaled).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(data[[i, j]], restored[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_minmax_scaler_custom_range() {
        let data = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let mut scaler = MinMaxScaler::with_range(-1.0, 1.0).unwrap();
        let scaled = scaler.fit_transform(&data).unwrap();

        assert_relative_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[[1, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_standard_scaler() {
        // Data: [2, 4, 6, 8] → mean=5, std≈2.582
        let data = Array2::from_shape_vec((4, 1), vec![2.0, 4.0, 6.0, 8.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        assert_relative_eq!(scaler.mean_[0], 5.0, epsilon = 1e-10);
        // Scaled mean should be 0
        let scaled_mean = scaled.column(0).sum() / 4.0;
        assert_relative_eq!(scaled_mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_standard_scaler_inverse() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
            .unwrap();

        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        let restored = scaler.inverse_transform(&scaled).unwrap();

        for i in 0..4 {
            for j in 0..2 {
                assert_relative_eq!(data[[i, j]], restored[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_robust_scaler() {
        // Data with outliers: [1, 2, 3, 4, 5, 100]
        let data = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]).unwrap();

        let mut scaler = RobustScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Median should be 3.5
        assert_relative_eq!(scaler.median_[0], 3.5, epsilon = 1e-10);
        // IQR = Q75 - Q25 = 4.75 - 2.25 = 2.5
        assert_relative_eq!(scaler.iqr_[0], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_robust_scaler_inverse() {
        let data = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]).unwrap();

        let mut scaler = RobustScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        let restored = scaler.inverse_transform(&scaled).unwrap();

        for i in 0..6 {
            assert_relative_eq!(data[[i, 0]], restored[[i, 0]], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_label_encoder() {
        let mut encoder = LabelEncoder::new();
        let labels = ["cat", "dog", "bird", "cat", "dog"];
        let codes = encoder.fit_transform(&labels).unwrap();

        // Sorted: ["bird", "cat", "dog"] → bird=0, cat=1, dog=2
        assert_eq!(codes, vec![1, 2, 0, 1, 2]);
        assert_eq!(encoder.n_classes(), 3);

        let decoded = encoder.inverse_transform(&codes).unwrap();
        assert_eq!(decoded, vec!["cat", "dog", "bird", "cat", "dog"]);
    }

    #[test]
    fn test_label_encoder_unknown() {
        let mut encoder = LabelEncoder::new();
        encoder.fit(&["cat", "dog"]).unwrap();
        let result = encoder.transform(&["cat", "fish"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoder() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 10.0, 2.0, 20.0, 1.0, 10.0, 3.0, 20.0],
        )
        .unwrap();

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Col 0: categories [1, 2, 3] → 3 columns
        // Col 1: categories [10, 20] → 2 columns
        // Total: 5 output columns
        assert_eq!(encoded.dim(), (4, 5));

        // Row 0: col0=1 → [1,0,0], col1=10 → [1,0]
        assert_relative_eq!(encoded[[0, 0]], 1.0);
        assert_relative_eq!(encoded[[0, 1]], 0.0);
        assert_relative_eq!(encoded[[0, 2]], 0.0);
        assert_relative_eq!(encoded[[0, 3]], 1.0);
        assert_relative_eq!(encoded[[0, 4]], 0.0);
    }

    #[test]
    fn test_train_test_split() {
        let result = TrainTestSplit::split(100, 0.2, 42).unwrap();
        assert_eq!(result.train_indices.len() + result.test_indices.len(), 100);
        assert_eq!(result.test_indices.len(), 20);

        // No overlap
        let train_set: std::collections::HashSet<usize> = result.train_indices.iter().cloned().collect();
        let test_set: std::collections::HashSet<usize> = result.test_indices.iter().cloned().collect();
        assert!(train_set.intersection(&test_set).count() == 0);
    }

    #[test]
    fn test_time_split() {
        let result = TrainTestSplit::time_split(100, 0.2).unwrap();
        assert_eq!(result.train_indices.len(), 80);
        assert_eq!(result.test_indices.len(), 20);

        // All train indices < all test indices
        let max_train = *result.train_indices.iter().max().unwrap();
        let min_test = *result.test_indices.iter().min().unwrap();
        assert!(max_train < min_test);
    }

    #[test]
    fn test_split_array() {
        let data = Array2::from_shape_vec(
            (10, 3),
            (0..30).map(|i| i as f64).collect(),
        )
        .unwrap();

        let (train, test) = TrainTestSplit::split_array(&data, 0.3, 42).unwrap();
        assert_eq!(train.nrows(), 7);
        assert_eq!(test.nrows(), 3);
        assert_eq!(train.ncols(), 3);
        assert_eq!(test.ncols(), 3);
    }

    #[test]
    fn test_add_lags() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = FeatureEngineering::add_lags(&data, &[1, 2]).unwrap();

        assert_eq!(result.dim(), (5, 3));
        assert_relative_eq!(result[[0, 0]], 1.0);
        assert!(result[[0, 1]].is_nan()); // lag 1: no prior
        assert!(result[[0, 2]].is_nan()); // lag 2: no prior
        assert_relative_eq!(result[[2, 1]], 2.0); // lag 1 at index 2
        assert_relative_eq!(result[[3, 2]], 2.0); // lag 2 at index 3: data[1]=2.0
    }

    #[test]
    fn test_add_rolling_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result =
            FeatureEngineering::add_rolling_stats(&data, &[3], RollingStat::Mean).unwrap();

        assert_eq!(result.dim(), (5, 2));
        assert!(result[[0, 1]].is_nan());
        assert!(result[[1, 1]].is_nan());
        assert_relative_eq!(result[[2, 1]], 2.0); // mean of [1, 2, 3]
        assert_relative_eq!(result[[3, 1]], 3.0); // mean of [2, 3, 4]
        assert_relative_eq!(result[[4, 1]], 4.0); // mean of [3, 4, 5]
    }

    #[test]
    fn test_add_rolling_std() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = FeatureEngineering::add_rolling_stats(&data, &[3], RollingStat::Std).unwrap();

        // [2,4,6]: mean=4, var=(4+0+4)/2=4, std=2
        assert_relative_eq!(result[[2, 1]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_add_cross_features() {
        let data = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let result = FeatureEngineering::add_cross_features(&data).unwrap();

        // 3 columns → 3 cross features (0*1, 0*2, 1*2)
        assert_eq!(result.dim(), (3, 6));
        assert_relative_eq!(result[[0, 3]], 2.0); // 1 * 2
        assert_relative_eq!(result[[0, 4]], 3.0); // 1 * 3
        assert_relative_eq!(result[[0, 5]], 6.0); // 2 * 3
    }

    #[test]
    fn test_add_ratio_features() {
        let data = Array2::from_shape_vec((3, 3), vec![10.0, 2.0, 5.0, 20.0, 4.0, 10.0, 30.0, 6.0, 15.0])
            .unwrap();
        let result = FeatureEngineering::add_ratio_features(&data, &[0, 2], &[1, 1]).unwrap();

        // 3 original + 2 ratios
        assert_eq!(result.dim(), (3, 5));
        assert_relative_eq!(result[[0, 3]], 5.0); // 10/2
        assert_relative_eq!(result[[0, 4]], 2.5); // 5/2
    }

    #[test]
    fn test_fill_nan() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, f64::NAN, 4.0]).unwrap();
        let filled = FeatureEngineering::fill_nan(&data, 0.0);
        assert_relative_eq!(filled[[0, 0]], 1.0);
        assert_relative_eq!(filled[[0, 1]], 0.0);
        assert_relative_eq!(filled[[1, 0]], 0.0);
        assert_relative_eq!(filled[[1, 1]], 4.0);
    }

    #[test]
    fn test_drop_nan_rows() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, f64::NAN],
        )
        .unwrap();
        let cleaned = FeatureEngineering::drop_nan_rows(&data);
        assert_eq!(cleaned.dim(), (2, 2)); // rows [1,2] and [5,6] survive
        assert_relative_eq!(cleaned[[0, 0]], 1.0);
        assert_relative_eq!(cleaned[[0, 1]], 2.0);
        assert_relative_eq!(cleaned[[1, 0]], 5.0);
        assert_relative_eq!(cleaned[[1, 1]], 6.0);
    }

    #[test]
    fn test_scaler_not_fitted() {
        let scaler = MinMaxScaler::new();
        let data = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let result = scaler.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_minmax_constant_column() {
        let data = Array2::from_shape_vec((3, 2), vec![5.0, 1.0, 5.0, 2.0, 5.0, 3.0]).unwrap();
        let mut scaler = MinMaxScaler::new();
        let scaled = scaler.fit_transform(&data).unwrap();
        // Constant column should map to 0.5 (midpoint of [0, 1])
        assert_relative_eq!(scaled[[0, 0]], 0.5, epsilon = 1e-10);
    }

}
