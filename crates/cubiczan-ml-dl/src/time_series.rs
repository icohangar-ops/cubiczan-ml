//! # Deep Learning for Time Series
//!
//! Time series datasets, sequence prediction, anomaly detection,
//! and full ML pipelines for financial time series.

use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Configuration for creating windowed time series datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Number of time steps in input window.
    pub window_size: usize,
    /// Number of time steps to forecast ahead.
    pub horizon: usize,
    /// Step stride between windows (1 = no overlap skip).
    pub stride: usize,
}

impl WindowConfig {
    pub fn new(window_size: usize, horizon: usize) -> Self {
        Self { window_size, horizon, stride: 1 }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
}

/// Windowed time series dataset for ML model training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesDataset {
    /// Input windows. Shape: [samples, window_size, features].
    pub inputs: Vec<f32>,
    /// Target windows. Shape: [samples, horizon] or [samples, horizon, features].
    pub targets: Vec<f32>,
    pub num_samples: usize,
    pub num_features: usize,
    pub window_size: usize,
    pub horizon: usize,
}

impl TimeSeriesDataset {
    /// Create a windowed dataset from a 2D data array (rows=timesteps, cols=features).
    pub fn from_array(data: &Array2<f64>, config: &WindowConfig) -> Self {
        let (n_timesteps, n_features) = data.dim();
        assert!(
            n_timesteps >= config.window_size + config.horizon,
            "Not enough timesteps: {} < {} + {}",
            n_timesteps, config.window_size, config.horizon
        );

        let max_start = n_timesteps - config.window_size - config.horizon + 1;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let mut start = 0;
        while start + config.window_size + config.horizon <= n_timesteps {
            // Input window
            for t in start..start + config.window_size {
                for f in 0..n_features {
                    inputs.push(data[[t, f]] as f32);
                }
            }
            // Target (next horizon values of feature 0)
            for t in start + config.window_size..start + config.window_size + config.horizon {
                targets.push(data[[t, 0]] as f32);
            }
            start += config.stride;
        }

        let num_samples = inputs.len() / (config.window_size * n_features);
        Self {
            inputs,
            targets,
            num_samples,
            num_features: n_features,
            window_size: config.window_size,
            horizon: config.horizon,
        }
    }

    /// Split into train/test sets.
    pub fn train_test_split(&self, test_ratio: f64) -> (TimeSeriesDataset, TimeSeriesDataset) {
        let test_size = (self.num_samples as f64 * test_ratio) as usize;
        let train_size = self.num_samples - test_size;

        let features_per_window = self.window_size * self.num_features;
        let targets_per_sample = self.horizon;

        let train_inputs = self.inputs[..train_size * features_per_window].to_vec();
        let test_inputs = self.inputs[train_size * features_per_window..].to_vec();
        let train_targets = self.targets[..train_size * targets_per_sample].to_vec();
        let test_targets = self.targets[train_size * targets_per_sample..].to_vec();

        (
            TimeSeriesDataset {
                inputs: train_inputs, targets: train_targets,
                num_samples: train_size, num_features: self.num_features,
                window_size: self.window_size, horizon: self.horizon,
            },
            TimeSeriesDataset {
                inputs: test_inputs, targets: test_targets,
                num_samples: test_size, num_features: self.num_features,
                window_size: self.window_size, horizon: self.horizon,
            },
        )
    }

    /// Get a single input window as a 2D array.
    pub fn get_input_window(&self, index: usize) -> Array2<f32> {
        let start = index * self.window_size * self.num_features;
        let end = start + self.window_size * self.num_features;
        Array2::from_shape_vec(
            (self.window_size, self.num_features),
            self.inputs[start..end].to_vec(),
        ).unwrap()
    }
}

/// Feature normalizer for consistent preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureNormalizer {
    means: Vec<f64>,
    stds: Vec<f64>,
}

impl FeatureNormalizer {
    /// Fit normalizer from training data.
    pub fn fit(data: &Array2<f64>) -> Self {
        let (n, _features) = data.dim();
        let means: Vec<f64> = (0..data.ncols())
            .map(|c| data.column(c).sum() / n as f64)
            .collect();
        let stds: Vec<f64> = (0..data.ncols())
            .map(|c| {
                let mean = means[c];
                let var: f64 = data.column(c).iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
                var.sqrt().max(1e-8)
            })
            .collect();
        Self { means, stds }
    }

    /// Normalize data using fitted parameters.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut result = data.clone();
        for (col_idx, mut col) in result.columns_mut().into_iter().enumerate() {
            let mean = self.means[col_idx];
            let std = self.stds[col_idx];
            for val in col.iter_mut() {
                *val = (*val - mean) / std;
            }
        }
        result
    }

    /// Inverse transform (denormalize) predictions.
    pub fn inverse_transform(&self, data: &Array1<f64>) -> Array1<f64> {
        data.iter().enumerate().map(|(i, v)| v * self.sts()[i] + self.means[i]).collect()
    }

    fn sts(&self) -> &[f64] { &self.stds }
}

/// Single-step sequence prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub predicted: Vec<f64>,
    pub actual: Vec<f64>,
    pub mse: f64,
    pub mae: f64,
    pub mape: f64,
    pub horizon: usize,
}

impl ForecastResult {
    /// Calculate metrics from predicted vs actual.
    pub fn from_arrays(predicted: &[f64], actual: &[f64]) -> Self {
        let n = predicted.len().min(actual.len());
        if n == 0 {
            return Self { predicted: vec![], actual: vec![], mse: 0.0, mae: 0.0, mape: 0.0, horizon: 0 };
        }
        let mse: f64 = (0..n).map(|i| (predicted[i] - actual[i]).powi(2)).sum::<f64>() / n as f64;
        let mae: f64 = (0..n).map(|i| (predicted[i] - actual[i]).abs()).sum::<f64>() / n as f64;
        let mape: f64 = (0..n)
            .filter(|&i| actual[i].abs() > 1e-8)
            .map(|i| ((predicted[i] - actual[i]).abs() / actual[i].abs()) * 100.0)
            .sum::<f64>() / n.max(1) as f64;
        Self {
            predicted: predicted[..n].to_vec(),
            actual: actual[..n].to_vec(),
            mse: mae.sqrt(), // RMSE
            mae,
            mape,
            horizon: n,
        }
    }
}

/// Simple sequence predictor using linear regression (baseline).
pub struct SequencePredictor {
    weights: Vec<f64>,
    bias: f64,
}

impl SequencePredictor {
    pub fn new() -> Self {
        Self { weights: vec![0.0; 1], bias: 0.0 }
    }

    /// Train a simple linear model on windowed data.
    pub fn fit(&mut self, dataset: &TimeSeriesDataset) -> Result<()> {
        self.weights = vec![1.0 / dataset.window_size as f64; dataset.window_size];
        self.bias = 0.0;
        // Simple moving average as baseline prediction
        Ok(())
    }

    /// Predict next value from a window.
    pub fn predict(&self, window: &[f64]) -> f64 {
        if window.is_empty() { return 0.0; }
        window.iter().sum::<f64>() / window.len() as f64 + self.bias
    }

    /// Predict multiple steps ahead.
    pub fn predict_multi(&self, window: &[f64], steps: usize) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(steps);
        let mut current = window.to_vec();
        for _ in 0..steps {
            let pred = self.predict(&current);
            predictions.push(pred);
            current.push(pred);
            if current.len() > window.len() {
                current.remove(0);
            }
        }
        predictions
    }
}

/// Multi-step predictor with different strategies.
pub struct MultiStepPredictor {
    strategy: MultiStepStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiStepStrategy {
    Recursive,
    Direct,
    MIMO, // Multi-input multi-output
}

impl MultiStepPredictor {
    pub fn new(strategy: MultiStepStrategy) -> Self {
        Self { strategy }
    }

    /// Predict the next `horizon` values.
    pub fn predict(&self, window: &[f64], horizon: usize) -> Vec<f64> {
        match &self.strategy {
            MultiStepStrategy::Recursive => {
                let predictor = SequencePredictor::new();
                predictor.predict_multi(window, horizon)
            }
            MultiStepStrategy::Direct | MultiStepStrategy::MIMO => {
                // For direct/MIMO, we'd need separate models per horizon.
                // Simple fallback: weighted average with decay.
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let trend = if window.len() >= 2 {
                    (window[window.len() - 1] - window[window.len() - 2])
                } else { 0.0 };
                (0..horizon).map(|i| mean + trend * (i as f64 + 1.0) * 0.5).collect()
            }
        }
    }
}

/// Anomaly detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub index: usize,
    pub score: f64,
    pub is_anomaly: bool,
    pub threshold: f64,
}

/// Anomaly detector using statistical methods.
pub struct AnomalyDetector {
    method: AnomalyMethod,
    threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyMethod {
    ZScore { window: usize },
    IQR { window: usize },
    RollingMA { window: usize, std_multiplier: f64 },
}

impl AnomalyDetector {
    pub fn z_score(window: usize, threshold: f64) -> Self {
        Self { method: AnomalyMethod::ZScore { window }, threshold }
    }

    pub fn detect(&self, values: &[f64]) -> Vec<AnomalyResult> {
        match &self.method {
            AnomalyMethod::ZScore { window } => self.detect_zscore(values, *window),
            AnomalyMethod::IQR { window } => self.detect_iqr(values, *window),
            AnomalyMethod::RollingMA { window, std_multiplier } => self.detect_rolling(values, *window, *std_multiplier),
        }
    }

    fn detect_zscore(&self, values: &[f64], window: usize) -> Vec<AnomalyResult> {
        let mut results = Vec::new();
        for i in window..values.len() {
            let slice = &values[i - window..i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let std = (slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window as f64).sqrt().max(1e-8);
            let z = (values[i] - mean) / std;
            results.push(AnomalyResult {
                index: i,
                score: z.abs(),
                is_anomaly: z.abs() > self.threshold,
                threshold: self.threshold,
            });
        }
        results
    }

    fn detect_iqr(&self, values: &[f64], window: usize) -> Vec<AnomalyResult> {
        let mut results = Vec::new();
        for i in window..values.len() {
            let mut sorted: Vec<f64> = values[i - window..i].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1 = sorted[sorted.len() / 4];
            let q3 = sorted[3 * sorted.len() / 4];
            let iqr = q3 - q1;
            let lower = q1 - self.threshold * iqr;
            let upper = q3 + self.threshold * iqr;
            let val = values[i];
            let score = if val < lower { (lower - val) / iqr.max(1e-8) }
                        else if val > upper { (val - upper) / iqr.max(1e-8) }
                        else { 0.0 };
            results.push(AnomalyResult {
                index: i, score, is_anomaly: score > 0.0, threshold: self.threshold,
            });
        }
        results
    }

    fn detect_rolling(&self, values: &[f64], window: usize, std_mult: f64) -> Vec<AnomalyResult> {
        let mut results = Vec::new();
        for i in window..values.len() {
            let mean = values[i - window..i].iter().sum::<f64>() / window as f64;
            let std = (values[i - window..i].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window as f64).sqrt().max(1e-8);
            let deviation = (values[i] - mean).abs();
            let score = deviation / std;
            results.push(AnomalyResult {
                index: i, score, is_anomaly: score > std_mult, threshold: std_mult,
            });
        }
        results
    }
}

/// Walk-forward cross-validation for time series.
pub struct WalkForwardValidator {
    pub train_size: usize,
    pub test_size: usize,
    pub step_size: usize,
}

impl WalkForwardValidator {
    pub fn new(train_size: usize, test_size: usize, step_size: usize) -> Self {
        Self { train_size, test_size, step_size }
    }

    /// Generate train/test splits for walk-forward validation.
    pub fn splits(&self, total_len: usize) -> Vec<(usize, usize)> {
        let mut splits = Vec::new();
        let mut start = 0;
        while start + self.train_size + self.test_size <= total_len {
            splits.push((start, start + self.train_size));
            start += self.step_size;
        }
        splits
    }
}

/// Full time series ML pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub window_size: usize,
    pub horizon: usize,
    pub train_test_ratio: f64,
    pub normalize: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { window_size: 20, horizon: 1, train_test_ratio: 0.2, normalize: true }
    }
}

/// End-to-end time series pipeline.
pub struct TimeSeriesPipeline {
    config: PipelineConfig,
}

impl TimeSeriesPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Run the full pipeline: preprocess → predict → evaluate.
    pub fn run(&self, data: &Array2<f64>) -> Result<ForecastResult> {
        let normalizer = if self.config.normalize {
            Some(FeatureNormalizer::fit(data))
        } else {
            None
        };

        let normalized = match &normalizer {
            Some(n) => n.transform(data),
            None => data.clone(),
        };

        let wc = WindowConfig::new(self.config.window_size, self.config.horizon);
        let dataset = TimeSeriesDataset::from_array(&normalized, &wc);

        let (train, test) = dataset.train_test_split(self.config.train_test_ratio);

        let mut predictor = SequencePredictor::new();
        predictor.fit(&train)?;

        let n = normalized.nrows();
        let last_window = normalized.slice(ndarray::s![n - self.config.window_size.., 0]).to_vec();
        let predicted = predictor.predict_multi(&last_window, self.config.horizon);

        let n_rows = data.nrows();
        let actual = data.column(0).slice(ndarray::s![n_rows - self.config.horizon..]).to_vec();

        Ok(ForecastResult::from_arrays(&predicted, &actual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample_data() -> Array2<f64> {
        let values: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5 + (i % 10) as f64 * 2.0).collect();
        Array2::from_shape_vec((100, 1), values).unwrap()
    }

    #[test]
    fn test_windowed_dataset() {
        let data = sample_data();
        let config = WindowConfig::new(10, 1);
        let ds = TimeSeriesDataset::from_array(&data, &config);
        assert_eq!(ds.num_samples, 90);
        assert_eq!(ds.window_size, 10);
    }

    #[test]
    fn test_train_test_split() {
        let data = sample_data();
        let config = WindowConfig::new(10, 1);
        let ds = TimeSeriesDataset::from_array(&data, &config);
        let (train, test) = ds.train_test_split(0.2);
        assert!(train.num_samples > test.num_samples);
        assert_eq!(train.num_samples + test.num_samples, ds.num_samples);
    }

    #[test]
    fn test_feature_normalizer() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let normalizer = FeatureNormalizer::fit(&data);
        let normalized = normalizer.transform(&data);
        // Mean of normalized data should be ~0
        let col_mean: f64 = normalized.column(0).sum() / normalized.nrows() as f64;
        assert!(col_mean.abs() < 1e-6);
    }

    #[test]
    fn test_sequence_predictor() {
        let predictor = SequencePredictor::new();
        let window = vec![10.0, 20.0, 30.0];
        let pred = predictor.predict(&window);
        assert!((pred - 20.0).abs() < 0.01); // mean
    }

    #[test]
    fn test_anomaly_detector() {
        let values: Vec<f64> = (0..100).map(|i| {
            if i == 50 { 1000.0 } else { 10.0 + (i as f64 % 10.0) }
        }).collect();
        let detector = AnomalyDetector::z_score(20, 2.0);
        let results = detector.detect(&values);
        let anomalies: Vec<_> = results.iter().filter(|r| r.is_anomaly).collect();
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_walk_forward() {
        let wf = WalkForwardValidator::new(60, 10, 10);
        let splits = wf.splits(100);
        assert!(!splits.is_empty());
    }

    #[test]
    fn test_pipeline() {
        let data = sample_data();
        let pipeline = TimeSeriesPipeline::new(PipelineConfig::default());
        let result = pipeline.run(&data);
        assert!(result.is_ok());
    }
}
