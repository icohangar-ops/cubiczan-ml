//! # Pre-built TensorFlow Model Interfaces
//!
//! Ready-to-use wrappers for common financial ML model architectures
//! running on TensorFlow via Rust.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use crate::session::{SessionConfig, TfSession};

/// LSTM model for time series forecasting.
pub struct TfLSTM {
    session: TfSession,
    /// Number of time steps in input sequence.
    pub sequence_length: usize,
    /// Number of input features per time step.
    pub num_features: usize,
    /// Number of output units (forecast horizon).
    pub forecast_horizon: usize,
}

impl TfLSTM {
    /// Load a trained LSTM model from a SavedModel directory.
    pub fn load(
        model_path: &Path,
        sequence_length: usize,
        num_features: usize,
        forecast_horizon: usize,
        config: SessionConfig,
    ) -> Result<Self> {
        let session = TfSession::from_saved_model(
            "lstm_forecast",
            model_path,
            vec!["lstm_input".to_string()],
            vec!["lstm_output".to_string()],
            config,
        )?;

        Ok(Self {
            session,
            sequence_length,
            num_features,
            forecast_horizon,
        })
    }

    /// Run forecasting inference.
    ///
    /// Input shape: [batch, sequence_length, num_features]
    /// Output shape: [batch, forecast_horizon]
    pub fn forecast(&mut self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let results = self.session.run(&[("lstm_input", input.clone())])?;
        results
            .into_iter()
            .next()
            .map(|(_, t)| t)
            .context("No LSTM output")
    }

    /// Forecast a single sequence (convenience method).
    ///
    /// Input shape: [sequence_length, num_features]
    pub fn forecast_single(&mut self, sequence: &ndarray::Array2<f32>) -> Result<Vec<f32>> {
        let batched = sequence.clone().insert_axis(ndarray::Axis(0));
        let input = batched.into_dyn();
        let output = self.forecast(&input)?;
        Ok(output.iter().cloned().collect())
    }
}

/// Transformer model for sequence classification tasks.
pub struct TfTransformer {
    session: TfSession,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of classes.
    pub num_classes: usize,
}

impl TfTransformer {
    /// Load a trained Transformer classifier.
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        num_classes: usize,
        config: SessionConfig,
    ) -> Result<Self> {
        let session = TfSession::from_saved_model(
            "transformer_cls",
            model_path,
            vec!["input_ids".to_string(), "attention_mask".to_string()],
            vec!["logits".to_string()],
            config,
        )?;

        Ok(Self {
            session,
            max_seq_len,
            num_classes,
        })
    }

    /// Run classification inference.
    ///
    /// Returns logits of shape [batch, num_classes].
    pub fn classify(
        &mut self,
        input_ids: &ArrayD<f32>,
        attention_mask: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        let results = self.session.run(&[
            ("input_ids", input_ids.clone()),
            ("attention_mask", attention_mask.clone()),
        ])?;
        results
            .into_iter()
            .next()
            .map(|(_, t)| t)
            .context("No transformer output")
    }

    /// Classify and get the predicted class indices.
    pub fn predict_classes(
        &mut self,
        input_ids: &ArrayD<f32>,
        attention_mask: &ArrayD<f32>,
    ) -> Result<Vec<usize>> {
        let logits = self.classify(input_ids, attention_mask)?;
        let classes: Vec<usize> = logits
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();
        Ok(classes)
    }
}

/// Generic classifier for tabular financial data.
pub struct TfClassifier {
    session: TfSession,
    /// Number of input features.
    pub num_features: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Class labels.
    pub class_labels: Vec<String>,
}

impl TfClassifier {
    /// Load a trained classifier.
    pub fn load(
        model_path: &Path,
        num_features: usize,
        num_classes: usize,
        class_labels: Vec<String>,
        config: SessionConfig,
    ) -> Result<Self> {
        let session = TfSession::from_saved_model(
            "tabular_classifier",
            model_path,
            vec!["features".to_string()],
            vec!["probabilities".to_string()],
            config,
        )?;

        Ok(Self {
            session,
            num_features,
            num_classes,
            class_labels,
        })
    }

    /// Predict class probabilities for a batch of samples.
    pub fn predict_proba(&mut self, features: &ArrayD<f32>) -> Result<ClassificationOutput> {
        let results = self.session.run(&[("features", features.clone())])?;
        let probs = results
            .into_iter()
            .next()
            .map(|(_, t)| t)
            .context("No classifier output")?;

        let predicted_classes: Vec<usize> = probs
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        let predicted_labels: Vec<String> = predicted_classes
            .iter()
            .map(|&idx| {
                self.class_labels
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", idx))
            })
            .collect();

        Ok(ClassificationOutput {
            probabilities: probs,
            predicted_classes,
            predicted_labels,
        })
    }
}

/// Output from a classification prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationOutput {
    /// Probability distribution over classes. Shape: [batch, num_classes]
    pub probabilities: ArrayD<f32>,
    /// Predicted class index per sample.
    pub predicted_classes: Vec<usize>,
    /// Human-readable class labels.
    pub predicted_labels: Vec<String>,
}

/// Risk scoring model for financial risk assessment.
pub struct TfRiskModel {
    session: TfSession,
    /// Risk categories from low to high.
    pub risk_labels: Vec<String>,
    /// Minimum score threshold for "high risk".
    pub high_risk_threshold: f32,
}

impl TfRiskModel {
    /// Load a trained risk model.
    pub fn load(
        model_path: &Path,
        risk_labels: Vec<String>,
        high_risk_threshold: f32,
        config: SessionConfig,
    ) -> Result<Self> {
        let session = TfSession::from_saved_model(
            "risk_scorer",
            model_path,
            vec!["risk_features".to_string()],
            vec!["risk_score".to_string(), "risk_class".to_string()],
            config,
        )?;

        Ok(Self {
            session,
            risk_labels,
            high_risk_threshold,
        })
    }

    /// Score a batch of entities for risk.
    pub fn score(&mut self, features: &ArrayD<f32>) -> Result<Vec<RiskAssessment>> {
        let results = self.session.run(&[("risk_features", features.clone())])?;

        let batch_size = features.shape()[0];
        let mut assessments = Vec::with_capacity(batch_size);

        // Extract risk scores (first output)
        if let Some((_, scores)) = results.first() {
            for i in 0..batch_size {
                let score = scores[[i, 0]];
                let level = if score >= self.high_risk_threshold {
                    RiskLevel::High
                } else if score >= self.high_risk_threshold * 0.6 {
                    RiskLevel::Medium
                } else {
                    RiskLevel::Low
                };
                assessments.push(RiskAssessment { score, level });
            }
        }

        Ok(assessments)
    }
}

/// Risk assessment result for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Continuous risk score (0.0 to 1.0).
    pub score: f32,
    /// Categorical risk level.
    pub level: RiskLevel,
}

/// Risk severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "LOW"),
            RiskLevel::Medium => write!(f, "MEDIUM"),
            RiskLevel::High => write!(f, "HIGH"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_display() {
        assert_eq!(format!("{}", RiskLevel::Low), "LOW");
        assert_eq!(format!("{}", RiskLevel::High), "HIGH");
    }

    #[test]
    fn test_classification_output_creation() {
        let probs = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![0.1, 0.7, 0.2, 0.8, 0.1, 0.1])
            .unwrap();
        let output = ClassificationOutput {
            probabilities: probs,
            predicted_classes: vec![1, 0],
            predicted_labels: vec!["positive".to_string(), "negative".to_string()],
        };
        assert_eq!(output.predicted_labels[0], "positive");
    }
}
