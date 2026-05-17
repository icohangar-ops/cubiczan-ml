//! # Inference Engines
//!
//! Fast inference engines for Candle models.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Generic inference engine trait.
pub trait InferenceEngine: Send + Sync {
    /// Run inference on a batch of inputs.
    fn infer(&self, inputs: &[(&str, ArrayD<f32>)]) -> Result<Vec<(String, ArrayD<f32>)>>;
    /// Engine name.
    fn name(&self) -> &str;
    /// Framework identifier.
    fn framework(&self) -> Framework;
}

/// Supported ML frameworks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Framework {
    Candle,
}

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Framework::Candle => write!(f, "candle"),
        }
    }
}

/// Result from an inference call.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub outputs: Vec<(String, ArrayD<f32>)>,
    pub latency_us: u64,
    pub framework: Framework,
    pub model_name: String,
}

/// Candle-based inference engine.
pub struct CandleInferenceEngine {
    model_name: String,
    device: String,
    dtype: String,
}

impl CandleInferenceEngine {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
        }
    }

    pub fn with_device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self
    }

    pub fn from_pretrained(model_path: &Path, device: &str) -> Result<Self> {
        if !model_path.exists() {
            anyhow::bail!("Model path not found: {}", model_path.display());
        }
        info!(path = %model_path.display(), device, "Loading Candle model");
        Ok(Self {
            model_name: model_path.to_string_lossy().to_string(),
            device: device.to_string(),
            dtype: "f32".to_string(),
        })
    }
}

impl InferenceEngine for CandleInferenceEngine {
    fn infer(&self, inputs: &[(&str, ArrayD<f32>)]) -> Result<Vec<(String, ArrayD<f32>)>> {
        let start = Instant::now();

        // In production, this runs the actual candle model.
        // Here we simulate output shapes based on input batch size.
        let batch = inputs.first().map(|(_, t)| t.shape()[0]).unwrap_or(1);
        let results = vec![("output".to_string(), ArrayD::zeros(ndarray::IxDyn(&[batch, 1])))];

        debug!(latency_us = start.elapsed().as_micros(), "Candle inference");
        Ok(results)
    }

    fn name(&self) -> &str { "Candle" }
    fn framework(&self) -> Framework { Framework::Candle }
}

/// Batch inference runner.
pub struct BatchInference<E: InferenceEngine> {
    engine: E,
    batch_size: usize,
}

impl<E: InferenceEngine> BatchInference<E> {
    pub fn new(engine: E, batch_size: usize) -> Self {
        Self { engine, batch_size }
    }

    /// Process multiple inputs in batches.
    pub fn run(
        &self,
        all_inputs: &[Vec<(&str, ArrayD<f32>)>],
    ) -> Result<Vec<InferenceResult>> {
        let mut results = Vec::with_capacity(all_inputs.len());
        for inputs in all_inputs {
            let start = Instant::now();
            let outputs = self.engine.infer(inputs)?;
            results.push(InferenceResult {
                outputs,
                latency_us: start.elapsed().as_micros() as u64,
                framework: self.engine.framework(),
                model_name: self.engine.name().to_string(),
            });
        }
        Ok(results)
    }
}

/// Model loader utility.
pub struct ModelLoader;

impl ModelLoader {
    /// Detect model format from file extension.
    pub fn detect_format(path: &Path) -> Result<ModelFormat> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "safetensors" => Ok(ModelFormat::Safetensors),
            "pt" | "pth" => Ok(ModelFormat::PyTorch),
            "onnx" => Ok(ModelFormat::Onnx),
            "bin" => Ok(ModelFormat::Bincode),
            "ot" => Ok(ModelFormat::Candle),
            _ => anyhow::bail!("Unknown model format: .{}", ext),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelFormat {
    Safetensors,
    PyTorch,
    Onnx,
    Bincode,
    Candle,
}

/// Inference benchmarking.
pub struct InferenceBenchmark;

impl InferenceBenchmark {
    /// Run a benchmark with warmup + timed runs.
    pub fn run<E: InferenceEngine>(
        engine: &E,
        inputs: &[(&str, ArrayD<f32>)],
        warmup_runs: usize,
        timed_runs: usize,
    ) -> ProfilingResult {
        // Warmup
        for _ in 0..warmup_runs {
            let _ = engine.infer(inputs);
        }

        // Timed runs
        let mut latencies = Vec::with_capacity(timed_runs);
        for _ in 0..timed_runs {
            let start = Instant::now();
            let _ = engine.infer(inputs);
            latencies.push(start.elapsed().as_micros() as u64);
        }

        let total: u64 = latencies.iter().sum();
        let avg = total / timed_runs.max(1) as u64;
        let p50 = percentile(&latencies, 50);
        let p95 = percentile(&latencies, 95);
        let p99 = percentile(&latencies, 99);

        ProfilingResult {
            framework: engine.framework().to_string(),
            warmup_runs,
            timed_runs,
            total_us: total,
            avg_us: avg,
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
            min_us: *latencies.iter().min().unwrap_or(&0),
            max_us: *latencies.iter().max().unwrap_or(&0),
            throughput: if avg > 0 { 1_000_000 / avg } else { 0 },
        }
    }
}

fn percentile(sorted: &[u64], p: usize) -> u64 {
    if sorted.is_empty() { return 0; }
    let mut v = sorted.to_vec();
    v.sort();
    let idx = (p as f64 / 100.0 * (v.len() - 1) as f64).round() as usize;
    v[idx.min(v.len() - 1)]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingResult {
    pub framework: String,
    pub warmup_runs: usize,
    pub timed_runs: usize,
    pub total_us: u64,
    pub avg_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub throughput: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_engine() {
        let engine = CandleInferenceEngine::new("test");
        assert_eq!(engine.name(), "Candle");
        assert_eq!(engine.framework(), Framework::Candle);
    }

    #[test]
    fn test_candle_infer() {
        let engine = CandleInferenceEngine::new("test");
        let input = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
        let result = engine.infer(&[("input", input)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_format_detection() {
        assert!(matches!(
            ModelLoader::detect_format(Path::new("model.safetensors")).unwrap(),
            ModelFormat::Safetensors
        ));
        assert!(matches!(
            ModelLoader::detect_format(Path::new("model.pt")).unwrap(),
            ModelFormat::PyTorch
        ));
    }

    #[test]
    fn test_percentile() {
        let data = vec![10, 20, 30, 40, 50];
        assert_eq!(percentile(&data, 50), 30);
    }

    #[test]
    fn test_framework_display() {
        assert_eq!(format!("{}", Framework::Candle), "candle");
    }
}
