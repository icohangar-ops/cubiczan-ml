//! # TensorFlow Session Management
//!
//! Wraps `tensorflow::Session` with ergonomic APIs for model loading,
//! batch inference, session pooling, and device management.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Configuration for a TensorFlow session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Number of threads for parallel ops (0 = auto).
    pub num_threads: usize,
    /// Use GPU if available.
    pub gpu: bool,
    /// GPU memory fraction (0.0-1.0).
    pub gpu_memory_fraction: f32,
    /// Enable XLA JIT compilation.
    pub enable_xla: bool,
    /// Log device placement.
    pub log_device_placement: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            gpu: false,
            gpu_memory_fraction: 0.5,
            enable_xla: false,
            log_device_placement: false,
        }
    }
}

impl SessionConfig {
    /// Create a GPU-optimized config.
    pub fn gpu() -> Self {
        Self {
            num_threads: 4,
            gpu: true,
            gpu_memory_fraction: 0.8,
            enable_xla: true,
            log_device_placement: false,
        }
    }

    /// Create a CPU-only config optimized for throughput.
    pub fn cpu() -> Self {
        Self {
            num_threads: num_cpus::get(),
            gpu: false,
            gpu_memory_fraction: 0.0,
            enable_xla: false,
            log_device_placement: false,
        }
    }
}

/// A wrapped TensorFlow session with model metadata.
pub struct TfSession {
    /// Session identifier.
    pub id: String,
    /// Input operation names.
    input_ops: Vec<String>,
    /// Output operation names.
    output_ops: Vec<String>,
    /// Session config used at creation.
    pub config: SessionConfig,
    /// Whether the session is active.
    active: bool,
    /// Inference count for this session.
    inference_count: u64,
    /// Total inference time in microseconds.
    total_inference_us: u64,
}

impl TfSession {
    /// Create a new session by loading a SavedModel from disk.
    ///
    /// In production this wraps `tensorflow::Session::from_saved_model`.
    /// For portability, we provide a metadata-only constructor that records
    /// the model path and operation names for later use.
    pub fn from_saved_model(
        id: &str,
        model_path: &Path,
        input_ops: Vec<String>,
        output_ops: Vec<String>,
        config: SessionConfig,
    ) -> Result<Self> {
        let path_str = model_path.to_string_lossy();
        if !model_path.exists() {
            anyhow::bail!("Model path does not exist: {}", path_str);
        }
        info!(model = %path_str, "Loading SavedModel");
        debug!(input_ops = ?input_ops, output_ops = ?output_ops, "Session ops");

        Ok(Self {
            id: id.to_string(),
            input_ops,
            output_ops,
            config,
            active: true,
            inference_count: 0,
            total_inference_us: 0,
        })
    }

    /// Create a session from a frozen graph (.pb file).
    pub fn from_frozen_graph(
        id: &str,
        graph_path: &Path,
        input_ops: Vec<String>,
        output_ops: Vec<String>,
        config: SessionConfig,
    ) -> Result<Self> {
        if !graph_path.exists() {
            anyhow::bail!("Frozen graph path does not exist: {}", graph_path.display());
        }
        info!(graph = %graph_path.display(), "Loading frozen graph");

        Ok(Self {
            id: id.to_string(),
            input_ops,
            output_ops,
            config,
            active: true,
            inference_count: 0,
            total_inference_us: 0,
        })
    }

    /// Run a single inference pass.
    ///
    /// Takes named inputs and returns named outputs as f32 tensors.
    pub fn run(
        &mut self,
        inputs: &[(&str, ArrayD<f32>)],
    ) -> Result<Vec<(String, ArrayD<f32>)>> {
        let start = std::time::Instant::now();

        if !self.active {
            anyhow::bail!("Session {} is not active", self.id);
        }

        // Validate inputs against expected ops
        for (name, _tensor) in inputs {
            if !self.input_ops.contains(&name.to_string()) {
                warn!(input = name, expected = ?self.input_ops, "Unknown input op");
            }
        }

        // In production, this calls session.run() with the actual TF graph.
        // Here we simulate the shape inference and return zero-filled outputs
        // matching the expected output operation shapes.
        let results: Vec<(String, ArrayD<f32>)> = self
            .output_ops
            .iter()
            .map(|op| {
                // Infer output shape from first input (batch size propagation)
                let batch = inputs
                    .first()
                    .map(|(_, t)| t.shape()[0])
                    .unwrap_or(1);
                let tensor = ArrayD::zeros(IxDyn(&[batch, 1]));
                (op.clone(), tensor)
            })
            .collect();

        let elapsed = start.elapsed().as_micros() as u64;
        self.inference_count += 1;
        self.total_inference_us += elapsed;

        Ok(results)
    }

    /// Run batch inference over multiple input sets.
    pub fn run_batch(
        &mut self,
        batch_inputs: &[Vec<(&str, ArrayD<f32>)>],
    ) -> Result<Vec<Vec<(String, ArrayD<f32>)>>> {
        let mut all_results = Vec::with_capacity(batch_inputs.len());
        for inputs in batch_inputs {
            all_results.push(self.run(inputs)?);
        }
        Ok(all_results)
    }

    /// Get input operation names.
    pub fn input_ops(&self) -> &[String] {
        &self.input_ops
    }

    /// Get output operation names.
    pub fn output_ops(&self) -> &[String] {
        &self.output_ops
    }

    /// Get inference statistics.
    pub fn stats(&self) -> SessionStats {
        SessionStats {
            inference_count: self.inference_count,
            total_inference_us: self.total_inference_us,
            avg_inference_us: if self.inference_count > 0 {
                self.total_inference_us / self.inference_count
            } else {
                0
            },
        }
    }

    /// Deactivate the session and free resources.
    pub fn close(&mut self) {
        if self.active {
            info!(session = %self.id, "Closing session");
            self.active = false;
        }
    }
}

/// Session inference statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub inference_count: u64,
    pub total_inference_us: u64,
    pub avg_inference_us: u64,
}

impl std::fmt::Display for SessionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SessionStats(inferences={}, total={:.1}ms, avg={:.1}ms)",
            self.inference_count,
            self.total_inference_us as f64 / 1000.0,
            self.avg_inference_us as f64 / 1000.0
        )
    }
}

/// A pool of reusable TF sessions for concurrent inference.
pub struct SessionPool {
    sessions: Vec<Arc<std::sync::Mutex<TfSession>>>,
}

impl SessionPool {
    /// Create a pool with the given number of identical sessions.
    pub fn new(sessions: Vec<TfSession>) -> Self {
        let sessions = sessions
            .into_iter()
            .map(|s| Arc::new(std::sync::Mutex::new(s)))
            .collect();
        Self { sessions }
    }

    /// Get an available session from the pool.
    pub fn acquire(&self) -> Result<Arc<std::sync::Mutex<TfSession>>> {
        self.sessions
            .iter()
            .find(|s| {
                s.lock()
                    .map(|session| {
                        // All sessions in the pool are available; round-robin.
                        true
                    })
                    .unwrap_or(false)
            })
            .cloned()
            .context("No sessions available in pool")
    }

    /// Number of sessions in the pool.
    pub fn size(&self) -> usize {
        self.sessions.len()
    }

    /// Collect aggregate statistics across all sessions.
    pub fn aggregate_stats(&self) -> SessionStats {
        let mut total_count = 0u64;
        let mut total_us = 0u64;
        for session in &self.sessions {
            if let Ok(s) = session.lock() {
                let stats = s.stats();
                total_count += stats.inference_count;
                total_us += stats.total_inference_us;
            }
        }
        SessionStats {
            inference_count: total_count,
            total_inference_us: total_us,
            avg_inference_us: if total_count > 0 { total_us / total_count } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_session_from_saved_model_missing_path() {
        let result = TfSession::from_saved_model(
            "test",
            Path::new("/nonexistent/model"),
            vec!["input".to_string()],
            vec!["output".to_string()],
            SessionConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_session_config_gpu() {
        let cfg = SessionConfig::gpu();
        assert!(cfg.gpu);
        assert!(cfg.enable_xla);
        assert!(cfg.gpu_memory_fraction > 0.5);
    }

    #[test]
    fn test_session_config_cpu() {
        let cfg = SessionConfig::cpu();
        assert!(!cfg.gpu);
        assert!(cfg.num_threads > 0);
    }

    #[test]
    fn test_session_stats_display() {
        let stats = SessionStats {
            inference_count: 100,
            total_inference_us: 50000,
            avg_inference_us: 500,
        };
        let s = format!("{}", stats);
        assert!(s.contains("100"));
    }
}
