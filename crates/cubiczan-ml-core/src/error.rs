//! # ML Error Types
//!
//! Unified error types for the CubicZan ML ecosystem.

use std::fmt;

/// The core error type for CubicZan ML operations.
///
/// Provides variants covering the common failure modes across the entire ML
/// pipeline: data validation, computation, serialization, and configuration.
#[derive(Debug, Clone)]
pub enum MlError {
    /// The provided input data was invalid (wrong shape, empty, NaN, etc.).
    InvalidInput(String),
    /// A numerical computation failed (singular matrix, divergence, etc.).
    ComputationError(String),
    /// Serialization / deserialization (bincode, JSON, etc.) failed.
    SerializationError(String),
    /// A required parameter or configuration value is missing or wrong.
    ConfigError(String),
    /// The requested device is not available.
    DeviceError(String),
    /// A model has not been fitted yet.
    NotFitted(String),
    /// Dimension / shape mismatch between arrays.
    ShapeMismatch { expected: String, actual: String },
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            MlError::ComputationError(msg) => write!(f, "computation error: {}", msg),
            MlError::SerializationError(msg) => write!(f, "serialization error: {}", msg),
            MlError::ConfigError(msg) => write!(f, "config error: {}", msg),
            MlError::DeviceError(msg) => write!(f, "device error: {}", msg),
            MlError::NotFitted(msg) => write!(f, "not fitted: {}", msg),
            MlError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for MlError {}

// ---------------------------------------------------------------------------
// Conversions from downstream error types
// ---------------------------------------------------------------------------

impl From<serde_json::Error> for MlError {
    fn from(e: serde_json::Error) -> Self {
        MlError::SerializationError(e.to_string())
    }
}

impl From<bincode::Error> for MlError {
    fn from(e: bincode::Error) -> Self {
        MlError::SerializationError(e.to_string())
    }
}

impl From<String> for MlError {
    fn from(s: String) -> Self {
        MlError::InvalidInput(s)
    }
}

impl From<&str> for MlError {
    fn from(s: &str) -> Self {
        MlError::InvalidInput(s.to_string())
    }
}

/// A type alias for `Result<T, MlError>` used throughout the CubicZan ML ecosystem.
pub type Result<T> = std::result::Result<T, MlError>;
