//! # Device Types
//!
//! Enumerates compute devices available for ML workloads.

/// Represents the compute device on which to perform operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU computation.
    Cpu,
    /// GPU computation via CUDA, with an optional device index.
    Cuda(usize),
}

impl DeviceType {
    /// Returns `true` if this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceType::Cpu)
    }

    /// Returns `true` if this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, DeviceType::Cuda(_))
    }

    /// Returns the CUDA device index, or `None` for CPU.
    pub fn cuda_index(&self) -> Option<usize> {
        match self {
            DeviceType::Cpu => None,
            DeviceType::Cuda(idx) => Some(*idx),
        }
    }
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda(idx) => write!(f, "cuda:{}", idx),
        }
    }
}
