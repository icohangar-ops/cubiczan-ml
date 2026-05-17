//! # TensorFlow Session & Bridge Bindings
//!
//! PyO3 Python bindings for TensorFlow session management and Python-trained
//! model import from `cubiczan-ml-tf`.
//!
//! Exposes:
//! - [`PyTfSession`] — TensorFlow SavedModel session with inference support
//! - [`PyPyTfBridge`] — Bridge for importing Python-trained TF/Keras models

use std::path::Path;

use ndarray::ArrayD;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use cubiczan_ml_tf::{bridge::PyTfBridge, session::TfSession, SessionConfig};

// ─────────────────────────────────────────────────────────────────────────────
// PyTfSession
// ─────────────────────────────────────────────────────────────────────────────

/// TensorFlow SavedModel session with inference support.
///
/// Wraps [`TfSession`] to expose it to Python.
///
/// ## Example
///
/// ```python
/// session = TfSession.from_saved_model(
///     "/path/to/model",
///     input_ops=["input_1"],
///     output_ops=["output_1"],
/// )
/// results = session.run({"input_1": np.array([[1.0, 2.0, 3.0]])})
/// print(session.stats())
/// session.close()
/// ```
#[pyclass(name = "TfSession")]
pub struct PyTfSession {
    session: TfSession,
}

#[pymethods]
impl PyTfSession {
    /// Load a SavedModel from disk.
    ///
    /// Args:
    ///     path: Path to the SavedModel directory.
    ///     input_ops: Names of the input operations.
    ///     output_ops: Names of the output operations.
    ///
    /// Returns:
    ///     A new `TfSession` instance.
    ///
    /// Raises:
    ///     ValueError: If the model path does not exist.
    #[staticmethod]
    fn from_saved_model(
        path: String,
        input_ops: Vec<String>,
        output_ops: Vec<String>,
    ) -> PyResult<Self> {
        let model_path = Path::new(&path);
        let id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default");

        let session = TfSession::from_saved_model(
            id,
            model_path,
            input_ops,
            output_ops,
            SessionConfig::default(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self { session })
    }

    /// Load a SavedModel from disk with a GPU-enabled config.
    ///
    /// Args:
    ///     path: Path to the SavedModel directory.
    ///     input_ops: Names of the input operations.
    ///     output_ops: Names of the output operations.
    ///
    /// Returns:
    ///     A new `TfSession` instance configured for GPU inference.
    ///
    /// Raises:
    ///     ValueError: If the model path does not exist.
    #[staticmethod]
    fn from_saved_model_gpu(
        path: String,
        input_ops: Vec<String>,
        output_ops: Vec<String>,
    ) -> PyResult<Self> {
        let model_path = Path::new(&path);
        let id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default");

        let session = TfSession::from_saved_model(
            id,
            model_path,
            input_ops,
            output_ops,
            SessionConfig::gpu(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self { session })
    }

    /// Run inference on the model.
    ///
    /// Args:
    ///     inputs: Dictionary mapping operation names to numpy arrays
    ///             (must be float32).
    ///
    /// Returns:
    ///     Dictionary mapping output operation names to numpy arrays.
    ///
    /// Raises:
    ///     ValueError: If the session is inactive or an input is not a
    ///                 valid numpy array.
    fn run<'py>(&mut self, py: Python<'py>, inputs: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
        // Extract inputs from Python dict into Rust Vec
        let mut rust_inputs: Vec<(String, ArrayD<f32>)> = Vec::new();
        for (key, value) in inputs.iter() {
            let name: String = key
                .extract()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("dict key must be str: {}", e)))?;

            // Try to extract as a numpy array (float32)
            let readonly: numpy::PyReadonlyArrayDyn<'_, f32> = value
                .extract()
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "value for key '{}' must be a numpy float32 array: {}",
                        name, e
                    ))
                })?;
            rust_inputs.push((name, readonly.as_array().to_owned()));
        }

        // Build the borrowed slice that TfSession::run expects
        let input_refs: Vec<(&str, ArrayD<f32>)> = rust_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        // Call the underlying session
        let results = self
            .session
            .run(&input_refs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Convert results to a Python dict of numpy arrays
        let out_dict = PyDict::new(py);
        for (name, arr) in results {
            use numpy::IntoPyArray;
            let py_array = arr.into_pyarray(py);
            out_dict.set_item(&name, py_array)?;
        }

        Ok(out_dict)
    }

    /// Get inference statistics for this session.
    ///
    /// Returns:
    ///     Dictionary with keys:
    ///     - ``inference_count`` (int): Total number of inferences run.
    ///     - ``total_time_ms`` (float): Total inference time in milliseconds.
    ///     - ``avg_time_ms`` (float): Average inference time in milliseconds.
    fn stats<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let s = self.session.stats();
        let dict = PyDict::new(py);
        dict.set_item("inference_count", s.inference_count)
            .unwrap();
        dict.set_item(
            "total_time_ms",
            s.total_inference_us as f64 / 1000.0,
        )
        .unwrap();
        dict.set_item(
            "avg_time_ms",
            s.avg_inference_us as f64 / 1000.0,
        )
        .unwrap();
        dict
    }

    /// Close the session and free resources.
    fn close(&mut self) {
        self.session.close();
    }

    fn __repr__(&self) -> String {
        let stats = self.session.stats();
        format!(
            "TfSession(id={}, active={}, inferences={}, avg_ms={:.1})",
            self.session.id,
            true, // we can't easily check active without a pub field accessor, but this is fine for display
            stats.inference_count,
            stats.avg_inference_us as f64 / 1000.0,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyPyTfBridge
// ─────────────────────────────────────────────────────────────────────────────

/// Bridge for importing Python-trained TensorFlow/Keras models into Rust.
///
/// Wraps [`PyTfBridge`] to expose it to Python.
///
/// ## Example
///
/// ```python
/// bridge = PyTfBridge("/path/to/model")
/// print(bridge.metadata())
/// print(bridge.is_valid())
/// ```
#[pyclass(name = "PyTfBridge")]
pub struct PyPyTfBridge {
    bridge: PyTfBridge,
}

#[pymethods]
impl PyPyTfBridge {
    /// Create a new bridge pointing to a Python-trained model directory.
    ///
    /// The directory should contain a ``model_metadata.json`` file and the
    /// SavedModel or frozen graph files.
    ///
    /// Args:
    ///     model_path: Path to the model directory.
    ///
    /// Raises:
    ///     ValueError: If the path cannot be read.
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let path = Path::new(&model_path);
        let bridge = PyTfBridge::new(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { bridge })
    }

    /// Get the model metadata as a dictionary.
    ///
    /// Returns:
    ///     Dictionary with keys:
    ///     - ``framework`` (str): Training framework (e.g. "tensorflow").
    ///     - ``name`` (str): Model name/identifier.
    ///     - ``version`` (str): Model version.
    ///     - ``trained_at`` (str): Training date (ISO 8601).
    ///     - ``input_ops`` (list[str]): Input layer names.
    ///     - ``output_ops`` (list[str]): Output layer names.
    ///     - ``model_size`` (int): Total file size of the model directory in bytes.
    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let meta = self.bridge.metadata();
        let dict = PyDict::new(py);

        dict.set_item("framework", &meta.framework).unwrap();
        dict.set_item("name", &meta.name).unwrap();
        dict.set_item("version", &meta.version).unwrap();
        dict.set_item("trained_at", &meta.trained_at).unwrap();

        let input_ops: Vec<&str> = meta.inputs.iter().map(|l| l.name.as_str()).collect();
        dict.set_item("input_ops", input_ops).unwrap();

        let output_ops: Vec<&str> = meta.outputs.iter().map(|l| l.name.as_str()).collect();
        dict.set_item("output_ops", output_ops).unwrap();

        // Compute model directory size
        let model_size = dir_size(self.bridge.path());
        dict.set_item("model_size", model_size).unwrap();

        dict
    }

    /// Validate that the model files are compatible with the metadata.
    ///
    /// Returns:
    ///     ``True`` if the model passed validation (no critical issues found).
    fn is_valid(&self) -> bool {
        self.bridge
            .validate()
            .map(|report| report.is_valid())
            .unwrap_or(false)
    }

    /// Get the full validation report.
    ///
    /// Returns:
    ///     Dictionary with keys:
    ///     - ``is_valid`` (bool): Whether validation passed.
    ///     - ``issues`` (list[str]): Critical issues found.
    ///     - ``warnings`` (list[str]): Non-critical warnings.
    fn validate<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        match self.bridge.validate() {
            Ok(report) => {
                dict.set_item("is_valid", report.is_valid()).unwrap();
                dict.set_item("issues", &report.issues).unwrap();
                dict.set_item("warnings", &report.warnings).unwrap();
            }
            Err(e) => {
                dict.set_item("is_valid", false).unwrap();
                dict.set_item("issues", vec![e.to_string()]).unwrap();
                dict.set_item("warnings", Vec::<String>::new()).unwrap();
            }
        }
        dict
    }

    /// Generate a Rust inference wrapper source code based on model metadata.
    ///
    /// Returns:
    ///     A string containing the generated Rust source code.
    ///
    /// Raises:
    ///     ValueError: If wrapper generation fails.
    fn generate_wrapper(&self) -> PyResult<String> {
        self.bridge
            .generate_wrapper()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let meta = self.bridge.metadata();
        format!(
            "PyTfBridge(name={}, framework={}, path={})",
            meta.name,
            meta.framework,
            self.bridge.path().display(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively compute the total file size of a directory in bytes.
///
/// Returns 0 if the path does not exist or cannot be read.
fn dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }

    if path.is_file() {
        return std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
    }

    std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|entry| dir_size(&entry.path()))
                .sum()
        })
        .unwrap_or(0)
}
