//! PyO3 bindings for the `cubiczan-ml-nlp` crate.
//!
//! Exposes financial NLP primitives (sentiment analysis, NER, zero-shot
//! classification) to Python.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use cubiczan_ml_nlp::classifier::ZeroShotClassifier;
use cubiczan_ml_nlp::ner::FinancialNER;
use cubiczan_ml_nlp::sentiment::FinSentimentAnalyzer;

// ---------------------------------------------------------------------------
// Helper: compound score → label
// ---------------------------------------------------------------------------

/// Map a compound sentiment score to a simple bullish / bearish / neutral label.
#[pyfunction]
#[pyo3(signature = (compound))]
pub fn py_sentiment_label(compound: f64) -> String {
    if compound > 0.05 {
        "bullish".to_string()
    } else if compound < -0.05 {
        "bearish".to_string()
    } else {
        "neutral".to_string()
    }
}

// ---------------------------------------------------------------------------
// PyFinSentimentAnalyzer
// ---------------------------------------------------------------------------

/// Financial sentiment analyzer with sector-aware lexicon scoring.
///
/// Wraps [`cubiczan_ml_nlp::sentiment::FinSentimentAnalyzer`].
#[pyclass(name = "FinSentimentAnalyzer")]
pub struct PyFinSentimentAnalyzer {
    inner: FinSentimentAnalyzer,
}

#[pymethods]
impl PyFinSentimentAnalyzer {
    #[new]
    fn new() -> Self {
        Self {
            inner: FinSentimentAnalyzer::new(),
        }
    }

    /// Analyze a single text and return a dict with keys:
    /// positive, negative, neutral, compound, confidence, label.
    fn analyze<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let score = self
            .inner
            .analyze(text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("positive", score.positive)?;
        dict.set_item("negative", score.negative)?;
        dict.set_item("neutral", score.neutral)?;
        dict.set_item("compound", score.compound)?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("label", score.dominant_label())?;
        Ok(dict)
    }

    /// Analyze a batch of texts and return a list of dicts.
    fn analyze_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let scores = self
            .inner
            .analyze_batch(&refs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let mut results = Vec::with_capacity(scores.len());
        for score in &scores {
            let dict = PyDict::new(py);
            dict.set_item("positive", score.positive)?;
            dict.set_item("negative", score.negative)?;
            dict.set_item("neutral", score.neutral)?;
            dict.set_item("compound", score.compound)?;
            dict.set_item("confidence", score.confidence)?;
            dict.set_item("label", score.dominant_label())?;
            results.push(dict);
        }
        Ok(results)
    }

    /// Analyze text with sector-specific weighting (e.g. "earnings", "crypto", "fed").
    fn analyze_with_sector<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        sector: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let score = self
            .inner
            .analyze_with_sector(text, sector)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("positive", score.positive)?;
        dict.set_item("negative", score.negative)?;
        dict.set_item("neutral", score.neutral)?;
        dict.set_item("compound", score.compound)?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("label", score.dominant_label())?;
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// PyFinancialNER
// ---------------------------------------------------------------------------

/// Financial named-entity recognizer using rule-based extraction.
///
/// Wraps [`cubiczan_ml_nlp::ner::FinancialNER`].
#[pyclass(name = "FinancialNER")]
pub struct PyFinancialNER {
    inner: FinancialNER,
}

#[pymethods]
impl PyFinancialNER {
    #[new]
    fn new() -> Self {
        Self {
            inner: FinancialNER::new(),
        }
    }

    /// Extract named entities from text.
    ///
    /// Returns a list of dicts, each with keys: text, entity_type, start, end.
    fn extract<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let entities = self.inner.extract(text);
        let mut results = Vec::with_capacity(entities.len());
        for entity in &entities {
            let dict = PyDict::new(py);
            dict.set_item("text", &entity.text)?;
            dict.set_item("entity_type", entity.entity_type.to_string())?;
            dict.set_item("start", entity.start)?;
            dict.set_item("end", entity.end)?;
            results.push(dict);
        }
        Ok(results)
    }

    /// Extract monetary amounts from text and return them as floats.
    fn extract_amounts(&self, text: &str) -> PyResult<Vec<f64>> {
        Ok(self.inner.extract_amounts(text))
    }
}

// ---------------------------------------------------------------------------
// PyZeroShotClassifier
// ---------------------------------------------------------------------------

/// Zero-shot text classifier that categorises text into arbitrary labels.
///
/// Wraps [`cubiczan_ml_nlp::classifier::ZeroShotClassifier`].
#[pyclass(name = "ZeroShotClassifier")]
pub struct PyZeroShotClassifier {
    inner: ZeroShotClassifier,
}

#[pymethods]
impl PyZeroShotClassifier {
    #[new]
    fn new(candidate_labels: Vec<String>) -> Self {
        Self {
            inner: ZeroShotClassifier::new(candidate_labels),
        }
    }

    /// Classify text against the candidate labels.
    ///
    /// Returns a dict with keys: label, score, and a "scores" dict mapping
    /// every candidate label to its probability.
    fn classify<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .classify_zero_shot(text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let scores_dict = PyDict::new(py);
        for (label, prob) in &result.probabilities {
            scores_dict.set_item(label, *prob)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("label", &result.label)?;
        dict.set_item("score", result.score)?;
        dict.set_item("scores", scores_dict)?;
        Ok(dict)
    }
}
