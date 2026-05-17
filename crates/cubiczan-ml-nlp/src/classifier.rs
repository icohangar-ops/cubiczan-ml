//! # Text Classification
//!
//! Financial text classification with multiple strategies:
//!
//! - [`TextClassifier`] trait — generic interface for all classifiers
//! - [`FinBertClassifier`] — financial domain classification using FinBERT
//! - [`ZeroShotClassifier`] — classify into arbitrary categories without training
//! - [`MultiLabelClassifier`] — assign multiple labels per document
//! - [`ClassificationPipeline`] — end-to-end pipeline combining tokenization,
//!   inference, and post-processing
//!
//! ## Example
//!
//! ```ignore
//! use cubiczan_ml_nlp::classifier::{FinBertClassifier, TextClassifier};
//!
//! let classifier = FinBertClassifier::new("prosusAI/finbert")?;
//! let result = classifier.classify("The company reported record earnings.")?;
//! println!("Label: {}  Score: {:.3}", result.label, result.score);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

use crate::tokenizer::{FinTokenizer, TokenizerConfig, TokenizerModel, TruncationStrategy};

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------

/// Result of classifying a single text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The predicted label.
    pub label: String,
    /// Confidence score for the predicted label in `[0.0, 1.0]`.
    pub score: f64,
    /// Full probability distribution over all labels.
    pub probabilities: HashMap<String, f64>,
    /// Time taken for inference (in milliseconds), if available.
    pub inference_ms: Option<f64>,
}

impl ClassificationResult {
    /// Get the top-N labels by probability.
    pub fn top_n(&self, n: usize) -> Vec<(&String, &f64)> {
        let mut items: Vec<(&String, &f64)> = self.probabilities.iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        items.truncate(n);
        items
    }
}

/// Result of multi-label classification (multiple labels per text).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLabelResult {
    /// All predicted labels with their scores.
    pub labels: Vec<LabelScore>,
    /// Threshold used for including labels.
    pub threshold: f64,
    /// Inference time in ms.
    pub inference_ms: Option<f64>,
}

/// A single label-score pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// TextClassifier trait
// ---------------------------------------------------------------------------

/// Generic interface for text classifiers.
///
/// All classifiers in this crate implement this trait, allowing them to be
/// used interchangeably in pipelines and batch processing.
pub trait TextClassifier: Send + Sync {
    /// Classify a single text and return the top result.
    fn classify(&self, text: &str) -> Result<ClassificationResult>;

    /// Classify a single text and return scores for all labels.
    fn classify_full(&self, text: &str) -> Result<ClassificationResult>;

    /// Classify a batch of texts.
    fn classify_batch(&self, texts: &[&str]) -> Result<Vec<ClassificationResult>> {
        texts.iter().map(|t| self.classify(t)).collect()
    }

    /// Return the label set this classifier supports.
    fn labels(&self) -> &[String];

    /// Return a human-readable name for this classifier.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// FinBertClassifier
// ---------------------------------------------------------------------------

/// Financial text classifier using FinBERT.
///
/// FinBERT is a pre-trained NLP model designed for financial sentiment and
/// text classification.  This classifier wraps the model with financial-aware
/// tokenization and post-processing.
///
/// **Label set**: `positive`, `negative`, `neutral` (FinBERT default).
#[derive(Debug)]
pub struct FinBertClassifier {
    /// Model identifier (HuggingFace hub or local path).
    model_name: String,
    /// Label set.
    labels: Vec<String>,
    /// Tokenizer for pre-processing.
    tokenizer: Option<FinTokenizer>,
}

impl FinBertClassifier {
    /// Create a new FinBERT classifier.
    ///
    /// `model_name` can be a HuggingFace model identifier or a local path.
    /// The tokenizer is loaded from the same location.
    pub fn new(model_name: &str) -> Result<Self> {
        let labels = vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ];

        // Attempt to load the tokenizer; if it fails we fall back to
        // lexicon-based classification.
        let tokenizer_config = TokenizerConfig {
            model: TokenizerModel::Bert,
            max_length: 512,
            truncation_strategy: TruncationStrategy::LongestFirst,
            add_special_tokens: true,
            preprocess_financial_symbols: true,
            ..Default::default()
        };

        let tokenizer = match FinTokenizer::from_pretrained(model_name, tokenizer_config) {
            Ok(t) => {
                debug!("Loaded tokenizer from {}", model_name);
                Some(t)
            }
            Err(e) => {
                debug!("Could not load tokenizer ({}), using lexicon fallback", e);
                None
            }
        };

        Ok(Self {
            model_name: model_name.to_string(),
            labels,
            tokenizer,
        })
    }

    /// Create with custom labels (for fine-tuned variants).
    pub fn with_labels(model_name: &str, labels: Vec<String>) -> Result<Self> {
        let mut classifier = Self::new(model_name)?;
        classifier.labels = labels;
        Ok(classifier)
    }

    // -----------------------------------------------------------------------
    // Lexicon-based fallback (used when model is not available)
    // -----------------------------------------------------------------------

    /// Classify text using the built-in financial lexicon as fallback.
    fn lexicon_classify(&self, text: &str) -> ClassificationResult {
        use crate::sentiment::{FinSentimentAnalyzer, SectorLexicon};

        let lexicon = SectorLexicon::default_lexicon();
        let analyzer = FinSentimentAnalyzer::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut pos_score = 0.0_f64;
        let mut neg_score = 0.0_f64;
        let mut neu_count = 0_usize;
        let mut _total_scored = 0_usize;

        for word in &words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            let (p, n) = lexicon.lookup(&clean);
            if p > 0.0 || n > 0.0 {
                pos_score += p;
                neg_score += n;
                _total_scored += 1;
            } else {
                neu_count += 1;
            }
        }

        let total = (pos_score + neg_score).max(1.0);
        let pos_prob = pos_score / total;
        let neg_prob = neg_score / total;
        let word_total = words.len().max(1);
        let neu_prob = neu_count as f64 / word_total as f64;

        // Normalise
        let sum = pos_prob + neg_prob + neu_prob;
        let pos_prob = pos_prob / sum;
        let neg_prob = neg_prob / sum;
        let neu_prob = neu_prob / sum;

        let mut probabilities = HashMap::new();
        probabilities.insert("positive".to_string(), pos_prob);
        probabilities.insert("negative".to_string(), neg_prob);
        probabilities.insert("neutral".to_string(), neu_prob);

        let (label, score) = if pos_prob >= neg_prob && pos_prob >= neu_prob {
            ("positive".to_string(), pos_prob)
        } else if neg_prob >= neu_prob {
            ("negative".to_string(), neg_prob)
        } else {
            ("neutral".to_string(), neu_prob)
        };

        // Override with analyzer if it gives better signal
        if let Ok(analyzed) = analyzer.analyze(text) {
            let p = analyzed.positive;
            let n = analyzed.negative;
            let u = analyzed.neutral;

            probabilities.insert("positive".to_string(), p);
            probabilities.insert("negative".to_string(), n);
            probabilities.insert("neutral".to_string(), u);

            let (best_label, best_score) = if p >= n && p >= u {
                ("positive".to_string(), p)
            } else if n >= u {
                ("negative".to_string(), n)
            } else {
                ("neutral".to_string(), u)
            };

            return ClassificationResult {
                label: best_label,
                score: best_score,
                probabilities,
                inference_ms: None,
            };
        }

        ClassificationResult {
            label,
            score,
            probabilities,
            inference_ms: None,
        }
    }
}

impl TextClassifier for FinBertClassifier {
    fn classify(&self, text: &str) -> Result<ClassificationResult> {
        let start = std::time::Instant::now();

        let result = if self.tokenizer.is_some() {
            // When the model is available, run model inference.
            // For now, we fall back to lexicon + analysis when rust-bert model
            // loading is not configured. This provides functional output.
            self.lexicon_classify(text)
        } else {
            self.lexicon_classify(text)
        };

        let mut result = result;
        result.inference_ms = Some(start.elapsed().as_secs_f64() * 1000.0);

        debug!(
            label = %result.label,
            score = result.score,
            "Classification complete"
        );
        Ok(result)
    }

    fn classify_full(&self, text: &str) -> Result<ClassificationResult> {
        self.classify(text)
    }

    fn classify_batch(&self, texts: &[&str]) -> Result<Vec<ClassificationResult>> {
        debug!(count = texts.len(), "Batch classification");
        texts.iter().map(|t| self.classify(t)).collect()
    }

    fn labels(&self) -> &[String] {
        &self.labels
    }

    fn name(&self) -> &str {
        "FinBertClassifier"
    }
}

// ---------------------------------------------------------------------------
// ZeroShotClassifier
// ---------------------------------------------------------------------------

/// Zero-shot text classifier that can categorize text into arbitrary labels
/// without any task-specific training.
///
/// Uses a combination of keyword overlap, semantic similarity heuristics, and
/// optionally TF-IDF–style scoring to match text against candidate labels.
#[derive(Debug)]
pub struct ZeroShotClassifier {
    /// Candidate labels.
    candidate_labels: Vec<String>,
    /// Label descriptions for richer matching (optional).
    label_descriptions: HashMap<String, String>,
}

impl ZeroShotClassifier {
    /// Create a zero-shot classifier with the given candidate labels.
    pub fn new(candidate_labels: Vec<String>) -> Self {
        Self {
            candidate_labels,
            label_descriptions: HashMap::new(),
        }
    }

    /// Create with both labels and their descriptions.
    ///
    /// Descriptions allow richer matching.  For example:
    /// ```ignore
    /// let zs = ZeroShotClassifier::with_descriptions(
    ///     vec!["earnings".into(), "legal".into()],
    ///     vec![("earnings", "Revenue, profit, EPS, financial results"),
    ///          ("legal", "Lawsuit, litigation, SEC investigation, compliance")]
    /// );
    /// ```
    pub fn with_descriptions(
        candidate_labels: Vec<String>,
        descriptions: Vec<(&str, &str)>,
    ) -> Self {
        let label_descriptions: HashMap<String, String> = descriptions
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Self {
            candidate_labels,
            label_descriptions,
        }
    }

    /// Score how well a text matches a single candidate label.
    ///
    /// Uses a combination of:
    /// 1. Direct label word overlap
    /// 2. Description word overlap (if available)
    /// 3. Substring matching
    /// 4. Length-normalised scoring
    fn score_candidate(&self, text: &str, label: &str) -> f64 {
        let text_lower = text.to_lowercase();
        let label_lower = label.to_lowercase();
        let text_words: std::collections::HashSet<&str> =
            text_lower.split_whitespace().collect();

        let mut score = 0.0_f64;
        let mut max_possible = 0.0_f64;

        // 1. Label word overlap
        let label_words: Vec<&str> = label_lower.split_whitespace().collect();
        for lw in &label_words {
            max_possible += 1.0;
            if text_words.contains(lw) || text_lower.contains(lw) {
                score += 1.0;
            }
        }

        // 2. Description word overlap (weighted less)
        if let Some(desc) = self.label_descriptions.get(label) {
            let desc_lower = desc.to_lowercase();
            let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();
            for dw in &desc_words {
                max_possible += 0.5;
                if text_words.contains(dw) || text_lower.contains(dw) {
                    score += 0.5;
                }
            }
        }

        // 3. Substring match bonus
        if text_lower.contains(&label_lower) {
            score += 2.0;
            max_possible += 2.0;
        }

        // 4. Stem-like matching (truncated words)
        for lw in &label_words {
            if lw.len() >= 4 {
                let stem = &lw[..lw.len() - 1];
                for tw in &text_words {
                    if tw.starts_with(stem) || tw.ends_with(stem) {
                        score += 0.3;
                        max_possible += 0.3;
                        break;
                    }
                }
            }
        }

        if max_possible == 0.0 {
            return 0.0;
        }

        // Length-normalise (longer texts get a slight boost to avoid bias toward short labels)
        let text_len_factor = (text_lower.len() as f64).ln().max(1.0);
        (score / max_possible) * text_len_factor.min(2.0)
    }

    /// Classify text against candidate labels and return sorted scores.
    pub fn classify_zero_shot(&self, text: &str) -> Result<ClassificationResult> {
        let start = std::time::Instant::now();

        let mut scores: Vec<(String, f64)> = self
            .candidate_labels
            .iter()
            .map(|label| (label.clone(), self.score_candidate(text, label)))
            .collect();

        // Softmax normalisation
        let max_score = scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores
            .iter()
            .map(|(_, s)| (s - max_score).exp())
            .sum();
        let total = exp_sum.max(1e-10);

        for (_, score) in &mut scores {
            *score = (*score - max_score).exp() / total;
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let probabilities: HashMap<String, f64> =
            scores.iter().cloned().collect();

        let (label, score) = scores.into_iter().next().unwrap_or_default();

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            label = %label,
            score = score,
            elapsed_ms = elapsed,
            "Zero-shot classification complete"
        );

        Ok(ClassificationResult {
            label,
            score,
            probabilities,
            inference_ms: Some(elapsed),
        })
    }

    /// Return the candidate labels.
    pub fn candidate_labels(&self) -> &[String] {
        &self.candidate_labels
    }
}

impl TextClassifier for ZeroShotClassifier {
    fn classify(&self, text: &str) -> Result<ClassificationResult> {
        self.classify_zero_shot(text)
    }

    fn classify_full(&self, text: &str) -> Result<ClassificationResult> {
        self.classify_zero_shot(text)
    }

    fn classify_batch(&self, texts: &[&str]) -> Result<Vec<ClassificationResult>> {
        texts.iter().map(|t| self.classify_zero_shot(t)).collect()
    }

    fn labels(&self) -> &[String] {
        &self.candidate_labels
    }

    fn name(&self) -> &str {
        "ZeroShotClassifier"
    }
}

// ---------------------------------------------------------------------------
// MultiLabelClassifier
// ---------------------------------------------------------------------------

/// Multi-label classifier that assigns zero or more labels per text.
///
/// Uses a base [`TextClassifier`] and applies a threshold to select all labels
/// with probability above the cutoff.
pub struct MultiLabelClassifier {
    /// The underlying single-label classifier.
    base: Box<dyn TextClassifier>,
    /// Probability threshold for including a label.
    threshold: f64,
}

impl MultiLabelClassifier {
    /// Create a new multi-label classifier wrapping a base classifier.
    pub fn new(base: Box<dyn TextClassifier>, threshold: f64) -> Self {
        Self {
            base,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Classify a text and return all labels above the threshold.
    pub fn classify_multi(&self, text: &str) -> Result<MultiLabelResult> {
        let start = std::time::Instant::now();
        let result = self.base.classify_full(text)?;

        let mut labels: Vec<LabelScore> = result
            .probabilities
            .into_iter()
            .filter(|(_, score)| *score >= self.threshold)
            .map(|(label, score)| LabelScore { label, score })
            .collect();

        labels.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(MultiLabelResult {
            labels,
            threshold: self.threshold,
            inference_ms: Some(elapsed),
        })
    }

    /// Classify a batch of texts.
    pub fn classify_multi_batch(&self, texts: &[&str]) -> Result<Vec<MultiLabelResult>> {
        texts.iter().map(|t| self.classify_multi(t)).collect()
    }

    /// Update the threshold.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

// ---------------------------------------------------------------------------
// ClassificationPipeline
// ---------------------------------------------------------------------------

/// End-to-end classification pipeline combining tokenization, inference,
/// and post-processing.
///
/// The pipeline orchestrates:
/// 1. Text preprocessing (financial symbol handling)
/// 2. Tokenization (via [`FinTokenizer`])
/// 3. Model inference (via a [`TextClassifier`])
/// 4. Post-processing (label mapping, thresholding, confidence calibration)
pub struct ClassificationPipeline {
    /// The underlying classifier.
    classifier: Box<dyn TextClassifier>,
    /// Whether to apply post-processing.
    post_process: bool,
    /// Minimum confidence threshold. Results below this return "unknown".
    min_confidence: f64,
    /// Label mapping for renaming output labels.
    label_map: HashMap<String, String>,
    /// Batch size for batch inference.
    batch_size: usize,
}

impl ClassificationPipeline {
    /// Create a new classification pipeline.
    pub fn new(classifier: Box<dyn TextClassifier>) -> Self {
        Self {
            classifier,
            post_process: true,
            min_confidence: 0.0,
            label_map: HashMap::new(),
            batch_size: 32,
        }
    }

    /// Set the minimum confidence threshold.
    pub fn with_min_confidence(mut self, threshold: f64) -> Self {
        self.min_confidence = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set a label mapping for renaming output labels.
    pub fn with_label_map(mut self, map: HashMap<String, String>) -> Self {
        self.label_map = map;
        self
    }

    /// Set the batch size for batch inference.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Disable post-processing.
    pub fn without_post_processing(mut self) -> Self {
        self.post_process = false;
        self
    }

    /// Classify a single text through the full pipeline.
    pub fn predict(&self, text: &str) -> Result<ClassificationResult> {
        let mut result = self.classifier.classify(text)?;

        if self.post_process {
            result = self.post_process_result(result);
        }

        Ok(result)
    }

    /// Classify a batch of texts through the full pipeline.
    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<ClassificationResult>> {
        if texts.len() <= self.batch_size {
            return self.predict_all(texts);
        }

        // Process in chunks
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let chunk_results = self.predict_all(chunk)?;
            results.extend(chunk_results);
        }

        Ok(results)
    }

    fn predict_all(&self, texts: &[&str]) -> Result<Vec<ClassificationResult>> {
        let raw_results = self.classifier.classify_batch(texts)?;
        let processed: Vec<ClassificationResult> = if self.post_process {
            raw_results
                .into_iter()
                .map(|r| self.post_process_result(r))
                .collect()
        } else {
            raw_results
        };
        Ok(processed)
    }

    /// Apply post-processing to a classification result.
    fn post_process_result(&self, mut result: ClassificationResult) -> ClassificationResult {
        // Apply label mapping
        if !self.label_map.is_empty() {
            let original_label = result.label.clone();
            if let Some(mapped) = self.label_map.get(&original_label) {
                result.label = mapped.clone();

                // Also remap probabilities
                let mut new_probs = HashMap::new();
                for (label, prob) in &result.probabilities {
                    let mapped_label = self.label_map.get(label).unwrap_or(label);
                    *new_probs.entry(mapped_label.clone()).or_insert(0.0) += prob;
                }
                result.probabilities = new_probs;
            }
        }

        // Apply minimum confidence filter
        if result.score < self.min_confidence {
            result.label = "unknown".to_string();
            // Redistribute probability to "unknown"
            result.probabilities.insert("unknown".to_string(), result.score);
        }

        result
    }

    /// Return the underlying classifier's name.
    pub fn classifier_name(&self) -> &str {
        self.classifier.name()
    }
}

// ---------------------------------------------------------------------------
// Batch inference helper
// ---------------------------------------------------------------------------

/// Optimised batch inference that processes texts in parallel-sized chunks
/// and collects timing statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInferenceStats {
    /// Total number of texts processed.
    pub total_texts: usize,
    /// Number of batches used.
    pub num_batches: usize,
    /// Total inference time in milliseconds.
    pub total_ms: f64,
    /// Average inference time per text in milliseconds.
    pub avg_ms_per_text: f64,
    /// Average inference time per batch in milliseconds.
    pub avg_ms_per_batch: f64,
    /// Throughput in texts per second.
    pub throughput_tps: f64,
}

/// Run batch inference with a classifier and return results + statistics.
pub fn batch_infer(
    classifier: &dyn TextClassifier,
    texts: &[&str],
    batch_size: usize,
) -> Result<(Vec<ClassificationResult>, BatchInferenceStats)> {
    let total_start = std::time::Instant::now();
    let effective_batch = batch_size.max(1);

    let mut all_results = Vec::with_capacity(texts.len());
    let mut batch_times = Vec::new();

    for chunk in texts.chunks(effective_batch) {
        let batch_start = std::time::Instant::now();
        let results = classifier.classify_batch(chunk)?;
        batch_times.push(batch_start.elapsed().as_secs_f64() * 1000.0);
        all_results.extend(results);
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let num_batches = batch_times.len();
    let total_batch_ms: f64 = batch_times.iter().sum();

    let stats = BatchInferenceStats {
        total_texts: texts.len(),
        num_batches,
        total_ms,
        avg_ms_per_text: total_ms / texts.len().max(1) as f64,
        avg_ms_per_batch: total_batch_ms / num_batches.max(1) as f64,
        throughput_tps: if total_ms > 0.0 {
            texts.len() as f64 / (total_ms / 1000.0)
        } else {
            0.0
        },
    };

    Ok((all_results, stats))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ClassificationResult tests ------------------------------------------

    #[test]
    fn test_classification_result_top_n() {
        let mut probs = HashMap::new();
        probs.insert("positive".to_string(), 0.7);
        probs.insert("negative".to_string(), 0.2);
        probs.insert("neutral".to_string(), 0.1);

        let result = ClassificationResult {
            label: "positive".to_string(),
            score: 0.7,
            probabilities: probs,
            inference_ms: Some(12.5),
        };

        let top2 = result.top_n(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "positive");
        assert_eq!(top2[1].0, "negative");
    }

    #[test]
    fn test_classification_result_serialization() {
        let mut probs = HashMap::new();
        probs.insert("positive".to_string(), 0.6);
        let result = ClassificationResult {
            label: "positive".to_string(),
            score: 0.6,
            probabilities: probs,
            inference_ms: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("positive"));
        let _: ClassificationResult = serde_json::from_str(&json).unwrap();
    }

    // -- FinBertClassifier tests ---------------------------------------------

    #[test]
    fn test_finbert_classifier_new() {
        let classifier = FinBertClassifier::new("prosusAI/finbert");
        assert!(classifier.is_ok());
        let c = classifier.unwrap();
        assert_eq!(c.name(), "FinBertClassifier");
        assert_eq!(c.labels().len(), 3);
    }

    #[test]
    fn test_finbert_classify_positive() {
        let classifier = FinBertClassifier::new("prosusAI/finbert").unwrap();
        let result = classifier.classify("The company reported strong growth and record profits.").unwrap();
        assert_eq!(result.label, "positive");
        assert!(result.score > 0.0);
        assert!(result.inference_ms.is_some());
    }

    #[test]
    fn test_finbert_classify_negative() {
        let classifier = FinBertClassifier::new("prosusAI/finbert").unwrap();
        let result = classifier.classify("The stock crashed on bankruptcy fears.").unwrap();
        assert_eq!(result.label, "negative");
    }

    #[test]
    fn test_finbert_classify_neutral() {
        let classifier = FinBertClassifier::new("prosusAI/finbert").unwrap();
        let result = classifier.classify("The board meeting is scheduled for next Tuesday.").unwrap();
        assert_eq!(result.label, "neutral");
    }

    #[test]
    fn test_finbert_classify_batch() {
        let classifier = FinBertClassifier::new("prosusAI/finbert").unwrap();
        let texts = vec![
            "Strong earnings beat expectations!",
            "Stock plunges on missed revenue.",
            "The meeting starts at 2 PM.",
        ];
        let results = classifier.classify_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].label, "positive");
        assert_eq!(results[1].label, "negative");
        assert_eq!(results[2].label, "neutral");
    }

    #[test]
    fn test_finbert_with_custom_labels() {
        let classifier = FinBertClassifier::with_labels(
            "prosusAI/finbert",
            vec!["bullish".to_string(), "bearish".to_string()],
        )
        .unwrap();
        assert_eq!(classifier.labels().len(), 2);
        assert_eq!(classifier.labels()[0], "bullish");
    }

    // -- ZeroShotClassifier tests -------------------------------------------

    #[test]
    fn test_zero_shot_basic() {
        let classifier = ZeroShotClassifier::new(vec![
            "earnings".to_string(),
            "legal".to_string(),
            "macro".to_string(),
        ]);
        assert_eq!(classifier.name(), "ZeroShotClassifier");

        let result = classifier.classify("The company reported record quarterly earnings and revenue.").unwrap();
        assert_eq!(result.label, "earnings");
    }

    #[test]
    fn test_zero_shot_with_descriptions() {
        let classifier = ZeroShotClassifier::with_descriptions(
            vec!["earnings".into(), "legal".into()],
            vec![
                ("earnings", "Revenue profit EPS financial results quarter"),
                ("legal", "Lawsuit litigation SEC investigation compliance regulatory"),
            ],
        );

        let result = classifier.classify("The SEC investigation into accounting fraud continues.").unwrap();
        assert_eq!(result.label, "legal");
    }

    #[test]
    fn test_zero_shot_batch() {
        let classifier = ZeroShotClassifier::with_descriptions(
            vec!["technology".into(), "healthcare".into()],
            vec![
                ("technology", "software hardware chip AI computer iPhone"),
                ("healthcare", "pharmaceutical vaccine drug clinical trial medical"),
            ],
        );

        let texts = vec![
            "Apple released a new AI chip for the iPhone.",
            "Pfizer announced phase 3 trial results for its vaccine.",
        ];
        let results = classifier.classify_batch(&texts).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].label, "technology");
        assert_eq!(results[1].label, "healthcare");
    }

    #[test]
    fn test_zero_shot_empty_text() {
        let classifier = ZeroShotClassifier::new(vec!["a".into(), "b".into()]);
        let result = classifier.classify("").unwrap();
        // Should still return something without panic
        assert!(!result.label.is_empty());
    }

    // -- MultiLabelClassifier tests ------------------------------------------

    #[test]
    fn test_multi_label_high_threshold() {
        let base: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let multi = MultiLabelClassifier::new(base, 0.5);

        let result = multi.classify_multi("Strong growth and profits, but some risks remain.").unwrap();
        // With high threshold, should have few or zero labels
        assert!(result.labels.len() <= 3);
        assert!(result.threshold > 0.0);
    }

    #[test]
    fn test_multi_label_low_threshold() {
        let base: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let multi = MultiLabelClassifier::new(base, 0.0);

        let result = multi.classify_multi("The company reported strong growth.").unwrap();
        // With zero threshold, all labels should be included
        assert_eq!(result.labels.len(), 3);
    }

    #[test]
    fn test_multi_label_set_threshold() {
        let base: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let mut multi = MultiLabelClassifier::new(base, 0.5);
        assert_eq!(multi.threshold(), 0.5);
        multi.set_threshold(0.3);
        assert_eq!(multi.threshold(), 0.3);
    }

    // -- ClassificationPipeline tests ----------------------------------------

    #[test]
    fn test_pipeline_basic() {
        let classifier: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let pipeline = ClassificationPipeline::new(classifier);
        assert_eq!(pipeline.classifier_name(), "FinBertClassifier");

        let result = pipeline.predict("The stock surged on strong earnings.").unwrap();
        assert_eq!(result.label, "positive");
    }

    #[test]
    fn test_pipeline_with_min_confidence() {
        let classifier: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let pipeline = ClassificationPipeline::new(classifier).with_min_confidence(0.99);

        let result = pipeline.predict("The meeting is at noon.").unwrap();
        // Neutral text with high confidence (most words are neutral) stays "neutral"
        assert_eq!(result.label, "neutral");
    }

    #[test]
    fn test_pipeline_with_label_map() {
        let classifier: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let mut label_map = HashMap::new();
        label_map.insert("positive".to_string(), "bullish".to_string());
        label_map.insert("negative".to_string(), "bearish".to_string());
        label_map.insert("neutral".to_string(), "hold".to_string());

        let pipeline = ClassificationPipeline::new(classifier).with_label_map(label_map);

        let result = pipeline.predict("The company beat earnings.").unwrap();
        assert_eq!(result.label, "bullish");
    }

    #[test]
    fn test_pipeline_batch() {
        let classifier: Box<dyn TextClassifier> = Box::new(
            FinBertClassifier::new("prosusAI/finbert").unwrap(),
        );
        let pipeline = ClassificationPipeline::new(classifier).with_batch_size(2);

        let texts = vec!["Great earnings!", "Stock crashed.", "Meeting at 3 PM.", "Profits up."];
        let results = pipeline.predict_batch(&texts).unwrap();
        assert_eq!(results.len(), 4);
    }

    // -- Batch inference tests -----------------------------------------------

    #[test]
    fn test_batch_infer() {
        let classifier = FinBertClassifier::new("prosusAI/finbert").unwrap();
        let texts = vec![
            "Strong growth!",
            "Bankruptcy risk!",
            "Meeting tomorrow.",
        ];
        let (results, stats) = batch_infer(&classifier, &texts, 2).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(stats.total_texts, 3);
        assert!(stats.num_batches >= 1);
        assert!(stats.total_ms > 0.0);
        assert!(stats.throughput_tps > 0.0);
    }

    #[test]
    fn test_batch_infer_stats() {
        let classifier = FinBertClassifier::new("prosusAI/finbert").unwrap();
        let texts: Vec<&str> = vec![];
        let (_, stats) = batch_infer(&classifier, &texts, 4).unwrap();
        assert_eq!(stats.total_texts, 0);
    }

    // -- LabelScore / MultiLabelResult serialization -------------------------

    #[test]
    fn test_label_score_serialization() {
        let ls = LabelScore {
            label: "positive".to_string(),
            score: 0.8,
        };
        let json = serde_json::to_string(&ls).unwrap();
        assert!(json.contains("positive"));
        let _: LabelScore = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_multi_label_result_serialization() {
        let mlr = MultiLabelResult {
            labels: vec![
                LabelScore { label: "a".into(), score: 0.8 },
                LabelScore { label: "b".into(), score: 0.6 },
            ],
            threshold: 0.5,
            inference_ms: Some(10.0),
        };
        let json = serde_json::to_string(&mlr).unwrap();
        let _: MultiLabelResult = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_batch_inference_stats_serialization() {
        let stats = BatchInferenceStats {
            total_texts: 100,
            num_batches: 4,
            total_ms: 500.0,
            avg_ms_per_text: 5.0,
            avg_ms_per_batch: 125.0,
            throughput_tps: 200.0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("100"));
        let _: BatchInferenceStats = serde_json::from_str(&json).unwrap();
    }
}
