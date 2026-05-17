//! # Financial Sentiment Analysis
//!
//! Domain-specific sentiment analysis for financial text, including earnings
//! calls, SEC filings, news headlines, and social media (Reddit/Twitter).
//!
//! ## Key Types
//!
//! - [`SentimentScore`] — per-label probability distribution (positive / negative / neutral / compound)
//! - [`FinSentimentAnalyzer`] — main analyzer combining lexicon-based + model-based approaches
//! - [`FedTone`] — hawkish / dovish / neutral classification for central bank communications
//! - [`SocialMediaSentimentAnalyzer`] — Reddit / Twitter specific preprocessing and scoring
//! - [`SentimentAggregation`] — aggregate sentiment across document collections
//!
//! ## Example
//!
//! ```ignore
//! use cubiczan_ml_nlp::sentiment::FinSentimentAnalyzer;
//!
//! let analyzer = FinSentimentAnalyzer::new()?;
//! let score = analyzer.analyze("Apple exceeded Q3 revenue expectations by 12%.")?;
//! println!("positive={:.2}  negative={:.2}  compound={:.2}",
//!           score.positive, score.negative, score.compound);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

// ---------------------------------------------------------------------------
// SentimentScore
// ---------------------------------------------------------------------------

/// Normalised sentiment score for a single piece of text.
///
/// Each field is a `f64` in `[0.0, 1.0]`.  The `compound` score ranges from
/// `[-1.0, 1.0]` and is computed as `positive - negative`, optionally scaled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    /// Probability / intensity of positive sentiment.
    pub positive: f64,
    /// Probability / intensity of negative sentiment.
    pub negative: f64,
    /// Probability / intensity of neutral sentiment.
    pub neutral: f64,
    /// Composite score in `[-1.0, 1.0]`.
    pub compound: f64,
    /// Optional confidence value in `[0.0, 1.0]`.
    pub confidence: f64,
}

impl SentimentScore {
    /// Create a new score.  Panics if any value is out of range.
    pub fn new(positive: f64, negative: f64, neutral: f64, compound: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&positive),
            "positive must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&negative),
            "negative must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&neutral),
            "neutral must be in [0, 1]"
        );
        assert!(
            (-1.0..=1.0).contains(&compound),
            "compound must be in [-1, 1]"
        );
        let confidence = Self::default_confidence(positive, negative, neutral);
        Self {
            positive,
            negative,
            neutral,
            compound,
            confidence,
        }
    }

    /// Create with an explicit confidence value.
    pub fn with_confidence(
        positive: f64,
        negative: f64,
        neutral: f64,
        compound: f64,
        confidence: f64,
    ) -> Self {
        Self {
            positive,
            negative,
            neutral,
            compound,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Compute a simple confidence: 1 − entropy(normalised probs).
    fn default_confidence(pos: f64, neg: f64, neu: f64) -> f64 {
        let total = pos + neg + neu;
        if total == 0.0 {
            return 0.0;
        }
        let p = pos / total;
        let n = neg / total;
        let u = neu / total;
        // Shannon entropy
        let entropy = -p.mul_add(p.ln(), -n.mul_add(n.ln(), -u.mul_add(u.ln(), 0.0)));
        // Normalise so that uniform distribution → 0 confidence.
        (1.0 - entropy / std::f64::consts::LN_2).max(0.0)
    }

    /// Dominant label ("positive", "negative", or "neutral").
    pub fn dominant_label(&self) -> &'static str {
        if self.positive >= self.negative && self.positive >= self.neutral {
            "positive"
        } else if self.negative >= self.neutral {
            "negative"
        } else {
            "neutral"
        }
    }

    /// A fully neutral score.
    pub fn neutral() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.0)
    }

    /// A strongly positive score.
    pub fn strongly_positive() -> Self {
        Self::new(0.95, 0.02, 0.03, 0.93)
    }

    /// A strongly negative score.
    pub fn strongly_negative() -> Self {
        Self::new(0.02, 0.95, 0.03, -0.93)
    }
}

impl Default for SentimentScore {
    fn default() -> Self {
        Self::neutral()
    }
}

// ---------------------------------------------------------------------------
// Sector-specific lexicons
// ---------------------------------------------------------------------------

/// A simple lexicon entry: word → (positive_weight, negative_weight).
type LexiconEntry = (f64, f64);

/// Built-in financial sentiment lexicon organised by sector / domain.
#[derive(Debug, Clone)]
pub struct SectorLexicon {
    /// General financial terms.
    general: HashMap<String, LexiconEntry>,
    /// Earnings-call specific terms.
    earnings: HashMap<String, LexiconEntry>,
    /// Central bank / monetary policy terms.
    central_bank: HashMap<String, LexiconEntry>,
    /// Crypto / DeFi terms.
    crypto: HashMap<String, LexiconEntry>,
    /// Social media slang mapped to sentiment.
    social: HashMap<String, LexiconEntry>,
}

impl SectorLexicon {
    /// Build a default lexicon with curated financial terms.
    pub fn default_lexicon() -> Self {
        let mut general: HashMap<String, LexiconEntry> = HashMap::new();
        // Positive general terms
        for (word, score) in [
            ("beat", 0.8),
            ("exceeded", 0.9),
            ("growth", 0.6),
            ("profit", 0.7),
            ("surge", 0.8),
            ("rally", 0.7),
            ("outperform", 0.8),
            ("upgrade", 0.7),
            ("bullish", 0.9),
            ("dividend", 0.4),
            ("buyback", 0.5),
            ("strong", 0.6),
            ("record", 0.5),
            ("innovation", 0.4),
            ("expansion", 0.5),
            ("margin", 0.3),
        ] {
            general.insert(word.to_string(), (score, 0.0));
        }
        // Negative general terms
        for (word, score) in [
            ("missed", 0.8),
            ("decline", 0.7),
            ("loss", 0.8),
            ("drop", 0.6),
            ("crash", 0.9),
            ("bearish", 0.9),
            ("downgrade", 0.8),
            ("debt", 0.4),
            ("default", 0.9),
            ("bankruptcy", 1.0),
            ("litigation", 0.6),
            ("investigation", 0.5),
            ("layoff", 0.7),
            ("cuts", 0.5),
            ("warning", 0.5),
            ("risk", 0.3),
        ] {
            general.insert(word.to_string(), (0.0, score));
        }

        // Earnings specific
        let mut earnings: HashMap<String, LexiconEntry> = HashMap::new();
        for (word, pos, neg) in [
            ("eps", 0.5, 0.3),
            ("guidance", 0.3, 0.4),
            ("outlook", 0.3, 0.3),
            ("revenue", 0.3, 0.2),
            ("forecast", 0.2, 0.3),
            ("earnings", 0.3, 0.3),
            ("quarterly", 0.0, 0.0),
            ("sequential", 0.1, 0.1),
            ("year-over-year", 0.2, 0.1),
            ("yoy", 0.2, 0.1),
            ("above expectations", 0.9, 0.0),
            ("below expectations", 0.0, 0.9),
            ("in-line", 0.2, 0.0),
        ] {
            earnings.insert(word.to_string(), (pos, neg));
        }

        // Central bank
        let mut central_bank: HashMap<String, LexiconEntry> = HashMap::new();
        for (word, pos, neg) in [
            ("rate hike", 0.0, 0.5),
            ("rate cut", 0.6, 0.0),
            ("hawkish", 0.0, 0.7),
            ("dovish", 0.7, 0.0),
            ("tightening", 0.0, 0.6),
            ("easing", 0.6, 0.0),
            ("inflation", 0.0, 0.5),
            ("disinflation", 0.5, 0.0),
            ("recession", 0.0, 0.8),
            ("soft landing", 0.7, 0.0),
            ("hard landing", 0.0, 0.8),
            ("quantitative easing", 0.5, 0.0),
            ("quantitative tightening", 0.0, 0.5),
            ("pause", 0.3, 0.0),
            ("patience", 0.3, 0.0),
            ("data dependent", 0.1, 0.1),
        ] {
            central_bank.insert(word.to_string(), (pos, neg));
        }

        // Crypto
        let mut crypto: HashMap<String, LexiconEntry> = HashMap::new();
        for (word, pos, neg) in [
            ("halving", 0.6, 0.0),
            ("bull run", 0.8, 0.0),
            ("pump", 0.6, 0.3),
            ("dump", 0.0, 0.8),
            ("rug pull", 0.0, 1.0),
            ("hack", 0.0, 0.9),
            ("exploit", 0.0, 0.8),
            ("tvl", 0.4, 0.0),
            ("apy", 0.5, 0.0),
            ("staking", 0.3, 0.0),
            ("airdrop", 0.5, 0.0),
            ("fomo", 0.4, 0.2),
            ("fud", 0.0, 0.6),
            ("dyor", 0.0, 0.0),
            ("rekt", 0.0, 0.8),
            ("to the moon", 0.9, 0.0),
        ] {
            crypto.insert(word.to_string(), (pos, neg));
        }

        // Social media slang
        let mut social: HashMap<String, LexiconEntry> = HashMap::new();
        for (word, pos, neg) in [
            ("apes", 0.5, 0.0),
            ("diamond hands", 0.7, 0.0),
            ("paper hands", 0.0, 0.7),
            ("yolo", 0.5, 0.3),
            ("stonks", 0.6, 0.0),
            ("bagholder", 0.0, 0.8),
            ("tendies", 0.6, 0.0),
            ("dd", 0.3, 0.0),
            ("shill", 0.1, 0.3),
            ("fud", 0.0, 0.5),
            ("hopium", 0.5, 0.2),
        ] {
            social.insert(word.to_string(), (pos, neg));
        }

        Self {
            general,
            earnings,
            central_bank,
            crypto,
            social,
        }
    }

    /// Look up a word across all sub-lexicons and return the combined score.
    ///
    /// Returns `(positive_weight, negative_weight)`.
    pub fn lookup(&self, word: &str) -> LexiconEntry {
        let lower = word.to_lowercase();
        let mut total_pos = 0.0;
        let mut total_neg = 0.0;
        let mut found = false;

        for lexicon in [&self.general, &self.earnings, &self.central_bank, &self.crypto, &self.social] {
            if let Some((p, n)) = lexicon.get(&lower) {
                total_pos += p;
                total_neg += n;
                found = true;
            }
        }

        if found {
            (total_pos.min(1.0), total_neg.min(1.0))
        } else {
            (0.0, 0.0)
        }
    }

    /// Add a custom entry to the general lexicon.
    pub fn add_entry(&mut self, word: &str, pos: f64, neg: f64) {
        self.general.insert(word.to_lowercase(), (pos, neg));
    }
}

// ---------------------------------------------------------------------------
// FinSentimentAnalyzer
// ---------------------------------------------------------------------------

/// Financial sentiment analyzer combining lexicon-based scoring with optional
/// ML model inference via `rust-bert`.
///
/// When no model is loaded the analyzer falls back to a high-quality
/// lexicon-based approach with sector-aware weighting and negation handling.
#[derive(Debug)]
pub struct FinSentimentAnalyzer {
    lexicon: SectorLexicon,
    /// Optional loaded model name (for future rust-bert integration).
    model_name: Option<String>,
}

impl FinSentimentAnalyzer {
    /// Create a new analyzer with the built-in financial lexicon.
    ///
    /// To use a pre-trained model pass its HuggingFace identifier via
    /// [`Self::with_model`].
    pub fn new() -> Self {
        Self {
            lexicon: SectorLexicon::default_lexicon(),
            model_name: None,
        }
    }

    /// Create an analyzer configured to use a specific sentiment model.
    ///
    /// Currently this records the model name for future integration.
    pub fn with_model(model_name: &str) -> Self {
        Self {
            lexicon: SectorLexicon::default_lexicon(),
            model_name: Some(model_name.to_string()),
        }
    }

    /// Analyze a single text and return a [`SentimentScore`].
    #[instrument(skip(self))]
    pub fn analyze(&self, text: &str) -> Result<SentimentScore> {
        let (pos, neg, neu, word_count) = self.lexicon_score(text);
        let compound = (pos - neg).clamp(-1.0, 1.0);
        let score = if word_count == 0 {
            SentimentScore::neutral()
        } else {
            SentimentScore::new(
                pos.clamp(0.0, 1.0),
                neg.clamp(0.0, 1.0),
                neu.clamp(0.0, 1.0),
                compound,
            )
        };
        debug!(
            text_len = text.len(),
            ?score,
            "Sentiment analysis complete"
        );
        Ok(score)
    }

    /// Analyze a batch of texts.
    pub fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<SentimentScore>> {
        texts.iter().map(|t| self.analyze(t)).collect()
    }

    /// Analyze with sector weighting.  The `sector` parameter biases the
    /// lexicon toward a specific domain (e.g. `"earnings"`, `"crypto"`).
    pub fn analyze_with_sector(&self, text: &str, sector: &str) -> Result<SentimentScore> {
        let (mut pos, mut neg, neu, word_count) = self.lexicon_score(text);

        // Boost sector-specific terms
        let sector_lexicon: &HashMap<String, LexiconEntry> = match sector.to_lowercase().as_str() {
            "earnings" | "earnings_call" => &self.lexicon.earnings,
            "central_bank" | "fed" | "monetary" => &self.lexicon.central_bank,
            "crypto" | "defi" | "web3" => &self.lexicon.crypto,
            "social" | "reddit" | "twitter" => &self.lexicon.social,
            _ => &self.lexicon.general,
        };

        // Re-scan with sector boost
        let words: Vec<&str> = text.split_whitespace().collect();
        for window in words.windows(3) {
            let bigram = format!("{} {}", window[0], window[1]);
            let trigram = format!("{} {} {}", window[0], window[1], window[2]);
            for phrase in [&bigram, &trigram] {
                if let Some((p, n)) = sector_lexicon.get(phrase.as_str()) {
                    pos += p * 0.3;
                    neg += n * 0.3;
                }
            }
        }

        let compound = (pos - neg).clamp(-1.0, 1.0);
        let score = if word_count == 0 {
            SentimentScore::neutral()
        } else {
            SentimentScore::new(
                pos.clamp(0.0, 1.0),
                neg.clamp(0.0, 1.0),
                neu.clamp(0.0, 1.0),
                compound,
            )
        };
        Ok(score)
    }

    // -----------------------------------------------------------------------
    // Internal lexicon scoring
    // -----------------------------------------------------------------------

    /// Score text using the lexicon with negation handling.
    ///
    /// Returns `(positive_sum, negative_sum, neutral_proportion, word_count)`.
    fn lexicon_score(&self, text: &str) -> (f64, f64, f64, usize) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        if word_count == 0 {
            return (0.0, 0.0, 1.0, 0);
        }

        let mut pos_sum = 0.0_f64;
        let mut neg_sum = 0.0_f64;
        let mut scored = 0_usize;
        let mut negate_next = false;
        let mut intensifiers = 0_i32;

        // Negation words and intensifiers
        let negation_words = [
            "not", "no", "never", "neither", "nor", "n't", "cannot", "can't", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
            "weren't", "haven't", "hasn't", "hadn't",
        ];
        let intensifier_words = [
            "very", "extremely", "incredibly", "remarkably", "significantly", "substantially",
            "materially", "exceptionally", "particularly",
        ];

        for word in &words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();

            // Check for negation
            if negation_words.contains(&clean.as_str()) {
                negate_next = true;
                continue;
            }

            // Check for intensifier
            if intensifier_words.contains(&clean.as_str()) {
                intensifiers = (intensifiers + 1).min(3);
                continue;
            }

            // Single-word lookup
            let (p, n) = self.lexicon.lookup(&clean);

            // Bigram and trigram lookup (via the lexicon's sector dicts)
            // We skip bigram/trigram here for speed; analyze_with_sector handles them.

            if p > 0.0 || n > 0.0 {
                let multiplier = if intensifiers > 0 {
                    1.0 + 0.25 * intensifiers as f64
                } else {
                    1.0
                };

                if negate_next {
                    // Flip sentiment
                    pos_sum += n * multiplier;
                    neg_sum += p * multiplier;
                } else {
                    pos_sum += p * multiplier;
                    neg_sum += n * multiplier;
                }
                scored += 1;
            }

            // Reset state after applying
            negate_next = false;
            intensifiers = 0;
        }

        // Normalise
        let total_scored = scored as f64;
        let pos = if total_scored > 0.0 {
            (pos_sum / total_scored).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let neg = if total_scored > 0.0 {
            (neg_sum / total_scored).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let neu = (1.0 - pos - neg).max(0.0);

        (pos, neg, neu, word_count)
    }
}

impl Default for FinSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FedTone — Hawkish / Dovish / Neutral
// ---------------------------------------------------------------------------

/// Classification of central bank / Fed communication tone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FedTone {
    /// Indicating tighter monetary policy or concern about inflation.
    Hawkish,
    /// Indicating looser monetary policy or concern about economic slowdown.
    Dovish,
    /// Balanced or non-committal.
    Neutral,
}

impl std::fmt::Display for FedTone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FedTone::Hawkish => write!(f, "hawkish"),
            FedTone::Dovish => write!(f, "dovish"),
            FedTone::Neutral => write!(f, "neutral"),
        }
    }
}

/// Hawkish / dovish indicator phrases and their weights.
struct FedIndicator {
    phrase: &'static str,
    hawkish_weight: f64,
    dovish_weight: f64,
}

/// Returns the built-in Fed-speak indicator list.
fn fed_indicators() -> Vec<FedIndicator> {
    vec![
        FedIndicator { phrase: "rate hike", hawkish_weight: 0.8, dovish_weight: 0.0 },
        FedIndicator { phrase: "rate hikes", hawkish_weight: 0.8, dovish_weight: 0.0 },
        FedIndicator { phrase: "rate cut", hawkish_weight: 0.0, dovish_weight: 0.8 },
        FedIndicator { phrase: "rate cuts", hawkish_weight: 0.0, dovish_weight: 0.8 },
        FedIndicator { phrase: "higher for longer", hawkish_weight: 0.9, dovish_weight: 0.0 },
        FedIndicator { phrase: "tightening", hawkish_weight: 0.7, dovish_weight: 0.0 },
        FedIndicator { phrase: "easing", hawkish_weight: 0.0, dovish_weight: 0.7 },
        FedIndicator { phrase: "restrictive", hawkish_weight: 0.7, dovish_weight: 0.0 },
        FedIndicator { phrase: "accommodative", hawkish_weight: 0.0, dovish_weight: 0.7 },
        FedIndicator { phrase: "inflationary pressures", hawkish_weight: 0.7, dovish_weight: 0.0 },
        FedIndicator { phrase: "disinflation", hawkish_weight: 0.0, dovish_weight: 0.6 },
        FedIndicator { phrase: "soft landing", hawkish_weight: 0.0, dovish_weight: 0.6 },
        FedIndicator { phrase: "hard landing", hawkish_weight: 0.3, dovish_weight: 0.3 },
        FedIndicator { phrase: "recession risk", hawkish_weight: 0.0, dovish_weight: 0.5 },
        FedIndicator { phrase: "labor market tightness", hawkish_weight: 0.6, dovish_weight: 0.0 },
        FedIndicator { phrase: "wage growth", hawkish_weight: 0.5, dovish_weight: 0.0 },
        FedIndicator { phrase: "demand cooling", hawkish_weight: 0.0, dovish_weight: 0.4 },
        FedIndicator { phrase: "patient", hawkish_weight: 0.0, dovish_weight: 0.4 },
        FedIndicator { phrase: "data dependent", hawkish_weight: 0.0, dovish_weight: 0.3 },
        FedIndicator { phrase: "need to do more", hawkish_weight: 0.8, dovish_weight: 0.0 },
        FedIndicator { phrase: "additional tightening", hawkish_weight: 0.8, dovish_weight: 0.0 },
        FedIndicator { phrase: "quantitative tightening", hawkish_weight: 0.6, dovish_weight: 0.0 },
        FedIndicator { phrase: "quantitative easing", hawkish_weight: 0.0, dovish_weight: 0.6 },
        FedIndicator { phrase: "balance sheet reduction", hawkish_weight: 0.6, dovish_weight: 0.0 },
    ]
}

/// Decode a Fed / central bank text into a [`FedTone`] classification.
///
/// Returns the tone along with a confidence score and counts of matched
/// indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedToneResult {
    /// Classified tone.
    pub tone: FedTone,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Raw hawkish score.
    pub hawkish_score: f64,
    /// Raw dovish score.
    pub dovish_score: f64,
    /// Number of hawkish indicators matched.
    pub hawkish_matches: usize,
    /// Number of dovish indicators matched.
    pub dovish_matches: usize,
}

/// Analyze Fed-speak and classify as hawkish / dovish / neutral.
pub fn decode_fed_speak(text: &str) -> FedToneResult {
    let lower = text.to_lowercase();
    let indicators = fed_indicators();

    let mut hawkish_score = 0.0_f64;
    let mut dovish_score = 0.0_f64;
    let mut hawkish_matches = 0_usize;
    let mut dovish_matches = 0_usize;

    for indicator in &indicators {
        if lower.contains(indicator.phrase) {
            hawkish_score += indicator.hawkish_weight;
            dovish_score += indicator.dovish_weight;
            if indicator.hawkish_weight > 0.0 {
                hawkish_matches += 1;
            }
            if indicator.dovish_weight > 0.0 {
                dovish_matches += 1;
            }
        }
    }

    let total = hawkish_score + dovish_score;
    let (tone, confidence) = if total < 0.3 {
        (FedTone::Neutral, 0.5)
    } else {
        let hawkish_ratio = hawkish_score / total;
        let confidence = ((hawkish_ratio - 0.5).abs() * 2.0).clamp(0.3, 1.0);
        if hawkish_ratio > 0.55 {
            (FedTone::Hawkish, confidence)
        } else if hawkish_ratio < 0.45 {
            (FedTone::Dovish, confidence)
        } else {
            (FedTone::Neutral, confidence * 0.5)
        }
    };

    FedToneResult {
        tone,
        confidence,
        hawkish_score,
        dovish_score,
        hawkish_matches,
        dovish_matches,
    }
}

// ---------------------------------------------------------------------------
// Earnings call tone analysis
// ---------------------------------------------------------------------------

/// Tone analysis result for earnings call segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningsToneResult {
    /// Overall sentiment score.
    pub sentiment: SentimentScore,
    /// Forward-looking tone (optimistic / cautious / neutral).
    pub forward_tone: ForwardTone,
    /// Key bullish phrases found.
    pub bullish_phrases: Vec<String>,
    /// Key bearish phrases found.
    pub bearish_phrases: Vec<String>,
}

/// Forward-looking tone classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForwardTone {
    /// Optimistic about future performance.
    Optimistic,
    /// Cautious about future headwinds.
    Cautious,
    /// No clear directional signal.
    Neutral,
}

/// Analyze an earnings call text segment for tone.
pub fn analyze_earnings_tone(text: &str) -> EarningsToneResult {
    let analyzer = FinSentimentAnalyzer::with_model("finbert");
    let sentiment = analyzer.analyze_with_sector(text, "earnings").unwrap_or_default();

    let lower = text.to_lowercase();

    let bullish_keywords = [
        "confident", "excited", "growth trajectory", "strong momentum", "market opportunity",
        "innovation pipeline", "upside", "raise guidance", "expand", "accelerate",
        "record revenue", "record earnings", "beating expectations",
    ];
    let bearish_keywords = [
        "headwinds", "cautious", "uncertain", "macro challenges", "pressures", "softness",
        "delay", "lower guidance", "restructuring", "headcount reduction", "cost cutting",
        "supply chain", "margin pressure", "deterioration",
    ];

    let mut bullish_phrases: Vec<String> = Vec::new();
    let mut bearish_phrases: Vec<String> = Vec::new();

    for kw in &bullish_keywords {
        if lower.contains(kw) {
            bullish_phrases.push(kw.to_string());
        }
    }
    for kw in &bearish_keywords {
        if lower.contains(kw) {
            bearish_phrases.push(kw.to_string());
        }
    }

    let forward_tone = match (bullish_phrases.len(), bearish_phrases.len()) {
        (b, c) if b > c + 1 => ForwardTone::Optimistic,
        (b, c) if c > b + 1 => ForwardTone::Cautious,
        _ => ForwardTone::Neutral,
    };

    EarningsToneResult {
        sentiment,
        forward_tone,
        bullish_phrases,
        bearish_phrases,
    }
}

// ---------------------------------------------------------------------------
// SocialMediaSentimentAnalyzer
// ---------------------------------------------------------------------------

/// Sentiment analyzer tuned for social media financial discussion.
///
/// Handles Reddit (r/wallstreetbets, r/stocks) and Twitter / X posts with
/// specific preprocessing for slang, cashtags (`$AAPL`), emojis, and
/// all-caps emphasis.
#[derive(Debug)]
pub struct SocialMediaSentimentAnalyzer {
    inner: FinSentimentAnalyzer,
}

impl SocialMediaSentimentAnalyzer {
    /// Create a new social-media-aware sentiment analyzer.
    pub fn new() -> Self {
        Self {
            inner: FinSentimentAnalyzer::with_model("finbert"),
        }
    }

    /// Pre-process social media text before analysis.
    ///
    /// - Strips URLs
    /// - Converts cashtags (`$TICKER`) to uppercase words
    /// - Normalises repeated characters ("sooo" → "so")
    /// - Converts common emojis to text
    /// - Strips `@mentions`
    fn preprocess_social(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            // Strip URLs
            if c == 'h' && chars.peek() == Some(&'t') {
                // Check for http
                let rest: String = chars.by_ref().take(4).collect();
                if rest.starts_with("ttp") {
                    // Skip until whitespace
                    while let Some(&nc) = chars.peek() {
                        if nc.is_whitespace() { chars.next(); break; }
                        chars.next();
                    }
                    continue;
                } else {
                    out.push(c);
                    out.push_str(&rest);
                }
                continue;
            }

            // Strip @mentions
            if c == '@' {
                while let Some(&nc) = chars.peek() {
                    if nc.is_alphanumeric() || nc == '_' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push(' ');
                continue;
            }

            // Convert $TICKER cashtags — keep the ticker text
            if c == '$' {
                out.push(' ');
                out.push(c);
                out.push(' ');
                continue;
            }

            // Common emoji → text
            match c {
                '🚀' => { out.push_str(" moon "); }
                '📈' => { out.push_str(" bullish "); }
                '📉' => { out.push_str(" bearish "); }
                '💰' => { out.push_str(" money "); }
                '🔴' => { out.push_str(" negative "); }
                '🟢' => { out.push_str(" positive "); }
                '💎' => { out.push_str(" diamond hands "); }
                '🤲' => { out.push_str(" paper hands "); }
                '🧻' => { out.push_str(" paper hands "); }
                _ => out.push(c),
            }
        }

        // Normalise repeated characters (limit to 2 consecutive)
        let normalized = normalize_repeated_chars(&out);
        normalized
    }

    /// Analyze a social media post.
    pub fn analyze(&self, text: &str) -> Result<SentimentScore> {
        let cleaned = Self::preprocess_social(text);
        self.inner.analyze_with_sector(&cleaned, "social")
    }

    /// Analyze a batch of social media posts.
    pub fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<SentimentScore>> {
        texts.iter().map(|t| self.analyze(t)).collect()
    }
}

impl Default for SocialMediaSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Collapse runs of the same character to at most 2 repetitions.
fn normalize_repeated_chars(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if chars.peek() == Some(&c) {
            out.push(c);
            while chars.peek() == Some(&c) {
                chars.next();
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SentimentAggregation
// ---------------------------------------------------------------------------

/// Aggregate sentiment scores across a collection of documents or text
/// segments with configurable weighting strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAggregation {
    /// Number of documents aggregated.
    pub count: usize,
    /// Mean positive score.
    pub mean_positive: f64,
    /// Mean negative score.
    pub mean_negative: f64,
    /// Mean neutral score.
    pub mean_neutral: f64,
    /// Mean compound score.
    pub mean_compound: f64,
    /// Standard deviation of compound scores.
    pub std_compound: f64,
    /// Minimum compound score.
    pub min_compound: f64,
    /// Maximum compound score.
    pub max_compound: f64,
    /// Aggregated dominant label.
    pub dominant_label: String,
    /// Overall confidence (mean).
    pub mean_confidence: f64,
}

impl SentimentAggregation {
    /// Aggregate a slice of [`SentimentScore`] values.
    pub fn from_scores(scores: &[SentimentScore]) -> Self {
        if scores.is_empty() {
            return Self {
                count: 0,
                mean_positive: 0.0,
                mean_negative: 0.0,
                mean_neutral: 0.0,
                mean_compound: 0.0,
                std_compound: 0.0,
                min_compound: 0.0,
                max_compound: 0.0,
                dominant_label: "neutral".to_string(),
                mean_confidence: 0.0,
            };
        }

        let n = scores.len() as f64;

        let sum_positive: f64 = scores.iter().map(|s| s.positive).sum();
        let sum_negative: f64 = scores.iter().map(|s| s.negative).sum();
        let sum_neutral: f64 = scores.iter().map(|s| s.neutral).sum();
        let sum_compound: f64 = scores.iter().map(|s| s.compound).sum();
        let sum_confidence: f64 = scores.iter().map(|s| s.confidence).sum();

        let mean_positive = sum_positive / n;
        let mean_negative = sum_negative / n;
        let mean_neutral = sum_neutral / n;
        let mean_compound = sum_compound / n;
        let mean_confidence = sum_confidence / n;

        let variance: f64 = scores
            .iter()
            .map(|s| (s.compound - mean_compound).powi(2))
            .sum::<f64>()
            / n;
        let std_compound = variance.sqrt();

        let min_compound = scores.iter().map(|s| s.compound).fold(f64::INFINITY, f64::min);
        let max_compound = scores.iter().map(|s| s.compound).fold(f64::NEG_INFINITY, f64::max);

        // Dominant label from mean scores
        let dominant_label = if mean_positive >= mean_negative && mean_positive >= mean_neutral {
            "positive"
        } else if mean_negative >= mean_neutral {
            "negative"
        } else {
            "neutral"
        }
        .to_string();

        Self {
            count: scores.len(),
            mean_positive,
            mean_negative,
            mean_neutral,
            mean_compound,
            std_compound,
            min_compound,
            max_compound,
            dominant_label,
            mean_confidence,
        }
    }

    /// Weighted aggregation where each score has an associated weight (e.g. recency, source credibility).
    pub fn from_weighted_scores(scores: &[(SentimentScore, f64)]) -> Self {
        if scores.is_empty() {
            return Self::from_scores(&[]);
        }

        let total_weight: f64 = scores.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return Self::from_scores(&[]);
        }

        let weighted_scores: Vec<SentimentScore> = scores
            .iter()
            .map(|(s, w)| SentimentScore::with_confidence(
                s.positive * w / total_weight,
                s.negative * w / total_weight,
                s.neutral * w / total_weight,
                s.compound * w / total_weight,
                s.confidence,
            ))
            .collect();

        Self::from_scores(&weighted_scores)
    }
}

// ---------------------------------------------------------------------------
// Confidence Calibration
// ---------------------------------------------------------------------------

/// Calibrate confidence scores using temperature scaling.
///
/// Higher temperature → softer probabilities (less confident).
/// Lower temperature → sharper probabilities (more confident).
pub fn calibrate_confidence(score: &SentimentScore, temperature: f64) -> SentimentScore {
    let t = temperature.max(0.01);
    // Apply softmax-like temperature scaling to [positive, negative, neutral]
    let logits = [score.positive.ln(), score.negative.ln(), score.neutral.ln()];
    let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
    let max_logit = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = scaled.iter().map(|l| (l - max_logit).exp()).sum();
    let probs: Vec<f64> = scaled.iter().map(|l| (l - max_logit).exp() / exp_sum).collect();

    // Compound stays the same (it's a difference, not a probability)
    SentimentScore::with_confidence(
        probs[0],
        probs[1],
        probs[2],
        score.compound,
        score.confidence,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SentimentScore tests ------------------------------------------------

    #[test]
    fn test_sentiment_score_new() {
        let s = SentimentScore::new(0.7, 0.2, 0.1, 0.5);
        assert!((s.positive - 0.7).abs() < 1e-9);
        assert!((s.negative - 0.2).abs() < 1e-9);
        assert!(s.confidence > 0.0);
    }

    #[test]
    fn test_sentiment_score_dominant_label() {
        let pos = SentimentScore::new(0.8, 0.1, 0.1, 0.7);
        assert_eq!(pos.dominant_label(), "positive");

        let neg = SentimentScore::new(0.1, 0.8, 0.1, -0.7);
        assert_eq!(neg.dominant_label(), "negative");

        let neu = SentimentScore::new(0.1, 0.1, 0.8, 0.0);
        assert_eq!(neu.dominant_label(), "neutral");
    }

    #[test]
    fn test_sentiment_score_predefined() {
        let n = SentimentScore::neutral();
        assert_eq!(n.dominant_label(), "neutral");

        let p = SentimentScore::strongly_positive();
        assert_eq!(p.dominant_label(), "positive");

        let n = SentimentScore::strongly_negative();
        assert_eq!(n.dominant_label(), "negative");
    }

    #[test]
    #[should_panic]
    fn test_sentiment_score_out_of_range() {
        SentimentScore::new(1.5, 0.0, 0.0, 0.0);
    }

    #[test]
    fn test_sentiment_score_serialization() {
        let s = SentimentScore::new(0.6, 0.3, 0.1, 0.3);
        let json = serde_json::to_string(&s).unwrap();
        let deserialized: SentimentScore = serde_json::from_str(&json).unwrap();
        assert!((deserialized.positive - 0.6).abs() < 1e-9);
    }

    // -- Lexicon tests -------------------------------------------------------

    #[test]
    fn test_lexicon_lookup_positive() {
        let lex = SectorLexicon::default_lexicon();
        let (p, n) = lex.lookup("bullish");
        assert!(p > 0.5);
        assert!(n < 0.01);
    }

    #[test]
    fn test_lexicon_lookup_negative() {
        let lex = SectorLexicon::default_lexicon();
        let (p, n) = lex.lookup("crash");
        assert!(n > 0.5);
        assert!(p < 0.01);
    }

    #[test]
    fn test_lexicon_lookup_unknown() {
        let lex = SectorLexicon::default_lexicon();
        let (p, n) = lex.lookup("xylophone");
        assert!((p - 0.0).abs() < 1e-9);
        assert!((n - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lexicon_add_entry() {
        let mut lex = SectorLexicon::default_lexicon();
        lex.add_entry("moonscape", 0.9, 0.0);
        let (p, _) = lex.lookup("moonscape");
        assert!(p > 0.8);
    }

    // -- FinSentimentAnalyzer tests ------------------------------------------

    #[test]
    fn test_analyze_positive_text() {
        let analyzer = FinSentimentAnalyzer::new();
        let score = analyzer.analyze("The company reported strong growth and record profits.").unwrap();
        assert!(score.positive > 0.3);
        assert_eq!(score.dominant_label(), "positive");
    }

    #[test]
    fn test_analyze_negative_text() {
        let analyzer = FinSentimentAnalyzer::new();
        let score = analyzer.analyze("The stock crashed after bankruptcy fears intensified.").unwrap();
        assert!(score.negative > 0.3);
        assert_eq!(score.dominant_label(), "negative");
    }

    #[test]
    fn test_analyze_neutral_text() {
        let analyzer = FinSentimentAnalyzer::new();
        let score = analyzer.analyze("The meeting is scheduled for Tuesday at noon.").unwrap();
        assert!(score.neutral > 0.5);
        assert_eq!(score.dominant_label(), "neutral");
    }

    #[test]
    fn test_analyze_negation() {
        let analyzer = FinSentimentAnalyzer::new();
        let positive = analyzer.analyze("Stocks surged on strong earnings.").unwrap();
        let negated = analyzer.analyze("Stocks did not surge on weak earnings.").unwrap();
        // The negated version should have lower positive than the original
        assert!(negated.positive <= positive.positive);
    }

    #[test]
    fn test_analyze_batch() {
        let analyzer = FinSentimentAnalyzer::new();
        let texts = vec![
            "Record profits this quarter!",
            "The company faces bankruptcy.",
            "Meeting at 3 PM tomorrow.",
        ];
        let scores = analyzer.analyze_batch(&texts).unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].dominant_label(), "positive");
        assert_eq!(scores[1].dominant_label(), "negative");
        assert_eq!(scores[2].dominant_label(), "neutral");
    }

    #[test]
    fn test_analyze_with_sector_crypto() {
        let analyzer = FinSentimentAnalyzer::new();
        let score = analyzer.analyze_with_sector("BTC had a massive bull run after the halving", "crypto").unwrap();
        assert!(score.positive > 0.2);
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = FinSentimentAnalyzer::new();
        let score = analyzer.analyze("").unwrap();
        assert_eq!(score.dominant_label(), "neutral");
    }

    // -- Fed speak tests -----------------------------------------------------

    #[test]
    fn test_fed_speak_hawkish() {
        let result = decode_fed_speak(
            "We may need additional rate hikes to combat inflationary pressures. \
             The labor market remains tight and we are committed to bringing \
             inflation back to target.",
        );
        assert_eq!(result.tone, FedTone::Hawkish);
        assert!(result.hawkish_score > result.dovish_score);
    }

    #[test]
    fn test_fed_speak_dovish() {
        let result = decode_fed_speak(
            "We see disinflationary trends and may consider rate cuts if the \
             data supports it. We remain patient and data dependent.",
        );
        assert_eq!(result.tone, FedTone::Dovish);
        assert!(result.dovish_score > result.hawkish_score);
    }

    #[test]
    fn test_fed_speak_neutral() {
        let result = decode_fed_speak("The committee reviewed the economic data.");
        assert_eq!(result.tone, FedTone::Neutral);
    }

    #[test]
    fn test_fed_tone_display() {
        assert_eq!(FedTone::Hawkish.to_string(), "hawkish");
        assert_eq!(FedTone::Dovish.to_string(), "dovish");
        assert_eq!(FedTone::Neutral.to_string(), "neutral");
    }

    #[test]
    fn test_fed_tone_serialization() {
        let result = FedToneResult {
            tone: FedTone::Hawkish,
            confidence: 0.85,
            hawkish_score: 2.4,
            dovish_score: 0.3,
            hawkish_matches: 3,
            dovish_matches: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("hawkish"));
        let _: FedToneResult = serde_json::from_str(&json).unwrap();
    }

    // -- Earnings tone tests -------------------------------------------------

    #[test]
    fn test_earnings_tone_optimistic() {
        let result = analyze_earnings_tone(
            "We are confident about our growth trajectory and strong momentum. \
             We are raising guidance for the full year driven by record revenue.",
        );
        assert_eq!(result.forward_tone, ForwardTone::Optimistic);
        assert!(!result.bullish_phrases.is_empty());
    }

    #[test]
    fn test_earnings_tone_cautious() {
        let result = analyze_earnings_tone(
            "We see macro headwinds and supply chain pressures. We are implementing \
             cost cutting measures and a headcount reduction.",
        );
        assert_eq!(result.forward_tone, ForwardTone::Cautious);
        assert!(!result.bearish_phrases.is_empty());
    }

    #[test]
    fn test_earnings_tone_neutral() {
        let result = analyze_earnings_tone("Revenue was in line with expectations.");
        assert_eq!(result.forward_tone, ForwardTone::Neutral);
    }

    // -- Social media tests --------------------------------------------------

    #[test]
    fn test_social_preprocess_urls() {
        let out = SocialMediaSentimentAnalyzer::preprocess_social(
            "Check this out https://example.com/stock Amazing!"
        );
        assert!(!out.contains("https://"));
    }

    #[test]
    fn test_social_preprocess_mentions() {
        let out = SocialMediaSentimentAnalyzer::preprocess_social("@user123 is bullish on AAPL");
        assert!(!out.contains("@user123"));
        assert!(out.contains("bullish"));
    }

    #[test]
    fn test_social_preprocess_emoji() {
        let out = SocialMediaSentimentAnalyzer::preprocess_social("BTC 🚀🚀🚀 to the moon");
        assert!(out.contains("moon"));
    }

    #[test]
    fn test_social_preprocess_cashtag() {
        let out = SocialMediaSentimentAnalyzer::preprocess_social("$AAPL earnings beat");
        assert!(out.contains("$ AAPL"));
    }

    #[test]
    fn test_normalize_repeated_chars() {
        let out = normalize_repeated_chars("sooooo good");
        assert_eq!(out, "soo good");
    }

    #[test]
    fn test_social_analyzer() {
        let analyzer = SocialMediaSentimentAnalyzer::new();
        let score = analyzer.analyze("🚀 AAPL to the moon diamond hands stonks 📈").unwrap();
        assert!(score.positive > 0.0);
    }

    // -- Aggregation tests ---------------------------------------------------

    #[test]
    fn test_aggregation_from_scores() {
        let scores = vec![
            SentimentScore::new(0.8, 0.1, 0.1, 0.7),
            SentimentScore::new(0.6, 0.2, 0.2, 0.4),
            SentimentScore::new(0.9, 0.05, 0.05, 0.85),
        ];
        let agg = SentimentAggregation::from_scores(&scores);
        assert_eq!(agg.count, 3);
        assert!(agg.mean_positive > 0.7);
        assert!(agg.mean_compound > 0.5);
        assert_eq!(agg.dominant_label, "positive");
        assert!(agg.min_compound <= agg.max_compound);
    }

    #[test]
    fn test_aggregation_empty() {
        let agg = SentimentAggregation::from_scores(&[]);
        assert_eq!(agg.count, 0);
        assert_eq!(agg.dominant_label, "neutral");
    }

    #[test]
    fn test_weighted_aggregation() {
        let scores = vec![
            (SentimentScore::new(0.9, 0.05, 0.05, 0.85), 3.0),
            (SentimentScore::new(0.1, 0.8, 0.1, -0.7), 1.0),
        ];
        let agg = SentimentAggregation::from_weighted_scores(&scores);
        assert_eq!(agg.count, 2);
        assert_eq!(agg.dominant_label, "positive"); // higher weight on positive
    }

    #[test]
    fn test_aggregation_serialization() {
        let scores = vec![SentimentScore::new(0.5, 0.3, 0.2, 0.2)];
        let agg = SentimentAggregation::from_scores(&scores);
        let json = serde_json::to_string(&agg).unwrap();
        assert!(json.contains("\"count\":1"));
        let _: SentimentAggregation = serde_json::from_str(&json).unwrap();
    }

    // -- Calibration tests ---------------------------------------------------

    #[test]
    fn test_calibrate_confidence() {
        let score = SentimentScore::new(0.7, 0.2, 0.1, 0.5);
        let calibrated = calibrate_confidence(&score, 2.0);
        // Higher temperature should push probabilities toward uniform
        assert!(calibrated.positive < score.positive);
        assert!(calibrated.negative > score.negative);
    }

    #[test]
    fn test_calibrate_low_temperature() {
        let score = SentimentScore::new(0.7, 0.2, 0.1, 0.5);
        let calibrated = calibrate_confidence(&score, 0.1);
        // Lower temperature should sharpen the dominant class
        assert!(calibrated.positive > score.positive);
    }

    #[test]
    fn test_calibrate_identity() {
        let score = SentimentScore::new(0.7, 0.2, 0.1, 0.5);
        let calibrated = calibrate_confidence(&score, 1.0);
        // Temperature of 1.0 should roughly preserve (softmax of logits/t ≈ softmax of logits when t=1)
        assert!((calibrated.positive - score.positive).abs() < 0.01);
    }
}
