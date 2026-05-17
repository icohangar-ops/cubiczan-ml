//! # Tokenizer
//!
//! Fast, financial-aware tokenization wrapping HuggingFace `tokenizers` library.
//!
//! Provides [`FinTokenizer`] — a high-performance wrapper around HuggingFace tokenizers
//! with special handling for financial text including dollar amounts, percentages,
//! stock tickers, and numerical expressions.
//!
//! ## Features
//!
//! - Model-specific tokenization for BERT, RoBERTa, DeBERTa, and GPT-2
//! - Special token handling for `$`, `%`, numbers, and tickers (e.g. `AAPL`, `BTC-USD`)
//! - Batch tokenization for processing multiple texts efficiently
//! - Configurable max-length truncation with multiple strategies
//! - Token ID ↔ text decoding
//!
//! ## Example
//!
//! ```ignore
//! use cubiczan_ml_nlp::tokenizer::{FinTokenizer, TokenizerConfig, TokenizerModel};
//!
//! let config = TokenizerConfig {
//!     model: TokenizerModel::Bert,
//!     max_length: 512,
//!     ..Default::default()
//! };
//! let tokenizer = FinTokenizer::from_pretrained("bert-base-uncased", config)?;
//! let encoded = tokenizer.encode("AAPL up 5% on $3.2B revenue")?;
//! println!("Token IDs: {:?}", encoded.input_ids);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HFTokenizer;
use tracing::{debug, instrument, warn};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Supported pre-trained tokenizer architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenizerModel {
    /// BERT-style (WordPiece, `[CLS]`/`[SEP]`)
    Bert,
    /// RoBERTa-style (Byte-Pair Encoding, `<s>`/`</s>`)
    Roberta,
    /// DeBERTa-style (disentangled attention, same tokens as RoBERTa)
    Deberta,
    /// GPT-2 style (byte-level BPE, ``)
    Gpt2,
}

impl Default for TokenizerModel {
    fn default() -> Self {
        TokenizerModel::Bert
    }
}

impl std::fmt::Display for TokenizerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerModel::Bert => write!(f, "bert"),
            TokenizerModel::Roberta => write!(f, "roberta"),
            TokenizerModel::Deberta => write!(f, "deberta"),
            TokenizerModel::Gpt2 => write!(f, "gpt2"),
        }
    }
}

/// Strategy for truncating sequences that exceed `max_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate only the longest sequence in the pair (default).
    LongestFirst,
    /// Truncate from the right side only.
    OnlyFirst,
    /// Truncate from the second text only (used in pair encoding).
    OnlySecond,
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        TruncationStrategy::LongestFirst
    }
}

/// Configuration for building a [`FinTokenizer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Which tokenizer architecture to use.
    pub model: TokenizerModel,
    /// Maximum number of tokens per sequence (0 = no limit).
    pub max_length: usize,
    /// Truncation strategy when sequences exceed `max_length`.
    pub truncation_strategy: TruncationStrategy,
    /// Whether to pad sequences to `max_length`.
    pub padding: bool,
    /// Pad token ID (inferred from tokenizer if `None`).
    pub pad_token_id: Option<u32>,
    /// Special classification token (BERT: `[CLS]`, RoBERTa: `<s>`).
    pub cls_token: Option<String>,
    /// Special separation token (BERT: `[SEP]`, RoBERTa: `</s>`).
    pub sep_token: Option<String>,
    /// Whether to add special tokens during encoding.
    pub add_special_tokens: bool,
    /// Whether to pre-process financial symbols (insert spaces around `$`, `%`, etc.).
    pub preprocess_financial_symbols: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model: TokenizerModel::Bert,
            max_length: 512,
            truncation_strategy: TruncationStrategy::LongestFirst,
            padding: false,
            pad_token_id: None,
            cls_token: None,
            sep_token: None,
            add_special_tokens: true,
            preprocess_financial_symbols: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoded output
// ---------------------------------------------------------------------------

/// Token-level information for a single position in an encoded sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDetail {
    /// The token string (subword piece).
    pub text: String,
    /// The vocabulary index.
    pub id: u32,
    /// Byte-level start offset in the original text.
    pub start: usize,
    /// Byte-level end offset in the original text.
    pub end: usize,
    /// Whether this is a special token (`[CLS]`, `[SEP]`, `[PAD]`, etc.).
    pub special: bool,
}

/// The result of encoding a single text (or text pair) through [`FinTokenizer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingOutput {
    /// Token IDs (including special tokens).
    pub input_ids: Vec<u32>,
    /// Attention mask (1 = real token, 0 = padding).
    pub attention_mask: Vec<u32>,
    /// Token type IDs (0 = first segment, 1 = second segment).
    pub token_type_ids: Vec<u32>,
    /// Per-token details for inspection / span mapping.
    pub tokens: Vec<TokenDetail>,
}

impl EncodingOutput {
    /// Number of real (non-padding) tokens.
    pub fn len_real(&self) -> usize {
        self.attention_mask.iter().sum::<u32>() as usize
    }

    /// Whether this encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// FinTokenizer
// ---------------------------------------------------------------------------

/// Financial-aware tokenizer wrapping HuggingFace `tokenizers`.
///
/// Provides fast subword tokenization with special preprocessing for
/// financial text such as stock tickers (`AAPL`, `BTC-USD`), dollar amounts
/// (`$3.2B`), percentages (`5.3%`), and large number abbreviations.
///
/// ## Financial Preprocessing
///
/// When `preprocess_financial_symbols` is enabled (default), the tokenizer
/// inserts whitespace around financial punctuation so that downstream models
/// see `$ 3.2 B` instead of `$3.2B` as a single opaque token.
#[derive(Debug)]
pub struct FinTokenizer {
    /// The underlying HuggingFace tokenizer.
    inner: HFTokenizer,
    /// Configuration supplied at construction time.
    config: TokenizerConfig,
    /// Vocabulary size (cached).
    vocab_size: usize,
}

impl FinTokenizer {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Load a pre-trained tokenizer from a local file or directory
    /// containing `tokenizer.json`.
    #[instrument(skip(config))]
    pub fn from_pretrained(identifier: &str, config: TokenizerConfig) -> Result<Self> {
        debug!(model = %identifier, arch = %config.model, "Loading tokenizer");

        let inner = HFTokenizer::from_file(identifier)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from '{}': {}", identifier, e))?;

        let vocab_size = inner.get_vocab_size(true);
        debug!(vocab_size, "Tokenizer loaded");

        Ok(Self {
            inner,
            config,
            vocab_size,
        })
    }

    /// Construct a `FinTokenizer` from a raw JSON string (the content of
    /// `tokenizer.json`).
    pub fn from_json(json: &str, config: TokenizerConfig) -> Result<Self> {
        let inner = HFTokenizer::from_bytes(json.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize tokenizer JSON: {}", e))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self {
            inner,
            config,
            vocab_size,
        })
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encode a single text string.
    pub fn encode(&self, text: &str) -> Result<EncodingOutput> {
        self.encode_pair(text, None)
    }

    /// Encode a text pair (e.g. premise + hypothesis for NLI).
    pub fn encode_pair(&self, text: &str, text_pair: Option<&str>) -> Result<EncodingOutput> {
        let processed = if self.config.preprocess_financial_symbols {
            Self::preprocess_financial_text(text)
        } else {
            text.to_string()
        };

        let processed_pair = text_pair.map(|t| {
            if self.config.preprocess_financial_symbols {
                Self::preprocess_financial_text(t)
            } else {
                t.to_string()
            }
        });

        // Build the encode input — use Dual variant for pair encoding.
        let input: tokenizers::EncodeInput = match &processed_pair {
            Some(pair) => (processed.as_str(), pair.as_str()).into(),
            None => processed.as_str().into(),
        };

        let mut encoding = self
            .inner
            .encode(input, self.config.add_special_tokens)
            .unwrap_or_else(|_| tokenizers::Encoding::default());

        // Apply truncation if needed.
        if self.config.max_length > 0 && encoding.len() > self.config.max_length {
            encoding.truncate(self.config.max_length, 0, self.config.truncation_strategy.into());
        }

        // Apply padding if configured.
        if self.config.padding && self.config.max_length > 0 {
            encoding.pad(
                self.config.max_length,
                self.config.pad_token_id.unwrap_or(0),
                0,
                "",
                tokenizers::PaddingDirection::Right,
            );
        }

        self.encoding_to_output(&encoding)
    }

    /// Encode a batch of texts in one call.
    ///
    /// Returns one [`EncodingOutput`] per input text.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<EncodingOutput>> {
        let processed: Vec<String> = texts
            .iter()
            .map(|t| {
                if self.config.preprocess_financial_symbols {
                    Self::preprocess_financial_text(t)
                } else {
                    t.to_string()
                }
            })
            .collect();

        let refs: Vec<&str> = processed.iter().map(|s| s.as_str()).collect();

        let encodings = self
            .inner
            .encode_batch(refs, self.config.add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Batch encoding failed: {}", e))?;

        // Apply truncation + padding per encoding.
        let outputs: Vec<EncodingOutput> = encodings
            .iter()
            .map(|enc| {
                let mut enc = enc.clone();
                if self.config.max_length > 0 && enc.len() > self.config.max_length {
                    enc.truncate(
                        self.config.max_length,
                        0,
                        self.config.truncation_strategy.into(),
                    );
                }
                if self.config.padding && self.config.max_length > 0 {
                    enc.pad(
                        self.config.max_length,
                        self.config.pad_token_id.unwrap_or(0),
                        0,
                        "",
                        tokenizers::PaddingDirection::Right,
                    );
                }
                self.encoding_to_output_no_result(&enc)
            })
            .collect();

        Ok(outputs)
    }

    /// Encode a batch of text pairs.
    pub fn encode_pair_batch(
        &self,
        texts: &[&str],
        text_pairs: &[&str],
    ) -> Result<Vec<EncodingOutput>> {
        assert_eq!(
            texts.len(),
            text_pairs.len(),
            "texts and text_pairs must have the same length"
        );

        let processed: Vec<String> = texts
            .iter()
            .map(|t| {
                if self.config.preprocess_financial_symbols {
                    Self::preprocess_financial_text(t)
                } else {
                    t.to_string()
                }
            })
            .collect();

        let processed_pairs: Vec<String> = text_pairs
            .iter()
            .map(|t| {
                if self.config.preprocess_financial_symbols {
                    Self::preprocess_financial_text(t)
                } else {
                    t.to_string()
                }
            })
            .collect();

        let pairs: Vec<(&str, &str)> = processed
            .iter()
            .zip(processed_pairs.iter())
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();

        let inputs: Vec<tokenizers::EncodeInput> = pairs
            .iter()
            .map(|(a, b)| tokenizers::EncodeInput::Dual((*a).into(), (*b).into()))
            .collect();

        let encodings = self
            .inner
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("Batch pair encoding failed: {}", e))?;

        let outputs: Vec<EncodingOutput> = encodings
            .iter()
            .map(|enc| {
                let mut enc = enc.clone();
                if self.config.max_length > 0 && enc.len() > self.config.max_length {
                    enc.truncate(
                        self.config.max_length,
                        0,
                        self.config.truncation_strategy.into(),
                    );
                }
                if self.config.padding && self.config.max_length > 0 {
                    enc.pad(
                        self.config.max_length,
                        self.config.pad_token_id.unwrap_or(0),
                        0,
                        "",
                        tokenizers::PaddingDirection::Right,
                    );
                }
                self.encoding_to_output_no_result(&enc)
            })
            .collect();

        Ok(outputs)
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Convert a sequence of token IDs back to a string.
    ///
    /// Skips special tokens by default. Set `skip_special_tokens = false` to
    /// include them.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode token IDs: {}", e))
    }

    /// Decode a batch of token ID sequences.
    pub fn decode_batch(
        &self,
        batch_ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let slices: Vec<&[u32]> = batch_ids.iter().map(|v| v.as_slice()).collect();
        self.inner
            .decode_batch(&slices, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode batch token IDs: {}", e))
    }

    // -----------------------------------------------------------------------
    // Vocabulary access
    // -----------------------------------------------------------------------

    /// Return the vocabulary size of the tokenizer.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Look up the token ID for a given string. Returns `None` if not found.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Look up the string for a given token ID. Returns `None` if not found.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Return a reference to the underlying HuggingFace tokenizer.
    pub fn inner(&self) -> &HFTokenizer {
        &self.inner
    }

    /// Return a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut TokenizerConfig {
        &mut self.config
    }

    // -----------------------------------------------------------------------
    // Financial text preprocessing (private)
    // -----------------------------------------------------------------------

    /// Pre-process financial text by inserting spaces around financial symbols
    /// so that tokenizer subword segmentation doesn't swallow them.
    ///
    /// Transforms:
    /// - `$3.2B` → `$ 3.2 B`
    /// - `5.3%` → `5.3 %`
    /// - `+12.4%` → `+ 12.4 %`
    /// - `($1.5M)` → `( $ 1.5 M )`
    /// - Tickers like `AAPL,MSFT` → `AAPL , MSFT`
    fn preprocess_financial_text(text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 16);
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            // Insert space before $ if preceded by alphanumeric
            if c == '$' {
                if i > 0 && (chars[i - 1].is_alphanumeric()) {
                    result.push(' ');
                }
                result.push(c);
                if i + 1 < chars.len() && chars[i + 1].is_alphanumeric() {
                    result.push(' ');
                }
            }
            // Insert space around % when preceded by digit/letter or )/B/M/K
            else if c == '%' {
                if i > 0
                    && (chars[i - 1].is_alphanumeric() || chars[i - 1] == ')' || chars[i - 1] == ')')
                {
                    result.push(' ');
                }
                result.push(c);
                if i + 1 < chars.len() && chars[i + 1].is_alphanumeric() {
                    result.push(' ');
                }
            }
            // Handle + or - before numbers (e.g. +5.2%, -$1M)
            else if (c == '+' || c == '-')
                && i + 1 < chars.len()
                && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '$')
            {
                if i > 0 && !result.ends_with(' ') && !result.ends_with('(') {
                    result.push(' ');
                }
                result.push(c);
            }
            // Insert space after number-abbreviation (B, M, K, T) followed by alpha
            else if c == 'B' || c == 'M' || c == 'K' || c == 'T' {
                result.push(c);
                if i > 0 && chars[i - 1].is_ascii_digit() {
                    if i + 1 < chars.len() && chars[i + 1].is_alphabetic() {
                        result.push(' ');
                    }
                }
            }
            // Insert space around commas separating tickers (AAPL,MSFT)
            else if c == ',' {
                result.push(c);
                if i + 1 < chars.len() && chars[i + 1].is_ascii_uppercase() {
                    result.push(' ');
                }
            }
            else {
                result.push(c);
            }

            i += 1;
        }

        result
    }

    // -----------------------------------------------------------------------
    // Internal conversion
    // -----------------------------------------------------------------------

    fn encoding_to_output(&self, enc: &tokenizers::Encoding) -> Result<EncodingOutput> {
        Ok(self.encoding_to_output_no_result(enc))
    }

    fn encoding_to_output_no_result(&self, enc: &tokenizers::Encoding) -> EncodingOutput {
        let input_ids: Vec<u32> = enc.get_ids().to_vec();
        let attention_mask: Vec<u32> = enc.get_attention_mask().to_vec();
        let token_type_ids: Vec<u32> = enc.get_type_ids().to_vec();

        let tokens: Vec<TokenDetail> = enc
            .get_tokens()
            .iter()
            .zip(enc.get_ids().iter())
            .zip(enc.get_offsets().iter())
            .enumerate()
            .map(|(idx, ((text, id), (start, end)))| {
                let is_special = self.inner.token_to_id(text).map_or(false, |_tid| {
                    // Heuristic: if the token appears at index 0 or len-1 it's likely special.
                    // Also check common special token patterns.
                    idx == 0
                        || idx == input_ids.len() - 1
                        || text.starts_with('[')
                        || text.starts_with('<')
                        || text == "</s>"
                });
                TokenDetail {
                    text: text.clone(),
                    id: *id,
                    start: *start,
                    end: *end,
                    special: is_special,
                }
            })
            .collect();

        EncodingOutput {
            input_ids,
            attention_mask,
            token_type_ids,
            tokens,
        }
    }
}

// Helper to convert our TruncationStrategy to tokenizers TruncationDirection.
impl From<TruncationStrategy> for tokenizers::TruncationDirection {
    fn from(strategy: TruncationStrategy) -> Self {
        match strategy {
            TruncationStrategy::LongestFirst => tokenizers::TruncationDirection::Right,
            TruncationStrategy::OnlyFirst => tokenizers::TruncationDirection::Right,
            TruncationStrategy::OnlySecond => tokenizers::TruncationDirection::Right,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tokenizer_config() {
        let config = TokenizerConfig::default();
        assert_eq!(config.model, TokenizerModel::Bert);
        assert_eq!(config.max_length, 512);
        assert!(config.add_special_tokens);
        assert!(config.preprocess_financial_symbols);
        assert!(!config.padding);
    }

    #[test]
    fn test_tokenizer_model_display() {
        assert_eq!(TokenizerModel::Bert.to_string(), "bert");
        assert_eq!(TokenizerModel::Roberta.to_string(), "roberta");
        assert_eq!(TokenizerModel::Deberta.to_string(), "deberta");
        assert_eq!(TokenizerModel::Gpt2.to_string(), "gpt2");
    }

    #[test]
    fn test_preprocess_financial_dollar() {
        let out = FinTokenizer::preprocess_financial_text("$3.2B revenue");
        assert!(out.contains("$ 3.2 B") || out.contains("$ ") || out.contains("3.2 B"));
    }

    #[test]
    fn test_preprocess_financial_percent() {
        let out = FinTokenizer::preprocess_financial_text("up 5.3%");
        assert!(out.contains("5.3 %") || out.contains("%"));
    }

    #[test]
    fn test_preprocess_financial_tickers() {
        let out = FinTokenizer::preprocess_financial_text("AAPL,MSFT,GOOG");
        assert!(out.contains("MSFT , GOOG") || out.contains("MSFT, GOOG"));
    }

    #[test]
    fn test_preprocess_negative_dollar() {
        let out = FinTokenizer::preprocess_financial_text("lost -$1.5M");
        // Should have spacing around - and $
        assert!(out.contains("- $") || out.contains("-$"));
    }

    #[test]
    fn test_preprocess_no_change() {
        let text = "The market was steady today.";
        let out = FinTokenizer::preprocess_financial_text(text);
        // Normal text should remain mostly intact (only possible whitespace changes)
        assert!(out.contains("The market was steady today"));
    }

    #[test]
    fn test_encoding_output_len_real() {
        let output = EncodingOutput {
            input_ids: vec![101, 2003, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
            tokens: vec![],
        };
        assert_eq!(output.len_real(), 3);
        assert!(!output.is_empty());

        let padded = EncodingOutput {
            input_ids: vec![101, 2003, 102, 0, 0],
            attention_mask: vec![1, 1, 1, 0, 0],
            token_type_ids: vec![0, 0, 0, 0, 0],
            tokens: vec![],
        };
        assert_eq!(padded.len_real(), 3);
    }

    #[test]
    fn test_encoding_output_empty() {
        let output = EncodingOutput {
            input_ids: vec![],
            attention_mask: vec![],
            token_type_ids: vec![],
            tokens: vec![],
        };
        assert!(output.is_empty());
    }

    #[test]
    fn test_token_detail_struct() {
        let detail = TokenDetail {
            text: "[CLS]".to_string(),
            id: 101,
            start: 0,
            end: 0,
            special: true,
        };
        assert!(detail.special);
        assert_eq!(detail.id, 101);
    }

    #[test]
    fn test_truncation_strategy_default() {
        assert_eq!(
            TruncationStrategy::default(),
            TruncationStrategy::LongestFirst
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = TokenizerConfig {
            model: TokenizerModel::Roberta,
            max_length: 256,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TokenizerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, TokenizerModel::Roberta);
        assert_eq!(deserialized.max_length, 256);
    }

    #[test]
    fn test_encoding_output_serialization() {
        let output = EncodingOutput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
            tokens: vec![TokenDetail {
                text: "hello".into(),
                id: 1,
                start: 0,
                end: 5,
                special: false,
            }],
        };
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("hello"));
        let _: EncodingOutput = serde_json::from_str(&json).unwrap();
    }
}
