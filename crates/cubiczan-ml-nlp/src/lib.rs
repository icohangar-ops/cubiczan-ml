//! # cubiczan-ml-nlp
//!
//! Financial NLP module for the Cubiczan ML ecosystem.
//!
//! Provides a comprehensive suite of natural language processing tools
//! specifically designed for financial text analysis, including:
//!
//! - **Tokenization**: Fast, financial-aware tokenization wrapping HuggingFace tokenizers
//! - **Sentiment Analysis**: Financial sentiment with sector-specific lexicons and Fed-speak decoding
//! - **Text Classification**: FinBERT, zero-shot, and multi-label classification pipelines
//! - **Named Entity Recognition**: Financial NER for companies, currencies, amounts, SEC entities
//! - **Embeddings**: Sentence embeddings with caching and cosine similarity
//! - **Summarization**: Extractive and abstractive summarization for earnings calls and SEC filings
//!
//! ## Example
//!
//! ```ignore
//! use cubiczan_ml_nlp::{tokenizer::FinTokenizer, sentiment::FinSentimentAnalyzer};
//!
//! let tokenizer = FinTokenizer::from_pretrained("bert-base-uncased")?;
//! let encoded = tokenizer.encode("AAPL reported $3.2B revenue.")?;
//!
//! let analyzer = FinSentimentAnalyzer::new()?;
//! let score = analyzer.analyze("The company beat earnings expectations.")?;
//! println!("Sentiment: {:?}", score);
//! ```

pub mod classifier;
pub mod embeddings;
pub mod ner;
pub mod sentiment;
pub mod summarizer;
pub mod tokenizer;

// Re-exports of the most commonly used types for ergonomic imports.
pub use classifier::{
    ClassificationPipeline, ClassificationResult, FinBertClassifier, MultiLabelClassifier,
    TextClassifier, ZeroShotClassifier,
};
pub use embeddings::{BatchEmbedder, EmbeddingCache, EmbeddingModel, TextEmbedder};
pub use ner::{FinancialNER, NEREntity, SECEntityExtractor};
pub use sentiment::{
    FedTone, FinSentimentAnalyzer, SentimentAggregation, SentimentScore,
    SocialMediaSentimentAnalyzer,
};
pub use summarizer::{
    AbstractiveSummarizer, EarningsCallSummarizer, ExtractiveSummarizer, KeyPhraseExtractor,
    SECFilingSummarizer,
};
pub use tokenizer::{FinTokenizer, TokenizerConfig, TokenizerModel, TruncationStrategy};
