//! # Text Embeddings
//!
//! Sentence embedding generation, batch processing, similarity search, and caching.

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Supported embedding models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingModel {
    AllMiniLmL6V2,
    AllMiniLmL12V2,
    BERTBaseUncased,
    RoBERTaBase,
    FinBERT,
    Custom { name: String },
}

impl EmbeddingModel {
    /// Get model identifier for HuggingFace hub.
    pub fn model_id(&self) -> &str {
        match self {
            EmbeddingModel::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingModel::AllMiniLmL12V2 => "sentence-transformers/all-MiniLM-L12-v2",
            EmbeddingModel::BERTBaseUncased => "sentence-transformers/bert-base-nli-mean-tokens",
            EmbeddingModel::RoBERTaBase => "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
            EmbeddingModel::FinBERT => "ProsusAI/finbert",
            EmbeddingModel::Custom { name } => name,
        }
    }

    /// Expected embedding dimension.
    pub fn dimension(&self) -> usize {
        match self {
            EmbeddingModel::AllMiniLmL6V2 => 384,
            EmbeddingModel::AllMiniLmL12V2 => 384,
            EmbeddingModel::BERTBaseUncased => 768,
            EmbeddingModel::RoBERTaBase => 768,
            EmbeddingModel::FinBERT => 768,
            EmbeddingModel::Custom { name: _ } => 768, // assume 768 for custom
        }
    }
}

/// Text embedder: generates sentence embeddings.
pub struct TextEmbedder {
    model: EmbeddingModel,
    /// Normalized embeddings flag.
    normalize: bool,
}

impl TextEmbedder {
    pub fn new(model: EmbeddingModel) -> Self {
        Self { model, normalize: true }
    }

    pub fn finbert() -> Self {
        Self::new(EmbeddingModel::FinBERT)
    }

    /// Get the embedding model.
    pub fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    /// Generate embedding for a single text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // In production, this uses rust-bert or candle-transformers.
        // Here we produce a deterministic hash-based embedding for testing.
        let dim = self.model.dimension();
        let mut embedding = Vec::with_capacity(dim);
        let bytes = text.as_bytes();
        for i in 0..dim {
            let byte_idx = i % bytes.len();
            let val = bytes[byte_idx] as f32 / 255.0;
            // Add some position-dependent variation
            let pos_mod = ((i as f32 * 0.01 * byte_idx as f32).sin() + 1.0) * 0.5;
            embedding.push(val * pos_mod);
        }
        if self.normalize {
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            for v in embedding.iter_mut() { *v /= norm; }
        }
        Ok(embedding)
    }

    /// Embedding dimension.
    pub fn dimension(&self) -> usize {
        self.model.dimension()
    }
}

/// Batch embedder for efficient bulk processing.
pub struct BatchEmbedder {
    embedder: TextEmbedder,
    batch_size: usize,
    cache: EmbeddingCache,
}

impl BatchEmbedder {
    pub fn new(model: EmbeddingModel, batch_size: usize) -> Self {
        Self {
            embedder: TextEmbedder::new(model),
            batch_size,
            cache: EmbeddingCache::new(1000),
        }
    }

    /// Embed multiple texts, using cache for duplicates.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_cached(t)).collect()
    }

    fn embed_cached(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(embedding) = self.cache.get(text) {
            return Ok(embedding);
        }
        let embedding = self.embedder.embed(text)?;
        self.cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }

    pub fn cache_hits(&self) -> usize { self.cache.hits }
    pub fn cache_size(&self) -> usize { self.cache.size() }
}

/// LRU-style embedding cache.
pub struct EmbeddingCache {
    capacity: usize,
    entries: HashMap<String, Vec<f32>>,
    access_order: Vec<String>,
    hits: usize,
}

impl EmbeddingCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, entries: HashMap::new(), access_order: Vec::new(), hits: 0 }
    }

    pub fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(embedding) = self.entries.get(key) {
            self.hits += 1;
            // Move to end (most recently used)
            self.access_order.retain(|k| k != key);
            self.access_order.push(key.to_string());
            Some(embedding.clone())
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: String, value: Vec<f32>) {
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            if let Some(evict) = self.access_order.first() {
                self.entries.remove(evict);
                self.access_order.remove(0);
            }
        }
        self.access_order.retain(|k| k != &key);
        self.access_order.push(key.clone());
        self.entries.insert(key, value);
    }

    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hits(&self) -> usize { self.hits }
}

/// Calculate cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-8 { return 0.0; }
    dot / denom
}

/// Find the most similar texts to a query embedding.
pub fn find_similar(query: &[f32], candidates: &[(String, Vec<f32>)], top_k: usize) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = candidates
        .iter()
        .map(|(text, emb)| (text.clone(), cosine_similarity(query, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder() {
        let embedder = TextEmbedder::finbert();
        let emb = embedder.embed("The company beat earnings").unwrap();
        assert_eq!(emb.len(), 768);
        // Normalized: norm should be ~1.0
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deterministic() {
        let embedder = TextEmbedder::new(EmbeddingModel::AllMiniLmL6V2);
        let a = embedder.embed("hello").unwrap();
        let b = embedder.embed("hello").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let embedder = TextEmbedder::new(EmbeddingModel::AllMiniLmL6V2);
        let a = embedder.embed("hello").unwrap();
        let b = embedder.embed("goodbye").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let embedder = TextEmbedder::new(EmbeddingModel::AllMiniLmL6V2);
        let emb = embedder.embed("test").unwrap();
        let sim = cosine_similarity(&emb, &emb);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cache() {
        let mut cache = EmbeddingCache::new(2);
        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);
        assert_eq!(cache.size(), 2);
        cache.insert("c".to_string(), vec![3.0]); // evicts "a"
        assert_eq!(cache.size(), 2);
        assert!(cache.get("a").is_none());
    }

    #[test]
    fn test_find_similar() {
        let embedder = TextEmbedder::new(EmbeddingModel::AllMiniLmL6V2);
        let query = embedder.embed("earnings beat").unwrap();
        let candidates = vec![
            ("revenue growth".to_string(), embedder.embed("revenue growth").unwrap()),
            ("the weather".to_string(), embedder.embed("the weather").unwrap()),
        ];
        let similar = find_similar(&query, &candidates, 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_batch_embedder_cache() {
        let mut batcher = BatchEmbedder::new(EmbeddingModel::AllMiniLmL6V2, 32);
        let texts = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
        let results = batcher.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(batcher.cache_hits(), 1); // "hello" cached on second encounter
    }
}
