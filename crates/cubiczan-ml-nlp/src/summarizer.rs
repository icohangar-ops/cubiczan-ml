//! # Text Summarization
//!
//! Extractive and abstractive summarization for earnings calls, SEC filings,
//! and financial documents. Includes key phrase extraction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extractive summarizer: selects the most important sentences from text.
pub struct ExtractiveSummarizer {
    num_sentences: usize,
    min_length: usize,
}

impl ExtractiveSummarizer {
    pub fn new(num_sentences: usize) -> Self {
        Self { num_sentences, min_length: 20 }
    }

    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            current.push(c);
            if ".!?".contains(c) {
                let trimmed = current.trim().to_string();
                if trimmed.len() >= self.min_length { sentences.push(trimmed); }
                current.clear();
            }
        }
        if sentences.is_empty() { sentences.push(text.to_string()); }
        sentences
    }

    fn score_sentences(&self, sentences: &[String]) -> Vec<(usize, f64)> {
        let stop_words: std::collections::HashSet<&str> = [
            "the","a","an","is","are","was","were","be","been","have","has","had",
            "do","does","did","will","would","could","should","may","might","to",
            "of","in","for","on","with","at","by","from","as","into","and","or",
            "if","while","about","it","its","this","that","we","our","they","their",
        ].iter().cloned().collect();

        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for s in sentences {
            for w in s.split_whitespace() {
                let clean: String = w.to_lowercase().chars().take_while(|c| c.is_alphanumeric()).collect();
                if clean.len() > 2 && !stop_words.contains(clean.as_str()) {
                    *word_freq.entry(clean).or_insert(0) += 1;
                }
            }
        }
        let max_freq = word_freq.values().copied().max().unwrap_or(1);

        sentences.iter().enumerate().map(|(i, s)| {
            let score = s.split_whitespace()
                .filter_map(|w| {
                    let clean: String = w.to_lowercase().chars().take_while(|c| c.is_alphanumeric()).collect();
                    word_freq.get(&clean).copied().map(|f| f as f64 / max_freq as f64)
                }).sum::<f64>() / s.split_whitespace().count().max(1) as f64;
            (i, score)
        }).collect()
    }

    pub fn summarize(&self, text: &str) -> String {
        let sentences = self.split_sentences(text);
        if sentences.len() <= self.num_sentences { return text.to_string(); }
        let mut scored = self.score_sentences(&sentences);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<(usize, f64)> = scored.into_iter().take(self.num_sentences).collect();
        let mut ordered = top; ordered.sort_by_key(|(i, _)| *i);
        ordered.iter().map(|(i, _)| sentences[*i].as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// Abstractive summarizer (wraps extractive as fallback).
pub struct AbstractiveSummarizer {
    extractive: ExtractiveSummarizer,
    model_name: String,
}

impl AbstractiveSummarizer {
    pub fn new(num_sentences: usize) -> Self {
        Self { extractive: ExtractiveSummarizer::new(num_sentences), model_name: "t5-base".into() }
    }
    pub fn from_model(name: &str) -> Self {
        Self { extractive: ExtractiveSummarizer::new(5), model_name: name.into() }
    }
    pub fn summarize(&self, text: &str) -> String { self.extractive.summarize(text) }
    pub fn model(&self) -> &str { &self.model_name }
}

/// Key phrase extractor.
pub struct KeyPhraseExtractor { max_phrases: usize }

impl KeyPhraseExtractor {
    pub fn new(max_phrases: usize) -> Self { Self { max_phrases } }
    pub fn extract(&self, text: &str) -> Vec<(String, f64)> {
        let mut freq: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        for line in text.split('.') {
            let words: Vec<&str> = line.split_whitespace().collect();
            for w in words.windows(2) {
                let phrase = format!("{} {}", w[0], w[1]);
                if phrase.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    || ["revenue","earnings","margin","growth","risk"].iter().any(|k| phrase.contains(k)) {
                    *freq.entry(phrase.to_lowercase()).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
        let mut phrases: Vec<(String, f64)> = freq.into_iter()
            .filter(|(_, c)| *c >= 2)
            .map(|(p, c)| (p, c as f64 / total.max(1) as f64))
            .collect();
        phrases.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        phrases.truncate(self.max_phrases);
        phrases
    }
}

/// Earnings call summarizer.
pub struct EarningsCallSummarizer {
    extractive: ExtractiveSummarizer,
    key_phrases: KeyPhraseExtractor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningsSummary {
    pub summary: String,
    pub key_phrases: Vec<(String, f64)>,
    pub speaker_sections: usize,
    pub sentiment_tone: ToneAnalysis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tone { Bullish, Bearish, Neutral }

impl std::fmt::Display for Tone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Tone::Bullish => write!(f,"Bullish"), Tone::Bearish => write!(f,"Bearish"), Tone::Neutral => write!(f,"Neutral") }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneAnalysis { pub tone: Tone, pub sentiment_score: f64, pub bullish_count: usize, pub bearish_count: usize }

impl EarningsCallSummarizer {
    pub fn new() -> Self { Self { extractive: ExtractiveSummarizer::new(7), key_phrases: KeyPhraseExtractor::new(10) } }

    pub fn summarize(&self, transcript: &str) -> EarningsSummary {
        let lower = transcript.to_lowercase();
        let bullish = ["beat","exceeded","record","growth","strong","confident","optimistic","momentum"];
        let bearish = ["decline","miss","weakness","challenge","headwind","uncertainty","pressure"];
        let bc = bullish.iter().filter(|w| lower.contains(*w)).count();
        let bearc = bearish.iter().filter(|w| lower.contains(*w)).count();
        let total = bc + bearc;
        let score = if total == 0 { 0.5 } else { bc as f64 / total as f64 };
        let tone = if score > 0.65 { Tone::Bullish } else if score < 0.35 { Tone::Bearish } else { Tone::Neutral };
        let section_re = regex::Regex::new(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*):").unwrap();
        let mut sections = 0u32;
        for line in transcript.lines() { if section_re.is_match(line) { sections += 1; } }
        EarningsSummary { summary: self.extractive.summarize(transcript), key_phrases: self.key_phrases.extract(transcript), speaker_sections: sections as usize, sentiment_tone: ToneAnalysis { tone, sentiment_score: score, bullish_count: bc, bearish_count: bearc } }
    }
}

/// SEC filing summarizer.
pub struct SECFilingSummarizer { extractive: ExtractiveSummarizer }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SECFilingSummary { pub summary: String, pub exhibits: Vec<String>, pub financial_metrics: Vec<(String, f64)>, pub key_phrases: Vec<(String, f64)> }

impl SECFilingSummarizer {
    pub fn new() -> Self { Self { extractive: ExtractiveSummarizer::new(10) } }
    pub fn summarize(&self, text: &str) -> SECFilingSummary {
        let exhibits: Vec<String> = regex::Regex::new(r"Exhibit\s+[\d.]+[a-z]?").unwrap()
            .find_iter(text).map(|m| m.as_str().to_string()).collect();
        let mut metrics = Vec::new();
        for (pat, name) in [
            (r"(?:revenue|Revenue)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "revenue"),
            (r"(?:net income|Net Income)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "net_income"),
            (r"(?:EBITDA)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "ebitda"),
        ] {
            if let Ok(re) = regex::Regex::new(pat) {
                if let Some(caps) = re.captures(text) {
                    if let Some(m) = caps.get(1) {
                        if let Ok(n) = m.as_str().replace(',', "").parse::<f64>() { metrics.push((name.into(), n)); }
                    }
                }
            }
        }
        SECFilingSummary { summary: self.extractive.summarize(text), exhibits, financial_metrics: metrics, key_phrases: KeyPhraseExtractor::new(15).extract(text) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractive() {
        let s = ExtractiveSummarizer::new(2);
        let text = "Apple reported record revenue of 94.8 billion dollars in Q4 2024. The company exceeded expectations. iPhone sales drove growth with a 15 percent increase. Services hit new highs.";
        let sum = s.summarize(text);
        assert!(sum.len() < text.len());
    }

    #[test]
    fn test_earnings_tone() {
        let s = EarningsCallSummarizer::new();
        let t = s.summarize("CEO: We beat earnings with record growth. Strong momentum.");
        assert_eq!(t.sentiment_tone.tone, Tone::Bullish);
    }

    #[test]
    fn test_sec_summary() {
        let s = SECFilingSummarizer::new();
        let r = s.summarize("Revenue of $52.6 billion. Exhibit 10.1 Material Contract.");
        assert!(!r.exhibits.is_empty());
    }
}
