//! # Named Entity Recognition
//!
//! Financial NER for extracting companies, currencies, amounts, percentages,
//! dates, and SEC-specific entities from text.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tagged entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NEREntity {
    pub text: String,
    pub entity_type: EntityType,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Company, Currency, Money, Percent, Date, Person, Location,
    Ticker, FinancialMetric, LegalEntity, Exhibit, Section, Unknown,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Company => write!(f, "ORG"),
            EntityType::Money => write!(f, "MONEY"),
            EntityType::Percent => write!(f, "PERCENT"),
            EntityType::Ticker => write!(f, "TICKER"),
            EntityType::Date => write!(f, "DATE"),
            EntityType::Currency => write!(f, "CURRENCY"),
            EntityType::Exhibit => write!(f, "EXHIBIT"),
            EntityType::Section => write!(f, "SECTION"),
            EntityType::FinancialMetric => write!(f, "METRIC"),
            _ => write!(f, "MISC"),
        }
    }
}

/// Financial NER using rule-based extraction.
pub struct FinancialNER {
    tickers: HashMap<String, String>,
}

impl FinancialNER {
    pub fn new() -> Self { Self { tickers: HashMap::new() } }

    pub fn add_tickers(&mut self, tickers: HashMap<String, String>) {
        self.tickers = tickers;
    }

    pub fn extract(&self, text: &str) -> Vec<NEREntity> {
        let mut entities = Vec::new();
        let patterns: Vec<(EntityType, &str)> = vec![
            (EntityType::Money, r"\$[\d,.]+(?:\s*(?:million|billion|trillion|M|B|T|K))?"),
            (EntityType::Percent, r"\d+(?:\.\d+)?(?:\s*(?:%|percent|basis points|bps))"),
            (EntityType::Ticker, r"\$[A-Z]{1,5}(?:\.[A-Z])?|\([A-Z]{1,5}(?:\.[A-Z])?\)"),
            (EntityType::Date, r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"),
            (EntityType::Currency, r"\b(?:USD|EUR|GBP|JPY|CNY|CHF|CAD|AUD|INR|SGD|HKD)\b"),
        ];
        for (entity_type, pattern) in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for mat in re.find_iter(text) {
                    let clean = mat.as_str().trim_start_matches('$').trim_matches('(').trim_matches(')').to_string();
                    entities.push(NEREntity {
                        text: clean, entity_type: *entity_type, score: 0.95,
                        start: mat.start(), end: mat.end(),
                    });
                }
            }
        }
        entities
    }

    pub fn extract_amounts(&self, text: &str) -> Vec<f64> {
        self.extract(text).into_iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .filter_map(|e| parse_money(&e.text)).collect()
    }
}

fn parse_money(text: &str) -> Option<f64> {
    let clean = text.trim_start_matches('$').replace(',', "");
    let num_part: String = clean.chars().take_while(|c| c.is_digit(10) || *c == '.').collect();
    let value: f64 = num_part.parse().ok()?;
    let multiplier = if clean.contains("trillion") || clean.contains("T") { 1e12 }
        else if clean.contains("billion") || clean.contains("B") { 1e9 }
        else if clean.contains("million") || clean.contains("M") { 1e6 }
        else if clean.contains("K") { 1e3 } else { 1.0 };
    Some(value * multiplier)
}

/// SEC filing entity extractor.
pub struct SECEntityExtractor { inner: FinancialNER }

impl SECEntityExtractor {
    pub fn new() -> Self { Self { inner: FinancialNER::new() } }

    pub fn extract_sec_entities(&self, text: &str) -> Vec<NEREntity> {
        let mut entities = self.inner.extract(text);
        let exhibit_re = regex::Regex::new(r"Exhibit\s+[\d.]+[a-z]?").unwrap();
        for mat in exhibit_re.find_iter(text) {
            entities.push(NEREntity {
                text: mat.as_str().to_string(), entity_type: EntityType::Exhibit,
                score: 0.98, start: mat.start(), end: mat.end(),
            });
        }
        let section_re = regex::Regex::new(r"(?:PART|ITEM|Item)\s+[IVXLCDM\d]+(?:\.)?").unwrap();
        for mat in section_re.find_iter(text) {
            entities.push(NEREntity {
                text: mat.as_str().to_string(), entity_type: EntityType::Section,
                score: 0.97, start: mat.start(), end: mat.end(),
            });
        }
        entities
    }

    pub fn extract_financial_metrics(&self, text: &str) -> Vec<(String, f64)> {
        let mut metrics = Vec::new();
        let patterns = vec![
            (r"(?:revenue|Revenue)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "revenue"),
            (r"(?:net income|Net Income)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "net_income"),
            (r"(?:EPS|earnings per share)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "eps"),
            (r"(?:EBITDA)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "ebitda"),
            (r"(?:total assets|Total Assets)\s+(?:of\s+)?(?:\$)?([\d,.]+)", "total_assets"),
        ];
        for (pattern, name) in patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(caps) = re.captures(text) {
                    if let Some(m) = caps.get(1) {
                        if let Ok(num) = m.as_str().replace(',', "").parse::<f64>() {
                            metrics.push((name.to_string(), num));
                        }
                    }
                }
            }
        }
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ner_money() {
        let ner = FinancialNER::new();
        let text = "AAPL reported $3.2B in revenue, up 15% from last year.";
        let entities = ner.extract(text);
        let money: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Money).collect();
        assert!(!money.is_empty());
    }

    #[test]
    fn test_parse_money() {
        assert_eq!(parse_money("$3.2B"), Some(3.2e9));
        assert_eq!(parse_money("$500,000"), Some(500_000.0));
    }

    #[test]
    fn test_sec_extract() {
        let extractor = SECEntityExtractor::new();
        let text = "Exhibit 10.1 Material Contract. See Item 1A for risk factors.";
        let entities = extractor.extract_sec_entities(text);
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Exhibit));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Section));
    }
}
