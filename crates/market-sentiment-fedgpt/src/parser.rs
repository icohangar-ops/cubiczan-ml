//! Fed statement/FOMC minute tokenizer and entity recognizer.
//!
//! Provides text normalization, keyword extraction, sentence segmentation,
//! and basic entity recognition for Federal Reserve text analysis.

use regex::Regex;
use std::collections::HashMap;

/// Category of a recognized entity in the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityCategory {
    Date,
    Percentage,
    BasisPoints,
    RateTarget,
    EconomicIndicator,
    MonetaryTerm,
}

/// A recognized entity in the text.
#[derive(Debug, Clone)]
pub struct RecognizedEntity {
    pub text: String,
    pub category: EntityCategory,
    pub start: usize,
    pub end: usize,
}

/// Extracted keyword with category.
#[derive(Debug, Clone)]
pub struct ExtractedKeyword {
    pub keyword: String,
    pub category: KeywordCategory,
    pub weight: f64,
}

/// Keyword categories for Fed text analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeywordCategory {
    RateRelated,
    Inflation,
    Employment,
    Growth,
    FinancialStability,
    ForwardGuidance,
    Risk,
}

/// A parsed sentence from the Fed statement.
#[derive(Debug, Clone)]
pub struct ParsedSentence {
    pub text: String,
    pub keywords: Vec<ExtractedKeyword>,
    pub entities: Vec<RecognizedEntity>,
    pub index: usize,
}

/// Parsed tokens from a Fed statement.
#[derive(Debug, Clone)]
pub struct ParsedTokens {
    pub sentences: Vec<ParsedSentence>,
    pub keywords: Vec<ExtractedKeyword>,
    pub entities: Vec<RecognizedEntity>,
    pub normalized_text: String,
    pub word_count: usize,
    pub rate_action_hint: Option<crate::types::RateAction>,
    pub target_rate_low: Option<f64>,
    pub target_rate_high: Option<f64>,
    pub hawkish_keyword_count: usize,
    pub dovish_keyword_count: usize,
}

/// Fed text parser — tokenizes, normalizes, and extracts entities.
pub struct FedTextParser {
    sentence_splitter: Regex,
    whitespace_re: Regex,
    rate_target_re: Regex,
    percentage_re: Regex,
    basis_points_re: Regex,
    date_re: Regex,
    economic_indicator_re: Regex,
}

impl FedTextParser {
    /// Create a new parser with compiled regex patterns.
    pub fn new() -> Self {
        Self {
            // Split on sentence-ending punctuation followed by space or end
            sentence_splitter: Regex::new(r"[.!?]\s+").unwrap(),
            // Collapse multiple whitespace
            whitespace_re: Regex::new(r"\s+").unwrap(),
            // Match rate target ranges like "5.25 to 5.50 percent"
            rate_target_re: Regex::new(
                r"(\d+\.?\d*)\s*(?:to|and|–|-)\s*(\d+\.?\d*)\s*percent"
            ).unwrap(),
            // Match percentages like "2 percent" or "2%"
            percentage_re: Regex::new(r"(\d+\.?\d*)\s*percent").unwrap(),
            // Match basis points mentions
            basis_points_re: Regex::new(
                r"(\d+)\s*basis\s*point|(\d+)\s*bps|(\d+)\s*bp",
            ).unwrap(),
            // Match date patterns like "June 12, 2024" or "2024-06-12"
            date_re: Regex::new(
                r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}",
            ).unwrap(),
            // Match economic indicators
            economic_indicator_re: Regex::new(
                r"(?:GDP|CPI|PCE|unemployment rate|nonfarm payrolls|consumer price index|personal consumption expenditures|core inflation|jobless claims)",
            ).unwrap(),
        }
    }

    /// Parse a Fed statement text into structured tokens.
    pub fn parse(&self, text: &str) -> ParsedTokens {
        let normalized = self.normalize_text(text);
        let sentences = self.segment_sentences(&normalized);
        let entities = self.recognize_entities(&normalized);
        let keywords = self.extract_keywords(&normalized);

        let word_count = normalized.split_whitespace().count();

        let (rate_action_hint, target_low, target_high) =
            self.detect_rate_action(&normalized, &keywords);

        let hawkish_kw_count = keywords.iter()
            .filter(|k| matches!(k.category, KeywordCategory::RateRelated) && (
                k.keyword.contains("hike") || k.keyword.contains("increase")
                || k.keyword.contains("raise") || k.keyword.contains("tighten")
            )).count();
        let dovish_kw_count = keywords.iter()
            .filter(|k| matches!(k.category, KeywordCategory::RateRelated) && (
                k.keyword.contains("cut") || k.keyword.contains("decrease")
                || k.keyword.contains("lower") || k.keyword.contains("reduce")
                || k.keyword.contains("easing")
            )).count();

        ParsedTokens {
            sentences,
            keywords,
            entities,
            normalized_text: normalized,
            word_count,
            rate_action_hint,
            target_rate_low: target_low,
            target_rate_high: target_high,
            hawkish_keyword_count: hawkish_kw_count,
            dovish_keyword_count: dovish_kw_count,
        }
    }

    /// Normalize text: lowercase, collapse whitespace, trim.
    pub fn normalize_text(&self, text: &str) -> String {
        let text = text.to_lowercase();
        let text = self.whitespace_re.replace_all(&text, " ");
        text.trim().to_string()
    }

    /// Split text into sentences.
    pub fn segment_sentences(&self, text: &str) -> Vec<ParsedSentence> {
        let raw_sentences: Vec<&str> = self.sentence_splitter
            .split(text)
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?'))
            .collect();

        raw_sentences.into_iter().enumerate().map(|(idx, s)| {
            let trimmed = s.trim().to_string();
            let keywords = self.extract_keywords(&trimmed);
            let entities = self.recognize_entities(&trimmed);
            ParsedSentence {
                text: trimmed,
                keywords,
                entities,
                index: idx,
            }
        }).collect()
    }

    /// Extract keywords from text with categories and weights.
    pub fn extract_keywords(&self, text: &str) -> Vec<ExtractedKeyword> {
        let mut keywords = Vec::new();

        // Rate-related keywords
        let rate_keywords = [
            ("federal funds rate", 1.5),
            ("target range", 1.3),
            ("interest rate", 1.2),
            ("rate hike", 1.8),
            ("rate cut", 1.8),
            ("rate increase", 1.6),
            ("rate decrease", 1.6),
            ("basis point", 1.4),
            ("tightening", 1.5),
            ("easing", 1.5),
            ("accommodative", 1.3),
            ("restrictive", 1.3),
            ("normalization", 1.2),
            ("quantitative tightening", 1.5),
            ("quantitative easing", 1.5),
            ("forward guidance", 1.1),
            ("monetary policy", 1.3),
            ("maintain the target", 1.4),
            ("lower the target", 1.6),
            ("raise the target", 1.6),
            ("increase the target", 1.5),
            ("reduce the target", 1.5),
            ("hold steady", 1.2),
        ];

        // Inflation keywords
        let inflation_keywords = [
            ("inflation", 1.3),
            ("inflationary", 1.4),
            ("disinflation", 1.2),
            ("core inflation", 1.5),
            ("price stability", 1.4),
            ("price pressures", 1.3),
            ("2 percent objective", 1.6),
            ("2 percent goal", 1.6),
            ("inflation expectations", 1.4),
            ("elevated inflation", 1.5),
            ("sticky inflation", 1.5),
            ("moderating inflation", 1.2),
            ("transitory", 1.1),
        ];

        // Employment keywords
        let employment_keywords = [
            ("labor market", 1.3),
            ("employment", 1.2),
            ("unemployment", 1.3),
            ("job gains", 1.2),
            ("payroll", 1.2),
            ("workforce", 1.1),
            ("wage growth", 1.3),
            ("labor force participation", 1.2),
            ("jobless", 1.2),
            ("tight labor", 1.4),
            ("strong employment", 1.3),
            ("slowing employment", 1.3),
        ];

        // Growth keywords
        let growth_keywords = [
            ("economic activity", 1.3),
            ("economic growth", 1.3),
            ("gdp", 1.3),
            ("expansion", 1.2),
            ("contraction", 1.3),
            ("recession", 1.5),
            ("recovery", 1.2),
            ("moderate growth", 1.2),
            ("strong pace", 1.3),
            ("soft landing", 1.4),
            ("modest growth", 1.1),
            ("slowing economy", 1.3),
            ("robust activity", 1.3),
        ];

        // Financial stability keywords
        let stability_keywords = [
            ("financial stability", 1.4),
            ("systemic risk", 1.5),
            ("banking sector", 1.3),
            ("credit conditions", 1.3),
            ("liquidity", 1.3),
            ("leverage", 1.2),
            ("asset valuations", 1.2),
            ("volatility", 1.1),
            ("market functioning", 1.3),
        ];

        // Forward guidance keywords
        let guidance_keywords = [
            ("anticipates", 1.1),
            ("expects", 1.0),
            ("projects", 1.0),
            ("likely to", 1.0),
            ("prepared to", 1.2),
            ("appropriate to", 1.1),
            ("data dependent", 1.0),
            ("patient", 1.1),
            ("gradual", 1.0),
        ];

        // Risk keywords
        let risk_keywords = [
            ("downside risk", 1.3),
            ("upside risk", 1.3),
            ("uncertainty", 1.2),
            ("risks remain", 1.1),
            ("balanced risks", 1.1),
        ];

        let add_keywords = |pairs: &[(&str, f64)], category: KeywordCategory, output: &mut Vec<ExtractedKeyword>| {
            for (keyword, weight) in pairs {
                if text.contains(keyword) {
                    output.push(ExtractedKeyword {
                        keyword: keyword.to_string(),
                        category,
                        weight: *weight,
                    });
                }
            }
        };

        add_keywords(&rate_keywords, KeywordCategory::RateRelated, &mut keywords);
        add_keywords(&inflation_keywords, KeywordCategory::Inflation, &mut keywords);
        add_keywords(&employment_keywords, KeywordCategory::Employment, &mut keywords);
        add_keywords(&growth_keywords, KeywordCategory::Growth, &mut keywords);
        add_keywords(&stability_keywords, KeywordCategory::FinancialStability, &mut keywords);
        add_keywords(&guidance_keywords, KeywordCategory::ForwardGuidance, &mut keywords);
        add_keywords(&risk_keywords, KeywordCategory::Risk, &mut keywords);

        // Deduplicate by keyword text (keep highest weight)
        let mut best: HashMap<String, ExtractedKeyword> = HashMap::new();
        for kw in keywords {
            best.entry(kw.keyword.clone())
                .and_modify(|existing| {
                    if kw.weight > existing.weight {
                        *existing = kw.clone();
                    }
                })
                .or_insert(kw);
        }

        let mut result: Vec<_> = best.into_values().collect();
        result.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Recognize entities in the text.
    pub fn recognize_entities(&self, text: &str) -> Vec<RecognizedEntity> {
        let mut entities = Vec::new();

        // Rate targets
        for cap in self.rate_target_re.find_iter(text) {
            entities.push(RecognizedEntity {
                text: cap.as_str().to_string(),
                category: EntityCategory::RateTarget,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Percentages
        for cap in self.percentage_re.find_iter(text) {
            entities.push(RecognizedEntity {
                text: cap.as_str().to_string(),
                category: EntityCategory::Percentage,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Basis points
        for cap in self.basis_points_re.find_iter(text) {
            entities.push(RecognizedEntity {
                text: cap.as_str().to_string(),
                category: EntityCategory::BasisPoints,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Dates
        for cap in self.date_re.find_iter(text) {
            entities.push(RecognizedEntity {
                text: cap.as_str().to_string(),
                category: EntityCategory::Date,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Economic indicators
        for cap in self.economic_indicator_re.find_iter(text) {
            entities.push(RecognizedEntity {
                text: cap.as_str().to_string(),
                category: EntityCategory::EconomicIndicator,
                start: cap.start(),
                end: cap.end(),
            });
        }

        // Sort by position
        entities.sort_by_key(|e| e.start);
        entities
    }

    /// Detect rate action from text and keywords.
    fn detect_rate_action(&self, text: &str, keywords: &[ExtractedKeyword])
        -> (Option<crate::types::RateAction>, Option<f64>, Option<f64>) {
        use crate::types::RateAction;

        let mut action = None;

        if text.contains("lower the target")
            || text.contains("reduce the target")
            || text.contains("cut the rate")
            || text.contains("reduce the rate")
            || text.contains("decrease the rate")
            || text.contains("lower the rate")
            || text.contains("decided to lower")
            || text.contains("25 basis point reduction")
        {
            action = Some(RateAction::Cut);
        } else if text.contains("raise the target")
            || text.contains("increase the target")
            || text.contains("increase the rate")
            || text.contains("rate hike")
            || text.contains("rate increase")
            || text.contains("increase the federal funds rate")
            || text.contains("decided to raise")
        {
            action = Some(RateAction::Hike);
        } else if text.contains("maintain the target")
            || text.contains("hold steady")
            || text.contains("hold the rate")
            || text.contains("decided to maintain")
            || text.contains("no change")
        {
            action = Some(RateAction::Hold);
        }

        // Check for emergency language
        if text.contains("emergency") {
            action = match action {
                Some(RateAction::Hike) | None => Some(RateAction::EmergencyHike),
                Some(RateAction::Cut) => Some(RateAction::EmergencyCut),
                other => other,
            };
        }

        // Parse rate target range
        let mut target_low = None;
        let mut target_high = None;
        if let Some(caps) = self.rate_target_re.captures(text) {
            if let (Some(lo), Some(hi)) = (caps.get(1), caps.get(2)) {
                if let (Ok(lo_val), Ok(hi_val)) = (
                    lo.as_str().parse::<f64>(),
                    hi.as_str().parse::<f64>(),
                ) {
                    target_low = Some(lo_val);
                    target_high = Some(hi_val);
                }
            }
        }

        // Filter rate-related keywords for action hints
        if action.is_none() {
            for kw in keywords {
                if kw.category == KeywordCategory::RateRelated {
                    if kw.keyword.contains("hike") || kw.keyword.contains("increase") {
                        action = Some(RateAction::Hike);
                        break;
                    } else if kw.keyword.contains("cut") || kw.keyword.contains("decrease")
                        || kw.keyword.contains("reduce") || kw.keyword.contains("easing") {
                        action = Some(RateAction::Cut);
                        break;
                    } else if kw.keyword.contains("maintain") || kw.keyword.contains("hold") {
                        action = Some(RateAction::Hold);
                        break;
                    }
                }
            }
        }

        (action, target_low, target_high)
    }
}

impl Default for FedTextParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parser() -> FedTextParser {
        FedTextParser::new()
    }

    #[test]
    fn test_normalize_text() {
        let parser = make_parser();
        let result = parser.normalize_text("  The  Committee\n\nmet  today.  ");
        assert_eq!(result, "the committee met today.");
    }

    #[test]
    fn test_normalize_text_lowercases() {
        let parser = make_parser();
        let result = parser.normalize_text("FEDERAL RESERVE");
        assert_eq!(result, "federal reserve");
    }

    #[test]
    fn test_segment_sentences() {
        let parser = make_parser();
        let text = "first sentence. second sentence! third sentence?";
        let sentences = parser.segment_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0].text, "first sentence");
        assert_eq!(sentences[1].text, "second sentence");
        assert_eq!(sentences[2].text, "third sentence");
    }

    #[test]
    fn test_segment_sentences_with_empty() {
        let parser = make_parser();
        let text = "only one sentence.";
        let sentences = parser.segment_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "only one sentence");
    }

    #[test]
    fn test_extract_keywords_hawkish() {
        let parser = make_parser();
        let text = "the committee decided to maintain the target range for the federal funds rate. inflation remains elevated.";
        let keywords = parser.extract_keywords(text);

        let kw_texts: Vec<&str> = keywords.iter().map(|k| k.keyword.as_str()).collect();
        assert!(kw_texts.contains(&"federal funds rate"), "Should find 'federal funds rate' in {:?}", kw_texts);
        assert!(kw_texts.contains(&"inflation"), "Should find 'inflation' in {:?}", kw_texts);
        assert!(kw_texts.contains(&"maintain the target"), "Should find 'maintain the target' in {:?}", kw_texts);
    }

    #[test]
    fn test_extract_keywords_dovish() {
        let parser = make_parser();
        let text = "the committee decided to lower the target range. the labor market \
                   is moderating. gradual easing expected.";
        let keywords = parser.extract_keywords(text);

        let kw_texts: Vec<&str> = keywords.iter().map(|k| k.keyword.as_str()).collect();
        assert!(kw_texts.contains(&"lower the target") || kw_texts.contains(&"easing"));
        assert!(kw_texts.contains(&"labor market"));
    }

    #[test]
    fn test_extract_keywords_categories() {
        let parser = make_parser();
        let text = "gdp growth was moderate. unemployment rate is low. financial stability \
                   risks are balanced.";
        let keywords = parser.extract_keywords(text);

        let categories: Vec<KeywordCategory> = keywords.iter().map(|k| k.category).collect();
        assert!(categories.contains(&KeywordCategory::Growth));
        assert!(categories.contains(&KeywordCategory::Employment));
        assert!(categories.contains(&KeywordCategory::FinancialStability));
    }

    #[test]
    fn test_extract_keywords_deduplication() {
        let parser = make_parser();
        let text = "inflation is a concern. inflation expectations remain anchored.";
        let keywords = parser.extract_keywords(text);

        let inflation_count = keywords.iter().filter(|k| k.keyword == "inflation").count();
        assert_eq!(inflation_count, 1);
    }

    #[test]
    fn test_recognize_entities_rate_target() {
        let parser = make_parser();
        let text = "maintain the target range at 5.25 to 5.50 percent";
        let entities = parser.recognize_entities(text);

        let rate_targets: Vec<_> = entities.iter()
            .filter(|e| e.category == EntityCategory::RateTarget)
            .collect();
        assert_eq!(rate_targets.len(), 1);
        assert!(rate_targets[0].text.contains("5.25"));
        assert!(rate_targets[0].text.contains("5.50"));
    }

    #[test]
    fn test_recognize_entities_percentage() {
        let parser = make_parser();
        let text = "inflation at 2 percent";
        let entities = parser.recognize_entities(text);

        let pcts: Vec<_> = entities.iter()
            .filter(|e| e.category == EntityCategory::Percentage)
            .collect();
        assert_eq!(pcts.len(), 1);
    }

    #[test]
    fn test_recognize_entities_basis_points() {
        let parser = make_parser();
        let text = "cut by 25 basis points";
        let entities = parser.recognize_entities(text);

        let bps: Vec<_> = entities.iter()
            .filter(|e| e.category == EntityCategory::BasisPoints)
            .collect();
        assert_eq!(bps.len(), 1);
    }

    #[test]
    fn test_recognize_entities_bps_shorthand() {
        let parser = make_parser();
        let text = "adjustment of 50 bps";
        let entities = parser.recognize_entities(text);

        let bps: Vec<_> = entities.iter()
            .filter(|e| e.category == EntityCategory::BasisPoints)
            .collect();
        assert_eq!(bps.len(), 1);
    }

    #[test]
    fn test_recognize_entities_date() {
        let parser = make_parser();
        let text = "on June 12, 2024 the committee met";
        let entities = parser.recognize_entities(text);

        let dates: Vec<_> = entities.iter()
            .filter(|e| e.category == EntityCategory::Date)
            .collect();
        assert_eq!(dates.len(), 1);
    }

    #[test]
    fn test_recognize_entities_economic_indicator() {
        let parser = make_parser();
        let text = "CPI data showed moderating core inflation";
        let entities = parser.recognize_entities(text);

        let indicators: Vec<_> = entities.iter()
            .filter(|e| e.category == EntityCategory::EconomicIndicator)
            .collect();
        assert!(!indicators.is_empty());
    }

    #[test]
    fn test_detect_rate_action_hold() {
        let parser = make_parser();
        let text = "the committee decided to maintain the target range for the federal funds rate";
        let (action, _, _) = parser.detect_rate_action(text, &[]);
        assert_eq!(action, Some(crate::types::RateAction::Hold));
    }

    #[test]
    fn test_detect_rate_action_cut() {
        let parser = make_parser();
        let text = "the committee decided to lower the target range";
        let (action, _, _) = parser.detect_rate_action(text, &[]);
        assert_eq!(action, Some(crate::types::RateAction::Cut));
    }

    #[test]
    fn test_detect_rate_action_hike() {
        let parser = make_parser();
        let text = "the committee decided to increase the target range";
        let (action, _, _) = parser.detect_rate_action(text, &[]);
        assert_eq!(action, Some(crate::types::RateAction::Hike));
    }

    #[test]
    fn test_detect_rate_target_range() {
        let parser = make_parser();
        let text = "target range at 5.25 to 5.50 percent";
        let (_, low, high) = parser.detect_rate_action(text, &[]);
        assert_eq!(low, Some(5.25));
        assert_eq!(high, Some(5.50));
    }

    #[test]
    fn test_full_parse() {
        let parser = make_parser();
        let text = "The Committee decided to maintain the target range for the federal \
                   funds rate at 5.25 to 5.50 percent. Inflation remains elevated at \
                   3.2 percent. The labor market continues to be tight.";
        let tokens = parser.parse(text);

        assert!(!tokens.sentences.is_empty());
        assert!(!tokens.keywords.is_empty());
        assert!(!tokens.entities.is_empty());
        assert!(tokens.word_count > 0);
        assert_eq!(tokens.rate_action_hint, Some(crate::types::RateAction::Hold));
        assert_eq!(tokens.target_rate_low, Some(5.25));
        assert_eq!(tokens.target_rate_high, Some(5.50));
    }
}
