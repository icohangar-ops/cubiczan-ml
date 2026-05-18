//! Sentiment analysis: keyword-based NLP on SEC filing text.

use crate::config::regulatory_keywords;

/// Positive keywords for supply-chain sentiment.
const POSITIVE_WORDS: &[&str] = &[
    "increase", "growth", "positive", "improve", "opportunity", "strong",
    "diversification", "innovation", "initiative", "secured", "optimistic",
    "expansion", "development", "investing", "reduce", "transition",
    "beneficial", "favorable", "progress", "advancement",
];

/// Negative keywords for supply-chain sentiment.
const NEGATIVE_WORDS: &[&str] = &[
    "risk", "uncertainty", "disruption", "constraint", "restriction",
    "volatile", "decline", "decrease", "challenge", "pressure",
    "ban", "sanction", "nationalization", "scrutiny", "compliance cost",
    "elevated", "tension", "dependency", "concern", "warning",
];

/// Intensifier words that amplify the next word's sentiment.
const INTENSIFIERS: &[&str] = &["significantly", "highly", "severely", "extremely", "substantially"];

/// Diminisher words that reduce the next word's sentiment.
const DIMINISHERS: &[&str] = &["slightly", "marginally", "partially", "somewhat"];

/// Strip common punctuation from a word.
fn clean_word(word: &str) -> &str {
    word.trim_matches(|c: char| c == '.' || c == ',' || c == ';' || c == ':'
        || c == '!' || c == '?' || c == '(' || c == ')' || c == '[' || c == ']'
        || c == '{' || c == '}' || c == '"' || c == '\'')
}

/// Simple keyword-based sentiment analyzer.
///
/// Returns a float between -1.0 (very negative) and 1.0 (very positive).
pub fn simple_sentiment_analyzer(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    let total_words = words.len();

    if total_words == 0 {
        return 0.0;
    }

    let mut positive_count = 0.0_f64;
    let mut negative_count = 0.0_f64;

    for (i, word) in words.iter().enumerate() {
        let cleaned = clean_word(word);

        // Check for intensifiers/diminishers modifying the next word
        let modifier = if i > 0 {
            let prev = clean_word(words[i - 1]);
            if INTENSIFIERS.contains(&prev) {
                1.5
            } else if DIMINISHERS.contains(&prev) {
                0.5
            } else {
                1.0
            }
        } else {
            1.0
        };

        if POSITIVE_WORDS.contains(&cleaned) {
            positive_count += modifier;
        } else if NEGATIVE_WORDS.contains(&cleaned) {
            negative_count += modifier;
        }
    }

    // Normalize
    let raw_score = (positive_count - negative_count) / (total_words as f64 * 0.05).max(1.0);
    raw_score.clamp(-1.0, 1.0)
}

/// Score regulatory risk based on keyword presence in text.
///
/// Returns a float between 0.0 (no risk) and 100.0 (maximum risk).
pub fn regulatory_risk_scorer(text: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let keywords = regulatory_keywords();
    let mut total_risk = 0.0_f64;
    let mut keyword_count = 0;

    for (keyword, &weight) in &keywords {
        if text_lower.contains(keyword) {
            let count = text_lower.matches(keyword).count();
            total_risk += weight * count as f64;
            keyword_count += count;
        }
    }

    if keyword_count == 0 {
        return 10.0; // Baseline risk
    }

    let avg_risk = total_risk / keyword_count as f64;
    let risk_score = avg_risk * 60.0 + 15.0;
    risk_score.clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_positive() {
        let text = "Growth is strong and opportunities increase significantly.";
        let score = simple_sentiment_analyzer(text);
        assert!(score > 0.0, "Expected positive sentiment, got {}", score);
    }

    #[test]
    fn test_sentiment_negative() {
        let text = "Risks are elevated and supply chain disruption threatens operations.";
        let score = simple_sentiment_analyzer(text);
        assert!(score < 0.0, "Expected negative sentiment, got {}", score);
    }

    #[test]
    fn test_sentiment_neutral() {
        let text = "The meeting was held on Tuesday.";
        let score = simple_sentiment_analyzer(text);
        assert!(score.abs() < 0.5, "Expected neutral sentiment, got {}", score);
    }

    #[test]
    fn test_reg_risk_high() {
        let text = "export ban and nationalization risk with trade sanctions.";
        let risk = regulatory_risk_scorer(text);
        assert!(risk > 50.0, "Expected high regulatory risk, got {}", risk);
    }

    #[test]
    fn test_reg_risk_low() {
        let text = "The weather is nice today.";
        let risk = regulatory_risk_scorer(text);
        assert_eq!(risk, 10.0, "Expected baseline risk");
    }

    #[test]
    fn test_reg_risk_positive_keywords() {
        let text = "free trade agreement and supply chain diversification initiative.";
        let risk = regulatory_risk_scorer(text);
        assert!(risk < 20.0, "Expected low risk with positive keywords, got {}", risk);
    }
}
