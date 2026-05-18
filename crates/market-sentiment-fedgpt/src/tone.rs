//! Deterministic sentiment scoring for Federal Reserve text.
//!
//! Provides hawkish/dovish keyword weighting, tone shift detection,
//! sentiment time-series construction, and multi-dimensional scoring.

use crate::parser::{FedTextParser, ParsedTokens};
use crate::types::{MonetaryPolicy, SentimentScore};

/// Hawkish keyword -> positive weight, dovish keyword -> negative weight.
struct KeywordWeights {
    hawkish: Vec<(&'static str, f64)>,
    dovish: Vec<(&'static str, f64)>,
}

impl KeywordWeights {
    fn new() -> Self {
        Self {
            hawkish: vec![
                ("increase", 0.6),
                ("raise", 0.7),
                ("tighten", 0.8),
                ("tightening", 0.8),
                ("restrictive", 0.7),
                ("elevated inflation", 0.9),
                ("inflationary pressures", 0.8),
                ("upside risk to inflation", 1.0),
                ("strong pace", 0.5),
                ("tight labor", 0.6),
                ("strong employment", 0.5),
                ("wage growth", 0.5),
                ("additional rate increases", 1.2),
                ("prepared to raise", 0.9),
                ("rate hike", 1.0),
                ("quantitative tightening", 0.8),
                ("remain elevated", 0.7),
                ("pressures", 0.4),
                ("vigilant", 0.5),
                ("hawkish", 1.0),
                ("inflation remains elevated", 1.0),
                ("committed to returning inflation", 0.8),
                ("further tightening", 0.9),
                ("strongly committed", 0.6),
                ("restrictive stance", 0.8),
                ("appropriate to raise", 0.9),
                ("concerned about inflation", 0.7),
            ],
            dovish: vec![
                ("decrease", 0.6),
                ("lower", 0.7),
                ("cut", 0.8),
                ("easing", 0.8),
                ("accommodative", 0.7),
                ("moderating", 0.6),
                ("disinflation", 0.8),
                ("gradual", 0.4),
                ("patient", 0.5),
                ("moderate pace", 0.4),
                ("slowing", 0.5),
                ("moderating inflation", 0.7),
                ("soft landing", 0.6),
                ("premature to tighten", 0.9),
                ("additional easing", 0.8),
                ("prepared to cut", 0.9),
                ("rate cut", 1.0),
                ("quantitative easing", 0.8),
                ("lower the target", 0.9),
                ("reduce the rate", 0.8),
                ("dovish", 1.0),
                ("inflation has moderated", 0.7),
                ("appropriate to ease", 0.8),
                ("inflation moving toward 2 percent", 0.8),
                ("gradual adjustments", 0.6),
                ("appropriate to lower", 0.9),
                ("measured pace", 0.4),
                ("labor market moderating", 0.5),
                ("gradual reduction", 0.6),
            ],
        }
    }
}

/// Dimension-specific keyword weights for multi-dimensional scoring.
struct DimensionWeights {
    inflation_hawkish: Vec<(&'static str, f64)>,
    inflation_dovish: Vec<(&'static str, f64)>,
    employment_hawkish: Vec<(&'static str, f64)>,
    employment_dovish: Vec<(&'static str, f64)>,
    growth_hawkish: Vec<(&'static str, f64)>,
    growth_dovish: Vec<(&'static str, f64)>,
    stability_hawkish: Vec<(&'static str, f64)>,
    stability_dovish: Vec<(&'static str, f64)>,
}

impl DimensionWeights {
    fn new() -> Self {
        Self {
            inflation_hawkish: vec![
                ("elevated inflation", 1.0),
                ("inflation remains elevated", 1.2),
                ("inflationary pressures", 0.9),
                ("upside risk to inflation", 1.1),
                ("committed to returning inflation", 0.8),
                ("price pressures", 0.7),
                ("inflation expectations", 0.6),
                ("wage growth", 0.5),
            ],
            inflation_dovish: vec![
                ("moderating inflation", 1.0),
                ("disinflation", 1.1),
                ("inflation has moderated", 1.1),
                ("inflation moving toward 2 percent", 1.0),
                ("price stability", 0.7),
            ],
            employment_hawkish: vec![
                ("tight labor", 1.0),
                ("strong employment", 0.9),
                ("wage growth", 0.7),
                ("labor market tight", 0.9),
            ],
            employment_dovish: vec![
                ("labor market moderating", 0.9),
                ("slowing employment", 0.8),
                ("unemployment rising", 0.7),
                ("softening labor", 0.8),
            ],
            growth_hawkish: vec![
                ("strong pace", 0.8),
                ("robust activity", 0.8),
                ("expansion", 0.6),
            ],
            growth_dovish: vec![
                ("slowing economy", 0.8),
                ("modest growth", 0.6),
                ("contraction", 0.9),
                ("recession", 1.0),
                ("soft landing", 0.7),
            ],
            stability_hawkish: vec![
                ("systemic risk", 1.0),
                ("asset valuations", 0.6),
                ("leverage", 0.5),
            ],
            stability_dovish: vec![
                ("stable", 0.5),
                ("resilient", 0.5),
                ("sound", 0.4),
            ],
        }
    }
}

/// Time-series entry for tracking sentiment over time.
#[derive(Debug, Clone)]
pub struct SentimentTimePoint {
    pub date: chrono::NaiveDate,
    pub score: SentimentScore,
}

/// Deterministic tone analyzer for Fed text.
pub struct ToneAnalyzer {
    parser: FedTextParser,
    keyword_weights: KeywordWeights,
    dimension_weights: DimensionWeights,
    /// Threshold for classifying as Hawkish vs Neutral.
    hawkish_threshold: f64,
    /// Threshold for classifying as Neutral vs Dovish.
    dovish_threshold: f64,
}

impl ToneAnalyzer {
    pub fn new() -> Self {
        Self {
            parser: FedTextParser::new(),
            keyword_weights: KeywordWeights::new(),
            dimension_weights: DimensionWeights::new(),
            hawkish_threshold: 0.2,
            dovish_threshold: -0.2,
        }
    }

    /// Analyze a single Fed statement and produce a sentiment score.
    pub fn analyze(&self, text: &str) -> SentimentScore {
        let tokens = self.parser.parse(text);
        let mut score = self.score_tokens(&tokens);
        score.confidence = self.compute_confidence(&tokens);
        score
    }

    /// Analyze with tone shift detection against a prior statement.
    pub fn analyze_with_shift(&self, current_text: &str, prior_text: &str) -> SentimentScore {
        let current_tokens = self.parser.parse(current_text);
        let prior_tokens = self.parser.parse(prior_text);

        let mut score = self.score_tokens(&current_tokens);

        let prior_score = self.score_tokens(&prior_tokens);
        let shift = score.overall - prior_score.overall;
        score.tone_shift = Some(shift);

        // Adjust confidence based on text length
        score.confidence = self.compute_confidence(&current_tokens);

        score
    }

    /// Build a sentiment time series from multiple statements.
    pub fn build_time_series(&self, statements: &[(chrono::NaiveDate, &str)])
        -> Vec<SentimentTimePoint> {
        let mut series = Vec::new();
        let mut prev_score: Option<f64> = None;

        for (date, text) in statements {
            let tokens = self.parser.parse(text);
            let mut score = self.score_tokens(&tokens);

            if let Some(prev) = prev_score {
                score.tone_shift = Some(score.overall - prev);
            }

            score.confidence = self.compute_confidence(&tokens);
            prev_score = Some(score.overall);

            series.push(SentimentTimePoint {
                date: *date,
                score,
            });
        }

        series
    }

    /// Compute the multi-dimensional sentiment score from parsed tokens.
    fn score_tokens(&self, tokens: &ParsedTokens) -> SentimentScore {
        let text = &tokens.normalized_text;

        // Overall score from hawkish/dovish keywords
        let overall = self.compute_overall_score(text);

        // Dimension scores
        let inflation = self.compute_dimension_score(
            text,
            &self.dimension_weights.inflation_hawkish,
            &self.dimension_weights.inflation_dovish,
        );
        let employment = self.compute_dimension_score(
            text,
            &self.dimension_weights.employment_hawkish,
            &self.dimension_weights.employment_dovish,
        );
        let growth = self.compute_dimension_score(
            text,
            &self.dimension_weights.growth_hawkish,
            &self.dimension_weights.growth_dovish,
        );
        let financial_stability = self.compute_dimension_score(
            text,
            &self.dimension_weights.stability_hawkish,
            &self.dimension_weights.stability_dovish,
        );

        // Determine monetary policy stance
        let monetary_policy = if overall > self.hawkish_threshold {
            MonetaryPolicy::Hawkish
        } else if overall < self.dovish_threshold {
            MonetaryPolicy::Dovish
        } else {
            MonetaryPolicy::Neutral
        };

        // Count keywords
        let hawkish_count = self.keyword_weights.hawkish.iter()
            .filter(|(kw, _)| text.contains(*kw))
            .count();
        let dovish_count = self.keyword_weights.dovish.iter()
            .filter(|(kw, _)| text.contains(*kw))
            .count();

        SentimentScore {
            overall: overall.clamp(-1.0, 1.0),
            inflation: inflation.clamp(-1.0, 1.0),
            employment: employment.clamp(-1.0, 1.0),
            growth: growth.clamp(-1.0, 1.0),
            financial_stability: financial_stability.clamp(-1.0, 1.0),
            monetary_policy,
            confidence: 0.0, // set separately
            tone_shift: None,
            hawkish_keyword_count: hawkish_count,
            dovish_keyword_count: dovish_count,
        }
    }

    /// Compute the overall hawkish/dovish score.
    fn compute_overall_score(&self, text: &str) -> f64 {
        let hawkish_sum: f64 = self.keyword_weights.hawkish.iter()
            .filter(|(kw, _)| text.contains(*kw))
            .map(|(_, w)| *w)
            .sum();

        let dovish_sum: f64 = self.keyword_weights.dovish.iter()
            .filter(|(kw, _)| text.contains(*kw))
            .map(|(_, w)| *w)
            .sum();

        let total = hawkish_sum + dovish_sum;
        if total == 0.0 {
            return 0.0;
        }

        (hawkish_sum - dovish_sum) / total
    }

    /// Compute a dimension-specific score.
    fn compute_dimension_score(
        &self,
        text: &str,
        hawkish: &[(&str, f64)],
        dovish: &[(&str, f64)],
    ) -> f64 {
        let hawkish_sum: f64 = hawkish.iter()
            .filter(|(kw, _)| text.contains(*kw))
            .map(|(_, w)| *w)
            .sum();

        let dovish_sum: f64 = dovish.iter()
            .filter(|(kw, _)| text.contains(*kw))
            .map(|(_, w)| *w)
            .sum();

        let total = hawkish_sum + dovish_sum;
        if total == 0.0 {
            return 0.0;
        }

        (hawkish_sum - dovish_sum) / total
    }

    /// Get a reference to the internal parser.
    pub fn parser(&self) -> &FedTextParser {
        &self.parser
    }

    /// Compute confidence based on text length and keyword density.
    pub fn compute_confidence(&self, tokens: &ParsedTokens) -> f64 {
        let word_count = tokens.word_count as f64;
        let keyword_count = tokens.keywords.len() as f64;

        // Minimum word count for reasonable confidence
        if word_count < 20.0 {
            return 0.1;
        }

        // Keyword density: higher density = more signal
        let density = keyword_count / word_count;
        let density_factor = (density * 100.0).min(1.0);

        // Word count factor: longer text = more context
        let length_factor = (word_count / 200.0).min(1.0);

        // Combine factors
        let confidence = (density_factor * 0.6 + length_factor * 0.4).clamp(0.0, 1.0);

        // Boost if there's a clear directional signal
        let total_kw = tokens.hawkish_keyword_count + tokens.dovish_keyword_count;
        if total_kw > 0 {
            let dominance = tokens.hawkish_keyword_count.max(tokens.dovish_keyword_count) as f64
                / total_kw as f64;
            return (confidence * 0.7 + dominance * 0.3).clamp(0.0, 1.0);
        }

        confidence
    }
}

impl Default for ToneAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_analyzer() -> ToneAnalyzer {
        ToneAnalyzer::new()
    }

    #[test]
    fn test_hawkish_statement() {
        let analyzer = make_analyzer();
        let text = "The Committee is strongly committed to returning inflation to its \
                   2 percent objective. Inflation remains elevated. Additional rate \
                   increases may be appropriate. The labor market continues to be tight \
                   with strong wage growth.";
        let score = analyzer.analyze(text);

        assert!(score.overall > 0.0, "Hawkish text should have positive score, got {}", score.overall);
        assert_eq!(score.monetary_policy, MonetaryPolicy::Hawkish);
        assert!(score.hawkish_keyword_count > 0);
    }

    #[test]
    fn test_dovish_statement() {
        let analyzer = make_analyzer();
        let text = "The Committee decided to lower the target range. Inflation has \
                   moderated significantly. The labor market is moderating. Gradual \
                   easing is appropriate. A soft landing appears achievable.";
        let score = analyzer.analyze(text);

        assert!(score.overall < 0.0, "Dovish text should have negative score, got {}", score.overall);
        assert_eq!(score.monetary_policy, MonetaryPolicy::Dovish);
        assert!(score.dovish_keyword_count > 0);
    }

    #[test]
    fn test_neutral_statement() {
        let analyzer = make_analyzer();
        let text = "The Committee will carefully assess incoming information. The economic \
                   outlook is uncertain. The Committee will proceed carefully as it \
                   monitors developments.";
        let score = analyzer.analyze(text);

        // Should be neutral or near-neutral
        assert!(score.overall.abs() <= 0.3, "Neutral text should score near 0, got {}", score.overall);
    }

    #[test]
    fn test_tone_shift_detection() {
        let analyzer = make_analyzer();
        let prior = "The Committee is strongly committed to tightening. Inflation remains \
                     elevated and additional rate increases may be appropriate.";
        let current = "Inflation has moderated. The Committee will proceed carefully and \
                      assess whether additional easing is warranted.";

        let score = analyzer.analyze_with_shift(current, prior);

        assert!(score.tone_shift.is_some());
        // Current should be more dovish than prior, so shift should be negative
        assert!(score.tone_shift.unwrap() < 0.0);
    }

    #[test]
    fn test_tone_shift_same_tone() {
        let analyzer = make_analyzer();
        let text = "The Committee will maintain the target range. Inflation is elevated.";
        let score = analyzer.analyze_with_shift(text, text);

        assert!(score.tone_shift.is_some());
        assert!(score.tone_shift.unwrap().abs() < 0.01);
    }

    #[test]
    fn test_inflation_dimension() {
        let analyzer = make_analyzer();
        let text = "Inflation remains elevated with upside risk to inflation. Price pressures \
                   are building. Inflationary pressures are a concern.";
        let score = analyzer.analyze(text);

        assert!(score.inflation > 0.0, "Elevated inflation text should have positive inflation score");
    }

    #[test]
    fn test_employment_dimension() {
        let analyzer = make_analyzer();
        let text = "The labor market moderating. Slowing employment is evident.";
        let score = analyzer.analyze(text);

        assert!(score.employment < 0.0, "Slowing employment should have negative employment score");
    }

    #[test]
    fn test_growth_dimension() {
        let analyzer = make_analyzer();
        let text = "The economy is in recession. Contraction in economic activity.";
        let score = analyzer.analyze(text);

        assert!(score.growth < 0.0, "Recession text should have negative growth score");
    }

    #[test]
    fn test_score_clamping() {
        let analyzer = make_analyzer();
        let score = analyzer.analyze("test");
        assert!(score.overall >= -1.0);
        assert!(score.overall <= 1.0);
        assert!(score.inflation >= -1.0);
        assert!(score.inflation <= 1.0);
    }

    #[test]
    fn test_confidence_low_for_short_text() {
        let analyzer = make_analyzer();
        let score = analyzer.analyze("The Committee met today.");
        assert!(score.confidence < 0.5, "Short text should have low confidence");
    }

    #[test]
    fn test_confidence_high_for_detailed_text() {
        let analyzer = make_analyzer();
        let text = "The Committee decided to maintain the target range for the federal \
                   funds rate at 5.25 to 5.50 percent. Inflation remains elevated and \
                   the labor market continues to be tight. The Committee is strongly \
                   committed to returning inflation to its 2 percent objective. \
                   Economic activity has been expanding at a moderate pace. \
                   Job gains have been robust in recent months. The Committee will \
                   continue to assess incoming information and is prepared to adjust \
                   monetary policy as appropriate. The Committee is attentive to \
                   inflation risks and will act to ensure price stability.";
        let score = analyzer.analyze(text);
        assert!(score.confidence > 0.3, "Detailed text should have reasonable confidence, got {}", score.confidence);
    }

    #[test]
    fn test_build_time_series() {
        let analyzer = make_analyzer();

        let statements = vec![
            (chrono::NaiveDate::from_ymd_opt(2024, 3, 20).unwrap(), "inflation elevated, rate hike needed"),
            (chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(), "inflation moderating, hold steady"),
            (chrono::NaiveDate::from_ymd_opt(2024, 9, 18).unwrap(), "inflation at target, rate cut expected"),
        ];

        let series = analyzer.build_time_series(
            &statements.iter().map(|(d, t)| (*d, *t)).collect::<Vec<_>>()
        );

        assert_eq!(series.len(), 3);
        assert!(series[0].score.overall > 0.0);
        assert!(series[2].score.overall < 0.0);

        // First point should have no shift
        assert!(series[0].score.tone_shift.is_none());
        // Subsequent points should have shifts
        assert!(series[1].score.tone_shift.is_some());
        assert!(series[2].score.tone_shift.is_some());
    }

    #[test]
    fn test_keyword_counts() {
        let analyzer = make_analyzer();
        let text = "Additional rate increases may be appropriate. The Committee is \
                   prepared to lower the target range. Gradual easing expected.";
        let score = analyzer.analyze(text);
        assert!(score.hawkish_keyword_count > 0);
        assert!(score.dovish_keyword_count > 0);
    }

    #[test]
    fn test_no_statement_is_none_sentiment() {
        let analyzer = make_analyzer();
        let score = analyzer.analyze("");
        assert_eq!(score.overall, 0.0);
        assert_eq!(score.monetary_policy, MonetaryPolicy::Neutral);
    }
}
