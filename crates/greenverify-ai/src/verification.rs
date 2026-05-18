use crate::types::{ESGCategory, VerificationClaim, VerificationStatus};
use std::collections::HashMap;

/// Claim verification engine for greenwashing detection and ESG claim validation.
#[derive(Debug, Clone)]
pub struct VerificationEngine;

/// Result of a greenwashing risk assessment.
#[derive(Debug, Clone)]
pub struct GreenwashingAssessment {
    pub risk_score: f64,           // 0-100, higher = more risk
    pub risk_level: GreenwashingRiskLevel,
    pub flags: Vec<String>,
    pub claim_score_gap: f64,      // difference between claimed and actual
}

/// Greenwashing risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreenwashingRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Evidence quality assessment result.
#[derive(Debug, Clone)]
pub struct EvidenceQuality {
    pub score: f64,               // 0-100
    pub evidence_count: usize,
    pub source_diversity: f64,    // 0-1
    pub has_independent_sources: bool,
}

/// Cross-reference consistency result.
#[derive(Debug, Clone)]
pub struct ConsistencyCheck {
    pub consistent: bool,
    pub discrepancies: Vec<String>,
    pub alignment_score: f64,     // 0-100
}

/// Temporal consistency result.
#[derive(Debug, Clone)]
pub struct TemporalConsistency {
    pub improving: bool,
    pub trend: TemporalTrend,
    pub average_improvement: f64,
    pub score: f64,               // 0-100
}

/// Temporal trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalTrend {
    Improving,
    Stable,
    Declining,
}

/// Red flag detection result.
#[derive(Debug, Clone)]
pub struct RedFlagResult {
    pub has_red_flags: bool,
    pub flags: Vec<String>,
    pub severity: RedFlagSeverity,
}

/// Red flag severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedFlagSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Claim intensity parsed from text.
#[derive(Debug, Clone)]
pub struct ClaimIntensity {
    pub claimed_level: f64,  // 0-100, how strong the claim sounds
    pub keywords_found: Vec<String>,
}

impl VerificationEngine {
    pub fn new() -> Self {
        Self
    }

    // ─── Greenwashing Detection ────────────────────────────────

    /// Parse the intensity level of an environmental claim (0-100).
    pub fn parse_claim_intensity(claim_text: &str) -> ClaimIntensity {
        let lower = claim_text.to_lowercase();
        let mut score: f64 = 0.0;
        let mut keywords = Vec::new();

        // Strong claims
        let strong_words = [
            ("carbon neutral", 25.0),
            ("net zero", 25.0),
            ("zero emissions", 30.0),
            ("climate positive", 30.0),
            ("100% renewable", 25.0),
            ("fully sustainable", 25.0),
            ("completely green", 25.0),
        ];

        // Moderate claims
        let moderate_words = [
            ("reducing emissions", 15.0),
            ("sustainable", 15.0),
            ("green", 10.0),
            ("eco-friendly", 15.0),
            ("renewable", 15.0),
            ("low carbon", 15.0),
            ("carbon reduction", 15.0),
            ("climate target", 15.0),
        ];

        // Weak claims
        let weak_words = [
            ("committed to", 5.0),
            ("working towards", 5.0),
            ("plan to", 5.0),
            ("aiming to", 5.0),
            ("goal of", 5.0),
            ("aspiring to", 5.0),
        ];

        for (word, pts) in &strong_words {
            if lower.contains(word) {
                score += pts;
                keywords.push(word.to_string());
            }
        }
        for (word, pts) in &moderate_words {
            if lower.contains(word) {
                score += pts;
                keywords.push(word.to_string());
            }
        }
        for (word, pts) in &weak_words {
            if lower.contains(word) {
                score += pts;
                keywords.push(word.to_string());
            }
        }

        ClaimIntensity {
            claimed_level: score.min(100.0),
            keywords_found: keywords,
        }
    }

    /// Detect greenwashing risk: gap between claim intensity and actual ESG metrics.
    pub fn greenwashing_risk(
        claim_text: &str,
        actual_esg_score: f64,
        environmental_score: f64,
    ) -> GreenwashingAssessment {
        let intensity = Self::parse_claim_intensity(claim_text);
        let gap = (intensity.claimed_level - actual_esg_score).max(0.0);

        // Environmental claims with low env score are extra suspicious
        let env_gap = if claim_text.to_lowercase().contains("carbon")
            || claim_text.to_lowercase().contains("emissions")
            || claim_text.to_lowercase().contains("climate")
            || claim_text.to_lowercase().contains("renewable")
        {
            (intensity.claimed_level - environmental_score).max(0.0)
        } else {
            0.0
        };

        let mut risk_score: f64 = 0.0;
        let mut flags = Vec::new();

        // Base risk from claim-actual gap
        if gap > 50.0 {
            risk_score += 40.0;
            flags.push("Large gap between claims and actual ESG score".into());
        } else if gap > 30.0 {
            risk_score += 25.0;
            flags.push("Moderate gap between claims and actual ESG score".into());
        } else if gap > 15.0 {
            risk_score += 10.0;
            flags.push("Small gap between claims and actual ESG score".into());
        }

        // Environmental-specific risk
        if env_gap > 40.0 {
            risk_score += 30.0;
            flags.push("Environmental claims significantly exceed environmental performance".into());
        } else if env_gap > 20.0 {
            risk_score += 15.0;
        }

        // Weak language with low scores is a red flag
        let has_weak = intensity.keywords_found.iter().any(|k| {
            k == "committed to" || k == "working towards" || k == "plan to"
        });
        if has_weak && actual_esg_score < 40.0 {
            risk_score += 15.0;
            flags.push("Aspirational language with poor actual performance".into());
        }

        let risk_level = match risk_score {
            r if r >= 60.0 => GreenwashingRiskLevel::Critical,
            r if r >= 40.0 => GreenwashingRiskLevel::High,
            r if r >= 20.0 => GreenwashingRiskLevel::Medium,
            _ => GreenwashingRiskLevel::Low,
        };

        GreenwashingAssessment {
            risk_score: risk_score.min(100.0),
            risk_level,
            flags,
            claim_score_gap: gap,
        }
    }

    // ─── Evidence Quality Assessment ───────────────────────────

    /// Assess evidence quality for a claim.
    pub fn evidence_quality(
        evidence_urls: &[String],
        has_independent_verification: bool,
    ) -> EvidenceQuality {
        let evidence_count = evidence_urls.len();
        let source_diversity = Self::compute_source_diversity(evidence_urls);

        let count_score = (evidence_count as f64 * 10.0).min(30.0);
        let diversity_score = source_diversity * 30.0;
        let independent_score = if has_independent_verification { 40.0 } else { 5.0 };

        let score = count_score + diversity_score + independent_score;

        EvidenceQuality {
            score: score.min(100.0),
            evidence_count,
            source_diversity,
            has_independent_sources: has_independent_verification,
        }
    }

    /// Compute how diverse evidence sources are (0-1).
    /// Different domains count as diverse sources.
    fn compute_source_diversity(urls: &[String]) -> f64 {
        if urls.is_empty() {
            return 0.0;
        }

        let mut domains: std::collections::HashSet<String> = std::collections::HashSet::new();
        for url in urls {
            let domain = url
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split('/')
                .next()
                .unwrap_or("")
                .to_string();
            domains.insert(domain);
        }

        // diversity = unique domains / total urls (capped at 1.0)
        (domains.len() as f64 / urls.len() as f64).min(1.0)
    }

    // ─── Cross-Reference Consistency ───────────────────────────

    /// Check consistency between claims and computed scores.
    pub fn cross_reference_consistency(
        claims: &[(ESGCategory, &str, f64)], // (category, claim, claimed_score)
        actual_scores: &HashMap<ESGCategory, f64>,
    ) -> ConsistencyCheck {
        let mut discrepancies = Vec::new();
        let mut total_alignment = 0.0;
        let count = claims.len().max(1) as f64;

        for (category, claim, claimed_score) in claims {
            let actual = actual_scores.get(category).copied().unwrap_or(0.0);
            let diff = (claimed_score - actual).abs();

            if diff > 20.0 {
                discrepancies.push(format!(
                    "Claim '{}' (claimed score: {:.1}) has {:.1} point gap with actual ({:.1})",
                    claim, claimed_score, diff, actual
                ));
            }

            // Alignment: closer to actual = higher
            let alignment = (100.0 - diff).max(0.0);
            total_alignment += alignment;
        }

        let alignment_score = total_alignment / count;
        let consistent = discrepancies.is_empty() && alignment_score >= 70.0;

        ConsistencyCheck {
            consistent,
            discrepancies,
            alignment_score,
        }
    }

    // ─── Temporal Consistency ──────────────────────────────────

    /// Check if metrics are improving over time.
    pub fn temporal_consistency(historical_scores: &[f64]) -> TemporalConsistency {
        if historical_scores.is_empty() {
            return TemporalConsistency {
                improving: false,
                trend: TemporalTrend::Stable,
                average_improvement: 0.0,
                score: 50.0,
            };
        }

        if historical_scores.len() == 1 {
            return TemporalConsistency {
                improving: false,
                trend: TemporalTrend::Stable,
                average_improvement: 0.0,
                score: 50.0,
            };
        }

        let mut improvements = Vec::new();
        for window in historical_scores.windows(2) {
            improvements.push(window[1] - window[0]);
        }

        let avg_improvement = improvements.iter().sum::<f64>() / improvements.len() as f64;

        let (trend, score) = if avg_improvement > 3.0 {
            (TemporalTrend::Improving, (50.0 + avg_improvement * 5.0).min(100.0))
        } else if avg_improvement < -3.0 {
            (TemporalTrend::Declining, (50.0 + avg_improvement * 5.0).max(0.0))
        } else {
            (TemporalTrend::Stable, 50.0)
        };

        TemporalConsistency {
            improving: trend == TemporalTrend::Improving,
            trend,
            average_improvement: avg_improvement,
            score,
        }
    }

    // ─── Red Flag Detection ────────────────────────────────────

    /// Detect red flags that suggest greenwashing.
    pub fn detect_red_flags(
        actual_esg_score: f64,
        environmental_score: f64,
        claim_intensity: f64,
        has_temporal_improvement: bool,
        evidence_count: usize,
    ) -> RedFlagResult {
        let mut flags = Vec::new();
        let mut severity_score = 0.0;

        // High claims, low scores
        if claim_intensity > 60.0 && actual_esg_score < 40.0 {
            flags.push("Strong claims with low overall ESG score".into());
            severity_score += 25.0;
        }

        // Environmental claims with poor env performance
        if claim_intensity > 50.0 && environmental_score < 35.0 {
            flags.push("Environmental claims with poor environmental performance".into());
            severity_score += 30.0;
        }

        // No improvement
        if !has_temporal_improvement && claim_intensity > 40.0 {
            flags.push("Aspirational claims with no demonstrated improvement".into());
            severity_score += 20.0;
        }

        // Minimal evidence
        if evidence_count == 0 && claim_intensity > 20.0 {
            flags.push("Claims without supporting evidence".into());
            severity_score += 25.0;
        } else if evidence_count < 3 && claim_intensity > 40.0 {
            flags.push("Strong claims with insufficient evidence".into());
            severity_score += 10.0;
        }

        let severity = match severity_score {
            s if s >= 50.0 => RedFlagSeverity::Critical,
            s if s >= 30.0 => RedFlagSeverity::High,
            s if s >= 15.0 => RedFlagSeverity::Medium,
            s if s > 0.0 => RedFlagSeverity::Low,
            _ => RedFlagSeverity::None,
        };

        RedFlagResult {
            has_red_flags: !flags.is_empty(),
            flags,
            severity,
        }
    }

    // ─── Full Claim Verification ───────────────────────────────

    /// Verify a single claim against available evidence and scores.
    pub fn verify_claim(
        &self,
        claim: &mut VerificationClaim,
        actual_score: f64,
        evidence_urls: &[String],
    ) {
        let intensity = Self::parse_claim_intensity(&claim.claim);
        let gap = intensity.claimed_level - actual_score;

        claim.evidence_urls = evidence_urls.to_vec();

        if gap.abs() <= 15.0 && !evidence_urls.is_empty() {
            claim.status = VerificationStatus::Verified;
            claim.verifier = "greenverify-ai".into();
        } else if gap.abs() <= 30.0 {
            claim.status = VerificationStatus::Pending;
            claim.verifier = "greenverify-ai".into();
        } else if gap > 30.0 {
            claim.status = VerificationStatus::Failed;
            claim.verifier = "greenverify-ai".into();
        } else {
            // Negative gap means actual exceeds claims — verified
            claim.status = VerificationStatus::Verified;
            claim.verifier = "greenverify-ai".into();
        }
    }
}

impl Default for VerificationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_claim_intensity_strong() {
        let ci = VerificationEngine::parse_claim_intensity("We are carbon neutral and net zero");
        assert!(ci.claimed_level >= 50.0);
        assert!(ci.keywords_found.contains(&"carbon neutral".into()));
        assert!(ci.keywords_found.contains(&"net zero".into()));
    }

    #[test]
    fn test_parse_claim_intensity_moderate() {
        let ci = VerificationEngine::parse_claim_intensity("We are reducing emissions and using renewable energy");
        assert!(ci.claimed_level >= 20.0);
    }

    #[test]
    fn test_parse_claim_intensity_weak() {
        let ci = VerificationEngine::parse_claim_intensity("We are committed to sustainability");
        assert!(ci.claimed_level >= 5.0);
        assert!(ci.keywords_found.contains(&"committed to".into()));
    }

    #[test]
    fn test_parse_claim_intensity_none() {
        let ci = VerificationEngine::parse_claim_intensity("We sell products");
        assert_eq!(ci.claimed_level, 0.0);
        assert!(ci.keywords_found.is_empty());
    }

    #[test]
    fn test_greenwashing_risk_low() {
        let result = VerificationEngine::greenwashing_risk("committed to reducing emissions", 75.0, 70.0);
        assert_eq!(result.risk_level, GreenwashingRiskLevel::Low);
    }

    #[test]
    fn test_greenwashing_risk_high() {
        let result = VerificationEngine::greenwashing_risk(
            "We are carbon neutral and 100% renewable", 20.0, 15.0
        );
        assert!(matches!(result.risk_level, GreenwashingRiskLevel::High | GreenwashingRiskLevel::Critical));
    }

    #[test]
    fn test_greenwashing_risk_claims_gap() {
        let result = VerificationEngine::greenwashing_risk(
            "We are carbon neutral", 5.0, 3.0
        );
        assert!(result.claim_score_gap > 0.0);
        assert!(!result.flags.is_empty());
    }

    #[test]
    fn test_evidence_quality_full() {
        let urls = vec![
            "https://source1.com/report".into(),
            "https://source2.org/data".into(),
            "https://source3.net/study".into(),
        ];
        let eq = VerificationEngine::evidence_quality(&urls, true);
        assert!(eq.score > 50.0);
        assert_eq!(eq.evidence_count, 3);
        assert!(eq.has_independent_sources);
    }

    #[test]
    fn test_evidence_quality_none() {
        let eq = VerificationEngine::evidence_quality(&[], false);
        assert_eq!(eq.score, 5.0);
        assert_eq!(eq.source_diversity, 0.0);
    }

    #[test]
    fn test_evidence_quality_same_source() {
        let urls = vec![
            "https://same.com/a".into(),
            "https://same.com/b".into(),
        ];
        let eq = VerificationEngine::evidence_quality(&urls, false);
        assert_eq!(eq.source_diversity, 0.5);
    }

    #[test]
    fn test_source_diversity() {
        let urls = vec![
            "https://a.com/1".into(),
            "https://b.com/2".into(),
        ];
        assert_eq!(VerificationEngine::compute_source_diversity(&urls), 1.0);
    }

    #[test]
    fn test_cross_reference_consistent() {
        let claims = vec![
            (ESGCategory::Environmental, "Good environmental score", 72.0),
            (ESGCategory::Social, "Good social score", 68.0),
        ];
        let mut actual = HashMap::new();
        actual.insert(ESGCategory::Environmental, 70.0);
        actual.insert(ESGCategory::Social, 70.0);

        let result = VerificationEngine::cross_reference_consistency(&claims, &actual);
        assert!(result.consistent);
        assert!(result.discrepancies.is_empty());
    }

    #[test]
    fn test_cross_reference_inconsistent() {
        let claims = vec![
            (ESGCategory::Environmental, "Great environmental score", 95.0),
        ];
        let mut actual = HashMap::new();
        actual.insert(ESGCategory::Environmental, 30.0);

        let result = VerificationEngine::cross_reference_consistency(&claims, &actual);
        assert!(!result.consistent);
        assert!(!result.discrepancies.is_empty());
    }

    #[test]
    fn test_temporal_consistency_improving() {
        let scores = vec![40.0, 50.0, 60.0, 70.0];
        let result = VerificationEngine::temporal_consistency(&scores);
        assert!(result.improving);
        assert_eq!(result.trend, TemporalTrend::Improving);
        assert!(result.average_improvement > 0.0);
    }

    #[test]
    fn test_temporal_consistency_declining() {
        let scores = vec![70.0, 60.0, 50.0, 40.0];
        let result = VerificationEngine::temporal_consistency(&scores);
        assert!(!result.improving);
        assert_eq!(result.trend, TemporalTrend::Declining);
        assert!(result.average_improvement < 0.0);
    }

    #[test]
    fn test_temporal_consistency_stable() {
        let scores = vec![50.0, 51.0, 49.0, 50.0];
        let result = VerificationEngine::temporal_consistency(&scores);
        assert_eq!(result.trend, TemporalTrend::Stable);
    }

    #[test]
    fn test_temporal_consistency_empty() {
        let result = VerificationEngine::temporal_consistency(&[]);
        assert_eq!(result.trend, TemporalTrend::Stable);
        assert!(!result.improving);
    }

    #[test]
    fn test_temporal_consistency_single() {
        let result = VerificationEngine::temporal_consistency(&[50.0]);
        assert_eq!(result.trend, TemporalTrend::Stable);
    }

    #[test]
    fn test_red_flags_clean() {
        let result = VerificationEngine::detect_red_flags(75.0, 70.0, 10.0, true, 5);
        assert!(!result.has_red_flags);
        assert_eq!(result.severity, RedFlagSeverity::None);
    }

    #[test]
    fn test_red_flags_high_claims_low_score() {
        let result = VerificationEngine::detect_red_flags(25.0, 20.0, 70.0, false, 0);
        assert!(result.has_red_flags);
        assert!(matches!(result.severity, RedFlagSeverity::High | RedFlagSeverity::Critical));
    }

    #[test]
    fn test_red_flags_no_evidence() {
        let result = VerificationEngine::detect_red_flags(30.0, 25.0, 50.0, false, 0);
        assert!(result.has_red_flags);
    }

    #[test]
    fn test_red_flags_no_improvement() {
        let result = VerificationEngine::detect_red_flags(50.0, 50.0, 50.0, false, 3);
        assert!(result.has_red_flags);
    }

    #[test]
    fn test_verify_claim_verified() {
        let engine = VerificationEngine::new();
        let mut claim = VerificationClaim::new("c1", "Acme", ESGCategory::Environmental, "committed to sustainability");
        engine.verify_claim(&mut claim, 15.0, &["https://evidence.com".into()]);
        // claim intensity ~5, actual 15, gap ~10, has evidence -> verified
        assert!(matches!(claim.status, VerificationStatus::Verified));
    }

    #[test]
    fn test_verify_claim_failed() {
        let engine = VerificationEngine::new();
        let mut claim = VerificationClaim::new("c2", "Acme", ESGCategory::Environmental, "We are carbon neutral and net zero");
        engine.verify_claim(&mut claim, 15.0, &[]);
        // claim intensity ~50, actual 15, gap ~35 -> failed
        assert!(matches!(claim.status, VerificationStatus::Failed));
    }

    #[test]
    fn test_verify_claim_pending() {
        let engine = VerificationEngine::new();
        let mut claim = VerificationClaim::new("c3", "Acme", ESGCategory::Environmental, "We are reducing emissions");
        engine.verify_claim(&mut claim, 20.0, &["https://evidence.com".into()]);
        // claim intensity ~15, actual 20, negative gap -> verified (actual exceeds claim)
        assert!(matches!(claim.status, VerificationStatus::Verified));
    }

    #[test]
    fn test_default_engine() {
        let _engine = VerificationEngine::default();
    }
}
