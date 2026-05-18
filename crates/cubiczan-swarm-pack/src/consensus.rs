//! LMSR consensus scoring and anti-sycophancy metrics.
//!
//! This module implements:
//! - Logarithmic Market Scoring Rule (LMSR) for consensus measurement
//! - Text similarity via Jaccard word overlap
//! - Anti-sycophancy risk assessment based on model diversity,
//!   position variance, and text similarity patterns

use crate::governance::compute_heterogeneity_score;
use crate::types::AgentVote;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// LMSR Scorer
// ---------------------------------------------------------------------------

/// Logarithmic Market Scoring Rule scorer for measuring consensus.
///
/// The LMSR provides a principled way to aggregate beliefs:
/// - `cost(q)` = b × ln(Σ exp(q_i / b)) — the cost function
/// - `price(q, i)` = exp(q_i / b) / Σ exp(q_j / b) — the market price
/// - `score_confidences(confidences)` — agreement × mean confidence
///
/// The `b` (liquidity) parameter controls price sensitivity.
/// Higher b = more liquid market (less sensitive to individual bets).
pub struct LMSRScorer {
    pub liquidity: f64,
}

impl LMSRScorer {
    /// Create a new LMSR scorer with the given liquidity parameter.
    ///
    /// Panics if liquidity <= 0.
    pub fn new(liquidity: f64) -> Self {
        assert!(liquidity > 0.0, "LMSR liquidity must be positive");
        LMSRScorer { liquidity }
    }

    /// Compute the cost function C(q) = b × ln(Σ exp(q_i / b)).
    ///
    /// This represents the total cost of purchasing the current
    /// quantity vector q from the automated market maker.
    pub fn cost(&self, quantities: &[f64]) -> f64 {
        if quantities.is_empty() {
            return 0.0;
        }

        let b = self.liquidity;
        let sum_exp: f64 = quantities
            .iter()
            .map(|&q| (q / b).exp())
            .sum();

        if sum_exp <= 0.0 {
            return 0.0;
        }

        b * sum_exp.ln()
    }

    /// Compute the market price for a specific outcome.
    ///
    /// price(q, i) = exp(q_i / b) / Σ exp(q_j / b)
    ///
    /// Returns the probability estimate for outcome i given the
    /// current quantity vector.
    pub fn price(&self, quantities: &[f64], outcome_idx: usize) -> f64 {
        if quantities.is_empty() || outcome_idx >= quantities.len() {
            return 0.0;
        }

        let b = self.liquidity;
        let qi_exp = (quantities[outcome_idx] / b).exp();
        let sum_exp: f64 = quantities.iter().map(|&q| (q / b).exp()).sum();

        if sum_exp <= 0.0 {
            return 0.0;
        }

        qi_exp / sum_exp
    }

    /// Compute a consensus score from a list of confidence values.
    ///
    /// The score is: agreement × mean_confidence
    /// where agreement = 1 - min(4 × variance, 1.0)
    ///
    /// High agreement + high confidence = high consensus.
    /// Low agreement (high variance) penalizes the score.
    pub fn score_consensus(&self, confidences: &[f64]) -> f64 {
        if confidences.is_empty() {
            return 0.0;
        }

        if confidences.len() == 1 {
            return confidences[0];
        }

        let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let variance = confidences
            .iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>()
            / confidences.len() as f64;

        let agreement = 1.0 - (4.0 * variance).min(1.0);
        agreement * mean
    }

    /// Compute the cost of moving from one quantity vector to another.
    ///
    /// This is the cost difference: C(q_new) - C(q_old).
    pub fn cost_difference(&self, q_old: &[f64], q_new: &[f64]) -> f64 {
        self.cost(q_new) - self.cost(q_old)
    }
}

// ---------------------------------------------------------------------------
// Text similarity
// ---------------------------------------------------------------------------

/// Compute pairwise Jaccard similarity of word sets for a list of texts.
///
/// Returns a matrix where result[i][j] is the Jaccard similarity
/// between texts[i] and texts[j].
pub fn compute_text_similarity(texts: &[&str]) -> Vec<Vec<f64>> {
    let n = texts.len();
    let word_sets: Vec<HashSet<&str>> = texts
        .iter()
        .map(|text| text.split_whitespace().collect())
        .collect();

    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let intersection = word_sets[i].intersection(&word_sets[j]).count();
            let union = word_sets[i].union(&word_sets[j]).count();

            matrix[i][j] = if union == 0 {
                1.0 // Both empty
            } else {
                intersection as f64 / union as f64
            };
        }
    }

    matrix
}

/// Compute the Jaccard similarity between two texts.
pub fn text_similarity(text_a: &str, text_b: &str) -> f64 {
    let set_a: HashSet<&str> = text_a.split_whitespace().collect();
    let set_b: HashSet<&str> = text_b.split_whitespace().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

// ---------------------------------------------------------------------------
// Sycophancy assessment
// ---------------------------------------------------------------------------

/// Risk level for sycophancy assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl RiskLevel {
    /// Classify a numeric risk score into a risk level.
    pub fn from_score(score: f64) -> RiskLevel {
        if score < 0.3 {
            RiskLevel::Low
        } else if score < 0.6 {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        }
    }
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "low"),
            RiskLevel::Medium => write!(f, "medium"),
            RiskLevel::High => write!(f, "high"),
        }
    }
}

/// Assessment of sycophancy risk in a group of agents.
#[derive(Debug, Clone)]
pub struct SycophancyAssessment {
    /// Model diversity score (from heterogeneity computation).
    pub model_diversity: f64,
    /// Variance in position embeddings (using hash as proxy).
    pub position_variance: f64,
    /// Average pairwise text similarity of positions.
    pub avg_text_similarity: f64,
    /// Overall sycophancy risk score (0.0 to 1.0).
    pub sycophancy_risk: f64,
    /// Classified risk level.
    pub risk_level: RiskLevel,
}

impl SycophancyAssessment {
    /// Create a new assessment with the given components.
    pub fn new(
        model_diversity: f64,
        position_variance: f64,
        avg_text_similarity: f64,
    ) -> Self {
        // Sycophancy risk is a weighted combination:
        // - Low model diversity → higher risk (weight: 0.4)
        // - Low position variance → higher risk (weight: 0.35)
        // - High text similarity → higher risk (weight: 0.25)
        let diversity_risk = 1.0 - model_diversity;
        let variance_risk = 1.0 - position_variance.clamp(0.0, 1.0);
        let similarity_risk = avg_text_similarity;

        let sycophancy_risk = 0.4 * diversity_risk + 0.35 * variance_risk + 0.25 * similarity_risk;
        let risk_level = RiskLevel::from_score(sycophancy_risk);

        SycophancyAssessment {
            model_diversity,
            position_variance,
            avg_text_similarity,
            sycophancy_risk,
            risk_level,
        }
    }
}

/// Assess the sycophancy risk for a group of agent votes.
///
/// Combines:
/// - Model diversity (from heterogeneity scoring)
/// - Position variance (using hash-based embedding proxy)
/// - Text similarity (pairwise Jaccard of reasoning/position texts)
pub fn assess_sycophancy_risk(
    model_names: &[&str],
    positions: &[&str],
    confidences: &[f64],
) -> SycophancyAssessment {
    // Model diversity from heterogeneity score
    let (model_diversity, _) = compute_heterogeneity_score(model_names);

    // Position variance using hash-based embedding proxy
    let position_variance = compute_position_variance(positions);

    // Average text similarity
    let avg_text_similarity = if positions.len() >= 2 {
        let sim_matrix = compute_text_similarity(positions);
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..sim_matrix.len() {
            for j in (i + 1)..sim_matrix.len() {
                total += sim_matrix[i][j];
                count += 1;
            }
        }
        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    SycophancyAssessment::new(model_diversity, position_variance, avg_text_similarity)
}

/// Compute position variance using hash-based embedding.
///
/// This is a lightweight proxy for semantic variance when true
/// embeddings are not available. We use the hash of each position
/// string as a 1-dimensional "embedding" and compute the variance.
fn compute_position_variance(positions: &[&str]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }

    if positions.len() == 1 {
        return 0.0;
    }

    // Use a simple hash-based approach
    let hashes: Vec<f64> = positions
        .iter()
        .map(|p| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            p.hash(&mut hasher);
            let hash_val = hasher.finish() as f64;
            // Normalize to [0, 1]
            (hash_val % 10000.0) / 10000.0
        })
        .collect();

    let mean = hashes.iter().sum::<f64>() / hashes.len() as f64;
    let variance = hashes
        .iter()
        .map(|h| (h - mean).powi(2))
        .sum::<f64>()
        / hashes.len() as f64;

    // Normalize variance to [0, 1] range
    // Max variance for uniform [0,1] is 1/12 ≈ 0.0833
    (variance * 12.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- LMSR Scorer tests --

    #[test]
    fn test_lmsr_new() {
        let scorer = LMSRScorer::new(10.0);
        assert!((scorer.liquidity - 10.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_lmsr_zero_liquidity() {
        let _ = LMSRScorer::new(0.0);
    }

    #[test]
    #[should_panic]
    fn test_lmsr_negative_liquidity() {
        let _ = LMSRScorer::new(-1.0);
    }

    #[test]
    fn test_lmsr_cost_empty() {
        let scorer = LMSRScorer::new(10.0);
        assert_eq!(scorer.cost(&[]), 0.0);
    }

    #[test]
    fn test_lmsr_cost_uniform() {
        let scorer = LMSRScorer::new(10.0);
        // All quantities equal → each price should be 1/n
        let quantities = vec![100.0, 100.0, 100.0];
        let _cost = scorer.cost(&quantities);
        let p0 = scorer.price(&quantities, 0);
        let p1 = scorer.price(&quantities, 1);
        let p2 = scorer.price(&quantities, 2);
        assert!((p0 - 1.0 / 3.0).abs() < 1e-10);
        assert!((p1 - 1.0 / 3.0).abs() < 1e-10);
        assert!((p2 - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lmsr_price_skewed() {
        let scorer = LMSRScorer::new(10.0);
        let quantities = vec![100.0, 0.0, 0.0];
        let p0 = scorer.price(&quantities, 0);
        assert!(p0 > 0.9); // Very high price for dominant outcome
    }

    #[test]
    fn test_lmsr_price_out_of_range() {
        let scorer = LMSRScorer::new(10.0);
        assert_eq!(scorer.price(&[1.0, 2.0], 5), 0.0);
    }

    #[test]
    fn test_lmsr_price_empty() {
        let scorer = LMSRScorer::new(10.0);
        assert_eq!(scorer.price(&[], 0), 0.0);
    }

    #[test]
    fn test_lmsr_score_consensus_empty() {
        let scorer = LMSRScorer::new(10.0);
        assert_eq!(scorer.score_consensus(&[]), 0.0);
    }

    #[test]
    fn test_lmsr_score_consensus_single() {
        let scorer = LMSRScorer::new(10.0);
        let score = scorer.score_consensus(&[0.8]);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_lmsr_score_consensus_high_agreement() {
        let scorer = LMSRScorer::new(10.0);
        let score = scorer.score_consensus(&[0.9, 0.9, 0.9, 0.9]);
        // mean = 0.9, variance = 0, agreement = 1.0
        // score = 1.0 * 0.9 = 0.9
        assert!((score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_lmsr_score_consensus_low_agreement() {
        let scorer = LMSRScorer::new(10.0);
        let score = scorer.score_consensus(&[0.0, 1.0]);
        // mean = 0.5, variance = 0.25, agreement = 1 - min(1.0, 1.0) = 0.0
        assert!(score < 0.1);
    }

    #[test]
    fn test_lmsr_score_consensus_moderate() {
        let scorer = LMSRScorer::new(10.0);
        let score = scorer.score_consensus(&[0.7, 0.8, 0.75]);
        // Should be high due to low variance
        assert!(score > 0.7);
    }

    #[test]
    fn test_lmsr_cost_difference() {
        let scorer = LMSRScorer::new(10.0);
        let q_old = vec![10.0, 10.0];
        let q_new = vec![20.0, 10.0];
        let diff = scorer.cost_difference(&q_old, &q_new);
        // Moving q[0] from 10 to 20 should cost something positive
        assert!(diff > 0.0);
    }

    // -- Text similarity tests --

    #[test]
    fn test_text_similarity_identical() {
        assert!((text_similarity("hello world", "hello world") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_text_similarity_disjoint() {
        assert!((text_similarity("hello world", "foo bar baz") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_text_similarity_partial() {
        let sim = text_similarity("the quick brown fox", "the quick blue fox");
        // intersection = {the, quick, fox} = 3, union = {the, quick, brown, fox, blue} = 5
        assert!((sim - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_text_similarity_empty() {
        assert!((text_similarity("", "") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_text_similarity_matrix() {
        let texts = vec!["hello world", "hello there", "foo bar"];
        let matrix = compute_text_similarity(&texts);

        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10); // self-similarity
        assert!(matrix[0][1] > 0.0); // "hello" shared
        assert!((matrix[0][2] - 0.0).abs() < 1e-10); // no shared words
    }

    // -- Sycophancy assessment tests --

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_score(0.1), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(0.29), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(0.3), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_score(0.59), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_score(0.6), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(1.0), RiskLevel::High);
    }

    #[test]
    fn test_assessment_new_high_diversity() {
        let assessment = SycophancyAssessment::new(0.9, 0.8, 0.2);
        assert!(assessment.sycophancy_risk < 0.3);
        assert_eq!(assessment.risk_level, RiskLevel::Low);
    }

    #[test]
    fn test_assessment_new_low_diversity() {
        let assessment = SycophancyAssessment::new(0.1, 0.1, 0.9);
        assert!(assessment.sycophancy_risk > 0.5);
        assert_eq!(assessment.risk_level, RiskLevel::High);
    }

    #[test]
    fn test_assess_sycophancy_diverse() {
        let models = ["gpt-4", "claude-3-opus", "llama-3.1-70b", "gemini-pro"];
        let positions = ["I think the answer is yes because X",
                         "The evidence suggests yes due to Y",
                         "Based on analysis, I agree with Z",
                         "Data indicates a positive outcome"];
        let confidences = [0.8, 0.7, 0.9, 0.75];

        let assessment = assess_sycophancy_risk(&models, &positions, &confidences);
        // High model diversity should reduce risk
        assert!(assessment.model_diversity > 0.5);
    }

    #[test]
    fn test_assess_sycophancy_monoculture() {
        let models = ["gpt-4", "gpt-4o", "gpt-4-turbo"];
        let positions = [
            "I agree with the previous assessment",
            "I concur with the above reasoning",
            "The prior analysis is correct",
        ];
        let confidences = [0.9, 0.9, 0.9];

        let assessment = assess_sycophancy_risk(&models, &positions, &confidences);
        // Low model diversity + similar text → higher risk
        assert!(assessment.sycophancy_risk > 0.3);
    }

    #[test]
    fn test_assess_sycophancy_empty() {
        let assessment = assess_sycophancy_risk(&[], &[], &[]);
        assert_eq!(assessment.model_diversity, 0.0);
    }

    #[test]
    fn test_position_variance_empty() {
        assert_eq!(compute_position_variance(&[]), 0.0);
    }

    #[test]
    fn test_position_variance_single() {
        assert_eq!(compute_position_variance(&["hello"]), 0.0);
    }

    #[test]
    fn test_risk_level_display() {
        assert_eq!(RiskLevel::Low.to_string(), "low");
        assert_eq!(RiskLevel::Medium.to_string(), "medium");
        assert_eq!(RiskLevel::High.to_string(), "high");
    }
}
