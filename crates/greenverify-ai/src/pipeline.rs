use crate::types::{ESGCategory, ESGRating, ESGReport, ESGScore, EnvironmentalMetric, SocialMetric, VerificationClaim};
use crate::environmental::EnvironmentalScorer;
use crate::social::SocialScorer;
use crate::governance::GovernanceScorer;
use crate::verification::VerificationEngine;
use crate::bonds::GreenBondAnalyzer;

/// Full ESG analysis pipeline orchestrating all scoring modules.
#[derive(Debug, Clone)]
pub struct ESGPipeline {
    environmental_scorer: EnvironmentalScorer,
    social_scorer: SocialScorer,
    governance_scorer: GovernanceScorer,
    verification_engine: VerificationEngine,
    bond_analyzer: GreenBondAnalyzer,
}

/// Input data for a single entity ESG analysis.
#[derive(Debug, Clone)]
pub struct EntityInput {
    pub name: String,
    pub industry: String,
    pub revenue: f64,
    pub employees: u32,
    pub recycling_rate: f64,
    pub environmental: EnvironmentalMetric,
    pub social: SocialMetric,
    // Governance inputs
    pub independent_ratio: f64,
    pub board_size: u32,
    pub ceo_to_median_pay_ratio: f64,
    pub auditor_name: String,
    pub has_cumulative_voting: bool,
    pub proxy_access_pct: f64,
    pub has_majority_voting: bool,
    pub has_say_on_pay: bool,
    pub dual_class_shares: bool,
    pub anti_corruption_text: String,
    pub disclosure_text: String,
    // Historical scores for temporal analysis
    pub historical_scores: Vec<f64>,
    // Claims
    pub claims: Vec<(ESGCategory, String)>,
}

impl EntityInput {
    /// Create a minimal input with required fields.
    pub fn new(name: &str, industry: &str) -> Self {
        Self {
            name: name.to_string(),
            industry: industry.to_string(),
            revenue: 0.0,
            employees: 0,
            recycling_rate: 0.0,
            environmental: EnvironmentalMetric::zero(),
            social: SocialMetric::zero(),
            independent_ratio: 0.0,
            board_size: 0,
            ceo_to_median_pay_ratio: 0.0,
            auditor_name: String::new(),
            has_cumulative_voting: false,
            proxy_access_pct: 0.0,
            has_majority_voting: false,
            has_say_on_pay: false,
            dual_class_shares: false,
            anti_corruption_text: String::new(),
            disclosure_text: String::new(),
            historical_scores: Vec::new(),
            claims: Vec::new(),
        }
    }
}

impl ESGPipeline {
    pub fn new() -> Self {
        Self {
            environmental_scorer: EnvironmentalScorer::new(),
            social_scorer: SocialScorer::new(),
            governance_scorer: GovernanceScorer::new(),
            verification_engine: VerificationEngine::new(),
            bond_analyzer: GreenBondAnalyzer::new(),
        }
    }

    // ─── Full ESG Scoring ──────────────────────────────────────

    /// Run full ESG analysis on a single entity.
    pub fn analyze_entity(&self, input: &EntityInput) -> ESGScore {
        let env_score = self.score_environmental(input);
        let soc_score = self.score_social(input);
        let gov_score = self.score_governance(input);

        let mut score = ESGScore::new(env_score, soc_score, gov_score);

        // Adjust confidence based on data completeness
        let confidence = self.compute_confidence(input);
        score.confidence = confidence;

        // Add extra factors
        score.weighted_factors.insert("temporal_consistency".into(),
            self.temporal_consistency_score(&input.historical_scores));
        score.weighted_factors.insert("data_completeness".into(),
            self.data_completeness_score(input));

        score
    }

    /// Score the environmental pillar.
    pub fn score_environmental(&self, input: &EntityInput) -> f64 {
        self.environmental_scorer.composite_environmental_score(
            &input.environmental,
            input.revenue,
            input.employees,
            &input.industry,
            input.recycling_rate,
        )
    }

    /// Score the social pillar.
    pub fn score_social(&self, input: &EntityInput) -> f64 {
        self.social_scorer.composite_social_score(&input.social, &input.industry)
    }

    /// Score the governance pillar.
    pub fn score_governance(&self, input: &EntityInput) -> f64 {
        let gov_metric = self.governance_scorer.composite_governance_score(
            input.independent_ratio,
            input.board_size,
            input.ceo_to_median_pay_ratio,
            &input.auditor_name,
            input.has_cumulative_voting,
            input.proxy_access_pct,
            input.has_majority_voting,
            input.has_say_on_pay,
            input.dual_class_shares,
            &input.anti_corruption_text,
            &input.disclosure_text,
        );
        gov_metric.average()
    }

    // ─── Confidence & Completeness ─────────────────────────────

    /// Compute confidence score based on data completeness.
    fn compute_confidence(&self, input: &EntityInput) -> f64 {
        let mut checks = 0;
        let mut total = 0;

        // Revenue
        total += 1;
        if input.revenue > 0.0 { checks += 1; }

        // Employees
        total += 1;
        if input.employees > 0 { checks += 1; }

        // Environmental data
        total += 6;
        let env = &input.environmental;
        if env.carbon_emissions > 0.0 { checks += 1; }
        if env.energy_consumption > 0.0 { checks += 1; }
        if env.water_usage > 0.0 { checks += 1; }
        if env.waste_generated > 0.0 { checks += 1; }
        if env.renewable_pct > 0.0 { checks += 1; }
        if env.biodiversity_impact > 0.0 { checks += 1; }

        // Social data
        total += 6;
        let soc = &input.social;
        if soc.employee_satisfaction > 0.0 { checks += 1; }
        if soc.diversity_score > 0.0 { checks += 1; }
        if soc.labor_practices > 0.0 { checks += 1; }
        if soc.community_investment > 0.0 { checks += 1; }
        if soc.data_privacy > 0.0 { checks += 1; }
        if soc.supply_chain_labor > 0.0 { checks += 1; }

        // Governance data
        total += 4;
        if input.independent_ratio > 0.0 { checks += 1; }
        if input.board_size > 0 { checks += 1; }
        if input.ceo_to_median_pay_ratio > 0.0 { checks += 1; }
        if !input.auditor_name.is_empty() { checks += 1; }

        // Historical data
        total += 1;
        if !input.historical_scores.is_empty() { checks += 1; }

        if total == 0 { return 0.5; }
        checks as f64 / total as f64
    }

    /// Score data completeness (0-100).
    fn data_completeness_score(&self, input: &EntityInput) -> f64 {
        self.compute_confidence(input) * 100.0
    }

    // ─── Temporal Consistency ──────────────────────────────────

    /// Compute temporal consistency score.
    fn temporal_consistency_score(&self, historical: &[f64]) -> f64 {
        let tc = VerificationEngine::temporal_consistency(historical);
        tc.score
    }

    // ─── ESG Rating Conversion ─────────────────────────────────

    /// Convert a numeric score (0-100) to an ESGRating.
    pub fn score_to_rating(score: f64) -> ESGRating {
        ESGRating::from_score(score)
    }

    /// Get all possible ESG ratings from best to worst.
    pub fn all_ratings() -> Vec<ESGRating> {
        vec![
            ESGRating::AAA,
            ESGRating::AA,
            ESGRating::A,
            ESGRating::BBB,
            ESGRating::BB,
            ESGRating::B,
            ESGRating::CCC,
            ESGRating::CC,
            ESGRating::C,
            ESGRating::D,
        ]
    }

    // ─── Multi-Entity Comparison ───────────────────────────────

    /// Analyze and compare multiple entities, returning ranked results.
    pub fn compare_entities(&self, inputs: &[EntityInput]) -> Vec<EntityRanking> {
        let mut rankings: Vec<EntityRanking> = inputs
            .iter()
            .map(|input| {
                let score = self.analyze_entity(input);
                EntityRanking {
                    name: input.name.clone(),
                    score: score.overall,
                    rating: score.rating,
                    environmental: score.environmental,
                    social: score.social,
                    governance: score.governance,
                    confidence: score.confidence,
                }
            })
            .collect();

        // Sort by overall score descending
        rankings.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        rankings
    }

    /// Get the top N entities by ESG score.
    pub fn top_entities(&self, inputs: &[EntityInput], n: usize) -> Vec<EntityRanking> {
        let rankings = self.compare_entities(inputs);
        rankings.into_iter().take(n).collect()
    }

    // ─── Report Generation ─────────────────────────────────────

    /// Generate a full ESG report for an entity.
    pub fn generate_report(&self, input: &EntityInput, period: &str) -> ESGReport {
        let scores = self.analyze_entity(input);

        // Verify claims
        let mut claims = Vec::new();
        for (i, (category, claim_text)) in input.claims.iter().enumerate() {
            let mut claim = VerificationClaim::new(
                &format!("{}_{}", input.name, i),
                &input.name,
                *category,
                claim_text,
            );
            let actual = match category {
                ESGCategory::Environmental => scores.environmental,
                ESGCategory::Social => scores.social,
                ESGCategory::Governance => scores.governance,
            };
            self.verification_engine.verify_claim(&mut claim, actual, &[]);
            claims.push(claim);
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        let env = &scores.environmental;
        let soc = &scores.social;
        let gov = &scores.governance;

        if *env < 50.0 {
            recommendations.push("Improve environmental performance: reduce carbon emissions and increase renewable energy usage.".into());
        }
        if *soc < 50.0 {
            recommendations.push("Strengthen social practices: improve diversity, employee satisfaction, and supply chain labor standards.".into());
        }
        if *gov < 50.0 {
            recommendations.push("Enhance governance: increase board independence, improve transparency, and strengthen anti-corruption policies.".into());
        }
        if *env >= 70.0 && *soc >= 70.0 && *gov >= 70.0 {
            recommendations.push("Strong ESG performance across all pillars. Maintain current practices and consider leadership initiatives.".into());
        } else if *env >= 50.0 && *soc >= 50.0 && *gov >= 50.0 {
            recommendations.push("Moderate ESG performance. Focus on areas below 70 to reach leadership levels.".into());
        }

        // Verification summary
        let verified_count = claims.iter()
            .filter(|c| c.status == crate::types::VerificationStatus::Verified)
            .count();
        let total_claims = claims.len();
        let verification_summary = if total_claims == 0 {
            "No claims to verify.".to_string()
        } else {
            format!(
                "{}/{} claims verified. Overall ESG rating: {} (score: {:.1}).",
                verified_count, total_claims, scores.rating, scores.overall
            )
        };

        ESGReport {
            entity: input.name.clone(),
            period: period.to_string(),
            scores,
            claims,
            recommendations,
            verification_summary,
        }
    }

    /// Generate comparison summary text.
    pub fn comparison_summary(&self, rankings: &[EntityRanking]) -> String {
        if rankings.is_empty() {
            return "No entities to compare.".into();
        }
        let mut lines = vec![format!("ESG Entity Comparison ({} entities)", rankings.len())];
        for (i, r) in rankings.iter().enumerate() {
            lines.push(format!(
                "  {}. {} — Score: {:.1} | Rating: {} | E: {:.1} S: {:.1} G: {:.1}",
                i + 1, r.name, r.score, r.rating, r.environmental, r.social, r.governance
            ));
        }
        lines.join("\n")
    }
}

impl Default for ESGPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Ranked entity result for comparison.
#[derive(Debug, Clone)]
pub struct EntityRanking {
    pub name: String,
    pub score: f64,
    pub rating: ESGRating,
    pub environmental: f64,
    pub social: f64,
    pub governance: f64,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline() -> ESGPipeline {
        ESGPipeline::new()
    }

    fn sample_input() -> EntityInput {
        let mut input = EntityInput::new("Acme Corp", "technology");
        input.revenue = 1e9;
        input.employees = 5000;
        input.recycling_rate = 70.0;
        input.environmental = EnvironmentalMetric {
            carbon_emissions: 50000.0,
            energy_consumption: 200000.0,
            water_usage: 100000.0,
            waste_generated: 5000.0,
            renewable_pct: 60.0,
            biodiversity_impact: 30.0,
        };
        input.social = SocialMetric {
            employee_satisfaction: 78.0,
            diversity_score: 72.0,
            labor_practices: 80.0,
            community_investment: 2.5,
            data_privacy: 85.0,
            supply_chain_labor: 70.0,
        };
        input.independent_ratio = 0.75;
        input.board_size = 10;
        input.ceo_to_median_pay_ratio = 50.0;
        input.auditor_name = "Deloitte".into();
        input.has_cumulative_voting = true;
        input.proxy_access_pct = 3.0;
        input.has_majority_voting = true;
        input.has_say_on_pay = true;
        input.anti_corruption_text = "Strong anti-corruption and whistleblower policies with ethics compliance".into();
        input.disclosure_text = "Comprehensive disclosure on climate, carbon, emissions, water, waste, biodiversity, governance, board".into();
        input.historical_scores = vec![40.0, 50.0, 60.0, 70.0];
        input.claims.push((ESGCategory::Environmental, "Committed to reducing carbon emissions".into()));
        input
    }

    #[test]
    fn test_analyze_entity() {
        let p = pipeline();
        let input = sample_input();
        let score = p.analyze_entity(&input);
        assert!(score.overall > 0.0);
        assert!(score.overall <= 100.0);
        assert!(score.environmental > 0.0);
        assert!(score.social > 0.0);
        assert!(score.governance > 0.0);
    }

    #[test]
    fn test_score_environmental() {
        let p = pipeline();
        let input = sample_input();
        let score = p.score_environmental(&input);
        assert!(score > 0.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_score_social() {
        let p = pipeline();
        let input = sample_input();
        let score = p.score_social(&input);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_governance() {
        let p = pipeline();
        let input = sample_input();
        let score = p.score_governance(&input);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_to_rating() {
        assert_eq!(ESGPipeline::score_to_rating(92.0), ESGRating::AAA);
        assert_eq!(ESGPipeline::score_to_rating(75.0), ESGRating::A);
        assert_eq!(ESGPipeline::score_to_rating(5.0), ESGRating::D);
    }

    #[test]
    fn test_all_ratings() {
        let ratings = ESGPipeline::all_ratings();
        assert_eq!(ratings.len(), 10);
        assert_eq!(ratings[0], ESGRating::AAA);
        assert_eq!(ratings[9], ESGRating::D);
    }

    #[test]
    fn test_compare_entities() {
        let p = pipeline();
        let input1 = sample_input();
        let input2 = {
            let mut inp = EntityInput::new("Beta Corp", "technology");
            inp.revenue = 1e9;
            inp.employees = 5000;
            inp.environmental = EnvironmentalMetric::zero();
            inp.social = SocialMetric::zero();
            inp
        };

        let rankings = p.compare_entities(&[input1, input2]);
        assert_eq!(rankings.len(), 2);
        // First entity should have higher score
        assert!(rankings[0].score >= rankings[1].score);
    }

    #[test]
    fn test_top_entities() {
        let p = pipeline();
        let inputs = (0..5).map(|i| {
            let mut inp = EntityInput::new(&format!("Entity {}", i), "technology");
            inp.revenue = 1e9;
            inp.employees = 1000;
            inp.environmental = EnvironmentalMetric::zero();
            inp.social = SocialMetric::zero();
            inp
        }).collect::<Vec<_>>();

        let top = p.top_entities(&inputs, 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_generate_report() {
        let p = pipeline();
        let input = sample_input();
        let report = p.generate_report(&input, "2024");
        assert_eq!(report.entity, "Acme Corp");
        assert_eq!(report.period, "2024");
        assert!(report.scores.overall > 0.0);
    }

    #[test]
    fn test_generate_report_recommendations() {
        let p = pipeline();
        let input = sample_input();
        let report = p.generate_report(&input, "2024");
        // With good scores, should have positive recommendation
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_generate_report_with_claims() {
        let p = pipeline();
        let input = sample_input();
        let report = p.generate_report(&input, "2024");
        assert_eq!(report.claims.len(), 1);
    }

    #[test]
    fn test_comparison_summary() {
        let p = pipeline();
        let rankings = p.compare_entities(&[sample_input()]);
        let summary = p.comparison_summary(&rankings);
        assert!(summary.contains("Acme Corp"));
    }

    #[test]
    fn test_comparison_summary_empty() {
        let p = pipeline();
        let summary = p.comparison_summary(&[]);
        assert_eq!(summary, "No entities to compare.");
    }

    #[test]
    fn test_entity_input_new() {
        let input = EntityInput::new("Test", "technology");
        assert_eq!(input.name, "Test");
        assert_eq!(input.revenue, 0.0);
    }

    #[test]
    fn test_default_pipeline() {
        let p = ESGPipeline::default();
        let input = EntityInput::new("Test", "technology");
        let _ = p.analyze_entity(&input);
    }

    #[test]
    fn test_confidence_with_full_data() {
        let p = pipeline();
        let input = sample_input();
        let score = p.analyze_entity(&input);
        assert!(score.confidence > 0.5);
    }

    #[test]
    fn test_confidence_with_minimal_data() {
        let p = pipeline();
        let input = EntityInput::new("Empty Corp", "technology");
        let score = p.analyze_entity(&input);
        assert!(score.confidence < 0.5);
    }
}
