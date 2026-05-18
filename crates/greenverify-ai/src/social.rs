use crate::types::{ESGRating, SocialMetric};
use std::collections::HashMap;

/// Social scoring engine.
#[derive(Debug, Clone)]
pub struct SocialScorer {
    industry_benchmarks: HashMap<String, SocialBenchmark>,
}

#[derive(Debug, Clone)]
pub struct SocialBenchmark {
    pub avg_satisfaction: f64,
    pub avg_diversity: f64,
    pub avg_turnover: f64,
    pub avg_gender_pay_gap: f64,
}

impl SocialScorer {
    pub fn new() -> Self {
        let mut benchmarks = HashMap::new();
        benchmarks.insert("technology".into(), SocialBenchmark {
            avg_satisfaction: 75.0,
            avg_diversity: 60.0,
            avg_turnover: 13.0,
            avg_gender_pay_gap: 15.0,
        });
        benchmarks.insert("finance".into(), SocialBenchmark {
            avg_satisfaction: 70.0,
            avg_diversity: 55.0,
            avg_turnover: 15.0,
            avg_gender_pay_gap: 20.0,
        });
        benchmarks.insert("manufacturing".into(), SocialBenchmark {
            avg_satisfaction: 65.0,
            avg_diversity: 40.0,
            avg_turnover: 18.0,
            avg_gender_pay_gap: 18.0,
        });
        benchmarks.insert("healthcare".into(), SocialBenchmark {
            avg_satisfaction: 70.0,
            avg_diversity: 50.0,
            avg_turnover: 20.0,
            avg_gender_pay_gap: 12.0,
        });
        Self { industry_benchmarks: benchmarks }
    }

    fn get_benchmark(&self, industry: &str) -> SocialBenchmark {
        self.industry_benchmarks.get(industry).cloned().unwrap_or(SocialBenchmark {
            avg_satisfaction: 70.0,
            avg_diversity: 50.0,
            avg_turnover: 15.0,
            avg_gender_pay_gap: 17.0,
        })
    }

    // ─── Workforce Diversity ───────────────────────────────────

    /// Score workforce diversity across multiple dimensions.
    pub fn diversity_score(
        &self,
        gender_pct: f64,      // % female in workforce
        minority_pct: f64,     // % underrepresented minorities
        leadership_diversity: f64, // 0-100 composite
        industry: &str,
    ) -> f64 {
        let gender_score = match gender_pct {
            g if g >= 45.0 => 90.0,
            g if g >= 40.0 => 80.0,
            g if g >= 35.0 => 70.0,
            g if g >= 25.0 => 55.0,
            _ => 40.0,
        };

        let minority_score = match minority_pct {
            m if m >= 40.0 => 90.0,
            m if m >= 30.0 => 80.0,
            m if m >= 20.0 => 65.0,
            m if m >= 10.0 => 50.0,
            _ => 35.0,
        };

        let benchmark = self.get_benchmark(industry);
        let relative_diversity = if benchmark.avg_diversity > 0.0 {
            (leadership_diversity / benchmark.avg_diversity * 50.0).min(50.0)
        } else {
            50.0
        };

        (gender_score * 0.35 + minority_score * 0.35 + relative_diversity * 0.30)
    }

    // ─── Employee Turnover & Satisfaction ──────────────────────

    /// Score employee satisfaction relative to industry.
    pub fn satisfaction_score(&self, satisfaction: f64, industry: &str) -> f64 {
        let benchmark = self.get_benchmark(industry);
        let relative = satisfaction / benchmark.avg_satisfaction;
        if relative >= 1.15 {
            90.0
        } else if relative >= 1.0 {
            75.0
        } else if relative >= 0.9 {
            60.0
        } else if relative >= 0.8 {
            45.0
        } else {
            30.0
        }
    }

    /// Score based on employee turnover rate.
    pub fn turnover_score(turnover_pct: f64, industry: &str) -> f64 {
        match turnover_pct {
            t if t <= 5.0 => 95.0,
            t if t <= 10.0 => 80.0,
            t if t <= 15.0 => 65.0,
            t if t <= 20.0 => 50.0,
            t if t <= 30.0 => 35.0,
            _ => 20.0,
        }
    }

    /// Combined workforce quality score.
    pub fn workforce_score(
        &self,
        satisfaction: f64,
        turnover_pct: f64,
        training_hours_per_employee: f64,
        industry: &str,
    ) -> f64 {
        let sat = self.satisfaction_score(satisfaction, industry) * 0.35;
        let turn = Self::turnover_score(turnover_pct, industry) * 0.35;
        let training = (training_hours_per_employee * 3.0).min(30.0); // max 30 points

        sat + turn + training
    }

    // ─── Community Impact ──────────────────────────────────────

    /// Score community investment and impact.
    pub fn community_impact_score(
        community_investment_pct: f64, // % of pre-tax profit
        volunteer_hours_per_employee: f64,
        has_community_programs: bool,
    ) -> f64 {
        let invest_score = match community_investment_pct {
            p if p >= 5.0 => 40.0,
            p if p >= 2.0 => 30.0,
            p if p >= 1.0 => 20.0,
            p if p >= 0.5 => 10.0,
            _ => 5.0,
        };

        let volunteer_score = (volunteer_hours_per_employee * 2.0).min(30.0);
        let program_score = if has_community_programs { 30.0 } else { 5.0 };

        invest_score + volunteer_score + program_score
    }

    // ─── Data Privacy & Security ───────────────────────────────

    /// Score data privacy and security practices.
    pub fn data_privacy_score(
        has_dpo: bool,             // Data Protection Officer
        has_encryption: bool,
        breach_count: u32,         // in last 3 years
        has_privacy_framework: bool,
        gdpr_compliant: bool,
    ) -> f64 {
        let mut score = 0.0;
        if has_dpo { score += 15.0; }
        if has_encryption { score += 20.0; }
        if has_privacy_framework { score += 20.0; }
        if gdpr_compliant { score += 20.0; }

        // Penalty for breaches
        let breach_penalty = (breach_count as f64 * 10.0).min(40.0);
        score -= breach_penalty;

        score.max(0.0)
    }

    // ─── Supply Chain Labor Standards ──────────────────────────

    /// Score supply chain labor standards.
    pub fn supply_chain_score(
        supplier_audit_pct: f64,     // % of suppliers audited
        living_wage_compliance: f64, // % of suppliers paying living wage
        child_labor_free: bool,
        has_supply_chain_policy: bool,
    ) -> f64 {
        let audit_score = (supplier_audit_pct * 0.3).min(30.0);
        let wage_score = (living_wage_compliance * 0.3).min(30.0);

        let mut score = audit_score + wage_score;
        if child_labor_free { score += 20.0; }
        if has_supply_chain_policy { score += 20.0; }

        score.min(100.0)
    }

    // ─── Human Rights Due Diligence ────────────────────────────

    /// Score human rights due diligence.
    pub fn human_rights_score(
        has_hr_policy: bool,
        has_grievance_mechanism: bool,
        conducts_impact_assessments: bool,
        hr_incidents: u32,
    ) -> f64 {
        let mut score = 0.0;
        if has_hr_policy { score += 30.0; }
        if has_grievance_mechanism { score += 25.0; }
        if conducts_impact_assessments { score += 25.0; }

        let incident_penalty = (hr_incidents as f64 * 10.0).min(30.0);
        score -= incident_penalty;

        score.max(0.0)
    }

    // ─── Composite Social Score ────────────────────────────────

    /// Compute overall social score.
    pub fn composite_social_score(
        &self,
        metrics: &SocialMetric,
        industry: &str,
    ) -> f64 {
        let diversity = metrics.diversity_score * 0.20;
        let satisfaction = self.satisfaction_score(metrics.employee_satisfaction, industry) * 0.20;
        let community = (metrics.community_investment * 4.0).min(20.0); // normalize to 0-20
        let privacy = metrics.data_privacy * 0.20;
        let labor = metrics.labor_practices * 0.10;
        let supply_chain = metrics.supply_chain_labor * 0.10;

        diversity + satisfaction + community + privacy + labor + supply_chain
    }

    /// Generate detailed social scoring breakdown.
    pub fn detailed_score(
        &self,
        metrics: &SocialMetric,
        industry: &str,
    ) -> SocialScoreBreakdown {
        let overall = self.composite_social_score(metrics, industry);
        let rating = ESGRating::from_score(overall);

        SocialScoreBreakdown {
            overall,
            rating,
            diversity: metrics.diversity_score,
            satisfaction: self.satisfaction_score(metrics.employee_satisfaction, industry),
            community: (metrics.community_investment * 4.0).min(20.0),
            data_privacy: metrics.data_privacy,
            labor_practices: metrics.labor_practices,
            supply_chain: metrics.supply_chain_labor,
        }
    }
}

impl Default for SocialScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Detailed social scoring breakdown.
#[derive(Debug, Clone)]
pub struct SocialScoreBreakdown {
    pub overall: f64,
    pub rating: ESGRating,
    pub diversity: f64,
    pub satisfaction: f64,
    pub community: f64,
    pub data_privacy: f64,
    pub labor_practices: f64,
    pub supply_chain: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> SocialMetric {
        SocialMetric {
            employee_satisfaction: 78.0,
            diversity_score: 72.0,
            labor_practices: 80.0,
            community_investment: 2.5,
            data_privacy: 85.0,
            supply_chain_labor: 70.0,
        }
    }

    #[test]
    fn test_diversity_score_high() {
        let scorer = SocialScorer::new();
        let score = scorer.diversity_score(48.0, 45.0, 80.0, "technology");
        assert!(score > 70.0);
    }

    #[test]
    fn test_diversity_score_low() {
        let scorer = SocialScorer::new();
        let score = scorer.diversity_score(15.0, 10.0, 30.0, "technology");
        assert!(score < 50.0);
    }

    #[test]
    fn test_satisfaction_score_above_avg() {
        let scorer = SocialScorer::new();
        let score = scorer.satisfaction_score(85.0, "technology");
        // 85 / 75 = 1.13 >= 1.0 -> 75
        assert_eq!(score, 75.0);
    }

    #[test]
    fn test_satisfaction_score_below_avg() {
        let scorer = SocialScorer::new();
        let score = scorer.satisfaction_score(50.0, "technology");
        // 50 / 75 = 0.67 < 0.8 -> 30
        assert_eq!(score, 30.0);
    }

    #[test]
    fn test_turnover_score() {
        assert_eq!(SocialScorer::turnover_score(3.0, "tech"), 95.0);
        assert_eq!(SocialScorer::turnover_score(12.0, "tech"), 65.0);
        assert_eq!(SocialScorer::turnover_score(35.0, "tech"), 20.0);
    }

    #[test]
    fn test_workforce_score() {
        let scorer = SocialScorer::new();
        let score = scorer.workforce_score(80.0, 8.0, 20.0, "technology");
        assert!(score > 50.0);
    }

    #[test]
    fn test_community_impact_high() {
        let score = SocialScorer::community_impact_score(5.0, 25.0, true);
        assert!(score > 80.0);
    }

    #[test]
    fn test_community_impact_low() {
        let score = SocialScorer::community_impact_score(0.1, 2.0, false);
        assert!(score < 20.0);
    }

    #[test]
    fn test_data_privacy_score_full() {
        let score = SocialScorer::data_privacy_score(true, true, 0, true, true);
        assert_eq!(score, 75.0);
    }

    #[test]
    fn test_data_privacy_score_with_breaches() {
        let score = SocialScorer::data_privacy_score(true, true, 3, true, true);
        // 75 - 30 = 45
        assert_eq!(score, 45.0);
    }

    #[test]
    fn test_data_privacy_score_minimal() {
        let score = SocialScorer::data_privacy_score(false, false, 0, false, false);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_supply_chain_score_full() {
        let score = SocialScorer::supply_chain_score(90.0, 85.0, true, true);
        assert!(score > 70.0);
    }

    #[test]
    fn test_supply_chain_score_minimal() {
        let score = SocialScorer::supply_chain_score(10.0, 10.0, false, false);
        // 3 + 3 = 6
        assert!((score - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_human_rights_score_full() {
        let score = SocialScorer::human_rights_score(true, true, true, 0);
        assert_eq!(score, 80.0);
    }

    #[test]
    fn test_human_rights_score_with_incidents() {
        let score = SocialScorer::human_rights_score(true, true, true, 5);
        // 80 - 30 = 50
        assert_eq!(score, 50.0);
    }

    #[test]
    fn test_composite_social_score() {
        let scorer = SocialScorer::new();
        let m = sample_metrics();
        let score = scorer.composite_social_score(&m, "technology");
        assert!(score > 0.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_detailed_score() {
        let scorer = SocialScorer::new();
        let m = sample_metrics();
        let breakdown = scorer.detailed_score(&m, "technology");
        assert!(breakdown.overall > 0.0);
        assert!(breakdown.diversity > 0.0);
        assert!(breakdown.satisfaction > 0.0);
    }

    #[test]
    fn test_default_scorer() {
        let scorer = SocialScorer::default();
        assert!(scorer.industry_benchmarks.contains_key("technology"));
    }

    #[test]
    fn test_diversity_unknown_industry() {
        let scorer = SocialScorer::new();
        let score = scorer.diversity_score(40.0, 30.0, 60.0, "unknown");
        assert!(score > 0.0);
    }
}
