use crate::types::{CarbonCredit, GreenBond, VerificationStatus};

/// Green bond and carbon credit analysis engine.
#[derive(Debug, Clone)]
pub struct GreenBondAnalyzer {
    green_categories: Vec<String>,
    recognized_standards: Vec<String>,
}

/// Green bond impact verification result.
#[derive(Debug, Clone)]
pub struct GreenBondAssessment {
    pub issuer: String,
    pub impact_score: f64,         // 0-100
    pub green_category_match: bool,
    pub verified_pct_score: f64,
    pub green_project_count: usize,
}

/// Greenium analysis result.
#[derive(Debug, Clone)]
pub struct GreeniumAnalysis {
    pub has_greenium: bool,
    pub greenium_basis_points: f64,
    pub significance: GreeniumSignificance,
}

/// Significance level of observed greenium.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreeniumSignificance {
    None,
    Low,
    Moderate,
    High,
}

/// Carbon credit quality assessment result.
#[derive(Debug, Clone)]
pub struct CarbonCreditQuality {
    pub credit_id: String,
    pub overall_score: f64,        // 0-100
    pub vintage_score: f64,
    pub standard_score: f64,
    pub verification_score: f64,
    pub quality_rating: CarbonCreditRating,
}

/// Carbon credit quality rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarbonCreditRating {
    Premium,
    High,
    Standard,
    Low,
    Poor,
}

/// Additionality verification result.
#[derive(Debug, Clone)]
pub struct AdditionalityResult {
    pub score: f64,                // 0-100
    pub likely_additional: bool,
    pub factors: Vec<String>,
}

/// Permanence risk assessment result.
#[derive(Debug, Clone)]
pub struct PermanenceRisk {
    pub score: f64,                // 0-100, higher = more risk
    pub risk_level: PermanenceRiskLevel,
    pub mitigation_factors: Vec<String>,
}

/// Permanence risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermanenceRiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

impl GreenBondAnalyzer {
    pub fn new() -> Self {
        Self {
            green_categories: vec![
                "renewable energy".into(),
                "solar".into(),
                "wind".into(),
                "energy efficiency".into(),
                "clean transportation".into(),
                "green building".into(),
                "water management".into(),
                "waste management".into(),
                "sustainable agriculture".into(),
                "biodiversity".into(),
                "pollution prevention".into(),
                "climate adaptation".into(),
                "sustainable forestry".into(),
            ],
            recognized_standards: vec![
                "vcs".into(),
                "gold standard".into(),
                "gs".into(),
                "acm0013".into(),
                "cdm".into(),
                "car".into(),
                "arb".into(),
                "aac".into(),
                "puro.earth".into(),
                "climate action reserve".into(),
                "american carbon registry".into(),
                "art".into(),
            ],
        }
    }

    // ─── Green Bond Impact Verification ────────────────────────

    /// Verify green bond impact by checking use-of-proceeds alignment.
    pub fn verify_green_bond_impact(&self, bond: &GreenBond) -> GreenBondAssessment {
        let lower_proceeds = bond.use_of_proceeds.to_lowercase();

        let mut matching_categories = 0;
        for cat in &self.green_categories {
            if lower_proceeds.contains(cat.as_str()) {
                matching_categories += 1;
            }
        }

        // Also check green_projects
        let mut project_matches = 0;
        for project in &bond.green_projects {
            let lower = project.to_lowercase();
            for cat in &self.green_categories {
                if lower.contains(cat.as_str()) {
                    project_matches += 1;
                    break;
                }
            }
        }

        let total_matches = matching_categories + project_matches;
        let green_category_match = total_matches > 0;

        let category_score = (total_matches as f64 * 15.0).min(50.0);
        let verified_score = (bond.verified_pct / 100.0) * 30.0;
        let project_score = (bond.green_projects.len() as f64 * 5.0).min(20.0);

        let impact_score = category_score + verified_score + project_score;

        GreenBondAssessment {
            issuer: bond.issuer.clone(),
            impact_score: impact_score.min(100.0),
            green_category_match,
            verified_pct_score: verified_score,
            green_project_count: bond.green_projects.len(),
        }
    }

    /// Check if a use-of-proceeds description aligns with any green category.
    pub fn is_green_use_of_proceeds(&self, use_of_proceeds: &str) -> bool {
        let lower = use_of_proceeds.to_lowercase();
        self.green_categories.iter().any(|cat| lower.contains(cat.as_str()))
    }

    // ─── Greenium Analysis ─────────────────────────────────────

    /// Analyze greenium (yield difference between green bond and benchmark).
    pub fn analyze_greenium(
        green_bond_yield: f64,
        benchmark_yield: f64,
    ) -> GreeniumAnalysis {
        // Greenium = green bond yield minus benchmark (negative means cheaper = greenium exists)
        let greenium = green_bond_yield - benchmark_yield;
        let basis_points = greenium * 10000.0; // convert decimal to bps

        let has_greenium = basis_points < -2.0; // more than 2bps greenium

        let significance = match basis_points {
            bp if bp <= -15.0 => GreeniumSignificance::High,
            bp if bp <= -5.0 => GreeniumSignificance::Moderate,
            bp if bp <= -2.0 => GreeniumSignificance::Low,
            _ => GreeniumSignificance::None,
        };

        GreeniumAnalysis {
            has_greenium,
            greenium_basis_points: basis_points,
            significance,
        }
    }

    // ─── Carbon Credit Quality Assessment ──────────────────────

    /// Assess carbon credit quality.
    pub fn assess_carbon_credit(&self, credit: &CarbonCredit) -> CarbonCreditQuality {
        let vintage_score = self.vintage_year_score(credit.vintage_year);
        let standard_score = self.standard_score(&credit.standard);
        let verification_score = self.verification_score(credit.verification_status);

        let overall = vintage_score * 0.25 + standard_score * 0.40 + verification_score * 0.35;

        let quality_rating = match overall {
            s if s >= 80.0 => CarbonCreditRating::Premium,
            s if s >= 65.0 => CarbonCreditRating::High,
            s if s >= 50.0 => CarbonCreditRating::Standard,
            s if s >= 30.0 => CarbonCreditRating::Low,
            _ => CarbonCreditRating::Poor,
        };

        CarbonCreditQuality {
            credit_id: credit.credit_id.clone(),
            overall_score: overall,
            vintage_score,
            standard_score,
            verification_score,
            quality_rating,
        }
    }

    /// Score based on credit vintage year (recent = better).
    fn vintage_year_score(&self, year: u32) -> f64 {
        // Assume current year context — use relative scoring
        // Credits issued within last 5 years are best
        match year {
            y if y >= 2023 => 95.0,
            y if y >= 2020 => 85.0,
            y if y >= 2017 => 70.0,
            y if y >= 2014 => 55.0,
            y if y >= 2010 => 40.0,
            _ => 20.0,
        }
    }

    /// Score based on carbon credit standard.
    fn standard_score(&self, standard: &str) -> f64 {
        let lower = standard.to_lowercase();

        // Tier 1: Most rigorous
        let tier1 = ["gold standard", "gs"];
        // Tier 2: Well-recognized
        let tier2 = ["vcs", "cdm", "climate action reserve", "car"];
        // Tier 3: Regional/other recognized
        let tier3 = ["arb", "aac", "puro", "art"];

        if tier1.iter().any(|t| lower.contains(t)) {
            95.0
        } else if tier2.iter().any(|t| lower.contains(t)) {
            80.0
        } else if tier3.iter().any(|t| lower.contains(t)) {
            65.0
        } else if self.recognized_standards.iter().any(|s| lower.contains(s.as_str())) {
            55.0
        } else if lower.is_empty() {
            10.0
        } else {
            40.0
        }
    }

    /// Score based on verification status.
    fn verification_score(&self, status: VerificationStatus) -> f64 {
        match status {
            VerificationStatus::Verified => 95.0,
            VerificationStatus::Pending => 50.0,
            VerificationStatus::Disputed => 20.0,
            VerificationStatus::Failed => 5.0,
            VerificationStatus::Expired => 30.0,
        }
    }

    // ─── Additionality Verification ────────────────────────────

    /// Score additionality — likelihood that the project wouldn't have happened otherwise.
    pub fn verify_additionality(
        &self,
        project_type: &str,
        region: &str,
        standard: &str,
        baseline_scenario_documented: bool,
        regulatory_surplus: bool,
    ) -> AdditionalityResult {
        let mut score = 0.0;
        let mut factors = Vec::new();

        // Project type scoring
        let project_lower = project_type.to_lowercase();
        let project_score = if project_lower.contains("renewable")
            || project_lower.contains("solar")
            || project_lower.contains("wind")
        {
            25.0
        } else if project_lower.contains("reforestation")
            || project_lower.contains("forestry")
        {
            20.0
        } else if project_lower.contains("efficiency")
            || project_lower.contains("methane")
        {
            22.0
        } else {
            15.0
        };
        score += project_score;

        // Region scoring (developing countries get higher additionality score)
        let region_lower = region.to_lowercase();
        let region_score = if region_lower.contains("africa")
            || region_lower.contains("southeast asia")
            || region_lower.contains("south asia")
            || region_lower.contains("latin america")
            || region_lower.contains("developing")
        {
            25.0
        } else if region_lower.contains("eastern europe")
            || region_lower.contains("middle east")
        {
            18.0
        } else {
            10.0
        };
        score += region_score;
        factors.push(format!("Region additionality score: {:.0}", region_score));

        // Standard score
        let std_score = self.standard_score(standard) * 0.20; // max ~19
        score += std_score;

        // Baseline documentation
        if baseline_scenario_documented {
            score += 15.0;
            factors.push("Baseline scenario documented".into());
        } else {
            factors.push("No baseline scenario documented".into());
        }

        // Regulatory surplus (project goes beyond regulation)
        if regulatory_surplus {
            score += 15.0;
            factors.push("Regulatory surplus demonstrated".into());
        } else {
            factors.push("May be required by regulation".into());
        }

        score = score.min(100.0);
        let likely_additional = score >= 55.0;

        AdditionalityResult {
            score,
            likely_additional,
            factors,
        }
    }

    // ─── Permanence Risk Scoring ───────────────────────────────

    /// Score permanence risk (higher = riskier).
    pub fn assess_permanence_risk(
        &self,
        credit_type: &str,
        standard: &str,
        has_buffer_pool: bool,
        has_insurance: bool,
    ) -> PermanenceRisk {
        let mut risk_score: f64 = 0.0;
        let mut mitigation_factors = Vec::new();

        // Credit type risk
        let type_lower = credit_type.to_lowercase();
        let type_risk = if type_lower.contains("nature")
            || type_lower.contains("forest")
            || type_lower.contains("reforestation")
            || type_lower.contains("afforestation")
        {
            60.0 // High inherent risk
        } else if type_lower.contains("soil")
            || type_lower.contains("agriculture")
        {
            50.0
        } else if type_lower.contains("renewable")
            || type_lower.contains("energy")
        {
            20.0 // Lower inherent risk
        } else if type_lower.contains("methane")
            || type_lower.contains("industrial")
        {
            30.0
        } else {
            40.0
        };
        risk_score += type_risk;

        // Standard-based buffer (better standards have lower risk)
        let std_lower = standard.to_lowercase();
        if std_lower.contains("gold standard") {
            risk_score -= 10.0;
        } else if std_lower.contains("vcs") {
            risk_score -= 5.0;
        }

        // Mitigation factors
        if has_buffer_pool {
            risk_score -= 15.0;
            mitigation_factors.push("Buffer pool in place".into());
        }
        if has_insurance {
            risk_score -= 10.0;
            mitigation_factors.push("Insurance coverage available".into());
        }

        risk_score = risk_score.clamp(0.0, 100.0);

        let risk_level = match risk_score {
            r if r >= 70.0 => PermanenceRiskLevel::VeryHigh,
            r if r >= 55.0 => PermanenceRiskLevel::High,
            r if r >= 40.0 => PermanenceRiskLevel::Medium,
            r if r >= 25.0 => PermanenceRiskLevel::Low,
            _ => PermanenceRiskLevel::VeryLow,
        };

        PermanenceRisk {
            score: risk_score,
            risk_level,
            mitigation_factors,
        }
    }
}

impl Default for GreenBondAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> GreenBondAnalyzer {
        GreenBondAnalyzer::new()
    }

    fn sample_bond() -> GreenBond {
        GreenBond {
            issuer: "World Bank".into(),
            amount: 500e6,
            maturity: "2030".into(),
            use_of_proceeds: "Renewable energy and energy efficiency projects".into(),
            verified_pct: 95.0,
            green_projects: vec!["Solar Farm Alpha".into(), "Wind Farm Beta".into()],
        }
    }

    fn sample_credit() -> CarbonCredit {
        CarbonCredit {
            credit_id: "cc-001".into(),
            originator: "Forest Co".into(),
            amount: 10000.0,
            vintage_year: 2024,
            standard: "VCS".into(),
            verification_status: VerificationStatus::Verified,
            retirements: vec![],
        }
    }

    #[test]
    fn test_verify_green_bond_impact() {
        let a = analyzer();
        let bond = sample_bond();
        let result = a.verify_green_bond_impact(&bond);
        assert!(result.green_category_match);
        assert!(result.impact_score > 0.0);
        assert_eq!(result.green_project_count, 2);
    }

    #[test]
    fn test_verify_green_bond_no_match() {
        let a = analyzer();
        let bond = GreenBond {
            issuer: "Bank".into(),
            amount: 100e6,
            maturity: "2028".into(),
            use_of_proceeds: "General corporate purposes".into(),
            verified_pct: 0.0,
            green_projects: vec![],
        };
        let result = a.verify_green_bond_impact(&bond);
        assert!(!result.green_category_match);
    }

    #[test]
    fn test_is_green_use_of_proceeds() {
        let a = analyzer();
        assert!(a.is_green_use_of_proceeds("Funding for renewable energy and solar power"));
        assert!(a.is_green_use_of_proceeds("sustainable forestry project"));
        assert!(!a.is_green_use_of_proceeds("General corporate spending"));
    }

    #[test]
    fn test_analyze_greenium_present() {
        // Green bond yields 3bps less than benchmark -> greenium exists
        let result = GreenBondAnalyzer::analyze_greenium(0.047, 0.050);
        assert!(result.has_greenium);
        assert!(result.greenium_basis_points < 0.0);
        assert!(matches!(result.significance, GreeniumSignificance::High));
    }

    #[test]
    fn test_analyze_greenium_none() {
        let result = GreenBondAnalyzer::analyze_greenium(0.050, 0.050);
        assert!(!result.has_greenium);
        assert_eq!(result.significance, GreeniumSignificance::None);
    }

    #[test]
    fn test_analyze_greenium_high() {
        let result = GreenBondAnalyzer::analyze_greenium(0.033, 0.050);
        assert!(result.has_greenium);
        assert_eq!(result.significance, GreeniumSignificance::High);
    }

    #[test]
    fn test_assess_carbon_credit_premium() {
        let a = analyzer();
        let credit = CarbonCredit {
            credit_id: "cc-p".into(),
            originator: "Org".into(),
            amount: 5000.0,
            vintage_year: 2024,
            standard: "Gold Standard".into(),
            verification_status: VerificationStatus::Verified,
            retirements: vec![],
        };
        let result = a.assess_carbon_credit(&credit);
        assert_eq!(result.quality_rating, CarbonCreditRating::Premium);
    }

    #[test]
    fn test_assess_carbon_credit_poor() {
        let a = analyzer();
        let credit = CarbonCredit {
            credit_id: "cc-bad".into(),
            originator: "Org".into(),
            amount: 1000.0,
            vintage_year: 2005,
            standard: "Unknown".into(),
            verification_status: VerificationStatus::Failed,
            retirements: vec![],
        };
        let result = a.assess_carbon_credit(&credit);
        assert_eq!(result.quality_rating, CarbonCreditRating::Poor);
    }

    #[test]
    fn test_vintage_year_score() {
        let a = analyzer();
        assert!(a.vintage_year_score(2024) > a.vintage_year_score(2015));
        assert!(a.vintage_year_score(2015) > a.vintage_year_score(2005));
    }

    #[test]
    fn test_standard_score() {
        let a = analyzer();
        assert!(a.standard_score("Gold Standard") > a.standard_score("VCS"));
        assert!(a.standard_score("VCS") > a.standard_score("Unknown"));
    }

    #[test]
    fn test_verification_score() {
        let a = analyzer();
        assert_eq!(a.verification_score(VerificationStatus::Verified), 95.0);
        assert_eq!(a.verification_score(VerificationStatus::Pending), 50.0);
        assert_eq!(a.verification_score(VerificationStatus::Failed), 5.0);
        assert_eq!(a.verification_score(VerificationStatus::Expired), 30.0);
    }

    #[test]
    fn test_verify_additionality_high() {
        let a = analyzer();
        let result = a.verify_additionality(
            "renewable energy solar",
            "Sub-Saharan Africa",
            "Gold Standard",
            true,
            true,
        );
        assert!(result.likely_additional);
        assert!(result.score > 50.0);
    }

    #[test]
    fn test_verify_additionality_low() {
        let a = analyzer();
        let result = a.verify_additionality(
            "generic project",
            "United States",
            "Unknown",
            false,
            false,
        );
        // May or may not be additional depending on scoring
        assert!(result.score < 60.0);
    }

    #[test]
    fn test_assess_permanence_risk_nature() {
        let a = analyzer();
        let result = a.assess_permanence_risk("nature-based reforestation", "VCS", false, false);
        assert_eq!(result.risk_level, PermanenceRiskLevel::High);
        assert!(result.mitigation_factors.is_empty());
    }

    #[test]
    fn test_assess_permanence_risk_renewable() {
        let a = analyzer();
        let result = a.assess_permanence_risk("renewable energy", "VCS", false, false);
        assert!(matches!(result.risk_level, PermanenceRiskLevel::Low | PermanenceRiskLevel::VeryLow));
    }

    #[test]
    fn test_assess_permanence_risk_mitigated() {
        let a = analyzer();
        let result = a.assess_permanence_risk("reforestation", "Gold Standard", true, true);
        assert!(result.mitigation_factors.len() >= 2);
        assert!(result.score < 60.0); // mitigations reduce risk
    }

    #[test]
    fn test_carbon_credit_quality_score_bounds() {
        let a = analyzer();
        let credit = sample_credit();
        let result = a.assess_carbon_credit(&credit);
        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 100.0);
    }

    #[test]
    fn test_default_analyzer() {
        let a = GreenBondAnalyzer::default();
        assert!(!a.green_categories.is_empty());
    }
}
