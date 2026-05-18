use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ESG category classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ESGCategory {
    Environmental,
    Social,
    Governance,
}

impl std::fmt::Display for ESGCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ESGCategory::Environmental => write!(f, "Environmental"),
            ESGCategory::Social => write!(f, "Social"),
            ESGCategory::Governance => write!(f, "Governance"),
        }
    }
}

/// ESG rating scale (AAA = best, D = worst).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ESGRating {
    AAA,
    AA,
    A,
    BBB,
    BB,
    B,
    CCC,
    CC,
    C,
    D,
}

impl ESGRating {
    /// Convert rating to a numeric score (0-100).
    pub fn to_score(&self) -> f64 {
        match self {
            ESGRating::AAA => 95.0,
            ESGRating::AA => 85.0,
            ESGRating::A => 75.0,
            ESGRating::BBB => 65.0,
            ESGRating::BB => 55.0,
            ESGRating::B => 45.0,
            ESGRating::CCC => 35.0,
            ESGRating::CC => 25.0,
            ESGRating::C => 15.0,
            ESGRating::D => 5.0,
        }
    }

    /// Convert a numeric score (0-100) to a rating.
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 90.0 => ESGRating::AAA,
            s if s >= 80.0 => ESGRating::AA,
            s if s >= 70.0 => ESGRating::A,
            s if s >= 60.0 => ESGRating::BBB,
            s if s >= 50.0 => ESGRating::BB,
            s if s >= 40.0 => ESGRating::B,
            s if s >= 30.0 => ESGRating::CCC,
            s if s >= 20.0 => ESGRating::CC,
            s if s >= 10.0 => ESGRating::C,
            _ => ESGRating::D,
        }
    }

    /// Is this an investment-grade rating?
    pub fn is_investment_grade(&self) -> bool {
        matches!(self, ESGRating::AAA | ESGRating::AA | ESGRating::A | ESGRating::BBB)
    }
}

impl std::fmt::Display for ESGRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ESGRating::AAA => write!(f, "AAA"),
            ESGRating::AA => write!(f, "AA"),
            ESGRating::A => write!(f, "A"),
            ESGRating::BBB => write!(f, "BBB"),
            ESGRating::BB => write!(f, "BB"),
            ESGRating::B => write!(f, "B"),
            ESGRating::CCC => write!(f, "CCC"),
            ESGRating::CC => write!(f, "CC"),
            ESGRating::C => write!(f, "C"),
            ESGRating::D => write!(f, "D"),
        }
    }
}

/// Environmental metrics for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalMetric {
    pub carbon_emissions: f64,      // tonnes CO2e
    pub energy_consumption: f64,     // MWh
    pub water_usage: f64,           // cubic meters
    pub waste_generated: f64,       // tonnes
    pub renewable_pct: f64,         // 0-100
    pub biodiversity_impact: f64,   // 0-100 (higher = worse impact)
}

impl EnvironmentalMetric {
    pub fn zero() -> Self {
        Self {
            carbon_emissions: 0.0,
            energy_consumption: 0.0,
            water_usage: 0.0,
            waste_generated: 0.0,
            renewable_pct: 0.0,
            biodiversity_impact: 0.0,
        }
    }

    pub fn carbon_intensity_per_revenue(&self, revenue: f64) -> f64 {
        if revenue < 1e-9 {
            return f64::MAX;
        }
        self.carbon_emissions / (revenue / 1e6) // per million revenue
    }

    pub fn carbon_intensity_per_employee(&self, employees: u32) -> f64 {
        if employees == 0 {
            return f64::MAX;
        }
        self.carbon_emissions / employees as f64
    }

    pub fn energy_intensity(&self) -> f64 {
        if self.energy_consumption < 1e-9 {
            return f64::MAX;
        }
        self.carbon_emissions / self.energy_consumption // tCO2e/MWh
    }
}

/// Social metrics for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialMetric {
    pub employee_satisfaction: f64,   // 0-100
    pub diversity_score: f64,         // 0-100
    pub labor_practices: f64,         // 0-100
    pub community_investment: f64,    // % of pre-tax profit
    pub data_privacy: f64,           // 0-100
    pub supply_chain_labor: f64,     // 0-100
}

impl SocialMetric {
    pub fn zero() -> Self {
        Self {
            employee_satisfaction: 0.0,
            diversity_score: 0.0,
            labor_practices: 0.0,
            community_investment: 0.0,
            data_privacy: 0.0,
            supply_chain_labor: 0.0,
        }
    }

    pub fn average(&self) -> f64 {
        let sum = self.employee_satisfaction + self.diversity_score + self.labor_practices
            + self.data_privacy + self.supply_chain_labor;
        sum / 5.0
    }
}

/// Governance metrics for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceMetric {
    pub board_independence: f64,      // 0-100
    pub executive_compensation: f64,  // 0-100 (reasonableness score)
    pub audit_quality: f64,          // 0-100
    pub shareholder_rights: f64,     // 0-100
    pub anti_corruption: f64,        // 0-100
    pub transparency: f64,           // 0-100
}

impl GovernanceMetric {
    pub fn zero() -> Self {
        Self {
            board_independence: 0.0,
            executive_compensation: 0.0,
            audit_quality: 0.0,
            shareholder_rights: 0.0,
            anti_corruption: 0.0,
            transparency: 0.0,
        }
    }

    pub fn average(&self) -> f64 {
        let sum = self.board_independence + self.executive_compensation + self.audit_quality
            + self.shareholder_rights + self.anti_corruption + self.transparency;
        sum / 6.0
    }
}

/// Composite ESG score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ESGScore {
    pub overall: f64,
    pub environmental: f64,
    pub social: f64,
    pub governance: f64,
    pub rating: ESGRating,
    pub confidence: f64,
    pub weighted_factors: HashMap<String, f64>,
}

impl ESGScore {
    pub fn new(env: f64, soc: f64, gov: f64) -> Self {
        let overall = env * 0.35 + soc * 0.30 + gov * 0.35;
        let rating = ESGRating::from_score(overall);
        let mut factors = HashMap::new();
        factors.insert("environmental".to_string(), env);
        factors.insert("social".to_string(), soc);
        factors.insert("governance".to_string(), gov);
        Self {
            overall,
            environmental: env,
            social: soc,
            governance: gov,
            rating,
            confidence: 0.5,
            weighted_factors: factors,
        }
    }
}

/// Verification status for ESG claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Verified,
    Pending,
    Failed,
    Disputed,
    Expired,
}

impl std::fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationStatus::Verified => write!(f, "Verified"),
            VerificationStatus::Pending => write!(f, "Pending"),
            VerificationStatus::Failed => write!(f, "Failed"),
            VerificationStatus::Disputed => write!(f, "Disputed"),
            VerificationStatus::Expired => write!(f, "Expired"),
        }
    }
}

/// An ESG-related claim requiring verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationClaim {
    pub claim_id: String,
    pub entity: String,
    pub category: ESGCategory,
    pub claim: String,
    pub evidence_urls: Vec<String>,
    pub status: VerificationStatus,
    pub verifier: String,
    pub timestamp: String,
}

impl VerificationClaim {
    pub fn new(claim_id: &str, entity: &str, category: ESGCategory, claim: &str) -> Self {
        Self {
            claim_id: claim_id.to_string(),
            entity: entity.to_string(),
            category,
            claim: claim.to_string(),
            evidence_urls: Vec::new(),
            status: VerificationStatus::Pending,
            verifier: String::new(),
            timestamp: String::new(),
        }
    }
}

/// Green bond information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreenBond {
    pub issuer: String,
    pub amount: f64,
    pub maturity: String,
    pub use_of_proceeds: String,
    pub verified_pct: f64,
    pub green_projects: Vec<String>,
}

/// Carbon credit information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonCredit {
    pub credit_id: String,
    pub originator: String,
    pub amount: f64,
    pub vintage_year: u32,
    pub standard: String,
    pub verification_status: VerificationStatus,
    pub retirements: Vec<CarbonRetirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonRetirement {
    pub amount: f64,
    pub date: String,
    pub beneficiary: String,
}

/// Full ESG report for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ESGReport {
    pub entity: String,
    pub period: String,
    pub scores: ESGScore,
    pub claims: Vec<VerificationClaim>,
    pub recommendations: Vec<String>,
    pub verification_summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esg_category_display() {
        assert_eq!(ESGCategory::Environmental.to_string(), "Environmental");
        assert_eq!(ESGCategory::Social.to_string(), "Social");
        assert_eq!(ESGCategory::Governance.to_string(), "Governance");
    }

    #[test]
    fn test_esg_rating_to_score() {
        assert!((ESGRating::AAA.to_score() - 95.0).abs() < 1e-9);
        assert!((ESGRating::BBB.to_score() - 65.0).abs() < 1e-9);
        assert!((ESGRating::D.to_score() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_esg_rating_from_score() {
        assert_eq!(ESGRating::from_score(92.0), ESGRating::AAA);
        assert_eq!(ESGRating::from_score(82.0), ESGRating::AA);
        assert_eq!(ESGRating::from_score(72.0), ESGRating::A);
        assert_eq!(ESGRating::from_score(62.0), ESGRating::BBB);
        assert_eq!(ESGRating::from_score(52.0), ESGRating::BB);
        assert_eq!(ESGRating::from_score(42.0), ESGRating::B);
        assert_eq!(ESGRating::from_score(32.0), ESGRating::CCC);
        assert_eq!(ESGRating::from_score(22.0), ESGRating::CC);
        assert_eq!(ESGRating::from_score(12.0), ESGRating::C);
        assert_eq!(ESGRating::from_score(2.0), ESGRating::D);
    }

    #[test]
    fn test_esg_rating_investment_grade() {
        assert!(ESGRating::AAA.is_investment_grade());
        assert!(ESGRating::AA.is_investment_grade());
        assert!(ESGRating::A.is_investment_grade());
        assert!(ESGRating::BBB.is_investment_grade());
        assert!(!ESGRating::BB.is_investment_grade());
        assert!(!ESGRating::D.is_investment_grade());
    }

    #[test]
    fn test_esg_rating_display() {
        assert_eq!(ESGRating::AAA.to_string(), "AAA");
        assert_eq!(ESGRating::D.to_string(), "D");
    }

    #[test]
    fn test_esg_rating_ord() {
        assert!(ESGRating::AAA < ESGRating::A);
        assert!(ESGRating::BB < ESGRating::B);
    }

    #[test]
    fn test_environmental_metric_zero() {
        let m = EnvironmentalMetric::zero();
        assert_eq!(m.carbon_emissions, 0.0);
        assert_eq!(m.renewable_pct, 0.0);
    }

    #[test]
    fn test_carbon_intensity_per_revenue() {
        let m = EnvironmentalMetric {
            carbon_emissions: 1000.0,
            energy_consumption: 5000.0,
            water_usage: 10000.0,
            waste_generated: 500.0,
            renewable_pct: 50.0,
            biodiversity_impact: 20.0,
        };
        let intensity = m.carbon_intensity_per_revenue(50e6);
        // 1000 / 50 = 20
        assert!((intensity - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_carbon_intensity_zero_revenue() {
        let m = EnvironmentalMetric::zero();
        assert_eq!(m.carbon_intensity_per_revenue(0.0), f64::MAX);
    }

    #[test]
    fn test_carbon_intensity_per_employee() {
        let m = EnvironmentalMetric {
            carbon_emissions: 1000.0,
            energy_consumption: 5000.0,
            water_usage: 10000.0,
            waste_generated: 500.0,
            renewable_pct: 50.0,
            biodiversity_impact: 20.0,
        };
        assert!((m.carbon_intensity_per_employee(100) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_energy_intensity() {
        let m = EnvironmentalMetric {
            carbon_emissions: 1000.0,
            energy_consumption: 5000.0,
            water_usage: 10000.0,
            waste_generated: 500.0,
            renewable_pct: 50.0,
            biodiversity_impact: 20.0,
        };
        // 1000 / 5000 = 0.2
        assert!((m.energy_intensity() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_social_metric_average() {
        let m = SocialMetric {
            employee_satisfaction: 80.0,
            diversity_score: 70.0,
            labor_practices: 75.0,
            community_investment: 2.0,
            data_privacy: 85.0,
            supply_chain_labor: 65.0,
        };
        // (80+70+75+85+65) / 5 = 75.0
        assert!((m.average() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_governance_metric_average() {
        let m = GovernanceMetric {
            board_independence: 80.0,
            executive_compensation: 70.0,
            audit_quality: 85.0,
            shareholder_rights: 75.0,
            anti_corruption: 90.0,
            transparency: 80.0,
        };
        // (80+70+85+75+90+80) / 6 = 80.0
        assert!((m.average() - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_esg_score_new() {
        let score = ESGScore::new(80.0, 70.0, 75.0);
        // overall = 80*0.35 + 70*0.30 + 75*0.35 = 28 + 21 + 26.25 = 75.25
        assert!((score.overall - 75.25).abs() < 1e-9);
        assert_eq!(score.rating, ESGRating::A);
        assert_eq!(score.confidence, 0.5);
    }

    #[test]
    fn test_verification_claim_new() {
        let claim = VerificationClaim::new(
            "c1", "Acme Corp", ESGCategory::Environmental, "Carbon neutral by 2030"
        );
        assert_eq!(claim.status, VerificationStatus::Pending);
        assert!(claim.evidence_urls.is_empty());
    }

    #[test]
    fn test_verification_status_display() {
        assert_eq!(VerificationStatus::Verified.to_string(), "Verified");
        assert_eq!(VerificationStatus::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_serde_roundtrip_esg_score() {
        let score = ESGScore::new(80.0, 70.0, 75.0);
        let json = serde_json::to_string(&score).unwrap();
        let score2: ESGScore = serde_json::from_str(&json).unwrap();
        assert!((score.overall - score2.overall).abs() < 1e-9);
    }

    #[test]
    fn test_serde_roundtrip_carbon_credit() {
        let cc = CarbonCredit {
            credit_id: "cc1".into(),
            originator: "Forest Co".into(),
            amount: 10000.0,
            vintage_year: 2024,
            standard: "VCS".into(),
            verification_status: VerificationStatus::Verified,
            retirements: vec![CarbonRetirement {
                amount: 5000.0,
                date: "2024-01-01".into(),
                beneficiary: "Acme".into(),
            }],
        };
        let json = serde_json::to_string(&cc).unwrap();
        let cc2: CarbonCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(cc.credit_id, cc2.credit_id);
        assert_eq!(cc2.retirements.len(), 1);
    }

    #[test]
    fn test_serde_roundtrip_green_bond() {
        let gb = GreenBond {
            issuer: "World Bank".into(),
            amount: 500e6,
            maturity: "2030".into(),
            use_of_proceeds: "Renewable energy".into(),
            verified_pct: 95.0,
            green_projects: vec!["Solar Farm A".into(), "Wind Farm B".into()],
        };
        let json = serde_json::to_string(&gb).unwrap();
        let gb2: GreenBond = serde_json::from_str(&json).unwrap();
        assert_eq!(gb2.issuer, "World Bank");
        assert_eq!(gb2.green_projects.len(), 2);
    }
}
