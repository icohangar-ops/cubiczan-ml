use serde::{Deserialize, Serialize};

/// Classification of mineral prospectivity potential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProspectivityClass {
    Prime,
    Strong,
    Watch,
    Early,
}

impl std::fmt::Display for ProspectivityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProspectivityClass::Prime => write!(f, "Prime"),
            ProspectivityClass::Strong => write!(f, "Strong"),
            ProspectivityClass::Watch => write!(f, "Watch"),
            ProspectivityClass::Early => write!(f, "Early"),
        }
    }
}

/// Risk severity level for supply chain assessments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Minimal => write!(f, "Minimal"),
            RiskLevel::Low => write!(f, "Low"),
            RiskLevel::Medium => write!(f, "Medium"),
            RiskLevel::High => write!(f, "High"),
            RiskLevel::Critical => write!(f, "Critical"),
        }
    }
}

/// Direction of price movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceDirection {
    Up,
    Down,
    Flat,
}

impl std::fmt::Display for PriceDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PriceDirection::Up => write!(f, "Up"),
            PriceDirection::Down => write!(f, "Down"),
            PriceDirection::Flat => write!(f, "Flat"),
        }
    }
}

/// Evidence scores supporting a prospectivity assessment (0-100 each).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectivityEvidence {
    pub geology: f64,
    pub geochemistry: f64,
    pub geophysics: f64,
    pub infrastructure: f64,
    pub policy: f64,
}

impl ProspectivityEvidence {
    /// Create a new evidence set with all fields at given values.
    pub fn new(
        geology: f64,
        geochemistry: f64,
        geophysics: f64,
        infrastructure: f64,
        policy: f64,
    ) -> Self {
        Self {
            geology,
            geochemistry,
            geophysics,
            infrastructure,
            policy,
        }
    }

    /// All fields set to zero.
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Clamp all evidence fields to [0, 100].
    pub fn clamped(&self) -> Self {
        Self {
            geology: self.geology.clamp(0.0, 100.0),
            geochemistry: self.geochemistry.clamp(0.0, 100.0),
            geophysics: self.geophysics.clamp(0.0, 100.0),
            infrastructure: self.infrastructure.clamp(0.0, 100.0),
            policy: self.policy.clamp(0.0, 100.0),
        }
    }

    /// Return the field name with the lowest score.
    pub fn weakest_field(&self) -> &str {
        let min = self.geology
            .min(self.geochemistry)
            .min(self.geophysics)
            .min(self.infrastructure)
            .min(self.policy);
        if (self.geology - min).abs() < f64::EPSILON {
            "geology"
        } else if (self.geochemistry - min).abs() < f64::EPSILON {
            "geochemistry"
        } else if (self.geophysics - min).abs() < f64::EPSILON {
            "geophysics"
        } else if (self.infrastructure - min).abs() < f64::EPSILON {
            "infrastructure"
        } else {
            "policy"
        }
    }
}

/// A geographic zone assessed for mineral prospectivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectivityZone {
    pub id: String,
    pub name: String,
    pub region: String,
    pub country: String,
    pub mineral_id: String,
    pub deposit_model: String,
    pub evidence: ProspectivityEvidence,
    pub confidence: f64,
}

/// Result of prospectivity scoring for a single zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectivityResult {
    pub zone_id: String,
    pub score: f64,
    pub class: ProspectivityClass,
    pub limiting_factor: String,
}

/// Individual risk factor scores (0-100 each).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub geopolitical: f64,
    pub environmental: f64,
    pub regulatory: f64,
    pub infrastructure: f64,
    pub labor: f64,
    pub market_concentration: f64,
}

impl RiskAssessment {
    /// All fields set to zero.
    pub fn zero() -> Self {
        Self {
            geopolitical: 0.0,
            environmental: 0.0,
            regulatory: 0.0,
            infrastructure: 0.0,
            labor: 0.0,
            market_concentration: 0.0,
        }
    }
}

/// Result of a risk assessment including composite score and level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskResult {
    pub composite: f64,
    pub level: RiskLevel,
    pub description: String,
    pub breakdown: RiskAssessment,
}

/// Herfindahl-Hirschman Index result for market concentration analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HHIResult {
    pub hhi: u32,
    pub level: String,
    pub concentration: String,
}

/// A single price observation with optional trading volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub date: chrono::NaiveDate,
    pub price: f64,
    pub volume: Option<f64>,
}

/// The change between two consecutive price observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceChange {
    pub absolute: f64,
    pub percentage: f64,
    pub direction: PriceDirection,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prospectivity_class_display() {
        assert_eq!(ProspectivityClass::Prime.to_string(), "Prime");
        assert_eq!(ProspectivityClass::Strong.to_string(), "Strong");
        assert_eq!(ProspectivityClass::Watch.to_string(), "Watch");
        assert_eq!(ProspectivityClass::Early.to_string(), "Early");
    }

    #[test]
    fn test_risk_level_display() {
        assert_eq!(RiskLevel::Critical.to_string(), "Critical");
        assert_eq!(RiskLevel::Minimal.to_string(), "Minimal");
    }

    #[test]
    fn test_price_direction_display() {
        assert_eq!(PriceDirection::Up.to_string(), "Up");
        assert_eq!(PriceDirection::Down.to_string(), "Down");
        assert_eq!(PriceDirection::Flat.to_string(), "Flat");
    }

    #[test]
    fn test_evidence_new_and_zero() {
        let e = ProspectivityEvidence::new(10.0, 20.0, 30.0, 40.0, 50.0);
        assert_eq!(e.geology, 10.0);
        let z = ProspectivityEvidence::zero();
        assert_eq!(z.geology, 0.0);
        assert_eq!(z.policy, 0.0);
    }

    #[test]
    fn test_evidence_clamped() {
        let e = ProspectivityEvidence::new(-10.0, 50.0, 150.0, 0.0, 100.0);
        let c = e.clamped();
        assert_eq!(c.geology, 0.0);
        assert_eq!(c.geochemistry, 50.0);
        assert_eq!(c.geophysics, 100.0);
    }

    #[test]
    fn test_evidence_weakest_field() {
        let e = ProspectivityEvidence::new(80.0, 90.0, 20.0, 70.0, 60.0);
        assert_eq!(e.weakest_field(), "geophysics");
    }

    #[test]
    fn test_risk_assessment_zero() {
        let r = RiskAssessment::zero();
        assert_eq!(r.geopolitical, 0.0);
        assert_eq!(r.market_concentration, 0.0);
    }

    #[test]
    fn test_price_point_construction() {
        let pp = PricePoint {
            date: chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            price: 42.5,
            volume: Some(1000.0),
        };
        assert_eq!(pp.price, 42.5);
        assert!(pp.volume.is_some());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let evidence = ProspectivityEvidence::new(75.0, 80.0, 65.0, 50.0, 90.0);
        let json = serde_json::to_string(&evidence).unwrap();
        let deserialized: ProspectivityEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.geology, 75.0);
        assert_eq!(deserialized.policy, 90.0);
    }

    #[test]
    fn test_zone_construction() {
        let zone = ProspectivityZone {
            id: "Z1".to_string(),
            name: "Test Zone".to_string(),
            region: "West".to_string(),
            country: "US".to_string(),
            mineral_id: "LI".to_string(),
            deposit_model: "Brine".to_string(),
            evidence: ProspectivityEvidence::new(80.0, 70.0, 60.0, 50.0, 90.0),
            confidence: 0.85,
        };
        assert_eq!(zone.country, "US");
    }

    #[test]
    fn test_risk_result_construction() {
        let rr = RiskResult {
            composite: 65.0,
            level: RiskLevel::High,
            description: "test".to_string(),
            breakdown: RiskAssessment::zero(),
        };
        assert_eq!(rr.level, RiskLevel::High);
    }

    #[test]
    fn test_hhi_result_construction() {
        let hhi = HHIResult {
            hhi: 3200,
            level: "highly_concentrated".to_string(),
            concentration: "Oligopoly".to_string(),
        };
        assert_eq!(hhi.hhi, 3200);
    }

    #[test]
    fn test_price_change_construction() {
        let pc = PriceChange {
            absolute: 2.5,
            percentage: 5.0,
            direction: PriceDirection::Up,
        };
        assert_eq!(pc.direction, PriceDirection::Up);
    }

    #[test]
    fn test_prospectivity_class_equality() {
        assert_eq!(ProspectivityClass::Prime, ProspectivityClass::Prime);
        assert_ne!(ProspectivityClass::Prime, ProspectivityClass::Strong);
    }

    #[test]
    fn test_risk_level_ordering() {
        let levels = [
            RiskLevel::Minimal,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ];
        for i in 0..levels.len() - 1 {
            assert_ne!(levels[i], levels[i + 1]);
        }
    }
}
