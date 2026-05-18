use crate::types::*;
use serde::{Deserialize, Serialize};

/// Alert generated when risk conditions are met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    pub mineral: String,
    pub country: String,
    pub level: RiskLevel,
    pub composite: f64,
    pub message: String,
}

/// Calculate composite risk score from individual risk factors.
///
/// Weights: geopolitical×0.25, environmental×0.15, regulatory×0.20,
/// infrastructure×0.15, labor×0.10, market_concentration×0.15
///
/// Classification: ≥80 Critical, ≥60 High, ≥40 Medium, ≥20 Low, <20 Minimal
pub fn calculate_composite_risk(assessment: &RiskAssessment) -> (f64, RiskLevel, String) {
    let composite = assessment.geopolitical * 0.25
        + assessment.environmental * 0.15
        + assessment.regulatory * 0.20
        + assessment.infrastructure * 0.15
        + assessment.labor * 0.10
        + assessment.market_concentration * 0.15;

    let (level, description) = classify_risk(composite);
    (composite, level, description)
}

fn classify_risk(composite: f64) -> (RiskLevel, String) {
    if composite >= 80.0 {
        (
            RiskLevel::Critical,
            "Critical risk requiring immediate mitigation action.".to_string(),
        )
    } else if composite >= 60.0 {
        (
            RiskLevel::High,
            "High risk — active monitoring and contingency plans required.".to_string(),
        )
    } else if composite >= 40.0 {
        (
            RiskLevel::Medium,
            "Moderate risk — periodic review recommended.".to_string(),
        )
    } else if composite >= 20.0 {
        (
            RiskLevel::Low,
            "Low risk — standard operating procedures sufficient.".to_string(),
        )
    } else {
        (
            RiskLevel::Minimal,
            "Minimal risk — no special measures required.".to_string(),
        )
    }
}

/// Calculate the Herfindahl-Hirschman Index from market share fractions.
///
/// `market_shares` should sum to 1.0. Each share is squared after multiplying by 100.
/// HHI < 1500: unconcentrated, 1500–2500: moderate, > 2500: highly concentrated.
pub fn calculate_hhi(market_shares: &[f64]) -> HHIResult {
    let hhi: f64 = market_shares.iter().map(|s| (s * 100.0).powi(2)).sum();
    let hhi_u32 = hhi as u32;

    let (level, concentration) = if hhi_u32 < 1500 {
        ("unconcentrated", "Competitive market")
    } else if hhi_u32 <= 2500 {
        ("moderate_concentration", "Moderately concentrated")
    } else {
        ("highly_concentrated", "Oligopolistic market")
    };

    HHIResult {
        hhi: hhi_u32,
        level: level.to_string(),
        concentration: concentration.to_string(),
    }
}

/// Calculate the effective number of competitors from HHI.
/// Returns 10000 / HHI.
pub fn effective_number_of_competitors(hhi: u32) -> f64 {
    if hhi == 0 {
        return f64::INFINITY;
    }
    10000.0 / hhi as f64
}

/// Compare two risk results over time.
///
/// Returns +1 if risk got worse, -1 if risk improved, 0 if unchanged.
pub fn compare_risk_trend(current: &RiskResult, previous: &RiskResult) -> i8 {
    if current.composite > previous.composite + 0.5 {
        1 // worse
    } else if current.composite < previous.composite - 0.5 {
        -1 // better
    } else {
        0
    }
}

/// Generate a risk matrix: minerals × countries.
pub fn generate_risk_matrix(
    minerals: &[String],
    countries: &[String],
    risk_fn: impl Fn(&str, &str) -> RiskResult,
) -> Vec<Vec<RiskResult>> {
    minerals
        .iter()
        .map(|mineral| countries.iter().map(|country| risk_fn(mineral, country)).collect())
        .collect()
}

/// Generate risk alerts for results that are High or Critical, or where
/// composite score increased by more than 20 from previous.
pub fn generate_risk_alerts(results: &[(&str, &str, RiskResult, Option<&RiskResult>)]) -> Vec<RiskAlert> {
    results
        .iter()
        .filter_map(|(mineral, country, current, previous)| {
            let should_alert = current.level == RiskLevel::High
                || current.level == RiskLevel::Critical
                || previous
                    .map(|p| current.composite - p.composite > 20.0)
                    .unwrap_or(false);

            if should_alert {
                let msg = if current.level == RiskLevel::Critical {
                    format!(
                        "CRITICAL: {} supply from {} at risk score {:.1}",
                        mineral, country, current.composite
                    )
                } else if let Some(prev) = previous {
                    if current.composite - prev.composite > 20.0 {
                        format!(
                            "ESCALATION: {} risk for {} jumped {:.1} → {:.1}",
                            mineral,
                            country,
                            prev.composite,
                            current.composite
                        )
                    } else {
                        format!(
                            "WARNING: High risk for {} supply from {} ({:.1})",
                            mineral, country, current.composite
                        )
                    }
                } else {
                    format!(
                        "WARNING: High risk for {} supply from {} ({:.1})",
                        mineral, country, current.composite
                    )
                };
                Some(RiskAlert {
                    mineral: mineral.to_string(),
                    country: country.to_string(),
                    level: current.level,
                    composite: current.composite,
                    message: msg,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Calculate supply chain diversification score.
/// Returns a value from 0 to 1, where 1 is fully diversified.
pub fn diversification_score(market_shares: &[f64]) -> f64 {
    if market_shares.is_empty() {
        return 0.0;
    }
    let hhi = calculate_hhi(market_shares);
    // Normalize: 0 HHI → score 1.0, 10000 HHI → score 0.0
    1.0 - (hhi.hhi as f64 / 10000.0)
}

/// Identify the top N risk factors from an assessment.
pub fn top_risk_factors(assessment: &RiskAssessment, n: usize) -> Vec<(&str, f64)> {
    let mut factors: Vec<(&str, f64)> = vec![
        ("geopolitical", assessment.geopolitical),
        ("environmental", assessment.environmental),
        ("regulatory", assessment.regulatory),
        ("infrastructure", assessment.infrastructure),
        ("labor", assessment.labor),
        ("market_concentration", assessment.market_concentration),
    ];
    factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    factors.truncate(n);
    factors
}

/// Calculate weighted average risk across multiple assessments.
pub fn average_risk(assessments: &[RiskAssessment]) -> RiskAssessment {
    if assessments.is_empty() {
        return RiskAssessment::zero();
    }
    let n = assessments.len() as f64;
    RiskAssessment {
        geopolitical: assessments.iter().map(|a| a.geopolitical).sum::<f64>() / n,
        environmental: assessments.iter().map(|a| a.environmental).sum::<f64>() / n,
        regulatory: assessments.iter().map(|a| a.regulatory).sum::<f64>() / n,
        infrastructure: assessments.iter().map(|a| a.infrastructure).sum::<f64>() / n,
        labor: assessments.iter().map(|a| a.labor).sum::<f64>() / n,
        market_concentration: assessments.iter().map(|a| a.market_concentration).sum::<f64>() / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_assessment() -> RiskAssessment {
        RiskAssessment {
            geopolitical: 60.0,
            environmental: 40.0,
            regulatory: 70.0,
            infrastructure: 30.0,
            labor: 20.0,
            market_concentration: 80.0,
        }
    }

    #[test]
    fn test_composite_risk_critical() {
        let a = RiskAssessment {
            geopolitical: 90.0,
            environmental: 90.0,
            regulatory: 90.0,
            infrastructure: 90.0,
            labor: 90.0,
            market_concentration: 90.0,
        };
        let (score, level, _) = calculate_composite_risk(&a);
        assert_eq!(level, RiskLevel::Critical);
        assert!((score - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_composite_risk_minimal() {
        let a = RiskAssessment::zero();
        let (score, level, _) = calculate_composite_risk(&a);
        assert_eq!(level, RiskLevel::Minimal);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_composite_risk_medium() {
        let a = RiskAssessment {
            geopolitical: 40.0,
            environmental: 40.0,
            regulatory: 40.0,
            infrastructure: 40.0,
            labor: 40.0,
            market_concentration: 40.0,
        };
        let (_, level, _) = calculate_composite_risk(&a);
        assert_eq!(level, RiskLevel::Medium);
    }

    #[test]
    fn test_composite_risk_high() {
        let a = RiskAssessment {
            geopolitical: 70.0,
            environmental: 70.0,
            regulatory: 70.0,
            infrastructure: 70.0,
            labor: 70.0,
            market_concentration: 70.0,
        };
        let (_, level, _) = calculate_composite_risk(&a);
        assert_eq!(level, RiskLevel::High);
    }

    #[test]
    fn test_composite_risk_low() {
        let a = RiskAssessment {
            geopolitical: 10.0,
            environmental: 10.0,
            regulatory: 10.0,
            infrastructure: 10.0,
            labor: 10.0,
            market_concentration: 10.0,
        };
        // All values at 10.0 → composite = 10.0 which is Minimal (< 20)
        let (_, level, _) = calculate_composite_risk(&a);
        assert_eq!(level, RiskLevel::Minimal);
    }

    #[test]
    fn test_hhi_unconcentrated() {
        // Equal shares among many competitors
        let shares = vec![0.1; 10]; // 10 competitors, each 10%
        let result = calculate_hhi(&shares);
        assert_eq!(result.hhi, 1000);
        assert_eq!(result.level, "unconcentrated");
    }

    #[test]
    fn test_hhi_moderate() {
        // HHI = 1225+625+400+100+100 = 2450 (moderate)
        let shares = vec![0.35, 0.25, 0.2, 0.1, 0.1];
        let result = calculate_hhi(&shares);
        assert!(result.hhi > 1500);
        assert!(result.hhi <= 2500);
    }

    #[test]
    fn test_hhi_monopoly() {
        let shares = vec![1.0]; // Single company
        let result = calculate_hhi(&shares);
        assert_eq!(result.hhi, 10000);
        assert_eq!(result.level, "highly_concentrated");
    }

    #[test]
    fn test_effective_competitors() {
        let competitors = effective_number_of_competitors(2500);
        assert!((competitors - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_effective_competitors_zero_hhi() {
        let competitors = effective_number_of_competitors(0);
        assert!(competitors.is_infinite());
    }

    #[test]
    fn test_compare_risk_worse() {
        let current = RiskResult {
            composite: 70.0,
            level: RiskLevel::High,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let previous = RiskResult {
            composite: 40.0,
            level: RiskLevel::Medium,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        assert_eq!(compare_risk_trend(&current, &previous), 1);
    }

    #[test]
    fn test_compare_risk_better() {
        let current = RiskResult {
            composite: 30.0,
            level: RiskLevel::Low,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let previous = RiskResult {
            composite: 60.0,
            level: RiskLevel::High,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        assert_eq!(compare_risk_trend(&current, &previous), -1);
    }

    #[test]
    fn test_compare_risk_same() {
        let current = RiskResult {
            composite: 50.0,
            level: RiskLevel::Medium,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let previous = RiskResult {
            composite: 50.0,
            level: RiskLevel::Medium,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        assert_eq!(compare_risk_trend(&current, &previous), 0);
    }

    #[test]
    fn test_risk_matrix_dimensions() {
        let minerals = vec!["Lithium".to_string(), "Cobalt".to_string()];
        let countries = vec!["AU".to_string(), "CD".to_string()];
        let matrix = generate_risk_matrix(&minerals, &countries, |_, _| RiskResult {
            composite: 50.0,
            level: RiskLevel::Medium,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        });
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_risk_alerts_critical() {
        let critical_result = RiskResult {
            composite: 85.0,
            level: RiskLevel::Critical,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let alerts = generate_risk_alerts(&[("Lithium", "CD", critical_result, None)]);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].level, RiskLevel::Critical);
    }

    #[test]
    fn test_risk_alerts_escalation() {
        let current = RiskResult {
            composite: 70.0,
            level: RiskLevel::Medium,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let previous = RiskResult {
            composite: 40.0,
            level: RiskLevel::Medium,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let alerts = generate_risk_alerts(&[("Cobalt", "AU", current, Some(&previous))]);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("ESCALATION"));
    }

    #[test]
    fn test_risk_alerts_no_alert() {
        let low_result = RiskResult {
            composite: 15.0,
            level: RiskLevel::Minimal,
            description: String::new(),
            breakdown: RiskAssessment::zero(),
        };
        let alerts = generate_risk_alerts(&[("Iron", "AU", low_result, None)]);
        assert_eq!(alerts.len(), 0);
    }

    #[test]
    fn test_diversification_score() {
        let shares = vec![1.0]; // Monopoly
        let score = diversification_score(&shares);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_diversification_score_competitive() {
        let shares = vec![0.1; 10];
        let score = diversification_score(&shares);
        assert!(score > 0.8);
    }

    #[test]
    fn test_top_risk_factors() {
        let a = sample_assessment();
        let top = top_risk_factors(&a, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "market_concentration"); // 80.0 is highest
    }

    #[test]
    fn test_average_risk() {
        let assessments = vec![
            RiskAssessment {
                geopolitical: 100.0,
                environmental: 0.0,
                regulatory: 100.0,
                infrastructure: 0.0,
                labor: 100.0,
                market_concentration: 0.0,
            },
            RiskAssessment {
                geopolitical: 0.0,
                environmental: 100.0,
                regulatory: 0.0,
                infrastructure: 100.0,
                labor: 0.0,
                market_concentration: 100.0,
            },
        ];
        let avg = average_risk(&assessments);
        assert!((avg.geopolitical - 50.0).abs() < 0.001);
        assert!((avg.environmental - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_average_risk_empty() {
        let avg = average_risk(&[]);
        assert!((avg.geopolitical - 0.0).abs() < 0.001);
    }
}
