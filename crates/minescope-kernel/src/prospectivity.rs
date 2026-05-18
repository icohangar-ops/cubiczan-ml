use crate::types::*;

/// Calculate a weighted prospectivity score from evidence fields.
///
/// Weights: geology×0.30, geochemistry×0.25, geophysics×0.20, infrastructure×0.10, policy×0.15
///
/// Returns `(score, limiting_factor_name)` where the limiting factor is the
/// evidence field with the lowest contribution (weighted score).
pub fn calculate_prospectivity_score(evidence: &ProspectivityEvidence) -> (f64, String) {
    let weights = [
        ("geology", 0.30),
        ("geochemistry", 0.25),
        ("geophysics", 0.20),
        ("infrastructure", 0.10),
        ("policy", 0.15),
    ];
    let fields = [
        evidence.geology,
        evidence.geochemistry,
        evidence.geophysics,
        evidence.infrastructure,
        evidence.policy,
    ];

    let mut score = 0.0;
    let mut min_contribution = f64::INFINITY;
    let mut limiting = "geology".to_string();

    for (i, (name, weight)) in weights.iter().enumerate() {
        let contribution = fields[i] * weight;
        score += contribution;
        if contribution < min_contribution {
            min_contribution = contribution;
            limiting = name.to_string();
        }
    }

    (score, limiting)
}

/// Classify a prospectivity score into a discrete class.
///
/// - ≥ 80 → Prime
/// - ≥ 60 → Strong
/// - ≥ 40 → Watch
/// - < 40 → Early
pub fn classify_prospectivity(score: f64) -> ProspectivityClass {
    if score >= 80.0 {
        ProspectivityClass::Prime
    } else if score >= 60.0 {
        ProspectivityClass::Strong
    } else if score >= 40.0 {
        ProspectivityClass::Watch
    } else {
        ProspectivityClass::Early
    }
}

/// Score a single prospectivity zone and return the full result.
pub fn score_zone(zone: &ProspectivityZone) -> ProspectivityResult {
    let (score, limiting_factor) = calculate_prospectivity_score(&zone.evidence);
    let class = classify_prospectivity(score);
    ProspectivityResult {
        zone_id: zone.id.clone(),
        score,
        class,
        limiting_factor,
    }
}

/// Score multiple zones and return results sorted by score descending.
pub fn rank_zones(zones: &[ProspectivityZone]) -> Vec<ProspectivityResult> {
    let mut results: Vec<ProspectivityResult> = zones.iter().map(score_zone).collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Calculate the confidence-adjusted score by multiplying raw score by confidence.
pub fn confidence_adjusted_score(zone: &ProspectivityZone) -> f64 {
    let (score, _) = calculate_prospectivity_score(&zone.evidence);
    score * zone.confidence
}

/// Filter zones by minimum prospectivity class.
pub fn filter_by_class(zones: &[ProspectivityZone], min_class: ProspectivityClass) -> Vec<ProspectivityZone> {
    let class_order = |c: &ProspectivityClass| -> u8 {
        match c {
            ProspectivityClass::Early => 0,
            ProspectivityClass::Watch => 1,
            ProspectivityClass::Strong => 2,
            ProspectivityClass::Prime => 3,
        }
    };
    let min_order = class_order(&min_class);
    zones.iter()
        .filter(|z| {
            let (score, _) = calculate_prospectivity_score(&z.evidence);
            class_order(&classify_prospectivity(score)) >= min_order
        })
        .cloned()
        .collect()
}

/// Identify zones where a specific evidence field is the limiting factor.
pub fn zones_limited_by<'a>(zones: &'a [ProspectivityZone], field: &str) -> Vec<&'a ProspectivityZone> {
    zones.iter()
        .filter(|z| calculate_prospectivity_score(&z.evidence).1 == field)
        .collect()
}

/// Calculate the average prospectivity score across all zones.
pub fn average_prospectivity(zones: &[ProspectivityZone]) -> f64 {
    if zones.is_empty() {
        return 0.0;
    }
    let total: f64 = zones.iter()
        .map(|z| calculate_prospectivity_score(&z.evidence).0)
        .sum();
    total / zones.len() as f64
}

/// Determine how much improvement in the limiting factor is needed to reach a target class.
pub fn gap_to_target(zone: &ProspectivityZone, target_class: ProspectivityClass) -> f64 {
    let (score, limiting) = calculate_prospectivity_score(&zone.evidence);
    let threshold = match target_class {
        ProspectivityClass::Prime => 80.0,
        ProspectivityClass::Strong => 60.0,
        ProspectivityClass::Watch => 40.0,
        ProspectivityClass::Early => 0.0,
    };
    if score >= threshold {
        return 0.0;
    }
    let deficit = threshold - score;
    let weight = match limiting.as_str() {
        "geology" => 0.30,
        "geochemistry" => 0.25,
        "geophysics" => 0.20,
        "infrastructure" => 0.10,
        "policy" => 0.15,
        _ => return deficit,
    };
    deficit / weight
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _sample_evidence() -> ProspectivityEvidence {
        ProspectivityEvidence::new(80.0, 70.0, 60.0, 50.0, 90.0)
    }

    fn sample_zone(id: &str, geo: f64, geochem: f64, geoph: f64, infra: f64, policy: f64) -> ProspectivityZone {
        ProspectivityZone {
            id: id.to_string(),
            name: format!("Zone {}", id),
            region: "Test".to_string(),
            country: "AU".to_string(),
            mineral_id: "LI".to_string(),
            deposit_model: "Pegmatite".to_string(),
            evidence: ProspectivityEvidence::new(geo, geochem, geoph, infra, policy),
            confidence: 0.9,
        }
    }

    #[test]
    fn test_score_perfect_evidence() {
        let evidence = ProspectivityEvidence::new(100.0, 100.0, 100.0, 100.0, 100.0);
        let (score, _) = calculate_prospectivity_score(&evidence);
        assert!((score - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_score_zero_evidence() {
        let evidence = ProspectivityEvidence::zero();
        let (score, _) = calculate_prospectivity_score(&evidence);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_weighted_calculation() {
        // geology=100*0.30 + geochemistry=0*0.25 + geophysics=0*0.20 + infrastructure=0*0.10 + policy=0*0.15 = 30.0
        let evidence = ProspectivityEvidence::new(100.0, 0.0, 0.0, 0.0, 0.0);
        let (score, limiting) = calculate_prospectivity_score(&evidence);
        assert!((score - 30.0).abs() < 0.001);
        assert_ne!(limiting, "geology");
    }

    #[test]
    fn test_classify_prime() {
        assert_eq!(classify_prospectivity(80.0), ProspectivityClass::Prime);
        assert_eq!(classify_prospectivity(95.0), ProspectivityClass::Prime);
    }

    #[test]
    fn test_classify_strong() {
        assert_eq!(classify_prospectivity(60.0), ProspectivityClass::Strong);
        assert_eq!(classify_prospectivity(79.9), ProspectivityClass::Strong);
    }

    #[test]
    fn test_classify_watch() {
        assert_eq!(classify_prospectivity(40.0), ProspectivityClass::Watch);
        assert_eq!(classify_prospectivity(59.9), ProspectivityClass::Watch);
    }

    #[test]
    fn test_classify_early() {
        assert_eq!(classify_prospectivity(0.0), ProspectivityClass::Early);
        assert_eq!(classify_prospectivity(39.9), ProspectivityClass::Early);
    }

    #[test]
    fn test_score_zone() {
        let zone = sample_zone("Z1", 80.0, 80.0, 80.0, 80.0, 80.0);
        let result = score_zone(&zone);
        assert_eq!(result.zone_id, "Z1");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_rank_zones_sorts_descending() {
        let zones = vec![
            sample_zone("A", 20.0, 20.0, 20.0, 20.0, 20.0),
            sample_zone("B", 100.0, 100.0, 100.0, 100.0, 100.0),
            sample_zone("C", 50.0, 50.0, 50.0, 50.0, 50.0),
        ];
        let ranked = rank_zones(&zones);
        assert!(ranked[0].score >= ranked[1].score);
        assert!(ranked[1].score >= ranked[2].score);
        assert_eq!(ranked[0].zone_id, "B");
    }

    #[test]
    fn test_limiting_factor_identifies_weakest() {
        // Infrastructure has lowest weight (0.10), so with low infra score
        // it will have lowest contribution
        let evidence = ProspectivityEvidence::new(80.0, 80.0, 80.0, 10.0, 80.0);
        let (_, limiting) = calculate_prospectivity_score(&evidence);
        assert_eq!(limiting, "infrastructure");
    }

    #[test]
    fn test_confidence_adjusted_score() {
        let zone = sample_zone("Z1", 100.0, 100.0, 100.0, 100.0, 100.0);
        let adjusted = confidence_adjusted_score(&zone);
        assert!((adjusted - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_by_class() {
        let zones = vec![
            sample_zone("A", 90.0, 90.0, 90.0, 90.0, 90.0), // Prime
            sample_zone("B", 20.0, 20.0, 20.0, 20.0, 20.0), // Early
        ];
        let filtered = filter_by_class(&zones, ProspectivityClass::Strong);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "A");
    }

    #[test]
    fn test_zones_limited_by() {
        let zones = vec![
            sample_zone("A", 80.0, 80.0, 80.0, 10.0, 80.0),
            sample_zone("B", 80.0, 80.0, 80.0, 80.0, 80.0),
        ];
        let limited = zones_limited_by(&zones, "infrastructure");
        // Both zones have infrastructure as limiting factor (lowest weight × value)
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn test_average_prospectivity_empty() {
        assert!((average_prospectivity(&[]) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_average_prospectivity_nonempty() {
        let zones = vec![
            sample_zone("A", 100.0, 100.0, 100.0, 100.0, 100.0),
            sample_zone("B", 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let avg = average_prospectivity(&zones);
        assert!((avg - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_gap_to_target_already_met() {
        let zone = sample_zone("Z1", 100.0, 100.0, 100.0, 100.0, 100.0);
        let gap = gap_to_target(&zone, ProspectivityClass::Prime);
        assert!(gap < 0.001);
    }

    #[test]
    fn test_gap_to_target_needs_improvement() {
        let zone = sample_zone("Z1", 50.0, 50.0, 50.0, 50.0, 50.0);
        let gap = gap_to_target(&zone, ProspectivityClass::Prime);
        assert!(gap > 0.0);
    }
}
