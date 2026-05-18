// scope-vantage/src/risk.rs — Supply chain risk scoring, resilience, Monte Carlo

use crate::graph::TradeFlowGraph;
use crate::types::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use statrs::distribution::Uniform;
use std::collections::HashMap;

/// Geopolitical risk score per country (0.0 – 1.0).
pub fn geopolitical_risk_map() -> HashMap<&'static str, f64> {
    let mut m = HashMap::new();
    m.insert("156", 0.65);  // China
    m.insert("643", 0.70);  // Russia
    m.insert("410", 0.25);  // South Korea
    m.insert("840", 0.15);  // USA
    m.insert("276", 0.10);  // Germany
    m.insert("356", 0.35);  // India
    m.insert("764", 0.30);  // Thailand
    m.insert("586", 0.55);  // Pakistan
    m.insert("364", 0.80);  // Iran
    m.insert("682", 0.40);  // Saudi Arabia
    m.insert("032", 0.50);  // Argentina
    m.insert("710", 0.45);  // South Africa
    m.insert("818", 0.50);  // Egypt
    m.insert("792", 0.35);  // Turkey
    m.insert("380", 0.10);  // Italy
    m.insert("392", 0.15);  // Japan
    m.insert("250", 0.10);  // France
    m.insert("826", 0.10);  // UK
    m.insert("124", 0.08);  // Canada
    m.insert("036", 0.08);  // Australia
    m
}

/// Get geopolitical risk score for a country, defaulting to 0.3.
pub fn get_geopolitical_risk(country_code: &str) -> f64 {
    let map = geopolitical_risk_map();
    map.get(country_code).copied().unwrap_or(0.3)
}

/// Single-source dependency risk: what fraction of imports come from one top supplier?
/// Returns a risk score in [0.0, 1.0].
pub fn single_source_dependency_risk(import_sources: &[(String, f64)]) -> f64 {
    if import_sources.is_empty() {
        return 0.0;
    }
    let total: f64 = import_sources.iter().map(|(_, v)| v).sum();
    if total == 0.0 {
        return 0.0;
    }
    let max_share = import_sources
        .iter()
        .map(|(_, v)| v / total)
        .fold(0.0_f64, f64::max);
    max_share
}

/// Supplier diversity index (inverse HHI). Returns a score in [0.0, 1.0].
/// Higher = more diversified.
pub fn supplier_diversity_index(import_sources: &[(String, f64)]) -> f64 {
    let hhi = TradeFlowGraph::hhi(import_sources);
    // Convert HHI [0, 1] to diversity [0, 1]
    1.0 - hhi
}

/// Compute the resilience score for a country importing a commodity.
pub fn compute_resilience_score(
    importer_code: &str,
    commodity: &CommodityCode,
    import_sources: &[(String, f64)],
    price_cv: f64,
) -> ResilienceScore {
    let mut score = ResilienceScore::new(importer_code, commodity.clone());

    // 1. Import concentration (HHI)
    let hhi = TradeFlowGraph::hhi(import_sources);
    score.import_concentration = hhi * 100.0;
    score.factors.push(
        RiskFactor::new("import_concentration", hhi, 0.25)
            .with_description("HHI-based import concentration risk"),
    );

    // 2. Supplier diversity
    let diversity = supplier_diversity_index(import_sources);
    score.supplier_diversity = diversity * 100.0;
    score.factors.push(
        RiskFactor::new("supplier_diversity", 1.0 - diversity, 0.15)
            .with_description("Lack of supplier diversity"),
    );

    // 3. Single-source dependency
    let ss_risk = single_source_dependency_risk(import_sources);
    score.factors.push(
        RiskFactor::new("single_source_dep", ss_risk, 0.20)
            .with_description("Largest single supplier share"),
    );

    // 4. Geopolitical risk overlay (max risk among top 3 suppliers)
    let mut top_suppliers: Vec<_> = import_sources.to_vec();
    top_suppliers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let geo_risk = top_suppliers
        .iter()
        .take(3)
        .map(|(code, _)| get_geopolitical_risk(code))
        .fold(0.0_f64, f64::max);
    score.geopolitical_risk = geo_risk * 100.0;
    score.factors.push(
        RiskFactor::new("geopolitical", geo_risk, 0.25)
            .with_description("Max geopolitical risk among top 3 suppliers"),
    );

    // 5. Price volatility (cap CV at 1.0 for scoring)
    let pv_risk = price_cv.min(1.0);
    score.price_volatility = pv_risk * 100.0;
    score.factors.push(
        RiskFactor::new("price_volatility", pv_risk, 0.15)
            .with_description("Coefficient of variation of commodity price"),
    );

    score.compute_overall();
    score
}

/// Disruption impact simulation: compute the percentage of imports affected
/// by a disruption scenario for a given importer.
pub fn disruption_impact(
    _importer_code: &str,
    commodity: &CommodityCode,
    import_sources: &[(String, f64)],
    scenario: &DisruptionScenario,
) -> f64 {
    if !scenario.affected_commodities.iter().any(|c| c.as_str() == commodity.as_str()) {
        return 0.0;
    }
    let total_import: f64 = import_sources.iter().map(|(_, v)| v).sum();
    if total_import == 0.0 {
        return 0.0;
    }
    let affected: f64 = import_sources
        .iter()
        .filter(|(code, _)| scenario.affected_countries.contains(code))
        .map(|(_, v)| v)
        .sum();
    (affected / total_import) * scenario.severity
}

/// Monte Carlo simulation result.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    pub mean_impact: f64,
    pub std_impact: f64,
    pub percentile_5: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub scenario_count: usize,
}

/// Run Monte Carlo scenario analysis over multiple disruption scenarios.
pub fn monte_carlo_simulation(
    importer_code: &str,
    commodity: &CommodityCode,
    import_sources: &[(String, f64)],
    scenarios: &[DisruptionScenario],
    n_simulations: usize,
    seed: u64,
) -> MonteCarloResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0, 1.0);
    let mut impacts = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        let mut total_impact = 0.0;
        for scenario in scenarios {
            // Random draw: does the scenario trigger?
            let dist = uniform.unwrap_or_else(|_| Uniform::new(0.0, 1.0).unwrap());
            let draw: f64 = dist.sample(&mut rng);
            if draw < scenario.probability {
                total_impact += disruption_impact(importer_code, commodity, import_sources, scenario);
            }
        }
        impacts.push(total_impact);
    }

    impacts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = impacts.len() as f64;
    let mean = impacts.iter().sum::<f64>() / n;
    let variance = impacts.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let p5 = impacts[(0.05 * n as f64) as usize];
    let p95 = impacts[((0.95 * n as f64) as usize).min(impacts.len() - 1)];
    let p99 = impacts[((0.99 * n as f64) as usize).min(impacts.len() - 1)];

    MonteCarloResult {
        mean_impact: mean,
        std_impact: std_dev,
        percentile_5: p5,
        percentile_95: p95,
        percentile_99: p99,
        scenario_count: scenarios.len(),
    }
}

/// Generate a disruption impact distribution using normal approximation.
pub fn impact_distribution_params(
    impacts: &[f64],
) -> Option<(f64, f64)> {
    if impacts.is_empty() {
        return None;
    }
    let n = impacts.len() as f64;
    let mean = impacts.iter().sum::<f64>() / n;
    let variance = impacts.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    Some((mean, std))
}

/// Value-at-Risk at given confidence level from impact samples.
pub fn value_at_risk(impacts: &[f64], confidence: f64) -> f64 {
    if impacts.is_empty() || confidence <= 0.0 || confidence >= 1.0 {
        return 0.0;
    }
    let mut sorted = impacts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_import_sources() -> Vec<(String, f64)> {
        vec![
            ("156".to_string(), 600.0),  // China
            ("276".to_string(), 200.0),  // Germany
            ("410".to_string(), 100.0),  // South Korea
            ("392".to_string(), 50.0),   // Japan
        ]
    }

    #[test]
    fn geopolitical_risk_known() {
        let risk = get_geopolitical_risk("156");
        assert!((risk - 0.65).abs() < 1e-9);
    }

    #[test]
    fn geopolitical_risk_unknown() {
        let risk = get_geopolitical_risk("999");
        assert!((risk - 0.3).abs() < 1e-9);
    }

    #[test]
    fn single_source_dependency() {
        let sources = sample_import_sources();
        // China = 600/950 ≈ 0.6316
        let risk = single_source_dependency_risk(&sources);
        assert!((risk - 0.631578).abs() < 1e-5);
    }

    #[test]
    fn single_source_dependency_empty() {
        let risk = single_source_dependency_risk(&[]);
        assert_eq!(risk, 0.0);
    }

    #[test]
    fn single_source_dependency_single_supplier() {
        let sources = vec![("156".to_string(), 100.0)];
        let risk = single_source_dependency_risk(&sources);
        assert!((risk - 1.0).abs() < 1e-9);
    }

    #[test]
    fn supplier_diversity_index_diversified() {
        let sources = vec![
            ("A".to_string(), 25.0),
            ("B".to_string(), 25.0),
            ("C".to_string(), 25.0),
            ("D".to_string(), 25.0),
        ];
        let di = supplier_diversity_index(&sources);
        // HHI = 0.25, diversity = 0.75
        assert!((di - 0.75).abs() < 1e-9);
    }

    #[test]
    fn supplier_diversity_index_concentrated() {
        let sources = vec![("A".to_string(), 100.0)];
        let di = supplier_diversity_index(&sources);
        // HHI = 1.0, diversity = 0.0
        assert!((di - 0.0).abs() < 1e-9);
    }

    #[test]
    fn compute_resilience_score_produces_result() {
        let sources = sample_import_sources();
        let code = CommodityCode::new("870323").unwrap();
        let score = compute_resilience_score("840", &code, &sources, 0.3);
        assert!(score.overall > 0.0 && score.overall <= 100.0);
        assert!(score.import_concentration > 0.0);
        assert!(score.geopolitical_risk > 0.0);
        assert!(!score.factors.is_empty());
    }

    #[test]
    fn compute_resilience_score_empty_sources() {
        let code = CommodityCode::new("870323").unwrap();
        let score = compute_resilience_score("840", &code, &[], 0.0);
        // No risk → resilience 100
        assert!((score.overall - 100.0).abs() < 1e-6);
    }

    #[test]
    fn disruption_impact_matches_commodity() {
        let sources = sample_import_sources();
        let scenario = DisruptionScenario::new(
            "S1",
            "China trade restrictions",
            0.8,
            6,
            0.5,
        )
        .with_countries(&["156"])
        .with_commodities(&["870323"])
        .unwrap();

        let code = CommodityCode::new("870323").unwrap();
        let impact = disruption_impact("840", &code, &sources, &scenario);
        // China share = 600/950, severity = 0.8
        let expected = (600.0 / 950.0) * 0.8;
        assert!((impact - expected).abs() < 1e-6);
    }

    #[test]
    fn disruption_impact_wrong_commodity() {
        let sources = sample_import_sources();
        let scenario = DisruptionScenario::new(
            "S1",
            "China trade restrictions",
            0.8,
            6,
            0.5,
        )
        .with_countries(&["156"])
        .with_commodities(&["270900"])
        .unwrap();

        let code = CommodityCode::new("870323").unwrap();
        let impact = disruption_impact("840", &code, &sources, &scenario);
        assert_eq!(impact, 0.0);
    }

    #[test]
    fn disruption_impact_no_affected_countries() {
        let sources = sample_import_sources();
        let scenario = DisruptionScenario::new("S2", "X event", 0.5, 3, 0.3)
            .with_countries(&["999"])
            .with_commodities(&["870323"])
            .unwrap();

        let code = CommodityCode::new("870323").unwrap();
        let impact = disruption_impact("840", &code, &sources, &scenario);
        assert_eq!(impact, 0.0);
    }

    #[test]
    fn monte_carlo_simulation_runs() {
        let sources = sample_import_sources();
        let code = CommodityCode::new("870323").unwrap();
        let scenarios = vec![
            DisruptionScenario::new("MC1", "Scenario A", 0.6, 6, 0.4)
                .with_countries(&["156"])
                .with_commodities(&["870323"])
                .unwrap(),
            DisruptionScenario::new("MC2", "Scenario B", 0.3, 3, 0.2)
                .with_countries(&["276"])
                .with_commodities(&["870323"])
                .unwrap(),
        ];
        let result = monte_carlo_simulation("840", &code, &sources, &scenarios, 1000, 42);
        assert_eq!(result.scenario_count, 2);
        assert!(result.mean_impact >= 0.0);
        assert!(result.percentile_95 >= result.percentile_5);
        assert!(result.percentile_99 >= result.percentile_95);
    }

    #[test]
    fn monte_carlo_deterministic_with_seed() {
        let sources = sample_import_sources();
        let code = CommodityCode::new("870323").unwrap();
        let scenarios = vec![
            DisruptionScenario::new("MC1", "S", 0.5, 6, 0.5)
                .with_countries(&["156"])
                .with_commodities(&["870323"])
                .unwrap(),
        ];
        let r1 = monte_carlo_simulation("840", &code, &sources, &scenarios, 500, 123);
        let r2 = monte_carlo_simulation("840", &code, &sources, &scenarios, 500, 123);
        assert!((r1.mean_impact - r2.mean_impact).abs() < 1e-12);
    }

    #[test]
    fn test_impact_distribution_params() {
        let impacts = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (mean, std) = impact_distribution_params(&impacts).unwrap();
        assert!((mean - 0.3).abs() < 1e-9);
        assert!(std > 0.0);
    }

    #[test]
    fn test_impact_distribution_empty() {
        assert!(impact_distribution_params(&[]).is_none());
    }

    #[test]
    fn value_at_risk_95() {
        let impacts: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let var95 = value_at_risk(&impacts, 0.95);
        // VaR at 95% = 5th percentile (worst 5% of outcomes)
        // index = (1-0.95)*100 = 5, sorted[5] = 0.05
        assert!((var95 - 0.05).abs() < 0.02);
    }

    #[test]
    fn value_at_risk_edge_cases() {
        assert_eq!(value_at_risk(&[], 0.95), 0.0);
        assert_eq!(value_at_risk(&[0.5], 0.0), 0.0);
        assert_eq!(value_at_risk(&[0.5], 1.0), 0.0);
    }

    #[test]
    fn risk_factor_with_description() {
        let rf = RiskFactor::new("test", 0.5, 0.3).with_description("a test factor");
        assert_eq!(rf.description, "a test factor");
    }
}
