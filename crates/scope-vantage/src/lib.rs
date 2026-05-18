// scope-vantage/src/lib.rs — Supply chain intelligence crate

pub mod types;
pub mod trade;
pub mod commodity;
pub mod graph;
pub mod risk;
pub mod pipeline;

// Re-export key types at crate root
pub use types::{
    CommodityCode, Country, TradeFlow, TradeRecord,
    SupplyChainNode, SupplyChainEdge, RiskFactor, ResilienceScore, DisruptionScenario,
};
pub use trade::{
    country_registry, resolve_country, normalize_hs_code,
    parse_csv_row, parse_csv_batch, parse_json_record, parse_json_batch,
    dedup_records, bilateral_flows,
};
pub use commodity::{
    PriceObservation, PriceFeed, CommodityCategory,
    categorize_hs2, categorize_commodity,
    price_correlation_matrix, supply_demand_estimate,
};
pub use graph::TradeFlowGraph;
pub use risk::{
    geopolitical_risk_map, get_geopolitical_risk,
    single_source_dependency_risk, supplier_diversity_index,
    compute_resilience_score, disruption_impact,
    monte_carlo_simulation, MonteCarloResult,
    value_at_risk,
};
pub use pipeline::{run_pipeline, run_batch_pipeline, quick_risk_summary, PipelineOutput};

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compute a data integrity hash for a set of trade records (SHA-3 256).
pub fn data_integrity_hash(records: &[TradeRecord]) -> String {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    for rec in records {
        let line = format!(
            "{}|{}|{}|{}|{}|{}",
            rec.reporter.code,
            rec.partner.code,
            rec.commodity.0,
            rec.trade_value_usd,
            rec.net_weight_kg,
            rec.year,
        );
        hasher.update(line.as_bytes());
    }
    let result = hasher.finalize();
    hex::encode(result)
}

// ─── Integration Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::collections::HashMap;

    fn full_dataset() -> Vec<&'static str> {
        vec![
            // USA imports vehicles from multiple countries
            "840,156,870323,import,5000.0,500.0,2023",
            "840,276,870323,import,3000.0,300.0,2023",
            "840,392,870323,import,2000.0,200.0,2023",
            "840,410,870323,import,1500.0,150.0,2023",
            "840,250,870323,import,500.0,50.0,2023",
            // Exports from those countries to USA
            "156,840,870323,export,5000.0,500.0,2023",
            "276,840,870323,export,3000.0,300.0,2023",
            "392,840,870323,export,2000.0,200.0,2023",
            "410,840,870323,export,1500.0,150.0,2023",
            "250,840,870323,export,500.0,50.0,2023",
            // Cross-border trade
            "156,276,870323,export,1000.0,100.0,2023",
            "276,410,870323,export,800.0,80.0,2023",
            "410,392,870323,export,600.0,60.0,2023",
            // Energy imports
            "840,643,270900,import,8000.0,80000.0,2023",
            "840,682,270900,import,6000.0,60000.0,2023",
            "840,156,270900,import,4000.0,40000.0,2023",
            "643,840,270900,export,8000.0,80000.0,2023",
            "682,840,270900,export,6000.0,60000.0,2023",
            "156,840,270900,export,4000.0,40000.0,2023",
        ]
    }

    #[test]
    fn end_to_end_vehicles_pipeline() {
        let code = CommodityCode::new("870323").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&full_dataset(), &code, &feeds).unwrap();

        assert!(output.total_records_processed > 10);
        assert!(output.graph_node_count >= 5);
        assert!(!output.country_scores.is_empty());
        assert!(!output.top_risk_countries.is_empty());
    }

    #[test]
    fn end_to_end_energy_pipeline() {
        let code = CommodityCode::new("270900").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&full_dataset(), &code, &feeds).unwrap();

        assert!(output.category.contains("Energy"));
        assert!(output.graph_node_count >= 3);
    }

    #[test]
    fn end_to_end_with_monte_carlo() {
        let code = CommodityCode::new("270900").unwrap();
        let sources = vec![
            ("643".to_string(), 8000.0),
            ("682".to_string(), 6000.0),
            ("156".to_string(), 4000.0),
        ];
        let scenarios = vec![
            DisruptionScenario::new("GEO", "Russia sanctions", 0.9, 12, 0.4)
                .with_countries(&["643"])
                .with_commodities(&["270900"])
                .unwrap(),
            DisruptionScenario::new("OIL", "OPEC cuts", 0.5, 6, 0.3)
                .with_countries(&["682", "156"])
                .with_commodities(&["270900"])
                .unwrap(),
        ];
        let mc = monte_carlo_simulation("840", &code, &sources, &scenarios, 2000, 42);
        assert!(mc.mean_impact >= 0.0);
        assert!(mc.percentile_95 > 0.0);
    }

    #[test]
    fn data_integrity_hash_deterministic() {
        let records = vec![
            TradeRecord::new(
                Country::new("840", "USA"),
                Country::new("156", "China"),
                CommodityCode::new("870323").unwrap(),
                TradeFlow::Import,
                100.0,
                10.0,
                2023,
            ),
        ];
        let h1 = data_integrity_hash(&records);
        let h2 = data_integrity_hash(&records);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-3 256 hex
    }

    #[test]
    fn data_integrity_hash_different_records() {
        let r1 = vec![TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "China"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            100.0, 10.0, 2023,
        )];
        let r2 = vec![TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "China"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            200.0, 10.0, 2023,
        )];
        assert_ne!(data_integrity_hash(&r1), data_integrity_hash(&r2));
    }

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn json_roundtrip_commodity() {
        let code = CommodityCode::new("870323").unwrap();
        let json = serde_json::to_string(&code).unwrap();
        let decoded: CommodityCode = serde_json::from_str(&json).unwrap();
        assert_eq!(code, decoded);
    }

    #[test]
    fn json_roundtrip_risk_factor() {
        let rf = RiskFactor::new("test", 0.5, 0.3).with_description("desc");
        let json = serde_json::to_string(&rf).unwrap();
        let decoded: RiskFactor = serde_json::from_str(&json).unwrap();
        assert_eq!(rf.name, decoded.name);
        assert!((rf.score - decoded.score).abs() < 1e-12);
    }

    #[test]
    fn json_roundtrip_resilience_score() {
        let mut rs = ResilienceScore::new("840", CommodityCode::new("270900").unwrap());
        rs.factors.push(RiskFactor::new("x", 0.4, 0.5));
        rs.compute_overall();
        let json = serde_json::to_string(&rs).unwrap();
        let decoded: ResilienceScore = serde_json::from_str(&json).unwrap();
        assert!((rs.overall - decoded.overall).abs() < 1e-12);
    }

    #[test]
    fn json_roundtrip_disruption() {
        let ds = DisruptionScenario::new("S1", "test", 0.8, 6, 0.5)
            .with_countries(&["840", "156"])
            .with_commodities(&["870323"])
            .unwrap();
        let json = serde_json::to_string(&ds).unwrap();
        let decoded: DisruptionScenario = serde_json::from_str(&json).unwrap();
        assert_eq!(ds.id, decoded.id);
        assert_eq!(ds.affected_countries.len(), decoded.affected_countries.len());
    }

    #[test]
    fn end_to_end_batch_commodities() {
        let c1 = CommodityCode::new("870323").unwrap();
        let c2 = CommodityCode::new("270900").unwrap();
        let c3 = CommodityCode::new("999999").unwrap(); // not in data
        let results = run_batch_pipeline(&full_dataset(), &[c1, c2, c3], &HashMap::new());
        assert_eq!(results.len(), 3);
        // First two should have data
        assert!(results[0].total_records_processed > 0);
        assert!(results[1].total_records_processed > 0);
        // Third should be empty
        assert_eq!(results[2].unique_trade_flows, 0);
        assert_eq!(results[2].graph_node_count, 0);
    }
}
