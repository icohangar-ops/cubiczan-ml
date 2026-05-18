// scope-vantage/src/pipeline.rs — End-to-end pipeline: trade → commodity → graph → risk

use crate::commodity::{categorize_commodity, PriceFeed};
use crate::graph::TradeFlowGraph;
use crate::risk::*;
use crate::trade::{bilateral_flows, dedup_records, parse_csv_batch};
use crate::types::*;
use anyhow::Result;
use std::collections::HashMap;

/// Full pipeline output for a single commodity analysis.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub commodity_code: CommodityCode,
    pub category: String,
    pub total_records_processed: usize,
    pub unique_trade_flows: usize,
    pub graph_node_count: usize,
    pub graph_edge_count: usize,
    pub country_scores: HashMap<String, ResilienceScore>,
    pub top_risk_countries: Vec<(String, f64)>,  // (country_code, overall risk)
    pub critical_nodes: Vec<(String, f64)>,
}

impl PipelineOutput {
    pub fn summary(&self) -> String {
        format!(
            "Commodity {} ({}): {} records → {} flows, {} nodes, {} edges. \
             Top risk: {:?}. Critical nodes: {:?}",
            self.commodity_code.as_str(),
            self.category,
            self.total_records_processed,
            self.unique_trade_flows,
            self.graph_node_count,
            self.graph_edge_count,
            self.top_risk_countries.first(),
            self.critical_nodes,
        )
    }
}

/// Run the full supply chain intelligence pipeline.
///
/// 1. Parse CSV rows → TradeRecords
/// 2. Deduplicate
/// 3. Build bilateral flows
/// 4. Construct trade flow graph
/// 5. For each importer of the target commodity, compute resilience score
/// 6. Identify critical nodes
pub fn run_pipeline(
    csv_rows: &[&str],
    target_commodity: &CommodityCode,
    price_feeds: &HashMap<String, PriceFeed>,
) -> Result<PipelineOutput> {
    // Step 1: Parse
    let (records, _errors) = parse_csv_batch(csv_rows);
    let total_processed = records.len();

    // Step 2: Filter to target commodity (and similar chapters)
    let filtered: Vec<TradeRecord> = records
        .iter()
        .filter(|r| r.commodity.as_str() == target_commodity.as_str())
        .cloned()
        .collect();

    // Step 3: Dedup
    let deduped = dedup_records(filtered);

    // Step 4: Bilateral flows
    let flows = bilateral_flows(&deduped);
    let unique_flows = flows.len();

    // Step 5: Build graph with edges
    let edges: Vec<SupplyChainEdge> = deduped
        .iter()
        .filter_map(|r| {
            let (src, tgt) = match r.flow {
                TradeFlow::Export => (r.reporter.code.clone(), r.partner.code.clone()),
                TradeFlow::Import => (r.partner.code.clone(), r.reporter.code.clone()),
                TradeFlow::ReExport => return None,
            };
            Some(SupplyChainEdge::new(
                &src, &tgt, r.commodity.clone(), r.trade_value_usd, r.net_weight_kg, r.year,
            ))
        })
        .collect();
    let graph = TradeFlowGraph::from_edges(&edges);

    // Step 6: Identify critical nodes (low threshold for testing)
    let critical = graph.critical_nodes(0.0);

    // Step 7: Compute resilience scores for each importer
    // Build import sources map: importer → [(source, value)]
    let mut import_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for (src, tgt, _commodity, value) in &flows {
        import_map.entry(tgt.clone()).or_default().push((src.clone(), *value));
    }

    let mut country_scores: HashMap<String, ResilienceScore> = HashMap::new();
    for (importer, sources) in &import_map {
        let price_cv = price_feeds
            .get(target_commodity.as_str())
            .map(|f| f.cv())
            .unwrap_or(0.0);
        let score = compute_resilience_score(importer, target_commodity, sources, price_cv);
        country_scores.insert(importer.clone(), score);
    }

    // Step 8: Rank countries by risk (lower resilience = higher risk)
    let mut top_risk: Vec<(String, f64)> = country_scores
        .iter()
        .map(|(code, score)| (code.clone(), 100.0 - score.overall))
        .collect();
    top_risk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let category = format!("{:?}", categorize_commodity(target_commodity));

    Ok(PipelineOutput {
        commodity_code: target_commodity.clone(),
        category,
        total_records_processed: total_processed,
        unique_trade_flows: unique_flows,
        graph_node_count: graph.node_count(),
        graph_edge_count: graph.edge_count(),
        country_scores,
        top_risk_countries: top_risk,
        critical_nodes: critical,
    })
}

/// Quick risk summary: given import sources, return a single risk number.
pub fn quick_risk_summary(import_sources: &[(String, f64)], _commodity: &CommodityCode) -> (f64, f64) {
    let hhi = TradeFlowGraph::hhi(import_sources);
    let ss = single_source_dependency_risk(import_sources);
    let geo: f64 = import_sources
        .iter()
        .take(3)
        .map(|(code, _)| get_geopolitical_risk(code))
        .fold(0.0_f64, f64::max);
    let composite = 0.4 * hhi + 0.3 * ss + 0.3 * geo;
    let diversity = supplier_diversity_index(import_sources);
    (composite, diversity)
}

/// Batch pipeline: run the pipeline for multiple commodities.
pub fn run_batch_pipeline(
    csv_rows: &[&str],
    commodities: &[CommodityCode],
    price_feeds: &HashMap<String, PriceFeed>,
) -> Vec<PipelineOutput> {
    commodities
        .iter()
        .filter_map(|c| run_pipeline(csv_rows, c, price_feeds).ok())
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commodity::PriceObservation;

    fn sample_csv_data() -> Vec<&'static str> {
        vec![
            "840,156,870323,import,1000.0,100.0,2023",
            "840,276,870323,import,500.0,50.0,2023",
            "840,410,870323,import,300.0,30.0,2023",
            "840,156,870323,import,200.0,20.0,2023", // dup (different value)
            "156,840,870323,export,1500.0,150.0,2023",
            "276,840,870323,export,500.0,50.0,2023",
            "410,840,870323,export,300.0,30.0,2023",
            "840,156,270900,import,2000.0,2000.0,2023",
            "840,643,270900,import,500.0,500.0,2023",
            "156,840,270900,export,2500.0,2500.0,2023",
            "276,840,270900,export,300.0,300.0,2023",
            "410,276,870323,export,100.0,10.0,2023",
        ]
    }

    #[test]
    fn run_pipeline_basic() {
        let code = CommodityCode::new("870323").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&sample_csv_data(), &code, &feeds).unwrap();
        assert_eq!(output.commodity_code.as_str(), "870323");
        assert!(output.total_records_processed > 0);
        assert!(output.graph_node_count >= 3);
    }

    #[test]
    fn run_pipeline_with_price_feed() {
        let code = CommodityCode::new("870323").unwrap();
        let mut feed = PriceFeed::new("870323");
        for i in 0..20 {
            feed.add(PriceObservation::new("870323", i, 50.0 + (i as f64) * 0.5, 100.0));
        }
        let mut feeds = HashMap::new();
        feeds.insert("870323".to_string(), feed);

        let output = run_pipeline(&sample_csv_data(), &code, &feeds).unwrap();
        assert!(!output.country_scores.is_empty());
        // Every score should have price_volatility > 0
        for score in output.country_scores.values() {
            assert!(score.price_volatility > 0.0);
        }
    }

    #[test]
    fn run_pipeline_summary() {
        let code = CommodityCode::new("870323").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&sample_csv_data(), &code, &feeds).unwrap();
        let summary = output.summary();
        assert!(summary.contains("870323"));
        assert!(summary.contains("Manufactures"));
    }

    #[test]
    fn run_pipeline_energy_commodity() {
        let code = CommodityCode::new("270900").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&sample_csv_data(), &code, &feeds).unwrap();
        assert!(output.category.contains("Energy"));
    }

    #[test]
    fn run_pipeline_empty_data() {
        let code = CommodityCode::new("870323").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&[], &code, &feeds).unwrap();
        assert_eq!(output.total_records_processed, 0);
        assert_eq!(output.graph_node_count, 0);
    }

    #[test]
    fn test_run_batch_pipeline() {
        let c1 = CommodityCode::new("870323").unwrap();
        let c2 = CommodityCode::new("270900").unwrap();
        let feeds = HashMap::new();
        let results = run_batch_pipeline(&sample_csv_data(), &[c1, c2], &feeds);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_run_batch_pipeline_no_match() {
        let c = CommodityCode::new("999999").unwrap();
        let feeds = HashMap::new();
        let results = run_batch_pipeline(&sample_csv_data(), &[c], &feeds);
        // Pipeline should still succeed but with no matching flows
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unique_trade_flows, 0);
        assert_eq!(results[0].graph_node_count, 0);
    }

    #[test]
    fn test_quick_risk_summary() {
        let sources = vec![
            ("156".to_string(), 600.0),
            ("276".to_string(), 200.0),
            ("410".to_string(), 100.0),
            ("392".to_string(), 50.0),
        ];
        let code = CommodityCode::new("870323").unwrap();
        let (risk, diversity) = quick_risk_summary(&sources, &code);
        assert!(risk > 0.0 && risk <= 1.0);
        assert!(diversity >= 0.0 && diversity <= 1.0);
    }

    #[test]
    fn test_quick_risk_summary_empty() {
        let code = CommodityCode::new("870323").unwrap();
        let (risk, diversity) = quick_risk_summary(&[], &code);
        assert_eq!(risk, 0.0);
        assert_eq!(diversity, 1.0); // no sources = max diversity (vacuously)
    }

    #[test]
    fn pipeline_country_scores_valid_range() {
        let code = CommodityCode::new("870323").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&sample_csv_data(), &code, &feeds).unwrap();
        for score in output.country_scores.values() {
            assert!(score.overall >= 0.0 && score.overall <= 100.0);
            assert!(score.import_concentration >= 0.0 && score.import_concentration <= 100.0);
            assert!(score.geopolitical_risk >= 0.0 && score.geopolitical_risk <= 100.0);
        }
    }

    #[test]
    fn pipeline_top_risk_sorted() {
        let code = CommodityCode::new("870323").unwrap();
        let feeds = HashMap::new();
        let output = run_pipeline(&sample_csv_data(), &code, &feeds).unwrap();
        for i in 1..output.top_risk_countries.len() {
            assert!(output.top_risk_countries[i - 1].1 >= output.top_risk_countries[i].1);
        }
    }
}
