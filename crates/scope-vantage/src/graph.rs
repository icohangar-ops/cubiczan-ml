// scope-vantage/src/graph.rs — Trade flow graph: adjacency, centrality, HHI, shortest paths

use crate::types::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// A weighted directed graph of trade flows.
#[derive(Debug, Clone)]
pub struct TradeFlowGraph {
    /// adjacency[source][target] = total value USD
    pub adjacency: HashMap<String, HashMap<String, f64>>,
    /// adjacency_by_commodity[source][target][(commodity, year)] = value
    pub adjacency_by_commodity: HashMap<String, HashMap<String, Vec<SupplyChainEdge>>>,
    pub all_nodes: HashSet<String>,
}

impl TradeFlowGraph {
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            adjacency_by_commodity: HashMap::new(),
            all_nodes: HashSet::new(),
        }
    }

    /// Build the graph from bilateral trade flow tuples.
    pub fn from_bilateral_flows(flows: &[(String, String, String, f64)]) -> Self {
        let mut graph = Self::new();
        for (src, tgt, _commodity, value) in flows {
            if *value <= 0.0 {
                continue;
            }
            graph.all_nodes.insert(src.clone());
            graph.all_nodes.insert(tgt.clone());
            *graph
                .adjacency
                .entry(src.clone())
                .or_default()
                .entry(tgt.clone())
                .or_insert(0.0) += value;
        }
        graph
    }

    /// Build with commodity-resolved edges.
    pub fn from_edges(edges: &[SupplyChainEdge]) -> Self {
        let mut graph = Self::new();
        for edge in edges {
            if edge.value_usd <= 0.0 {
                continue;
            }
            graph.all_nodes.insert(edge.source.clone());
            graph.all_nodes.insert(edge.target.clone());
            *graph
                .adjacency
                .entry(edge.source.clone())
                .or_default()
                .entry(edge.target.clone())
                .or_insert(0.0) += edge.value_usd;
            graph
                .adjacency_by_commodity
                .entry(edge.source.clone())
                .or_default()
                .entry(edge.target.clone())
                .or_default()
                .push(edge.clone());
        }
        graph
    }

    /// Total outgoing trade value from a node.
    pub fn out_degree_value(&self, node: &str) -> f64 {
        self.adjacency
            .get(node)
            .map(|targets| targets.values().sum())
            .unwrap_or(0.0)
    }

    /// Total incoming trade value to a node.
    pub fn in_degree_value(&self, node: &str) -> f64 {
        let mut total = 0.0;
        for (_src, targets) in &self.adjacency {
            if let Some(val) = targets.get(node) {
                total += val;
            }
        }
        total
    }

    /// BFS shortest path (unweighted hop count). Returns (distance, path) or None.
    pub fn shortest_path(&self, source: &str, target: &str) -> Option<(usize, Vec<String>)> {
        if source == target {
            return Some((0, vec![source.to_string()]));
        }
        if !self.all_nodes.contains(source) || !self.all_nodes.contains(target) {
            return None;
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        queue.push_back((source.to_string(), vec![source.to_string()]));
        visited.insert(source.to_string());

        while let Some((current, path)) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for neighbor in neighbors.keys() {
                    if neighbor == target {
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        return Some((new_path.len() - 1, new_path));
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back((neighbor.clone(), new_path));
                    }
                }
            }
        }
        None
    }

    /// Reachable nodes from a source.
    pub fn reachable(&self, source: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(source.to_string());
        visited.insert(source.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for neighbor in neighbors.keys() {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        visited
    }

    /// Betweenness centrality for a single node (unweighted).
    pub fn betweenness_centrality(&self, node: &str) -> f64 {
        let nodes: Vec<&String> = self.all_nodes.iter().collect();
        let mut count = 0usize;
        let mut total = 0usize;

        for src in &nodes {
            for tgt in &nodes {
                if src == tgt || *src == node || *tgt == node {
                    continue;
                }
                total += 1;
                if let Some((_dist, path)) = self.shortest_path(src, tgt) {
                    if path.len() >= 3 && path.contains(&node.to_string()) {
                        // Check it passes through node (not start/end)
                        if path[1..path.len() - 1].contains(&node.to_string()) {
                            count += 1;
                        }
                    }
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64
        }
    }

    /// Degree centrality (fraction of nodes this node connects to).
    pub fn degree_centrality(&self, node: &str) -> f64 {
        let n = self.all_nodes.len();
        if n <= 1 {
            return 0.0;
        }
        let mut connections = HashSet::new();
        // Outgoing
        if let Some(targets) = self.adjacency.get(node) {
            for t in targets.keys() {
                connections.insert(t.clone());
            }
        }
        // Incoming
        for (src, targets) in &self.adjacency {
            if src != node && targets.contains_key(node) {
                connections.insert(src.clone());
            }
        }
        connections.len() as f64 / (n - 1) as f64
    }

    /// Herfindahl-Hirschman Index for imports into a country for a commodity.
    /// Takes a list of (source_country, value_usd) import shares.
    /// Returns HHI in [0, 1].
    pub fn hhi(shares: &[(String, f64)]) -> f64 {
        let total: f64 = shares.iter().map(|(_, v)| v).sum();
        if total == 0.0 {
            return 0.0;
        }
        let hhi: f64 = shares
            .iter()
            .map(|(_, v)| {
                let share = v / total;
                share * share
            })
            .sum();
        hhi
    }

    /// Import concentration score for a specific commodity and importing country.
    /// Returns HHI. Provide import sources as (source, value).
    pub fn import_concentration_score(
        &self,
        _importer: &str,
        _commodity: &str,
        import_sources: &[(String, f64)],
    ) -> f64 {
        Self::hhi(import_sources)
    }

    /// Identify critical nodes based on betweenness centrality above threshold.
    pub fn critical_nodes(&self, threshold: f64) -> Vec<(String, f64)> {
        let mut results: Vec<(String, f64)> = self
            .all_nodes
            .iter()
            .map(|n| {
                let bc = self.betweenness_centrality(n);
                (n.clone(), bc)
            })
            .filter(|(_, bc)| *bc >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Top importers for a given commodity.
    pub fn top_importers(
        &self,
        commodity_code: &str,
        top_n: usize,
    ) -> Vec<(String, f64)> {
        let mut import_totals: HashMap<String, f64> = HashMap::new();
        for (_src, targets) in &self.adjacency_by_commodity {
            for (tgt, edges) in targets {
                for edge in edges {
                    if edge.commodity.as_str() == commodity_code {
                        *import_totals.entry(tgt.clone()).or_insert(0.0) += edge.value_usd;
                    }
                }
            }
        }
        let mut vec: Vec<(String, f64)> = import_totals.into_iter().collect();
        vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(top_n);
        vec
    }

    /// Top exporters for a given commodity.
    pub fn top_exporters(
        &self,
        commodity_code: &str,
        top_n: usize,
    ) -> Vec<(String, f64)> {
        let mut export_totals: HashMap<String, f64> = HashMap::new();
        for (src, targets) in &self.adjacency_by_commodity {
            for (_tgt, edges) in targets {
                for edge in edges {
                    if edge.source == *src && edge.commodity.as_str() == commodity_code {
                        *export_totals.entry(src.clone()).or_insert(0.0) += edge.value_usd;
                    }
                }
            }
        }
        let mut vec: Vec<(String, f64)> = export_totals.into_iter().collect();
        vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(top_n);
        vec
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.all_nodes.len()
    }

    /// Number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|m| m.len()).sum()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_flows() -> Vec<(String, String, String, f64)> {
        vec![
            ("A".to_string(), "B".to_string(), "870323".to_string(), 100.0),
            ("A".to_string(), "C".to_string(), "870323".to_string(), 200.0),
            ("B".to_string(), "C".to_string(), "870323".to_string(), 50.0),
            ("B".to_string(), "D".to_string(), "870323".to_string(), 150.0),
            ("C".to_string(), "D".to_string(), "870323".to_string(), 80.0),
            ("A".to_string(), "D".to_string(), "270900".to_string(), 300.0),
        ]
    }

    #[test]
    fn graph_from_flows() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        assert_eq!(graph.node_count(), 4); // A, B, C, D
        assert!(graph.edge_count() >= 5);
    }

    #[test]
    fn graph_out_degree() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let out_a = graph.out_degree_value("A");
        assert!((out_a - 600.0).abs() < 1e-6);
    }

    #[test]
    fn graph_in_degree() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let in_d = graph.in_degree_value("D");
        // From B: 150, from C: 80, from A: 300 = 530
        assert!((in_d - 530.0).abs() < 1e-6);
    }

    #[test]
    fn graph_in_degree_missing_node() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        assert_eq!(graph.in_degree_value("Z"), 0.0);
    }

    #[test]
    fn shortest_path_direct() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let result = graph.shortest_path("A", "B");
        assert!(result.is_some());
        let (dist, path) = result.unwrap();
        assert_eq!(dist, 1);
        assert_eq!(path, vec!["A", "B"]);
    }

    #[test]
    fn shortest_path_multi_hop() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let result = graph.shortest_path("A", "D");
        assert!(result.is_some());
        let (dist, _path) = result.unwrap();
        assert_eq!(dist, 1); // A->D is direct
    }

    #[test]
    fn shortest_path_no_connection() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let result = graph.shortest_path("D", "A");
        assert!(result.is_none()); // D has no outgoing edges
    }

    #[test]
    fn shortest_path_same_node() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let result = graph.shortest_path("A", "A");
        assert!(result.is_some());
        let (dist, path) = result.unwrap();
        assert_eq!(dist, 0);
        assert_eq!(path, vec!["A"]);
    }

    #[test]
    fn shortest_path_nonexistent() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        assert!(graph.shortest_path("X", "Y").is_none());
    }

    #[test]
    fn reachable_from_a() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let reachable = graph.reachable("A");
        assert!(reachable.contains("A"));
        assert!(reachable.contains("B"));
        assert!(reachable.contains("C"));
        assert!(reachable.contains("D"));
    }

    #[test]
    fn reachable_isolated() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let reachable = graph.reachable("D");
        assert_eq!(reachable.len(), 1); // Only D itself
    }

    #[test]
    fn degree_centrality() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let dc_a = graph.degree_centrality("A");
        assert!(dc_a > 0.0);
        let dc_d = graph.degree_centrality("D");
        assert!(dc_d > 0.0);
    }

    #[test]
    fn hhi_concentrated() {
        let shares = vec![("A".to_string(), 90.0), ("B".to_string(), 10.0)];
        let hhi = TradeFlowGraph::hhi(&shares);
        // (0.9^2 + 0.1^2) = 0.81 + 0.01 = 0.82
        assert!((hhi - 0.82).abs() < 1e-9);
    }

    #[test]
    fn hhi_diversified() {
        let shares = vec![
            ("A".to_string(), 25.0),
            ("B".to_string(), 25.0),
            ("C".to_string(), 25.0),
            ("D".to_string(), 25.0),
        ];
        let hhi = TradeFlowGraph::hhi(&shares);
        // 4 * 0.25^2 = 0.25
        assert!((hhi - 0.25).abs() < 1e-9);
    }

    #[test]
    fn hhi_empty() {
        let hhi = TradeFlowGraph::hhi(&[]);
        assert_eq!(hhi, 0.0);
    }

    #[test]
    fn hhi_zero_values() {
        let shares = vec![("A".to_string(), 0.0), ("B".to_string(), 0.0)];
        let hhi = TradeFlowGraph::hhi(&shares);
        assert_eq!(hhi, 0.0);
    }

    #[test]
    fn import_concentration_score() {
        let graph = TradeFlowGraph::from_bilateral_flows(&sample_flows());
        let sources = vec![
            ("A".to_string(), 80.0),
            ("B".to_string(), 20.0),
        ];
        let score = graph.import_concentration_score("C", "870323", &sources);
        assert!((score - 0.68).abs() < 1e-9);
    }

    #[test]
    fn from_edges() {
        let edges = vec![
            SupplyChainEdge::new("X", "Y", CommodityCode::new("870323").unwrap(), 100.0, 10.0, 2023),
            SupplyChainEdge::new("X", "Z", CommodityCode::new("270900").unwrap(), 200.0, 20.0, 2023),
        ];
        let graph = TradeFlowGraph::from_edges(&edges);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn top_importers() {
        let edges = vec![
            SupplyChainEdge::new("A", "B", CommodityCode::new("870323").unwrap(), 100.0, 10.0, 2023),
            SupplyChainEdge::new("C", "B", CommodityCode::new("870323").unwrap(), 50.0, 5.0, 2023),
            SupplyChainEdge::new("A", "C", CommodityCode::new("870323").unwrap(), 30.0, 3.0, 2023),
        ];
        let graph = TradeFlowGraph::from_edges(&edges);
        let top = graph.top_importers("870323", 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "B"); // 150 total
    }

    #[test]
    fn top_exporters() {
        let edges = vec![
            SupplyChainEdge::new("A", "B", CommodityCode::new("870323").unwrap(), 100.0, 10.0, 2023),
            SupplyChainEdge::new("A", "C", CommodityCode::new("870323").unwrap(), 30.0, 3.0, 2023),
            SupplyChainEdge::new("B", "C", CommodityCode::new("870323").unwrap(), 20.0, 2.0, 2023),
        ];
        let graph = TradeFlowGraph::from_edges(&edges);
        let top = graph.top_exporters("870323", 2);
        assert_eq!(top[0].0, "A"); // 130 total
    }

    #[test]
    fn skip_zero_value_flows() {
        let flows = vec![
            ("A".to_string(), "B".to_string(), "870323".to_string(), 0.0),
            ("A".to_string(), "B".to_string(), "870323".to_string(), -5.0),
        ];
        let graph = TradeFlowGraph::from_bilateral_flows(&flows);
        assert_eq!(graph.edge_count(), 0);
    }
}
