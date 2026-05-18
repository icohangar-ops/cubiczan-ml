// scope-vantage/src/types.rs — Core domain types for supply chain intelligence

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HS6 commodity code — 6-digit Harmonized System classification.
/// The string is always exactly 6 digits (zero-padded), validated on construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommodityCode(pub String);

impl CommodityCode {
    /// Construct an HS6 code, validating length and digit-only content.
    pub fn new(raw: &str) -> anyhow::Result<Self> {
        let trimmed = raw.trim();
        if trimmed.len() != 6 || !trimmed.chars().all(|c| c.is_ascii_digit()) {
            anyhow::bail!("HS6 code must be exactly 6 digits, got: '{}'", trimmed);
        }
        Ok(Self(trimmed.to_string()))
    }

    /// 2-digit chapter.
    pub fn chapter(&self) -> &str {
        &self.0[0..2]
    }

    /// 4-digit heading.
    pub fn heading(&self) -> &str {
        &self.0[0..4]
    }

    /// Full 6-digit subheading.
    pub fn subheading(&self) -> &str {
        &self.0
    }

    /// Display as string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CommodityCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// ISO 3166-1 numeric or alpha-2 country identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Country {
    pub code: String,
    pub name: String,
}

impl Country {
    pub fn new(code: &str, name: &str) -> Self {
        Self {
            code: code.trim().to_uppercase(),
            name: name.trim().to_string(),
        }
    }

    /// ISO 3166-1 numeric (3 digits) or alpha-2.
    pub fn is_numeric(&self) -> bool {
        self.code.len() == 3 && self.code.chars().all(|c| c.is_ascii_digit())
    }
}

/// Direction of trade flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeFlow {
    Import,
    Export,
    ReExport,
}

impl TradeFlow {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "import" | "m" | "1" => Ok(TradeFlow::Import),
            "export" | "x" | "2" => Ok(TradeFlow::Export),
            "re-export" | "reexport" | "re_export" | "3" => Ok(TradeFlow::ReExport),
            _ => anyhow::bail!("Unknown trade flow: '{}'", s),
        }
    }
}

/// A single trade record from UN Comtrade or similar source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub reporter: Country,
    pub partner: Country,
    pub commodity: CommodityCode,
    pub flow: TradeFlow,
    pub trade_value_usd: f64,
    pub net_weight_kg: f64,
    pub year: u32,
    pub date: Option<NaiveDate>,
    pub flags: u8,
}

impl TradeRecord {
    pub fn new(
        reporter: Country,
        partner: Country,
        commodity: CommodityCode,
        flow: TradeFlow,
        trade_value_usd: f64,
        net_weight_kg: f64,
        year: u32,
    ) -> Self {
        Self {
            reporter,
            partner,
            commodity,
            flow,
            trade_value_usd,
            net_weight_kg,
            year,
            date: None,
            flags: 0,
        }
    }

    /// Unit price in USD/kg.
    pub fn unit_price(&self) -> Option<f64> {
        if self.net_weight_kg > 0.0 {
            Some(self.trade_value_usd / self.net_weight_kg)
        } else {
            None
        }
    }

    /// Validate the record fields.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.trade_value_usd < 0.0 {
            anyhow::bail!("Negative trade value");
        }
        if self.net_weight_kg < 0.0 {
            anyhow::bail!("Negative net weight");
        }
        if self.year < 1900 || self.year > 2100 {
            anyhow::bail!("Year out of range");
        }
        if self.reporter.code == self.partner.code {
            anyhow::bail!("Reporter and partner cannot be the same country");
        }
        Ok(())
    }

    /// Dedup key: (reporter, partner, commodity, flow, year).
    pub fn dedup_key(&self) -> String {
        format!(
            "{}_{}_{}_{}_{}",
            self.reporter.code,
            self.partner.code,
            self.commodity.0,
            match self.flow {
                TradeFlow::Import => "I",
                TradeFlow::Export => "X",
                TradeFlow::ReExport => "R",
            },
            self.year
        )
    }
}

/// A node in the supply-chain graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyChainNode {
    pub country: Country,
    pub total_export_value: f64,
    pub total_import_value: f64,
    pub commodity_breakdown: HashMap<String, f64>,
}

impl SupplyChainNode {
    pub fn new(country: Country) -> Self {
        Self {
            country,
            total_export_value: 0.0,
            total_import_value: 0.0,
            commodity_breakdown: HashMap::new(),
        }
    }

    pub fn trade_volume(&self) -> f64 {
        self.total_export_value + self.total_import_value
    }
}

/// A directed edge in the trade-flow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyChainEdge {
    pub source: String,
    pub target: String,
    pub commodity: CommodityCode,
    pub value_usd: f64,
    pub weight_kg: f64,
    pub year: u32,
}

impl SupplyChainEdge {
    pub fn new(
        source: &str,
        target: &str,
        commodity: CommodityCode,
        value_usd: f64,
        weight_kg: f64,
        year: u32,
    ) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
            commodity,
            value_usd,
            weight_kg,
            year,
        }
    }
}

/// Individual risk factor for a node or edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub score: f64,       // 0.0 – 1.0
    pub weight: f64,      // 0.0 – 1.0
    pub description: String,
}

impl RiskFactor {
    pub fn new(name: &str, score: f64, weight: f64) -> Self {
        Self {
            name: name.to_string(),
            score: score.clamp(0.0, 1.0),
            weight: weight.clamp(0.0, 1.0),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Weighted contribution.
    pub fn contribution(&self) -> f64 {
        self.score * self.weight
    }
}

/// Aggregated resilience score for a supply-chain path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceScore {
    pub country_code: String,
    pub commodity: CommodityCode,
    pub overall: f64,               // 0 – 100
    pub import_concentration: f64,  // HHI-scaled 0 – 100
    pub supplier_diversity: f64,    // 0 – 100
    pub geopolitical_risk: f64,     // 0 – 100
    pub price_volatility: f64,      // 0 – 100
    pub factors: Vec<RiskFactor>,
}

impl ResilienceScore {
    pub fn new(country_code: &str, commodity: CommodityCode) -> Self {
        Self {
            country_code: country_code.to_string(),
            commodity,
            overall: 0.0,
            import_concentration: 0.0,
            supplier_diversity: 0.0,
            geopolitical_risk: 0.0,
            price_volatility: 0.0,
            factors: Vec::new(),
        }
    }

    pub fn compute_overall(&mut self) {
        if self.factors.is_empty() {
            self.overall = 100.0;
            return;
        }
        let total_weight: f64 = self.factors.iter().map(|f| f.weight).sum();
        let weighted_score: f64 = self.factors.iter().map(|f| f.contribution()).sum();
        // Invert: risk 0 → resilience 100, risk 1 → resilience 0
        let avg_risk = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };
        self.overall = (1.0 - avg_risk) * 100.0;
    }
}

/// A disruption scenario for Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisruptionScenario {
    pub id: String,
    pub description: String,
    pub affected_countries: Vec<String>,
    pub affected_commodities: Vec<CommodityCode>,
    pub severity: f64,            // 0.0 – 1.0
    pub duration_months: u32,
    pub probability: f64,         // 0.0 – 1.0
}

impl DisruptionScenario {
    pub fn new(id: &str, description: &str, severity: f64, duration_months: u32, probability: f64) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            affected_countries: Vec::new(),
            affected_commodities: Vec::new(),
            severity: severity.clamp(0.0, 1.0),
            duration_months,
            probability: probability.clamp(0.0, 1.0),
        }
    }

    pub fn with_countries(mut self, codes: &[&str]) -> Self {
        self.affected_countries = codes.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_commodities(mut self, codes: &[&str]) -> anyhow::Result<Self> {
        for c in codes {
            self.affected_commodities.push(CommodityCode::new(c)?);
        }
        Ok(self)
    }

    /// Expected impact = probability × severity × duration_months.
    pub fn expected_impact(&self) -> f64 {
        self.probability * self.severity * (self.duration_months as f64) / 12.0
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commodity_code_valid() {
        let code = CommodityCode::new("010101").unwrap();
        assert_eq!(code.chapter(), "01");
        assert_eq!(code.heading(), "0101");
        assert_eq!(code.subheading(), "010101");
    }

    #[test]
    fn commodity_code_invalid_length() {
        assert!(CommodityCode::new("12345").is_err());
        assert!(CommodityCode::new("1234567").is_err());
    }

    #[test]
    fn commodity_code_invalid_chars() {
        assert!(CommodityCode::new("ABCDEF").is_err());
        assert!(CommodityCode::new("01a101").is_err());
    }

    #[test]
    fn commodity_code_display() {
        let code = CommodityCode::new("870323").unwrap();
        assert_eq!(format!("{}", code), "870323");
    }

    #[test]
    fn country_numeric_detection() {
        let c_num = Country::new("840", "United States");
        assert!(c_num.is_numeric());
        let c_alpha = Country::new("US", "United States");
        assert!(!c_alpha.is_numeric());
    }

    #[test]
    fn trade_flow_from_str() {
        assert_eq!(TradeFlow::from_str("import").unwrap(), TradeFlow::Import);
        assert_eq!(TradeFlow::from_str("M").unwrap(), TradeFlow::Import);
        assert_eq!(TradeFlow::from_str("export").unwrap(), TradeFlow::Export);
        assert_eq!(TradeFlow::from_str("X").unwrap(), TradeFlow::Export);
        assert_eq!(TradeFlow::from_str("re-export").unwrap(), TradeFlow::ReExport);
        assert!(TradeFlow::from_str("unknown").is_err());
    }

    #[test]
    fn trade_record_valid() {
        let rec = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "China"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            1_000_000.0,
            50_000.0,
            2023,
        );
        assert!(rec.validate().is_ok());
        assert_eq!(rec.unit_price().unwrap(), 20.0);
    }

    #[test]
    fn trade_record_invalid_negative_value() {
        let rec = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "China"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            -1.0,
            50_000.0,
            2023,
        );
        assert!(rec.validate().is_err());
    }

    #[test]
    fn trade_record_same_country_fails() {
        let rec = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("840", "USA"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            1.0,
            1.0,
            2023,
        );
        assert!(rec.validate().is_err());
    }

    #[test]
    fn trade_record_dedup_key() {
        let rec = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "CHN"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            100.0,
            10.0,
            2023,
        );
        let key = rec.dedup_key();
        assert!(key.contains("840_156"));
        assert!(key.contains("870323"));
    }

    #[test]
    fn risk_factor_clamping() {
        let rf = RiskFactor::new("test", 1.5, 2.0);
        assert_eq!(rf.score, 1.0);
        assert_eq!(rf.weight, 1.0);
        let rf2 = RiskFactor::new("test2", -0.5, -0.1);
        assert_eq!(rf2.score, 0.0);
        assert_eq!(rf2.weight, 0.0);
    }

    #[test]
    fn risk_factor_contribution() {
        let rf = RiskFactor::new("geo", 0.6, 0.3);
        assert!((rf.contribution() - 0.18).abs() < 1e-9);
    }

    #[test]
    fn resilience_score_compute() {
        let mut rs = ResilienceScore::new("840", CommodityCode::new("270900").unwrap());
        rs.factors.push(RiskFactor::new("conc", 0.4, 0.5));
        rs.factors.push(RiskFactor::new("geo", 0.6, 0.5));
        rs.compute_overall();
        // avg_risk = (0.4*0.5 + 0.6*0.5) / 1.0 = 0.5
        assert!((rs.overall - 50.0).abs() < 1e-6);
    }

    #[test]
    fn resilience_score_empty_factors() {
        let mut rs = ResilienceScore::new("840", CommodityCode::new("270900").unwrap());
        rs.compute_overall();
        assert_eq!(rs.overall, 100.0);
    }

    #[test]
    fn disruption_scenario_expected_impact() {
        let ds = DisruptionScenario::new("SC1", "Trade war", 0.8, 12, 0.3);
        assert!((ds.expected_impact() - 0.24).abs() < 1e-9);
    }

    #[test]
    fn disruption_scenario_clamping() {
        let ds = DisruptionScenario::new("SC2", "Crisis", 2.0, 6, -0.5);
        assert_eq!(ds.severity, 1.0);
        assert_eq!(ds.probability, 0.0);
    }

    #[test]
    fn supply_chain_node_volume() {
        let mut node = SupplyChainNode::new(Country::new("840", "USA"));
        node.total_export_value = 500.0;
        node.total_import_value = 300.0;
        assert_eq!(node.trade_volume(), 800.0);
    }
}
