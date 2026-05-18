//! Configuration: minerals, scaling constants, regulatory keywords, mock data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported minerals with their metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineralConfig {
    pub symbol: &'static str,
    pub alpha_vantage_symbol: &'static str,
    pub unit: &'static str,
    pub typical_price_range: (f64, f64),
    pub description: &'static str,
}

/// All supported minerals.
pub static MINERALS: &[(&str, MineralConfig)] = &[
    ("LITHIUM", MineralConfig {
        symbol: "LITHIUM",
        alpha_vantage_symbol: "LITHIUM",
        unit: "USD/mt",
        typical_price_range: (10_000.0, 20_000.0),
        description: "Critical for EV batteries, energy storage",
    }),
    ("NICKEL", MineralConfig {
        symbol: "NICKEL",
        alpha_vantage_symbol: "NICKEL",
        unit: "USD/mt",
        typical_price_range: (15_000.0, 25_000.0),
        description: "Essential for stainless steel and EV batteries",
    }),
    ("COBALT", MineralConfig {
        symbol: "COBALT",
        alpha_vantage_symbol: "COBALT",
        unit: "USD/mt",
        typical_price_range: (25_000.0, 40_000.0),
        description: "Key component in lithium-ion battery cathodes",
    }),
];

/// On-chain scaling factors (must match Solidity contract).
pub const PRICE_SCALE: i64 = 1_0000_0000; // 1e8
pub const SCORE_SCALE: i64 = 100;         // composite: -100..100 → -10000..10000
pub const SENTIMENT_SCALE: i64 = 10_000;  // sentiment: -1.0..1.0 → -10000..10000
pub const REG_RISK_SCALE: i64 = 100;      // reg risk: 0..100 → 0..10000

/// Regulatory keywords and their risk weights.
pub fn regulatory_keywords() -> HashMap<&'static str, f64> {
    let mut m = HashMap::new();
    // High-risk
    m.insert("export ban", 0.95);
    m.insert("export restriction", 0.90);
    m.insert("trade sanction", 0.95);
    m.insert("supply chain disruption", 0.85);
    m.insert("nationalization", 1.00);
    m.insert("strategic reserve", 0.60);
    m.insert("tariff", 0.70);
    m.insert("duty increase", 0.75);
    m.insert("quota restriction", 0.80);
    m.insert("license requirement", 0.65);
    // Medium-risk
    m.insert("environmental regulation", 0.50);
    m.insert("emission standard", 0.45);
    m.insert("labor regulation", 0.40);
    m.insert("sustainability requirement", 0.35);
    m.insert("due diligence", 0.30);
    m.insert("esg compliance", 0.35);
    m.insert("carbon tax", 0.55);
    m.insert("mining permit", 0.50);
    // Positive (reduce risk)
    m.insert("free trade agreement", -0.30);
    m.insert("supply chain diversification", -0.40);
    m.insert("recycling initiative", -0.25);
    m.insert("innovation incentive", -0.20);
    m.insert("production increase", -0.30);
    m.insert("new mine", -0.35);
    m.insert("stock release", -0.25);
    m
}

/// Mock SEC filing excerpts for sentiment analysis (demo mode).
pub fn mock_sec_filings() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("LITHIUM", "\
    The company acknowledges significant supply chain risks related to lithium procurement. \
    While global lithium production has increased by 15% year-over-year, regulatory pressures \
    in key mining regions of Chile and Australia have created uncertainty. Export restrictions \
    in certain jurisdictions may impact our cost structure. The company is actively pursuing \
    supply chain diversification strategies and has secured long-term agreements with three \
    additional suppliers. Environmental regulations regarding mining operations continue to evolve, \
    with new sustainability requirements expected to increase compliance costs by approximately \
    8-12% over the next fiscal year. Despite these challenges, innovation incentives in battery \
    technology and recycling initiatives present opportunities for cost optimization.");
    m.insert("NICKEL", "\
    Nickel supply remains constrained due to the recent implementation of export restrictions \
    by Indonesia, the world's largest nickel producer. The trade sanction environment has created \
    additional uncertainty for our stainless steel and battery material divisions. Nationalization \
    risks in certain African mining jurisdictions have been discussed in recent government \
    proceedings. On the positive side, new mine development in Canada and Australia is expected \
    to come online by Q3 2026, potentially easing supply constraints. The company is investing \
    in recycling initiatives to reduce dependence on primary nickel supply. Tariff adjustments \
    on imported nickel products are expected to affect our cost basis by approximately 5-7%. \
    Environmental regulation compliance costs continue to trend upward.");
    m.insert("COBALT", "\
    Cobalt procurement faces heightened regulatory scrutiny following new due diligence \
    requirements under the EU Battery Regulation. Supply chain disruption risks remain elevated \
    due to geopolitical tensions in the Democratic Republic of Congo, which produces approximately \
    70% of global cobalt supply. The company has implemented comprehensive ESG compliance programs \
    and is transitioning to recycled cobalt sources, which now represent 12% of our cobalt input. \
    Export ban discussions in key producing nations have contributed to market volatility. \
    Production increases from new mining operations in Indonesia and Australia are expected to \
    partially offset supply constraints. Labor regulation changes in producing regions may impact \
    mining costs. The company's supply chain diversification efforts have reduced single-source \
    dependency from 45% to 32% over the past year.");
    m
}

/// Default HashKey Chain testnet configuration.
pub const DEFAULT_RPC_URL: &str = "https://hashkeychain-testnet.alt.technology";
pub const DEFAULT_CHAIN_ID: u64 = 133;
