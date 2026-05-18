//! Cross-Exchange Arbitrage — Signal comparison across dYdX, GMX, Synthetix.
//!
//! Aggregates market data from multiple perpetual futures exchanges, normalizes
//! it into canonical types, and detects arbitrage opportunities by comparing
//! funding rates, mark prices, and sentiment signals across venues.
//!
//! # Supported Exchanges
//!
//! - **dYdX** (v4) — primary exchange, full orderbook + funding data
//! - **GMX** (v2 on Arbitrum) — GLP pool price + funding
//! - **Synthetix** (Perps V2/V3 on Optimism/Base) — skew-based funding + oracle price
//!
//! # Arbitrage Detection
//!
//! The module computes two classes of opportunity:
//!
//! 1. **Price arbitrage** — directional spread between exchanges exceeds threshold
//! 2. **Funding arbitrage** — opposing funding rates allow carry-trade profit

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported exchange identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Exchange {
    Dydx,
    Gmx,
    Synthetix,
}

impl Exchange {
    pub fn as_str(&self) -> &'static str {
        match self {
            Exchange::Dydx => "DYDX",
            Exchange::Gmx => "GMX",
            Exchange::Synthetix => "SYNTHETIX",
        }
    }

    pub fn all() -> &'static [Exchange] {
        &[Exchange::Dydx, Exchange::Gmx, Exchange::Synthetix]
    }
}

/// Symbol mapping — how a perp pair maps across exchanges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMap {
    pub base_asset: String,
    /// Map of exchange → local ticker symbol.
    pub tickers: HashMap<Exchange, String>,
}

impl SymbolMap {
    /// Standard BTC perp mapping.
    pub fn btc() -> Self {
        let mut tickers = HashMap::new();
        tickers.insert(Exchange::Dydx, "BTC-USD".into());
        tickers.insert(Exchange::Gmx, "BTC".into());
        tickers.insert(Exchange::Synthetix, "sBTC".into());
        Self {
            base_asset: "BTC".into(),
            tickers,
        }
    }

    /// Standard ETH perp mapping.
    pub fn eth() -> Self {
        let mut tickers = HashMap::new();
        tickers.insert(Exchange::Dydx, "ETH-USD".into());
        tickers.insert(Exchange::Gmx, "ETH".into());
        tickers.insert(Exchange::Synthetix, "sETH".into());
        Self {
            base_asset: "ETH".into(),
            tickers,
        }
    }

    /// Get the ticker for a given exchange.
    pub fn ticker_for(&self, exchange: Exchange) -> Option<&str> {
        self.tickers.get(&exchange).map(|s| s.as_str())
    }
}

/// Normalized market snapshot from a single exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeSnapshot {
    pub exchange: Exchange,
    pub ticker: String,
    pub base_asset: String,
    pub mark_price: f64,
    pub index_price: f64,
    /// Annualized funding rate (%).
    pub funding_rate_annualized: f64,
    /// Open interest in USD.
    pub open_interest_usd: f64,
    /// 24h volume in USD.
    pub volume_24h_usd: f64,
    /// Bid/ask spread as a fraction of price.
    pub spread_bps: f64,
    /// Unix timestamp (ms) of the snapshot.
    pub timestamp_ms: i64,
}

impl ExchangeSnapshot {
    /// Compute the basis (mark - index) as a percentage of index price.
    pub fn basis_pct(&self) -> f64 {
        if self.index_price > 0.0 {
            (self.mark_price - self.index_price) / self.index_price * 100.0
        } else {
            0.0
        }
    }
}

/// A detected arbitrage opportunity between two exchanges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub base_asset: String,
    pub buy_exchange: Exchange,
    pub sell_exchange: Exchange,
    /// Price spread as a percentage.
    pub spread_pct: f64,
    /// Estimated profit after fees (bps).
    pub profit_after_fees_bps: f64,
    /// Buy side ticker.
    pub buy_ticker: String,
    /// Sell side ticker.
    pub sell_ticker: String,
    /// Whether this is a funding-rate carry trade.
    pub is_funding_arb: bool,
    /// Opportunity strength: LOW, MEDIUM, HIGH.
    pub strength: ArbitrageStrength,
    /// Unix timestamp (ms).
    pub timestamp_ms: i64,
}

/// Opportunity strength rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ArbitrageStrength {
    #[default]
    Low,
    Medium,
    High,
}

impl ArbitrageStrength {
    pub fn from_spread_pct(spread_pct: f64) -> Self {
        if spread_pct < 0.05 {
            ArbitrageStrength::Low
        } else if spread_pct < 0.15 {
            ArbitrageStrength::Medium
        } else {
            ArbitrageStrength::High
        }
    }
}

/// A funding-rate arbitrage opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbitrage {
    pub base_asset: String,
    /// Exchange with positive funding (go short here).
    pub short_exchange: Exchange,
    /// Exchange with negative funding (go long here).
    pub long_exchange: Exchange,
    /// Combined annualized yield from the carry (%).
    pub combined_yield_pct: f64,
    /// Risk that the funding rates converge before profit is captured.
    pub convergence_risk: RiskLevel,
    pub timestamp_ms: i64,
}

/// Cross-exchange comparison result for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossExchangeComparison {
    pub base_asset: String,
    pub snapshots: Vec<ExchangeSnapshot>,
    pub price_opportunities: Vec<ArbitrageOpportunity>,
    pub funding_opportunities: Vec<FundingArbitrage>,
    /// Aggregate consensus signal considering all exchanges.
    pub consensus_signal: Signal,
    /// Aggregate confidence across exchanges.
    pub consensus_confidence: f64,
}

/// Configuration for arbitrage detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    /// Minimum spread % to consider for price arbitrage.
    pub min_spread_pct: f64,
    /// Estimated round-trip trading fees in bps per exchange.
    pub fee_bps_per_exchange: f64,
    /// Minimum funding rate spread (annualized %) for carry trade.
    pub min_funding_spread_pct: f64,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_spread_pct: 0.03,
            fee_bps_per_exchange: 5.0,
            min_funding_spread_pct: 5.0,
        }
    }
}

/// The cross-exchange arbitrage engine.
pub struct ArbitrageEngine {
    config: ArbitrageConfig,
    /// Symbol mappings for supported assets.
    symbol_maps: HashMap<String, SymbolMap>,
}

impl Default for ArbitrageEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ArbitrageEngine {
    /// Create a new engine with default symbol maps and config.
    pub fn new() -> Self {
        let mut symbol_maps = HashMap::new();
        symbol_maps.insert("BTC".into(), SymbolMap::btc());
        symbol_maps.insert("ETH".into(), SymbolMap::eth());
        Self {
            config: ArbitrageConfig::default(),
            symbol_maps,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ArbitrageConfig) -> Self {
        let mut engine = Self::new();
        engine.config = config;
        engine
    }

    /// Add a custom symbol map.
    pub fn add_symbol_map(&mut self, asset: &str, map: SymbolMap) {
        self.symbol_maps.insert(asset.to_uppercase(), map);
    }

    /// Detect price arbitrage opportunities across exchanges for a single asset.
    pub fn detect_price_arbitrage(
        &self,
        snapshots: &[ExchangeSnapshot],
        timestamp_ms: i64,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        for i in 0..snapshots.len() {
            for j in (i + 1)..snapshots.len() {
                let a = &snapshots[i];
                let b = &snapshots[j];

                // Compute spread
                let spread_pct = (a.mark_price - b.mark_price).abs()
                    / a.mark_price.min(b.mark_price).max(f64::EPSILON)
                    * 100.0;

                if spread_pct < self.config.min_spread_pct {
                    continue;
                }

                // Determine buy/sell direction
                let (_buy, _sell) = if a.mark_price < b.mark_price {
                    (a, b)
                } else {
                    (b, a)
                };

                let total_fees_bps = self.config.fee_bps_per_exchange * 2.0;
                let spread_bps = spread_pct * 100.0;
                let profit_after_fees_bps = spread_bps - total_fees_bps;

                if profit_after_fees_bps <= 0.0 {
                    continue;
                }

                opportunities.push(ArbitrageOpportunity {
                    base_asset: a.base_asset.clone(),
                    buy_exchange: a.exchange,
                    sell_exchange: b.exchange,
                    spread_pct,
                    profit_after_fees_bps,
                    buy_ticker: a.ticker.clone(),
                    sell_ticker: b.ticker.clone(),
                    is_funding_arb: false,
                    strength: ArbitrageStrength::from_spread_pct(spread_pct),
                    timestamp_ms,
                });
            }
        }

        opportunities
    }

    /// Detect funding-rate arbitrage opportunities.
    pub fn detect_funding_arbitrage(
        &self,
        snapshots: &[ExchangeSnapshot],
        timestamp_ms: i64,
    ) -> Vec<FundingArbitrage> {
        let mut opportunities = Vec::new();

        for i in 0..snapshots.len() {
            for j in (i + 1)..snapshots.len() {
                let a = &snapshots[i];
                let b = &snapshots[j];

                // We want one exchange with high positive funding and one with low/negative
                // Go short on the high-funding exchange, long on the low/negative one
                let (short_side, long_side) = if a.funding_rate_annualized > b.funding_rate_annualized {
                    (a, b)
                } else {
                    (b, a)
                };

                let combined_yield = short_side.funding_rate_annualized - long_side.funding_rate_annualized;

                if combined_yield < self.config.min_funding_spread_pct {
                    continue;
                }

                let convergence_risk = if combined_yield > 20.0 {
                    RiskLevel::Low
                } else if combined_yield > 10.0 {
                    RiskLevel::Medium
                } else {
                    RiskLevel::High
                };

                opportunities.push(FundingArbitrage {
                    base_asset: a.base_asset.clone(),
                    short_exchange: short_side.exchange,
                    long_exchange: long_side.exchange,
                    combined_yield_pct: combined_yield,
                    convergence_risk,
                    timestamp_ms,
                });
            }
        }

        opportunities
    }

    /// Run a full cross-exchange comparison for an asset.
    pub fn compare(&self, asset: &str, snapshots: Vec<ExchangeSnapshot>) -> CrossExchangeComparison {
        let timestamp_ms = snapshots
            .first()
            .map(|s| s.timestamp_ms)
            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

        let price_opps = self.detect_price_arbitrage(&snapshots, timestamp_ms);
        let funding_opps = self.detect_funding_arbitrage(&snapshots, timestamp_ms);

        // Compute consensus signal from all exchanges
        let (consensus_signal, consensus_confidence) =
            self.compute_cross_exchange_consensus(&snapshots);

        CrossExchangeComparison {
            base_asset: asset.to_uppercase(),
            snapshots,
            price_opportunities: price_opps,
            funding_opportunities: funding_opps,
            consensus_signal,
            consensus_confidence,
        }
    }

    /// Compute an aggregate consensus signal from multiple exchange snapshots.
    ///
    /// Uses funding rate direction, basis, and volume to produce a
    /// cross-exchange weighted signal.
    fn compute_cross_exchange_consensus(
        &self,
        snapshots: &[ExchangeSnapshot],
    ) -> (Signal, f64) {
        if snapshots.is_empty() {
            return (Signal::Neutral, 0.0);
        }

        let mut long_score = 0.0f64;
        let mut short_score = 0.0f64;
        let mut total_weight = 0.0f64;

        for snap in snapshots {
            // Weight by volume (log-scaled to avoid dominance)
            let weight = (snap.volume_24h_usd.log10().max(0.0)).max(1.0);
            total_weight += weight;

            // Funding rate signal: high positive = crowded longs → short
            if snap.funding_rate_annualized > 15.0 {
                short_score += weight * (snap.funding_rate_annualized / 30.0).min(1.0);
            } else if snap.funding_rate_annualized < -5.0 {
                long_score += weight * (snap.funding_rate_annualized.abs() / 20.0).min(1.0);
            }

            // Basis signal: positive basis = overvalued → short
            let basis = snap.basis_pct();
            if basis > 0.1 {
                short_score += weight * (basis / 0.5).min(1.0);
            } else if basis < -0.1 {
                long_score += weight * (basis.abs() / 0.5).min(1.0);
            }
        }

        if total_weight == 0.0 {
            return (Signal::Neutral, 0.0);
        }

        let signal = if long_score > short_score * 1.5 {
            Signal::Long
        } else if short_score > long_score * 1.5 {
            Signal::Short
        } else {
            Signal::Neutral
        };

        let confidence = ((long_score.max(short_score) / total_weight) * 50.0)
            .clamp(10.0, 85.0);

        (signal, confidence)
    }
}

/// Generate a mock exchange snapshot for testing.
pub fn mock_snapshot(exchange: Exchange, base_asset: &str, price: f64, funding_annual: f64) -> ExchangeSnapshot {
    ExchangeSnapshot {
        exchange,
        ticker: format!("{}-USD", base_asset),
        base_asset: base_asset.to_uppercase(),
        mark_price: price,
        index_price: price * (1.0 - funding_annual / 10000.0),
        funding_rate_annualized: funding_annual,
        open_interest_usd: 500_000_000.0,
        volume_24h_usd: 100_000_000.0,
        spread_bps: 1.5,
        timestamp_ms: chrono::Utc::now().timestamp_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_as_str() {
        assert_eq!(Exchange::Dydx.as_str(), "DYDX");
        assert_eq!(Exchange::Gmx.as_str(), "GMX");
        assert_eq!(Exchange::Synthetix.as_str(), "SYNTHETIX");
    }

    #[test]
    fn test_symbol_map_btc() {
        let map = SymbolMap::btc();
        assert_eq!(map.base_asset, "BTC");
        assert_eq!(map.ticker_for(Exchange::Dydx), Some("BTC-USD"));
        assert_eq!(map.ticker_for(Exchange::Gmx), Some("BTC"));
    }

    #[test]
    fn test_snapshot_basis_pct() {
        let snap = mock_snapshot(Exchange::Dydx, "BTC", 67050.0, 10.0);
        // index_price = 67050 * (1 - 10/10000) ≈ 67050 * 0.999 ≈ 66983.05
        // basis = (67050 - 66983.05) / 66983.05 * 100 ≈ 0.1%
        let basis = snap.basis_pct();
        assert!(basis > 0.0);
        assert!(basis < 1.0);
    }

    #[test]
    fn test_detect_price_arbitrage_significant_spread_no_profit() {
        let engine = ArbitrageEngine::new();
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "BTC", 67000.0, 10.0),
            mock_snapshot(Exchange::Gmx, "BTC", 67050.0, 8.0),  // 0.075% spread
        ];

        let opps = engine.detect_price_arbitrage(&snapshots, 0);
        // Spread is 0.075% but after fees (10 bps) there's no profit, so filtered out
        assert_eq!(opps.len(), 0);
    }

    #[test]
    fn test_detect_price_arbitrage_profitable() {
        let mut engine = ArbitrageEngine::new();
        engine.config.fee_bps_per_exchange = 2.0; // Lower fees
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "BTC", 67000.0, 10.0),
            mock_snapshot(Exchange::Gmx, "BTC", 67100.0, 8.0), // ~0.15% spread
        ];

        let opps = engine.detect_price_arbitrage(&snapshots, 0);
        assert_eq!(opps.len(), 1);
        // spread_pct ≈ 0.149, spread_bps ≈ 14.9, fees = 4, profit ≈ 10.9
        assert!(opps[0].profit_after_fees_bps > 0.0);
        // 67100/67000 - 1 = 0.149%, ArbitrageStrength threshold: 0.15 → MEDIUM, not HIGH
        assert_eq!(opps[0].strength, ArbitrageStrength::Medium);
    }

    #[test]
    fn test_detect_price_arbitrage_no_opportunity() {
        let engine = ArbitrageEngine::new();
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "BTC", 67000.0, 10.0),
            mock_snapshot(Exchange::Gmx, "BTC", 67001.0, 10.0), // Tiny spread
        ];

        let opps = engine.detect_price_arbitrage(&snapshots, 0);
        assert!(opps.is_empty());
    }

    #[test]
    fn test_detect_funding_arbitrage() {
        let engine = ArbitrageEngine::new();
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "ETH", 3500.0, 25.0),  // high positive
            mock_snapshot(Exchange::Gmx, "ETH", 3500.0, 5.0),   // low
            mock_snapshot(Exchange::Synthetix, "ETH", 3500.0, -5.0), // negative
        ];

        let opps = engine.detect_funding_arbitrage(&snapshots, 0);
        assert!(opps.len() >= 2); // DYDX-GMX and DYDX-Synth pairs

        // Should have one with short=DYDX, long=Synthetix
        let best = opps.iter().max_by_key(|o| o.combined_yield_pct as i64).unwrap();
        assert_eq!(best.short_exchange, Exchange::Dydx);
        assert_eq!(best.long_exchange, Exchange::Synthetix);
        assert_eq!(best.combined_yield_pct, 30.0); // 25 - (-5)
    }

    #[test]
    fn test_detect_funding_arbitrage_below_threshold() {
        let engine = ArbitrageEngine::new();
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "BTC", 67000.0, 8.0),
            mock_snapshot(Exchange::Gmx, "BTC", 67000.0, 5.0),
        ];

        let opps = engine.detect_funding_arbitrage(&snapshots, 0);
        // Combined yield = 3.0, below threshold of 5.0
        assert!(opps.is_empty());
    }

    #[test]
    fn test_full_cross_exchange_comparison() {
        let engine = ArbitrageEngine::new();
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "BTC", 67000.0, 20.0),
            mock_snapshot(Exchange::Gmx, "BTC", 67030.0, 10.0),
            mock_snapshot(Exchange::Synthetix, "BTC", 66980.0, -5.0),
        ];

        let result = engine.compare("BTC", snapshots);
        assert_eq!(result.base_asset, "BTC");
        assert_eq!(result.snapshots.len(), 3);
        // Should detect funding arb (DYDX 20 vs Synthetix -5)
        assert!(!result.funding_opportunities.is_empty());
        // Price arb may or may not be profitable depending on spread vs fees
        // (Synthetix 66980 vs GMX 67030 = 0.075% spread, but 10 bps fees)
    }

    #[test]
    fn test_cross_exchange_consensus_short_signal() {
        let engine = ArbitrageEngine::new();
        // All exchanges have high positive funding → crowded longs
        let snapshots = vec![
            mock_snapshot(Exchange::Dydx, "BTC", 67000.0, 30.0),
            mock_snapshot(Exchange::Gmx, "BTC", 67000.0, 25.0),
        ];

        let result = engine.compare("BTC", snapshots);
        assert_eq!(result.consensus_signal, Signal::Short);
        assert!(result.consensus_confidence > 10.0);
    }

    #[test]
    fn test_cross_exchange_consensus_empty() {
        let engine = ArbitrageEngine::new();
        let result = engine.compare("BTC", vec![]);
        assert_eq!(result.consensus_signal, Signal::Neutral);
        assert_eq!(result.consensus_confidence, 0.0);
    }

    #[test]
    fn test_arbitrage_strength_classification() {
        assert_eq!(ArbitrageStrength::from_spread_pct(0.02), ArbitrageStrength::Low);
        assert_eq!(ArbitrageStrength::from_spread_pct(0.05), ArbitrageStrength::Medium);
        assert_eq!(ArbitrageStrength::from_spread_pct(0.1), ArbitrageStrength::Medium);
        assert_eq!(ArbitrageStrength::from_spread_pct(0.2), ArbitrageStrength::High);
    }

    #[test]
    fn test_arbitrage_config_default() {
        let cfg = ArbitrageConfig::default();
        assert_eq!(cfg.min_spread_pct, 0.03);
        assert_eq!(cfg.fee_bps_per_exchange, 5.0);
        assert_eq!(cfg.min_funding_spread_pct, 5.0);
    }

    #[test]
    fn test_exchange_snapshot_serde() {
        let snap = mock_snapshot(Exchange::Dydx, "ETH", 3500.0, 12.0);
        let json = serde_json::to_string(&snap).unwrap();
        let restored: ExchangeSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.exchange, Exchange::Dydx);
        assert_eq!(restored.base_asset, "ETH");
        assert_eq!(restored.mark_price, 3500.0);
    }

    #[test]
    fn test_arbitrage_opportunity_serde() {
        let opp = ArbitrageOpportunity {
            base_asset: "BTC".into(),
            buy_exchange: Exchange::Dydx,
            sell_exchange: Exchange::Gmx,
            spread_pct: 0.15,
            profit_after_fees_bps: 5.0,
            buy_ticker: "BTC-USD".into(),
            sell_ticker: "BTC".into(),
            is_funding_arb: false,
            strength: ArbitrageStrength::High,
            timestamp_ms: 1704067200000,
        };
        let json = serde_json::to_string(&opp).unwrap();
        // Exchange variants serialize with their Rust names: Dydx, Gmx
        assert!(json.contains("Dydx") || json.contains("DYDX"));
        assert!(json.contains("Gmx") || json.contains("GMX"));
    }
}
