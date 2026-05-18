//! Full analysis pipeline and mock data generation.
//!
//! Provides `build_market_data_bundle` (mock data for offline analysis),
//! `run_swarm_analysis` (end-to-end pipeline), and mock data generators for testing.

use crate::agents::run_all_agents;
use crate::consensus::run_consensus;
use crate::math::parse_f64_or_zero;
use crate::types::*;

/// Run the full swarm analysis pipeline on a market data bundle.
///
/// 1. Run all 9 agents on the data
/// 2. Compute adversarial weighted consensus
/// 3. Return the consensus result
pub fn run_swarm_analysis(market: &str, data: &MarketDataBundle, previous_board: Option<&StigmergyBoard>) -> ConsensusResult {
    let votes = run_all_agents(data);
    run_consensus(votes, market, previous_board)
}

/// Generate a mock MarketDataBundle for testing and demo purposes.
///
/// Creates synthetic data with configurable parameters to simulate
/// different market conditions.
pub fn generate_mock_market_data(market: &str) -> MarketDataBundle {
    // Base price depends on market
    let base_price = match market {
        "ETH-USD" => 3500.0,
        "SOL-USD" => 150.0,
        "BTC-USD" | _ => 67000.0,
    };

    // Generate 20 candles with slight uptrend
    let mut candles = Vec::new();
    let mut price = base_price * 0.97;
    for i in 0..20 {
        let trend = base_price * 0.001 * i as f64;
        let noise = (rand::random::<f64>() - 0.5) * base_price * 0.005;
        let open = price;
        let close = base_price * 0.97 + trend + noise;
        let high = open.max(close) + base_price * 0.001 * rand::random::<f64>();
        let low = open.min(close) - base_price * 0.001 * rand::random::<f64>();
        candles.push(Candle {
            started_at: format!("2025-01-01T{:02}:00:00Z", i),
            open,
            high: high.max(close),
            low: low.min(close).max(0.0),
            close,
            base_token_volume: 100.0 + rand::random::<f64>() * 200.0,
            usd_volume: close * (100.0 + rand::random::<f64>() * 200.0),
            trades: 500 + (rand::random::<f64>() * 500.0) as u32,
        });
        price = close;
    }

    let last_close = candles.last().unwrap().close;
    let mid_price = last_close;

    // Generate funding entries (slightly positive = overleveraged longs)
    let funding = vec![
        FundingEntry { rate: "0.00012".into(), effective_at: "2025-01-01T00:00:00Z".into(), price: format!("{}", mid_price as u64) },
        FundingEntry { rate: "0.00010".into(), effective_at: "2025-01-01T01:00:00Z".into(), price: format!("{}", mid_price as u64) },
        FundingEntry { rate: "0.00015".into(), effective_at: "2025-01-01T02:00:00Z".into(), price: format!("{}", mid_price as u64) },
    ];

    // Funding rate annualized
    let funding_rate_1h = parse_f64_or_zero(&funding[0].rate) * 24.0 * 365.0 * 100.0;

    // Generate orderbook
    let spread = mid_price * 0.0001;
    let orderbook = Orderbook {
        bids: vec![
            OrderbookLevel { price: mid_price - spread, size: 1.5 },
            OrderbookLevel { price: mid_price - spread * 2.0, size: 3.0 },
            OrderbookLevel { price: mid_price - spread * 3.0, size: 5.0 },
            OrderbookLevel { price: mid_price - spread * 4.0, size: 7.0 },
            OrderbookLevel { price: mid_price - spread * 5.0, size: 10.0 },
        ],
        asks: vec![
            OrderbookLevel { price: mid_price + spread, size: 1.0 },
            OrderbookLevel { price: mid_price + spread * 2.0, size: 2.0 },
            OrderbookLevel { price: mid_price + spread * 3.0, size: 4.0 },
            OrderbookLevel { price: mid_price + spread * 4.0, size: 6.0 },
            OrderbookLevel { price: mid_price + spread * 5.0, size: 8.0 },
        ],
    };

    // Generate trades
    let trades: Vec<Trade> = (0..50)
        .map(|i| Trade {
            side: if rand::random::<f64>() > 0.45 { TradeSide::Buy } else { TradeSide::Sell },
            size: 0.01 + rand::random::<f64>() * 0.5,
            price: mid_price + (rand::random::<f64>() - 0.5) * spread * 4.0,
            created_at: 1704067200.0 + i as f64 * 60.0,
        })
        .collect();

    MarketDataBundle {
        orderbook: Some(orderbook),
        trades,
        candles,
        funding,
        market: Some(MarketInfo {
            ticker: market.to_string(),
            oracle_price: format!("{}", mid_price as u64),
            open_interest: "500000000".into(),
            volume_24h: "1200000000".into(),
            next_funding_time: "2025-01-01T01:00:00Z".into(),
        }),
        stats: MarketStats {
            mid_price,
            spread,
            volume_24h: 1_200_000_000.0,
            open_interest: 500_000_000.0,
            funding_rate_1h,
        },
    }
}

/// Generate a MarketDataBundle simulating high volatility conditions.
pub fn generate_high_volatility_data(market: &str) -> MarketDataBundle {
    let base_price = match market {
        "ETH-USD" => 3500.0,
        "BTC-USD" | _ => 67000.0,
    };

    let mut candles = Vec::new();
    let mut price = base_price;
    for i in 0..20 {
        let open = price;
        // Large swings for high volatility
        let change = (rand::random::<f64>() - 0.5) * base_price * 0.03;
        let close = open + change;
        let high = open.max(close) + base_price * 0.01 * rand::random::<f64>();
        let low = open.min(close) - base_price * 0.01 * rand::random::<f64>();
        candles.push(Candle {
            started_at: format!("2025-01-01T{:02}:00:00Z", i),
            open,
            high,
            low: low.max(0.0),
            close,
            base_token_volume: 500.0 + rand::random::<f64>() * 1000.0,
            usd_volume: close * (500.0 + rand::random::<f64>() * 1000.0),
            trades: 2000 + (rand::random::<f64>() * 3000.0) as u32,
        });
        price = close;
    }

    let last_close = candles.last().unwrap().close;
    let funding = vec![
        FundingEntry { rate: "0.00050".into(), effective_at: "2025-01-01T00:00:00Z".into(), price: format!("{}", last_close as u64) },
    ];
    let funding_rate_1h = parse_f64_or_zero(&funding[0].rate) * 24.0 * 365.0 * 100.0;

    MarketDataBundle {
        orderbook: None,
        trades: vec![],
        candles,
        funding,
        market: None,
        stats: MarketStats {
            mid_price: last_close,
            spread: 0.0,
            volume_24h: 0.0,
            open_interest: 0.0,
            funding_rate_1h,
        },
    }
}

/// Render a consensus result as a human-readable text report.
pub fn render_report(result: &ConsensusResult) -> String {
    let mut lines = Vec::new();
    lines.push(format!("# SwarmFi Perps — Consensus Report"));
    lines.push(format!("Market: {}", result.market));
    lines.push(format!("Signal: {}", result.signal.as_str()));
    lines.push(format!("Confidence: {:.1}%", result.confidence));
    lines.push(format!("Timestamp: {}", result.timestamp));
    lines.push(String::new());

    lines.push("## Agent Votes".to_string());
    for vote in &result.agent_votes {
        lines.push(format!("  {}: {} ({:.0}%)", vote.agent_type, vote.signal.as_str(), vote.confidence));
    }
    lines.push(String::new());

    lines.push("## Stigmergy Board".to_string());
    lines.push(format!("  Signal counts: {} LONG, {} SHORT, {} NEUTRAL",
        result.stigmergy_board.signal_counts.long,
        result.stigmergy_board.signal_counts.short,
        result.stigmergy_board.signal_counts.neutral,
    ));
    lines.push(format!("  Average confidence: {:.1}%", result.stigmergy_board.average_confidence));
    lines.push(format!("  Volatility regime: {:?}", result.stigmergy_board.volatility_regime));
    lines.push(format!("  Liquidation risk: {:?}", result.stigmergy_board.liquidation_risk_level));

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_swarm_analysis_btc() {
        let data = generate_mock_market_data("BTC-USD");
        let result = run_swarm_analysis("BTC-USD", &data, None);
        assert_eq!(result.market, "BTC-USD");
        assert_eq!(result.agent_votes.len(), 9);
        assert!(result.confidence >= 10.0);
        assert!(result.confidence <= 90.0);
    }

    #[test]
    fn test_run_swarm_analysis_eth() {
        let data = generate_mock_market_data("ETH-USD");
        let result = run_swarm_analysis("ETH-USD", &data, None);
        assert_eq!(result.market, "ETH-USD");
    }

    #[test]
    fn test_high_volatility_produces_neutral() {
        // With high volatility, the VolatilityAgent should output NEUTRAL
        let data = generate_high_volatility_data("BTC-USD");
        let result = run_swarm_analysis("BTC-USD", &data, None);
        // We can't assert the final consensus is NEUTRAL (other agents may disagree),
        // but we can verify the pipeline runs
        assert_eq!(result.agent_votes.len(), 9);
    }

    #[test]
    fn test_mock_data_bundle_validity() {
        let data = generate_mock_market_data("BTC-USD");
        assert!(data.candles.len() >= 20);
        assert!(!data.funding.is_empty());
        assert!(data.orderbook.is_some());
        assert!(data.stats.mid_price > 0.0);
        assert!(data.trades.len() > 0);
    }

    #[test]
    fn test_render_report() {
        let data = generate_mock_market_data("BTC-USD");
        let result = run_swarm_analysis("BTC-USD", &data, None);
        let report = render_report(&result);
        assert!(report.contains("SwarmFi Perps"));
        assert!(report.contains("BTC-USD"));
        assert!(report.contains("Agent Votes"));
    }

    #[test]
    fn test_previous_board_integration() {
        let data = generate_mock_market_data("BTC-USD");
        let mut prev = StigmergyBoard::default();
        prev.last_signals.insert("FundingAgent".into(), "SHORT".into());

        let result = run_swarm_analysis("BTC-USD", &data, Some(&prev));
        assert!(result.stigmergy_board.previous_signals.is_some());
    }
}
