//! Tier 2 benchmarks for swarmfi-perps
#![allow(clippy::all)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use swarmfi_perps::*;

/// Build a realistic MarketDataBundle for benchmarking agents.
fn make_bench_bundle() -> MarketDataBundle {
    let candles: Vec<Candle> = (0..20)
        .map(|i| {
            let base = 65000.0 + (i as f64) * 50.0;
            Candle {
                started_at: format!("2025-01-01T{:02}:00:00Z", i),
                open: base,
                high: base + 100.0,
                low: base - 50.0,
                close: base + 75.0,
                base_token_volume: 100.0,
                usd_volume: 6_500_000.0 + i as f64 * 100_000.0,
                trades: 500,
            }
        })
        .collect();

    MarketDataBundle {
        orderbook: Some(Orderbook {
            bids: (0..10)
                .map(|i| OrderbookLevel {
                    price: 66000.0 - (i as f64) * 100.0,
                    size: 1.0 + (i as f64) * 0.5,
                })
                .collect(),
            asks: (0..10)
                .map(|i| OrderbookLevel {
                    price: 66100.0 + (i as f64) * 100.0,
                    size: 0.5 + (i as f64) * 0.3,
                })
                .collect(),
        }),
        trades: (0..100)
            .map(|i| Trade {
                side: if i % 3 == 0 { TradeSide::Sell } else { TradeSide::Buy },
                size: 0.01 + (i as f64) * 0.01,
                price: 66000.0 + (i as f64 % 10.0) * 5.0,
                created_at: 1704067200.0 + i as f64 * 10.0,
            })
            .collect(),
        candles,
        funding: vec![
            FundingEntry { rate: "0.0001".into(), effective_at: "2025-01-01T00:00:00Z".into(), price: "66000".into() },
            FundingEntry { rate: "0.00015".into(), effective_at: "2025-01-01T01:00:00Z".into(), price: "66000".into() },
            FundingEntry { rate: "0.0002".into(), effective_at: "2025-01-01T02:00:00Z".into(), price: "66050".into() },
        ],
        market: Some(MarketInfo {
            ticker: "BTC-USD".into(),
            oracle_price: "66000".into(),
            open_interest: "500000000".into(),
            volume_24h: "1000000000".into(),
            next_funding_time: "2025-01-01T01:00:00Z".into(),
        }),
        stats: MarketStats {
            mid_price: 66050.0,
            spread: 100.0,
            volume_24h: 1_000_000_000.0,
            open_interest: 500_000_000.0,
            funding_rate_1h: 87.6,
        },
    }
}

fn bench_agent_consensus(c: &mut Criterion) {
    let bundle = make_bench_bundle();

    let mut group = c.benchmark_group("agents/run_all_agents");
    group.bench_function("9_agents_full_bundle", |b| {
        b.iter(|| run_all_agents(black_box(&bundle)))
    });
    group.finish();

    let votes = run_all_agents(&bundle);

    let mut group2 = c.benchmark_group("consensus/compute_consensus");
    group2.bench_function("9_votes", |b| {
        b.iter(|| compute_consensus(black_box(&votes), black_box(None)))
    });
    group2.bench_function("9_votes_with_prev_board", |b| {
        let mut prev = StigmergyBoard::default();
        for v in &votes {
            prev.last_signals.insert(v.agent_type.clone(), v.signal.as_str().to_string());
        }
        b.iter(|| compute_consensus(black_box(&votes), black_box(Some(&prev))))
    });
    group2.finish();

    c.benchmark_group("consensus/run_consensus")
        .bench_function("full_pipeline", |b| {
            let votes = run_all_agents(&bundle);
            b.iter(|| run_consensus(votes.clone(), "BTC-USD", black_box(None)))
        });
}

fn bench_individual_agents(c: &mut Criterion) {
    let bundle = make_bench_bundle();

    let mut group = c.benchmark_group("agents/individual");
    group.bench_function("funding_agent", |b| {
        b.iter(|| funding_agent(black_box(&bundle)))
    });
    group.bench_function("momentum_agent", |b| {
        b.iter(|| momentum_agent(black_box(&bundle)))
    });
    group.bench_function("volatility_agent", |b| {
        b.iter(|| volatility_agent(black_box(&bundle)))
    });
    group.bench_function("volume_agent", |b| {
        b.iter(|| volume_agent(black_box(&bundle)))
    });
    group.bench_function("orderbook_agent", |b| {
        b.iter(|| orderbook_agent(black_box(&bundle)))
    });
    group.bench_function("liquidation_agent", |b| {
        b.iter(|| liquidation_agent(black_box(&bundle)))
    });
    group.bench_function("mean_reversion_agent", |b| {
        b.iter(|| mean_reversion_agent(black_box(&bundle)))
    });
    group.bench_function("trend_agent", |b| {
        b.iter(|| trend_agent(black_box(&bundle)))
    });
    group.finish();
}

fn bench_sentiment_agent(c: &mut Criterion) {
    let bundle = make_bench_bundle();
    let core_votes = vec![
        funding_agent(&bundle),
        momentum_agent(&bundle),
        volatility_agent(&bundle),
        volume_agent(&bundle),
        orderbook_agent(&bundle),
        liquidation_agent(&bundle),
        mean_reversion_agent(&bundle),
        trend_agent(&bundle),
    ];

    c.benchmark_group("agents/sentiment_agent")
        .bench_function("synthesize_8_votes", |b| {
            b.iter(|| sentiment_agent(black_box(&bundle), black_box(&core_votes)))
        });
}

fn bench_math_utilities(c: &mut Criterion) {
    use swarmfi_perps::math::{clamp, sma, std_dev};

    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1 + (i as f64 % 7.0)).collect();

    let mut group = c.benchmark_group("math");
    group.bench_function("sma_1000_values", |b| {
        b.iter(|| sma(black_box(&values)))
    });
    group.bench_function("std_dev_1000_values", |b| {
        b.iter(|| std_dev(black_box(&values)))
    });
    group.bench_function("clamp", |b| {
        b.iter(|| clamp(black_box(15.0), 0.0, 10.0))
    });
    group.finish();
}

fn bench_agent_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus/agent_weights");
    group.bench_function("weights_hashmap", |b| {
        b.iter(|| agent_weights())
    });
    group.bench_function("weight_descriptions", |b| {
        b.iter(|| agent_weight_descriptions())
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_agent_consensus,
    bench_individual_agents,
    bench_sentiment_agent,
    bench_math_utilities,
    bench_agent_weights,
);
criterion_main!(benches);
