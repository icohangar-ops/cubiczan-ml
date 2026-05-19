//! Tier 2 benchmarks for critmin-oracle
#![allow(clippy::all)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use critmin_oracle_lib::*;

fn bench_sentiment_analysis(c: &mut Criterion) {
    // Realistic SEC filing text (from mock data)
    let sec_filing = "The company acknowledges significant supply chain risks related to lithium procurement. \
        While global lithium production has increased by 15% year-over-year, regulatory pressures \
        in key mining regions of Chile and Australia have created uncertainty. Export restrictions \
        in certain jurisdictions may impact our cost structure. The company is actively pursuing \
        supply chain diversification strategies and has secured long-term agreements with three \
        additional suppliers. Environmental regulations regarding mining operations continue to evolve, \
        with new sustainability requirements expected to increase compliance costs by approximately \
        8-12% over the next fiscal year. Despite these challenges, innovation incentives in battery \
        technology and recycling initiatives present opportunities for cost optimization.";

    let short_text = "The weather is nice today.";
    let negative_text = "Risks are elevated and supply chain disruption threatens operations with severe \
        nationalization risk and trade sanctions creating uncertainty.";

    let mut group = c.benchmark_group("sentiment/simple_sentiment_analyzer");
    group.bench_function("sec_filing_500_chars", |b| {
        b.iter(|| simple_sentiment_analyzer(black_box(sec_filing)))
    });
    group.bench_function("short_neutral_text", |b| {
        b.iter(|| simple_sentiment_analyzer(black_box(short_text)))
    });
    group.bench_function("negative_risk_text", |b| {
        b.iter(|| simple_sentiment_analyzer(black_box(negative_text)))
    });
    group.finish();
}

fn bench_regulatory_risk(c: &mut Criterion) {
    let high_risk_text = "export ban and nationalization risk with trade sanctions and supply chain disruption. \
        Tariff adjustments on imported products create additional uncertainty. \
        Environmental regulation compliance costs continue to trend upward.";

    let low_risk_text = "The weather is nice today.";
    let positive_text = "free trade agreement and supply chain diversification initiative with \
        innovation incentive and production increase from new mine development.";

    let mut group = c.benchmark_group("sentiment/regulatory_risk_scorer");
    group.bench_function("high_risk_text", |b| {
        b.iter(|| regulatory_risk_scorer(black_box(high_risk_text)))
    });
    group.bench_function("low_risk_baseline", |b| {
        b.iter(|| regulatory_risk_scorer(black_box(low_risk_text)))
    });
    group.bench_function("positive_keywords", |b| {
        b.iter(|| regulatory_risk_scorer(black_box(positive_text)))
    });
    group.finish();
}

fn bench_price_forecast(c: &mut Criterion) {
    // Generate a realistic 24-point price history
    let history = generate_mock_price_history("LITHIUM", 24);
    let short_history = generate_mock_price_history("NICKEL", 2);

    let mut group = c.benchmark_group("forecast/compute_price_forecast");
    group.bench_function("24_periods", |b| {
        b.iter(|| compute_price_forecast(black_box(&history)))
    });
    group.bench_function("short_history_2_periods", |b| {
        b.iter(|| compute_price_forecast(black_box(&short_history)))
    });
    group.finish();

    c.benchmark_group("forecast/generate_mock_price_history")
        .bench_function("24_periods_lithium", |b| {
            b.iter(|| generate_mock_price_history(black_box("LITHIUM"), 24))
        });
}

fn bench_keccak_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/crypto");
    group.bench_function("mineral_hash_bytes", |b| {
        b.iter(|| mineral_hash(black_box("LITHIUM")))
    });
    group.bench_function("mineral_hash_hex", |b| {
        b.iter(|| mineral_hash_hex(black_box("LITHIUM")))
    });
    group.bench_function("scale_price", |b| {
        b.iter(|| scale_price(black_box(15_250.50)))
    });
    group.bench_function("scale_composite", |b| {
        b.iter(|| scale_composite(black_box(42.5)))
    });
    group.bench_function("scale_sentiment", |b| {
        b.iter(|| scale_sentiment(black_box(0.73)))
    });
    group.bench_function("scale_reg_risk", |b| {
        b.iter(|| scale_reg_risk(black_box(65.0)))
    });
    group.finish();
}

fn bench_generate_mock_prices(c: &mut Criterion) {
    c.benchmark_group("prices")
        .bench_function("generate_mock_prices", |b| {
            b.iter(|| generate_mock_prices())
        });
}

criterion_group!(
    benches,
    bench_sentiment_analysis,
    bench_regulatory_risk,
    bench_price_forecast,
    bench_keccak_hashing,
    bench_generate_mock_prices,
);
criterion_main!(benches);
