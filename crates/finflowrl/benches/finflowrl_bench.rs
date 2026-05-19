//! Tier 2 benchmarks for finflowrl
#![allow(clippy::all)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use finflowrl::models::{MeanFlowPolicy, NoisePolicy};
use finflowrl::experts::{GLFTExpert, AvellanedaStoikovExpert};
use finflowrl::simulator::MarketSimulator;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

fn bench_meanflow_forward(c: &mut Criterion) {
    let policy = MeanFlowPolicy::new_default(6, 1);
    let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let x_t = Array1::from_vec(vec![0.5]);

    let mut group = c.benchmark_group("models/MeanFlowPolicy");
    group.bench_function("velocity_network", |b| {
        b.iter(|| policy.velocity_network(black_box(&x_t), black_box(0.5), black_box(&obs)))
    });
    group.bench_function("act_deterministic", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| policy.act(black_box(&obs), &mut rng, black_box(true)))
    });
    group.bench_function("act_stochastic", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| policy.act(black_box(&obs), &mut rng, black_box(false)))
    });
    group.bench_function("flow_loss", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        let expert_action = Array1::from_vec(vec![0.75]);
        b.iter(|| policy.flow_loss(black_box(&obs), black_box(&expert_action), &mut rng))
    });
    group.finish();
}

fn bench_meanflow_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("models/MeanFlowPolicy");
    group.bench_function("new_default_6x1", |b| {
        b.iter(|| MeanFlowPolicy::new_default(black_box(6), black_box(1)))
    });
    group.bench_function("new_custom_12x3", |b| {
        b.iter(|| MeanFlowPolicy::new(black_box(12), black_box(3), vec![256, 128, 64], 20))
    });
    group.finish();
}

fn bench_noise_policy(c: &mut Criterion) {
    let policy = NoisePolicy::new(6, 2, 64, 0.1);
    let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);

    let mut group = c.benchmark_group("models/NoisePolicy");
    group.bench_function("forward", |b| {
        b.iter(|| policy.forward(black_box(&obs)))
    });
    group.bench_function("act_stochastic", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| policy.act(black_box(&obs), &mut rng))
    });
    group.bench_function("get_mean", |b| {
        b.iter(|| policy.get_mean(black_box(&obs)))
    });
    group.finish();
}

fn bench_glft_expert(c: &mut Criterion) {
    let expert = GLFTExpert::new(6, 0.1, 10.0);
    let mut state = HashMap::new();
    state.insert("inventory".to_string(), 3.0);
    state.insert("mid_price".to_string(), 100.0);
    state.insert("prev_mid_price".to_string(), 99.5);
    state.insert("mid_price_change".to_string(), 0.5);
    state.insert("spread".to_string(), 0.02);
    state.insert("volatility".to_string(), 0.02);
    state.insert("order_imbalance".to_string(), 0.1);
    state.insert("hawkes_intensity".to_string(), 5.0);

    let mut group = c.benchmark_group("experts/GLFTExpert");
    group.bench_function("act", |b| {
        b.iter(|| expert.act(black_box(&state)))
    });
    group.bench_function("extract_features", |b| {
        b.iter(|| expert.extract_features(black_box(&state)))
    });
    group.finish();
}

fn bench_avellaneda_stoikov(c: &mut Criterion) {
    let expert = AvellanedaStoikovExpert::new(0.1, 0.02, 60.0, 1.5);

    let mut group = c.benchmark_group("experts/AvellanedaStoikov");
    group.bench_function("act", |b| {
        b.iter(|| expert.act(black_box(100.0), black_box(5.0), black_box(10.0)))
    });
    group.bench_function("get_reservation_price", |b| {
        b.iter(|| expert.get_reservation_price(black_box(100.0), black_box(5.0), black_box(10.0)))
    });
    group.bench_function("get_spread", |b| {
        b.iter(|| expert.get_spread(black_box(10.0)))
    });
    group.finish();
}

fn bench_market_simulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulator/MarketSimulator");
    group.bench_function("single_step", |b| {
        let mut sim = MarketSimulator::new(42);
        sim.reset(None);
        b.iter(|| sim.step())
    });
    group.bench_function("simulate_100_steps", |b| {
        b.iter(|| {
            let mut sim = MarketSimulator::new(42);
            black_box(sim.simulate(100))
        })
    });
    group.bench_function("simulate_1000_steps", |b| {
        b.iter(|| {
            let mut sim = MarketSimulator::new(42);
            black_box(sim.simulate(1000))
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_meanflow_forward,
    bench_meanflow_creation,
    bench_noise_policy,
    bench_glft_expert,
    bench_avellaneda_stoikov,
    bench_market_simulator,
);
criterion_main!(benches);
