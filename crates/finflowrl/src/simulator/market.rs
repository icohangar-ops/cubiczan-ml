/// Market Simulator — Jump-diffusion process + Hawkes process for order flow.
///
/// Models realistic market microstructure:
///   - Mid-price follows Merton jump-diffusion
///   - Order arrivals follow a self-exciting Hawkes process
///   - Supports configurable spread, volatility, and intensity parameters

use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, Poisson, StandardNormal};

/// Market state returned by a single simulation step.
#[derive(Debug, Clone)]
pub struct MarketState {
    pub mid_price: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub spread: f64,
    pub order_arrivals: u64,
    pub hawkes_intensity: f64,
    pub inventory_shock: i64,
}

/// Simulation data returned after running multiple steps.
#[derive(Debug, Clone)]
pub struct SimulationData {
    pub mid_prices: Vec<f64>,
    pub best_bids: Vec<f64>,
    pub best_asks: Vec<f64>,
    pub spreads: Vec<f64>,
    pub order_arrivals: Vec<u64>,
    pub hawkes_intensities: Vec<f64>,
    pub inventory_shocks: Vec<i64>,
}

/// Simulates a limit-order-book market environment.
///
/// The mid-price evolves via a Merton jump-diffusion:
///     dS = mu*dt + sigma*dW + J*dN
///
/// Order arrivals follow a Hawkes process with exponential kernel:
///     lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
#[derive(Debug, Clone)]
pub struct MarketSimulator {
    /// Initial mid-price.
    pub S0: f64,
    /// Drift.
    pub mu: f64,
    /// Volatility.
    pub sigma: f64,
    /// Jump intensity.
    pub jump_intensity: f64,
    /// Jump mean.
    pub jump_mean: f64,
    /// Jump std.
    pub jump_std: f64,
    /// Half spread.
    pub half_spread: f64,
    /// Hawkes base intensity.
    pub hawkes_mu: f64,
    /// Hawkes excitation.
    pub hawkes_alpha: f64,
    /// Hawkes decay.
    pub hawkes_beta: f64,
    /// Time step.
    pub dt: f64,

    /// Current mid-price.
    pub mid_price: f64,
    /// Current Hawkes intensity.
    pub hawkes_intensity: f64,
    /// Order history.
    pub order_history: Vec<OrderRecord>,
    /// RNG.
    pub rng: StdRng,
}

/// Record of orders in a time step.
#[derive(Debug, Clone)]
pub struct OrderRecord {
    pub t: usize,
    pub n_orders: u64,
    pub buy: u64,
    pub sell: u64,
}

impl MarketSimulator {
    /// Create a new market simulator with given seed.
    pub fn new(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self {
            S0: 100.0,
            mu: 0.0,
            sigma: 0.02,
            jump_intensity: 0.1,
            jump_mean: 0.0,
            jump_std: 0.01,
            half_spread: 0.01,
            hawkes_mu: 5.0,
            hawkes_alpha: 2.0,
            hawkes_beta: 10.0,
            dt: 1.0,
            mid_price: 100.0,
            hawkes_intensity: 5.0,
            order_history: Vec::new(),
            rng,
        }
    }

    /// Create with full parameter set.
    pub fn with_params(
        seed: u64,
        S0: f64,
        mu: f64,
        sigma: f64,
        jump_intensity: f64,
        jump_mean: f64,
        jump_std: f64,
        half_spread: f64,
        hawkes_mu: f64,
        hawkes_alpha: f64,
        hawkes_beta: f64,
        dt: f64,
    ) -> Self {
        let mut sim = Self::new(seed);
        sim.S0 = S0;
        sim.mu = mu;
        sim.sigma = sigma;
        sim.jump_intensity = jump_intensity;
        sim.jump_mean = jump_mean;
        sim.jump_std = jump_std;
        sim.half_spread = half_spread;
        sim.hawkes_mu = hawkes_mu;
        sim.hawkes_alpha = hawkes_alpha;
        sim.hawkes_beta = hawkes_beta;
        sim.dt = dt;
        sim
    }

    /// Reset the simulator to initial state. Returns starting mid-price.
    pub fn reset(&mut self, s0: Option<f64>) -> f64 {
        if let Some(s) = s0 {
            self.S0 = s;
        }
        self.mid_price = self.S0;
        self.order_history.clear();
        self.hawkes_intensity = self.hawkes_mu;
        self.mid_price
    }

    /// Advance one time step. Returns market state.
    pub fn step(&mut self) -> MarketState {
        // --- Mid-price dynamics (jump-diffusion) ---
        let dW: f64 = self.rng.sample::<f64, _>(StandardNormal) * self.dt.sqrt();
        let mut jump = 0.0;
        let n_jumps_dist = Poisson::new(self.jump_intensity * self.dt).unwrap();
        let n_jumps: f64 = self.rng.sample(n_jumps_dist);
        if n_jumps > 0.0 {
            let jump_dist = Normal::new(self.jump_mean, self.jump_std).unwrap();
            for _ in 0..n_jumps as usize {
                jump += self.rng.sample(jump_dist);
            }
        }

        let dS = self.mu * self.dt + self.sigma * dW + jump;
        self.mid_price += dS;
        self.mid_price = self.mid_price.max(1e-6); // floor

        // --- Hawkes order arrivals ---
        let lambda = (self.hawkes_intensity * self.dt).max(0.0);
        let order_dist = Poisson::new(lambda).unwrap();
        let n_orders: f64 = self.rng.sample(order_dist);

        // Update intensity: self-exciting decay + re-seed
        self.hawkes_intensity = (self.hawkes_mu + self.hawkes_alpha * n_orders)
            * (-self.hawkes_beta * self.dt).exp();
        self.hawkes_intensity = self.hawkes_intensity.max(self.hawkes_mu * 0.5);

        // Random walk on half-spread for realism
        let spread_noise: f64 = self.rng.sample::<f64, _>(StandardNormal) * 0.001;
        let effective_half_spread = (self.half_spread + spread_noise).max(1e-5);

        let best_bid = self.mid_price - effective_half_spread;
        let best_ask = self.mid_price + effective_half_spread;

        // Inventory shock: net order flow imbalance
        let buy_orders = if n_orders > 0.0 {
            self.rng.gen_range(0..=n_orders as u64)
        } else {
            0
        };
        let sell_orders = n_orders as u64 - buy_orders;
        let inventory_shock = buy_orders as i64 - sell_orders as i64;

        self.order_history.push(OrderRecord {
            t: self.order_history.len(),
            n_orders: n_orders as u64,
            buy: buy_orders,
            sell: sell_orders,
        });

        MarketState {
            mid_price: self.mid_price,
            best_bid,
            best_ask,
            spread: best_ask - best_bid,
            order_arrivals: n_orders as u64,
            hawkes_intensity: self.hawkes_intensity,
            inventory_shock,
        }
    }

    /// Run simulation for n_steps, return arrays of market data.
    pub fn simulate(&mut self, n_steps: usize) -> SimulationData {
        self.reset(None);
        let mut mid_prices = Vec::with_capacity(n_steps);
        let mut best_bids = Vec::with_capacity(n_steps);
        let mut best_asks = Vec::with_capacity(n_steps);
        let mut spreads = Vec::with_capacity(n_steps);
        let mut order_arrivals = Vec::with_capacity(n_steps);
        let mut hawkes_intensities = Vec::with_capacity(n_steps);
        let mut inventory_shocks = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let state = self.step();
            mid_prices.push(state.mid_price);
            best_bids.push(state.best_bid);
            best_asks.push(state.best_ask);
            spreads.push(state.spread);
            order_arrivals.push(state.order_arrivals);
            hawkes_intensities.push(state.hawkes_intensity);
            inventory_shocks.push(state.inventory_shock);
        }

        SimulationData {
            mid_prices,
            best_bids,
            best_asks,
            spreads,
            order_arrivals,
            hawkes_intensities,
            inventory_shocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let sim = MarketSimulator::new(42);
        assert!((sim.S0 - 100.0).abs() < 1e-10);
        assert!((sim.sigma - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_simulator_reset() {
        let mut sim = MarketSimulator::new(42);
        let price = sim.reset(Some(150.0));
        assert!((price - 150.0).abs() < 1e-10);
        assert!((sim.mid_price - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulator_step() {
        let mut sim = MarketSimulator::new(42);
        sim.reset(None);
        let state = sim.step();
        assert!(state.mid_price > 0.0);
        assert!(state.best_bid < state.best_ask);
    }

    #[test]
    fn test_simulator_simulate() {
        let mut sim = MarketSimulator::new(42);
        let data = sim.simulate(100);
        assert_eq!(data.mid_prices.len(), 100);
        assert_eq!(data.best_bids.len(), 100);
        assert!(data.mid_prices.iter().all(|&p| p > 0.0));
        assert!(data.spreads.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_simulator_deterministic() {
        let mut sim1 = MarketSimulator::new(123);
        let mut sim2 = MarketSimulator::new(123);
        let d1 = sim1.simulate(50);
        let d2 = sim2.simulate(50);
        for i in 0..50 {
            assert!(
                (d1.mid_prices[i] - d2.mid_prices[i]).abs() < 1e-10,
                "Mismatch at step {}: {} vs {}",
                i,
                d1.mid_prices[i],
                d2.mid_prices[i]
            );
        }
    }

    #[test]
    fn test_parallel_simulations_match_sequential() {
        let seeds: Vec<u64> = vec![10, 20, 30, 40, 50];
        let n_steps = 100;

        // Sequential
        let seq_results: Vec<SimulationData> = seeds
            .iter()
            .map(|&seed| {
                let mut sim = MarketSimulator::new(seed);
                sim.simulate(n_steps)
            })
            .collect();

        // Parallel
        let par_results: Vec<SimulationData> = seeds
            .par_iter()
            .map(|&seed| {
                let mut sim = MarketSimulator::new(seed);
                sim.simulate(n_steps)
            })
            .collect();

        assert_eq!(seq_results.len(), par_results.len());
        for (i, (s, p)) in seq_results.iter().zip(par_results.iter()).enumerate() {
            assert_eq!(s.mid_prices.len(), p.mid_prices.len(),
                "Length mismatch at seed index {}", i);
            for j in 0..s.mid_prices.len() {
                assert!(
                    (s.mid_prices[j] - p.mid_prices[j]).abs() < 1e-10,
                    "Price mismatch at seed index {}, step {}: {} vs {}",
                    i, j, s.mid_prices[j], p.mid_prices[j]
                );
            }
        }
    }
}
