/// HFT Gym-style Environment for market-making agents.
///
/// Implements an OpenAI Gym-compatible interface wrapping the MarketSimulator.
/// Observation: [inventory, mid_price, spread, volatility, order_imbalance, hawkes_intensity]
/// Action: continuous target_position in [-max_position, +max_position]
/// Reward: PnL change - inventory_risk_penalty - transaction_cost

use crate::simulator::market::{MarketSimulator, MarketState};
use ndarray::Array1;

/// Info dict returned alongside (obs, reward, done) from each step.
#[derive(Debug, Clone)]
pub struct StepInfo {
    pub pnl: f64,
    pub step_pnl: f64,
    pub inventory: f64,
    pub mid_price: f64,
    pub trade_qty: f64,
}

/// High-Frequency Trading environment.
#[derive(Debug, Clone)]
pub struct HFTEnv {
    pub max_steps: usize,
    pub max_position: f64,
    pub transaction_cost: f64,
    pub inventory_penalty: f64,
    pub reward_scale: f64,
    pub obs_dim: usize,
    pub act_dim: usize,

    /// Internal state
    pub current_step: usize,
    pub inventory: f64,
    pub cash: f64,
    pub mid_price: f64,
    pub prev_mid_price: f64,
    pub total_pnl: f64,
    pub done: bool,

    /// Market simulator
    pub sim: MarketSimulator,
}

impl HFTEnv {
    /// Create a new HFT environment.
    pub fn new(seed: u64) -> Self {
        let sim = MarketSimulator::new(seed);
        Self {
            max_steps: 1000,
            max_position: 10.0,
            transaction_cost: 0.001,
            inventory_penalty: 0.01,
            reward_scale: 1.0,
            obs_dim: 6,
            act_dim: 1,
            current_step: 0,
            inventory: 0.0,
            cash: 0.0,
            mid_price: 100.0,
            prev_mid_price: 100.0,
            total_pnl: 0.0,
            done: false,
            sim,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        seed: u64,
        max_steps: usize,
        max_position: f64,
        transaction_cost: f64,
        inventory_penalty: f64,
        reward_scale: f64,
    ) -> Self {
        let mut env = Self::new(seed);
        env.max_steps = max_steps;
        env.max_position = max_position;
        env.transaction_cost = transaction_cost;
        env.inventory_penalty = inventory_penalty;
        env.reward_scale = reward_scale;
        env
    }

    /// Reset environment to initial state. Returns observation.
    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = 0;
        self.inventory = 0.0;
        self.cash = 0.0;
        self.total_pnl = 0.0;
        self.done = false;
        self.sim.reset(None);

        let state = self.sim.step();
        self.mid_price = state.mid_price;
        self.prev_mid_price = self.mid_price;

        self.get_obs(&state)
    }

    /// Execute one environment step.
    ///
    /// Returns: (observation, reward, done, info)
    pub fn step(&mut self, action: &Array1<f64>) -> (Array1<f64>, f64, bool, StepInfo) {
        if self.done {
            let obs = self.reset();
            return (obs, 0.0, true, StepInfo {
                pnl: self.total_pnl,
                step_pnl: 0.0,
                inventory: self.inventory,
                mid_price: self.mid_price,
                trade_qty: 0.0,
            });
        }

        let target_pos = if action.len() > 0 {
            action[0].max(-self.max_position).min(self.max_position)
        } else {
            0.0
        };

        // Get market state
        let state = self.sim.step();
        self.prev_mid_price = self.mid_price;
        self.mid_price = state.mid_price;

        // Execute trade to reach target position
        let trade_qty = target_pos - self.inventory;
        let trade_cost = trade_qty.abs() * self.transaction_cost;
        self.inventory = target_pos;
        self.cash -= trade_cost;

        // Mark-to-market PnL
        let unrealized_pnl = self.inventory * (self.mid_price - self.prev_mid_price);
        let realized_pnl = -trade_cost;
        let step_pnl = unrealized_pnl + realized_pnl;

        // Inventory risk penalty (quadratic)
        let inv_penalty = self.inventory_penalty * (self.inventory / self.max_position).powi(2);

        let reward = (step_pnl - inv_penalty) * self.reward_scale;
        self.total_pnl += step_pnl;

        self.current_step += 1;
        self.done = self.current_step >= self.max_steps;

        let obs = self.get_obs(&state);
        let info = StepInfo {
            pnl: self.total_pnl,
            step_pnl,
            inventory: self.inventory,
            mid_price: self.mid_price,
            trade_qty,
        };

        (obs, reward, self.done, info)
    }

    /// Construct observation vector from market state.
    fn get_obs(&self, state: &MarketState) -> Array1<f64> {
        let inv_ratio = self.inventory / self.max_position.max(1.0);
        let mid_change = if self.mid_price > 0.0 {
            (self.mid_price - self.prev_mid_price) / self.mid_price
        } else {
            0.0
        };
        let spread = state.spread;
        let vol_proxy = spread * 10.0; // proxy volatility
        let inv_shock = state.inventory_shock as f64 / 10.0;
        let hawkes = state.hawkes_intensity / 20.0;

        Array1::from_vec(vec![
            inv_ratio,
            mid_change,
            spread,
            vol_proxy,
            inv_shock,
            hawkes,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_env_creation() {
        let env = HFTEnv::new(42);
        assert_eq!(env.obs_dim, 6);
        assert_eq!(env.act_dim, 1);
    }

    #[test]
    fn test_env_reset() {
        let mut env = HFTEnv::new(42);
        let obs = env.reset();
        assert_eq!(obs.len(), 6);
        for v in obs.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_env_step() {
        let mut env = HFTEnv::new(42);
        env.reset();
        let action = Array1::from_vec(vec![0.5]);
        let (obs2, reward, _done, info) = env.step(&action);
        assert_eq!(obs2.len(), 6);
        assert!(reward.is_finite());
        // info fields
        assert!(info.pnl.is_finite());
        assert!(info.inventory.is_finite());
    }

    #[test]
    fn test_env_full_episode() {
        let mut env = HFTEnv::with_params(42, 50, 10.0, 0.001, 0.01, 1.0);
        let mut rng = StdRng::seed_from_u64(99);
        env.reset();
        for _ in 0..50 {
            let noise: f64 = rng.sample::<f64, _>(rand_distr::StandardNormal) * 0.1;
            let action = Array1::from_vec(vec![noise]);
            let (_, _, done, _) = env.step(&action);
            if done {
                break;
            }
        }
        assert!(env.done);
    }

    #[test]
    fn test_env_position_clipping() {
        let mut env = HFTEnv::with_params(42, 1000, 5.0, 0.001, 0.01, 1.0);
        env.reset();
        let action = Array1::from_vec(vec![100.0]);
        let (_, _, _, info) = env.step(&action);
        assert!(info.inventory.abs() <= 5.0 + 1e-10);
    }
}
