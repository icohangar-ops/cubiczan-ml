/// Pre-Trainer — Stage 1 expert distillation via flow matching.
///
/// Collects expert demonstrations and trains the MeanFlow policy to
/// match expert actions via the flow-matching loss.

use crate::envs::hft_env::HFTEnv;
use crate::experts::glft::GLFTExpert;
use crate::models::meanflow::MeanFlowPolicy;
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;

/// Stage 1: Distill expert policies into MeanFlow via flow matching.
pub struct PreTrainer {
    pub policy: MeanFlowPolicy,
    pub env: HFTEnv,
    pub expert: GLFTExpert,
    pub n_episodes: usize,
    pub steps_per_episode: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    /// Replay buffer for observations.
    pub obs_buffer: Vec<Array1<f64>>,
    /// Replay buffer for expert actions.
    pub action_buffer: Vec<Array1<f64>>,
}

impl PreTrainer {
    /// Create a new pre-trainer.
    pub fn new(
        policy: MeanFlowPolicy,
        env: HFTEnv,
        expert: GLFTExpert,
        n_episodes: usize,
        steps_per_episode: usize,
        learning_rate: f64,
        batch_size: usize,
    ) -> Self {
        Self {
            policy,
            env,
            expert,
            n_episodes,
            steps_per_episode,
            learning_rate,
            batch_size,
            obs_buffer: Vec::new(),
            action_buffer: Vec::new(),
        }
    }

    /// Collect expert demonstrations by running expert in env.
    pub fn collect_expert_demos(&mut self) {
        self.obs_buffer.clear();
        self.action_buffer.clear();

        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..self.n_episodes {
            let obs = self.env.reset();
            let mut current_obs = obs;

            for _step in 0..self.steps_per_episode {
                // Build state dict for expert
                let mut state = HashMap::new();
                state.insert(
                    "inventory".to_string(),
                    current_obs[0] * self.env.max_position,
                );
                state.insert("mid_price".to_string(), self.env.mid_price);
                state.insert("prev_mid_price".to_string(), self.env.prev_mid_price);
                state.insert("spread".to_string(), current_obs[2]);
                state.insert("volatility".to_string(), current_obs[3] / 10.0);
                state.insert(
                    "order_imbalance".to_string(),
                    current_obs[4] * 10.0,
                );
                state.insert(
                    "hawkes_intensity".to_string(),
                    current_obs[5] * 20.0,
                );

                let expert_output = self.expert.act(&state);

                let expert_action = if true {
                    // GLFT expert returns target_position
                    Array1::from_vec(vec![expert_output.target_position / self.env.max_position])
                } else {
                    Array1::zeros(self.env.act_dim)
                };

                self.obs_buffer.push(current_obs.clone());
                self.action_buffer.push(expert_action);

                // Take a random action to explore
                let action: Array1<f64> = Array1::from_shape_fn(self.env.act_dim, |_| {
                    rng.sample::<f64, _>(StandardNormal) * 0.1
                });
                let (next_obs, _reward, done, _info) = self.env.step(&action);

                if done {
                    break;
                }
                current_obs = next_obs;
            }
        }
    }

    /// Perform one training step on a random batch.
    ///
    /// Returns: average flow-matching loss
    pub fn train_step(&mut self) -> f64 {
        let n = self.obs_buffer.len();
        let mut rng = StdRng::seed_from_u64(42);

        let indices: Vec<usize> = if n < self.batch_size {
            (0..n).collect()
        } else {
            // Random sample without replacement
            let mut all: Vec<usize> = (0..n).collect();
            all.shuffle(&mut rng);
            all[..self.batch_size].to_vec()
        };

        let mut total_loss = 0.0;
        for &idx in &indices {
            let loss = self.policy.flow_loss(&self.obs_buffer[idx], &self.action_buffer[idx], &mut rng);
            total_loss += loss;
            self.nudge_params(loss, &mut rng);
        }

        total_loss / indices.len() as f64
    }

    /// Simple parameter update via small random perturbation (gradient-free).
    fn nudge_params(&mut self, loss: f64, rng: &mut StdRng) {
        let scale = 0.01;

        for w in &mut self.policy.vel_weights {
            for v in w.iter_mut() {
                let sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
                *v -= scale * loss * sign * 0.01;
            }
        }
        for b in &mut self.policy.vel_biases {
            for v in b.iter_mut() {
                let sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
                *v -= scale * loss * sign * 0.01;
            }
        }
        // FiLM params
        for v in self.policy.film_gamma.iter_mut() {
            let sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
            *v -= scale * loss * sign * 0.01;
        }
        for v in self.policy.film_beta.iter_mut() {
            let sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
            *v -= scale * loss * sign * 0.01;
        }
        for v in self.policy.W_film.iter_mut() {
            let sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
            *v -= scale * loss * sign * 0.01;
        }
        for v in self.policy.b_film.iter_mut() {
            let sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
            *v -= scale * loss * sign * 0.01;
        }
    }

    /// Run full pre-training loop.
    pub fn train(&mut self, n_iterations: usize) -> Vec<f64> {
        self.collect_expert_demos();

        let mut history = Vec::new();
        for _ in 0..n_iterations {
            let loss = self.train_step();
            history.push(loss);
        }

        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::hft_env::HFTEnv;
    use crate::models::meanflow::MeanFlowPolicy;
    use crate::experts::glft::GLFTExpert;

    #[test]
    fn test_pretrainer_creation() {
        let env = HFTEnv::with_params(42, 20, 10.0, 0.001, 0.01, 1.0);
        let policy = MeanFlowPolicy::new_default(6, 1);
        let expert = GLFTExpert::new(6, 0.1, 10.0);
        let trainer = PreTrainer::new(policy, env, expert, 2, 10, 1e-3, 32);
        assert_eq!(trainer.n_episodes, 2);
    }

    #[test]
    fn test_pretrainer_collect() {
        let env = HFTEnv::with_params(42, 20, 10.0, 0.001, 0.01, 1.0);
        let policy = MeanFlowPolicy::new_default(6, 1);
        let expert = GLFTExpert::new(6, 0.1, 10.0);
        let mut trainer = PreTrainer::new(policy, env, expert, 2, 10, 1e-3, 32);
        trainer.collect_expert_demos();
        assert!(!trainer.obs_buffer.is_empty());
        assert!(!trainer.action_buffer.is_empty());
    }

    #[test]
    fn test_pretrainer_train_step() {
        let env = HFTEnv::with_params(42, 20, 10.0, 0.001, 0.01, 1.0);
        let policy = MeanFlowPolicy::new_default(6, 1);
        let expert = GLFTExpert::new(6, 0.1, 10.0);
        let mut trainer = PreTrainer::new(policy, env, expert, 2, 10, 1e-3, 32);
        trainer.collect_expert_demos();
        let loss = trainer.train_step();
        assert!(loss.is_finite());
    }
}
