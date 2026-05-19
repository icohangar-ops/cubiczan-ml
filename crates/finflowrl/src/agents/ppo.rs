/// PPO Agent — Proximal Policy Optimization with a numpy MLP.
///
/// Lightweight PPO implementation for fine-tuning the MeanFlow policy.
/// Supports save/load and on-policy rollouts.

use ndarray::{Array1, Array2, Zip};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::fs;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PpoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Dimension mismatch: expected obs_dim={0}, got {1}")]
    DimensionMismatch(usize, usize),
}

/// Softmax function for action probabilities.
///
/// Uses a manual slice loop for the exp() computation so LLVM can
/// auto-vectorize through contiguous memory (ndarray's mapv closure
/// obscures the inner loop from the vectorizer).
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = x.clone();
    // In-place subtraction + exp via contiguous slice
    if let Some(slice) = exps.as_slice_mut() {
        for v in slice.iter_mut() {
            *v = (*v - max_val).exp();
        }
    } else {
        exps = exps.mapv(|v| (v - max_val).exp());
    }
    let sum_exp: f64 = exps.sum();
    // In-place division
    if let Some(slice) = exps.as_slice_mut() {
        let inv_sum = 1.0 / sum_exp;
        for v in slice.iter_mut() {
            *v *= inv_sum;
        }
    } else {
        exps = exps / sum_exp;
    }
    exps
}

/// Simple multi-layer perceptron policy network.
#[derive(Debug, Clone)]
pub struct MLPPolicy {
    pub obs_dim: usize,
    pub act_dim: usize,
    pub hidden_sizes: Vec<usize>,
    /// Layer weights.
    pub weights: Vec<Array2<f64>>,
    /// Layer biases.
    pub biases: Vec<Array1<f64>>,
}

impl MLPPolicy {
    /// Create a new MLP policy with He initialization.
    pub fn new(obs_dim: usize, act_dim: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut prev_dim = obs_dim;

        for &h in &hidden_sizes {
            let scale = (2.0 / prev_dim as f64).sqrt();
            let w = Array2::from_shape_fn((prev_dim, h), |_| {
                rng.sample::<f64, _>(StandardNormal) * scale
            });
            let b = Array1::zeros(h);
            weights.push(w);
            biases.push(b);
            prev_dim = h;
        }

        // Output layer
        let scale = (2.0 / prev_dim as f64).sqrt();
        let w_out = Array2::from_shape_fn((prev_dim, act_dim), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        let b_out = Array1::zeros(act_dim);
        weights.push(w_out);
        biases.push(b_out);

        Self {
            obs_dim,
            act_dim,
            hidden_sizes,
            weights,
            biases,
        }
    }

    /// Forward pass. Returns action logits/values.
    pub fn forward(&self, obs: &Array1<f64>) -> Array1<f64> {
        let mut x = obs.clone();
        let n_layers = self.weights.len();
        for i in 0..n_layers {
            x = x.dot(&self.weights[i]) + &self.biases[i];
            if i < n_layers - 1 {
                // SIMD-friendly: manual slice loop for tanh.
                if let Some(slice) = x.as_slice_mut() {
                    for v in slice.iter_mut() {
                        *v = v.tanh();
                    }
                } else {
                    x = x.mapv(|v| v.tanh());
                }
            }
        }
        x
    }

    /// Sample action from policy.
    ///
    /// Returns: (action_index, log_prob)
    pub fn get_action(&self, obs: &Array1<f64>, deterministic: bool, rng: &mut StdRng) -> (usize, f64) {
        let logits = self.forward(obs);
        let probs = softmax(&logits);

        let action = if deterministic {
            // argmax
            let mut best_idx = 0;
            let mut best_val = probs[0];
            for i in 1..probs.len() {
                if probs[i] > best_val {
                    best_val = probs[i];
                    best_idx = i;
                }
            }
            best_idx
        } else {
            // Sample from categorical distribution
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            for i in 0..probs.len() {
                cumulative += probs[i];
                if r <= cumulative {
                    return (i, (probs[i] + 1e-10).ln());
                }
            }
            probs.len() - 1
        };

        (action, (probs[action] + 1e-10).ln())
    }

    /// Get scalar value estimate (uses first output neuron).
    pub fn get_value(&self, obs: &Array1<f64>) -> f64 {
        let logits = self.forward(obs);
        logits[0]
    }
}

/// PPO agent wrapping an MLP policy with clip-based updates.
#[derive(Debug, Clone)]
pub struct PPOAgent {
    pub policy: MLPPolicy,
    pub lr: f64,
    pub clip_ratio: f64,
    pub gamma: f64,
    pub lam: f64,
    pub epochs: usize,
    pub minibatch_size: usize,
}

impl PPOAgent {
    /// Create a new PPO agent.
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_sizes: Vec<usize>,
        lr: f64,
        clip_ratio: f64,
        gamma: f64,
        lam: f64,
    ) -> Self {
        let policy = MLPPolicy::new(obs_dim, act_dim, hidden_sizes);
        Self {
            policy,
            lr,
            clip_ratio,
            gamma,
            lam,
            epochs: 4,
            minibatch_size: 64,
        }
    }

    /// Save policy weights to JSON file.
    pub fn save(&self, path: &str) -> Result<(), PpoError> {
        let weights: Vec<Vec<Vec<f64>>> = self
            .policy
            .weights
            .iter()
            .map(|w| w.rows().into_iter().map(|r| r.to_vec()).collect())
            .collect();
        let biases: Vec<Vec<f64>> = self.policy.biases.iter().map(|b| b.to_vec()).collect();

        let data = serde_json::json!({
            "obs_dim": self.policy.obs_dim,
            "act_dim": self.policy.act_dim,
            "hidden_sizes": self.policy.hidden_sizes,
            "weights": weights,
            "biases": biases,
        });

        let content = serde_json::to_string_pretty(&data)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Load policy weights from JSON file.
    pub fn load(&mut self, path: &str) -> Result<(), PpoError> {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let obs_dim = data["obs_dim"].as_u64().unwrap_or(0) as usize;
        let act_dim = data["act_dim"].as_u64().unwrap_or(0) as usize;

        if obs_dim != self.policy.obs_dim {
            return Err(PpoError::DimensionMismatch(self.policy.obs_dim, obs_dim));
        }
        if act_dim != self.policy.act_dim {
            return Err(PpoError::DimensionMismatch(self.policy.act_dim, act_dim));
        }

        if let Some(weights_arr) = data["weights"].as_array() {
            for (i, w_val) in weights_arr.iter().enumerate() {
                if i < self.policy.weights.len() {
                    if let Some(rows) = w_val.as_array() {
                        let matrix: Vec<Vec<f64>> =
                            rows.iter().map(|r| json_to_f64_vec(r)).collect();
                        if !matrix.is_empty() {
                            let (nrows, ncols) = (matrix.len(), matrix[0].len());
                            if let Ok(arr2) = Array2::from_shape_vec(
                                (nrows, ncols),
                                matrix.into_iter().flatten().collect(),
                            ) {
                                self.policy.weights[i] = arr2;
                            }
                        }
                    }
                }
            }
        }

        if let Some(biases_arr) = data["biases"].as_array() {
            for (i, b_val) in biases_arr.iter().enumerate() {
                if i < self.policy.biases.len() {
                    let vec = json_to_f64_vec(b_val);
                    self.policy.biases[i] = Array1::from_vec(vec);
                }
            }
        }

        Ok(())
    }

    /// Select action given observation.
    ///
    /// Returns: (action, log_prob)
    pub fn select_action(
        &self,
        obs: &Array1<f64>,
        deterministic: bool,
        rng: &mut StdRng,
    ) -> (usize, f64) {
        self.policy.get_action(obs, deterministic, rng)
    }

    /// Compute Generalized Advantage Estimation.
    pub fn compute_gae(
        rewards: &[f64],
        values: &[f64],
        dones: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = rewards.len();
        let mut advantages = vec![0.0; n];
        let mut gae = 0.0;

        let gamma = 0.99;
        let lam = 0.95;

        for t in (0..n).rev() {
            let next_val = if t == n - 1 { 0.0 } else { values[t + 1] };
            let delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t];
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae;
            advantages[t] = gae;
        }

        let returns: Vec<f64> = advantages.iter().zip(values.iter()).map(|(a, v)| a + v).collect();
        (advantages, returns)
    }
}

/// Helper: JSON Value to Vec<f64>.
fn json_to_f64_vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .map(|x| x.as_f64().unwrap_or(0.0))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mlp_creation() {
        let policy = MLPPolicy::new(6, 3, vec![32, 32]);
        assert_eq!(policy.obs_dim, 6);
        assert_eq!(policy.act_dim, 3);
    }

    #[test]
    fn test_mlp_forward() {
        let policy = MLPPolicy::new(6, 3, vec![32, 32]);
        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let logits = policy.forward(&obs);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_mlp_get_action() {
        let policy = MLPPolicy::new(6, 3, vec![32, 32]);
        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut rng = StdRng::seed_from_u64(42);
        let (action, log_prob) = policy.get_action(&obs, false, &mut rng);
        assert!(action < 3);
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_ppo_creation() {
        let agent = PPOAgent::new(6, 3, vec![32, 32], 3e-4, 0.2, 0.99, 0.95);
        assert_eq!(agent.policy.obs_dim, 6);
    }

    #[test]
    fn test_ppo_select_action() {
        let agent = PPOAgent::new(6, 3, vec![32, 32], 3e-4, 0.2, 0.99, 0.95);
        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut rng = StdRng::seed_from_u64(42);
        let (action, _log_prob) = agent.select_action(&obs, false, &mut rng);
        assert!(action < 3);
    }

    #[test]
    fn test_ppo_save_load() {
        let agent = PPOAgent::new(6, 3, vec![32, 32], 3e-4, 0.2, 0.99, 0.95);
        let dir = std::env::temp_dir();
        let path = dir.join("finflowrl_test_ppo.json");
        let path_str = path.to_str().unwrap();

        agent.save(path_str).unwrap();

        let mut agent2 = PPOAgent::new(6, 3, vec![32, 32], 3e-4, 0.2, 0.99, 0.95);
        agent2.load(path_str).unwrap();

        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let (a1, _) = agent.select_action(&obs, true, &mut rng1);
        let (a2, _) = agent2.select_action(&obs, true, &mut rng2);
        assert_eq!(a1, a2);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_ppo_gae() {
        let rewards = vec![1.0, -0.5, 0.3, 0.8];
        let values = vec![0.5, 0.2, 0.1, 0.0];
        let dones = vec![0.0, 0.0, 0.0, 1.0];
        let (advs, rets) = PPOAgent::compute_gae(&rewards, &values, &dones);
        assert_eq!(advs.len(), 4);
        assert_eq!(rets.len(), 4);
    }
}
