/// Gaussian Noise Policy — simple exploration baseline.
///
/// Samples actions from a Gaussian distribution parameterised by a learned
/// mean, with fixed standard deviation.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Gaussian noise exploration policy.
#[derive(Debug, Clone)]
pub struct NoisePolicy {
    pub obs_dim: usize,
    pub act_dim: usize,
    pub noise_std: f64,
    /// First layer weights.
    pub W1: Array2<f64>,
    /// First layer biases.
    pub b1: Array1<f64>,
    /// Second layer weights.
    pub W2: Array2<f64>,
    /// Second layer biases.
    pub b2: Array1<f64>,
}

impl NoisePolicy {
    /// Create a new noise policy with He initialization.
    pub fn new(obs_dim: usize, act_dim: usize, hidden_size: usize, noise_std: f64) -> Self {
        let mut rng = rand::thread_rng();

        let scale1 = (2.0 / obs_dim as f64).sqrt();
        let W1 = Array2::from_shape_fn((obs_dim, hidden_size), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale1
        });
        let b1 = Array1::zeros(hidden_size);

        let scale2 = (2.0 / hidden_size as f64).sqrt();
        let W2 = Array2::from_shape_fn((hidden_size, act_dim), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale2
        });
        let b2 = Array1::zeros(act_dim);

        Self {
            obs_dim,
            act_dim,
            noise_std,
            W1,
            b1,
            W2,
            b2,
        }
    }

    /// Forward pass to get action mean.
    pub fn forward(&self, obs: &Array1<f64>) -> Array1<f64> {
        let x = obs.dot(&self.W1) + &self.b1;
        let x = x.mapv(|v| v.tanh());
        x.dot(&self.W2) + &self.b2
    }

    /// Sample action: mean + Gaussian noise.
    pub fn act(&self, obs: &Array1<f64>, rng: &mut StdRng) -> Array1<f64> {
        let mean = self.forward(obs);
        let noise: Array1<f64> = Array1::from_shape_fn(self.act_dim, |_| {
            rng.sample::<f64, _>(StandardNormal) * self.noise_std
        });
        mean + noise
    }

    /// Get deterministic action (mean without noise).
    pub fn get_mean(&self, obs: &Array1<f64>) -> Array1<f64> {
        self.forward(obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_noise_policy_creation() {
        let policy = NoisePolicy::new(6, 2, 64, 0.1);
        assert_eq!(policy.obs_dim, 6);
        assert_eq!(policy.act_dim, 2);
    }

    #[test]
    fn test_noise_policy_act() {
        let policy = NoisePolicy::new(6, 2, 64, 0.1);
        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut rng = StdRng::seed_from_u64(42);
        let action = policy.act(&obs, &mut rng);
        assert_eq!(action.len(), 2);
        for v in action.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_noise_policy_get_mean() {
        let policy = NoisePolicy::new(6, 2, 64, 0.1);
        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mean = policy.get_mean(&obs);
        assert_eq!(mean.len(), 2);
        for v in mean.iter() {
            assert!(v.is_finite());
        }
    }
}
