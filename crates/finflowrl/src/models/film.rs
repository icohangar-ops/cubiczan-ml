/// FiLM — Feature-wise Linear Modulation layer.
///
/// FiLM applies an affine transformation conditioned on auxiliary features:
///     output = gamma * input + beta
///
/// where gamma and beta are projected from the conditioning vector.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Feature-wise Linear Modulation layer.
///
/// Projects conditioning input into scale (gamma) and shift (beta)
/// parameters, then applies element-wise affine transform to the input.
#[derive(Debug, Clone)]
pub struct FiLMLayer {
    /// Weight for projecting condition to gamma.
    pub W_gamma: Array2<f64>,
    /// Bias for gamma.
    pub b_gamma: Array1<f64>,
    /// Weight for projecting condition to beta.
    pub W_beta: Array2<f64>,
    /// Bias for beta.
    pub b_beta: Array1<f64>,
    /// Dimension of the input to modulate.
    pub input_dim: usize,
    /// Dimension of the conditioning vector.
    pub cond_dim: usize,
}

impl FiLMLayer {
    /// Create a new FiLM layer with He initialization.
    pub fn new(input_dim: usize, cond_dim: usize) -> Self {
        let scale = (2.0 / cond_dim as f64).sqrt();
        let mut rng = rand::thread_rng();

        let W_gamma = Array2::from_shape_fn((cond_dim, input_dim), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        let b_gamma = Array1::from_elem(input_dim, 1.0);
        let W_beta = Array2::from_shape_fn((cond_dim, input_dim), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        let b_beta = Array1::zeros(input_dim);

        Self {
            W_gamma,
            b_gamma,
            W_beta,
            b_beta,
            input_dim,
            cond_dim,
        }
    }

    /// Apply FiLM modulation: `gamma * x + beta`.
    ///
    /// - `x`: input tensor of shape `(input_dim,)`
    /// - `cond`: conditioning vector of shape `(cond_dim,)`
    pub fn forward(&self, x: &Array1<f64>, cond: &Array1<f64>) -> Array1<f64> {
        let gamma = cond.dot(&self.W_gamma) + &self.b_gamma;
        let beta = cond.dot(&self.W_beta) + &self.b_beta;
        &gamma * x + &beta
    }

    /// Get parameters as a tuple.
    pub fn get_params(&self) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        (
            self.W_gamma.clone(),
            self.b_gamma.clone(),
            self.W_beta.clone(),
            self.b_beta.clone(),
        )
    }

    /// Set parameters from a tuple.
    pub fn set_params(
        &mut self,
        W_gamma: Array2<f64>,
        b_gamma: Array1<f64>,
        W_beta: Array2<f64>,
        b_beta: Array1<f64>,
    ) {
        self.W_gamma = W_gamma;
        self.b_gamma = b_gamma;
        self.W_beta = W_beta;
        self.b_beta = b_beta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: we don't have approx in deps, use manual comparison
    fn array_close(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_film_creation() {
        let film = FiLMLayer::new(8, 4);
        assert_eq!(film.input_dim, 8);
        assert_eq!(film.cond_dim, 4);
    }

    #[test]
    fn test_film_forward() {
        let film = FiLMLayer::new(8, 4);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let cond = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.2]);
        let out = film.forward(&x, &cond);
        assert_eq!(out.len(), 8);
        for v in out.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_film_get_set_params() {
        let film = FiLMLayer::new(8, 4);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let cond = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.2]);
        let out1 = film.forward(&x, &cond);

        let (wg, bg, wb, bb) = film.get_params();
        let mut film2 = FiLMLayer::new(8, 4);
        film2.set_params(wg, bg, wb, bb);
        let out2 = film2.forward(&x, &cond);

        assert!(array_close(&out1, &out2, 1e-10));
    }
}
