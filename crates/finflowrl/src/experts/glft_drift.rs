/// GLFT-Drift expert — extends GLFT with a learned drift correction term.
///
/// Adds a drift-aware component that adapts the quoting strategy when the
/// mid-price exhibits directional momentum.

use ndarray::Array1;
use std::collections::HashMap;

/// GLFT with drift correction for trending markets.
#[derive(Debug, Clone)]
pub struct GLFTDriftExpert {
    /// Number of features.
    pub n_features: usize,
    /// Risk aversion.
    pub risk_aversion: f64,
    /// Maximum position.
    pub max_position: f64,
    /// Window for drift computation.
    pub drift_window: usize,
    /// Threshold for drift risk management.
    pub drift_threshold: f64,
    /// Learnable weight vector.
    pub weights: Array1<f64>,
    /// Price history for drift computation.
    pub price_history: Vec<f64>,
}

impl GLFTDriftExpert {
    /// Create a new GLFT-Drift expert.
    pub fn new(
        n_features: usize,
        risk_aversion: f64,
        max_position: f64,
        drift_window: usize,
        drift_threshold: f64,
    ) -> Self {
        let mut weights = Array1::zeros(n_features);
        // GLFT base weights
        if n_features >= 6 {
            weights[0] = -0.5;
            weights[1] = 0.3;
            weights[2] = -0.2;
            weights[3] = 0.1;
            weights[4] = -0.15; // drift correction
            weights[5] = -0.1; // drift squared (nonlinear)
        }

        Self {
            n_features,
            risk_aversion,
            max_position,
            drift_window,
            drift_threshold,
            weights,
            price_history: Vec::new(),
        }
    }

    /// Compute rolling drift from price history.
    pub fn compute_drift(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }
        let window = if self.price_history.len() >= self.drift_window {
            &self.price_history[self.price_history.len() - self.drift_window..]
        } else {
            &self.price_history[..]
        };
        if window.len() < 2 {
            return 0.0;
        }
        let sum: f64 = window
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .sum();
        sum / (window.len() - 1) as f64
    }

    /// Extract extended feature vector including drift terms.
    pub fn extract_features(&self, state: &HashMap<String, f64>) -> Array1<f64> {
        let mut features = Array1::zeros(self.n_features);
        features[0] = state
            .get("inventory")
            .copied()
            .unwrap_or(0.0)
            / self.max_position.max(1.0);
        features[1] = state.get("mid_price_change").copied().unwrap_or(0.0);
        features[2] = state.get("spread").copied().unwrap_or(0.01);
        features[3] = state.get("volatility").copied().unwrap_or(0.02);
        if self.n_features > 4 {
            features[4] = state.get("order_imbalance").copied().unwrap_or(0.0);
        }
        if self.n_features > 5 {
            features[5] = state.get("hawkes_intensity").copied().unwrap_or(5.0) / 20.0;
        }
        // Drift features
        let drift = self.compute_drift();
        if self.n_features > 6 {
            features[6] = drift;
        }
        if self.n_features > 7 {
            features[7] = drift.powi(2);
        }
        features
    }

    /// Compute drift-aware action.
    pub fn act(&mut self, state: &HashMap<String, f64>) -> GlftDriftAction {
        let mid_price = state.get("mid_price").copied().unwrap_or(100.0);
        self.price_history.push(mid_price);
        // Keep bounded history
        if self.price_history.len() > 2000 {
            self.price_history = self.price_history[2000 - self.drift_window..].to_vec();
        }

        let features = self.extract_features(state);
        let drift = self.compute_drift();

        let mut raw_action = self.weights.dot(&features);

        // Reduce position in strong drift (risk management)
        if drift.abs() > self.drift_threshold {
            let scale = (0.5_f64).max(1.0 - drift.abs() / (2.0 * self.drift_threshold));
            raw_action *= scale;
        }

        let target_position = (raw_action * self.max_position)
            .max(-self.max_position)
            .min(self.max_position);

        GlftDriftAction {
            target_position,
            raw_action,
            drift,
            weights: self.weights.clone(),
            features,
        }
    }
}

/// Action returned by the GLFT-Drift expert.
#[derive(Debug, Clone)]
pub struct GlftDriftAction {
    pub target_position: f64,
    pub raw_action: f64,
    pub drift: f64,
    pub weights: Array1<f64>,
    pub features: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glft_drift_expert() {
        let mut expert = GLFTDriftExpert::new(8, 0.1, 10.0, 20, 0.005);
        let mut state = HashMap::new();
        state.insert("inventory".to_string(), 2.0);
        state.insert("mid_price".to_string(), 100.0);
        state.insert("mid_price_change".to_string(), 0.001);
        state.insert("spread".to_string(), 0.02);
        state.insert("volatility".to_string(), 0.02);
        state.insert("order_imbalance".to_string(), 0.1);
        state.insert("hawkes_intensity".to_string(), 5.0);

        let result = expert.act(&state);
        assert!(result.target_position.abs() <= 10.0);
    }
}
