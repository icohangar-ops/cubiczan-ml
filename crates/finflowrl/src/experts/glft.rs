//! GLFT (Generalized Linear Feature-based Trading) expert policy.
//!
//! A linear-quadratic market-making strategy that extends Avellaneda-Stoikov
//! with additional feature inputs (order imbalance, volatility, spread).

use ndarray::Array1;
use std::collections::HashMap;

/// Generalized Linear Feature-based Trading expert.
#[derive(Debug, Clone)]
pub struct GLFTExpert {
    /// Number of features.
    pub n_features: usize,
    /// Risk aversion.
    pub risk_aversion: f64,
    /// Maximum position.
    pub max_position: f64,
    /// Learnable weight vector.
    pub weights: Array1<f64>,
}

impl GLFTExpert {
    /// Create a new GLFT expert with heuristic weight initialisation.
    pub fn new(n_features: usize, risk_aversion: f64, max_position: f64) -> Self {
        let mut weights = Array1::zeros(n_features);
        // Heuristic initialisation: penalise inventory, reward mean-reversion
        if n_features >= 4 {
            weights[0] = -0.5; // inventory
            weights[1] = 0.3; // mid-price change (mean-reversion)
            weights[2] = -0.2; // spread
            weights[3] = 0.1; // volatility
        }

        Self {
            n_features,
            risk_aversion,
            max_position,
            weights,
        }
    }

    /// Extract feature vector from market state.
    ///
    /// Expected state keys: inventory, mid_price, prev_mid_price,
    /// spread, volatility, order_imbalance, ...
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
        features
    }

    /// Compute action from market state.
    pub fn act(&self, state: &HashMap<String, f64>) -> GlftAction {
        let features = self.extract_features(state);
        let raw_action = self.weights.dot(&features);

        // Clip to position limits
        let target_position = (raw_action * self.max_position)
            .max(-self.max_position)
            .min(self.max_position);

        GlftAction {
            target_position,
            raw_action,
            weights: self.weights.clone(),
            features,
        }
    }
}

/// Action returned by the GLFT expert.
#[derive(Debug, Clone)]
pub struct GlftAction {
    pub target_position: f64,
    pub raw_action: f64,
    pub weights: Array1<f64>,
    pub features: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("inventory".to_string(), 3.0);
        m.insert("mid_price".to_string(), 100.0);
        m.insert("prev_mid_price".to_string(), 99.5);
        m.insert("spread".to_string(), 0.02);
        m.insert("volatility".to_string(), 0.02);
        m.insert("order_imbalance".to_string(), 0.1);
        m
    }

    #[test]
    fn test_glft_expert() {
        let expert = GLFTExpert::new(6, 0.1, 10.0);
        let state = make_state();
        let result = expert.act(&state);
        assert!(result.target_position.abs() <= 10.0);
    }

    #[test]
    fn test_glft_features() {
        let expert = GLFTExpert::new(6, 0.1, 10.0);
        let mut state = HashMap::new();
        state.insert("inventory".to_string(), 0.0);
        state.insert("mid_price_change".to_string(), 0.0);
        state.insert("spread".to_string(), 0.01);
        state.insert("volatility".to_string(), 0.02);
        state.insert("order_imbalance".to_string(), 0.0);
        state.insert("hawkes_intensity".to_string(), 5.0);
        let features = expert.extract_features(&state);
        assert_eq!(features.len(), 6);
    }
}
