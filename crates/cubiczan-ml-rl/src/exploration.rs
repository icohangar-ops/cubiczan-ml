//! # Exploration Strategies
//!
//! Exploration-exploitation strategies for reinforcement learning agents
//! in financial trading environments.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Trait for exploration strategies.
pub trait ExplorationStrategy: Send + Sync {
    /// Select an action using the exploration strategy.
    /// Returns the selected action index.
    fn select_action(&mut self, q_values: &[f64]) -> usize;

    /// Update internal state after a step (e.g., decay epsilon).
    fn update(&mut self);

    /// Reset the strategy to initial state.
    fn reset(&mut self);
}

/// Epsilon-greedy exploration: random action with probability epsilon,
/// greedy (best known Q-value) otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpsilonGreedy {
    pub epsilon: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
    pub initial_epsilon: f64,
}

impl EpsilonGreedy {
    pub fn new(epsilon: f64, epsilon_min: f64, epsilon_decay: f64) -> Self {
        Self {
            epsilon,
            epsilon_min,
            epsilon_decay,
            initial_epsilon: epsilon,
        }
    }

    pub fn standard() -> Self {
        Self::new(1.0, 0.01, 0.995)
    }
}

impl ExplorationStrategy for EpsilonGreedy {
    fn select_action(&mut self, q_values: &[f64]) -> usize {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.epsilon {
            rng.random_range(0..q_values.len())
        } else {
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    fn update(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}

/// Boltzmann (softmax) exploration: actions selected with probability
/// proportional to exp(Q / temperature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoltzmannExploration {
    pub temperature: f64,
    pub temperature_min: f64,
    pub temperature_decay: f64,
    pub initial_temperature: f64,
}

impl BoltzmannExploration {
    pub fn new(temperature: f64, temperature_min: f64, temperature_decay: f64) -> Self {
        Self {
            temperature,
            temperature_min,
            temperature_decay,
            initial_temperature: temperature,
        }
    }
}

impl ExplorationStrategy for BoltzmannExploration {
    fn select_action(&mut self, q_values: &[f64]) -> usize {
        let mut rng = rand::rng();
        if self.temperature < 1e-8 {
            return q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        let exps: Vec<f64> = q_values.iter().map(|q| (q / self.temperature).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            return rng.random_range(0..q_values.len());
        }

        let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        let mut cum = 0.0;
        let r = rng.random::<f64>();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                return i;
            }
        }
        q_values.len() - 1
    }

    fn update(&mut self) {
        self.temperature = (self.temperature * self.temperature_decay).max(self.temperature_min);
    }

    fn reset(&mut self) {
        self.temperature = self.initial_temperature;
    }
}

/// Upper Confidence Bound 1 (UCB1) exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UCB1 {
    pub exploration_weight: f64,
    action_counts: Vec<u64>,
    total_steps: u64,
}

impl UCB1 {
    pub fn new(num_actions: usize, exploration_weight: f64) -> Self {
        Self {
            exploration_weight,
            action_counts: vec![0; num_actions],
            total_steps: 0,
        }
    }

    fn ucb_score(&self, action: usize, q_value: f64) -> f64 {
        let count = self.action_counts[action].max(1);
        let exploration = ((self.total_steps as f64).ln() / count as f64).sqrt();
        q_value + self.exploration_weight * exploration
    }
}

impl ExplorationStrategy for UCB1 {
    fn select_action(&mut self, q_values: &[f64]) -> usize {
        self.total_steps += 1;
        q_values
            .iter()
            .enumerate()
            .map(|(i, q)| (i, self.ucb_score(i, *q)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn update(&mut self) {
        // UCB1 updates are handled in select_action via total_steps
    }

    fn reset(&mut self) {
        self.action_counts.fill(0);
        self.total_steps = 0;
    }
}

/// Thompson Sampling exploration.
#[derive(Debug, Clone)]
pub struct ThompsonSampling {
    num_actions: usize,
    alpha: Vec<f64>, // success counts
    beta: Vec<f64>,  // failure counts
}

impl ThompsonSampling {
    pub fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            alpha: vec![1.0; num_actions],
            beta: vec![1.0; num_actions],
        }
    }

    /// Sample from Beta distribution and pick the action with highest sample.
    fn beta_sample(alpha: f64, beta: f64) -> f64 {
        // Simple approximation using the Gamma distribution relationship
        let mut rng = rand::rng();
        let u = rng.random::<f64>();
        // Using the logistic-normal approximation for Beta sampling
        let mu = (alpha / (alpha + beta)).ln();
        let v = (1.0 / (alpha + beta) + 1.0 / (alpha + beta + 1.0)).max(1e-10);
        let z = (u.ln() - (1.0 - u).ln()).sqrt(); // approx normal
        (mu + v.sqrt() * z).exp() / (1.0 + (mu + v.sqrt() * z).exp())
    }
}

impl ExplorationStrategy for ThompsonSampling {
    fn select_action(&mut self, _q_values: &[f64]) -> usize {
        (0..self.num_actions)
            .map(|i| (i, Self::beta_sample(self.alpha[i], self.beta[i])))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn update(&mut self) {
        // Thompson Sampling updates are per-action, handled externally
    }

    fn reset(&mut self) {
        self.alpha.fill(1.0);
        self.beta.fill(1.0);
    }
}

/// Entropy-regularized exploration that penalizes low-entropy policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyRegularized {
    pub entropy_coefficient: f64,
    pub epsilon: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
}

impl EntropyRegularized {
    pub fn new(entropy_coefficient: f64, epsilon: f64, epsilon_min: f64, epsilon_decay: f64) -> Self {
        Self { entropy_coefficient, epsilon, epsilon_min, epsilon_decay }
    }

    /// Calculate entropy of the Q-value distribution.
    pub fn entropy(&self, q_values: &[f64]) -> f64 {
        let exps: Vec<f64> = q_values.iter().map(|q| q.exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 { return 0.0; }
        let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        probs.iter().filter(|&&p| p > 1e-10).map(|&p| -p * p.ln()).sum()
    }
}

impl ExplorationStrategy for EntropyRegularized {
    fn select_action(&mut self, q_values: &[f64]) -> usize {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.epsilon {
            rng.random_range(0..q_values.len())
        } else {
            q_values.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0)
        }
    }

    fn update(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    fn reset(&mut self) {
        self.epsilon = 1.0;
    }
}

/// Schedule for decaying exploration over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationSchedule {
    /// Total training steps.
    pub total_steps: u64,
    /// Steps completed so far.
    pub current_step: u64,
    /// Type of schedule.
    pub schedule_type: ScheduleType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Linear { start: f64, end: f64 },
    Exponential { start: f64, end: f64, decay: f64 },
    Step { values: Vec<(f64, u64)> },
}

impl ExplorationSchedule {
    pub fn linear(total_steps: u64, start: f64, end: f64) -> Self {
        Self { total_steps, current_step: 0, schedule_type: ScheduleType::Linear { start, end } }
    }

    /// Get the exploration parameter value at the current step.
    pub fn current_value(&self) -> f64 {
        let progress = (self.current_step as f64) / (self.total_steps.max(1) as f64);
        match &self.schedule_type {
            ScheduleType::Linear { start, end } => start + (end - start) * progress.min(1.0),
            ScheduleType::Exponential { start, end, decay } => {
                let val = start * decay.powi(progress as i32);
                val.max(*end)
            }
            ScheduleType::Step { values } => {
                for (val, step) in values {
                    if self.current_step < *step { return *val; }
                }
                values.last().map(|(v, _)| *v).unwrap_or(0.01)
            }
        }
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_greedy_explores() {
        let mut eg = EpsilonGreedy::new(1.0, 0.01, 0.995);
        let q = vec![1.0, 2.0, 3.0];
        // With epsilon=1.0, should select randomly
        let mut counts = vec![0; 3];
        for _ in 0..1000 {
            counts[eg.select_action(&q)] += 1;
        }
        // Each action should be selected roughly 1/3 of the time
        for c in &counts {
            assert!(*c > 200);
        }
    }

    #[test]
    fn test_epsilon_greedy_decay() {
        let mut eg = EpsilonGreedy::new(1.0, 0.01, 0.5);
        for _ in 0..100 {
            eg.update();
        }
        assert!(eg.epsilon < 1.0);
        assert!(eg.epsilon >= eg.epsilon_min);
    }

    #[test]
    fn test_boltzmann_selects() {
        let mut boltz = BoltzmannExploration::new(1.0, 0.01, 0.99);
        let q = vec![0.0, 10.0, 0.0];
        let action = boltz.select_action(&q);
        assert!(action < 3);
    }

    #[test]
    fn test_ucb1_selects() {
        let mut ucb = UCB1::new(3, 1.0);
        let q = vec![1.0, 2.0, 3.0];
        let action = ucb.select_action(&q);
        assert!(action < 3);
    }

    #[test]
    fn test_entropy_regularized() {
        let mut er = EntropyRegularized::new(0.01, 1.0, 0.01, 0.995);
        let q = vec![1.0, 2.0, 3.0];
        let ent = er.entropy(&q);
        assert!(ent > 0.0);
    }

    #[test]
    fn test_linear_schedule() {
        let mut sched = ExplorationSchedule::linear(100, 1.0, 0.01);
        assert!((sched.current_value() - 1.0).abs() < 0.001);
        for _ in 0..50 {
            sched.step();
        }
        assert!(sched.current_value() < 1.0);
        assert!(sched.current_value() > 0.01);
    }
}
