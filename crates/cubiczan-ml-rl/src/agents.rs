//! # RL Agents
//!
//! Reinforcement learning agents for financial trading: Q-learning, DQN,
//! policy gradient, actor-critic, and ensemble agents.

use std::collections::HashMap;
use std::fmt::Debug;

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::exploration::ExplorationStrategy;

/// Core agent trait for all RL agents.
pub trait Agent: Debug + Send + Sync {
    /// Select an action given the current state representation.
    fn act(&mut self, state: &State) -> Action;

    /// Learn from experience (state, action, reward, next_state, done).
    fn learn(
        &mut self,
        state: &State,
        action: Action,
        reward: f64,
        next_state: &State,
        done: bool,
    );

    /// Save agent state to a serializable format.
    fn save(&self) -> Result<Vec<u8>>;

    /// Load agent state.
    fn load(&mut self, data: &[u8]) -> Result<()>;
}

/// Agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub name: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            discount_factor: 0.99,
            name: "agent".to_string(),
        }
    }
}

/// State representation (simplified feature vector).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub features: Vec<f64>,
    pub timestamp: i64,
}

impl State {
    pub fn new(features: Vec<f64>) -> Self {
        Self { features, timestamp: chrono::Utc::now().timestamp_millis() }
    }

    pub fn zeros(dim: usize) -> Self {
        Self { features: vec![0.0; dim], timestamp: 0 }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Action representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Hold = 0,
    Buy = 1,
    Sell = 2,
}

impl Action {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Action::Hold,
            1 => Action::Buy,
            2 => Action::Sell,
            _ => Action::Hold,
        }
    }

    pub fn index(&self) -> usize {
        *self as usize
    }

    pub fn all() -> [Action; 3] {
        [Action::Hold, Action::Buy, Action::Sell]
    }
}

/// Tabular Q-learning agent with epsilon-greedy exploration.
#[derive(Debug, Clone)]
pub struct QLearningAgent<E: ExplorationStrategy> {
    /// Q-table: state_key -> [Q(hold), Q(buy), Q(sell)].
    q_table: HashMap<String, Vec<f64>>,
    config: AgentConfig,
    exploration: E,
}

impl<E: ExplorationStrategy> QLearningAgent<E> {
    pub fn new(learning_rate: f64, discount_factor: f64, exploration: E) -> Self {
        Self {
            q_table: HashMap::new(),
            config: AgentConfig {
                learning_rate,
                discount_factor,
                name: "q_learning".to_string(),
            },
            exploration,
        }
    }

    fn state_key(state: &State) -> String {
        // Discretize features to create a hashable state key.
        state
            .features
            .iter()
            .map(|f| format!("{:.2}", f))
            .collect::<Vec<_>>()
            .join("|")
    }

    fn get_q_values(&self, state: &State) -> Vec<f64> {
        let key = Self::state_key(state);
        self.q_table
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![0.0; 3])
    }

    fn set_q_value(&mut self, state: &State, action: Action, value: f64) {
        let key = Self::state_key(state);
        let entry = self.q_table.entry(key).or_insert_with(|| vec![0.0; 3]);
        entry[action.index()] = value;
    }

    pub fn q_table_size(&self) -> usize {
        self.q_table.len()
    }
}

impl<E: ExplorationStrategy + std::fmt::Debug + 'static> Agent for QLearningAgent<E> {
    fn act(&mut self, state: &State) -> Action {
        let q_values = self.get_q_values(state);
        let idx = self.exploration.select_action(
            &q_values.iter().map(|q| *q as f64).collect::<Vec<_>>(),
        );
        Action::from_index(idx)
    }

    fn learn(
        &mut self,
        state: &State,
        action: Action,
        reward: f64,
        next_state: &State,
        done: bool,
    ) {
        let current_q = self
            .get_q_values(state)
            .get(action.index())
            .copied()
            .unwrap_or(0.0);

        let next_q = if done {
            0.0
        } else {
            self.get_q_values(next_state)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let target = reward + self.config.discount_factor * next_q;
        let new_q = current_q + self.config.learning_rate * (target - current_q);
        self.set_q_value(state, action, new_q);
        self.exploration.update();
    }

    fn save(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&(&self.q_table, &self.config))?)
    }

    fn load(&mut self, data: &[u8]) -> Result<()> {
        let (q_table, config): (HashMap<String, Vec<f64>>, AgentConfig) = serde_json::from_slice(data)?;
        self.q_table = q_table;
        self.config = config;
        Ok(())
    }
}

/// Deep Q-learning agent with replay buffer.
#[derive(Debug, Clone)]
pub struct DeepQLearningAgent {
    /// Simple weight-based approximation (placeholder for neural net).
    weights: Vec<f64>,
    config: AgentConfig,
    replay_buffer: ReplayBuffer,
    batch_size: usize,
    target_update_freq: usize,
    step_count: usize,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
}

impl DeepQLearningAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            weights: vec![0.0; 10],
            config,
            replay_buffer: ReplayBuffer::new(10000),
            batch_size: 32,
            target_update_freq: 100,
            step_count: 0,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
        }
    }

    fn forward(&self, state: &State) -> Vec<f64> {
        // Simple linear approximation: output = W^T * features + bias
        let mut output = vec![0.0; 3];
        for (i, w) in self.weights.iter().enumerate() {
            let feat = state.features.get(i).copied().unwrap_or(0.0);
            for j in 0..3 {
                output[j] += w * feat * (j as f64 + 1.0) * 0.01;
            }
        }
        output
    }

    fn update_weights(&mut self, state: &State, _action: Action, td_error: f64) {
        let lr = self.config.learning_rate;
        for (i, w) in self.weights.iter_mut().enumerate() {
            let feat = state.features.get(i).copied().unwrap_or(0.0);
            *w += lr * td_error * feat * 0.01;
        }
    }

    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }
}

impl Agent for DeepQLearningAgent {
    fn act(&mut self, state: &State) -> Action {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.epsilon {
            Action::from_index(rng.random_range(0..3))
        } else {
            let q = self.forward(state);
            Action::from_index(
                q.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0),
            )
        }
    }

    fn learn(&mut self, state: &State, action: Action, reward: f64, next_state: &State, done: bool) {
        let q_values = self.forward(state);
        let current_q = q_values[action.index()];

        let next_q = if done {
            0.0
        } else {
            let next_q = self.forward(next_state);
            next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        let td_error = reward + self.config.discount_factor * next_q - current_q;
        self.update_weights(state, action, td_error);

        self.replay_buffer.push(Transition {
            state: state.clone(),
            action,
            reward,
            next_state: next_state.clone(),
            done,
        });

        self.step_count += 1;
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    fn save(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&(self.weights.clone(), self.config.clone(), self.epsilon))?)
    }

    fn load(&mut self, data: &[u8]) -> Result<()> {
        let (weights, config, epsilon): (Vec<f64>, AgentConfig, f64) = serde_json::from_slice(data)?;
        self.weights = weights;
        self.config = config;
        self.epsilon = epsilon;
        Ok(())
    }
}

/// Policy gradient (REINFORCE) agent.
#[derive(Debug, Clone)]
pub struct PolicyGradientAgent {
    /// Policy weights: maps state features to action probabilities.
    policy_weights: Vec<Vec<f64>>,
    config: AgentConfig,
    episode_states: Vec<Vec<f64>>,
    episode_actions: Vec<usize>,
    episode_rewards: Vec<f64>,
    gamma: f64,
}

impl PolicyGradientAgent {
    pub fn new(num_features: usize, config: AgentConfig) -> Self {
        Self {
            policy_weights: vec![vec![0.0; num_features]; 3],
            config,
            episode_states: Vec::new(),
            episode_actions: Vec::new(),
            episode_rewards: Vec::new(),
            gamma: 0.99,
        }
    }

    fn softmax(&self, state: &State) -> Vec<f64> {
        let mut scores = vec![0.0; 3];
        for (action_idx, weights) in self.policy_weights.iter().enumerate() {
            let mut score = 0.0;
            for (w, f) in weights.iter().zip(state.features.iter()) {
                score += w * f;
            }
            scores[action_idx] = score;
        }
        // Numerically stable softmax
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Update policy weights after an episode completes.
    pub fn update_policy(&mut self) {
        if self.episode_rewards.is_empty() {
            return;
        }
        // Compute discounted returns
        let mut returns = Vec::with_capacity(self.episode_rewards.len());
        let mut g = 0.0;
        for &r in self.episode_rewards.iter().rev() {
            g = r + self.gamma * g;
            returns.push(g);
        }
        returns.reverse();

        // Normalize returns
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64)
            .sqrt()
            .max(1e-8);
        let normalized: Vec<f64> = returns.iter().map(|r| (r - mean) / std).collect();

        // Policy gradient update
        let lr = self.config.learning_rate;
        for ((state_feat, action_idx), ret) in
            self.episode_states.iter().zip(self.episode_actions.iter()).zip(normalized.iter())
        {
            let grad = ret * lr;
            for (w, f) in self.policy_weights[*action_idx].iter_mut().zip(state_feat.iter()) {
                *w += grad * f;
            }
        }

        self.episode_states.clear();
        self.episode_actions.clear();
        self.episode_rewards.clear();
    }
}

impl Agent for PolicyGradientAgent {
    fn act(&mut self, state: &State) -> Action {
        let probs = self.softmax(state);
        let mut rng = rand::rng();
        let r = rng.random::<f64>();
        let mut cum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                let action = Action::from_index(i);
                self.episode_states.push(state.features.clone());
                self.episode_actions.push(i);
                return action;
            }
        }
        Action::Hold
    }

    fn learn(&mut self, _state: &State, _action: Action, reward: f64, _next_state: &State, done: bool) {
        self.episode_rewards.push(reward);
        if done {
            self.update_policy();
        }
    }

    fn save(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&(self.policy_weights.clone(), self.config.clone()))?)
    }

    fn load(&mut self, data: &[u8]) -> Result<()> {
        let (weights, config): (Vec<Vec<f64>>, AgentConfig) = serde_json::from_slice(data)?;
        self.policy_weights = weights;
        self.config = config;
        Ok(())
    }
}

/// Actor-Critic agent.
#[derive(Debug, Clone)]
pub struct ActorCriticAgent {
    actor_weights: Vec<Vec<f64>>,
    critic_weights: Vec<f64>,
    config: AgentConfig,
}

impl ActorCriticAgent {
    pub fn new(num_features: usize, config: AgentConfig) -> Self {
        Self {
            actor_weights: vec![vec![0.0; num_features]; 3],
            critic_weights: vec![0.0; num_features],
            config,
        }
    }

    fn actor_probs(&self, state: &State) -> Vec<f64> {
        let mut scores = vec![0.0; 3];
        for (i, weights) in self.actor_weights.iter().enumerate() {
            let score: f64 = weights.iter().zip(state.features.iter()).map(|(w, f)| w * f).sum();
            scores[i] = score;
        }
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    fn critic_value(&self, state: &State) -> f64 {
        self.critic_weights
            .iter()
            .zip(state.features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }
}

impl Agent for ActorCriticAgent {
    fn act(&mut self, state: &State) -> Action {
        let probs = self.actor_probs(state);
        let mut rng = rand::rng();
        let r = rng.random::<f64>();
        let mut cum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                return Action::from_index(i);
            }
        }
        Action::Hold
    }

    fn learn(
        &mut self,
        state: &State,
        action: Action,
        reward: f64,
        next_state: &State,
        done: bool,
    ) {
        let lr = self.config.learning_rate;
        let td_target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.critic_value(next_state)
        };
        let td_error = td_target - self.critic_value(state);

        // Update critic
        for (w, f) in self.critic_weights.iter_mut().zip(state.features.iter()) {
            *w += lr * td_error * f;
        }

        // Update actor (policy gradient)
        let probs = self.actor_probs(state);
        for (i, weights) in self.actor_weights.iter_mut().enumerate() {
            let grad = if i == action.index() {
                td_error
            } else {
                -probs[i] * td_error
            };
            for (w, f) in weights.iter_mut().zip(state.features.iter()) {
                *w += lr * 0.1 * grad * f;
            }
        }
    }

    fn save(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&(self.actor_weights.clone(), self.critic_weights.clone(), self.config.clone()))?)
    }

    fn load(&mut self, data: &[u8]) -> Result<()> {
        let (aw, cw, config): (Vec<Vec<f64>>, Vec<f64>, AgentConfig) = serde_json::from_slice(data)?;
        self.actor_weights = aw;
        self.critic_weights = cw;
        self.config = config;
        Ok(())
    }
}

/// Ensemble agent that combines multiple agents via voting.
#[derive(Debug)]
pub struct EnsembleAgent {
    agents: Vec<Box<dyn Agent>>,
    voting_method: VotingMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingMethod {
    Majority,
    WeightedAverage { weights: Vec<f64> },
    BestQ,
}

impl EnsembleAgent {
    pub fn new(agents: Vec<Box<dyn Agent>>, voting_method: VotingMethod) -> Self {
        Self { agents, voting_method }
    }

    /// Add an agent to the ensemble.
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agents.push(agent);
    }

    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }
}

impl Agent for EnsembleAgent {
    fn act(&mut self, state: &State) -> Action {
        let votes: Vec<Action> = self.agents.iter_mut().map(|a| a.act(state)).collect();
        match &self.voting_method {
            VotingMethod::Majority => {
                let mut counts = HashMap::new();
                for v in &votes {
                    *counts.entry(*v).or_insert(0usize) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(a, _)| a)
                    .unwrap_or(Action::Hold)
            }
            VotingMethod::WeightedAverage { weights } => {
                let mut scores = [0.0; 3];
                for (i, action) in votes.iter().enumerate() {
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    scores[action.index()] += w;
                }
                let best = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Action::from_index(best)
            }
            VotingMethod::BestQ => {
                // First agent acts as primary
                votes.into_iter().next().unwrap_or(Action::Hold)
            }
        }
    }

    fn learn(&mut self, state: &State, action: Action, reward: f64, next_state: &State, done: bool) {
        for agent in self.agents.iter_mut() {
            agent.learn(state, action, reward, next_state, done);
        }
    }

    fn save(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&("ensemble", self.agents.len()))?)
    }

    fn load(&mut self, data: &[u8]) -> Result<()> {
        let _: (String, usize) = serde_json::from_slice(data)?;
        Ok(())
    }
}

/// Experience replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    buffer: Vec<Transition>,
    capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub state: State,
    pub action: Action,
    pub reward: f64,
    pub next_state: State,
    pub done: bool,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buffer: Vec::with_capacity(capacity), capacity }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::rng();
        (0..batch_size.min(self.buffer.len()))
            .map(|_| {
                let idx = rng.random_range(0..self.buffer.len());
                &self.buffer[idx]
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBufferConfig {
    pub capacity: usize,
    pub batch_size: usize,
    pub min_size_to_learn: usize,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self { capacity: 10000, batch_size: 32, min_size_to_learn: 100 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exploration::EpsilonGreedy;

    #[test]
    fn test_q_learning_basic() {
        let mut agent: QLearningAgent<EpsilonGreedy> = QLearningAgent::new(
            0.1, 0.99,
            EpsilonGreedy::new(0.3, 0.01, 0.99),
        );
        let state = State::new(vec![1.0, 2.0]);
        let _action = agent.act(&state);
        agent.learn(&state, Action::Buy, 1.0, &State::new(vec![1.5, 2.5]), false);
        assert_eq!(agent.q_table_size(), 1);
    }

    #[test]
    fn test_dqn_agent() {
        let mut agent = DeepQLearningAgent::new(AgentConfig::default());
        let state = State::new(vec![1.0, 0.5, -0.3]);
        let action = agent.act(&state);
        agent.learn(&state, action, 0.5, &State::new(vec![1.2, 0.6, -0.2]), false);
        assert!(agent.replay_buffer_size() == 1);
    }

    #[test]
    fn test_policy_gradient() {
        let mut agent = PolicyGradientAgent::new(3, AgentConfig::default());
        let state = State::new(vec![1.0, 2.0, 3.0]);
        let _action = agent.act(&state);
        agent.learn(&state, Action::Buy, 1.0, &State::new(vec![1.5, 2.5, 3.5]), false);
        agent.learn(&state, Action::Buy, 0.5, &State::new(vec![1.5, 2.5, 3.5]), true);
    }

    #[test]
    fn test_actor_critic() {
        let mut agent = ActorCriticAgent::new(3, AgentConfig::default());
        let state = State::new(vec![0.1, -0.2, 0.3]);
        let _action = agent.act(&state);
        agent.learn(&state, Action::Hold, 0.1, &State::new(vec![0.15, -0.18, 0.32]), false);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buf = ReplayBuffer::new(5);
        for i in 0..7 {
            buf.push(Transition {
                state: State::new(vec![i as f64]),
                action: Action::Buy,
                reward: i as f64,
                next_state: State::new(vec![(i + 1) as f64]),
                done: i == 6,
            });
        }
        assert_eq!(buf.len(), 5); // capacity limit
        let samples = buf.sample(3);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_action_conversions() {
        assert_eq!(Action::from_index(0), Action::Hold);
        assert_eq!(Action::from_index(1), Action::Buy);
        assert_eq!(Action::from_index(2), Action::Sell);
        assert_eq!(Action::Buy.index(), 1);
        assert_eq!(Action::all().len(), 3);
    }
}
