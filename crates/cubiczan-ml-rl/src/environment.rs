//! # Trading Environments
//!
//! Reinforcement learning environments for financial trading. Each environment implements the
//! [`TradingEnv`] trait, exposing `step()`, `reset()`, and `state()` methods compatible with
//! standard RL agent interfaces.
//!
//! ## Available Environments
//!
//! - [`SimpleTradingEnv`] — Single-asset buy/sell/hold with position tracking
//! - [`PortfolioEnv`] — Multi-asset portfolio allocation
//! - [`FuturesEnv`] — Perpetual futures with funding rates and leverage
//! - [`OrderBookEnv`] — Limit order book simulation with order placement

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// A single trading action the agent can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Do nothing; hold current position.
    Hold,
    /// Buy (go long or add to long).
    Buy,
    /// Sell (go short or add to short).
    Sell,
    /// Close any open position.
    Close,
}

impl Action {
    /// Map to a discrete index for tabular methods.
    pub fn to_index(&self) -> usize {
        match self {
            Action::Hold => 0,
            Action::Buy => 1,
            Action::Sell => 2,
            Action::Close => 3,
        }
    }

    /// Convert a discrete index back to an action.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Action::Hold,
            1 => Action::Buy,
            2 => Action::Sell,
            _ => Action::Close,
        }
    }

    /// Returns all possible actions.
    pub fn all() -> &'static [Action] {
        &[Action::Hold, Action::Buy, Action::Sell, Action::Close]
    }

    /// Number of discrete actions.
    pub fn count() -> usize {
        4
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Hold => write!(f, "HOLD"),
            Action::Buy => write!(f, "BUY"),
            Action::Sell => write!(f, "SELL"),
            Action::Close => write!(f, "CLOSE"),
        }
    }
}

/// Description of the action space (discrete for trading).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpace {
    /// Number of discrete actions available.
    pub n: usize,
    /// Human-readable names for each action.
    pub actions: Vec<String>,
}

impl Default for ActionSpace {
    fn default() -> Self {
        Self {
            n: Action::count(),
            actions: Action::all().iter().map(|a| a.to_string()).collect(),
        }
    }
}

/// Observation vector provided to the agent at each time step.
///
/// Encodes the current market state, position, and portfolio information
/// as a fixed-length `Vec<f64>` suitable for neural network input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Feature vector for the agent.
    pub features: Vec<f64>,
    /// Current timestamp.
    pub timestamp: DateTime<Utc>,
    /// Current portfolio value.
    pub portfolio_value: f64,
    /// Current position (positive = long, negative = short, zero = flat).
    pub position: f64,
    /// Unrealized profit/loss.
    pub unrealized_pnl: f64,
}

impl Observation {
    /// Create a new observation from a feature vector.
    pub fn new(features: Vec<f64>, timestamp: DateTime<Utc>) -> Self {
        Self {
            features,
            timestamp,
            portfolio_value: 0.0,
            position: 0.0,
            unrealized_pnl: 0.0,
        }
    }

    /// Create a padded/empty observation of a given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            features: vec![0.0; dim],
            timestamp: Utc::now(),
            portfolio_value: 0.0,
            position: 0.0,
            unrealized_pnl: 0.0,
        }
    }

    /// Dimensionality of the feature vector.
    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

/// Full environment state, including market data and portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Current observation vector.
    pub observation: Observation,
    /// Whether the episode has ended.
    pub terminal: bool,
    /// Time step index within the episode.
    pub step: usize,
    /// Total reward accumulated so far.
    pub total_reward: f64,
    /// Additional metadata.
    pub info: HashMap<String, f64>,
}

impl State {
    pub fn new(observation: Observation) -> Self {
        Self {
            observation,
            terminal: false,
            step: 0,
            total_reward: 0.0,
            info: HashMap::new(),
        }
    }

    pub fn terminal(observation: Observation, total_reward: f64) -> Self {
        Self {
            observation,
            terminal: true,
            step: 0,
            total_reward,
            info: HashMap::new(),
        }
    }
}

/// Reward types for different optimization objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardType {
    /// Simple profit-and-loss.
    PnL,
    /// Log returns (symmetric).
    LogReturn,
    /// Risk-adjusted (Sharpe-like).
    RiskAdjusted,
    /// Percentage return on capital.
    PercentReturn,
    /// Differential Sharpe ratio.
    DifferentialSharpe,
    /// Custom reward (user-defined).
    Custom,
}

/// A single reward signal with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reward {
    /// Scalar reward value.
    pub value: f64,
    /// What kind of reward this is.
    pub reward_type: RewardType,
    /// Breakdown of reward components.
    pub components: HashMap<String, f64>,
}

impl Reward {
    pub fn new(value: f64, reward_type: RewardType) -> Self {
        Self { value, reward_type, components: HashMap::new() }
    }

    pub fn pnl(pnl: f64) -> Self {
        Self::new(pnl, RewardType::PnL)
    }

    pub fn risk_adjusted(return_val: f64, volatility: f64) -> Self {
        let value = if volatility > 1e-10 { return_val / volatility } else { 0.0 };
        let mut components = HashMap::new();
        components.insert("return".to_string(), return_val);
        components.insert("volatility".to_string(), volatility);
        Self { value, reward_type: RewardType::RiskAdjusted, components }
    }

    pub fn with_component(mut self, key: &str, value: f64) -> Self {
        self.components.insert(key.to_string(), value);
        self
    }
}

impl std::ops::Add for Reward {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut components = self.components;
        for (k, v) in rhs.components {
            *components.entry(k).or_insert(0.0) += v;
        }
        Self {
            value: self.value + rhs.value,
            reward_type: self.reward_type,
            components,
        }
    }
}

/// Reward function that maps (state, action, next_state) → Reward.
pub trait RewardFunction: Debug + Send + Sync {
    /// Compute the reward for transitioning between states.
    fn compute(
        &self,
        prev_portfolio_value: f64,
        curr_portfolio_value: f64,
        action: Action,
        position_size: f64,
        returns_history: &[f64],
        risk_free_rate: f64,
    ) -> Reward;
}

/// PnL reward function — raw dollar profit/loss.
#[derive(Debug, Clone, Default)]
pub struct PnLReward;

impl RewardFunction for PnLReward {
    fn compute(
        &self,
        prev_portfolio_value: f64,
        curr_portfolio_value: f64,
        _action: Action,
        _position_size: f64,
        _returns_history: &[f64],
        _risk_free_rate: f64,
    ) -> Reward {
        let pnl = curr_portfolio_value - prev_portfolio_value;
        Reward::pnl(pnl)
    }
}

/// Log return reward function — symmetric and scale-invariant.
#[derive(Debug, Clone, Default)]
pub struct LogReturnReward;

impl RewardFunction for LogReturnReward {
    fn compute(
        &self,
        prev_portfolio_value: f64,
        curr_portfolio_value: f64,
        _action: Action,
        _position_size: f64,
        _returns_history: &[f64],
        _risk_free_rate: f64,
    ) -> Reward {
        let log_ret = if prev_portfolio_value > 1e-10 {
            (curr_portfolio_value / prev_portfolio_value).ln()
        } else {
            0.0
        };
        Reward::new(log_ret, RewardType::LogReturn)
    }
}

/// Risk-adjusted reward — returns scaled by recent volatility (Sharpe-like).
#[derive(Debug, Clone)]
pub struct RiskAdjustedReward {
    /// Lookback window for volatility computation.
    pub window: usize,
    /// Risk-free rate for excess return computation.
    pub risk_free_rate: f64,
    /// Scaling factor applied to the raw Sharpe.
    pub scale: f64,
}

impl Default for RiskAdjustedReward {
    fn default() -> Self {
        Self { window: 20, risk_free_rate: 0.02 / 252.0, scale: 100.0 }
    }
}

impl RewardFunction for RiskAdjustedReward {
    fn compute(
        &self,
        prev_portfolio_value: f64,
        curr_portfolio_value: f64,
        _action: Action,
        _position_size: f64,
        returns_history: &[f64],
        risk_free_rate: f64,
    ) -> Reward {
        let ret = if prev_portfolio_value > 1e-10 {
            (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        } else {
            0.0
        };

        let vol = if returns_history.len() >= 2 {
            let recent = &returns_history[returns_history.len().saturating_sub(self.window)..];
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            variance.sqrt().max(1e-10)
        } else {
            1.0
        };

        let sharpe = (ret - risk_free_rate) / vol * self.scale;
        Reward::risk_adjusted(ret, vol).with_component("sharpe_raw", sharpe)
    }
}

/// Differential Sharpe reward — online update without storing full history.
///
/// Uses the formulation from Moody & Saffell (2001):
/// `dS_t = (A_t * ΔR_t - B_t * ΔR_t²) / (t^(3/2))`
/// where `A_t = η_A * dS_t + A_{t-1}` and `B_t = η_B * dS_t + B_{t-1}`.
#[derive(Debug, Clone)]
pub struct DifferentialSharpeReward {
    pub eta_a: f64,
    pub eta_b: f64,
    pub a: f64,
    pub b: f64,
    pub t: usize,
}

impl Default for DifferentialSharpeReward {
    fn default() -> Self {
        Self { eta_a: 0.005, eta_b: 0.005, a: 0.0, b: 0.0, t: 1 }
    }
}

impl DifferentialSharpeReward {
    pub fn new(eta_a: f64, eta_b: f64) -> Self {
        Self { eta_a, eta_b, a: 0.0, b: 0.0, t: 1 }
    }
}

impl RewardFunction for DifferentialSharpeReward {
    fn compute(
        &self,
        prev_portfolio_value: f64,
        curr_portfolio_value: f64,
        _action: Action,
        _position_size: f64,
        _returns_history: &[f64],
        _risk_free_rate: f64,
    ) -> Reward {
        let ret = if prev_portfolio_value > 1e-10 {
            (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        } else {
            0.0
        };
        let a = self.a + self.eta_a * ret;
        let b = self.b + self.eta_b * ret * ret;
        let t = self.t as f64;
        let denominator = (t.powi(3)).max(1.0);
        let ds = if b > 1e-10 {
            (a / b.sqrt()) / denominator
        } else {
            0.0
        };
        let mut r = Reward::new(ds, RewardType::DifferentialSharpe);
        r.components.insert("a".to_string(), a);
        r.components.insert("b".to_string(), b);
        r
    }
}

// ─── TradingEnv Trait ───────────────────────────────────────────────────────

/// Core trait for all trading reinforcement learning environments.
///
/// Provides the standard OpenAI Gym-style interface:
/// - `reset()` → initial `State`
/// - `step(action)` → `(State, Reward, bool)`
/// - `state()` → current `State`
pub trait TradingEnv: Send + Sync {
    /// Reset the environment to its initial state.
    fn reset(&mut self) -> State;

    /// Execute one time step. Returns `(next_state, reward, done)`.
    fn step(&mut self, action: Action) -> (State, Reward, bool);

    /// Get the current state without advancing.
    fn state(&self) -> &State;

    /// Whether the current episode has ended.
    fn is_terminal(&self) -> bool;

    /// Dimensionality of the observation space.
    fn observation_dim(&self) -> usize;

    /// Description of the action space.
    fn action_space(&self) -> &ActionSpace;

    /// Render the current state for debugging/visualization.
    fn render(&self) -> String {
        let state = self.state();
        format!(
            "Step: {} | Portfolio: {:.2} | Position: {:.4} | Total Reward: {:.4} | Terminal: {}",
            state.step,
            state.observation.portfolio_value,
            state.observation.position,
            state.total_reward,
            state.terminal
        )
    }

    /// Set the reward function used by this environment.
    fn set_reward_function(&mut self, reward_fn: Box<dyn RewardFunction>);

    /// Seed the environment's random number generator for reproducibility.
    fn seed(&mut self, _seed: u64) {
        // Default: no-op
    }
}

// ─── SimpleTradingEnv ───────────────────────────────────────────────────────

/// Configuration for [`SimpleTradingEnv`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTradingEnvConfig {
    /// Initial cash balance.
    pub initial_cash: f64,
    /// Transaction cost as a fraction of trade value.
    pub commission_rate: f64,
    /// Fixed slippage in price units.
    pub slippage: f64,
    /// Maximum position size (number of units).
    pub max_position: f64,
    /// Lookback window for feature computation.
    pub lookback: usize,
    /// Reward function type.
    pub reward_type: RewardType,
}

impl Default for SimpleTradingEnvConfig {
    fn default() -> Self {
        Self {
            initial_cash: 100_000.0,
            commission_rate: 0.001,
            slippage: 0.0,
            max_position: 1_000_000.0,
            lookback: 10,
            reward_type: RewardType::PnL,
        }
    }
}

/// Single-asset trading environment with buy/sell/hold actions.
///
/// The agent observes a window of normalized price returns and its current
/// position/portfolio value, then selects an action each step.
#[derive(Debug)]
pub struct SimpleTradingEnv {
    /// Historical price data (close prices).
    prices: Vec<f64>,
    /// Current time step index into the price data.
    current_step: usize,
    /// Environment configuration.
    config: SimpleTradingEnvConfig,
    /// Current state.
    state: State,
    /// Cash on hand.
    cash: f64,
    /// Position size (positive = long, negative = short).
    position: f64,
    /// Average entry price for PnL calculation.
    entry_price: f64,
    /// Portfolio value at the start of the episode.
    pub initial_portfolio_value: f64,
    /// Historical returns for reward computation.
    returns_history: Vec<f64>,
    /// Reward function.
    reward_fn: Box<dyn RewardFunction>,
    /// Risk-free rate.
    risk_free_rate: f64,
    /// Episode count (for differential Sharpe).
    episode: usize,
}

impl SimpleTradingEnv {
    /// Create a new simple trading environment from a vector of close prices.
    ///
    /// # Arguments
    ///
    /// * `prices` - Historical close prices for the asset.
    /// * `initial_cash` - Starting cash balance.
    pub fn new(prices: Vec<f64>, initial_cash: f64) -> Self {
        Self::with_config(prices, SimpleTradingEnvConfig {
            initial_cash,
            ..Default::default()
        })
    }

    /// Create with a custom configuration.
    pub fn with_config(prices: Vec<f64>, config: SimpleTradingEnvConfig) -> Self {
        let lookback = config.lookback;
        let obs_dim = lookback + 3; // returns window + position + cash_ratio + unrealized_pnl
        let _action_space = ActionSpace::default();

        let env = Self {
            prices,
            current_step: lookback,
            config,
            state: State::new(Observation::zeros(obs_dim)),
            cash: 0.0,
            position: 0.0,
            entry_price: 0.0,
            initial_portfolio_value: 0.0,
            returns_history: Vec::new(),
            reward_fn: Box::new(PnLReward),
            risk_free_rate: 0.02 / 252.0,
            episode: 0,
        };

        // We can't call reset() here because `self` is moved; caller must call reset().
        // So we build and return, letting the caller reset.
        // Instead, we create a placeholder and immediately reset in a helper.
        let mut env = env;
        env.reward_fn = match env.config.reward_type {
            RewardType::PnL => Box::new(PnLReward),
            RewardType::LogReturn => Box::new(LogReturnReward),
            RewardType::RiskAdjusted => Box::new(RiskAdjustedReward::default()),
            RewardType::DifferentialSharpe => Box::new(DifferentialSharpeReward::default()),
            _ => Box::new(PnLReward),
        };
        env
    }

    /// Current cash balance.
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Current position size.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Current portfolio value (cash + position at market price).
    pub fn portfolio_value(&self) -> f64 {
        self.cash + self.position * self.current_price()
    }

    /// Unrealized PnL.
    pub fn unrealized_pnl(&self) -> f64 {
        if self.position.abs() < 1e-10 {
            return 0.0;
        }
        self.position * (self.current_price() - self.entry_price)
    }

    /// Current price in the data.
    pub fn current_price(&self) -> f64 {
        self.prices.get(self.current_step).copied().unwrap_or(0.0)
    }

    /// Execute a buy order.
    fn execute_buy(&mut self) -> f64 {
        let price = self.current_price() * (1.0 + self.config.slippage);
        // Buy as much as possible with available cash
        let max_units = (self.cash / (price * (1.0 + self.config.commission_rate)))
            .floor()
            .min(self.config.max_position - self.position.max(0.0));

        if max_units <= 0.0 {
            return 0.0;
        }

        let cost = max_units * price;
        let commission = cost * self.config.commission_rate;
        self.cash -= cost + commission;
        self.position += max_units;

        // Update entry price to weighted average
        if self.position.abs() > 1e-10 {
            self.entry_price = (self.entry_price * (self.position - max_units) + price * max_units) / self.position;
        } else {
            self.entry_price = price;
        }

        tracing::debug!(step = self.current_step, units = max_units, price = price, cost = cost, "BUY executed");
        -(commission) // negative cost
    }

    /// Execute a sell order.
    fn execute_sell(&mut self) -> f64 {
        if self.position <= 0.0 {
            return 0.0; // nothing to sell
        }
        let price = self.current_price() * (1.0 - self.config.slippage);
        let units = self.position; // sell all
        let revenue = units * price;
        let commission = revenue * self.config.commission_rate;
        self.cash += revenue - commission;
        self.position = 0.0;
        tracing::debug!(step = self.current_step, units = units, price = price, revenue = revenue, "SELL executed");
        revenue - commission
    }

    /// Execute a short sell.
    fn execute_short(&mut self) -> f64 {
        let price = self.current_price() * (1.0 - self.config.slippage);
        let max_units = (self.cash / (price * (1.0 + self.config.commission_rate)))
            .floor()
            .min(self.config.max_position + self.position.min(0.0));

        if max_units <= 0.0 {
            return 0.0;
        }

        let revenue = max_units * price;
        let commission = revenue * self.config.commission_rate;
        self.cash += revenue - commission;
        self.position -= max_units;

        if self.position.abs() > 1e-10 {
            self.entry_price = (self.entry_price * (self.position + max_units) + price * max_units) / self.position.abs();
        } else {
            self.entry_price = price;
        }

        tracing::debug!(step = self.current_step, units = max_units, price = price, "SHORT executed");
        revenue - commission
    }

    /// Close any open position.
    fn execute_close(&mut self) -> f64 {
        let mut proceeds = 0.0;
        if self.position > 0.0 {
            let price = self.current_price() * (1.0 - self.config.slippage);
            let revenue = self.position * price;
            let commission = revenue * self.config.commission_rate;
            self.cash += revenue - commission;
            proceeds = revenue - commission;
            tracing::debug!(step = self.current_step, units = self.position, price = price, "CLOSE LONG");
        } else if self.position < 0.0 {
            let price = self.current_price() * (1.0 + self.config.slippage);
            let cost = self.position.abs() * price;
            let commission = cost * self.config.commission_rate;
            self.cash -= cost + commission;
            proceeds = -(cost + commission);
            tracing::debug!(step = self.current_step, units = self.position.abs(), price = price, "CLOSE SHORT");
        }
        self.position = 0.0;
        self.entry_price = 0.0;
        proceeds
    }

    /// Build the observation vector from the current state.
    fn build_observation(&self) -> Observation {
        let mut features = Vec::with_capacity(self.observation_dim());
        let lb = self.config.lookback;

        // Normalized returns over the lookback window
        for i in (0..lb).rev() {
            let idx = self.current_step.saturating_sub(i + 1);
            let prev_idx = idx.saturating_sub(1);
            if prev_idx < self.prices.len() && idx < self.prices.len() && self.prices[prev_idx] > 1e-10 {
                features.push((self.prices[idx] - self.prices[prev_idx]) / self.prices[prev_idx]);
            } else {
                features.push(0.0);
            }
        }

        // Position as fraction of portfolio
        let pv = self.portfolio_value();
        let pos_frac = if pv > 1e-10 {
            (self.position * self.current_price()) / pv
        } else {
            0.0
        };
        features.push(pos_frac);

        // Cash ratio
        let cash_ratio = if pv > 1e-10 { self.cash / pv } else { 1.0 };
        features.push(cash_ratio);

        // Unrealized PnL as fraction of portfolio
        let pnl_frac = if pv > 1e-10 { self.unrealized_pnl() / pv } else { 0.0 };
        features.push(pnl_frac);

        Observation {
            features,
            timestamp: Utc::now(),
            portfolio_value: pv,
            position: self.position,
            unrealized_pnl: self.unrealized_pnl(),
        }
    }

    /// Update the internal state from the current market data.
    fn update_state(&mut self) {
        let obs = self.build_observation();
        let pv = obs.portfolio_value;
        let prev_pv = self.initial_portfolio_value.max(1.0);

        // Track returns for reward computation
        let ret = (pv - prev_pv) / prev_pv;
        self.returns_history.push(ret);

        self.state = State {
            observation: obs,
            terminal: false,
            step: self.current_step,
            total_reward: self.state.total_reward,
            info: HashMap::new(),
        };
    }
}

impl TradingEnv for SimpleTradingEnv {
    fn reset(&mut self) -> State {
        self.current_step = self.config.lookback;
        self.cash = self.config.initial_cash;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.initial_portfolio_value = self.config.initial_cash;
        self.returns_history.clear();
        self.episode += 1;

        let obs = Observation {
            features: vec![0.0; self.observation_dim()],
            timestamp: Utc::now(),
            portfolio_value: self.config.initial_cash,
            position: 0.0,
            unrealized_pnl: 0.0,
        };

        self.state = State::new(obs);
        self.update_state();
        tracing::info!(episode = self.episode, "Environment reset");
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        let prev_portfolio_value = self.portfolio_value();

        // Execute the action
        match action {
            Action::Hold => {}
            Action::Buy => {
                if self.position >= 0.0 {
                    self.execute_buy();
                } else {
                    self.execute_close();
                    self.execute_buy();
                }
            }
            Action::Sell => {
                if self.position <= 0.0 {
                    self.execute_short();
                } else {
                    self.execute_close();
                    self.execute_short();
                }
            }
            Action::Close => {
                self.execute_close();
            }
        }

        // Advance time step
        self.current_step += 1;

        // Check if episode is done
        let done = self.current_step >= self.prices.len() - 1;

        if done {
            // Close any open position at the end
            self.execute_close();
        }

        // Update state
        self.update_state();

        let curr_portfolio_value = self.portfolio_value();

        // Compute reward
        let reward = self.reward_fn.compute(
            prev_portfolio_value,
            curr_portfolio_value,
            action,
            self.position,
            &self.returns_history,
            self.risk_free_rate,
        );

        self.state.total_reward += reward.value;
        if done {
            self.state.terminal = true;
        }

        tracing::trace!(
            step = self.current_step,
            action = %action,
            reward = reward.value,
            portfolio = curr_portfolio_value,
            "Step completed"
        );

        (self.state.clone(), reward, done)
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn is_terminal(&self) -> bool {
        self.state.terminal || self.current_step >= self.prices.len() - 1
    }

    fn observation_dim(&self) -> usize {
        self.config.lookback + 3
    }

    fn action_space(&self) -> &ActionSpace {
        // This is a static action space; we return a reference to a static value.
        // We use a leaked Box to provide a 'static reference.
        static ACTION_SPACE: std::sync::OnceLock<ActionSpace> = std::sync::OnceLock::new();
        ACTION_SPACE.get_or_init(ActionSpace::default)
    }

    fn set_reward_function(&mut self, reward_fn: Box<dyn RewardFunction>) {
        self.reward_fn = reward_fn;
    }

    fn render(&self) -> String {
        let pv = self.portfolio_value();
        let pnl = pv - self.initial_portfolio_value;
        let ret_pct = if self.initial_portfolio_value > 1e-10 {
            pnl / self.initial_portfolio_value * 100.0
        } else {
            0.0
        };
        format!(
            "Step: {}/{} | Cash: {:.2} | Position: {:.4} | Price: {:.2} | \
             Portfolio: {:.2} | PnL: {:.2} ({:.2}%) | Total Reward: {:.4}",
            self.current_step,
            self.prices.len() - 1,
            self.cash,
            self.position,
            self.current_price(),
            pv,
            pnl,
            ret_pct,
            self.state.total_reward,
        )
    }
}

// ─── PortfolioEnv ───────────────────────────────────────────────────────────

/// State for the multi-asset portfolio environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    /// Portfolio weights for each asset.
    pub weights: Vec<f64>,
    /// Total portfolio value.
    pub portfolio_value: f64,
    /// Individual asset values.
    pub asset_values: Vec<f64>,
}

/// Multi-asset portfolio allocation environment.
///
/// The agent observes normalized returns for each asset and outputs a weight
/// vector that sums to 1.0, representing the portfolio allocation.
#[derive(Debug)]
pub struct PortfolioEnv {
    /// Asset price matrices: prices[asset][time].
    prices: Vec<Vec<f64>>,
    /// Number of assets.
    n_assets: usize,
    /// Current time step.
    current_step: usize,
    /// Lookback window.
    lookback: usize,
    /// Current portfolio state.
    portfolio: PortfolioState,
    /// Initial portfolio value.
    pub initial_value: f64,
    /// Transaction cost.
    pub commission_rate: f64,
    /// State structure for the trait.
    state: State,
    /// Risk-free rate.
    risk_free_rate: f64,
}

impl PortfolioEnv {
    /// Create a new multi-asset portfolio environment.
    ///
    /// # Arguments
    ///
    /// * `prices` - Vector of price series, one per asset. All must have the same length.
    /// * `initial_value` - Starting portfolio value.
    pub fn new(prices: Vec<Vec<f64>>, initial_value: f64) -> anyhow::Result<Self> {
        if prices.is_empty() {
            anyhow::bail!("At least one asset price series required");
        }
        let len = prices[0].len();
        for (i, p) in prices.iter().enumerate() {
            if p.len() != len {
                anyhow::bail!("Asset {} has {} prices, expected {}", i, p.len(), len);
            }
        }
        let n_assets = prices.len();
        let lookback = 10;
        let weights = vec![1.0 / n_assets as f64; n_assets];
        let asset_values = vec![initial_value / n_assets as f64; n_assets];

        Ok(Self {
            prices,
            n_assets,
            current_step: lookback,
            lookback,
            portfolio: PortfolioState { weights, portfolio_value: initial_value, asset_values },
            initial_value,
            commission_rate: 0.001,
            risk_free_rate: 0.02 / 252.0,
            state: State::new(Observation::zeros(n_assets * lookback + n_assets)),
        })
    }

    /// Get current prices for all assets.
    fn current_prices(&self) -> Vec<f64> {
        self.prices.iter()
            .map(|p| p.get(self.current_step).copied().unwrap_or(0.0))
            .collect()
    }

    /// Previous prices for all assets.
    fn previous_prices(&self) -> Vec<f64> {
        let step = self.current_step.saturating_sub(1);
        self.prices.iter()
            .map(|p| p.get(step).copied().unwrap_or(0.0))
            .collect()
    }

    /// Compute turnover between two weight vectors.
    fn turnover(old_weights: &[f64], new_weights: &[f64]) -> f64 {
        old_weights.iter()
            .zip(new_weights.iter())
            .map(|(o, n)| (o - n).abs())
            .sum::<f64>() / 2.0
    }

    /// Build observation from current state.
    fn build_observation(&self) -> Observation {
        let mut features = Vec::new();

        // Normalized returns for each asset over lookback
        for asset_prices in &self.prices {
            for i in (0..self.lookback).rev() {
                let idx = self.current_step.saturating_sub(i + 1);
                let prev_idx = idx.saturating_sub(1);
                if prev_idx < asset_prices.len() && asset_prices[prev_idx] > 1e-10 {
                    features.push((asset_prices[idx] - asset_prices[prev_idx]) / asset_prices[prev_idx]);
                } else {
                    features.push(0.0);
                }
            }
        }

        // Current weights
        features.extend_from_slice(&self.portfolio.weights);

        Observation {
            features,
            timestamp: Utc::now(),
            portfolio_value: self.portfolio.portfolio_value,
            position: 0.0, // N/A for portfolio env
            unrealized_pnl: self.portfolio.portfolio_value - self.initial_value,
        }
    }

    /// Rebalance portfolio to target weights.
    fn rebalance(&mut self, new_weights: &[f64]) {
        let turnover = Self::turnover(&self.portfolio.weights, new_weights);
        let tx_cost = turnover * self.portfolio.portfolio_value * self.commission_rate;
        self.portfolio.portfolio_value -= tx_cost;

        // Update asset values based on new weights
        for (i, val) in self.portfolio.asset_values.iter_mut().enumerate() {
            *val = self.portfolio.portfolio_value * new_weights[i];
        }
        self.portfolio.weights = new_weights.to_vec();
    }
}

/// Portfolio actions — represent target weight allocations.
#[derive(Debug, Clone)]
pub enum PortfolioAction {
    /// Specific weight vector.
    Weights(Vec<f64>),
    /// Equal weight allocation.
    EqualWeight,
    /// Momentum — increase weights for top performers.
    Momentum,
    /// Hold current allocation.
    Hold,
}

impl PortfolioEnv {
    /// Execute a portfolio rebalance action.
    pub fn step_portfolio(&mut self, action: PortfolioAction) -> (State, Reward, bool) {
        let prev_value = self.portfolio.portfolio_value;

        // Apply price changes from previous step to current step
        let prev_prices = self.previous_prices();
        let curr_prices = self.current_prices();
        for (i, val) in self.portfolio.asset_values.iter_mut().enumerate() {
            if prev_prices[i] > 1e-10 {
                *val *= curr_prices[i] / prev_prices[i];
            }
        }
        self.portfolio.portfolio_value = self.portfolio.asset_values.iter().sum();

        // Execute action
        let target_weights = match action {
            PortfolioAction::Weights(w) => w,
            PortfolioAction::EqualWeight => {
                let w = 1.0 / self.n_assets as f64;
                vec![w; self.n_assets]
            }
            PortfolioAction::Momentum => {
                // Allocate more to assets with higher recent returns
                let mut returns: Vec<(usize, f64)> = Vec::new();
                for (i, prices) in self.prices.iter().enumerate() {
                    let idx = self.current_step;
                    let prev_idx = idx.saturating_sub(self.lookback);
                    let r = if prices.get(prev_idx).copied().unwrap_or(0.0) > 1e-10 {
                        prices.get(idx).copied().unwrap_or(0.0) / prices.get(prev_idx).copied().unwrap_or(1.0) - 1.0
                    } else {
                        0.0
                    };
                    returns.push((i, r));
                }
                returns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut weights = vec![0.0; self.n_assets];
                for (rank, (idx, _)) in returns.iter().enumerate() {
                    // Exponential weighting favoring top performers
                    weights[*idx] = (-(rank as f64) * 0.5).exp();
                }
                let total: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= total.max(1e-10);
                }
                weights
            }
            PortfolioAction::Hold => self.portfolio.weights.clone(),
        };

        self.rebalance(&target_weights);
        self.current_step += 1;

        let done = self.current_step >= self.prices[0].len() - 1;

        // Build observation
        let obs = self.build_observation();
        self.state = State {
            observation: obs.clone(),
            terminal: done,
            step: self.current_step,
            total_reward: self.state.total_reward,
            info: HashMap::new(),
        };

        // Compute reward
        let ret = if prev_value > 1e-10 {
            (self.portfolio.portfolio_value - prev_value) / prev_value
        } else {
            0.0
        };
        let reward = Reward::new(ret, RewardType::PercentReturn)
            .with_component("portfolio_value", self.portfolio.portfolio_value);
        self.state.total_reward += reward.value;

        (self.state.clone(), reward, done)
    }
}

impl TradingEnv for PortfolioEnv {
    fn reset(&mut self) -> State {
        self.current_step = self.lookback;
        let eq_weight = 1.0 / self.n_assets as f64;
        self.portfolio = PortfolioState {
            weights: vec![eq_weight; self.n_assets],
            portfolio_value: self.initial_value,
            asset_values: vec![self.initial_value / self.n_assets as f64; self.n_assets],
        };

        let obs = self.build_observation();
        self.state = State::new(obs);
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        // Map discrete actions to portfolio actions
        let portfolio_action = match action {
            Action::Hold => PortfolioAction::Hold,
            Action::Buy => PortfolioAction::EqualWeight,
            Action::Sell => PortfolioAction::Momentum,
            Action::Close => PortfolioAction::Weights(vec![1.0; self.n_assets]),
        };
        self.step_portfolio(portfolio_action)
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn is_terminal(&self) -> bool {
        self.state.terminal || self.current_step >= self.prices[0].len() - 1
    }

    fn observation_dim(&self) -> usize {
        self.n_assets * self.lookback + self.n_assets
    }

    fn action_space(&self) -> &ActionSpace {
        static ACTION_SPACE: std::sync::OnceLock<ActionSpace> = std::sync::OnceLock::new();
        ACTION_SPACE.get_or_init(ActionSpace::default)
    }

    fn set_reward_function(&mut self, _reward_fn: Box<dyn RewardFunction>) {
        // Portfolio env uses its own reward logic
    }
}

// ─── FuturesEnv ─────────────────────────────────────────────────────────────

/// Configuration for perpetual futures environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRateConfig {
    /// Base funding rate per 8 hours.
    pub base_rate: f64,
    /// Interest rate component.
    pub interest_rate: f64,
    /// Premium index deviation factor.
    pub premium_factor: f64,
    /// Maximum funding rate clamp.
    pub max_rate: f64,
    /// Funding interval in hours.
    pub interval_hours: f64,
}

impl Default for FundingRateConfig {
    fn default() -> Self {
        Self {
            base_rate: 0.0001,
            interest_rate: 0.02 / 365.0 / 3.0,
            premium_factor: 0.05,
            max_rate: 0.003,
            interval_hours: 8.0,
        }
    }
}

impl FundingRateConfig {
    /// Compute the current funding rate based on market conditions.
    pub fn compute_rate(&self, mark_price: f64, index_price: f64) -> f64 {
        let premium = if index_price > 1e-10 {
            (mark_price - index_price) / index_price
        } else {
            0.0
        };
        let rate = self.interest_rate + self.premium_factor * premium;
        rate.clamp(-self.max_rate, self.max_rate)
    }
}

/// Perpetual futures trading environment with funding rates and leverage.
///
/// Models realistic futures mechanics including:
/// - Long/short positions with leverage
/// - Periodic funding rate payments
/// - Liquidation when margin is insufficient
/// - Mark price vs index price basis
#[derive(Debug)]
pub struct FuturesEnv {
    /// Mark price series (may differ from index).
    mark_prices: Vec<f64>,
    /// Index price series (underlying spot).
    index_prices: Vec<f64>,
    /// Current time step.
    current_step: usize,
    /// Funding rate configuration.
    funding_config: FundingRateConfig,
    /// Leverage multiplier.
    leverage: f64,
    /// Initial margin (collateral).
    pub initial_margin: f64,
    /// Current margin.
    margin: f64,
    /// Position size in contracts (positive = long).
    position: f64,
    /// Entry price for the position.
    entry_price: f64,
    /// Number of funding intervals elapsed.
    funding_intervals: u64,
    /// Lookback for features.
    lookback: usize,
    /// Environment state.
    state: State,
    /// Cumulative funding paid/received.
    cumulative_funding: f64,
    /// Commission rate.
    commission_rate: f64,
    /// Whether the position has been liquidated.
    liquidated: bool,
}

impl FuturesEnv {
    /// Create a new futures environment.
    ///
    /// # Arguments
    ///
    /// * `mark_prices` - Perpetual futures mark price series.
    /// * `index_prices` - Underlying index price series.
    /// * `initial_margin` - Starting margin/collateral.
    /// * `leverage` - Maximum leverage multiplier.
    pub fn new(
        mark_prices: Vec<f64>,
        index_prices: Vec<f64>,
        initial_margin: f64,
        leverage: f64,
    ) -> anyhow::Result<Self> {
        if mark_prices.len() != index_prices.len() {
            anyhow::bail!("Mark and index price series must have the same length");
        }
        let lookback = 10;
        Ok(Self {
            mark_prices,
            index_prices,
            current_step: lookback,
            funding_config: FundingRateConfig::default(),
            leverage,
            initial_margin,
            margin: initial_margin,
            position: 0.0,
            entry_price: 0.0,
            funding_intervals: 0,
            lookback,
            state: State::new(Observation::zeros(lookback + 5)),
            cumulative_funding: 0.0,
            commission_rate: 0.0005,
            liquidated: false,
        })
    }

    /// Set the funding rate configuration.
    pub fn with_funding_config(mut self, config: FundingRateConfig) -> Self {
        self.funding_config = config;
        self
    }

    /// Current mark price.
    pub fn mark_price(&self) -> f64 {
        self.mark_prices.get(self.current_step).copied().unwrap_or(0.0)
    }

    /// Current index price.
    pub fn index_price(&self) -> f64 {
        self.index_prices.get(self.current_step).copied().unwrap_or(0.0)
    }

    /// Unrealized PnL including funding.
    pub fn unrealized_pnl(&self) -> f64 {
        let pnl = self.position * (self.mark_price() - self.entry_price);
        pnl + self.cumulative_funding
    }

    /// Check if the position should be liquidated.
    fn check_liquidation(&self) -> bool {
        if self.position.abs() < 1e-10 {
            return false;
        }
        let pnl = self.unrealized_pnl();
        let maintenance_margin = self.initial_margin * 0.005; // 0.5% maintenance margin
        let available = self.margin + pnl;
        available < maintenance_margin
    }

    /// Apply funding payment.
    fn apply_funding(&mut self) {
        if self.position.abs() < 1e-10 {
            return;
        }
        let rate = self.funding_config.compute_rate(self.mark_price(), self.index_price());
        let notional = self.position.abs() * self.mark_price();
        let funding = rate * notional;
        // Long pays short when funding is positive (mark > index)
        let payment = if self.position > 0.0 { -funding } else { funding };
        self.margin += payment;
        self.cumulative_funding += payment;
        self.funding_intervals += 1;

        tracing::debug!(
            rate = rate,
            notional = notional,
            payment = payment,
            margin = self.margin,
            "Funding applied"
        );
    }

    /// Execute a long position.
    fn open_long(&mut self, contracts: f64) {
        let price = self.mark_price();
        let notional = contracts * price;
        let required_margin = notional / self.leverage;
        let commission = notional * self.commission_rate;

        if required_margin + commission > self.margin {
            tracing::warn!("Insufficient margin for long position");
            return;
        }

        self.margin -= required_margin + commission;
        self.position += contracts;

        if self.position.abs() > 1e-10 {
            self.entry_price = (self.entry_price * (self.position - contracts) + price * contracts) / self.position;
        } else {
            self.entry_price = price;
        }

        tracing::debug!(contracts = contracts, price = price, margin = self.margin, "Opened long");
    }

    /// Execute a short position.
    fn open_short(&mut self, contracts: f64) {
        let price = self.mark_price();
        let notional = contracts * price;
        let required_margin = notional / self.leverage;
        let commission = notional * self.commission_rate;

        if required_margin + commission > self.margin {
            tracing::warn!("Insufficient margin for short position");
            return;
        }

        self.margin -= required_margin + commission;
        self.position -= contracts;

        if self.position.abs() > 1e-10 {
            self.entry_price = (self.entry_price * (self.position + contracts) + price * contracts) / self.position.abs();
        } else {
            self.entry_price = price;
        }

        tracing::debug!(contracts = contracts, price = price, margin = self.margin, "Opened short");
    }

    /// Close the entire position.
    fn close_position(&mut self) -> f64 {
        if self.position.abs() < 1e-10 {
            return 0.0;
        }
        let price = self.mark_price();
        let notional = self.position.abs() * price;
        let commission = notional * self.commission_rate;
        let pnl = self.position * (price - self.entry_price);
        self.margin += pnl - commission;

        tracing::debug!(
            contracts = self.position.abs(),
            price = price,
            pnl = pnl,
            margin = self.margin,
            "Position closed"
        );

        let _contracts = self.position.abs();
        self.position = 0.0;
        self.entry_price = 0.0;
        pnl - commission
    }

    /// Build observation vector.
    fn build_observation(&self) -> Observation {
        let mut features = Vec::new();

        // Mark price returns over lookback
        for i in (0..self.lookback).rev() {
            let idx = self.current_step.saturating_sub(i + 1);
            let prev_idx = idx.saturating_sub(1);
            if prev_idx < self.mark_prices.len() && self.mark_prices[prev_idx] > 1e-10 {
                features.push((self.mark_prices[idx] - self.mark_prices[prev_idx]) / self.mark_prices[prev_idx]);
            } else {
                features.push(0.0);
            }
        }

        // Basis (mark - index) / index
        let basis = if self.index_price() > 1e-10 {
            (self.mark_price() - self.index_price()) / self.index_price()
        } else {
            0.0
        };
        features.push(basis);

        // Position (normalized by leverage)
        features.push(self.position / self.leverage);

        // Margin ratio
        features.push(self.margin / self.initial_margin);

        // Unrealized PnL ratio
        features.push(self.unrealized_pnl() / self.initial_margin);

        Observation {
            features,
            timestamp: Utc::now(),
            portfolio_value: self.margin + self.unrealized_pnl(),
            position: self.position,
            unrealized_pnl: self.unrealized_pnl(),
        }
    }

    /// Update internal state.
    fn update_state(&mut self) {
        let obs = self.build_observation();
        self.state = State {
            observation: obs,
            terminal: self.liquidated,
            step: self.current_step,
            total_reward: self.state.total_reward,
            info: HashMap::new(),
        };
    }
}

impl TradingEnv for FuturesEnv {
    fn reset(&mut self) -> State {
        self.current_step = self.lookback;
        self.margin = self.initial_margin;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.funding_intervals = 0;
        self.cumulative_funding = 0.0;
        self.liquidated = false;

        self.update_state();
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        let prev_value = self.margin + self.unrealized_pnl();

        match action {
            Action::Buy => {
                if self.position < 0.0 {
                    self.close_position();
                }
                // Open max long with available margin
                let available = self.margin * 0.95; // leave 5% buffer
                let price = self.mark_price();
                let notional = available * self.leverage;
                let contracts = if price > 1e-10 { notional / price } else { 0.0 };
                if contracts > 0.0 {
                    self.open_long(contracts);
                }
            }
            Action::Sell => {
                if self.position > 0.0 {
                    self.close_position();
                }
                let available = self.margin * 0.95;
                let price = self.mark_price();
                let notional = available * self.leverage;
                let contracts = if price > 1e-10 { notional / price } else { 0.0 };
                if contracts > 0.0 {
                    self.open_short(contracts);
                }
            }
            Action::Close => {
                self.close_position();
            }
            Action::Hold => {}
        }

        // Advance time
        self.current_step += 1;

        // Apply funding every `interval_hours` worth of steps (approx 3 steps per day = ~8 hours)
        if self.current_step % 3 == 0 {
            self.apply_funding();
        }

        // Check liquidation
        if self.check_liquidation() {
            tracing::warn!(
                margin = self.margin,
                pnl = self.unrealized_pnl(),
                "LIQUIDATED"
            );
            self.liquidated = true;
            self.position = 0.0;
            self.entry_price = 0.0;
        }

        let done = self.liquidated || self.current_step >= self.mark_prices.len() - 1;

        if done && !self.liquidated {
            self.close_position();
        }

        self.update_state();

        let curr_value = self.margin + self.unrealized_pnl();
        let reward = if prev_value > 1e-10 {
            Reward::new((curr_value - prev_value) / prev_value, RewardType::PercentReturn)
        } else {
            Reward::new(0.0, RewardType::PercentReturn)
        };

        self.state.total_reward += reward.value;

        (self.state.clone(), reward, done)
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn is_terminal(&self) -> bool {
        self.state.terminal
    }

    fn observation_dim(&self) -> usize {
        self.lookback + 5
    }

    fn action_space(&self) -> &ActionSpace {
        static ACTION_SPACE: std::sync::OnceLock<ActionSpace> = std::sync::OnceLock::new();
        ACTION_SPACE.get_or_init(ActionSpace::default)
    }

    fn set_reward_function(&mut self, _reward_fn: Box<dyn RewardFunction>) {
        // Futures env uses its own reward logic
    }
}

// ─── OrderBookEnv ───────────────────────────────────────────────────────────

/// Side of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Type of order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

/// Status of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// A single order in the order book environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: f64,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub status: OrderStatus,
    pub created_step: usize,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
}

impl Order {
    /// Create a new limit order.
    pub fn limit(id: &str, side: OrderSide, price: f64, quantity: f64, step: usize) -> Self {
        Self {
            id: id.to_string(),
            side,
            order_type: OrderType::Limit,
            price,
            quantity,
            filled_quantity: 0.0,
            status: OrderStatus::Pending,
            created_step: step,
            limit_price: Some(price),
            stop_price: None,
        }
    }

    /// Create a new market order.
    pub fn market(id: &str, side: OrderSide, quantity: f64, step: usize) -> Self {
        Self {
            id: id.to_string(),
            side,
            order_type: OrderType::Market,
            price: 0.0, // filled at market
            quantity,
            filled_quantity: 0.0,
            status: OrderStatus::Pending,
            created_step: step,
            limit_price: None,
            stop_price: None,
        }
    }

    /// Remaining quantity to fill.
    pub fn remaining(&self) -> f64 {
        self.quantity - self.filled_quantity
    }

    /// Fill ratio (0.0 to 1.0).
    pub fn fill_ratio(&self) -> f64 {
        if self.quantity < 1e-10 { 1.0 } else { self.filled_quantity / self.quantity }
    }
}

/// A simulated order book for limit order trading.
///
/// Provides a simplified limit-order-book simulation where the agent can
/// place limit orders at various price levels and observe fill dynamics.
#[derive(Debug)]
pub struct OrderBookEnv {
    /// Mid-price time series.
    mid_prices: Vec<f64>,
    /// Simulated spread as a fraction of mid price.
    spread_frac: f64,
    /// Current time step.
    current_step: usize,
    /// Lookback window for features.
    lookback: usize,
    /// Agent's cash balance.
    cash: f64,
    /// Current inventory (positive = long, negative = short).
    inventory: f64,
    /// Active orders.
    orders: Vec<Order>,
    /// Filled orders history.
    filled_orders: Vec<Order>,
    /// Next order ID counter.
    next_order_id: u64,
    /// Max inventory allowed.
    max_inventory: f64,
    /// Commission rate.
    commission_rate: f64,
    /// Environment state.
    state: State,
    /// Initial cash.
    pub initial_cash: f64,
}

impl OrderBookEnv {
    /// Create a new order book environment.
    ///
    /// # Arguments
    ///
    /// * `mid_prices` - Mid-price time series.
    /// * `initial_cash` - Starting cash.
    pub fn new(mid_prices: Vec<f64>, initial_cash: f64) -> Self {
        let lookback = 10;
        let obs_dim = lookback + 5; // returns + spread + inventory + cash + position_value
        let env = Self {
            mid_prices,
            spread_frac: 0.001,
            current_step: lookback,
            lookback,
            cash: initial_cash,
            inventory: 0.0,
            orders: Vec::new(),
            filled_orders: Vec::new(),
            next_order_id: 1,
            max_inventory: 1000.0,
            commission_rate: 0.0002,
            state: State::new(Observation::zeros(obs_dim)),
            initial_cash,
        };
        env
    }

    /// Current mid price.
    pub fn mid_price(&self) -> f64 {
        self.mid_prices.get(self.current_step).copied().unwrap_or(0.0)
    }

    /// Current best bid.
    pub fn best_bid(&self) -> f64 {
        self.mid_price() * (1.0 - self.spread_frac / 2.0)
    }

    /// Current best ask.
    pub fn best_ask(&self) -> f64 {
        self.mid_price() * (1.0 + self.spread_frac / 2.0)
    }

    /// Place a new order.
    pub fn place_order(&mut self, side: OrderSide, order_type: OrderType, price: f64, quantity: f64) -> &Order {
        let id = format!("ord-{}", self.next_order_id);
        self.next_order_id += 1;

        let order = match order_type {
            OrderType::Market => Order::market(&id, side, quantity, self.current_step),
            _ => Order::limit(&id, side, price, quantity, self.current_step),
        };

        self.orders.push(order);
        self.orders.last().unwrap()
    }

    /// Cancel an order by ID.
    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        if let Some(order) = self.orders.iter_mut().find(|o| o.id == order_id) {
            if order.status == OrderStatus::Pending {
                order.status = OrderStatus::Cancelled;
                return true;
            }
        }
        false
    }

    /// Process pending orders against the current market.
    fn process_orders(&mut self) -> Vec<Order> {
        let _mid = self.mid_price();
        let bid = self.best_bid();
        let ask = self.best_ask();
        let mut newly_filled = Vec::new();

        for order in &mut self.orders {
            if order.status != OrderStatus::Pending {
                continue;
            }

            // Check if limit price is crossed
            let should_fill = match (order.side, order.order_type) {
                (OrderSide::Buy, OrderType::Limit) => {
                    if let Some(lp) = order.limit_price {
                        ask <= lp // buy limit fills when ask <= limit price
                    } else {
                        false
                    }
                }
                (OrderSide::Sell, OrderType::Limit) => {
                    if let Some(lp) = order.limit_price {
                        bid >= lp // sell limit fills when bid >= limit price
                    } else {
                        false
                    }
                }
                (OrderSide::Buy, OrderType::Market) => true,
                (OrderSide::Sell, OrderType::Market) => true,
                _ => false,
            };

            if should_fill {
                let fill_price = match order.order_type {
                    OrderType::Market => match order.side {
                        OrderSide::Buy => ask,
                        OrderSide::Sell => bid,
                    },
                    _ => order.price,
                };

                // Check affordability / inventory limits
                let max_qty = match order.side {
                    OrderSide::Buy => {
                        let affordable = self.cash / (fill_price * (1.0 + self.commission_rate));
                        let inv_room = self.max_inventory - self.inventory;
                        order.remaining().min(affordable).max(0.0).min(inv_room.max(0.0))
                    }
                    OrderSide::Sell => {
                        let inv_room = self.max_inventory + self.inventory; // inventory can be negative
                        order.remaining().max(0.0).min(inv_room.max(0.0))
                    }
                };

                if max_qty > 0.0 {
                    let commission = max_qty * fill_price * self.commission_rate;
                    order.filled_quantity += max_qty;
                    order.price = fill_price;
                    order.status = OrderStatus::Filled;

                    // Update cash and inventory
                    match order.side {
                        OrderSide::Buy => {
                            self.cash -= max_qty * fill_price + commission;
                            self.inventory += max_qty;
                        }
                        OrderSide::Sell => {
                            self.cash += max_qty * fill_price - commission;
                            self.inventory -= max_qty;
                        }
                    }

                    newly_filled.push(order.clone());
                    self.filled_orders.push(order.clone());
                    tracing::debug!(
                        order_id = %order.id,
                        side = ?order.side,
                        qty = max_qty,
                        price = fill_price,
                        "Order filled"
                    );
                }
            }
        }

        // Remove filled and cancelled orders
        self.orders.retain(|o| o.status == OrderStatus::Pending);
        newly_filled
    }

    /// Build the observation vector.
    fn build_observation(&self) -> Observation {
        let mut features = Vec::new();

        // Normalized returns over lookback
        for i in (0..self.lookback).rev() {
            let idx = self.current_step.saturating_sub(i + 1);
            let prev_idx = idx.saturating_sub(1);
            if prev_idx < self.mid_prices.len() && self.mid_prices[prev_idx] > 1e-10 {
                features.push((self.mid_prices[idx] - self.mid_prices[prev_idx]) / self.mid_prices[prev_idx]);
            } else {
                features.push(0.0);
            }
        }

        // Spread (normalized)
        features.push(self.spread_frac);

        // Inventory (normalized)
        features.push(self.inventory / self.max_inventory);

        // Cash ratio
        let pv = self.portfolio_value();
        features.push(if pv > 1e-10 { self.cash / pv } else { 1.0 });

        // Unrealized PnL ratio
        let unrealized = self.inventory * self.mid_price();
        features.push(if pv > 1e-10 { unrealized / pv } else { 0.0 });

        Observation {
            features,
            timestamp: Utc::now(),
            portfolio_value: pv,
            position: self.inventory,
            unrealized_pnl: unrealized,
        }
    }

    /// Current portfolio value.
    pub fn portfolio_value(&self) -> f64 {
        self.cash + self.inventory * self.mid_price()
    }

    /// Update internal state.
    fn update_state(&mut self) {
        let obs = self.build_observation();
        self.state = State {
            observation: obs,
            terminal: false,
            step: self.current_step,
            total_reward: self.state.total_reward,
            info: HashMap::new(),
        };
    }
}

impl TradingEnv for OrderBookEnv {
    fn reset(&mut self) -> State {
        self.current_step = self.lookback;
        self.cash = self.initial_cash;
        self.inventory = 0.0;
        self.orders.clear();
        self.filled_orders.clear();
        self.next_order_id = 1;

        self.update_state();
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        let prev_value = self.portfolio_value();

        // Map actions to order placements
        match action {
            Action::Buy => {
                // Place a buy limit order at the best bid
                self.place_order(OrderSide::Buy, OrderType::Limit, self.best_ask(), 100.0);
            }
            Action::Sell => {
                // Place a sell limit order at the best ask
                self.place_order(OrderSide::Sell, OrderType::Limit, self.best_bid(), 100.0);
            }
            Action::Close => {
                // Cancel all orders and liquidate
                let ids: Vec<String> = self.orders.iter().map(|o| o.id.clone()).collect();
                for id in &ids {
                    self.cancel_order(id);
                }
                // Market sell to close inventory
                if self.inventory > 0.0 {
                    self.place_order(OrderSide::Sell, OrderType::Market, 0.0, self.inventory);
                } else if self.inventory < 0.0 {
                    self.place_order(OrderSide::Buy, OrderType::Market, 0.0, self.inventory.abs());
                }
            }
            Action::Hold => {}
        }

        // Process orders
        let _filled = self.process_orders();

        // Advance time
        self.current_step += 1;
        let done = self.current_step >= self.mid_prices.len() - 1;

        // Process orders at the new price level
        let _filled = self.process_orders();

        self.update_state();

        let curr_value = self.portfolio_value();
        let reward = if prev_value > 1e-10 {
            Reward::pnl(curr_value - prev_value)
        } else {
            Reward::pnl(0.0)
        };

        self.state.total_reward += reward.value;
        if done {
            self.state.terminal = true;
        }

        (self.state.clone(), reward, done)
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn is_terminal(&self) -> bool {
        self.state.terminal || self.current_step >= self.mid_prices.len() - 1
    }

    fn observation_dim(&self) -> usize {
        self.lookback + 5
    }

    fn action_space(&self) -> &ActionSpace {
        static ACTION_SPACE: std::sync::OnceLock<ActionSpace> = std::sync::OnceLock::new();
        ACTION_SPACE.get_or_init(ActionSpace::default)
    }

    fn set_reward_function(&mut self, _reward_fn: Box<dyn RewardFunction>) {
        // OrderBook env uses its own reward logic
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_prices(n: usize, start: f64, volatility: f64) -> Vec<f64> {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut prices = vec![start];
        for _ in 1..n {
            let ret = rng.random_range(-volatility..volatility);
            let last = *prices.last().unwrap();
            prices.push((last * (1.0 + ret)).max(0.01));
        }
        prices
    }

    #[test]
    fn test_action_conversions() {
        assert_eq!(Action::Hold.to_index(), 0);
        assert_eq!(Action::Buy.to_index(), 1);
        assert_eq!(Action::Sell.to_index(), 2);
        assert_eq!(Action::Close.to_index(), 3);
        assert_eq!(Action::from_index(0), Action::Hold);
        assert_eq!(Action::from_index(1), Action::Buy);
        assert_eq!(Action::all().len(), 4);
    }

    #[test]
    fn test_observation() {
        let obs = Observation::new(vec![1.0, 2.0, 3.0], Utc::now());
        assert_eq!(obs.dim(), 3);
        let obs2 = Observation::zeros(5);
        assert_eq!(obs2.dim(), 5);
        assert_eq!(obs2.features.len(), 5);
    }

    #[test]
    fn test_pnl_reward() {
        let reward_fn = PnLReward;
        let reward = reward_fn.compute(100_000.0, 101_000.0, Action::Buy, 1.0, &[], 0.0);
        assert!((reward.value - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_log_return_reward() {
        let reward_fn = LogReturnReward;
        let reward = reward_fn.compute(100.0, 110.0, Action::Buy, 1.0, &[], 0.0);
        let expected = (110.0f64 / 100.0f64).ln();
        assert!((reward.value - expected).abs() < 0.001);
    }

    #[test]
    fn test_risk_adjusted_reward() {
        let reward_fn = RiskAdjustedReward::default();
        let history = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.05, -0.03, 0.01, -0.02, 0.04];
        let reward = reward_fn.compute(100.000, 100.500, Action::Buy, 1.0, &history, 0.0001);
        // Just verify it doesn't panic and returns a finite value
        assert!(reward.value.is_finite());
    }

    #[test]
    fn test_differential_sharpe_reward() {
        let mut reward_fn = DifferentialSharpeReward::default();
        let history = vec![0.01, -0.02, 0.03];
        let reward = reward_fn.compute(100.0, 101.0, Action::Buy, 1.0, &history, 0.0);
        assert!(reward.value.is_finite());
    }

    #[test]
    fn test_simple_trading_env_reset() {
        let prices = generate_prices(200, 100.0, 0.02);
        let mut env = SimpleTradingEnv::new(prices, 100_000.0);
        let state = env.reset();
        assert!(!state.terminal);
        assert!((state.observation.portfolio_value - 100_000.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_trading_env_step() {
        let prices = generate_prices(200, 100.0, 0.02);
        let mut env = SimpleTradingEnv::new(prices, 100_000.0);
        env.reset();

        let (state, reward, done) = env.step(Action::Hold);
        assert!(!done);
        assert!(reward.value.is_finite());
        assert_eq!(state.step, env.config.lookback + 1);
    }

    #[test]
    fn test_simple_trading_env_full_episode() {
        let prices = generate_prices(100, 100.0, 0.03);
        let mut env = SimpleTradingEnv::new(prices, 100_000.0);
        env.reset();

        let mut total_reward = 0.0;
        let mut steps = 0;
        loop {
            let (state, reward, done) = env.step(Action::Hold);
            total_reward += reward.value;
            steps += 1;
            if done {
                break;
            }
        }
        assert!(steps > 0);
        assert!(total_reward.is_finite());
    }

    #[test]
    fn test_simple_trading_env_buy_sell() {
        let prices = generate_prices(200, 100.0, 0.02);
        let mut env = SimpleTradingEnv::new(prices, 100_000.0);
        env.reset();

        // Buy
        let (_, _, done) = env.step(Action::Buy);
        assert!(!done);
        assert!(env.position() > 0.0);
        assert!(env.cash() < 100_000.0); // spent some cash

        // Sell (close)
        let (_, _, _) = env.step(Action::Close);
        assert!((env.position() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reward_functions() {
        let mut reward = Reward::pnl(100.0);
        reward = reward + Reward::new(50.0, RewardType::PnL);
        assert!((reward.value - 150.0).abs() < 0.01);

        let ra = Reward::risk_adjusted(0.05, 0.1);
        assert!((ra.value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_env() {
        let prices_a = generate_prices(100, 100.0, 0.02);
        let prices_b = generate_prices(100, 50.0, 0.03);
        let mut env = PortfolioEnv::new(vec![prices_a, prices_b], 100_000.0).unwrap();
        let state = env.reset();
        assert!(!state.terminal);
        assert_eq!(env.n_assets, 2);

        let (_, reward, done) = env.step(Action::Hold);
        assert!(!done);
        assert!(reward.value.is_finite());
    }

    #[test]
    fn test_portfolio_momentum_action() {
        let prices_a = generate_prices(100, 100.0, 0.02);
        let prices_b = generate_prices(100, 50.0, 0.03);
        let mut env = PortfolioEnv::new(vec![prices_a, prices_b], 100_000.0).unwrap();
        env.reset();

        let (_, _, done) = env.step(Action::Sell); // maps to Momentum
        assert!(!done);
        let sum: f64 = env.portfolio.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // weights sum to 1
    }

    #[test]
    fn test_funding_rate_config() {
        let config = FundingRateConfig::default();
        let rate = config.compute_rate(101.0, 100.0);
        assert!(rate > 0.0); // positive when mark > index

        let rate_neg = config.compute_rate(99.0, 100.0);
        assert!(rate_neg < 0.0); // negative when mark < index

        assert!(rate <= config.max_rate);
        assert!(rate_neg >= -config.max_rate);
    }

    #[test]
    fn test_futures_env() {
        let mark = generate_prices(200, 100.0, 0.02);
        let index = mark.iter().map(|p| p * 0.99).collect(); // slight basis
        let mut env = FuturesEnv::new(mark, index, 10_000.0, 5.0).unwrap();
        let state = env.reset();
        assert!(!state.terminal);
        assert!(!env.liquidated);
    }

    #[test]
    fn test_futures_env_long_position() {
        let mark = generate_prices(200, 100.0, 0.02);
        let index = mark.clone();
        let mut env = FuturesEnv::new(mark, index, 10_000.0, 5.0).unwrap();
        env.reset();

        env.step(Action::Buy);
        assert!(env.position > 0.0);
        assert!(env.margin < 10_000.0); // used some margin

        env.step(Action::Close);
        assert!((env.position).abs() < 1e-10);
    }

    #[test]
    fn test_order_book_env() {
        let mid_prices = generate_prices(200, 100.0, 0.01);
        let mut env = OrderBookEnv::new(mid_prices, 100_000.0);
        let state = env.reset();
        assert!(!state.terminal);
        assert!((env.cash - 0.0).abs() > 0.0); // cash starts at 0 since we build but don't init
    }

    #[test]
    fn test_order_placement() {
        let mid_prices = generate_prices(200, 100.0, 0.01);
        let mut env = OrderBookEnv::new(mid_prices, 100_000.0);
        env.reset();

        env.place_order(OrderSide::Buy, OrderType::Limit, 99.0, 10.0);
        assert_eq!(env.orders.len(), 1);
        assert_eq!(env.orders[0].status, OrderStatus::Pending);

        env.place_order(OrderSide::Sell, OrderType::Market, 0.0, 5.0);
        assert_eq!(env.orders.len(), 2);
    }

    #[test]
    fn test_order_cancellation() {
        let mid_prices = generate_prices(200, 100.0, 0.01);
        let mut env = OrderBookEnv::new(mid_prices, 100_000.0);
        env.reset();

        let order = env.place_order(OrderSide::Buy, OrderType::Limit, 95.0, 10.0);
        let id = order.id.clone();
        assert!(env.cancel_order(&id));
        assert_eq!(env.orders[0].status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_simple_trading_env_render() {
        let prices = generate_prices(200, 100.0, 0.02);
        let mut env = SimpleTradingEnv::new(prices, 100_000.0);
        env.reset();
        let render = env.render();
        assert!(render.contains("Step:"));
        assert!(render.contains("Cash:"));
    }

    #[test]
    fn test_action_space() {
        let space = ActionSpace::default();
        assert_eq!(space.n, 4);
        assert_eq!(space.actions.len(), 4);
    }
}
