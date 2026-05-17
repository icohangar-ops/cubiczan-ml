//! # PyO3 bindings for cubiczan-ml-rl
//!
//! Exposes the RL crate's trading environments, Q-learning agents,
//! and backtest results to Python.

use pyo3::prelude::*;

use cubiczan_ml_rl::agents::{
    self, Agent, QLearningAgent,
};
use cubiczan_ml_rl::backtest::{self, BacktestEngine};
use cubiczan_ml_rl::environment::{Action as EnvAction, SimpleTradingEnv, TradingEnv};
use cubiczan_ml_rl::exploration::EpsilonGreedy;

// ---------------------------------------------------------------------------
// PyTradingEnv — wraps SimpleTradingEnv
// ---------------------------------------------------------------------------

/// Single-asset trading environment with buy/sell/hold/close actions.
///
/// Wraps the Rust ``SimpleTradingEnv`` and exposes a Gym-like interface
/// (``reset`` / ``step``) that returns plain Python dicts.
#[pyclass(name = "TradingEnv")]
pub struct PyTradingEnv {
    env: SimpleTradingEnv,
}

#[pymethods]
impl PyTradingEnv {
    /// Create a new trading environment.
    ///
    /// Parameters
    /// ----------
    /// prices : list[float]
    ///     Historical close prices.
    /// initial_cash : float
    ///     Starting cash balance.
    #[new]
    #[pyo3(signature = (prices, initial_cash))]
    pub fn new(prices: Vec<f64>, initial_cash: f64) -> Self {
        let mut env = SimpleTradingEnv::new(prices, initial_cash);
        // The Rust env must be reset before first use.
        let _ = env.reset();
        Self { env }
    }

    /// Reset the environment and return the initial state dict.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"features": list[float], "terminal": bool}``
    pub fn reset(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.env.reset();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("features", state.observation.features.clone())?;
        dict.set_item("terminal", state.terminal)?;
        Ok(dict.into())
    }

    /// Execute one time step.
    ///
    /// Parameters
    /// ----------
    /// action_idx : int
    ///     0 = Hold, 1 = Buy, 2 = Sell, 3 = Close.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"features", "reward", "terminal", "position", "cash", "portfolio_value"}``
    pub fn step(&mut self, py: Python<'_>, action_idx: usize) -> PyResult<PyObject> {
        let action = EnvAction::from_index(action_idx);
        let (state, reward, _done) = self.env.step(action);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("features", state.observation.features)?;
        dict.set_item("reward", reward.value)?;
        dict.set_item("terminal", state.terminal)?;
        dict.set_item("position", state.observation.position)?;
        dict.set_item("cash", self.env.cash())?;
        dict.set_item("portfolio_value", self.env.portfolio_value())?;
        Ok(dict.into())
    }

    /// Current price in the data series.
    pub fn current_price(&self) -> f64 {
        self.env.current_price()
    }

    /// Current cash balance.
    pub fn cash(&self) -> f64 {
        self.env.cash()
    }

    /// Current position size (positive = long, negative = short).
    pub fn position(&self) -> f64 {
        self.env.position()
    }

    /// Current portfolio value (cash + position * price).
    pub fn portfolio_value(&self) -> f64 {
        self.env.portfolio_value()
    }

    /// Whether the episode has ended.
    pub fn is_terminal(&self) -> bool {
        self.env.is_terminal()
    }

    /// Dimensionality of the observation (feature) space.
    pub fn observation_dim(&self) -> usize {
        self.env.observation_dim()
    }

    /// Render a human-readable summary of the current state.
    pub fn render(&self) -> String {
        self.env.render()
    }
}

// ---------------------------------------------------------------------------
// PyQLearningAgent — wraps QLearningAgent<EpsilonGreedy>
// ---------------------------------------------------------------------------

/// Tabular Q-learning agent with epsilon-greedy exploration.
///
/// Wraps the Rust ``QLearningAgent<EpsilonGreedy>``.  The agent operates
/// on a simplified ``agents::State`` (a plain ``Vec<f64>`` feature vector).
#[pyclass(name = "QLearningAgent")]
pub struct PyQLearningAgent {
    agent: QLearningAgent<EpsilonGreedy>,
}

#[pymethods]
impl PyQLearningAgent {
    /// Create a new Q-learning agent.
    ///
    /// Parameters
    /// ----------
    /// state_size : int
    ///     Dimensionality of the state vector (informational; the Q-table
    ///     is a HashMap so no pre-allocation is needed).
    /// action_size : int
    ///     Number of discrete actions (informational; the agent hardcodes 3
    ///     actions: Hold, Buy, Sell).
    /// learning_rate : float
    ///     Q-learning step size (alpha).
    /// discount_factor : float
    ///     Discount factor (gamma) for future rewards.
    /// epsilon : float
    ///     Initial exploration rate.  ``epsilon_min`` defaults to 0.01 and
    ///     ``epsilon_decay`` to 0.995.
    #[new]
    #[pyo3(signature = (_state_size, _action_size, learning_rate, discount_factor, epsilon))]
    pub fn new(
        _state_size: usize,
        _action_size: usize,
        learning_rate: f64,
        discount_factor: f64,
        epsilon: f64,
    ) -> Self {
        let exploration = EpsilonGreedy::new(epsilon, 0.01, 0.995);
        let agent = QLearningAgent::new(learning_rate, discount_factor, exploration);
        Self { agent }
    }

    /// Select an action given a state feature vector.
    ///
    /// Parameters
    /// ----------
    /// state : list[float]
    ///     Current observation features.
    ///
    /// Returns
    /// -------
    /// int
    ///     Selected action index (0 = Hold, 1 = Buy, 2 = Sell).
    pub fn select_action(&mut self, state: Vec<f64>) -> usize {
        let agent_state = agents::State::new(state);
        let action = self.agent.act(&agent_state);
        action.index()
    }

    /// Update the Q-table from a single transition.
    ///
    /// Parameters
    /// ----------
    /// state : list[float]
    ///     Current state features.
    /// action : int
    ///     Action index taken (0–2).
    /// reward : float
    ///     Scalar reward received.
    /// next_state : list[float]
    ///     Next state features.
    /// done : bool
    ///     Whether the episode ended after this transition.
    pub fn update(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        let s = agents::State::new(state);
        let a = agents::Action::from_index(action);
        let ns = agents::State::new(next_state);
        self.agent.learn(&s, a, reward, &ns, done);
    }

    /// Number of distinct state entries in the Q-table.
    pub fn q_table_size(&self) -> usize {
        self.agent.q_table_size()
    }
}

// ---------------------------------------------------------------------------
// PyBacktestResult — lightweight data class for backtest outcomes
// ---------------------------------------------------------------------------

/// Summary of a backtest run exposed to Python.
///
/// This is a thin data class.  To obtain one, use
/// :func:`run_backtest` which delegates to the Rust ``BacktestEngine``.
#[pyclass(name = "BacktestResult")]
#[derive(Clone)]
pub struct PyBacktestResult {
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub total_trades: f64,
    #[pyo3(get)]
    pub final_value: f64,
}

#[pymethods]
impl PyBacktestResult {
    /// Return a summary string.
    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(total_return={:.4}, sharpe_ratio={:.4}, max_drawdown={:.4}, \
             win_rate={:.4}, total_trades={:.0}, final_value={:.2})",
            self.total_return,
            self.sharpe_ratio,
            self.max_drawdown,
            self.win_rate,
            self.total_trades,
            self.final_value,
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level helper: run_backtest
// ---------------------------------------------------------------------------

/// Run a signal-based backtest on historical price data.
///
/// Parameters
/// ----------
/// prices : list[float]
///     Historical close prices.
/// signals : list[float]
///     Trading signals per bar.  Positive (>0.1) → buy, negative (<-0.1) → sell, else hold.
/// initial_capital : float
///     Starting cash balance (default 100 000).
///
/// Returns
/// -------
/// BacktestResult
///     Summary of the backtest performance.
#[pyfunction]
#[pyo3(signature = (prices, signals, initial_capital = 100_000.0))]
pub fn run_backtest(
    prices: Vec<f64>,
    signals: Vec<f64>,
    initial_capital: f64,
) -> PyBacktestResult {
    let result = BacktestEngine::run(
        &prices,
        &signals,
        initial_capital,
        &backtest::CommissionModel::default(),
        &backtest::SlippageModel::default(),
    );

    let report = &result.report;
    PyBacktestResult {
        total_return: report.total_return,
        sharpe_ratio: report.sharpe_ratio,
        max_drawdown: report.max_drawdown,
        win_rate: report.win_rate,
        total_trades: report.total_trades as f64,
        final_value: result.equity_curve.last().copied().unwrap_or(initial_capital),
    }
}

// ---------------------------------------------------------------------------
// Python-side tests (run via pytest after `pip install cubiczan-ml`)
// ---------------------------------------------------------------------------
