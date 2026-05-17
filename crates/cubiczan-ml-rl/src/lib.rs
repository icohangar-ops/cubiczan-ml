//! # CubicZan ML — Reinforcement Learning
//!
//! A comprehensive reinforcement learning framework for financial trading and portfolio
//! management. Provides trading environments, RL agents (Q-learning, DQN, policy gradient,
//! actor-critic), exploration strategies, trading policies, and a full backtesting engine.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    BacktestEngine                         │
//! │  ┌─────────┐   ┌──────────┐   ┌───────────────────────┐ │
//! │  │  Agent   │──▶│  Env     │──▶│  Reward Function      │ │
//! │  │(DQN/AC) │◀──│(Trading) │   │(PnL/Sharpe/RiskAdj)  │ │
//! │  └────┬────┘   └──────────┘   └───────────────────────┘ │
//! │       │                                                  │
//! │  ┌────▼──────────────────────────────────────┐          │
//! │  │  Exploration  │  Policy  │  Slippage/Comm  │          │
//! │  └───────────────────────────────────────────┘          │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```ignore
//! use cubiczan_ml_rl::{
//!     environment::{SimpleTradingEnv, TradingEnv},
//!     agents::{QLearningAgent, Agent},
//!     exploration::EpsilonGreedy,
//!     backtest::BacktestEngine,
//! };
//!
//! let mut env = SimpleTradingEnv::new(price_data, 100_000.0);
//! let mut agent = QLearningAgent::new(0.1, 0.99, EpsilonGreedy::new(1.0, 0.01, 0.995));
//!
//! // Train
//! for episode in 0..1000 {
//!     env.reset();
//!     let mut total_reward = 0.0;
//!     while !env.is_terminal() {
//!         let action = agent.act(&env.state());
//!         let (reward, _done) = env.step(action);
//!         agent.learn(env.state(), action, reward, env.state(), env.is_terminal());
//!         total_reward += reward;
//!     }
//! }
//!
//! // Backtest
//! let engine = BacktestEngine::new(env, agent);
//! let result = engine.run()?;
//! println!("Sharpe: {:.2}", result.report.sharpe_ratio);
//! ```

pub mod agents;
pub mod backtest;
pub mod environment;
pub mod exploration;
pub mod policy;

// Re-exports of the most commonly used types
pub use agents::{
    ActorCriticAgent, Agent, AgentConfig, DeepQLearningAgent, EnsembleAgent, PolicyGradientAgent,
    QLearningAgent, ReplayBuffer, ReplayBufferConfig,
};
pub use backtest::{
    BacktestEngine, BacktestResult, CommissionModel, MultiStrategyBacktest, PerformanceReport,
    SlippageModel, Trade, TradeDirection, TradeLog,
};
pub use environment::{
    Action, ActionSpace, FuturesEnv, FundingRateConfig, Order, OrderBookEnv, OrderSide,
    OrderStatus, OrderType, Observation, PortfolioEnv, PortfolioState, Reward, RewardFunction,
    RewardType, SimpleTradingEnv, State, TradingEnv,
};
pub use exploration::{
    BoltzmannExploration, EntropyRegularized, EpsilonGreedy, ExplorationSchedule,
    ExplorationStrategy, ThompsonSampling, UCB1,
};
pub use policy::{
    AdaptivePolicy, KellyPolicy, MeanReversionPolicy, MomentumPolicy, PolicyChain,
    RiskParityPolicy, TradingPolicy,
};
