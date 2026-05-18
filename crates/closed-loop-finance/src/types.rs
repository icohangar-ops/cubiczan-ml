//! # Core Types
//!
//! Fundamental types for the closed-loop finance system: loop phases, market regimes,
//! decision signals, actions, execution results, outcomes, portfolio state, and control parameters.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Phase in the O-D-E-L (Observe-Decide-Execute-Learn) cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopPhase {
    Observe,
    Decide,
    Execute,
    Learn,
    Evaluate,
}

impl std::fmt::Display for LoopPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopPhase::Observe => write!(f, "OBSERVE"),
            LoopPhase::Decide => write!(f, "DECIDE"),
            LoopPhase::Execute => write!(f, "EXECUTE"),
            LoopPhase::Learn => write!(f, "LEARN"),
            LoopPhase::Evaluate => write!(f, "EVALUATE"),
        }
    }
}

impl LoopPhase {
    /// All phases in cycle order.
    pub fn cycle_order() -> &'static [LoopPhase] {
        &[LoopPhase::Observe, LoopPhase::Decide, LoopPhase::Execute, LoopPhase::Learn, LoopPhase::Evaluate]
    }

    /// Next phase in the cycle.
    pub fn next(&self) -> LoopPhase {
        match self {
            LoopPhase::Observe => LoopPhase::Decide,
            LoopPhase::Decide => LoopPhase::Execute,
            LoopPhase::Execute => LoopPhase::Learn,
            LoopPhase::Learn => LoopPhase::Evaluate,
            LoopPhase::Evaluate => LoopPhase::Observe,
        }
    }
}

/// Current market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending,
    MeanReverting,
    Volatile,
    Quiet,
    Crisis,
    Recovery,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Trending => write!(f, "TRENDING"),
            MarketRegime::MeanReverting => write!(f, "MEAN_REVERTING"),
            MarketRegime::Volatile => write!(f, "VOLATILE"),
            MarketRegime::Quiet => write!(f, "QUIET"),
            MarketRegime::Crisis => write!(f, "CRISIS"),
            MarketRegime::Recovery => write!(f, "RECOVERY"),
        }
    }
}

impl MarketRegime {
    /// Risk multiplier for this regime (higher = more risky).
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            MarketRegime::Quiet => 0.5,
            MarketRegime::Trending => 0.8,
            MarketRegime::MeanReverting => 0.7,
            MarketRegime::Recovery => 1.0,
            MarketRegime::Volatile => 1.5,
            MarketRegime::Crisis => 2.5,
        }
    }

    /// Whether this is a defensive regime where we should reduce risk.
    pub fn is_defensive(&self) -> bool {
        matches!(self, MarketRegime::Crisis | MarketRegime::Volatile)
    }
}

/// An action the system can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Buy,
    Sell,
    Hold,
    Rebalance,
    Hedge,
    Delever,
    LeverageUp,
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Buy => write!(f, "BUY"),
            Action::Sell => write!(f, "SELL"),
            Action::Hold => write!(f, "HOLD"),
            Action::Rebalance => write!(f, "REBALANCE"),
            Action::Hedge => write!(f, "HEDGE"),
            Action::Delever => write!(f, "DELEVER"),
            Action::LeverageUp => write!(f, "LEVERAGE_UP"),
        }
    }
}

impl Action {
    /// Numeric score for the action: +1 for bullish, -1 for bearish, 0 for neutral.
    pub fn directional_score(&self) -> f64 {
        match self {
            Action::Buy | Action::LeverageUp => 1.0,
            Action::Sell | Action::Delever => -1.0,
            Action::Hold | Action::Rebalance | Action::Hedge => 0.0,
        }
    }
}

/// A decision signal produced by the decision engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSignal {
    /// Unique identifier for this decision.
    pub id: String,
    /// The action to take.
    pub action: Action,
    /// Confidence level in [0, 1].
    pub confidence: f64,
    /// Human-readable reasoning.
    pub reasoning: String,
    /// Risk assessment score in [0, 1] (higher = riskier).
    pub risk_assessment: f64,
    /// When this decision was made.
    pub timestamp: DateTime<Utc>,
    /// Target symbol (if applicable).
    pub symbol: Option<String>,
    /// Suggested position size as fraction of capital.
    pub position_size: f64,
    /// Factor scores that led to this decision.
    pub factor_scores: HashMap<String, f64>,
}

impl DecisionSignal {
    pub fn new(action: Action, confidence: f64, reasoning: impl Into<String>) -> Self {
        DecisionSignal {
            id: format!("dec_{}", Utc::now().timestamp_millis()),
            action,
            confidence: confidence.clamp(0.0, 1.0),
            reasoning: reasoning.into(),
            risk_assessment: 0.0,
            timestamp: Utc::now(),
            symbol: None,
            position_size: 0.0,
            factor_scores: HashMap::new(),
        }
    }

    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    pub fn with_risk(mut self, risk: f64) -> Self {
        self.risk_assessment = risk.clamp(0.0, 1.0);
        self
    }

    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size.clamp(0.0, 1.0);
        self
    }

    /// Whether the signal is actionable (above minimum confidence).
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        self.confidence >= min_confidence && self.action != Action::Hold
    }
}

/// Result of executing a trading action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// The action that was executed.
    pub action: Action,
    /// Price at which the order was filled.
    pub filled_price: f64,
    /// Quantity that was filled.
    pub filled_quantity: f64,
    /// Slippage as a fraction of price.
    pub slippage: f64,
    /// Total fees paid.
    pub fees: f64,
    /// When the execution occurred.
    pub timestamp: DateTime<Utc>,
    /// Whether the execution was successful.
    pub success: bool,
    /// Execution quality score in [0, 1].
    pub quality_score: f64,
    /// Symbol traded.
    pub symbol: String,
    /// Order type used.
    pub order_type: String,
}

/// Outcome of a completed trade for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// ID of the decision that led to this outcome.
    pub decision_id: String,
    /// Profit/loss in base currency.
    pub pnl: f64,
    /// Return as a percentage.
    pub return_pct: f64,
    /// Risk-adjusted return (return / risk taken).
    pub risk_adjusted_return: f64,
    /// Impact on drawdown.
    pub drawdown_impact: f64,
    /// Lesson learned from this outcome.
    pub lesson: String,
    /// Timestamp of the outcome.
    pub timestamp: DateTime<Utc>,
    /// Whether the trade was profitable.
    pub profitable: bool,
    /// Duration of the trade in seconds.
    pub duration_secs: f64,
    /// Market regime at the time of the trade.
    pub regime: MarketRegime,
}

impl Outcome {
    pub fn winning_lesson(&self) -> &str {
        if self.profitable { &self.lesson } else { "Loss" }
    }
}

/// A single position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Trading symbol (e.g., "BTC/USDT").
    pub symbol: String,
    /// Quantity held (positive = long, negative = short).
    pub quantity: f64,
    /// Average entry price.
    pub avg_entry: f64,
    /// Current market price.
    pub current_price: f64,
    /// Unrealized PnL.
    pub unrealized_pnl: f64,
    /// Portfolio weight (fraction of total value).
    pub weight: f64,
}

impl Position {
    pub fn unrealized_pnl_pct(&self) -> f64 {
        if self.avg_entry.abs() < 1e-15 || self.quantity.abs() < 1e-15 {
            return 0.0;
        }
        (self.current_price - self.avg_entry) / self.avg_entry * self.quantity.signum()
    }

    pub fn market_value(&self) -> f64 {
        self.quantity * self.current_price
    }
}

/// Full portfolio state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    /// Total portfolio value (cash + positions).
    pub total_value: f64,
    /// Available cash.
    pub cash: f64,
    /// Individual positions.
    pub positions: Vec<Position>,
    /// Current leverage ratio (total exposure / equity).
    pub leverage: f64,
    /// Total gross exposure as fraction of portfolio.
    pub exposure: f64,
    /// Maximum observed drawdown.
    pub max_drawdown: f64,
    /// Current Sharpe ratio.
    pub sharpe: f64,
    /// Timestamp of this state.
    pub timestamp: DateTime<Utc>,
}

impl Default for PortfolioState {
    fn default() -> Self {
        PortfolioState {
            total_value: 100_000.0,
            cash: 100_000.0,
            positions: Vec::new(),
            leverage: 1.0,
            exposure: 0.0,
            max_drawdown: 0.0,
            sharpe: 0.0,
            timestamp: Utc::now(),
        }
    }
}

/// A tunable control parameter in the feedback loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlParameter {
    /// Parameter name.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Minimum allowed value.
    pub min: f64,
    /// Maximum allowed value.
    pub max: f64,
    /// Current target value (set by the controller).
    pub target: f64,
}

impl ControlParameter {
    pub fn new(name: impl Into<String>, value: f64, min: f64, max: f64) -> Self {
        let v = value.clamp(min, max);
        ControlParameter {
            name: name.into(),
            value: v,
            min,
            max,
            target: v,
        }
    }

    /// Update the value, clamping to [min, max].
    pub fn set(&mut self, value: f64) {
        self.value = value.clamp(self.min, self.max);
    }

    /// Error between current value and target.
    pub fn error(&self) -> f64 {
        self.target - self.value
    }
}

/// Aggregated metrics for the closed-loop system.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoopMetrics {
    /// Number of completed O-D-E-L cycles.
    pub cycle_count: usize,
    /// Total PnL across all cycles.
    pub total_pnl: f64,
    /// Win rate across all trades.
    pub win_rate: f64,
    /// Average return per trade.
    pub avg_return: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
    /// Sharpe ratio of the loop.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Calmar ratio.
    pub calmar_ratio: f64,
}

/// A feedback signal from the control loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    /// Metric being monitored.
    pub metric: String,
    /// Actual observed value.
    pub actual: f64,
    /// Expected/target value.
    pub expected: f64,
    /// Deviation (actual - expected).
    pub deviation: f64,
    /// Suggested adjustment value.
    pub adjustment: f64,
}

impl FeedbackSignal {
    pub fn new(metric: impl Into<String>, actual: f64, expected: f64, adjustment: f64) -> Self {
        FeedbackSignal {
            metric: metric.into(),
            actual,
            expected,
            deviation: actual - expected,
            adjustment,
        }
    }

    /// Relative deviation as a fraction of expected.
    pub fn relative_deviation(&self) -> f64 {
        if self.expected.abs() < 1e-15 {
            0.0
        } else {
            self.deviation / self.expected.abs()
        }
    }
}

/// Observation data produced by the MarketObserver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketObservation {
    /// Current market regime.
    pub regime: MarketRegime,
    /// Current price.
    pub price: f64,
    /// Volatility level classification.
    pub volatility_regime: VolatilityRegime,
    /// Annualized volatility.
    pub volatility: f64,
    /// Trend strength score in [0, 1].
    pub trend_strength: f64,
    /// Trend direction (+1 bullish, -1 bearish, 0 neutral).
    pub trend_direction: f64,
    /// Liquidity score in [0, 1] (higher = more liquid).
    pub liquidity_score: f64,
    /// Average correlation across assets.
    pub avg_correlation: f64,
    /// Whether a risk event is detected.
    pub risk_event_detected: bool,
    /// Description of risk event (if any).
    pub risk_event_description: Option<String>,
    /// Recent returns (for factor computation).
    pub recent_returns: Vec<f64>,
    /// Observation timestamp.
    pub timestamp: DateTime<Utc>,
}

impl MarketObservation {
    /// Create a quiet/default observation with minimal data.
    pub fn quiet_default(price: f64) -> Self {
        MarketObservation {
            regime: MarketRegime::Quiet,
            price,
            volatility_regime: VolatilityRegime::Low,
            volatility: 0.0,
            trend_strength: 0.0,
            trend_direction: 0.0,
            liquidity_score: 0.5,
            avg_correlation: 0.5,
            risk_event_detected: false,
            risk_event_description: None,
            recent_returns: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Volatility regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

impl std::fmt::Display for VolatilityRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolatilityRegime::Low => write!(f, "LOW"),
            VolatilityRegime::Medium => write!(f, "MEDIUM"),
            VolatilityRegime::High => write!(f, "HIGH"),
            VolatilityRegime::Extreme => write!(f, "EXTREME"),
        }
    }
}

/// Fee model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    /// Fixed fee per trade.
    pub fixed_fee: f64,
    /// Percentage fee (e.g., 0.001 = 0.1%).
    pub percentage_fee: f64,
    /// Maker fee rate.
    pub maker_fee: f64,
    /// Taker fee rate.
    pub taker_fee: f64,
}

impl Default for FeeConfig {
    fn default() -> Self {
        FeeConfig {
            fixed_fee: 0.0,
            percentage_fee: 0.001,
            maker_fee: 0.0005,
            taker_fee: 0.001,
        }
    }
}

/// Slippage model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    /// Base slippage as fraction of price.
    pub base_slippage: f64,
    /// Volatility multiplier for slippage.
    pub volatility_multiplier: f64,
    /// Volume impact factor (how much order size affects slippage).
    pub volume_impact_factor: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        SlippageConfig {
            base_slippage: 0.0005,
            volatility_multiplier: 0.01,
            volume_impact_factor: 0.0001,
        }
    }
}

/// Configuration for the closed-loop pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Minimum confidence to act on a decision.
    pub min_confidence: f64,
    /// Maximum position size as fraction of capital.
    pub max_position_size: f64,
    /// Maximum leverage.
    pub max_leverage: f64,
    /// Maximum drawdown before circuit breaker.
    pub max_drawdown: f64,
    /// Risk-free rate for Sharpe calculation.
    pub risk_free_rate: f64,
    /// Periods per year for annualization.
    pub periods_per_year: f64,
    /// Fee configuration.
    pub fee_config: FeeConfig,
    /// Slippage configuration.
    pub slippage_config: SlippageConfig,
    /// Whether the loop is active.
    pub active: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            min_confidence: 0.3,
            max_position_size: 0.25,
            max_leverage: 2.0,
            max_drawdown: 0.15,
            risk_free_rate: 0.02,
            periods_per_year: 252.0,
            fee_config: FeeConfig::default(),
            slippage_config: SlippageConfig::default(),
            active: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_phase_cycle() {
        let phases = LoopPhase::cycle_order();
        assert_eq!(phases.len(), 5);
        assert_eq!(phases[0].next(), LoopPhase::Decide);
        assert_eq!(phases[4].next(), LoopPhase::Observe);
    }

    #[test]
    fn test_market_regime_risk() {
        assert!(MarketRegime::Crisis.risk_multiplier() > MarketRegime::Quiet.risk_multiplier());
        assert!(MarketRegime::Crisis.is_defensive());
        assert!(MarketRegime::Volatile.is_defensive());
        assert!(!MarketRegime::Trending.is_defensive());
    }

    #[test]
    fn test_action_directional() {
        assert!(Action::Buy.directional_score() > 0.0);
        assert!(Action::Sell.directional_score() < 0.0);
        assert_eq!(Action::Hold.directional_score(), 0.0);
    }

    #[test]
    fn test_decision_signal() {
        let sig = DecisionSignal::new(Action::Buy, 0.8, "Strong momentum")
            .with_symbol("BTC/USDT")
            .with_risk(0.3)
            .with_position_size(0.15);
        assert!(sig.is_actionable(0.5));
        assert!(!sig.is_actionable(0.9));
        assert_eq!(sig.symbol.as_deref(), Some("BTC/USDT"));
        assert!(sig.position_size > 0.0);
    }

    #[test]
    fn test_decision_signal_clamping() {
        let sig = DecisionSignal::new(Action::Sell, 1.5, "Overconfident");
        assert_eq!(sig.confidence, 1.0);
    }

    #[test]
    fn test_feedback_signal() {
        let fb = FeedbackSignal::new("sharpe", 1.5, 2.0, -0.25);
        assert_eq!(fb.deviation, -0.5);
        assert_eq!(fb.relative_deviation(), -0.25);
    }

    #[test]
    fn test_control_parameter() {
        let mut cp = ControlParameter::new("leverage", 1.5, 0.5, 3.0);
        assert_eq!(cp.error(), 0.0);
        cp.target = 2.0;
        assert!((cp.error() - 0.5).abs() < 1e-10);
        cp.set(5.0); // Should clamp
        assert_eq!(cp.value, 3.0);
    }

    #[test]
    fn test_control_parameter_clamp_on_create() {
        let cp = ControlParameter::new("test", -1.0, 0.0, 1.0);
        assert_eq!(cp.value, 0.0);
    }

    #[test]
    fn test_position_pnl() {
        let pos = Position {
            symbol: "BTC".into(),
            quantity: 1.0,
            avg_entry: 100.0,
            current_price: 110.0,
            unrealized_pnl: 10.0,
            weight: 0.5,
        };
        assert!((pos.unrealized_pnl_pct() - 0.1).abs() < 1e-10);
        assert_eq!(pos.market_value(), 110.0);
    }

    #[test]
    fn test_position_negative_quantity() {
        let pos = Position {
            symbol: "BTC".into(),
            quantity: -1.0,
            avg_entry: 100.0,
            current_price: 90.0,
            unrealized_pnl: 10.0,
            weight: 0.5,
        };
        assert!((pos.unrealized_pnl_pct() - 0.1).abs() < 1e-10); // short profit
    }

    #[test]
    fn test_portfolio_state_default() {
        let ps = PortfolioState::default();
        assert_eq!(ps.total_value, 100_000.0);
        assert!(ps.positions.is_empty());
    }

    #[test]
    fn test_volatility_regime_display() {
        assert_eq!(format!("{}", VolatilityRegime::High), "HIGH");
    }

    #[test]
    fn test_fee_config_default() {
        let fc = FeeConfig::default();
        assert!((fc.percentage_fee - 0.001).abs() < 1e-10);
        assert!(fc.maker_fee < fc.taker_fee);
    }

    #[test]
    fn test_pipeline_config_default() {
        let pc = PipelineConfig::default();
        assert!(pc.active);
        assert!(pc.min_confidence > 0.0);
    }

    #[test]
    fn test_outcome_profitable_flag() {
        let o = Outcome {
            decision_id: "d1".into(),
            pnl: 100.0,
            return_pct: 0.1,
            risk_adjusted_return: 0.05,
            drawdown_impact: 0.0,
            lesson: "Good trade".into(),
            timestamp: Utc::now(),
            profitable: true,
            duration_secs: 3600.0,
            regime: MarketRegime::Trending,
        };
        assert_eq!(o.winning_lesson(), "Good trade");
    }

    #[test]
    fn test_serde_roundtrip_decision_signal() {
        let sig = DecisionSignal::new(Action::Buy, 0.8, "test");
        let json = serde_json::to_string(&sig).unwrap();
        let sig2: DecisionSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(sig.action, sig2.action);
        assert_eq!(sig.confidence, sig2.confidence);
    }

    #[test]
    fn test_serde_roundtrip_portfolio_state() {
        let ps = PortfolioState::default();
        let json = serde_json::to_string(&ps).unwrap();
        let ps2: PortfolioState = serde_json::from_str(&json).unwrap();
        assert_eq!(ps.total_value, ps2.total_value);
    }
}
