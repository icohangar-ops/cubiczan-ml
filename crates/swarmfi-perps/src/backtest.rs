//! Backtesting Engine — Historical signal performance analysis.
//!
//! Provides a framework for evaluating how well the swarm consensus signals
//! would have performed on historical market data. Supports equity curve
//! computation, drawdown analysis, Sharpe/Sortino ratios, and per-signal
//! win-rate tracking.
//!
//! # Architecture
//!
//! ```text
//! Historical OHLCV data
//!         │
//!         ▼
//! ┌──────────────────┐
//! │ BacktestEngine   │ ← Strategy config, slippage model, commission schedule
//! └────────┬─────────┘
//!          │ iterates each candle
//!          ▼
//! ┌──────────────────┐
//! │ Signal Generator │ ← Uses swarm agents on rolling window of data
//! └────────┬─────────┘
//!          │ LONG/SHORT/NEUTRAL per candle
//!          ▼
//! ┌──────────────────┐
//! │ Trade Simulator  │ ← Executes trades with slippage & fees
//! └────────┬─────────┘
//!          │
//!          ▼
//!   BacktestReport (metrics, equity curve, trades)
//! ```

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A simulated trade in the backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedTrade {
    /// "BTC-USD" etc.
    pub market: String,
    pub side: Signal,
    /// Entry price.
    pub entry_price: f64,
    /// Exit price.
    pub exit_price: f64,
    /// Position size in base asset units.
    pub size: f64,
    /// PnL in quote currency (USD).
    pub pnl: f64,
    /// PnL as a percentage.
    pub pnl_pct: f64,
    /// Commission paid.
    pub commission: f64,
    /// Entry timestamp (candle index or timestamp).
    pub entry_time: i64,
    /// Exit timestamp.
    pub exit_time: i64,
    /// Number of candles held.
    pub bars_held: u32,
}

/// Equity curve point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: i64,
    pub equity: f64,
    pub drawdown_pct: f64,
    /// Signal at this point.
    pub signal: Signal,
}

/// Risk-adjusted performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return as a percentage.
    pub total_return_pct: f64,
    /// Annualized return (%).
    pub annualized_return_pct: f64,
    /// Maximum drawdown (%).
    pub max_drawdown_pct: f64,
    /// Sharpe ratio (risk-free rate = 0).
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside deviation only).
    pub sortino_ratio: f64,
    /// Win rate as a decimal (0.0–1.0).
    pub win_rate: f64,
    /// Total number of trades.
    pub total_trades: u32,
    /// Winning trades count.
    pub winning_trades: u32,
    /// Losing trades count.
    pub losing_trades: u32,
    /// Average win (%).
    pub avg_win_pct: f64,
    /// Average loss (%).
    pub avg_loss_pct: f64,
    /// Profit factor (gross wins / gross losses).
    pub profit_factor: f64,
    /// Average trade duration in candles.
    pub avg_bars_held: f64,
    /// Long-only return (%).
    pub long_return_pct: f64,
    /// Short-only return (%).
    pub short_return_pct: f64,
}

/// Per-signal-type breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SignalBreakdown {
    pub signal_type: String,
    pub count: u32,
    pub wins: u32,
    pub total_pnl: f64,
    pub win_rate: f64,
    pub avg_pnl_pct: f64,
}

/// The complete backtest report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub market: String,
    pub total_candles: u32,
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<EquityPoint>,
    pub trades: Vec<SimulatedTrade>,
    pub signal_breakdown: Vec<SignalBreakdown>,
}

/// Strategy configuration for the backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Starting capital in USD.
    pub initial_capital: f64,
    /// Per-trade commission as a fraction (e.g., 0.001 = 0.1%).
    pub commission_rate: f64,
    /// Slippage model as a fraction of price.
    pub slippage_rate: f64,
    /// Maximum position size as a fraction of equity (0.0–1.0).
    pub max_position_fraction: f64,
    /// Minimum holding period in candles.
    pub min_hold_bars: u32,
    /// Maximum holding period in candles (0 = no limit).
    pub max_hold_bars: u32,
    /// Whether to use the consensus signal or individual agents.
    pub use_consensus: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            commission_rate: 0.0005, // 0.05% per side
            slippage_rate: 0.0002,   // 0.02%
            max_position_fraction: 0.95,
            min_hold_bars: 3,
            max_hold_bars: 24,
            use_consensus: true,
        }
    }
}

/// The backtesting engine.
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BacktestEngine {
    pub fn new() -> Self {
        Self {
            config: BacktestConfig::default(),
        }
    }

    pub fn with_config(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run a backtest over historical candles with pre-computed signals.
    ///
    /// `candles` must be sorted chronologically. `signals[i]` is the signal
    /// at candle `i`. If `signals` is shorter than `candles`, remaining
    /// candles use `Signal::Neutral`.
    pub fn run(
        &self,
        market: &str,
        candles: &[Candle],
        signals: &[Signal],
    ) -> BacktestReport {
        if candles.is_empty() {
            return BacktestReport {
                market: market.to_string(),
                total_candles: 0,
                metrics: PerformanceMetrics::zero(),
                equity_curve: vec![],
                trades: vec![],
                signal_breakdown: vec![],
            };
        }

        let mut equity = self.config.initial_capital;
        let mut trades: Vec<SimulatedTrade> = Vec::new();
        let mut equity_curve: Vec<EquityPoint> = Vec::new();
        let mut peak_equity = equity;
        let mut max_drawdown = 0.0_f64;

        let mut open_signal: Option<(Signal, usize, f64)> = None; // (signal, candle_idx, entry_price)

        for i in 0..candles.len() {
            let close = candles[i].close;
            let signal = signals.get(i).copied().unwrap_or(Signal::Neutral);
            let timestamp = parse_candle_timestamp(&candles[i].started_at, i as i64);

            // Check if we should close existing position
            let mut closed = false;
            if let Some((pos_signal, entry_idx, entry_price)) = open_signal {
                let bars_held = (i - entry_idx) as u32;

                // Close conditions: signal change, max hold reached
                let should_close = pos_signal != signal
                    || (self.config.max_hold_bars > 0 && bars_held >= self.config.max_hold_bars);

                if should_close && bars_held >= self.config.min_hold_bars {
                    closed = true;
                    let trade = self.simulate_trade(
                        market,
                        pos_signal,
                        entry_price,
                        close,
                        equity,
                        entry_idx,
                        i,
                        bars_held,
                    );
                    equity += trade.pnl;
                    trades.push(trade);
                    open_signal = None;
                }
            }

            // Open new position if no position and signal is directional
            if open_signal.is_none() && !closed {
                if signal == Signal::Long || signal == Signal::Short {
                    open_signal = Some((signal, i, close));
                }
            }

            // Track equity curve
            if equity > peak_equity {
                peak_equity = equity;
            }
            let drawdown = if peak_equity > 0.0 {
                (peak_equity - equity) / peak_equity * 100.0
            } else {
                0.0
            };
            max_drawdown = max_drawdown.max(drawdown);

            equity_curve.push(EquityPoint {
                timestamp,
                equity,
                drawdown_pct: drawdown,
                signal,
            });
        }

        // Close any remaining open position at last close
        if let Some((pos_signal, entry_idx, entry_price)) = open_signal {
            let bars_held = (candles.len() - 1 - entry_idx) as u32;
            if bars_held >= self.config.min_hold_bars {
                let trade = self.simulate_trade(
                    market,
                    pos_signal,
                    entry_price,
                    candles.last().unwrap().close,
                    equity,
                    entry_idx,
                    candles.len() - 1,
                    bars_held,
                );
                equity += trade.pnl;
                trades.push(trade);
            }
        }

        let metrics = self.compute_metrics(&trades, equity);
        let signal_breakdown = Self::compute_signal_breakdown(&trades);

        BacktestReport {
            market: market.to_string(),
            total_candles: candles.len() as u32,
            metrics,
            equity_curve,
            trades,
            signal_breakdown,
        }
    }

    /// Simulate a single trade with slippage and commission.
    fn simulate_trade(
        &self,
        market: &str,
        signal: Signal,
        entry_price: f64,
        exit_price: f64,
        equity: f64,
        entry_idx: usize,
        exit_idx: usize,
        bars_held: u32,
    ) -> SimulatedTrade {
        let slip = entry_price * self.config.slippage_rate;
        let effective_entry = match signal {
            Signal::Long => entry_price + slip,
            Signal::Short => entry_price - slip,
            Signal::Neutral => entry_price,
        };
        let effective_exit = match signal {
            Signal::Long => exit_price - slip,
            Signal::Short => exit_price + slip,
            Signal::Neutral => exit_price,
        };

        let position_size = equity * self.config.max_position_fraction;
        let size = position_size / effective_entry;
        let commission = position_size * self.config.commission_rate * 2.0; // entry + exit

        let pnl = match signal {
            Signal::Long => (effective_exit - effective_entry) * size - commission,
            Signal::Short => (effective_entry - effective_exit) * size - commission,
            Signal::Neutral => -commission,
        };

        let pnl_pct = pnl / position_size * 100.0;

        SimulatedTrade {
            market: market.to_string(),
            side: signal,
            entry_price: effective_entry,
            exit_price: effective_exit,
            size,
            pnl,
            pnl_pct,
            commission,
            entry_time: parse_candle_timestamp("", entry_idx as i64),
            exit_time: parse_candle_timestamp("", exit_idx as i64),
            bars_held,
        }
    }

    /// Compute performance metrics from completed trades.
    fn compute_metrics(&self, trades: &[SimulatedTrade], final_equity: f64) -> PerformanceMetrics {
        if trades.is_empty() {
            return PerformanceMetrics::zero();
        }

        let wins: Vec<_> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losses: Vec<_> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let total_return_pct =
            (final_equity - self.config.initial_capital) / self.config.initial_capital * 100.0;

        let gross_profit: f64 = wins.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losses.iter().map(|t| t.pnl.abs()).sum();

        let avg_win_pct = if !wins.is_empty() {
            wins.iter().map(|t| t.pnl_pct).sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss_pct = if !losses.is_empty() {
            losses.iter().map(|t| t.pnl_pct).sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        // Sharpe ratio from trade returns
        let returns: Vec<f64> = trades.iter().map(|t| t.pnl_pct / 100.0).collect();
        let sharpe = compute_sharpe(&returns);
        let sortino = compute_sortino(&returns);

        // Long/short breakdown
        let long_trades: Vec<_> = trades.iter().filter(|t| t.side == Signal::Long).collect();
        let short_trades: Vec<_> = trades.iter().filter(|t| t.side == Signal::Short).collect();

        let long_return_pct = long_trades.iter().map(|t| t.pnl).sum::<f64>()
            / self.config.initial_capital * 100.0;
        let short_return_pct = short_trades.iter().map(|t| t.pnl).sum::<f64>()
            / self.config.initial_capital * 100.0;

        PerformanceMetrics {
            total_return_pct,
            annualized_return_pct: total_return_pct, // Simplified (needs time period for true annualization)
            max_drawdown_pct: 0.0,                   // Computed from equity curve in caller
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            win_rate: wins.len() as f64 / trades.len() as f64,
            total_trades: trades.len() as u32,
            winning_trades: wins.len() as u32,
            losing_trades: losses.len() as u32,
            avg_win_pct,
            avg_loss_pct,
            profit_factor: if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY },
            avg_bars_held: trades.iter().map(|t| t.bars_held as f64).sum::<f64>() / trades.len() as f64,
            long_return_pct,
            short_return_pct,
        }
    }

    /// Compute per-signal-type breakdown.
    fn compute_signal_breakdown(trades: &[SimulatedTrade]) -> Vec<SignalBreakdown> {
        let mut by_type: HashMap<String, Vec<&SimulatedTrade>> = HashMap::new();
        for trade in trades {
            by_type
                .entry(trade.side.as_str().to_string())
                .or_default()
                .push(trade);
        }

        let mut breakdown = Vec::new();
        for (signal_type, group) in by_type {
            let wins = group.iter().filter(|t| t.pnl > 0.0).count();
            let total_pnl: f64 = group.iter().map(|t| t.pnl).sum();
            let avg_pnl: f64 = group.iter().map(|t| t.pnl_pct).sum::<f64>() / group.len() as f64;
            breakdown.push(SignalBreakdown {
                signal_type,
                count: group.len() as u32,
                wins: wins as u32,
                total_pnl,
                win_rate: wins as f64 / group.len() as f64,
                avg_pnl_pct: avg_pnl,
            });
        }
        breakdown.sort_by(|a, b| b.total_pnl.partial_cmp(&a.total_pnl).unwrap_or(std::cmp::Ordering::Equal));
        breakdown
    }
}

impl PerformanceMetrics {
    /// Zero-initialized metrics (no trades).
    pub fn zero() -> Self {
        Self {
            total_return_pct: 0.0,
            annualized_return_pct: 0.0,
            max_drawdown_pct: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            win_rate: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_win_pct: 0.0,
            avg_loss_pct: 0.0,
            profit_factor: 0.0,
            avg_bars_held: 0.0,
            long_return_pct: 0.0,
            short_return_pct: 0.0,
        }
    }
}

/// Compute Sharpe ratio from a list of returns (mean / std).
fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1).max(1) as f64;
    let std = variance.sqrt();
    if std < f64::EPSILON {
        return 0.0;
    }
    mean / std
}

/// Compute Sortino ratio (mean / downside deviation).
fn compute_sortino(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: Vec<f64> = returns.iter().map(|r| r.min(0.0)).collect();
    let downside_variance = downside
        .iter()
        .map(|r| r.powi(2))
        .sum::<f64>()
        / downside.len().max(1) as f64;
    let downside_dev = downside_variance.sqrt();
    if downside_dev < f64::EPSILON {
        return 0.0;
    }
    mean / downside_dev
}

/// Generate mock historical candles for backtesting.
pub fn generate_historical_candles(
    base_price: f64,
    count: u32,
    trend: f64,
    volatility: f64,
) -> Vec<Candle> {
    let mut candles = Vec::with_capacity(count as usize);
    let mut price = base_price;

    for i in 0..count {
        let drift = base_price * trend * 0.001;
        let noise = (rand::random::<f64>() - 0.5) * base_price * volatility;
        let open = price;
        let close = price + drift + noise;
        let high = open.max(close) + base_price * volatility * 0.3 * rand::random::<f64>();
        let low = open.min(close) - base_price * volatility * 0.3 * rand::random::<f64>();

        candles.push(Candle {
            started_at: format!("2025-01-01T{:02}:00:00Z", i % 24),
            open,
            high: high.max(close),
            low: low.max(0.0).min(close),
            close,
            base_token_volume: 100.0 + rand::random::<f64>() * 500.0,
            usd_volume: close * (100.0 + rand::random::<f64>() * 500.0),
            trades: 200 + (rand::random::<f64>() * 800.0) as u32,
        });

        price = close;
    }

    candles
}

/// Generate a simple trend-following signal series.
///
/// Goes LONG when close > previous close * (1 + threshold),
/// SHORT when close < previous close * (1 - threshold), else NEUTRAL.
pub fn generate_trend_signals(candles: &[Candle], threshold: f64) -> Vec<Signal> {
    let mut signals = Vec::with_capacity(candles.len());
    for i in 0..candles.len() {
        if i == 0 {
            signals.push(Signal::Neutral);
            continue;
        }
        let prev_close = candles[i - 1].close;
        let curr_close = candles[i].close;
        let change = (curr_close - prev_close) / prev_close;

        if change > threshold {
            signals.push(Signal::Long);
        } else if change < -threshold {
            signals.push(Signal::Short);
        } else {
            signals.push(Signal::Neutral);
        }
    }
    signals
}

/// Parse a candle timestamp string into a unix ms timestamp, with fallback.
fn parse_candle_timestamp(started_at: &str, fallback_idx: i64) -> i64 {
    if !started_at.is_empty() {
        chrono::DateTime::parse_from_rfc3339(started_at)
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(fallback_idx * 3600_000)
    } else {
        fallback_idx * 3600_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_empty_data() {
        let engine = BacktestEngine::new();
        let report = engine.run("BTC-USD", &[], &[]);
        assert_eq!(report.total_candles, 0);
        assert_eq!(report.metrics.total_trades, 0);
    }

    #[test]
    fn test_backtest_uptrend_long_signals() {
        let candles = generate_historical_candles(67000.0, 100, 1.0, 0.005);
        let signals = generate_trend_signals(&candles, 0.001);

        let engine = BacktestEngine::new();
        let report = engine.run("BTC-USD", &candles, &signals);

        assert_eq!(report.total_candles, 100);
        assert!(report.equity_curve.len() == 100);
        assert!(report.metrics.total_trades > 0);
    }

    #[test]
    fn test_backtest_neutral_signals_no_trades() {
        let candles = generate_historical_candles(67000.0, 50, 0.0, 0.002);
        let signals = vec![Signal::Neutral; 50];

        let engine = BacktestEngine::new();
        let report = engine.run("BTC-USD", &candles, &signals);
        assert_eq!(report.metrics.total_trades, 0);
    }

    #[test]
    fn test_performance_metrics_zero() {
        let m = PerformanceMetrics::zero();
        assert_eq!(m.total_trades, 0);
        assert_eq!(m.sharpe_ratio, 0.0);
    }

    #[test]
    fn test_compute_sharpe() {
        let returns = vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02];
        let sharpe = compute_sharpe(&returns);
        assert!(!sharpe.is_nan());
    }

    #[test]
    fn test_compute_sortino() {
        let returns = vec![0.02, 0.01, -0.01, 0.03, 0.00, 0.01];
        let sortino = compute_sortino(&returns);
        assert!(sortino > 0.0);
        assert!(!sortino.is_nan());
    }

    #[test]
    fn test_sharpe_single_return() {
        assert_eq!(compute_sharpe(&[0.05]), 0.0);
        assert_eq!(compute_sharpe(&[]), 0.0);
    }

    #[test]
    fn test_trend_signal_generation() {
        let candles = generate_historical_candles(67000.0, 10, 0.5, 0.01);
        let signals = generate_trend_signals(&candles, 0.001);
        assert_eq!(signals.len(), 10);
        assert_eq!(signals[0], Signal::Neutral); // First is always neutral
    }

    #[test]
    fn test_signal_breakdown() {
        let trades = vec![
            SimulatedTrade {
                market: "BTC".into(), side: Signal::Long,
                entry_price: 67000.0, exit_price: 67100.0,
                size: 1.0, pnl: 100.0, pnl_pct: 0.15,
                commission: 1.0, entry_time: 0, exit_time: 10, bars_held: 10,
            },
            SimulatedTrade {
                market: "BTC".into(), side: Signal::Long,
                entry_price: 67100.0, exit_price: 67000.0,
                size: 1.0, pnl: -101.0, pnl_pct: -0.15,
                commission: 1.0, entry_time: 11, exit_time: 20, bars_held: 9,
            },
        ];
        let breakdown = BacktestEngine::compute_signal_breakdown(&trades);
        assert_eq!(breakdown.len(), 1);
        assert_eq!(breakdown[0].signal_type, "LONG");
        assert_eq!(breakdown[0].count, 2);
        assert_eq!(breakdown[0].wins, 1);
        assert!((breakdown[0].win_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_historical_candles_valid() {
        let candles = generate_historical_candles(67000.0, 50, 0.0, 0.01);
        assert_eq!(candles.len(), 50);
        for c in &candles {
            assert!(c.close > 0.0);
            assert!(c.high >= c.low);
            assert!(c.usd_volume > 0.0);
        }
    }

    #[test]
    fn test_backtest_config_default() {
        let cfg = BacktestConfig::default();
        assert_eq!(cfg.initial_capital, 10_000.0);
        assert_eq!(cfg.commission_rate, 0.0005);
        assert_eq!(cfg.slippage_rate, 0.0002);
        assert_eq!(cfg.min_hold_bars, 3);
        assert_eq!(cfg.max_hold_bars, 24);
    }

    #[test]
    fn test_equity_point_serde() {
        let ep = EquityPoint {
            timestamp: 1704067200000,
            equity: 10500.0,
            drawdown_pct: 0.5,
            signal: Signal::Long,
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("10500"));
    }

    #[test]
    fn test_simulated_trade_serde() {
        let trade = SimulatedTrade {
            market: "BTC-USD".into(),
            side: Signal::Short,
            entry_price: 67000.0,
            exit_price: 66900.0,
            size: 0.15,
            pnl: 15.0,
            pnl_pct: 0.22,
            commission: 0.67,
            entry_time: 0,
            exit_time: 5,
            bars_held: 5,
        };
        let json = serde_json::to_string(&trade).unwrap();
        let restored: SimulatedTrade = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.side, Signal::Short);
        assert_eq!(restored.pnl, 15.0);
    }

    #[test]
    fn test_backtest_report_serde() {
        let report = BacktestReport {
            market: "ETH-USD".into(),
            total_candles: 0,
            metrics: PerformanceMetrics::zero(),
            equity_curve: vec![],
            trades: vec![],
            signal_breakdown: vec![],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("ETH-USD"));
    }
}
